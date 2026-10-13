/*
 * Copyright 2025 WebAssembly Community Group participants
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// Rolls constant "table" globals into data segments and lazy factories.
//
// A table global is an immutable global whose initializer is an
// array.new_fixed (possibly wrapped in a struct.new with otherwise constant
// operands, e.g. a Java array object) whose elements all have the same shape:
// nested struct.new / array.new_fixed trees whose leaves are constants, where
// only numeric constants may differ between elements. For example, Unicode
// range tables:
//
//   (global $t (ref $Table) (struct.new $Table (global.get $vt) (i32.const 0)
//     (array.new_fixed $Arr 3
//       (struct.new $Range (global.get $vt2) (i32.const 0)
//         (array.new_fixed $Ints 3 (i32.const 97) (i32.const 122) (i32.const
//         1)))
//       ...)))
//
// Each such element costs ~17 bytes in the global section. This pass groups
// the tables by type and shape, writes the varying leaves of all elements of
// a group into columns of ONE passive data segment (as i8/i16/i32 depending
// on the value range, optionally as the difference to another column of the
// same element), and emits a factory
//
//   (func $tables$N (param $k i32) (result (ref $Table))
//     ;; cache check, then
//     ;; offsets = array.new_data (per-table element offsets),
//     ;; off = offsets[k], n = offsets[k+1] - off,
//     ;; column_j = array.new_data of the table's slice of column j,
//     ;; arr = array.new_default n; loop { arr[i] = element(column reads) }
//     ;; result = outer struct.new with arr, cached in a mutable array)
//
// and replaces (global.get $t_k) by (call $tables$N (i32.const k)). Object
// identity is preserved by the cache. Only globals used exclusively from
// function code are rolled, and only when the size model says it saves bytes.
// Members that force a wider column (outliers) are excluded when that pays.
// --pass-arg=roll-constant-tables-always@1 ignores the size model.
//
// This is meant for the end of a size-focused pipeline, before
// --roll-repeated-calls (whose loops absorb the new indexed call sites) and
// --remove-unused-module-elements.
//

#include <algorithm>
#include <functional>
#include <string>
#include <tuple>

#include "ir/module-utils.h"
#include "ir/names.h"
#include "ir/utils.h"
#include "pass.h"
#include "support/insert_ordered.h"
#include "wasm-binary.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

constexpr Index MaxDepth = 8;
constexpr Index MaxLeaves = 64;
// Fixed cost of a group: the factory function (code, function section entry,
// type), the cache global and its type, the segment header.
constexpr Index GroupOverhead = 170;
// Per column: the array.new_data, the local and the read in the factory.
constexpr Index ColumnOverhead = 14;
// At most this fraction of the elements may be excluded as outliers in one
// step.
constexpr Index OutlierPercent = 10;
constexpr Index MaxOutlierRounds = 6;

void appendLE(std::vector<char>& data, uint64_t value, Index width) {
  for (Index i = 0; i < width; i++) {
    data.push_back(char(uint8_t(value >> (8 * i))));
  }
}

struct RollConstantTables : public Pass {
  struct Member {
    Global* global = nullptr;
    // The element list.
    ArrayNewFixed* array = nullptr;
    // The leaves of each element, in pre-order.
    std::vector<std::vector<Expression*>> elements;
    Index initBytes = 0;
    Index uses = 0;
    Index firstUse = Index(-1);
  };

  struct Group {
    Type globalType;
    HeapType arrayType;
    std::vector<Member> members;
  };

  // How a varying column is stored.
  struct Encoding {
    // The column whose value of the same element is subtracted, or -1.
    int base = -1;
    // 1, 2, 4 or 8 bytes.
    Index width = 4;
    bool signed_ = false;
  };

  struct Plan {
    std::vector<Member*> members;
    // The varying leaf indexes.
    std::vector<Index> varying;
    // Per varying leaf (same order as varying), the encoding.
    std::vector<Encoding> encodings;
    Index numElements = 0;
    int64_t saving = 0;
  };

  struct Replacement {
    Name factory;
    Index index;
  };

  using GroupKey = std::tuple<Type, std::string>;

  Module* module = nullptr;
  std::unordered_map<Name, Index> globalIndexes;
  Index numGroups = 0;
  // Ignore the size model (for testing).
  bool always = false;

  void run(Module* module_) override {
    module = module_;
    if (!module->features.hasGC() || !module->features.hasBulkMemory()) {
      return;
    }
    always = !getArgumentOrDefault("roll-constant-tables-always", "").empty();
    computeGlobalIndexes();

    // Find the candidates and group them.
    InsertOrderedMap<GroupKey, Group> groups;
    std::unordered_map<Name, std::pair<Group*, Index>> candidateMap;
    for (auto& global : module->globals) {
      if (global->imported() || global->mutable_) {
        continue;
      }
      Member member;
      std::string shape;
      HeapType arrayType;
      if (!analyze(global.get(), member, shape, arrayType)) {
        continue;
      }
      auto& group = groups[{global->type, shape}];
      group.globalType = global->type;
      group.arrayType = arrayType;
      group.members.push_back(std::move(member));
      candidateMap[global->name] = {&group, Index(group.members.size() - 1)};
    }
    if (groups.empty()) {
      return;
    }

    // Globals used outside of function code must stay.
    std::unordered_set<Name> nonCodeUses;
    {
      struct Scanner : public PostWalker<Scanner> {
        std::unordered_set<Name>& names;
        Scanner(std::unordered_set<Name>& names) : names(names) {}
        void visitGlobalGet(GlobalGet* curr) { names.insert(curr->name); }
      } scanner(nonCodeUses);
      for (auto& global : module->globals) {
        if (!global->imported()) {
          scanner.walk(global->init);
        }
      }
      for (auto& segment : module->elementSegments) {
        if (segment->offset) {
          scanner.walk(segment->offset);
        }
        for (auto*& item : segment->data) {
          scanner.walk(item);
        }
      }
      for (auto& segment : module->dataSegments) {
        if (segment->offset) {
          scanner.walk(segment->offset);
        }
      }
      for (auto& ex : module->exports) {
        if (ex->kind == ExternalKind::Global) {
          nonCodeUses.insert(*ex->getInternalName());
        }
      }
    }

    // Count the uses in function code.
    struct FunctionInfo {
      std::vector<Expression**> gets;
    };
    ModuleUtils::ParallelFunctionAnalysis<FunctionInfo> analysis(
      *module, [&](Function* func, FunctionInfo& info) {
        if (func->imported()) {
          return;
        }
        struct Scanner : public PostWalker<Scanner> {
          FunctionInfo& info;
          const std::unordered_map<Name, std::pair<Group*, Index>>& candidates;
          Scanner(FunctionInfo& info,
                  const std::unordered_map<Name, std::pair<Group*, Index>>&
                    candidates)
            : info(info), candidates(candidates) {}
          void visitGlobalGet(GlobalGet* curr) {
            if (candidates.count(curr->name)) {
              info.gets.push_back(getCurrentPointer());
            }
          }
        };
        Scanner(info, candidateMap).walk(func->body);
      });
    std::vector<Expression**> allGets;
    for (auto& func : module->functions) {
      auto& info = analysis.map[func.get()];
      for (auto** ptr : info.gets) {
        auto& [group, i] = candidateMap[(*ptr)->cast<GlobalGet>()->name];
        auto& member = group->members[i];
        if (member.uses == 0) {
          member.firstUse = allGets.size();
        }
        member.uses++;
        allGets.push_back(ptr);
      }
    }

    // Plan and emit the groups.
    std::unordered_map<Name, Replacement> replacements;
    for (auto& [key, group] : groups) {
      auto& members = group.members;
      members.erase(std::remove_if(members.begin(),
                                   members.end(),
                                   [&](const Member& m) {
                                     return m.uses == 0 ||
                                            nonCodeUses.count(m.global->name);
                                   }),
                    members.end());
      if (members.empty()) {
        continue;
      }
      processGroup(group, replacements);
    }
    if (replacements.empty()) {
      return;
    }

    Builder builder(*module);
    for (auto** ptr : allGets) {
      auto* get = (*ptr)->cast<GlobalGet>();
      auto iter = replacements.find(get->name);
      if (iter == replacements.end()) {
        continue;
      }
      *ptr = builder.makeCall(iter->second.factory,
                              {builder.makeConst(int32_t(iter->second.index))},
                              get->type);
    }
    for (auto& [name, _] : replacements) {
      module->removeGlobal(name);
    }
  }

private:
  void computeGlobalIndexes() {
    Index importedGlobals = 0;
    for (auto& global : module->globals) {
      if (global->imported()) {
        importedGlobals++;
      }
    }
    Index nextImported = 0, nextDefined = importedGlobals;
    for (auto& global : module->globals) {
      globalIndexes[global->name] =
        global->imported() ? nextImported++ : nextDefined++;
    }
  }

  static bool isNumericConst(Expression* curr) {
    auto* c = curr->dynCast<Const>();
    return c && (c->type == Type::i32 || c->type == Type::i64 ||
                 c->type == Type::f32 || c->type == Type::f64);
  }

  // Whether an operand is a constant we can move into the factory.
  bool isValidLeaf(Expression* curr) {
    if (curr->is<Const>() || curr->is<RefNull>() || curr->is<RefFunc>()) {
      return true;
    }
    if (auto* get = curr->dynCast<GlobalGet>()) {
      return !module->getGlobal(get->name)->mutable_;
    }
    if (auto* i31 = curr->dynCast<RefI31>()) {
      return i31->value->is<Const>();
    }
    if (auto* structNew = curr->dynCast<StructNew>()) {
      // struct.new_default
      return structNew->operands.empty() && !structNew->desc;
    }
    return false;
  }

  static std::string hashString(Expression* curr) {
    return std::to_string(ExpressionAnalyzer::hash(curr));
  }

  // Checks the outer shape: an array.new_fixed, or a struct.new with constant
  // operands and exactly one array.new_fixed operand. Constant operands are
  // part of the shape (they must be identical in a group).
  bool
  analyzeOuter(Expression* init, std::string& shape, ArrayNewFixed*& array) {
    if (auto* arrayNew = init->dynCast<ArrayNewFixed>()) {
      array = arrayNew;
      shape += "A";
      shape += std::to_string(arrayNew->type.getHeapType().getID());
      return true;
    }
    auto* structNew = init->dynCast<StructNew>();
    if (!structNew || structNew->desc || !structNew->type.isRef()) {
      return false;
    }
    auto type = structNew->type.getHeapType();
    if (type.isShared() ||
        structNew->operands.size() != type.getStruct().fields.size()) {
      return false;
    }
    shape += '(';
    shape += std::to_string(type.getID());
    for (auto* operand : structNew->operands) {
      if (auto* arrayNew = operand->dynCast<ArrayNewFixed>()) {
        if (array) {
          return false;
        }
        array = arrayNew;
        shape += "A";
        shape += std::to_string(arrayNew->type.getHeapType().getID());
      } else if (isValidLeaf(operand)) {
        shape += '.';
        shape += hashString(operand);
      } else {
        return false;
      }
    }
    shape += ')';
    return array != nullptr;
  }

  // Flattens an element into its leaves and its shape. Numeric constants are
  // only represented by their type in the shape (they may vary); other
  // leaves are part of the shape.
  bool analyzeElement(Expression* curr,
                      std::vector<Expression*>& leaves,
                      std::string& shape,
                      Index depth) {
    if (depth > MaxDepth || leaves.size() > MaxLeaves) {
      return false;
    }
    if (auto* structNew = curr->dynCast<StructNew>()) {
      if (!structNew->operands.empty()) {
        if (structNew->desc || !structNew->type.isRef()) {
          return false;
        }
        auto type = structNew->type.getHeapType();
        if (type.isShared() ||
            structNew->operands.size() != type.getStruct().fields.size()) {
          return false;
        }
        shape += '(';
        shape += std::to_string(type.getID());
        for (auto* operand : structNew->operands) {
          if (!analyzeElement(operand, leaves, shape, depth + 1)) {
            return false;
          }
        }
        shape += ')';
        return true;
      }
    }
    if (auto* arrayNew = curr->dynCast<ArrayNewFixed>()) {
      shape += '[';
      shape += std::to_string(arrayNew->type.getHeapType().getID());
      shape += ':';
      shape += std::to_string(arrayNew->values.size());
      for (auto* value : arrayNew->values) {
        if (!analyzeElement(value, leaves, shape, depth + 1)) {
          return false;
        }
      }
      shape += ']';
      return true;
    }
    if (!isValidLeaf(curr) || (curr->is<Const>() && !isNumericConst(curr))) {
      return false;
    }
    leaves.push_back(curr);
    if (isNumericConst(curr)) {
      shape += '#';
      shape += std::to_string(curr->type.getID());
    } else {
      shape += '.';
      shape += hashString(curr);
    }
    return true;
  }

  bool analyze(Global* global,
               Member& member,
               std::string& shape,
               HeapType& arrayType) {
    if (!global->type.isRef() || global->type.isNullable()) {
      return false;
    }
    ArrayNewFixed* array = nullptr;
    if (!analyzeOuter(global->init, shape, array)) {
      return false;
    }
    if (global->type.getHeapType() != global->init->type.getHeapType()) {
      return false;
    }
    arrayType = array->type.getHeapType();
    auto element = arrayType.getArray().element;
    // We build the array with array.new_default and fill it with array.set.
    if (element.mutable_ != Mutable || !element.type.isDefaultable()) {
      return false;
    }
    if (array->values.empty()) {
      return false;
    }
    shape += '{';
    std::string elementShape;
    for (Index i = 0; i < array->values.size(); i++) {
      std::vector<Expression*> leaves;
      std::string currShape;
      if (!analyzeElement(array->values[i], leaves, currShape, 0)) {
        return false;
      }
      if (i == 0) {
        elementShape = currShape;
      } else if (currShape != elementShape) {
        return false;
      }
      member.elements.push_back(std::move(leaves));
    }
    shape += elementShape;
    shape += '}';
    member.global = global;
    member.array = array;
    member.initBytes = 3 + 1 + getOperandBytes(global->init) + 1;
    return true;
  }

  template<typename T> Index getLEBSize(const T& leb) {
    BufferWithRandomAccess buffer;
    buffer << leb;
    return buffer.size();
  }

  Index getOperandBytes(Expression* curr) {
    if (auto* c = curr->dynCast<Const>()) {
      auto type = c->type;
      if (type == Type::i32) {
        return 1 + getLEBSize(S32LEB(c->value.geti32()));
      } else if (type == Type::i64) {
        return 1 + getLEBSize(S64LEB(c->value.geti64()));
      } else if (type == Type::f32) {
        return 5;
      } else if (type == Type::f64) {
        return 9;
      }
      return 17;
    }
    if (auto* null = curr->dynCast<RefNull>()) {
      return null->type.getHeapType().isBasic() ? 2 : 3;
    }
    if (curr->is<RefFunc>()) {
      return 3;
    }
    if (auto* get = curr->dynCast<GlobalGet>()) {
      return 1 + getLEBSize(U32LEB(globalIndexes[get->name]));
    }
    if (auto* i31 = curr->dynCast<RefI31>()) {
      return 2 + getOperandBytes(i31->value);
    }
    if (auto* structNew = curr->dynCast<StructNew>()) {
      Index bytes = 4;
      for (auto* operand : structNew->operands) {
        bytes += getOperandBytes(operand);
      }
      return bytes;
    }
    if (auto* arrayNew = curr->dynCast<ArrayNewFixed>()) {
      Index bytes = 4 + getLEBSize(U32LEB(arrayNew->values.size()));
      for (auto* value : arrayNew->values) {
        bytes += getOperandBytes(value);
      }
      return bytes;
    }
    return 4;
  }

  // The value of a numeric leaf as raw bits (i32/f32 in the low 32 bits).
  static uint64_t getBits(Expression* leaf) {
    auto value = leaf->cast<Const>()->value;
    if (value.type == Type::i32) {
      return uint32_t(value.geti32());
    } else if (value.type == Type::i64) {
      return uint64_t(value.geti64());
    } else if (value.type == Type::f32) {
      return value.reinterpreti32();
    } else {
      assert(value.type == Type::f64);
      return uint64_t(value.reinterpreti64());
    }
  }

  // The narrowest width (1, 2 or 4 bytes) and signedness that can hold all
  // values in [min, max] (as signed 32-bit values).
  static std::pair<Index, bool> getWidth(int64_t min, int64_t max) {
    if (min >= 0) {
      return {max <= 0xff ? 1 : max <= 0xffff ? 2 : 4, false};
    }
    if (min >= -128 && max <= 127) {
      return {1, true};
    }
    if (min >= -32768 && max <= 32767) {
      return {2, true};
    }
    return {4, false};
  }

  // The stored value of varying column c (index into plan.varying) for an
  // element, given a base column (or -1).
  static uint32_t getStored(const Plan& plan,
                            const std::vector<Expression*>& leaves,
                            Index c,
                            int base) {
    uint32_t value = uint32_t(getBits(leaves[plan.varying[c]]));
    if (base >= 0) {
      value -= uint32_t(getBits(leaves[plan.varying[base]]));
    }
    return value;
  }

  static uint32_t
  getStored(const Plan& plan, const std::vector<Expression*>& leaves, Index c) {
    return getStored(plan, leaves, c, plan.encodings[c].base);
  }

  // The width needed by the elements of a member for column c under a base.
  static std::pair<Index, bool>
  getMemberWidth(const Plan& plan, Member* member, Index c, int base) {
    int64_t min = INT64_MAX, max = INT64_MIN;
    for (auto& leaves : member->elements) {
      int64_t value = int32_t(getStored(plan, leaves, c, base));
      min = std::min(min, value);
      max = std::max(max, value);
    }
    return getWidth(min, max);
  }

  // Chooses the encodings of the varying columns for the given members and
  // computes the saving.
  void evaluate(Plan& plan) {
    auto& members = plan.members;
    plan.numElements = 0;
    for (auto* member : members) {
      plan.numElements += member->elements.size();
    }
    auto& first = members[0]->elements[0];
    Index numLeaves = first.size();
    // Varying leaves.
    plan.varying.clear();
    for (Index i = 0; i < numLeaves; i++) {
      bool varies = false;
      for (auto* member : members) {
        for (auto& leaves : member->elements) {
          if (!ExpressionAnalyzer::equal(first[i], leaves[i])) {
            varies = true;
            break;
          }
        }
        if (varies) {
          break;
        }
      }
      if (varies) {
        plan.varying.push_back(i);
      }
    }
    // Encodings.
    plan.encodings.assign(plan.varying.size(), Encoding());
    int64_t dataBytes = 0;
    for (Index c = 0; c < plan.varying.size(); c++) {
      auto type = first[plan.varying[c]]->type;
      auto& enc = plan.encodings[c];
      if (type == Type::i64 || type == Type::f64) {
        enc.width = 8;
      } else if (type == Type::f32) {
        enc.width = 4;
      } else {
        // Try raw and each earlier i32 column as a base; keep the narrowest.
        enc.width = 5;
        for (int base = -1; base < int(c); base++) {
          if (base >= 0 && first[plan.varying[base]]->type != Type::i32) {
            continue;
          }
          int64_t min = INT64_MAX, max = INT64_MIN;
          for (auto* member : members) {
            for (auto& leaves : member->elements) {
              int64_t value = int32_t(getStored(plan, leaves, c, base));
              min = std::min(min, value);
              max = std::max(max, value);
            }
          }
          auto [width, signed_] = getWidth(min, max);
          if (width < enc.width) {
            enc = {base, width, signed_};
          }
        }
      }
      dataBytes += int64_t(enc.width) * plan.numElements;
    }
    // Offsets column.
    Index offsetsWidth = plan.numElements + 1 <= 0xffff ? 2 : 4;
    dataBytes += int64_t(offsetsWidth) * (members.size() + 1);
    int64_t saving = -int64_t(GroupOverhead) -
                     int64_t(ColumnOverhead) * plan.varying.size() - dataBytes;
    // A global.get becomes a call (opcode + 2-byte function index) and an
    // i32.const.
    const Index callBytes = 3;
    for (Index k = 0; k < members.size(); k++) {
      auto* member = members[k];
      Index getBytes =
        1 + getLEBSize(U32LEB(globalIndexes[member->global->name]));
      int64_t siteDelta =
        int64_t(callBytes + 1 + getLEBSize(S32LEB(int32_t(k)))) -
        int64_t(getBytes);
      saving += int64_t(member->initBytes) - int64_t(member->uses) * siteDelta;
    }
    plan.saving = saving;
  }

  void processGroup(Group& group,
                    std::unordered_map<Name, Replacement>& replacements) {
    // Assign the indexes in first-use order.
    std::vector<Member*> members;
    for (auto& member : group.members) {
      members.push_back(&member);
    }
    std::sort(members.begin(), members.end(), [](Member* a, Member* b) {
      return a->firstUse < b->firstUse;
    });
    Plan plan;
    plan.members = members;
    evaluate(plan);

    // Exclude outliers: for each column and each possible base, the members
    // that force the widest storage; exclude them if that improves the
    // saving.
    for (Index round = 0; round < MaxOutlierRounds; round++) {
      Plan best;
      best.saving = plan.saving;
      auto& first = plan.members[0]->elements[0];
      for (Index c = 0; c < plan.varying.size(); c++) {
        if (first[plan.varying[c]]->type != Type::i32) {
          continue;
        }
        for (int base = -1; base < int(c); base++) {
          if (base >= 0 && first[plan.varying[base]]->type != Type::i32) {
            continue;
          }
          Index maxWidth = 0;
          std::vector<Index> widths;
          for (auto* member : plan.members) {
            auto [width, signed_] = getMemberWidth(plan, member, c, base);
            widths.push_back(width);
            maxWidth = std::max(maxWidth, width);
          }
          if (maxWidth == 1) {
            continue;
          }
          std::vector<Member*> kept;
          Index excludedElements = 0;
          for (Index m = 0; m < plan.members.size(); m++) {
            if (widths[m] == maxWidth) {
              excludedElements += plan.members[m]->elements.size();
            } else {
              kept.push_back(plan.members[m]);
            }
          }
          if (kept.empty() ||
              excludedElements * 100 > plan.numElements * OutlierPercent) {
            continue;
          }
          Plan candidate;
          candidate.members = kept;
          evaluate(candidate);
          if (candidate.saving > best.saving) {
            best = std::move(candidate);
          }
        }
      }
      if (best.members.empty()) {
        break;
      }
      plan = std::move(best);
    }
    if (!always && plan.saving <= 0) {
      return;
    }
    emit(group, plan, replacements);
  }

  HeapType getColumnHeapType(const Encoding& enc, Type type) {
    if (type == Type::i32) {
      if (enc.width == 1) {
        return HeapTypes::getMutI8Array();
      } else if (enc.width == 2) {
        return HeapTypes::getMutI16Array();
      }
    }
    return HeapType(Array(Field(type, Mutable)));
  }

  void emit(Group& group,
            Plan& plan,
            std::unordered_map<Name, Replacement>& replacements) {
    Builder builder(*module);
    auto& members = plan.members;
    auto resultType = group.globalType;
    auto& first = members[0]->elements[0];
    std::string baseName = "tables$" + std::to_string(numGroups++);

    // The segment: offsets, then the columns.
    std::vector<char> data;
    Index offsetsWidth = plan.numElements + 1 <= 0xffff ? 2 : 4;
    Index offset = 0;
    for (auto* member : members) {
      appendLE(data, offset, offsetsWidth);
      offset += member->elements.size();
    }
    appendLE(data, offset, offsetsWidth);
    std::vector<Index> columnOffsets;
    for (Index c = 0; c < plan.varying.size(); c++) {
      auto& enc = plan.encodings[c];
      columnOffsets.push_back(data.size());
      for (auto* member : members) {
        for (auto& leaves : member->elements) {
          uint64_t value = enc.width == 8
                             ? getBits(leaves[plan.varying[c]])
                             : uint64_t(getStored(plan, leaves, c));
          appendLE(data, value, enc.width);
        }
      }
    }
    auto segmentName = Names::getValidDataSegmentName(*module, baseName);
    module->addDataSegment(Builder::makeDataSegment(
      segmentName, Name(), nullptr, data.data(), data.size()));

    // The cache.
    auto cacheElementType = resultType.with(Nullable);
    HeapType cacheType(Array(Field(cacheElementType, Mutable)));
    auto cacheName = Names::getValidGlobalName(*module, baseName + "$cache");
    auto cacheGlobalType = Type(cacheType, NonNullable);
    module->addGlobal(builder.makeGlobal(
      cacheName,
      cacheGlobalType,
      builder.makeArrayNew(cacheType,
                           builder.makeConst(int32_t(members.size()))),
      Builder::Immutable));

    // The factory.
    auto func =
      builder.makeFunction(Names::getValidFunctionName(*module, baseName),
                           Signature(Type::i32, resultType),
                           {});
    Index kLocal = 0;
    Index offLocal = Builder::addVar(func.get(), Type::i32);
    Index nLocal = Builder::addVar(func.get(), Type::i32);
    Index iLocal = Builder::addVar(func.get(), Type::i32);
    auto offsetsType = offsetsWidth == 2
                         ? HeapTypes::getMutI16Array()
                         : HeapType(Array(Field(Type::i32, Mutable)));
    Index offsetsLocal =
      Builder::addVar(func.get(), Type(offsetsType, NonNullable));
    std::vector<Index> columnLocals;
    std::vector<HeapType> columnTypes;
    for (Index c = 0; c < plan.varying.size(); c++) {
      auto type =
        getColumnHeapType(plan.encodings[c], first[plan.varying[c]]->type);
      columnTypes.push_back(type);
      columnLocals.push_back(
        Builder::addVar(func.get(), Type(type, NonNullable)));
    }
    auto* arrayNew = builder.makeArrayNew(
      group.arrayType, builder.makeLocalGet(nLocal, Type::i32));
    Index arrLocal = Builder::addVar(func.get(), arrayNew->type);
    Index resLocal = Builder::addVar(func.get(), resultType);

    auto i32 = Type::i32;
    auto k = [&]() { return builder.makeLocalGet(kLocal, i32); };
    auto i = [&]() { return builder.makeLocalGet(iLocal, i32); };
    auto n = [&]() { return builder.makeLocalGet(nLocal, i32); };
    auto off = [&]() { return builder.makeLocalGet(offLocal, i32); };
    auto offsets = [&]() {
      return builder.makeLocalGet(offsetsLocal, Type(offsetsType, NonNullable));
    };

    std::vector<Expression*> list;
    Name label("cached");
    list.push_back(builder.makeBrOn(
      BrOnNonNull,
      label,
      builder.makeArrayGet(builder.makeGlobalGet(cacheName, cacheGlobalType),
                           k(),
                           MemoryOrder::Unordered,
                           cacheElementType)));
    list.push_back(
      builder.makeLocalSet(offsetsLocal,
                           builder.makeArrayNewData(
                             offsetsType,
                             segmentName,
                             builder.makeConst(int32_t(0)),
                             builder.makeConst(int32_t(members.size() + 1)))));
    list.push_back(builder.makeLocalSet(
      offLocal,
      builder.makeArrayGet(offsets(), k(), MemoryOrder::Unordered, i32)));
    list.push_back(builder.makeLocalSet(
      nLocal,
      builder.makeBinary(
        SubInt32,
        builder.makeArrayGet(
          offsets(),
          builder.makeBinary(AddInt32, k(), builder.makeConst(int32_t(1))),
          MemoryOrder::Unordered,
          i32),
        off())));
    for (Index c = 0; c < plan.varying.size(); c++) {
      auto& enc = plan.encodings[c];
      Expression* byteOffset = off();
      if (enc.width > 1) {
        byteOffset = builder.makeBinary(
          MulInt32, byteOffset, builder.makeConst(int32_t(enc.width)));
      }
      if (columnOffsets[c] > 0) {
        byteOffset = builder.makeBinary(
          AddInt32, byteOffset, builder.makeConst(int32_t(columnOffsets[c])));
      }
      list.push_back(
        builder.makeLocalSet(columnLocals[c],
                             builder.makeArrayNewData(
                               columnTypes[c], segmentName, byteOffset, n())));
    }
    list.push_back(builder.makeLocalSet(arrLocal, arrayNew));

    // The element, reading the varying leaves from the columns.
    std::function<Expression*(Index)> read = [&](Index c) -> Expression* {
      auto& enc = plan.encodings[c];
      auto type = first[plan.varying[c]]->type;
      Expression* value = builder.makeArrayGet(
        builder.makeLocalGet(columnLocals[c],
                             Type(columnTypes[c], NonNullable)),
        i(),
        MemoryOrder::Unordered,
        type,
        enc.signed_);
      if (enc.base >= 0) {
        value = builder.makeBinary(AddInt32, read(enc.base), value);
      }
      return value;
    };
    std::unordered_map<Index, Index> leafToColumn;
    for (Index c = 0; c < plan.varying.size(); c++) {
      leafToColumn[plan.varying[c]] = c;
    }
    Index leaf = 0;
    std::function<Expression*(Expression*)> rebuild =
      [&](Expression* templ) -> Expression* {
      if (auto* structNew = templ->dynCast<StructNew>();
          structNew && !structNew->operands.empty()) {
        std::vector<Expression*> operands;
        for (auto* operand : structNew->operands) {
          operands.push_back(rebuild(operand));
        }
        return builder.makeStructNew(structNew->type.getHeapType(), operands);
      }
      if (auto* arrayNew = templ->dynCast<ArrayNewFixed>()) {
        std::vector<Expression*> values;
        for (auto* value : arrayNew->values) {
          values.push_back(rebuild(value));
        }
        return builder.makeArrayNewFixed(arrayNew->type.getHeapType(), values);
      }
      Index index = leaf++;
      if (auto iter = leafToColumn.find(index); iter != leafToColumn.end()) {
        return read(iter->second);
      }
      return ExpressionManipulator::copy(templ, *module);
    };
    auto* element = rebuild(members[0]->array->values[0]);
    assert(leaf == first.size());

    Name loopName("fill");
    auto* loop = builder.makeLoop(
      loopName,
      builder.makeBlock(
        {builder.makeArraySet(builder.makeLocalGet(arrLocal, arrayNew->type),
                              i(),
                              element,
                              MemoryOrder::Unordered),
         builder.makeBreak(
           loopName,
           nullptr,
           builder.makeBinary(
             LtUInt32,
             builder.makeLocalTee(
               iLocal,
               builder.makeBinary(AddInt32, i(), builder.makeConst(int32_t(1))),
               i32),
             n()))}));
    list.push_back(loop);

    // The outer object.
    Expression* result;
    auto* init = members[0]->global->init;
    if (init->is<ArrayNewFixed>()) {
      result = builder.makeLocalGet(arrLocal, arrayNew->type);
    } else {
      auto* structNew = init->cast<StructNew>();
      std::vector<Expression*> operands;
      for (auto* operand : structNew->operands) {
        if (operand->is<ArrayNewFixed>()) {
          operands.push_back(builder.makeLocalGet(arrLocal, arrayNew->type));
        } else {
          operands.push_back(ExpressionManipulator::copy(operand, *module));
        }
      }
      result = builder.makeStructNew(structNew->type.getHeapType(), operands);
    }
    list.push_back(
      builder.makeArraySet(builder.makeGlobalGet(cacheName, cacheGlobalType),
                           k(),
                           builder.makeLocalTee(resLocal, result, resultType),
                           MemoryOrder::Unordered));
    list.push_back(builder.makeLocalGet(resLocal, resultType));
    func->body = builder.makeBlock(label, list, resultType);
    auto factoryName = func->name;
    module->addFunction(std::move(func));

    for (Index k = 0; k < members.size(); k++) {
      replacements[members[k]->global->name] = {factoryName, k};
    }
  }
};

} // anonymous namespace

Pass* createRollConstantTablesPass() { return new RollConstantTables(); }

} // namespace wasm
