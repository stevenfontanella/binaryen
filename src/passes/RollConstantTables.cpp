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
// When all varying leaves are i32, a "varint" layout is considered as well:
// the segment holds [byte offsets (M+1)][one ULEB stream per table], where
// each table's stream is its element count followed by, per element and
// column, the ULEB (zigzag-encoded if negative residuals occur) of the value
// minus an optional predictor: an earlier column of the same element and/or a
// column of the previous element (the previous element's values start at 0
// for each table). Sorted range tables thus store (gap, length, category)
// bytes. The factory loads the table's byte range as an i8 array and decodes
// it with a shared ULEB reader ($tables$read, position in the mutable global
// $tables$pos). The layout with the smaller model size wins;
// --pass-arg=roll-constant-tables-mode@fixed|varint forces one.
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
// Varint layout: the byte slice, the position reset and the count read in the
// factory; per column the read call, the predictor adds and the local; the
// zigzag decoding of a signed column; the shared reader function and its
// position global (once per module).
constexpr Index VarintGroupOverhead = 30;
constexpr Index VarintColumnOverhead = 9;
constexpr Index ZigzagOverhead = 14;
constexpr Index ReaderOverhead = 75;
// At most this fraction of the elements may be excluded as outliers in one
// step.
constexpr Index OutlierPercent = 10;
constexpr Index MaxOutlierRounds = 6;

void appendLE(std::vector<char>& data, uint64_t value, Index width) {
  for (Index i = 0; i < width; i++) {
    data.push_back(char(uint8_t(value >> (8 * i))));
  }
}

Index getULEBSize(uint32_t value) {
  Index bytes = 1;
  while (value >= 128) {
    value >>= 7;
    bytes++;
  }
  return bytes;
}

void appendULEB(std::vector<char>& data, uint32_t value) {
  while (true) {
    uint8_t byte = value & 127;
    value >>= 7;
    if (value) {
      data.push_back(char(byte | 128));
    } else {
      data.push_back(char(byte));
      return;
    }
  }
}

uint32_t zigzag(uint32_t value) {
  int32_t s = int32_t(value);
  return (uint32_t(s) << 1) ^ uint32_t(s >> 31);
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
    // 1, 2, 4 or 8 bytes (fixed layout only).
    Index width = 4;
    // Fixed layout: signed reads. Varint layout: zigzag-encoded residuals.
    bool signed_ = false;
    // Varint layout only: the column whose value of the previous element is
    // subtracted as well (>= this column, or -1).
    int prevBase = -1;
  };

  struct Plan {
    std::vector<Member*> members;
    // The varying leaf indexes.
    std::vector<Index> varying;
    // Per varying leaf (same order as varying), the encoding.
    std::vector<Encoding> encodings;
    Index numElements = 0;
    int64_t saving = 0;
    // Whether the columns are stored as ULEB streams.
    bool varint = false;
  };

  struct Replacement {
    Name factory;
    Index index;
  };

  using GroupKey = std::tuple<Type, std::string>;

  enum class Mode { Auto, Fixed, Varint };

  Module* module = nullptr;
  std::unordered_map<Name, Index> globalIndexes;
  Index numGroups = 0;
  // Ignore the size model (for testing).
  bool always = false;
  Mode mode = Mode::Auto;
  // The shared ULEB reader, created on demand.
  Name readerName;
  Name posName;

  void run(Module* module_) override {
    module = module_;
    if (!module->features.hasGC() || !module->features.hasBulkMemory()) {
      return;
    }
    always = !getArgumentOrDefault("roll-constant-tables-always", "").empty();
    auto modeArg = getArgumentOrDefault("roll-constant-tables-mode", "auto");
    if (modeArg == "fixed") {
      mode = Mode::Fixed;
    } else if (modeArg == "varint") {
      mode = Mode::Varint;
    } else if (modeArg != "auto") {
      Fatal() << "Invalid roll-constant-tables-mode: " << modeArg;
    }
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

  // The residual stored for varying column c of an element in the varint
  // layout, given the previous element of the same table (or nullptr).
  static uint32_t getResidual(const Plan& plan,
                              const std::vector<Expression*>& leaves,
                              const std::vector<Expression*>* prevLeaves,
                              Index c,
                              int base,
                              int prevBase) {
    uint32_t value = uint32_t(getBits(leaves[plan.varying[c]]));
    if (base >= 0) {
      value -= uint32_t(getBits(leaves[plan.varying[base]]));
    }
    if (prevBase >= 0 && prevLeaves) {
      value -= uint32_t(getBits((*prevLeaves)[plan.varying[prevBase]]));
    }
    return value;
  }

  static uint32_t getResidual(const Plan& plan,
                              const std::vector<Expression*>& leaves,
                              const std::vector<Expression*>* prevLeaves,
                              Index c) {
    auto& enc = plan.encodings[c];
    auto value =
      getResidual(plan, leaves, prevLeaves, c, enc.base, enc.prevBase);
    return enc.signed_ ? zigzag(value) : value;
  }

  // The bytes of column c under the given predictors, and whether zigzag
  // encoding is needed.
  std::pair<int64_t, bool>
  getVarintColumnBytes(const Plan& plan, Index c, int base, int prevBase) {
    bool negative = false;
    for (auto* member : plan.members) {
      const std::vector<Expression*>* prev = nullptr;
      for (auto& leaves : member->elements) {
        if (int32_t(getResidual(plan, leaves, prev, c, base, prevBase)) < 0) {
          negative = true;
          break;
        }
        prev = &leaves;
      }
      if (negative) {
        break;
      }
    }
    int64_t bytes = 0;
    for (auto* member : plan.members) {
      const std::vector<Expression*>* prev = nullptr;
      for (auto& leaves : member->elements) {
        auto value = getResidual(plan, leaves, prev, c, base, prevBase);
        bytes += getULEBSize(negative ? zigzag(value) : value);
        prev = &leaves;
      }
    }
    return {bytes, negative};
  }

  // Computes the element count and the varying leaves of a plan.
  void computeVarying(Plan& plan) {
    auto& members = plan.members;
    plan.numElements = 0;
    for (auto* member : members) {
      plan.numElements += member->elements.size();
    }
    auto& first = members[0]->elements[0];
    Index numLeaves = first.size();
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
  }

  // The saving of a plan given its data bytes and fixed overhead.
  int64_t getSaving(const Plan& plan, int64_t overhead, int64_t dataBytes) {
    auto& members = plan.members;
    int64_t saving = -overhead - dataBytes;
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
    return saving;
  }

  // Chooses the varint encodings of the varying columns for the given members
  // and computes the saving. All varying leaves must be i32.
  void evaluateVarint(Plan& plan) {
    plan.varint = true;
    computeVarying(plan);
    auto& first = plan.members[0]->elements[0];
    for (auto index : plan.varying) {
      if (first[index]->type != Type::i32) {
        plan.saving = INT64_MIN;
        return;
      }
    }
    plan.encodings.assign(plan.varying.size(), Encoding());
    int64_t streamBytes = 0;
    int64_t overhead = GroupOverhead + VarintGroupOverhead +
                       (readerName.is() ? 0 : ReaderOverhead);
    for (auto* member : plan.members) {
      streamBytes += getULEBSize(member->elements.size());
    }
    Index numColumns = plan.varying.size();
    for (Index c = 0; c < numColumns; c++) {
      int64_t best = INT64_MAX;
      for (int base = -1; base < int(c); base++) {
        for (int prevBase = -1; prevBase < int(numColumns); prevBase++) {
          if (prevBase >= 0 && prevBase < int(c)) {
            continue;
          }
          auto [bytes, negative] =
            getVarintColumnBytes(plan, c, base, prevBase);
          if (negative) {
            bytes += ZigzagOverhead;
          }
          if (bytes < best) {
            best = bytes;
            plan.encodings[c] = {base, 0, negative, prevBase};
          }
        }
      }
      streamBytes += best;
      overhead += VarintColumnOverhead;
      if (plan.encodings[c].signed_) {
        streamBytes -= ZigzagOverhead;
        overhead += ZigzagOverhead;
      }
    }
    Index numOffsets = plan.members.size() + 1;
    Index offsetsWidth = 2 * numOffsets + streamBytes <= 0xffff ? 2 : 4;
    plan.saving =
      getSaving(plan, overhead, streamBytes + offsetsWidth * numOffsets);
  }

  // Chooses the fixed-width encodings of the varying columns for the given
  // members and computes the saving.
  void evaluate(Plan& plan) {
    auto& members = plan.members;
    computeVarying(plan);
    auto& first = members[0]->elements[0];
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
    plan.saving = getSaving(
      plan, GroupOverhead + ColumnOverhead * plan.varying.size(), dataBytes);
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
    // The varint layout over all members (ULEBs adapt to outliers).
    if (mode != Mode::Fixed) {
      Plan varint;
      varint.members = members;
      evaluateVarint(varint);
      if (varint.saving != INT64_MIN &&
          (mode == Mode::Varint || varint.saving > plan.saving)) {
        plan = std::move(varint);
      }
    }
    if (!always && plan.saving <= 0) {
      return;
    }
    emit(group, plan, replacements);
  }

  // Creates the shared ULEB reader (once):
  //
  //   (global $tables$pos (mut i32))
  //   (func $tables$read (param $b (ref $i8)) (result i32)
  //     loop { byte = b[pos]; pos++; v |= (byte & 127) << shift; shift += 7;
  //            br_if (byte & 128) } v)
  void ensureReader() {
    if (readerName.is()) {
      return;
    }
    Builder builder(*module);
    auto i32 = Type::i32;
    posName = Names::getValidGlobalName(*module, "tables$pos");
    module->addGlobal(builder.makeGlobal(
      posName, i32, builder.makeConst(int32_t(0)), Builder::Mutable));
    auto bytesType = Type(HeapTypes::getMutI8Array(), NonNullable);
    auto func =
      builder.makeFunction(Names::getValidFunctionName(*module, "tables$read"),
                           Signature(bytesType, i32),
                           {});
    Index bLocal = 0;
    Index vLocal = Builder::addVar(func.get(), i32);
    Index shiftLocal = Builder::addVar(func.get(), i32);
    Index byteLocal = Builder::addVar(func.get(), i32);
    auto pos = [&]() { return builder.makeGlobalGet(posName, i32); };
    auto byte = [&]() { return builder.makeLocalGet(byteLocal, i32); };
    Name loopName("read");
    auto* loop = builder.makeLoop(
      loopName,
      builder.makeBlock(
        {builder.makeLocalSet(
           byteLocal,
           builder.makeArrayGet(builder.makeLocalGet(bLocal, bytesType),
                                pos(),
                                MemoryOrder::Unordered,
                                i32,
                                false)),
         builder.makeGlobalSet(
           posName,
           builder.makeBinary(AddInt32, pos(), builder.makeConst(int32_t(1)))),
         builder.makeLocalSet(
           vLocal,
           builder.makeBinary(
             OrInt32,
             builder.makeLocalGet(vLocal, i32),
             builder.makeBinary(
               ShlInt32,
               builder.makeBinary(
                 AndInt32, byte(), builder.makeConst(int32_t(127))),
               builder.makeLocalGet(shiftLocal, i32)))),
         builder.makeLocalSet(
           shiftLocal,
           builder.makeBinary(AddInt32,
                              builder.makeLocalGet(shiftLocal, i32),
                              builder.makeConst(int32_t(7)))),
         builder.makeBreak(
           loopName,
           nullptr,
           builder.makeBinary(
             AndInt32, byte(), builder.makeConst(int32_t(128))))}));
    func->body =
      builder.makeBlock({loop, builder.makeLocalGet(vLocal, i32)}, i32);
    readerName = func->name;
    module->addFunction(std::move(func));
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

    // The segment: offsets, then the columns (fixed) or the per-table streams
    // (varint).
    std::vector<char> data;
    Index offsetsWidth = 2;
    std::vector<Index> columnOffsets;
    if (plan.varint) {
      ensureReader();
      // The streams, then the byte offsets in front of them.
      std::vector<char> streams;
      std::vector<Index> byteOffsets;
      for (auto* member : members) {
        byteOffsets.push_back(streams.size());
        appendULEB(streams, member->elements.size());
        const std::vector<Expression*>* prev = nullptr;
        for (auto& leaves : member->elements) {
          for (Index c = 0; c < plan.varying.size(); c++) {
            appendULEB(streams, getResidual(plan, leaves, prev, c));
          }
          prev = &leaves;
        }
      }
      byteOffsets.push_back(streams.size());
      Index headerBytes = 2 * byteOffsets.size();
      if (headerBytes + streams.size() > 0xffff) {
        offsetsWidth = 4;
        headerBytes = 4 * byteOffsets.size();
      }
      for (auto byteOffset : byteOffsets) {
        appendLE(data, headerBytes + byteOffset, offsetsWidth);
      }
      data.insert(data.end(), streams.begin(), streams.end());
    } else {
      offsetsWidth = plan.numElements + 1 <= 0xffff ? 2 : 4;
      Index offset = 0;
      for (auto* member : members) {
        appendLE(data, offset, offsetsWidth);
        offset += member->elements.size();
      }
      appendLE(data, offset, offsetsWidth);
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
    auto bytesType = Type(HeapTypes::getMutI8Array(), NonNullable);
    Index bytesLocal = Index(-1);
    Index tmpLocal = Index(-1);
    if (plan.varint) {
      bytesLocal = Builder::addVar(func.get(), bytesType);
      for (auto& enc : plan.encodings) {
        if (enc.signed_) {
          tmpLocal = Builder::addVar(func.get(), Type::i32);
          break;
        }
      }
      // The columns are decoded into i32 locals, which also hold the previous
      // element's values (0 at the start).
      for (Index c = 0; c < plan.varying.size(); c++) {
        columnLocals.push_back(Builder::addVar(func.get(), Type::i32));
      }
    } else {
      for (Index c = 0; c < plan.varying.size(); c++) {
        auto type =
          getColumnHeapType(plan.encodings[c], first[plan.varying[c]]->type);
        columnTypes.push_back(type);
        columnLocals.push_back(
          Builder::addVar(func.get(), Type(type, NonNullable)));
      }
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
    auto* nextOffset = builder.makeArrayGet(
      offsets(),
      builder.makeBinary(AddInt32, k(), builder.makeConst(int32_t(1))),
      MemoryOrder::Unordered,
      i32);
    // The statements decoding one element in the loop (varint).
    std::vector<Expression*> decodes;
    if (plan.varint) {
      // bytes = the table's byte range; pos = 0; n = read().
      list.push_back(builder.makeLocalSet(
        bytesLocal,
        builder.makeArrayNewData(
          HeapTypes::getMutI8Array(),
          segmentName,
          off(),
          builder.makeBinary(SubInt32, nextOffset, off()))));
      list.push_back(
        builder.makeGlobalSet(posName, builder.makeConst(int32_t(0))));
      auto read = [&]() {
        return builder.makeCall(
          readerName, {builder.makeLocalGet(bytesLocal, bytesType)}, i32);
      };
      list.push_back(builder.makeLocalSet(nLocal, read()));
      for (Index c = 0; c < plan.varying.size(); c++) {
        auto& enc = plan.encodings[c];
        Expression* value = read();
        if (enc.signed_) {
          // Zigzag: (tmp >>> 1) ^ -(tmp & 1)
          value = builder.makeBinary(
            XorInt32,
            builder.makeBinary(ShrUInt32,
                               builder.makeLocalTee(tmpLocal, value, i32),
                               builder.makeConst(int32_t(1))),
            builder.makeBinary(
              SubInt32,
              builder.makeConst(int32_t(0)),
              builder.makeBinary(AndInt32,
                                 builder.makeLocalGet(tmpLocal, i32),
                                 builder.makeConst(int32_t(1)))));
        }
        if (enc.base >= 0) {
          value = builder.makeBinary(
            AddInt32, value, builder.makeLocalGet(columnLocals[enc.base], i32));
        }
        if (enc.prevBase >= 0) {
          // Not yet overwritten in this iteration (prevBase >= c).
          assert(enc.prevBase >= int(c));
          value = builder.makeBinary(
            AddInt32,
            value,
            builder.makeLocalGet(columnLocals[enc.prevBase], i32));
        }
        decodes.push_back(builder.makeLocalSet(columnLocals[c], value));
      }
    } else {
      list.push_back(builder.makeLocalSet(
        nLocal, builder.makeBinary(SubInt32, nextOffset, off())));
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
        list.push_back(builder.makeLocalSet(
          columnLocals[c],
          builder.makeArrayNewData(
            columnTypes[c], segmentName, byteOffset, n())));
      }
    }
    list.push_back(builder.makeLocalSet(arrLocal, arrayNew));

    // The element, reading the varying leaves from the columns.
    std::function<Expression*(Index)> read = [&](Index c) -> Expression* {
      if (plan.varint) {
        return builder.makeLocalGet(columnLocals[c], i32);
      }
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
    auto body = decodes;
    body.push_back(
      builder.makeArraySet(builder.makeLocalGet(arrLocal, arrayNew->type),
                           i(),
                           element,
                           MemoryOrder::Unordered));
    body.push_back(builder.makeBreak(
      loopName,
      nullptr,
      builder.makeBinary(
        LtUInt32,
        builder.makeLocalTee(
          iLocal,
          builder.makeBinary(AddInt32, i(), builder.makeConst(int32_t(1))),
          i32),
        n())));
    auto* loop = builder.makeLoop(loopName, builder.makeBlock(body));
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
