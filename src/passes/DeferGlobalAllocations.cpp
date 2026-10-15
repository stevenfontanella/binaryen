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
// Defers the allocation of constant global objects to their first use.
//
// Many immutable globals are initialized with a struct.new of constant
// operands (string literal wrappers, enum constants, class metadata) and are
// only read by global.get in function code. Each such global costs a full
// global entry (type, mutability, struct.new, operands, end) even if it is
// used only once.
//
// This pass groups such globals by struct type and by the shape of their
// initializer (nested constant struct.news are allowed, e.g. an enum constant
// holding a freshly allocated string wrapper). The initializer is flattened
// into a list of constant leaves; the leaves that are identical in all
// members stay in a shared factory function, while the leaves that vary
// become columns of immutable array "tables" (one array.new_fixed per
// column). A mutable array caches the created objects so that each original
// global still corresponds to exactly one object (identity and mutable fields
// are preserved). The factory is
//
//   (func $factory (param $i i32) (result (ref $T))
//     (block $cached (result (ref $T))
//       (br_on_non_null $cached (array.get $cache (local.get $i)))
//       (array.set $cache (local.get $i)
//         (local.tee $obj (struct.new $T fixed... (array.get $column i)...)))
//       (local.get $obj)))
//
// and every (global.get $g_k) becomes (call $factory (i32.const k)). A global
// is deferred only when the model says it saves bytes, i.e. when its global
// entry is larger than its table bytes plus the extra bytes at its use sites,
// so globals with many uses stay eager. Groups are chunked so that every
// array.new_fixed stays below engine limits.
//
// Uses from other global initializers, segments, and exports keep a global
// eager. Groups whose members are mostly used as struct.new operands (e.g.
// vtables) are skipped by default, as a call on every allocation is a runtime
// cost; --pass-arg=defer-global-allocations-allocation-operands@1 includes
// them.
//
// --pass-arg=defer-global-allocations-import-bonus@N adds N bytes of
// estimated savings for every imported global whose only uses are in the
// initializer of a candidate: after deferral the import is only read from a
// table column, which allows a later pass (e.g. --string-consts-to-data) to
// remove the import entirely.
//
// This moves the allocation from instantiation to first use, and adds a call,
// an array read and a null check to every use. It is meant for the very end
// of a size-focused pipeline (before reorder-globals/reorder-functions).
//

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <tuple>

#include "ir/module-utils.h"
#include "ir/names.h"
#include "ir/utils.h"
#include "pass.h"
#include "support/hash.h"
#include "support/insert_ordered.h"
#include "wasm-binary.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

// Keep array.new_fixed well below the V8 limit of 10000 operands, and keep
// the indexes at most 2 bytes as an SLEB.
constexpr Index MaxChunkSize = 8192;

struct DeferGlobalAllocations : public Pass {
  // Per-function information about global.get uses.
  struct FunctionInfo {
    // All global.gets of candidates, in traversal order.
    std::vector<Expression**> gets;
    // Candidates used directly as operands of a struct.new.
    std::unordered_map<Name, Index> allocOperandUses;
    // Uses of imported globals.
    std::unordered_map<Name, Index> importUses;
  };

  struct Candidate {
    Global* global;
    StructNew* init;
    // The constant leaves of the initializer, in a pre-order traversal of the
    // nested struct.news.
    std::vector<Expression*> leaves;
    // Estimated size of the global entry in the binary.
    Index initBytes = 0;
    // Estimated bytes saved elsewhere (imports) if this global is deferred.
    Index bonus = 0;
    // Code uses.
    Index uses = 0;
    Index allocOperandUses = 0;
    // Order of first use in the code.
    Index firstUse = Index(-1);
  };

  struct Group {
    HeapType type;
    Type globalType;
    // The field each leaf is stored in.
    std::vector<Field> leafFields;
    std::vector<Candidate> members;
  };

  struct Deferral {
    Name factory;
    Index index;
  };

  using GroupKey = std::tuple<HeapType, Type, std::string>;

  Module* module;
  std::unordered_map<Name, Index> globalIndexes;
  std::unordered_map<Name, Index> functionIndexes;
  bool includeAllocationOperands = false;
  // Ignore the size model and defer every candidate (for testing, and for
  // startup-focused builds).
  bool always = false;
  Index importBonus = 0;
  Index numFactories = 0;
  Name segmentName;
  std::vector<char> segmentData;

  void run(Module* module_) override {
    module = module_;
    if (!module->features.hasGC()) {
      return;
    }
    segmentName = Names::getValidDataSegmentName(*module, "defer$data");
    includeAllocationOperands =
      !getArgumentOrDefault("defer-global-allocations-allocation-operands", "")
         .empty();
    always =
      !getArgumentOrDefault("defer-global-allocations-always", "").empty();
    importBonus = std::stoul(
      getArgumentOrDefault("defer-global-allocations-import-bonus", "0"));

    computeIndexes();

    // Find the structurally valid candidates and group them by type and
    // shape.
    InsertOrderedMap<GroupKey, Group> groups;
    std::unordered_map<Name, std::pair<Group*, Index>> candidateMap;
    for (auto& global : module->globals) {
      if (global->imported() || global->mutable_) {
        continue;
      }
      auto* init = global->init->dynCast<StructNew>();
      if (!init || init->desc || !init->type.isRef()) {
        continue;
      }
      auto type = init->type.getHeapType();
      // The global's type must be the struct type itself (possibly exact), so
      // that a call to the factory can have the same type as the global.get.
      if (type.isShared() || !global->type.isRef() ||
          global->type.isNullable() || global->type.getHeapType() != type ||
          init->operands.size() != type.getStruct().fields.size()) {
        continue;
      }
      Candidate candidate;
      std::vector<Field> leafFields;
      std::string shape;
      if (!flatten(init, candidate.leaves, leafFields, shape)) {
        continue;
      }
      candidate.global = global.get();
      candidate.init = init;
      candidate.initBytes = getGlobalEntryBytes(init);
      auto& group = groups[{type, global->type, shape}];
      group.type = type;
      group.globalType = global->type;
      group.leafFields = std::move(leafFields);
      group.members.push_back(std::move(candidate));
      candidateMap[global->name] = {&group, Index(group.members.size() - 1)};
    }
    if (groups.empty()) {
      return;
    }

    // Globals that are used outside of function code must stay eager. Also
    // count the uses of imported globals: those used only from a single
    // candidate's initializer may become removable once that candidate is
    // deferred.
    std::unordered_set<Name> nonCodeUses;
    // Uses of imports anywhere except in the initializers of candidates.
    std::unordered_map<Name, Index> importOtherUses;
    // Uses of imports in the initializers of candidates.
    std::unordered_map<Name, Index> importCandidateUses;
    {
      struct Scanner : public PostWalker<Scanner> {
        DeferGlobalAllocations& parent;
        std::unordered_set<Name>& names;
        std::unordered_map<Name, Index>* importUses = nullptr;
        Scanner(DeferGlobalAllocations& parent, std::unordered_set<Name>& names)
          : parent(parent), names(names) {}
        void visitGlobalGet(GlobalGet* curr) {
          names.insert(curr->name);
          if (parent.module->getGlobal(curr->name)->imported()) {
            (*importUses)[curr->name]++;
          }
        }
      } scanner(*this, nonCodeUses);
      for (auto& global : module->globals) {
        if (!global->imported()) {
          scanner.importUses = candidateMap.count(global->name)
                                 ? &importCandidateUses
                                 : &importOtherUses;
          scanner.walk(global->init);
        }
      }
      scanner.importUses = &importOtherUses;
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
          importOtherUses[*ex->getInternalName()]++;
        }
      }
    }

    // Count the uses in function code, in parallel.
    ModuleUtils::ParallelFunctionAnalysis<FunctionInfo> analysis(
      *module, [&](Function* func, FunctionInfo& info) {
        if (func->imported()) {
          return;
        }
        struct Scanner : public PostWalker<Scanner> {
          Module& module;
          FunctionInfo& info;
          const std::unordered_map<Name, std::pair<Group*, Index>>& candidates;
          Scanner(Module& module,
                  FunctionInfo& info,
                  const std::unordered_map<Name, std::pair<Group*, Index>>&
                    candidates)
            : module(module), info(info), candidates(candidates) {}
          void visitGlobalGet(GlobalGet* curr) {
            if (candidates.count(curr->name)) {
              info.gets.push_back(getCurrentPointer());
            } else if (module.getGlobal(curr->name)->imported()) {
              info.importUses[curr->name]++;
            }
          }
          void visitStructNew(StructNew* curr) {
            for (auto* operand : curr->operands) {
              if (auto* get = operand->dynCast<GlobalGet>()) {
                if (candidates.count(get->name)) {
                  info.allocOperandUses[get->name]++;
                }
              }
            }
          }
        };
        Scanner(*module, info, candidateMap).walk(func->body);
      });

    // Merge the per-function information in a deterministic order.
    std::vector<Expression**> allGets;
    for (auto& func : module->functions) {
      auto& info = analysis.map[func.get()];
      for (auto** ptr : info.gets) {
        auto& [group, i] = candidateMap[(*ptr)->cast<GlobalGet>()->name];
        auto& candidate = group->members[i];
        if (candidate.uses == 0) {
          candidate.firstUse = allGets.size();
        }
        candidate.uses++;
        allGets.push_back(ptr);
      }
      for (auto& [name, count] : info.allocOperandUses) {
        auto& [group, i] = candidateMap[name];
        group->members[i].allocOperandUses += count;
      }
      for (auto& [name, count] : info.importUses) {
        importOtherUses[name] += count;
      }
    }

    // Remove the candidates with non-code uses, and compute the bonuses.
    for (auto& [key, group] : groups) {
      auto& members = group.members;
      members.erase(std::remove_if(members.begin(),
                                   members.end(),
                                   [&](const Candidate& c) {
                                     return nonCodeUses.count(c.global->name);
                                   }),
                    members.end());
      if (importBonus == 0) {
        continue;
      }
      for (auto& member : members) {
        std::unordered_map<Name, Index> counts;
        for (auto* leaf : member.leaves) {
          if (auto* get = leaf->dynCast<GlobalGet>()) {
            if (module->getGlobal(get->name)->imported()) {
              counts[get->name]++;
            }
          }
        }
        for (auto& [name, count] : counts) {
          // All uses of the import must be in this initializer.
          if (!importOtherUses.count(name) &&
              importCandidateUses[name] == count) {
            member.bonus += importBonus;
          }
        }
      }
    }

    // Decide what to defer, and build the factories.
    std::unordered_map<Name, Deferral> deferrals;
    for (auto& [key, group] : groups) {
      processGroup(group, deferrals);
    }
    if (deferrals.empty()) {
      return;
    }

    // Replace the uses and remove the globals.
    Builder builder(*module);
    for (auto** ptr : allGets) {
      auto* get = (*ptr)->cast<GlobalGet>();
      auto iter = deferrals.find(get->name);
      if (iter == deferrals.end()) {
        continue;
      }
      auto& deferral = iter->second;
      *ptr = builder.makeCall(deferral.factory,
                              {builder.makeConst(int32_t(deferral.index))},
                              get->type);
    }
    for (auto& [name, _] : deferrals) {
      module->removeGlobal(name);
    }
    if (!segmentData.empty()) {
      module->addDataSegment(Builder::makeDataSegment(
        segmentName, Name(), nullptr, segmentData.data(), segmentData.size()));
    }
  }

private:
  void computeIndexes() {
    Index importedGlobals = 0, definedGlobals = 0;
    for (auto& global : module->globals) {
      if (global->imported()) {
        importedGlobals++;
      } else {
        definedGlobals++;
      }
    }
    Index nextImported = 0, nextDefined = importedGlobals;
    for (auto& global : module->globals) {
      globalIndexes[global->name] =
        global->imported() ? nextImported++ : nextDefined++;
    }
    Index importedFuncs = 0;
    for (auto& func : module->functions) {
      if (func->imported()) {
        importedFuncs++;
      }
    }
    nextImported = 0;
    nextDefined = importedFuncs;
    for (auto& func : module->functions) {
      functionIndexes[func->name] =
        func->imported() ? nextImported++ : nextDefined++;
    }
  }

  // Whether a struct.new is a nested allocation with operands that we can
  // flatten (as opposed to a struct.new_default, which is a leaf).
  static bool isNestedNew(Expression* curr) {
    auto* structNew = curr->dynCast<StructNew>();
    return structNew && !structNew->operands.empty();
  }

  // Whether an operand is a constant we can move into the factory or a table.
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

  // Flattens the operands of a struct.new (recursively through nested
  // struct.news) into leaves. The shape string identifies the nesting
  // structure, so that members of a group have the same leaves.
  bool flatten(StructNew* curr,
               std::vector<Expression*>& leaves,
               std::vector<Field>& leafFields,
               std::string& shape) {
    if (curr->desc || !curr->type.isRef()) {
      return false;
    }
    auto type = curr->type.getHeapType();
    if (type.isShared()) {
      return false;
    }
    auto& fields = type.getStruct().fields;
    if (curr->operands.size() != fields.size()) {
      return false;
    }
    shape += '(';
    shape += std::to_string(type.getID());
    for (Index i = 0; i < fields.size(); i++) {
      auto* operand = curr->operands[i];
      if (isNestedNew(operand)) {
        if (!flatten(operand->cast<StructNew>(), leaves, leafFields, shape)) {
          return false;
        }
      } else if (isValidLeaf(operand)) {
        leaves.push_back(operand);
        leafFields.push_back(fields[i]);
        shape += '.';
      } else {
        return false;
      }
    }
    shape += ')';
    return true;
  }

  template<typename T> Index getLEBSize(const T& leb) {
    BufferWithRandomAccess buffer;
    buffer << leb;
    return buffer.size();
  }

  // The estimated number of bytes for the binary encoding of a (constant)
  // operand expression.
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
    if (auto* func = curr->dynCast<RefFunc>()) {
      return 1 + getLEBSize(U32LEB(functionIndexes[func->func]));
    }
    if (auto* get = curr->dynCast<GlobalGet>()) {
      return 1 + getLEBSize(U32LEB(globalIndexes[get->name]));
    }
    if (auto* i31 = curr->dynCast<RefI31>()) {
      return 2 + getOperandBytes(i31->value);
    }
    // struct.new or struct.new_default: opcode and a type index, plus the
    // operands.
    Index bytes = 4;
    for (auto* operand : curr->cast<StructNew>()->operands) {
      bytes += getOperandBytes(operand);
    }
    return bytes;
  }

  Index getGlobalEntryBytes(StructNew* init) {
    // The type (ref $T) and mutability, the struct.new, and the end.
    return 3 + 1 + getOperandBytes(init) + 1;
  }

  Index getSLEBSize(Index value) { return getLEBSize(S32LEB(int32_t(value))); }

  static void appendLE(std::vector<char>& data, uint32_t value, Index width) {
    for (Index i = 0; i < width; i++) {
      data.push_back(char(uint8_t(value >> (8 * i))));
    }
  }

  static bool sameRefLeaf(Expression* a, Expression* b) {
    if (a->is<RefNull>() && b->is<RefNull>()) {
      return true;
    }
    if (auto* ga = a->dynCast<GlobalGet>()) {
      if (auto* gb = b->dynCast<GlobalGet>()) {
        return ga->name == gb->name;
      }
      return false;
    }
    if (auto* fa = a->dynCast<RefFunc>()) {
      if (auto* fb = b->dynCast<RefFunc>()) {
        return fa->func == fb->func;
      }
      return false;
    }
    return ExpressionAnalyzer::equal(a, b);
  }

  static bool isPaletteCandidate(const Field& field) {
    return field.type.isRef() &&
           field.type.getHeapType().getBottom() != HeapType::noext;
  }

  static int64_t getPackedOrI32Value(const Field& field, int32_t raw) {
    if (field.isPacked()) {
      if (field.packedType == Field::i8) {
        return int64_t(int8_t(uint8_t(raw)));
      }
      return int64_t(int16_t(uint16_t(raw)));
    }
    return int64_t(raw);
  }

  static void classifyI32Range(int64_t minVal,
                               int64_t maxVal,
                               Index& width,
                               bool& signed_,
                               HeapType& arrType) {
    if (minVal >= 0 && maxVal <= 255) {
      width = 1;
      signed_ = false;
      arrType = HeapTypes::getMutI8Array();
    } else if (minVal >= -128 && maxVal <= 127) {
      width = 1;
      signed_ = true;
      arrType = HeapTypes::getMutI8Array();
    } else if (minVal >= 0 && maxVal <= 65535) {
      width = 2;
      signed_ = false;
      arrType = HeapTypes::getMutI16Array();
    } else if (minVal >= -32768 && maxVal <= 32767) {
      width = 2;
      signed_ = true;
      arrType = HeapTypes::getMutI16Array();
    } else {
      width = 4;
      signed_ = false;
      arrType = HeapType(Array(Field(Type::i32, Mutable)));
    }
  }

  void processGroup(Group& group,
                    std::unordered_map<Name, Deferral>& deferrals) {
    auto& members = group.members;
    // Only globals that are used in code matter (unused ones are left to
    // other passes).
    members.erase(
      std::remove_if(members.begin(),
                     members.end(),
                     [](const Candidate& c) { return c.uses == 0; }),
      members.end());
    if (members.empty()) {
      return;
    }
    if (!includeAllocationOperands) {
      Index uses = 0, allocOperandUses = 0;
      for (auto& member : members) {
        uses += member.uses;
        allocOperandUses += member.allocOperandUses;
      }
      if (allocOperandUses * 2 > uses) {
        return;
      }
    }

    Index numLeaves = group.leafFields.size();
    auto varying = getVaryingColumns(members, numLeaves);

    // A global.get is replaced by a call (opcode + function index; the factory
    // is used often, so assume a 2-byte index) and an i32.const.
    const Index callBytes = 3;

    // Greedily select members, cheapest first, so that the index estimate is
    // meaningful.
    std::vector<Candidate*> sorted;
    for (auto& member : members) {
      sorted.push_back(&member);
    }
    std::sort(sorted.begin(), sorted.end(), [](Candidate* a, Candidate* b) {
      if (a->uses != b->uses) {
        return a->uses < b->uses;
      }
      return a->firstUse < b->firstUse;
    });
    std::vector<Candidate*> chosen;
    int64_t saving = 0;
    for (auto* member : sorted) {
      Index tableBytes = 0;
      for (auto i : varying) {
        tableBytes += getOperandBytes(member->leaves[i]);
      }
      Index index = chosen.size() % MaxChunkSize;
      Index getBytes =
        1 + getLEBSize(U32LEB(globalIndexes[member->global->name]));
      int64_t siteDelta =
        int64_t(callBytes + 1 + getSLEBSize(index)) - int64_t(getBytes);
      int64_t memberSaving = int64_t(member->initBytes) +
                             int64_t(member->bonus) - int64_t(tableBytes) -
                             int64_t(member->uses) * siteDelta;
      if (always || memberSaving > 0) {
        chosen.push_back(member);
        saving += memberSaving;
      }
    }
    if (chosen.empty()) {
      return;
    }
    // Recompute the columns over the chosen members only.
    std::vector<Candidate> chosenMembers;
    for (auto* member : chosen) {
      chosenMembers.push_back(*member);
    }
    varying = getVaryingColumns(chosenMembers, numLeaves);

    // Fixed costs per chunk: the factory function (code, function section
    // entry, type), the cache global and its type, and per table column the
    // global, its type, and the read in the factory.
    Index numChunks = (chosen.size() + MaxChunkSize - 1) / MaxChunkSize;
    Index overhead = numChunks * (70 + 20 * varying.size());
    if (!always && saving <= int64_t(overhead)) {
      return;
    }

    // Assign the indexes in first-use order, which keeps the constants in
    // long initialization functions in ascending order.
    std::sort(chosen.begin(), chosen.end(), [](Candidate* a, Candidate* b) {
      return a->firstUse < b->firstUse;
    });
    for (Index chunk = 0; chunk < numChunks; chunk++) {
      Index begin = chunk * MaxChunkSize;
      Index end = std::min(Index(chosen.size()), begin + MaxChunkSize);
      std::vector<Candidate*> chunkMembers(chosen.begin() + begin,
                                           chosen.begin() + end);
      auto factory = makeFactory(group, chunkMembers, varying);
      for (Index i = 0; i < chunkMembers.size(); i++) {
        deferrals[chunkMembers[i]->global->name] = {factory, i};
      }
    }
  }

  std::vector<Index> getVaryingColumns(std::vector<Candidate>& members,
                                       Index numLeaves) {
    std::vector<Index> varying;
    for (Index i = 0; i < numLeaves; i++) {
      auto* first = members[0].leaves[i];
      for (auto& member : members) {
        if (!ExpressionAnalyzer::equal(first, member.leaves[i])) {
          varying.push_back(i);
          break;
        }
      }
    }
    return varying;
  }

  Name makeFactory(Group& group,
                   std::vector<Candidate*>& members,
                   std::vector<Index>& varying) {
    Builder builder(*module);
    auto type = group.type;
    auto resultType = group.globalType;
    auto& leafFields = group.leafFields;
    std::string typeName;
    if (auto iter = module->typeNames.find(type);
        iter != module->typeNames.end()) {
      typeName = iter->second.name.toString();
    } else {
      typeName = std::to_string(numFactories);
    }
    numFactories++;
    std::string baseName = "defer$" + typeName;

    // The cache.
    // The cache holds nullable versions of the result type (which may be
    // exact).
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

    // The tables, indexed by leaf.
    struct Column {
      Name global;
      Type globalType;
      bool signed_ = false;
      bool addIndex = false;
      Index baseColumn = Index(-1);
      Name paletteGlobal;
      Type paletteGlobalType;
    };
    struct DataColumnInit {
      Name global;
      HeapType arrType;
      Index offset;
      Index count;
    };
    std::unordered_map<Index, Column> columns;
    std::vector<DataColumnInit> dataInits;
    std::vector<Index> prevI32Cols;
    Index N = members.size();
    bool canUseData = module->features.hasBulkMemory();
    for (auto i : varying) {
      auto& field = leafFields[i];
      HeapType tableType(Array(Field(field.type, Immutable)));
      if (field.isPacked()) {
        tableType = HeapType(Array(Field(field.packedType, Immutable)));
      }
      if (canUseData && N >= 10 && field.type == Type::i32 &&
          std::all_of(members.begin(), members.end(), [&](Candidate* m) {
            return m->leaves[i]->is<Const>();
          })) {
        auto evalResiduals = [&](auto getVal,
                                 Index& w,
                                 bool& s,
                                 HeapType& ht,
                                 Index& numDistinct) {
          int64_t mn = std::numeric_limits<int64_t>::max();
          int64_t mx = std::numeric_limits<int64_t>::min();
          std::unordered_set<int32_t> dist;
          for (Index k = 0; k < N; k++) {
            int64_t v = getVal(k);
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            dist.insert(int32_t(v));
          }
          classifyI32Range(mn, mx, w, s, ht);
          numDistinct = dist.size();
        };
        Index width = 4, bestDistinct = N;
        bool signed_ = false;
        HeapType arrType;
        evalResiduals(
          [&](Index k) {
            int32_t raw = members[k]->leaves[i]->cast<Const>()->value.geti32();
            return getPackedOrI32Value(field, raw);
          },
          width,
          signed_,
          arrType,
          bestDistinct);
        bool addIndex = false;
        Index baseColumn = Index(-1);
        if (!field.isPacked()) {
          Index plainDistinct = bestDistinct;
          Index wIdx = 4, dIdx = N;
          bool sIdx = false;
          HeapType htIdx;
          evalResiduals(
            [&](Index k) {
              int32_t raw =
                members[k]->leaves[i]->cast<Const>()->value.geti32();
              return int64_t(int32_t(uint32_t(raw) - uint32_t(k)));
            },
            wIdx,
            sIdx,
            htIdx,
            dIdx);
          if (wIdx < width ||
              (wIdx == width && dIdx * 2 <= plainDistinct &&
               plainDistinct - dIdx >= 8)) {
            width = wIdx;
            signed_ = sIdx;
            arrType = htIdx;
            bestDistinct = dIdx;
            addIndex = true;
          }
          for (Index prev : prevI32Cols) {
            Index wCol = 4, dCol = N;
            bool sCol = false;
            HeapType htCol;
            evalResiduals(
              [&](Index k) {
                int32_t raw =
                  members[k]->leaves[i]->cast<Const>()->value.geti32();
                int32_t baseRaw =
                  members[k]->leaves[prev]->cast<Const>()->value.geti32();
                return int64_t(int32_t(uint32_t(raw) - uint32_t(baseRaw)));
              },
              wCol,
              sCol,
              htCol,
              dCol);
            if (wCol < width ||
                (wCol == width && dCol < bestDistinct &&
                 dCol * 2 <= plainDistinct && plainDistinct - dCol >= 8)) {
              width = wCol;
              signed_ = sCol;
              arrType = htCol;
              bestDistinct = dCol;
              addIndex = false;
              baseColumn = prev;
            }
          }
          prevI32Cols.push_back(i);
        }
        Index offset = segmentData.size();
        for (Index k = 0; k < N; k++) {
          uint32_t raw =
            uint32_t(members[k]->leaves[i]->cast<Const>()->value.geti32());
          if (addIndex) {
            raw -= uint32_t(k);
          } else if (baseColumn != Index(-1)) {
            uint32_t baseRaw = uint32_t(
              members[k]->leaves[baseColumn]->cast<Const>()->value.geti32());
            raw -= baseRaw;
          }
          appendLE(segmentData, raw, width);
        }
        auto tableName = Names::getValidGlobalName(
          *module, baseName + "$table$" + std::to_string(i));
        auto tableGlobalType = Type(arrType, Nullable);
        module->addGlobal(builder.makeGlobal(tableName,
                                             tableGlobalType,
                                             builder.makeRefNull(arrType),
                                             Builder::Mutable));
        dataInits.push_back({tableName, arrType, offset, N});
        columns[i] = {tableName,
                      tableGlobalType,
                      signed_,
                      addIndex,
                      baseColumn,
                      Name(),
                      Type::none};
        continue;
      }
      if (canUseData && isPaletteCandidate(field)) {
        std::vector<Expression*> distinct;
        std::vector<uint8_t> indices;
        indices.reserve(N);
        bool ok = true;
        for (auto* member : members) {
          auto* leaf = member->leaves[i];
          Index idx = 0;
          while (idx < distinct.size() && !sameRefLeaf(distinct[idx], leaf)) {
            idx++;
          }
          if (idx == distinct.size()) {
            if (distinct.size() == 256) {
              ok = false;
              break;
            }
            distinct.push_back(leaf);
          }
          indices.push_back(uint8_t(idx));
        }
        Index D = distinct.size();
        if (ok && 9 + 3 * D + N + 25 < 9 + 3 * N) {
          std::vector<Expression*> palValues;
          palValues.reserve(D);
          for (auto* expr : distinct) {
            palValues.push_back(ExpressionManipulator::copy(expr, *module));
          }
          auto palName = Names::getValidGlobalName(
            *module, baseName + "$palette$" + std::to_string(i));
          auto palGlobalType = Type(tableType, NonNullable);
          module->addGlobal(
            builder.makeGlobal(palName,
                               palGlobalType,
                               builder.makeArrayNewFixed(tableType, palValues),
                               Builder::Immutable));
          Index offset = segmentData.size();
          for (uint8_t idx : indices) {
            segmentData.push_back(char(idx));
          }
          auto idxArrType = HeapTypes::getMutI8Array();
          auto tableName = Names::getValidGlobalName(
            *module, baseName + "$table$" + std::to_string(i));
          auto tableGlobalType = Type(idxArrType, Nullable);
          module->addGlobal(builder.makeGlobal(tableName,
                                               tableGlobalType,
                                               builder.makeRefNull(idxArrType),
                                               Builder::Mutable));
          dataInits.push_back({tableName, idxArrType, offset, N});
          columns[i] = {tableName,
                        tableGlobalType,
                        false,
                        false,
                        Index(-1),
                        palName,
                        palGlobalType};
          continue;
        }
      }
      std::vector<Expression*> values;
      for (auto* member : members) {
        values.push_back(
          ExpressionManipulator::copy(member->leaves[i], *module));
      }
      auto tableName = Names::getValidGlobalName(
        *module, baseName + "$table$" + std::to_string(i));
      auto tableGlobalType = Type(tableType, NonNullable);
      module->addGlobal(
        builder.makeGlobal(tableName,
                           tableGlobalType,
                           builder.makeArrayNewFixed(tableType, values),
                           Builder::Immutable));
      columns[i] = {tableName,
                    tableGlobalType,
                    false,
                    false,
                    Index(-1),
                    Name(),
                    Type::none};
    }

    // The factory.
    Index indexLocal = 0;
    Index objectLocal = 1;
    Name label("cached");
    auto* cacheGet =
      builder.makeArrayGet(builder.makeGlobalGet(cacheName, cacheGlobalType),
                           builder.makeLocalGet(indexLocal, Type::i32),
                           MemoryOrder::Unordered,
                           cacheElementType);
    auto* brOn = builder.makeBrOn(BrOnNonNull, label, cacheGet);

    // Rebuild the (nested) allocation from the first member, reading the
    // varying leaves from the tables.
    std::function<Expression*(Index)> readColumn = [&](Index i) -> Expression* {
      auto& col = columns.at(i);
      if (col.paletteGlobal.is()) {
        auto* idx = builder.makeArrayGet(
          builder.makeGlobalGet(col.global, col.globalType),
          builder.makeLocalGet(indexLocal, Type::i32),
          MemoryOrder::Unordered,
          Type::i32,
          false);
        return builder.makeArrayGet(
          builder.makeGlobalGet(col.paletteGlobal, col.paletteGlobalType),
          idx,
          MemoryOrder::Unordered,
          leafFields[i].type,
          false);
      }
      Expression* val = builder.makeArrayGet(
        builder.makeGlobalGet(col.global, col.globalType),
        builder.makeLocalGet(indexLocal, Type::i32),
        MemoryOrder::Unordered,
        leafFields[i].type,
        col.signed_);
      if (col.addIndex) {
        val = builder.makeBinary(
          AddInt32, val, builder.makeLocalGet(indexLocal, Type::i32));
      } else if (col.baseColumn != Index(-1)) {
        val = builder.makeBinary(AddInt32, val, readColumn(col.baseColumn));
      }
      return val;
    };
    Index leaf = 0;
    std::function<Expression*(StructNew*)> rebuild =
      [&](StructNew* templ) -> Expression* {
      std::vector<Expression*> operands;
      for (auto* operand : templ->operands) {
        if (isNestedNew(operand)) {
          operands.push_back(rebuild(operand->cast<StructNew>()));
          continue;
        }
        Index i = leaf++;
        if (columns.count(i)) {
          operands.push_back(readColumn(i));
        } else {
          operands.push_back(ExpressionManipulator::copy(operand, *module));
        }
      }
      return builder.makeStructNew(templ->type.getHeapType(), operands);
    };
    auto* structNew = rebuild(members[0]->init);
    assert(leaf == leafFields.size());
    auto* tee = builder.makeLocalTee(objectLocal, structNew, resultType);
    auto* cacheSet =
      builder.makeArraySet(builder.makeGlobalGet(cacheName, cacheGlobalType),
                           builder.makeLocalGet(indexLocal, Type::i32),
                           tee,
                           MemoryOrder::Unordered);
    std::vector<Expression*> blockList = {brOn};
    if (!dataInits.empty()) {
      std::vector<Expression*> setStmts;
      for (auto& dc : dataInits) {
        setStmts.push_back(builder.makeGlobalSet(
          dc.global,
          builder.makeArrayNewData(dc.arrType,
                                   segmentName,
                                   builder.makeConst(int32_t(dc.offset)),
                                   builder.makeConst(int32_t(dc.count)))));
      }
      Expression* initBody =
        setStmts.size() == 1 ? setStmts[0] : builder.makeBlock(setStmts);
      blockList.push_back(builder.makeIf(
        builder.makeRefIsNull(builder.makeGlobalGet(
          dataInits[0].global, Type(dataInits[0].arrType, Nullable))),
        initBody));
    }
    blockList.push_back(cacheSet);
    blockList.push_back(builder.makeLocalGet(objectLocal, resultType));
    auto* body = builder.makeBlock(label, blockList, resultType);
    auto factoryName = Names::getValidFunctionName(*module, baseName);
    module->addFunction(builder.makeFunction(
      factoryName, Signature(Type::i32, resultType), {resultType}, body));
    return factoryName;
  }
};

} // anonymous namespace

Pass* createDeferGlobalAllocationsPass() {
  return new DeferGlobalAllocations();
}

} // namespace wasm
