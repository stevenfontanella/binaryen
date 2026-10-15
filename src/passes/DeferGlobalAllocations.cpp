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

#include "ir/find_all.h"
#include "ir/iteration.h"
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
    // Candidates used inside a once-initialized region.
    std::unordered_map<Name, Index> onceUses;
    // Candidates used directly as operands of a struct.new outside once-guards.
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
    Index onceUses = 0;
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
  bool preserveThunkTypes = false;
  bool onceOnly = false;
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
    preserveThunkTypes =
      !getArgumentOrDefault("defer-global-allocations-preserve-thunk-types", "")
         .empty();
    onceOnly =
      !getArgumentOrDefault("defer-global-allocations-once-only", "").empty();
    always =
      !getArgumentOrDefault("defer-global-allocations-always", "").empty();
    importBonus = std::stoul(
      getArgumentOrDefault("defer-global-allocations-import-bonus", "0"));

    static const Module* lastPreservedModule = nullptr;
    if (preserveThunkTypes) {
      if (lastPreservedModule == module) {
        preserveThunkTypes = false;
        onceOnly = true;
      } else {
        lastPreservedModule = module;
      }
    }
    if (!preserveThunkTypes) {
      optimizeOnceGlobals(/*onlyArrays=*/true);
      segmentData.clear();
    }
    runGroupDeferral();
    optimizeOnceGlobals(/*onlyArrays=*/false);
  }

  void runGroupDeferral() {
    computeIndexes();

    std::unordered_set<Expression*> onceGets;
    computeOnceGetSet(onceGets);

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
          const std::unordered_set<Expression*>& onceGets;
          Scanner(Module& module,
                  FunctionInfo& info,
                  const std::unordered_map<Name, std::pair<Group*, Index>>&
                    candidates,
                  const std::unordered_set<Expression*>& onceGets)
            : module(module), info(info), candidates(candidates),
              onceGets(onceGets) {}
          void visitGlobalGet(GlobalGet* curr) {
            if (candidates.count(curr->name)) {
              info.gets.push_back(getCurrentPointer());
              if (onceGets.count(curr)) {
                info.onceUses[curr->name]++;
              }
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
        Scanner(*module, info, candidateMap, onceGets).walk(func->body);
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
      for (auto& [name, count] : info.onceUses) {
        auto& [group, i] = candidateMap[name];
        group->members[i].onceUses += count;
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
    // other passes). In once-only mode, only single-use once-guarded globals
    // are considered.
    members.erase(
      std::remove_if(members.begin(),
                     members.end(),
                     [&](const Candidate& c) {
                       if (c.uses == 0) {
                         return true;
                       }
                       if (onceOnly) {
                         return !(c.uses == 1 && c.onceUses == 1);
                       }
                       return false;
                     }),
      members.end());
    if (members.empty()) {
      return;
    }
    if (!includeAllocationOperands && !onceOnly) {
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

    bool allOnceSingleUse =
      !preserveThunkTypes &&
      std::all_of(chosen.begin(), chosen.end(), [](Candidate* c) {
        return c->uses == 1 && c->onceUses == 1;
      });
    // Fixed costs per chunk: the factory function (code, function section
    // entry, type), the cache global and its type (unless cacheless), and per
    // table column the global, its type, and the read in the factory.
    Index numChunks = (chosen.size() + MaxChunkSize - 1) / MaxChunkSize;
    Index baseFixed = allOnceSingleUse ? 25 : 70;
    Index overhead = numChunks * (baseFixed + 20 * varying.size());
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
      bool cacheless =
        !preserveThunkTypes &&
        std::all_of(chunkMembers.begin(), chunkMembers.end(), [](Candidate* c) {
          return c->uses == 1 && c->onceUses == 1;
        });
      auto factory = makeFactory(group, chunkMembers, varying, cacheless);
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
                   std::vector<Index>& varying,
                   bool cacheless) {
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

    // The cache (omitted when every member is single-use inside a once-guard).
    auto cacheElementType = resultType.with(Nullable);
    Name cacheName;
    Type cacheGlobalType = Type::none;
    if (!cacheless) {
      HeapType cacheType(Array(Field(cacheElementType, Mutable)));
      cacheName = Names::getValidGlobalName(*module, baseName + "$cache");
      cacheGlobalType = Type(cacheType, NonNullable);
      module->addGlobal(builder.makeGlobal(
        cacheName,
        cacheGlobalType,
        builder.makeArrayNew(cacheType,
                             builder.makeConst(int32_t(members.size()))),
        Builder::Immutable));
    }

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

    Expression* dataInitCheck = nullptr;
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
      dataInitCheck = builder.makeIf(
        builder.makeRefIsNull(builder.makeGlobalGet(
          dataInits[0].global, Type(dataInits[0].arrType, Nullable))),
        initBody);
    }

    auto factoryName = Names::getValidFunctionName(*module, baseName);
    if (cacheless) {
      Expression* body =
        dataInitCheck
          ? builder.makeBlock({dataInitCheck, structNew}, resultType)
          : structNew;
      module->addFunction(builder.makeFunction(
        factoryName, Signature(Type::i32, resultType), {}, body));
      return factoryName;
    }

    Index objectLocal = 1;
    Name label("cached");
    auto* cacheGet =
      builder.makeArrayGet(builder.makeGlobalGet(cacheName, cacheGlobalType),
                           builder.makeLocalGet(indexLocal, Type::i32),
                           MemoryOrder::Unordered,
                           cacheElementType);
    auto* brOn = builder.makeBrOn(BrOnNonNull, label, cacheGet);
    auto* tee = builder.makeLocalTee(objectLocal, structNew, resultType);
    auto* cacheSet =
      builder.makeArraySet(builder.makeGlobalGet(cacheName, cacheGlobalType),
                           builder.makeLocalGet(indexLocal, Type::i32),
                           tee,
                           MemoryOrder::Unordered);
    std::vector<Expression*> blockList = {brOn};
    if (dataInitCheck) {
      blockList.push_back(dataInitCheck);
    }
    blockList.push_back(cacheSet);
    blockList.push_back(builder.makeLocalGet(objectLocal, resultType));
    auto* body = builder.makeBlock(label, blockList, resultType);
    module->addFunction(builder.makeFunction(
      factoryName, Signature(Type::i32, resultType), {resultType}, body));
    return factoryName;
  }

  static bool hasImmutableSigField(Type t) {
    if (!t.isRef() || !t.getHeapType().isStruct()) {
      return false;
    }
    for (auto& field : t.getHeapType().getStruct().fields) {
      if (field.mutable_ == Immutable && field.type.isRef() &&
          field.type.getHeapType().isSignature()) {
        return true;
      }
    }
    return false;
  }

  bool isInlinableAllocExpr(Expression* curr, bool isRoot) const {
    if (auto* sn = curr->dynCast<StructNew>()) {
      if (sn->desc || sn->type == Type::unreachable) {
        return false;
      }
      if (preserveThunkTypes && hasImmutableSigField(sn->type)) {
        return false;
      }
      for (auto* op : sn->operands) {
        if (!isInlinableAllocExpr(op, false)) {
          return false;
        }
      }
      return true;
    }
    if (auto* anf = curr->dynCast<ArrayNewFixed>()) {
      if (anf->type == Type::unreachable) {
        return false;
      }
      if ((!isRoot && anf->values.size() > 16) || anf->values.size() > 256) {
        return false;
      }
      for (auto* val : anf->values) {
        if (!isInlinableAllocExpr(val, false)) {
          return false;
        }
      }
      return true;
    }
    if (auto* an = curr->dynCast<ArrayNew>()) {
      if (an->type == Type::unreachable) {
        return false;
      }
      if (an->init && !isInlinableAllocExpr(an->init, false)) {
        return false;
      }
      return isInlinableAllocExpr(an->size, false);
    }
    if (auto* and_ = curr->dynCast<ArrayNewData>()) {
      if (and_->type == Type::unreachable) {
        return false;
      }
      return isInlinableAllocExpr(and_->offset, false) &&
             isInlinableAllocExpr(and_->size, false);
    }
    if (isRoot) {
      return false;
    }
    if (curr->is<Const>() || curr->is<RefNull>() || curr->is<RefFunc>() ||
        curr->is<StringConst>()) {
      return true;
    }
    if (auto* i31 = curr->dynCast<RefI31>()) {
      return isInlinableAllocExpr(i31->value, false);
    }
    if (auto* gg = curr->dynCast<GlobalGet>()) {
      auto* g = module->getGlobalOrNull(gg->name);
      return g && !g->mutable_;
    }
    return false;
  }

  static bool isTrivialLocalInit(Expression* stmt) {
    auto* ls = stmt->dynCast<LocalSet>();
    if (!ls) {
      return false;
    }
    Expression* v = ls->value;
    return v->is<Const>() || v->is<RefNull>() || v->is<LocalGet>() ||
           v->is<RefFunc>();
  }

  static Expression* getFirstStmt(Expression* expr) {
    while (auto* b = expr->dynCast<Block>()) {
      Expression* next = nullptr;
      for (auto* item : b->list) {
        if (!isTrivialLocalInit(item)) {
          next = item;
          break;
        }
      }
      if (!next) {
        return expr;
      }
      expr = next;
    }
    return expr;
  }

  static Block* getEditableBlock(Expression* expr) {
    auto* b = expr->dynCast<Block>();
    while (b) {
      Index firstNonTrivial = 0;
      while (firstNonTrivial < b->list.size() &&
             isTrivialLocalInit(b->list[firstNonTrivial])) {
        firstNonTrivial++;
      }
      if (firstNonTrivial + 1 == b->list.size() &&
          b->list[firstNonTrivial]->is<Block>()) {
        b = b->list[firstNonTrivial]->cast<Block>();
      } else {
        break;
      }
    }
    return b;
  }

  void computeOnceGetSet(std::unordered_set<Expression*>& onceGets) {
    std::unordered_set<Name> nonCodeGlobalUses;
    std::unordered_set<Name> nonCallFuncUses;
    std::unordered_map<Name, Index> initGlobalUses;

    struct NonCodeScanner : public PostWalker<NonCodeScanner> {
      std::unordered_set<Name>& globalUses;
      std::unordered_set<Name>& funcUses;
      std::unordered_map<Name, Index>* initMap = nullptr;
      NonCodeScanner(std::unordered_set<Name>& globalUses,
                     std::unordered_set<Name>& funcUses)
        : globalUses(globalUses), funcUses(funcUses) {}
      void visitGlobalGet(GlobalGet* curr) {
        if (initMap) {
          (*initMap)[curr->name]++;
        } else {
          globalUses.insert(curr->name);
        }
      }
      void visitRefFunc(RefFunc* curr) { funcUses.insert(curr->func); }
    } nonCodeScanner(nonCodeGlobalUses, nonCallFuncUses);

    for (auto& g : module->globals) {
      if (!g->imported()) {
        nonCodeScanner.initMap = &initGlobalUses;
        nonCodeScanner.walk(g->init);
      }
    }
    nonCodeScanner.initMap = nullptr;
    for (auto& seg : module->elementSegments) {
      if (seg->offset) {
        nonCodeScanner.walk(seg->offset);
      }
      for (auto* item : seg->data) {
        nonCodeScanner.walk(item);
      }
    }
    for (auto& seg : module->dataSegments) {
      if (seg->offset) {
        nonCodeScanner.walk(seg->offset);
      }
    }
    for (auto& ex : module->exports) {
      if (ex->kind == ExternalKind::Global) {
        nonCodeGlobalUses.insert(*ex->getInternalName());
      } else if (ex->kind == ExternalKind::Function) {
        nonCallFuncUses.insert(*ex->getInternalName());
      }
    }
    if (module->start.is()) {
      nonCallFuncUses.insert(module->start);
    }

    struct FuncScanInfo {
      std::vector<std::pair<Name, Expression**>> gets;
      std::vector<std::pair<Name, Expression**>> sets;
      std::vector<Call*> calls;
      std::vector<If*> ifs;
      std::vector<Name> refFuncs;
    };

    ModuleUtils::ParallelFunctionAnalysis<FuncScanInfo> funcAnalysis(
      *module, [&](Function* func, FuncScanInfo& info) {
        if (func->imported()) {
          return;
        }
        struct Walker : public PostWalker<Walker> {
          FuncScanInfo& info;
          Walker(FuncScanInfo& info) : info(info) {}
          void visitGlobalGet(GlobalGet* curr) {
            info.gets.push_back({curr->name, getCurrentPointer()});
          }
          void visitGlobalSet(GlobalSet* curr) {
            info.sets.push_back({curr->name, getCurrentPointer()});
          }
          void visitCall(Call* curr) { info.calls.push_back(curr); }
          void visitIf(If* curr) { info.ifs.push_back(curr); }
          void visitRefFunc(RefFunc* curr) {
            info.refFuncs.push_back(curr->func);
          }
        };
        Walker(info).walk(func->body);
      });

    std::unordered_map<Name, std::vector<std::pair<Function*, Expression**>>>
      funcGets;
    std::unordered_map<Name, std::vector<std::pair<Function*, Expression**>>>
      funcSets;
    std::unordered_map<Name, std::vector<std::pair<Function*, Call*>>>
      funcCallers;
    std::unordered_map<Name, std::vector<std::pair<Function*, If*>>>
      flagGuardIfs;

    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      auto& info = funcAnalysis.map[func.get()];
      for (auto& [gname, ptr] : info.gets) {
        funcGets[gname].push_back({func.get(), ptr});
      }
      for (auto& [gname, ptr] : info.sets) {
        funcSets[gname].push_back({func.get(), ptr});
      }
      for (auto* call : info.calls) {
        funcCallers[call->target].push_back({func.get(), call});
      }
      for (auto rf : info.refFuncs) {
        nonCallFuncUses.insert(rf);
      }
    }

    std::unordered_set<Name> onceFlags;
    for (auto& g : module->globals) {
      if (g->imported() || !g->mutable_ || g->type != Type::i32 ||
          nonCodeGlobalUses.count(g->name) || initGlobalUses[g->name] > 0) {
        continue;
      }
      auto* c = g->init->dynCast<Const>();
      if (!c || c->value.geti32() != 0) {
        continue;
      }
      auto it = funcSets.find(g->name);
      if (it == funcSets.end() || it->second.empty()) {
        continue;
      }
      bool allNonZeroConst = true;
      for (auto& [_, setPtr] : it->second) {
        auto* gs = (*setPtr)->cast<GlobalSet>();
        auto* sc = gs->value->dynCast<Const>();
        if (!sc || sc->value.geti32() == 0) {
          allNonZeroConst = false;
          break;
        }
      }
      if (allNonZeroConst) {
        onceFlags.insert(g->name);
      }
    }

    auto isFlagEqzCond = [&](Expression* cond, Name& flagOut) -> bool {
      auto* u = cond->dynCast<Unary>();
      if (!u || u->op != EqZInt32) {
        return false;
      }
      auto* gg = u->value->dynCast<GlobalGet>();
      if (!gg || !onceFlags.count(gg->name) ||
          funcSets[gg->name].size() != 1) {
        return false;
      }
      flagOut = gg->name;
      return true;
    };

    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      for (auto* iff : funcAnalysis.map[func.get()].ifs) {
        Name flag;
        if (!iff->ifFalse && isFlagEqzCond(iff->condition, flag)) {
          flagGuardIfs[flag].push_back({func.get(), iff});
        }
      }
    }

    std::unordered_set<Name> onceHelperFuncs;
    std::unordered_map<Name, Function*> flagHelperFunc;
    for (auto& func : module->functions) {
      if (func->imported() || nonCallFuncUses.count(func->name)) {
        continue;
      }
      auto& callers = funcCallers[func->name];
      if (callers.empty()) {
        continue;
      }
      auto* firstSet = getFirstStmt(func->body)->dynCast<GlobalSet>();
      if (!firstSet || !onceFlags.count(firstSet->name) ||
          funcSets[firstSet->name].size() != 1) {
        continue;
      }
      Name flag = firstSet->name;
      auto& guards = flagGuardIfs[flag];
      if (guards.size() != callers.size() ||
          guards.size() != funcGets[flag].size()) {
        continue;
      }
      bool allMatch = true;
      for (auto& [_, iff] : guards) {
        auto* call = getFirstStmt(iff->ifTrue)->dynCast<Call>();
        if (!call || call != iff->ifTrue || call->target != func->name) {
          allMatch = false;
          break;
        }
      }
      if (allMatch) {
        onceHelperFuncs.insert(func->name);
        flagHelperFunc[flag] = func.get();
      }
    }

    auto isOnceIf = [&](If* iff) -> bool {
      if (iff->ifFalse) {
        return false;
      }
      Name flag;
      if (isFlagEqzCond(iff->condition, flag)) {
        Expression* first = getFirstStmt(iff->ifTrue);
        if (auto* gs = first->dynCast<GlobalSet>()) {
          return gs->name == flag;
        }
        if (auto* call = first->dynCast<Call>()) {
          auto it = flagHelperFunc.find(flag);
          return it != flagHelperFunc.end() &&
                 it->second->name == call->target;
        }
        return false;
      }
      if (auto* rin = iff->condition->dynCast<RefIsNull>()) {
        if (auto* gg = rin->value->dynCast<GlobalGet>()) {
          Name lazy = gg->name;
          auto* lg = module->getGlobalOrNull(lazy);
          if (lg && !lg->imported() && lg->mutable_ && lg->type.isRef() &&
              lg->type.isNullable() && lg->init->is<RefNull>() &&
              !nonCodeGlobalUses.count(lazy) && initGlobalUses[lazy] == 0 &&
              funcSets[lazy].size() == 1) {
            Expression* targetSet = *funcSets[lazy][0].second;
            for (auto* gs : FindAll<GlobalSet>(iff->ifTrue).list) {
              if (gs == targetSet && !gs->value->is<RefNull>()) {
                return true;
              }
            }
          }
        }
      }
      return false;
    };

    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      std::function<void(Expression*, bool, bool)> walkOnce =
        [&](Expression* curr, bool inOnce, bool inLoopOrTry) {
          if (!curr) {
            return;
          }
          if (curr->is<Loop>() || curr->is<Try>() || curr->is<TryTable>()) {
            inLoopOrTry = true;
          }
          if (auto* iff = curr->dynCast<If>()) {
            walkOnce(iff->condition, inOnce, inLoopOrTry);
            walkOnce(iff->ifTrue, inOnce || isOnceIf(iff), inLoopOrTry);
            if (iff->ifFalse) {
              walkOnce(iff->ifFalse, inOnce, inLoopOrTry);
            }
            return;
          }
          if (auto* gg = curr->dynCast<GlobalGet>()) {
            if (inOnce && !inLoopOrTry) {
              onceGets.insert(gg);
            }
            return;
          }
          for (auto* ch : ChildIterator(curr)) {
            walkOnce(ch, inOnce, inLoopOrTry);
          }
        };
      walkOnce(func->body, onceHelperFuncs.count(func->name) > 0, false);
    }
  }

  void optimizeOnceGlobals(bool onlyArrays) {
    std::unordered_set<Name> nonCodeGlobalUses;
    std::unordered_set<Name> nonCallFuncUses;
    std::unordered_map<Name, Index> initGlobalUses;

    struct NonCodeScanner : public PostWalker<NonCodeScanner> {
      std::unordered_set<Name>& globalUses;
      std::unordered_set<Name>& funcUses;
      std::unordered_map<Name, Index>* initMap = nullptr;
      NonCodeScanner(std::unordered_set<Name>& globalUses,
                     std::unordered_set<Name>& funcUses)
        : globalUses(globalUses), funcUses(funcUses) {}
      void visitGlobalGet(GlobalGet* curr) {
        if (initMap) {
          (*initMap)[curr->name]++;
        } else {
          globalUses.insert(curr->name);
        }
      }
      void visitRefFunc(RefFunc* curr) { funcUses.insert(curr->func); }
    } nonCodeScanner(nonCodeGlobalUses, nonCallFuncUses);

    for (auto& g : module->globals) {
      if (!g->imported()) {
        nonCodeScanner.initMap = &initGlobalUses;
        nonCodeScanner.walk(g->init);
      }
    }
    nonCodeScanner.initMap = nullptr;
    for (auto& seg : module->elementSegments) {
      if (seg->offset) {
        nonCodeScanner.walk(seg->offset);
      }
      for (auto* item : seg->data) {
        nonCodeScanner.walk(item);
      }
    }
    for (auto& seg : module->dataSegments) {
      if (seg->offset) {
        nonCodeScanner.walk(seg->offset);
      }
    }
    for (auto& ex : module->exports) {
      if (ex->kind == ExternalKind::Global) {
        nonCodeGlobalUses.insert(*ex->getInternalName());
      } else if (ex->kind == ExternalKind::Function) {
        nonCallFuncUses.insert(*ex->getInternalName());
      }
    }
    if (module->start.is()) {
      nonCallFuncUses.insert(module->start);
    }

    struct FuncScanInfo {
      std::vector<std::pair<Name, Expression**>> gets;
      std::vector<std::pair<Name, Expression**>> sets;
      std::vector<Call*> calls;
      std::vector<If*> ifs;
      std::vector<Name> refFuncs;
    };

    ModuleUtils::ParallelFunctionAnalysis<FuncScanInfo> funcAnalysis(
      *module, [&](Function* func, FuncScanInfo& info) {
        if (func->imported()) {
          return;
        }
        struct Walker : public PostWalker<Walker> {
          FuncScanInfo& info;
          Walker(FuncScanInfo& info) : info(info) {}
          void visitGlobalGet(GlobalGet* curr) {
            info.gets.push_back({curr->name, getCurrentPointer()});
          }
          void visitGlobalSet(GlobalSet* curr) {
            info.sets.push_back({curr->name, getCurrentPointer()});
          }
          void visitCall(Call* curr) { info.calls.push_back(curr); }
          void visitIf(If* curr) { info.ifs.push_back(curr); }
          void visitRefFunc(RefFunc* curr) {
            info.refFuncs.push_back(curr->func);
          }
        };
        Walker(info).walk(func->body);
      });

    std::unordered_map<Name, std::vector<std::pair<Function*, Expression**>>>
      funcGets;
    std::unordered_map<Name, std::vector<std::pair<Function*, Expression**>>>
      funcSets;
    std::unordered_map<Name, std::vector<std::pair<Function*, Call*>>>
      funcCallers;
    std::unordered_map<Name, std::vector<Name>> funcCallees;
    std::unordered_map<Name, std::vector<std::pair<Function*, If*>>>
      flagGuardIfs;

    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      auto& info = funcAnalysis.map[func.get()];
      for (auto& [gname, ptr] : info.gets) {
        funcGets[gname].push_back({func.get(), ptr});
      }
      for (auto& [gname, ptr] : info.sets) {
        funcSets[gname].push_back({func.get(), ptr});
      }
      auto& callees = funcCallees[func->name];
      for (auto* call : info.calls) {
        funcCallers[call->target].push_back({func.get(), call});
        callees.push_back(call->target);
      }
      for (auto rf : info.refFuncs) {
        nonCallFuncUses.insert(rf);
      }
    }

    // Identify once-flag globals: mutable i32 globals initialized to 0 whose
    // only sets are non-zero constants.
    std::unordered_set<Name> onceFlags;
    for (auto& g : module->globals) {
      if (g->imported() || !g->mutable_ || g->type != Type::i32 ||
          nonCodeGlobalUses.count(g->name) || initGlobalUses[g->name] > 0) {
        continue;
      }
      auto* c = g->init->dynCast<Const>();
      if (!c || c->value.geti32() != 0) {
        continue;
      }
      auto it = funcSets.find(g->name);
      if (it == funcSets.end() || it->second.empty()) {
        continue;
      }
      bool allNonZeroConst = true;
      for (auto& [_, setPtr] : it->second) {
        auto* gs = (*setPtr)->cast<GlobalSet>();
        auto* sc = gs->value->dynCast<Const>();
        if (!sc || sc->value.geti32() == 0) {
          allNonZeroConst = false;
          break;
        }
      }
      if (allNonZeroConst) {
        onceFlags.insert(g->name);
      }
    }

    auto isFlagEqzCond = [&](Expression* cond, Name& flagOut) -> bool {
      auto* u = cond->dynCast<Unary>();
      if (!u || u->op != EqZInt32) {
        return false;
      }
      auto* gg = u->value->dynCast<GlobalGet>();
      if (!gg || !onceFlags.count(gg->name) ||
          funcSets[gg->name].size() != 1) {
        return false;
      }
      flagOut = gg->name;
      return true;
    };

    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      for (auto* iff : funcAnalysis.map[func.get()].ifs) {
        Name flag;
        if (!iff->ifFalse && isFlagEqzCond(iff->condition, flag)) {
          flagGuardIfs[flag].push_back({func.get(), iff});
        }
      }
    }

    // Identify direct once-helper functions: functions starting with
    // (global.set $flag (i32.const 1)) whose every call is in
    // (if (i32.eqz (global.get $flag)) (then (call $fn))).
    std::unordered_set<Name> onceHelperFuncs;
    std::unordered_map<Name, Function*> flagHelperFunc;
    for (auto& func : module->functions) {
      if (func->imported() || nonCallFuncUses.count(func->name)) {
        continue;
      }
      auto& callers = funcCallers[func->name];
      if (callers.empty()) {
        continue;
      }
      auto* firstSet = getFirstStmt(func->body)->dynCast<GlobalSet>();
      if (!firstSet || !onceFlags.count(firstSet->name) ||
          funcSets[firstSet->name].size() != 1) {
        continue;
      }
      Name flag = firstSet->name;
      auto& guards = flagGuardIfs[flag];
      if (guards.size() != callers.size() ||
          guards.size() != funcGets[flag].size()) {
        continue;
      }
      bool allMatch = true;
      for (auto& [_, iff] : guards) {
        auto* call = getFirstStmt(iff->ifTrue)->dynCast<Call>();
        if (!call || call != iff->ifTrue || call->target != func->name) {
          allMatch = false;
          break;
        }
      }
      if (allMatch) {
        onceHelperFuncs.insert(func->name);
        flagHelperFunc[flag] = func.get();
      }
    }

    auto isOnceIf = [&](If* iff) -> bool {
      if (iff->ifFalse) {
        return false;
      }
      Name flag;
      if (isFlagEqzCond(iff->condition, flag)) {
        Expression* first = getFirstStmt(iff->ifTrue);
        if (auto* gs = first->dynCast<GlobalSet>()) {
          return gs->name == flag;
        }
        if (auto* call = first->dynCast<Call>()) {
          auto it = flagHelperFunc.find(flag);
          return it != flagHelperFunc.end() &&
                 it->second->name == call->target;
        }
        return false;
      }
      if (auto* rin = iff->condition->dynCast<RefIsNull>()) {
        if (auto* gg = rin->value->dynCast<GlobalGet>()) {
          Name lazy = gg->name;
          auto* lg = module->getGlobalOrNull(lazy);
          if (lg && !lg->imported() && lg->mutable_ && lg->type.isRef() &&
              lg->type.isNullable() && lg->init->is<RefNull>() &&
              !nonCodeGlobalUses.count(lazy) && initGlobalUses[lazy] == 0 &&
              funcSets[lazy].size() == 1) {
            Expression* targetSet = *funcSets[lazy][0].second;
            for (auto* gs : FindAll<GlobalSet>(iff->ifTrue).list) {
              if (gs == targetSet && !gs->value->is<RefNull>()) {
                return true;
              }
            }
          }
        }
      }
      return false;
    };

    std::unordered_set<Function*> modifiedFuncs;
    Builder builder(*module);

    // -------------------------------------------------------------------------
    // Pattern 1: Inline single-use immutable allocation globals in once-regions
    // -------------------------------------------------------------------------
    if (!preserveThunkTypes) {
      std::unordered_map<Name, Index> totalCodeGets;
      for (auto& [gname, vec] : funcGets) {
        totalCodeGets[gname] = vec.size();
      }
      std::unordered_map<Name, std::vector<std::pair<Function*, Expression**>>>
        onceGetLoc;

      for (auto& func : module->functions) {
        if (func->imported()) {
          continue;
        }
        Function* fn = func.get();
        std::function<void(Expression**, bool, bool)> walkOnce =
          [&](Expression** exprp, bool inOnce, bool inLoopOrTry) {
            if (!exprp || !*exprp) {
              return;
            }
            Expression* curr = *exprp;
            if (curr->is<Loop>() || curr->is<Try>() || curr->is<TryTable>()) {
              inLoopOrTry = true;
            }
            if (auto* iff = curr->dynCast<If>()) {
              walkOnce(&iff->condition, inOnce, inLoopOrTry);
              walkOnce(&iff->ifTrue, inOnce || isOnceIf(iff), inLoopOrTry);
              if (iff->ifFalse) {
                walkOnce(&iff->ifFalse, inOnce, inLoopOrTry);
              }
              return;
            }
            if (auto* gg = curr->dynCast<GlobalGet>()) {
              if (inOnce && !inLoopOrTry) {
                onceGetLoc[gg->name].push_back({fn, exprp});
              }
              return;
            }
            for (auto*& ch : ChildIterator(curr)) {
              walkOnce(&ch, inOnce, inLoopOrTry);
            }
          };
        walkOnce(&fn->body, onceHelperFuncs.count(fn->name) > 0, false);
      }

      bool changed = true;
      while (changed) {
        changed = false;
        std::vector<Name> globalsToRemove;
        for (auto& g : module->globals) {
          if (g->imported() || g->mutable_ ||
              nonCodeGlobalUses.count(g->name)) {
            continue;
          }
          if (onlyArrays &&
              !(g->init->is<ArrayNewFixed>() || g->init->is<ArrayNew>() ||
                g->init->is<ArrayNewData>() || g->init->is<ArrayNewElem>())) {
            continue;
          }
          Name gname = g->name;
          if (initGlobalUses[gname] != 0 || totalCodeGets[gname] != 1) {
            continue;
          }
          auto it = onceGetLoc.find(gname);
          if (it == onceGetLoc.end() || it->second.size() != 1) {
            continue;
          }
          if (!isInlinableAllocExpr(g->init, true)) {
            continue;
          }
          auto [targetFn, getPtr] = it->second[0];
          Expression* copyInit = ExpressionManipulator::copy(g->init, *module);
          *getPtr = copyInit;
          modifiedFuncs.insert(targetFn);

          std::function<void(Expression**)> recordInlined =
            [&](Expression** p) {
              Expression* e = *p;
              if (auto* gg = e->dynCast<GlobalGet>()) {
                initGlobalUses[gg->name]--;
                totalCodeGets[gg->name]++;
                onceGetLoc[gg->name].push_back({targetFn, p});
                return;
              }
              for (auto*& ch : ChildIterator(e)) {
                recordInlined(&ch);
              }
            };
          recordInlined(getPtr);
          globalsToRemove.push_back(gname);
          changed = true;
        }
        for (auto name : globalsToRemove) {
          module->removeGlobal(name);
        }
      }
    }
    if (onlyArrays) {
      for (auto* fn : modifiedFuncs) {
        ReFinalize().walkFunctionInModule(fn, module);
      }
      return;
    }

    // Rebuild funcGets after Pattern 1 inlining so Pattern 2/3 have accurate
    // pointers and counts.
    funcGets.clear();
    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      struct GetCollector : public PostWalker<GetCollector> {
        Function* fn;
        std::unordered_map<
          Name,
          std::vector<std::pair<Function*, Expression**>>>& funcGets;
        GetCollector(
          Function* fn,
          std::unordered_map<
            Name,
            std::vector<std::pair<Function*, Expression**>>>& funcGets)
          : fn(fn), funcGets(funcGets) {}
        void visitGlobalGet(GlobalGet* curr) {
          funcGets[curr->name].push_back({fn, getCurrentPointer()});
        }
      };
      GetCollector(func.get(), funcGets).walk(func->body);
    }

    // -------------------------------------------------------------------------
    // Pattern 2: Demote same-block 1-set mutable globals in once-regions
    // -------------------------------------------------------------------------
    {
      std::unordered_set<Name> candidateMutGlobals;
      for (auto& g : module->globals) {
        if (g->imported() || !g->mutable_ || onceFlags.count(g->name) ||
            nonCodeGlobalUses.count(g->name) || initGlobalUses[g->name] > 0) {
          continue;
        }
        bool isDefaultInit =
          g->init->is<RefNull>() ||
          (g->init->is<Const>() && g->init->cast<Const>()->value.isZero());
        if (!isDefaultInit) {
          continue;
        }
        auto setIt = funcSets.find(g->name);
        auto getIt = funcGets.find(g->name);
        if (setIt == funcSets.end() || setIt->second.size() != 1 ||
            getIt == funcGets.end() || getIt->second.empty()) {
          continue;
        }
        Function* fn = setIt->second[0].first;
        bool sameFn = true;
        for (auto& [gfn, _] : getIt->second) {
          if (gfn != fn) {
            sameFn = false;
            break;
          }
        }
        if (sameFn) {
          candidateMutGlobals.insert(g->name);
        }
      }

      if (!candidateMutGlobals.empty()) {
        std::vector<Name> demotedGlobals;
        for (auto& func : module->functions) {
          if (func->imported()) {
            continue;
          }
          Function* fn = func.get();
          std::function<void(Expression*, bool, bool)> walkBlocks =
            [&](Expression* curr, bool inOnce, bool inLoopOrTry) {
              if (!curr) {
                return;
              }
              if (curr->is<Loop>() || curr->is<Try>() || curr->is<TryTable>()) {
                inLoopOrTry = true;
              }
              if (auto* iff = curr->dynCast<If>()) {
                walkBlocks(iff->condition, inOnce, inLoopOrTry);
                walkBlocks(iff->ifTrue, inOnce || isOnceIf(iff), inLoopOrTry);
                if (iff->ifFalse) {
                  walkBlocks(iff->ifFalse, inOnce, inLoopOrTry);
                }
                return;
              }
              if (auto* b = curr->dynCast<Block>()) {
                if (inOnce && !inLoopOrTry) {
                  for (Index i = 0; i < b->list.size(); i++) {
                    auto* gs = b->list[i]->dynCast<GlobalSet>();
                    if (!gs || !candidateMutGlobals.count(gs->name)) {
                      continue;
                    }
                    Name gname = gs->name;
                    // Check 0 gets of gname in gs->value.
                    bool selfGet = false;
                    for (auto* gg : FindAll<GlobalGet>(gs->value).list) {
                      if (gg->name == gname) {
                        selfGet = true;
                        break;
                      }
                    }
                    if (selfGet) {
                      continue;
                    }
                    // Collect all GlobalGet** of gname in b->list[i + 1 ..].
                    std::vector<Expression**> afterGets;
                    std::function<void(Expression**)> collectGets =
                      [&](Expression** p) {
                        if (!p || !*p) {
                          return;
                        }
                        if (auto* gg = (*p)->dynCast<GlobalGet>()) {
                          if (gg->name == gname) {
                            afterGets.push_back(p);
                          }
                          return;
                        }
                        for (auto*& ch : ChildIterator(*p)) {
                          collectGets(&ch);
                        }
                      };
                    for (Index j = i + 1; j < b->list.size(); j++) {
                      collectGets(&b->list[j]);
                    }
                    if (afterGets.size() == funcGets[gname].size()) {
                      Type gtype = module->getGlobal(gname)->type;
                      Index loc = Builder::addVar(fn, gtype);
                      b->list[i] = builder.makeLocalSet(loc, gs->value);
                      for (auto** gp : afterGets) {
                        *gp = builder.makeLocalGet(loc, gtype);
                      }
                      modifiedFuncs.insert(fn);
                      demotedGlobals.push_back(gname);
                      candidateMutGlobals.erase(gname);
                      funcSets.erase(gname);
                      funcGets.erase(gname);
                    }
                  }
                }
              }
              for (auto* ch : ChildIterator(curr)) {
                walkBlocks(ch, inOnce, inLoopOrTry);
              }
            };
          walkBlocks(fn->body, onceHelperFuncs.count(fn->name) > 0, false);
        }
        for (auto name : demotedGlobals) {
          module->removeGlobal(name);
        }
      }
    }

    // -------------------------------------------------------------------------
    // Pattern 3: Eliminate redundant i32 once-flag globals
    // -------------------------------------------------------------------------
    {
      auto isGuaranteedNonNull = [&](Expression* val,
                                     const ExpressionList& stmts) -> bool {
        if (val->type.isRef() && val->type.isNonNullable()) {
          return true;
        }
        if (auto* lg = val->dynCast<LocalGet>()) {
          bool foundNonNullSet = false;
          for (Index i = 1; i + 1 < stmts.size(); i++) {
            for (auto* ls : FindAll<LocalSet>(stmts[i]).list) {
              if (ls->index == lg->index) {
                if (ls->value->type.isRef() &&
                    ls->value->type.isNonNullable()) {
                  foundNonNullSet = true;
                } else {
                  return false;
                }
              }
            }
          }
          return foundNonNullSet;
        }
        return false;
      };

      auto canExitOrModifyGlobals = [&](const ExpressionList& stmts,
                                        Name topBlockName) -> bool {
        for (Index i = 1; i + 1 < stmts.size(); i++) {
          Expression* s = stmts[i];
          if (!FindAll<GlobalSet>(s).list.empty() ||
              !FindAll<Return>(s).list.empty() ||
              !FindAll<Throw>(s).list.empty() ||
              !FindAll<Rethrow>(s).list.empty() ||
              !FindAll<Unreachable>(s).list.empty()) {
            return true;
          }
          if (topBlockName.is()) {
            for (auto* br : FindAll<Break>(s).list) {
              if (br->name == topBlockName) {
                return true;
              }
            }
            for (auto* sw : FindAll<Switch>(s).list) {
              if (sw->default_ == topBlockName ||
                  std::find(sw->targets.begin(),
                            sw->targets.end(),
                            topBlockName) != sw->targets.end()) {
                return true;
              }
            }
            for (auto* bron : FindAll<BrOn>(s).list) {
              if (bron->name == topBlockName) {
                return true;
              }
            }
          }
        }
        auto* lastSet = stmts.back()->cast<GlobalSet>();
        return !FindAll<GlobalSet>(lastSet->value).list.empty();
      };

      auto funcTouchesGlobal = [&](Function* f, Name g) -> bool {
        if (auto it = funcGets.find(g); it != funcGets.end()) {
          for (auto& [fn, _] : it->second) {
            if (fn == f) {
              return true;
            }
          }
        }
        if (auto it = funcSets.find(g); it != funcSets.end()) {
          for (auto& [fn, _] : it->second) {
            if (fn == f) {
              return true;
            }
          }
        }
        return false;
      };

      auto canReachTargetOrFlag = [&](const ExpressionList& stmts,
                                      Name flag,
                                      Name tg,
                                      Function* selfFn) -> bool {
        std::vector<Name> queue;
        std::unordered_set<Name> visited;
        auto scanExpr = [&](Expression* e) -> bool {
          for (auto* gg : FindAll<GlobalGet>(e).list) {
            if (gg->name == flag || gg->name == tg) {
              return true;
            }
          }
          for (auto* c : FindAll<Call>(e).list) {
            if (visited.insert(c->target).second) {
              queue.push_back(c->target);
            }
          }
          return false;
        };
        for (Index i = 1; i + 1 < stmts.size(); i++) {
          if (scanExpr(stmts[i])) {
            return true;
          }
        }
        if (scanExpr(stmts.back()->cast<GlobalSet>()->value)) {
          return true;
        }
        while (!queue.empty()) {
          Name fnName = queue.back();
          queue.pop_back();
          if (fnName == selfFn->name) {
            return true;
          }
          Function* f = module->getFunctionOrNull(fnName);
          if (!f || f->imported()) {
            continue;
          }
          if (funcTouchesGlobal(f, flag) || funcTouchesGlobal(f, tg)) {
            return true;
          }
          for (auto callee : funcCallees[fnName]) {
            if (visited.insert(callee).second) {
              queue.push_back(callee);
            }
          }
        }
        return false;
      };

      auto validateOnceBlock = [&](Block* topB,
                                   Name flag,
                                   Function* selfFn,
                                   Global*& tgGlobOut,
                                   Index& flagIdxOut) -> bool {
        if (!topB || topB->list.size() < 2) {
          return false;
        }
        Index flagIdx = 0;
        while (flagIdx < topB->list.size() &&
               isTrivialLocalInit(topB->list[flagIdx])) {
          flagIdx++;
        }
        if (flagIdx + 1 >= topB->list.size()) {
          return false;
        }
        auto* firstSet = topB->list[flagIdx]->dynCast<GlobalSet>();
        auto* lastSet = topB->list.back()->dynCast<GlobalSet>();
        if (!firstSet || firstSet->name != flag || !lastSet) {
          return false;
        }
        Name tg = lastSet->name;
        if (tg == flag || nonCodeGlobalUses.count(tg) ||
            initGlobalUses[tg] > 0 || funcSets[tg].size() != 1) {
          return false;
        }
        Global* tgGlob = module->getGlobalOrNull(tg);
        if (!tgGlob || tgGlob->imported() || !tgGlob->mutable_ ||
            !tgGlob->type.isRef() || !tgGlob->type.isNullable() ||
            !tgGlob->init->is<RefNull>()) {
          return false;
        }
        if (!isGuaranteedNonNull(lastSet->value, topB->list)) {
          return false;
        }
        if (canExitOrModifyGlobals(topB->list, topB->name)) {
          return false;
        }
        if (canReachTargetOrFlag(topB->list, flag, tg, selfFn)) {
          return false;
        }
        tgGlobOut = tgGlob;
        flagIdxOut = flagIdx;
        return true;
      };

      std::vector<Name> flagsToRemove;
      for (auto& g : module->globals) {
        Name flag = g->name;
        if (!onceFlags.count(flag) || funcSets[flag].size() != 1 ||
            funcGets[flag].empty()) {
          continue;
        }
        auto& guards = flagGuardIfs[flag];
        if (guards.size() != funcGets[flag].size()) {
          continue;
        }
        // Case 1: onceHelperFunc form
        if (auto it = flagHelperFunc.find(flag); it != flagHelperFunc.end()) {
          Function* helperFn = it->second;
          Block* topB = getEditableBlock(helperFn->body);
          Global* tgGlob = nullptr;
          Index flagIdx = 0;
          if (validateOnceBlock(topB, flag, helperFn, tgGlob, flagIdx)) {
            for (auto& [callerFn, iff] : guards) {
              iff->condition = builder.makeRefIsNull(
                builder.makeGlobalGet(tgGlob->name, tgGlob->type));
              modifiedFuncs.insert(callerFn);
            }
            topB->list.erase(topB->list.begin() + flagIdx);
            modifiedFuncs.insert(helperFn);
            flagsToRemove.push_back(flag);
            continue;
          }
        }
        // Case 2: inline onceIf form (1 get in the same function)
        if (guards.size() == 1) {
          auto [fn, iff] = guards[0];
          if (funcSets[flag][0].first == fn) {
            Block* topB = getEditableBlock(iff->ifTrue);
            Global* tgGlob = nullptr;
            Index flagIdx = 0;
            if (validateOnceBlock(topB, flag, fn, tgGlob, flagIdx)) {
              iff->condition = builder.makeRefIsNull(
                builder.makeGlobalGet(tgGlob->name, tgGlob->type));
              topB->list.erase(topB->list.begin() + flagIdx);
              modifiedFuncs.insert(fn);
              flagsToRemove.push_back(flag);
              continue;
            }
          }
        }
      }
      for (auto flag : flagsToRemove) {
        module->removeGlobal(flag);
      }
    }

    for (auto* fn : modifiedFuncs) {
      ReFinalize().walkFunctionInModule(fn, module);
    }
  }
};

} // anonymous namespace

Pass* createDeferGlobalAllocationsPass() {
  return new DeferGlobalAllocations();
}

} // namespace wasm
