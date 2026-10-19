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

#include "ir/effects.h"
#include "ir/find_all.h"
#include "ir/iteration.h"
#include "ir/manipulation.h"
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
    // Parameter indices that flow directly to struct.new operands.
    std::unordered_set<Index> allocParams;
    // (callee, argIndex, candidateGlobal) for calls passing a candidate.
    std::vector<std::tuple<Name, Index, Name>> callArgCandidates;
    // Uses of imported globals.
    std::unordered_map<Name, Index> importUses;
    // Total GlobalGet count and valid ArrayGet** sites for candidate arrays.
    std::unordered_map<Name, Index> arrayGlobalGets;
    std::unordered_map<Name, Index> arrayGlobalOnceGets;
    std::unordered_map<Name, std::vector<Expression**>> arrayGetSites;
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
    // Index in an absorbed array.new_fixed global, or Index(-1) if not absorbed.
    Index absorbedIndex = Index(-1);
  };

  struct SecondaryArrayInfo {
    Name arrGlobal;
    Index callIndexOffset = 0;
  };

  struct Group {
    HeapType type;
    Type globalType;
    // The field each leaf is stored in.
    std::vector<Field> leafFields;
    std::vector<Candidate> members;
    std::vector<std::pair<Name, Index>> primaryArrays;
    Index absorbedCount = 0;
    Name absorbedFactory;
    std::vector<Name> secondaryArrays;
    std::vector<SecondaryArrayInfo> secondaryArrayInfos;
  };

  struct Deferral {
    Name factory;
    Index index;
  };

  using GroupKey = std::tuple<HeapType, Type, std::string>;

  struct PaletteSlice {
    Index offset;
    Index length;
  };

  Module* module;
  std::unordered_map<Name, Index> globalIndexes;
  std::unordered_map<Name, Index> functionIndexes;
  std::unordered_map<Name, StructNew*> soleCandidateStructInits;
  std::unordered_map<Name, std::vector<PaletteSlice>> paletteSlices;
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
    paletteSlices.clear();
    includeAllocationOperands =
      !getArgumentOrDefault("defer-global-allocations-allocation-operands", "")
         .empty();
    preserveThunkTypes =
      !getArgumentOrDefault("defer-global-allocations-preserve-thunk-types", "")
         .empty();
    auto instArg = getArgumentOrDefault("defer-global-allocations", "");
    onceOnly =
      instArg == "once" || instArg == "once-only" ||
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
      paletteSlices.clear();
    }
    runGroupDeferral(/*skipPreservedThunkGroups=*/false);
    if (!onceOnly) {
      runGroupDeferral(/*skipPreservedThunkGroups=*/true);
      nullifySoleUsePaletteElements();
    }
    optimizeOnceGlobals(/*onlyArrays=*/false);
    if (!preserveThunkTypes && !always) {
      foldArrayGlobals();
    }
    if (!segmentData.empty()) {
      module->addDataSegment(Builder::makeDataSegment(
        segmentName, Name(), nullptr, segmentData.data(), segmentData.size()));
    }
  }

  void runGroupDeferral(bool skipPreservedThunkGroups = false) {
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
      if (skipPreservedThunkGroups && preserveThunkTypes) {
        bool hasSig = hasImmutableSigField(Type(type, NonNullable));
        if (!hasSig) {
          for (auto& f : leafFields) {
            if (f.mutable_ == Immutable && f.type.isRef() &&
                f.type.getHeapType().isSignature()) {
              hasSig = true;
              break;
            }
          }
        }
        if (hasSig) {
          continue;
        }
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
    std::unordered_set<Name> otherNonCodeUses;
    std::unordered_map<Name, Index> initGlobalUses;
    // Uses of imports anywhere except in the initializers of candidates.
    std::unordered_map<Name, Index> importOtherUses;
    // Uses of imports in the initializers of candidates.
    std::unordered_map<Name, Index> importCandidateUses;
    {
      struct Scanner : public PostWalker<Scanner> {
        DeferGlobalAllocations& parent;
        std::unordered_set<Name>& names;
        std::unordered_set<Name>* otherNames = nullptr;
        std::unordered_map<Name, Index>* initCounts = nullptr;
        std::unordered_map<Name, Index>* importUses = nullptr;
        Scanner(DeferGlobalAllocations& parent, std::unordered_set<Name>& names)
          : parent(parent), names(names) {}
        void visitGlobalGet(GlobalGet* curr) {
          names.insert(curr->name);
          if (initCounts) {
            (*initCounts)[curr->name]++;
          }
          if (otherNames) {
            otherNames->insert(curr->name);
          }
          if (parent.module->getGlobal(curr->name)->imported()) {
            (*importUses)[curr->name]++;
          }
        }
      } scanner(*this, nonCodeUses);
      scanner.initCounts = &initGlobalUses;
      for (auto& global : module->globals) {
        if (!global->imported()) {
          scanner.importUses = candidateMap.count(global->name)
                                 ? &importCandidateUses
                                 : &importOtherUses;
          scanner.walk(global->init);
        }
      }
      scanner.initCounts = nullptr;
      scanner.otherNames = &otherNonCodeUses;
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
          otherNonCodeUses.insert(*ex->getInternalName());
          importOtherUses[*ex->getInternalName()]++;
        }
      }
    }

    // Identify immutable array.new_fixed globals whose elements are GlobalGets
    // of members of a single candidate Group.
    std::unordered_map<Name, Group*> candidateArrays;
    std::vector<Name> candidateArrayOrder;
    std::unordered_set<Name> allDistinctArrays;
    for (auto& global : module->globals) {
      if (global->imported() || global->mutable_ ||
          nonCodeUses.count(global->name)) {
        continue;
      }
      auto* anf = global->init->dynCast<ArrayNewFixed>();
      if (!anf || anf->values.size() < 2 || anf->values.size() > MaxChunkSize) {
        continue;
      }
      Group* commonGroup = nullptr;
      bool valid = true;
      bool allDistinct = true;
      std::unordered_set<Name> seenElems;
      for (auto* val : anf->values) {
        auto* gg = val->dynCast<GlobalGet>();
        if (!gg || otherNonCodeUses.count(gg->name)) {
          valid = false;
          break;
        }
        if (!seenElems.insert(gg->name).second) {
          allDistinct = false;
        }
        auto it = candidateMap.find(gg->name);
        if (it == candidateMap.end()) {
          valid = false;
          break;
        }
        if (!commonGroup) {
          commonGroup = it->second.first;
        } else if (commonGroup != it->second.first) {
          valid = false;
          break;
        }
        if (isPreservedThunkGroup(*commonGroup)) {
          if (initGlobalUses[gg->name] != 1 || !allDistinct) {
            valid = false;
            break;
          }
        }
      }
      if (valid && commonGroup &&
          (!isPreservedThunkGroup(*commonGroup) ||
           commonGroup->primaryArrays.empty())) {
        candidateArrays[global->name] = commonGroup;
        candidateArrayOrder.push_back(global->name);
        if (allDistinct) {
          allDistinctArrays.insert(global->name);
        }
      }
    }
    std::stable_sort(
      candidateArrayOrder.begin(), candidateArrayOrder.end(), [&](Name a, Name b) {
        Group* ga = candidateArrays[a];
        Group* gb = candidateArrays[b];
        if (ga == gb && isPreservedThunkGroup(*ga)) {
          return false;
        }
        bool da = allDistinctArrays.count(a) > 0;
        bool db = allDistinctArrays.count(b) > 0;
        if (da != db) {
          return da;
        }
        return module->getGlobal(a)->init->cast<ArrayNewFixed>()->values.size() >
               module->getGlobal(b)->init->cast<ArrayNewFixed>()->values.size();
      });

    // Count the uses in function code, in parallel.
    ModuleUtils::ParallelFunctionAnalysis<FunctionInfo> analysis(
      *module, [&](Function* func, FunctionInfo& info) {
        if (func->imported()) {
          return;
        }
        struct Scanner : public PostWalker<Scanner> {
          DeferGlobalAllocations& parent;
          Module& module;
          Function* func;
          FunctionInfo& info;
          const std::unordered_map<Name, std::pair<Group*, Index>>& candidates;
          const std::unordered_map<Name, Group*>& candidateArrays;
          const std::unordered_set<Expression*>& onceGets;
          Scanner(DeferGlobalAllocations& parent,
                  Module& module,
                  Function* func,
                  FunctionInfo& info,
                  const std::unordered_map<Name, std::pair<Group*, Index>>&
                    candidates,
                  const std::unordered_map<Name, Group*>& candidateArrays,
                  const std::unordered_set<Expression*>& onceGets)
            : parent(parent), module(module), func(func), info(info),
              candidates(candidates), candidateArrays(candidateArrays),
              onceGets(onceGets) {}
          static Expression* unwrapRef(Expression* expr) {
            while (true) {
              if (auto* ra = expr->dynCast<RefAs>()) {
                expr = ra->value;
              } else if (auto* rc = expr->dynCast<RefCast>()) {
                expr = rc->ref;
              } else {
                return expr;
              }
            }
          }
          void visitGlobalGet(GlobalGet* curr) {
            if (candidates.count(curr->name)) {
              info.gets.push_back(getCurrentPointer());
              if (onceGets.count(curr)) {
                info.onceUses[curr->name]++;
              }
            } else if (candidateArrays.count(curr->name)) {
              info.arrayGlobalGets[curr->name]++;
              if (onceGets.count(curr)) {
                info.arrayGlobalOnceGets[curr->name]++;
              }
            } else if (module.getGlobal(curr->name)->imported()) {
              info.importUses[curr->name]++;
            }
          }
          void visitArrayGet(ArrayGet* curr) {
            if (curr->order != MemoryOrder::Unordered) {
              return;
            }
            if (auto* gg = curr->ref->dynCast<GlobalGet>()) {
              if (candidateArrays.count(gg->name)) {
                info.arrayGetSites[gg->name].push_back(getCurrentPointer());
              }
            }
          }
          void visitStructNew(StructNew* curr) {
            for (auto* operand : curr->operands) {
              Expression* base = unwrapRef(operand);
              if (auto* get = base->dynCast<GlobalGet>()) {
                auto it = candidates.find(get->name);
                if (it != candidates.end() &&
                    (parent.isPreservedThunkGroup(*it->second.first) ||
                     !onceGets.count(get))) {
                  info.allocOperandUses[get->name]++;
                }
              } else if (auto* lg = base->dynCast<LocalGet>()) {
                if (lg->index < func->getNumParams()) {
                  info.allocParams.insert(lg->index);
                }
              }
            }
          }
          void visitCall(Call* curr) {
            for (Index i = 0; i < curr->operands.size(); i++) {
              if (auto* get = unwrapRef(curr->operands[i])->dynCast<GlobalGet>()) {
                auto it = candidates.find(get->name);
                if (it != candidates.end() &&
                    (parent.isPreservedThunkGroup(*it->second.first) ||
                     !onceGets.count(get))) {
                  info.callArgCandidates.emplace_back(
                    curr->target, i, get->name);
                }
              }
            }
          }
        };
        Scanner(*this,
                *module,
                func,
                info,
                candidateMap,
                candidateArrays,
                onceGets)
          .walk(func->body);
      });

    // Merge the per-function information in a deterministic order.
    std::vector<Expression**> allGets;
    std::unordered_map<Name, std::unordered_set<Index>> funcAllocParams;
    std::unordered_map<Name, Index> totalArrayGlobalGets;
    std::unordered_map<Name, Index> totalArrayGlobalOnceGets;
    std::unordered_map<Name, std::vector<std::pair<Function*, Expression**>>>
      allArrayGetSites;
    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      auto& info = analysis.map[func.get()];
      if (!info.allocParams.empty()) {
        funcAllocParams[func->name] = info.allocParams;
      }
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
      for (auto& [name, count] : info.arrayGlobalGets) {
        totalArrayGlobalGets[name] += count;
      }
      for (auto& [name, count] : info.arrayGlobalOnceGets) {
        totalArrayGlobalOnceGets[name] += count;
      }
      for (auto& [name, sites] : info.arrayGetSites) {
        for (auto** site : sites) {
          allArrayGetSites[name].push_back({func.get(), site});
        }
      }
    }
    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      for (auto& [callee, argIdx, candName] :
           analysis.map[func.get()].callArgCandidates) {
        auto it = funcAllocParams.find(callee);
        if (it != funcAllocParams.end() && it->second.count(argIdx)) {
          auto& [group, i] = candidateMap[candName];
          group->members[i].allocOperandUses++;
        }
      }
    }

    // Absorb valid candidate array.new_fixed globals into their group.
    std::unordered_map<Group*, std::vector<Name>> groupCandidateArrays;
    for (Name arrName : candidateArrayOrder) {
      Group* grp = candidateArrays[arrName];
      Index totalGets = totalArrayGlobalGets[arrName];
      auto& sites = allArrayGetSites[arrName];
      if (totalGets == 0 || totalGets != sites.size()) {
        continue;
      }
      if (!allDistinctArrays.count(arrName) &&
          !module->features.hasBulkMemory()) {
        continue;
      }
      groupCandidateArrays[grp].push_back(arrName);
    }
    for (auto& [key, group] : groups) {
      Group* grp = &group;
      auto it = groupCandidateArrays.find(grp);
      if (it == groupCandidateArrays.end() || it->second.empty()) {
        continue;
      }
      if (isPreservedThunkGroup(*grp)) {
        for (Name arrName : it->second) {
          if (!grp->primaryArrays.empty()) {
            break;
          }
          auto* anf = module->getGlobal(arrName)->init->cast<ArrayNewFixed>();
          bool elementsUnusedInCode = true;
          for (auto* val : anf->values) {
            Name ename = val->cast<GlobalGet>()->name;
            auto& [_, idx] = candidateMap[ename];
            if (grp->members[idx].uses != 0) {
              elementsUnusedInCode = false;
            }
          }
          if (!elementsUnusedInCode) {
            continue;
          }
          Index totalGets = totalArrayGlobalGets[arrName];
          bool treatAsOnce = totalArrayGlobalOnceGets[arrName] == totalGets;
          grp->primaryArrays.push_back({arrName, 0});
          grp->absorbedCount = anf->values.size();
          for (Index k = 0; k < anf->values.size(); k++) {
            Name elemName = anf->values[k]->cast<GlobalGet>()->name;
            nonCodeUses.erase(elemName);
            auto& [_, idx] = candidateMap[elemName];
            auto& cand = grp->members[idx];
            cand.uses += 1;
            cand.onceUses += treatAsOnce ? 1 : 0;
            cand.absorbedIndex = k;
            for (auto* leaf : cand.leaves) {
              if (auto* lg = leaf->dynCast<GlobalGet>()) {
                if (module->getGlobal(lg->name)->imported()) {
                  importOtherUses[lg->name]--;
                  importCandidateUses[lg->name]++;
                }
              }
            }
          }
        }
        continue;
      }

      std::vector<Name> validArrays = it->second;
      while (!validArrays.empty()) {
        std::unordered_map<Name, Index> arrayElemUses;
        for (Name arrName : validArrays) {
          auto* anf = module->getGlobal(arrName)->init->cast<ArrayNewFixed>();
          for (auto* val : anf->values) {
            arrayElemUses[val->cast<GlobalGet>()->name]++;
          }
        }
        std::vector<Name> nextValid;
        for (Name arrName : validArrays) {
          auto* anf = module->getGlobal(arrName)->init->cast<ArrayNewFixed>();
          bool ok = true;
          for (auto* val : anf->values) {
            Name ename = val->cast<GlobalGet>()->name;
            if (initGlobalUses[ename] != arrayElemUses[ename]) {
              ok = false;
              break;
            }
          }
          if (ok) {
            nextValid.push_back(arrName);
          }
        }
        if (nextValid.size() == validArrays.size()) {
          break;
        }
        validArrays = std::move(nextValid);
      }
      if (validArrays.empty()) {
        continue;
      }

      // Filter out connected components of validArrays whose elements have too
      // many direct uses in code to be profitably deferred.
      std::vector<Name> filteredArrays;
      std::vector<bool> visitedArr(validArrays.size(), false);
      for (Index i = 0; i < validArrays.size(); i++) {
        if (visitedArr[i]) {
          continue;
        }
        std::vector<Index> compArrIdxs = {i};
        visitedArr[i] = true;
        std::unordered_set<Name> compElems;
        for (Index q = 0; q < compArrIdxs.size(); q++) {
          Name aName = validArrays[compArrIdxs[q]];
          auto* anf = module->getGlobal(aName)->init->cast<ArrayNewFixed>();
          for (auto* val : anf->values) {
            compElems.insert(val->cast<GlobalGet>()->name);
          }
          for (Index j = 0; j < validArrays.size(); j++) {
            if (visitedArr[j]) {
              continue;
            }
            auto* anfJ =
              module->getGlobal(validArrays[j])->init->cast<ArrayNewFixed>();
            bool shares = false;
            for (auto* val : anfJ->values) {
              if (compElems.count(val->cast<GlobalGet>()->name)) {
                shares = true;
                break;
              }
            }
            if (shares) {
              visitedArr[j] = true;
              compArrIdxs.push_back(j);
            }
          }
        }
        int64_t compSaving = 0;
        for (Name ename : compElems) {
          auto& [_, idx] = candidateMap[ename];
          auto& cand = grp->members[idx];
          Index getBytes =
            1 + getLEBSize(U32LEB(globalIndexes[cand.global->name]));
          int64_t siteDelta = 5 - int64_t(getBytes);
          int64_t elemBonus = cand.uses > 0 ? int64_t(getBytes) : 0;
          compSaving += int64_t(cand.initBytes) + int64_t(cand.bonus) +
                        elemBonus - 1 - int64_t(cand.uses) * siteDelta;
        }
        if (always || compSaving > 0) {
          for (Index aIdx : compArrIdxs) {
            filteredArrays.push_back(validArrays[aIdx]);
          }
        }
      }
      if (filteredArrays.empty()) {
        continue;
      }

      std::vector<std::pair<Name, Index>> primaryArrays;
      std::vector<Name> secArrays;
      std::unordered_set<Name> distinctAbsorbed;
      std::vector<Name> absorbedOrder;
      bool canUseData = module->features.hasBulkMemory();
      for (Name arrName : filteredArrays) {
        auto* anf = module->getGlobal(arrName)->init->cast<ArrayNewFixed>();
        bool canBePrimary =
          allDistinctArrays.count(arrName) &&
          absorbedOrder.size() + anf->values.size() <= MaxChunkSize;
        if (canBePrimary) {
          for (auto* val : anf->values) {
            if (distinctAbsorbed.count(val->cast<GlobalGet>()->name)) {
              canBePrimary = false;
              break;
            }
          }
        }
        if (canBePrimary) {
          primaryArrays.push_back({arrName, Index(absorbedOrder.size())});
          for (auto* val : anf->values) {
            Name ename = val->cast<GlobalGet>()->name;
            distinctAbsorbed.insert(ename);
            absorbedOrder.push_back(ename);
          }
        } else if (canUseData) {
          std::vector<Name> newElems;
          for (auto* val : anf->values) {
            Name ename = val->cast<GlobalGet>()->name;
            if (!distinctAbsorbed.count(ename) &&
                std::find(newElems.begin(), newElems.end(), ename) ==
                  newElems.end()) {
              newElems.push_back(ename);
            }
          }
          if (distinctAbsorbed.size() + newElems.size() <= 256) {
            secArrays.push_back(arrName);
            for (Name ename : newElems) {
              distinctAbsorbed.insert(ename);
              absorbedOrder.push_back(ename);
            }
          }
        }
      }
      if (absorbedOrder.empty()) {
        continue;
      }
      // Verify that every absorbed element's initializer uses are accounted for
      // by primaryArrays + secArrays.
      std::unordered_map<Name, Index> chosenArrElemUses;
      for (auto& [pArr, _] : primaryArrays) {
        auto* anf = module->getGlobal(pArr)->init->cast<ArrayNewFixed>();
        for (auto* val : anf->values) {
          chosenArrElemUses[val->cast<GlobalGet>()->name]++;
        }
      }
      for (Name sArr : secArrays) {
        auto* anf = module->getGlobal(sArr)->init->cast<ArrayNewFixed>();
        for (auto* val : anf->values) {
          chosenArrElemUses[val->cast<GlobalGet>()->name]++;
        }
      }
      bool allAccounted = true;
      for (Name ename : absorbedOrder) {
        if (initGlobalUses[ename] != chosenArrElemUses[ename]) {
          allAccounted = false;
          break;
        }
      }
      if (!allAccounted) {
        continue;
      }

      bool allArraysOnce = true;
      for (auto& [pArr, _] : primaryArrays) {
        if (totalArrayGlobalOnceGets[pArr] != totalArrayGlobalGets[pArr]) {
          allArraysOnce = false;
        }
      }
      for (Name sArr : secArrays) {
        if (totalArrayGlobalOnceGets[sArr] != totalArrayGlobalGets[sArr]) {
          allArraysOnce = false;
        }
      }
      bool hasOtherCodeMembers = false;
      bool allOtherCodeMembersOnce = true;
      for (auto& m : grp->members) {
        if (!distinctAbsorbed.count(m.global->name) && m.uses > 0) {
          hasOtherCodeMembers = true;
          if (!(m.uses == 1 && m.onceUses == 1)) {
            allOtherCodeMembersOnce = false;
          }
        }
      }
      bool treatAsOnce =
        allArraysOnce || (hasOtherCodeMembers && allOtherCodeMembersOnce);
      grp->primaryArrays = std::move(primaryArrays);
      grp->secondaryArrays = std::move(secArrays);
      grp->absorbedCount = absorbedOrder.size();
      for (Index k = 0; k < absorbedOrder.size(); k++) {
        Name elemName = absorbedOrder[k];
        nonCodeUses.erase(elemName);
        auto& [_, idx] = candidateMap[elemName];
        auto& cand = grp->members[idx];
        cand.uses += 1;
        cand.onceUses += treatAsOnce ? 1 : 0;
        cand.absorbedIndex = k;
        for (auto* leaf : cand.leaves) {
          if (auto* lg = leaf->dynCast<GlobalGet>()) {
            if (module->getGlobal(lg->name)->imported()) {
              importOtherUses[lg->name]--;
              importCandidateUses[lg->name]++;
            }
          }
        }
      }
    }

    // Record sole-use flat StructNew candidates (used only in candidate/array
    // initializers) so palette profitability accounts for post-deferral
    // nullification.
    if (!skipPreservedThunkGroups) {
      soleCandidateStructInits.clear();
    } else {
      for (auto it = soleCandidateStructInits.begin();
           it != soleCandidateStructInits.end();) {
        if (!module->getGlobalOrNull(it->first) ||
            otherNonCodeUses.count(it->first) ||
            initGlobalUses[it->first] == 0) {
          it = soleCandidateStructInits.erase(it);
        } else {
          ++it;
        }
      }
    }
    for (auto& [key, group] : groups) {
      if (group.leafFields.size() != group.type.getStruct().fields.size()) {
        continue;
      }
      for (auto& cand : group.members) {
        if (cand.uses == 0 && !otherNonCodeUses.count(cand.global->name) &&
            initGlobalUses[cand.global->name] > 0 &&
            !(preserveThunkTypes && hasImmutableSigRefFunc(cand.init))) {
          soleCandidateStructInits[cand.global->name] = cand.init;
        }
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
      processGroup(group, deferrals, allArrayGetSites);
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
    std::unordered_set<Function*> refinalizeFuncs;
    for (auto& [key, group] : groups) {
      if (!group.absorbedFactory.is()) {
        continue;
      }
      for (auto& [arrGlobal, baseIdx] : group.primaryArrays) {
        for (auto& [fn, site] : allArrayGetSites[arrGlobal]) {
          auto* ag = (*site)->cast<ArrayGet>();
          Expression* idx = ag->index;
          if (baseIdx != 0) {
            if (auto* c = idx->dynCast<Const>()) {
              idx = builder.makeConst(
                int32_t(uint32_t(c->value.geti32()) + baseIdx));
            } else {
              idx = builder.makeBinary(
                AddInt32, idx, builder.makeConst(int32_t(baseIdx)));
            }
          }
          *site =
            builder.makeCall(group.absorbedFactory, {idx}, group.globalType);
          refinalizeFuncs.insert(fn);
        }
        module->removeGlobal(arrGlobal);
      }
      for (auto& secInfo : group.secondaryArrayInfos) {
        for (auto& [fn, site] : allArrayGetSites[secInfo.arrGlobal]) {
          auto* ag = (*site)->cast<ArrayGet>();
          Expression* idx = ag->index;
          if (secInfo.callIndexOffset != 0) {
            if (auto* c = idx->dynCast<Const>()) {
              idx = builder.makeConst(int32_t(uint32_t(c->value.geti32()) +
                                              secInfo.callIndexOffset));
            } else {
              idx = builder.makeBinary(
                AddInt32,
                idx,
                builder.makeConst(int32_t(secInfo.callIndexOffset)));
            }
          }
          *site =
            builder.makeCall(group.absorbedFactory, {idx}, group.globalType);
          refinalizeFuncs.insert(fn);
        }
        module->removeGlobal(secInfo.arrGlobal);
      }
    }
    for (auto* fn : refinalizeFuncs) {
      ReFinalize().walkFunctionInModule(fn, module);
    }
    for (auto& [name, _] : deferrals) {
      module->removeGlobal(name);
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

  enum SelectPredMode {
    InvalidSelectPred,
    SelectDirectIdx,
    SelectEqConst,
    SelectRangeLtU,
    SelectRangeGeU,
    SelectRangeMid,
    SelectBitmask
  };

  struct SelectArmPlan {
    SelectPredMode mode = InvalidSelectPred;
    Index param1 = 0;
    Index param2 = 0;
    Index condExtra = 0;
  };

  bool planSelectChainSteps(
    const std::vector<Expression*>& distinct,
    const std::vector<std::vector<Index>>& distinctIndices,
    Index N,
    Index& defaultIdxOut,
    std::vector<std::pair<Index, SelectArmPlan>>& stepsOut) {
    Index D = distinct.size();
    if (D < 2 || D > 5) {
      return false;
    }
    Index numMasks = 1u << D;
    std::vector<Index> fMin(numMasks, N);
    std::vector<Index> fMax(numMasks, 0);
    std::vector<Index> fCount(numMasks, 0);
    for (Index mask = 1; mask < numMasks; mask++) {
      for (Index d = 0; d < D; d++) {
        if (mask & (1u << d)) {
          const auto& ks = distinctIndices[d];
          fMin[mask] = std::min(fMin[mask], ks.front());
          fMax[mask] = std::max(fMax[mask], ks.back());
          fCount[mask] += ks.size();
        }
      }
    }

    auto computeStepPlan = [&](Index mask, Index nextD) -> SelectArmPlan {
      const auto& k1 = distinctIndices[nextD];
      Index k1Min = k1.front();
      Index k1Max = k1.back();
      SelectArmPlan best;
      auto tryPlan = [&](SelectPredMode m, Index p1, Index p2, Index cost) {
        if (best.mode == InvalidSelectPred || cost < best.condExtra) {
          best = {m, p1, p2, cost};
        }
      };
      if (fCount[mask] == 1 && fMin[mask] == 0) {
        tryPlan(SelectDirectIdx, 0, 0, 0);
      }
      if (k1.size() == 1) {
        Index c = k1[0];
        tryPlan(SelectEqConst, c, 0, (c == 0) ? 1 : (2 + getSLEBSize(c)));
      }
      if (k1Max < fMin[mask]) {
        Index bestLim = k1Max + 1;
        Index bestLimCost = 2 + getSLEBSize(bestLim);
        for (Index lim = k1Max + 2; lim <= fMin[mask]; lim++) {
          Index c = 2 + getSLEBSize(lim);
          if (c < bestLimCost) {
            bestLim = lim;
            bestLimCost = c;
          }
        }
        tryPlan(SelectRangeLtU, bestLim, 0, bestLimCost);
      }
      if (k1Min > fMax[mask]) {
        Index bestStart = k1Min;
        Index bestStartCost = 2 + getSLEBSize(bestStart);
        for (Index st = fMax[mask] + 1; st < k1Min; st++) {
          Index c = 2 + getSLEBSize(st);
          if (c < bestStartCost) {
            bestStart = st;
            bestStartCost = c;
          }
        }
        tryPlan(SelectRangeGeU, bestStart, 0, bestStartCost);
      }
      {
        bool conflict = false;
        for (Index d = 0; d < D; d++) {
          if (mask & (1u << d)) {
            for (Index f : distinctIndices[d]) {
              if (f >= k1Min && f <= k1Max) {
                conflict = true;
                break;
              }
            }
            if (conflict) {
              break;
            }
          }
        }
        if (!conflict) {
          Index len = k1Max - k1Min + 1;
          tryPlan(SelectRangeMid,
                  k1Min,
                  len,
                  4 + getSLEBSize(k1Min) + getSLEBSize(len));
        }
      }
      if (k1Max < 32 && fMax[mask] < 32) {
        uint32_t bm = 0;
        for (Index idx : k1) {
          bm |= (1u << idx);
        }
        tryPlan(SelectBitmask, Index(bm), 0, 5 + getSLEBSize(Index(bm)));
      }
      return best;
    };

    struct DPState {
      Index condSum = Index(-1);
      Index defaultCount = 0;
      Index prevMask = 0;
      Index addedArm = Index(-1);
      SelectArmPlan plan;
    };
    std::vector<DPState> dp(numMasks);
    for (Index d = 0; d < D; d++) {
      Index m = 1u << d;
      dp[m].condSum = 0;
      dp[m].defaultCount = distinctIndices[d].size();
    }
    for (Index mask = 1; mask < numMasks; mask++) {
      if (dp[mask].condSum == Index(-1)) {
        continue;
      }
      for (Index nextD = 0; nextD < D; nextD++) {
        if (mask & (1u << nextD)) {
          continue;
        }
        SelectArmPlan p = computeStepPlan(mask, nextD);
        if (p.mode == InvalidSelectPred) {
          continue;
        }
        Index nextMask = mask | (1u << nextD);
        Index nextCost = dp[mask].condSum + p.condExtra;
        if (dp[nextMask].condSum == Index(-1) ||
            nextCost < dp[nextMask].condSum ||
            (nextCost == dp[nextMask].condSum &&
             dp[mask].defaultCount > dp[nextMask].defaultCount)) {
          dp[nextMask] = {nextCost, dp[mask].defaultCount, mask, nextD, p};
        }
      }
    }
    Index fullMask = numMasks - 1;
    if (dp[fullMask].condSum == Index(-1)) {
      return false;
    }
    stepsOut.clear();
    Index curMask = fullMask;
    while (dp[curMask].prevMask != 0) {
      stepsOut.push_back({dp[curMask].addedArm, dp[curMask].plan});
      curMask = dp[curMask].prevMask;
    }
    std::reverse(stepsOut.begin(), stepsOut.end());
    defaultIdxOut = 0;
    while ((curMask & (1u << defaultIdxOut)) == 0) {
      defaultIdxOut++;
    }
    return true;
  }

  Expression* buildSelectCond(Builder& builder,
                              const SelectArmPlan& p,
                              Expression* idxExpr) {
    if (p.mode == SelectDirectIdx) {
      return idxExpr;
    }
    if (p.mode == SelectEqConst) {
      if (p.param1 == 0) {
        return builder.makeUnary(EqZInt32, idxExpr);
      }
      return builder.makeBinary(
        EqInt32, idxExpr, builder.makeConst(int32_t(p.param1)));
    }
    if (p.mode == SelectRangeLtU) {
      return builder.makeBinary(
        LtUInt32, idxExpr, builder.makeConst(int32_t(p.param1)));
    }
    if (p.mode == SelectRangeGeU) {
      return builder.makeBinary(
        GeUInt32, idxExpr, builder.makeConst(int32_t(p.param1)));
    }
    if (p.mode == SelectRangeMid) {
      return builder.makeBinary(
        LtUInt32,
        builder.makeBinary(
          SubInt32, idxExpr, builder.makeConst(int32_t(p.param1))),
        builder.makeConst(int32_t(p.param2)));
    }
    return builder.makeBinary(
      AndInt32,
      builder.makeBinary(
        ShrUInt32, builder.makeConst(int32_t(p.param1)), idxExpr),
      builder.makeConst(int32_t(1)));
  }

  bool tryPlanSelectChain(
    const std::vector<Expression*>& distinct,
    const std::vector<std::vector<Index>>& distinctIndices,
    Index N,
    Type resType,
    Index indexLocal,
    Expression** outExpr,
    Index* outCodeCost) {
    Index defaultIdx = 0;
    std::vector<std::pair<Index, SelectArmPlan>> steps;
    if (!planSelectChainSteps(
          distinct, distinctIndices, N, defaultIdx, steps)) {
      return false;
    }
    Expression* vDef = distinct[defaultIdx];
    Index selectOpBytes = resType.isRef() ? 4 : 1;
    Index baseSiteCost = getOperandBytes(vDef);
    for (auto& [d, p] : steps) {
      baseSiteCost +=
        getOperandBytes(distinct[d]) + p.condExtra + selectOpBytes;
    }
    Index M = steps.size();
    Index idxExtra = (M <= 1) ? 0 : (2 * (M - 1));
    if (outCodeCost) {
      *outCodeCost = baseSiteCost + idxExtra;
    }
    if (outExpr) {
      Builder builder(*module);
      Expression* acc = ExpressionManipulator::copy(vDef, *module);
      for (Index m = 0; m < M; m++) {
        Index d = steps[m].first;
        const SelectArmPlan& p = steps[m].second;
        Expression* idxExpr = builder.makeLocalGet(indexLocal, Type::i32);
        Expression* cond = buildSelectCond(builder, p, idxExpr);
        Expression* trueExpr =
          ExpressionManipulator::copy(distinct[d], *module);
        acc = builder.makeSelect(cond, trueExpr, acc);
      }
      *outExpr = acc;
    }
    return true;
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

  struct ColumnCosts {
    std::unordered_map<const Candidate*, Index> memberTableBytes;
    Index chunkColumnOverhead = 0;
  };

  Index getEffectivePaletteCount(const std::vector<Expression*>& distinct) {
    Index D = distinct.size();
    std::vector<StructNew*> soleTempls;
    Index mergedCount = 0;
    Index nonSoleCount = 0;
    for (auto* expr : distinct) {
      if (auto* gg = expr->dynCast<GlobalGet>()) {
        auto it = soleCandidateStructInits.find(gg->name);
        if (it != soleCandidateStructInits.end()) {
          StructNew* sn = it->second;
          bool found = false;
          for (auto* prev : soleTempls) {
            if (ExpressionAnalyzer::equal(prev, sn)) {
              found = true;
              break;
            }
          }
          if (found) {
            mergedCount++;
          } else {
            soleTempls.push_back(sn);
          }
          continue;
        }
      }
      nonSoleCount++;
    }
    return nonSoleCount > 0 ? (D - mergedCount) : D;
  }

  ColumnCosts computeColumnCosts(const Group& group,
                                 const std::vector<Candidate*>& members,
                                 const std::vector<Index>& varying,
                                 bool hasSecArrays) {
    ColumnCosts costs;
    Index N = members.size();
    if (N == 0) {
      return costs;
    }
    for (auto* m : members) {
      costs.memberTableBytes[m] = 0;
    }
    if (isPreservedThunkGroup(group)) {
      for (auto* m : members) {
        Index tb = 0;
        for (auto i : varying) {
          tb += getOperandBytes(m->leaves[i]);
        }
        costs.memberTableBytes[m] = tb;
      }
      costs.chunkColumnOverhead = 20 * varying.size();
      return costs;
    }
    auto& leafFields = group.leafFields;
    bool canUseData = module->features.hasBulkMemory();
    std::vector<Index> uniqueVarying;
    std::vector<Index> prevI32Cols;
    std::unordered_map<Index, Index> i32ColWidths;
    bool hasW1Col = hasSecArrays;

    for (auto i : varying) {
      auto& field = leafFields[i];
      Index dupOf = Index(-1);
      for (Index prev : uniqueVarying) {
        if (leafFields[prev].type == field.type &&
            leafFields[prev].isPacked() == field.isPacked() &&
            (!field.isPacked() ||
             leafFields[prev].packedType == field.packedType)) {
          bool allEq = true;
          for (Index k = 0; k < N; k++) {
            if (!ExpressionAnalyzer::equal(members[k]->leaves[prev],
                                           members[k]->leaves[i])) {
              allEq = false;
              break;
            }
          }
          if (allEq) {
            dupOf = prev;
            break;
          }
        }
      }
      if (dupOf != Index(-1)) {
        continue;
      }
      uniqueVarying.push_back(i);
      if (canUseData && N >= 7 && field.type == Type::i32 &&
          std::all_of(members.begin(), members.end(), [&](Candidate* m) {
            return m->leaves[i]->is<Const>();
          })) {
        auto evalWidth = [&](auto getVal) -> Index {
          int64_t mn = std::numeric_limits<int64_t>::max();
          int64_t mx = std::numeric_limits<int64_t>::min();
          for (Index k = 0; k < N; k++) {
            int64_t v = getVal(k);
            mn = std::min(mn, v);
            mx = std::max(mx, v);
          }
          Index w = 4;
          bool s = false;
          HeapType ht;
          classifyI32Range(mn, mx, w, s, ht);
          return w;
        };
        Index width = evalWidth([&](Index k) {
          int32_t raw = members[k]->leaves[i]->cast<Const>()->value.geti32();
          return getPackedOrI32Value(field, raw);
        });
        if (!field.isPacked()) {
          width = std::min(width, evalWidth([&](Index k) {
                             int32_t raw = members[k]
                                             ->leaves[i]
                                             ->cast<Const>()
                                             ->value.geti32();
                             return int64_t(
                               int32_t(uint32_t(raw) - uint32_t(k)));
                           }));
          for (Index prev : prevI32Cols) {
            width = std::min(width, evalWidth([&](Index k) {
                               int32_t raw = members[k]
                                               ->leaves[i]
                                               ->cast<Const>()
                                               ->value.geti32();
                               int32_t baseRaw = members[k]
                                                   ->leaves[prev]
                                                   ->cast<Const>()
                                                   ->value.geti32();
                               return int64_t(
                                 int32_t(uint32_t(raw) - uint32_t(baseRaw)));
                             }));
          }
          prevI32Cols.push_back(i);
        }
        i32ColWidths[i] = width;
        if (width == 1) {
          hasW1Col = true;
        }
      }
    }

    if (!hasW1Col && canUseData) {
      Index combinedPalGainAt8 = 0;
      Index palCountAt8 = 0;
      for (Index i : uniqueVarying) {
        if (i32ColWidths.count(i) || !isPaletteCandidate(leafFields[i])) {
          continue;
        }
        std::vector<Expression*> distinct;
        std::vector<std::vector<Index>> distinctIndices;
        bool ok = true;
        Index elemBytesSum = 0;
        for (Index k = 0; k < N; k++) {
          auto* leaf = members[k]->leaves[i];
          elemBytesSum += getOperandBytes(leaf);
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
            distinctIndices.push_back({k});
          } else {
            distinctIndices[idx].push_back(k);
          }
        }
        if (!ok) {
          continue;
        }
        Index D = distinct.size();
        Index effD = getEffectivePaletteCount(distinct);
        Index distinctBytes = 0;
        for (auto* expr : distinct) {
          distinctBytes += getOperandBytes(expr);
        }
        if (D > effD) {
          Index savedPal = 3 * (D - effD);
          distinctBytes =
            distinctBytes > savedPal ? distinctBytes - savedPal : 0;
        }
        Index uncompressedTableCost =
          9 + getLEBSize(U32LEB(N)) + elemBytesSum + 6;
        Index w1PaletteCostAt8 =
          9 + getLEBSize(U32LEB(effD)) + distinctBytes + N + 8;
        Index selectCodeCost = Index(-1);
        if (D >= 2 && D <= 5 && D < N && D == effD &&
            tryPlanSelectChain(distinct,
                               distinctIndices,
                               N,
                               leafFields[i].type,
                               0,
                               nullptr,
                               &selectCodeCost) &&
            selectCodeCost <=
              std::min(uncompressedTableCost, w1PaletteCostAt8)) {
          continue;
        }
        if (9 + 3 * effD + N + 25 < 9 + 3 * N) {
          hasW1Col = true;
          break;
        }
        if (9 + 3 * effD + N + 8 < 9 + 3 * N) {
          palCountAt8++;
          combinedPalGainAt8 += (9 + 3 * N) - (9 + 3 * effD + N + 8);
          if (palCountAt8 >= 2 && combinedPalGainAt8 > 17) {
            hasW1Col = true;
            break;
          }
        }
      }
    }

    Index dataWidthCols[5] = {0, 0, 0, 0, 0};
    Index uncompressedCols = 0;
    Index paletteOverhead = 0;
    for (Index i : uniqueVarying) {
      auto& field = leafFields[i];
      if (auto it = i32ColWidths.find(i); it != i32ColWidths.end()) {
        Index w = it->second;
        for (auto* m : members) {
          costs.memberTableBytes[m] += w;
        }
        dataWidthCols[w]++;
        continue;
      }
      if (canUseData && isPaletteCandidate(field)) {
        std::vector<Expression*> distinct;
        std::vector<std::vector<Index>> distinctIndices;
        bool ok = true;
        Index elemBytesSum = 0;
        for (Index k = 0; k < N; k++) {
          auto* leaf = members[k]->leaves[i];
          elemBytesSum += getOperandBytes(leaf);
          if (!ok) {
            continue;
          }
          Index idx = 0;
          while (idx < distinct.size() && !sameRefLeaf(distinct[idx], leaf)) {
            idx++;
          }
          if (idx == distinct.size()) {
            if (distinct.size() == 256) {
              ok = false;
            } else {
              distinct.push_back(leaf);
              distinctIndices.push_back({k});
            }
          } else {
            distinctIndices[idx].push_back(k);
          }
        }
        Index D = distinct.size();
        Index effD = ok ? getEffectivePaletteCount(distinct) : D;
        Index w1Extra = hasW1Col ? 8 : 25;
        if (ok && 9 + 3 * effD + N + w1Extra < 9 + 3 * N) {
          for (auto* m : members) {
            costs.memberTableBytes[m] += 1;
          }
          Index distinctBytes = 0;
          for (auto* expr : distinct) {
            distinctBytes += getOperandBytes(expr);
          }
          if (D > effD) {
            Index savedPal = 3 * (D - effD);
            distinctBytes =
              distinctBytes > savedPal ? distinctBytes - savedPal : 0;
          }
          Index uncompressedTableCost =
            9 + getLEBSize(U32LEB(N)) + elemBytesSum + 6;
          Index w1PaletteCost =
            9 + getLEBSize(U32LEB(effD)) + distinctBytes + N + w1Extra;
          Index selectCodeCost = Index(-1);
          if (D >= 2 && D <= 5 && D < N && D == effD &&
              tryPlanSelectChain(distinct,
                                 distinctIndices,
                                 N,
                                 field.type,
                                 0,
                                 nullptr,
                                 &selectCodeCost) &&
              selectCodeCost <=
                std::min(uncompressedTableCost, w1PaletteCost)) {
            paletteOverhead += (selectCodeCost > N ? selectCodeCost - N : 0);
            continue;
          }
          Index palBytes = 9 + distinctBytes;
          paletteOverhead += palBytes + 6;
          dataWidthCols[1]++;
          hasW1Col = true;
          continue;
        }
      }
      for (auto* m : members) {
        costs.memberTableBytes[m] += getOperandBytes(m->leaves[i]);
      }
      uncompressedCols++;
    }
    costs.chunkColumnOverhead = 20 * uncompressedCols + paletteOverhead;
    for (Index w : {Index(1), Index(2), Index(4)}) {
      Index c = dataWidthCols[w];
      if (w == 1 && hasSecArrays) {
        c++;
      }
      if (c > 0) {
        costs.chunkColumnOverhead += 20 + 8 * (c - 1);
      }
    }
    return costs;
  }

  void processGroup(
    Group& group,
    std::unordered_map<Name, Deferral>& deferrals,
    const std::unordered_map<
      Name,
      std::vector<std::pair<Function*, Expression**>>>& allArrayGetSites) {
    auto& members = group.members;
    // Only globals that are used in code matter (unused ones are left to
    // other passes).
    members.erase(
      std::remove_if(members.begin(),
                     members.end(),
                     [&](const Candidate& c) { return c.uses == 0; }),
      members.end());
    if (members.empty()) {
      return;
    }
    bool preservedThunk = isPreservedThunkGroup(group);
    if (!includeAllocationOperands) {
      Index uses = 0, allocOperandUses = 0;
      for (auto& member : members) {
        uses += member.uses;
        allocOperandUses += member.allocOperandUses;
      }
      if (onceOnly ? allocOperandUses > 0 : allocOperandUses * 2 > uses) {
        if (preservedThunk) {
          return;
        }
        members.erase(std::remove_if(members.begin(),
                                     members.end(),
                                     [&](const Candidate& c) {
                                       if (!onceOnly &&
                                           c.allocOperandUses == 0) {
                                         return false;
                                       }
                                       return !(c.uses == 1 &&
                                                c.onceUses == 1);
                                     }),
                      members.end());
        if (members.empty()) {
          return;
        }
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
      if (a->absorbedIndex != b->absorbedIndex) {
        return a->absorbedIndex < b->absorbedIndex;
      }
      return a->firstUse < b->firstUse;
    });

    // Estimate column costs over viable members (excluding hot members that
    // cannot be profitably deferred even with 0 tableBytes).
    std::vector<Candidate*> viable;
    for (auto* member : sorted) {
      if (always || member->absorbedIndex != Index(-1)) {
        viable.push_back(member);
        continue;
      }
      Index getBytes =
        1 + getLEBSize(U32LEB(globalIndexes[member->global->name]));
      int64_t minSiteDelta = int64_t(callBytes + 2) - int64_t(getBytes);
      int64_t maxSaving = int64_t(member->initBytes) + int64_t(member->bonus) -
                          int64_t(member->uses) * minSiteDelta;
      if (maxSaving > 0) {
        viable.push_back(member);
      }
    }
    if (viable.empty()) {
      return;
    }
    std::vector<Candidate> viableCandidates;
    viableCandidates.reserve(viable.size());
    for (auto* m : viable) {
      viableCandidates.push_back(*m);
    }
    auto viableVarying = getVaryingColumns(viableCandidates, numLeaves);
    auto initialCosts = computeColumnCosts(
      group, viable, viableVarying, !group.secondaryArrays.empty());

    auto getMemberTableBytes = [&](const Candidate* m) -> Index {
      auto it = initialCosts.memberTableBytes.find(m);
      if (it != initialCosts.memberTableBytes.end()) {
        return it->second;
      }
      Index tb = 0;
      for (auto i : varying) {
        tb += getOperandBytes(m->leaves[i]);
      }
      return tb;
    };

    std::vector<Candidate*> chosen;
    int64_t saving = 0;
    int64_t secArraysBonus = 0;
    bool hasAbsorbedArrays =
      !group.primaryArrays.empty() || !group.secondaryArrays.empty();
    if (!preservedThunk && hasAbsorbedArrays) {
      std::vector<Candidate*> absMembers;
      int64_t absSaving = 0;
      for (auto* member : sorted) {
        if (member->absorbedIndex != Index(-1)) {
          absMembers.push_back(member);
          Index tableBytes = getMemberTableBytes(member);
          Index index = member->absorbedIndex;
          Index directUses = member->uses - 1;
          Index getBytes =
            1 + getLEBSize(U32LEB(globalIndexes[member->global->name]));
          int64_t siteDelta =
            int64_t(callBytes + 1 + getSLEBSize(index)) - int64_t(getBytes);
          int64_t elemBonus = directUses > 0 ? int64_t(getBytes) : 0;
          absSaving += int64_t(member->initBytes) + int64_t(member->bonus) +
                       elemBonus - int64_t(tableBytes) -
                       int64_t(directUses) * siteDelta;
        }
      }
      for (Name secArr : group.secondaryArrays) {
        auto* anf = module->getGlobal(secArr)->init->cast<ArrayNewFixed>();
        Index M = anf->values.size();
        auto siteIt = allArrayGetSites.find(secArr);
        Index numSites = siteIt != allArrayGetSites.end() ? siteIt->second.size() : 0;
        secArraysBonus += int64_t(9 + 2 * M) - int64_t(2 * numSites) - 18;
      }
      absSaving += secArraysBonus;
      if (absMembers.size() == group.absorbedCount &&
          (always || absSaving > 0)) {
        for (auto* m : absMembers) {
          chosen.push_back(m);
        }
        saving += absSaving;
      } else {
        group.primaryArrays.clear();
        group.secondaryArrays.clear();
        group.absorbedCount = 0;
        secArraysBonus = 0;
      }
    }
    for (auto* member : sorted) {
      if (!preservedThunk && member->absorbedIndex != Index(-1)) {
        continue;
      }
      Index tableBytes = getMemberTableBytes(member);
      Index index = chosen.size() % MaxChunkSize;
      int64_t siteDelta = 0;
      if (member->absorbedIndex == Index(-1)) {
        Index getBytes =
          1 + getLEBSize(U32LEB(globalIndexes[member->global->name]));
        siteDelta =
          int64_t(callBytes + 1 + getSLEBSize(index)) - int64_t(getBytes);
      }
      int64_t memberSaving = int64_t(member->initBytes) +
                             int64_t(member->bonus) - int64_t(tableBytes) -
                             int64_t(member->uses) * siteDelta;
      if (always || memberSaving > 0) {
        chosen.push_back(member);
        saving += memberSaving;
      }
    }
    if (preservedThunk && !group.primaryArrays.empty()) {
      Index chosenAbsorbed = 0;
      for (auto* c : chosen) {
        if (c->absorbedIndex != Index(-1)) {
          chosenAbsorbed++;
        }
      }
      if (chosenAbsorbed != group.absorbedCount) {
        chosen.erase(std::remove_if(chosen.begin(),
                                    chosen.end(),
                                    [](Candidate* c) {
                                      return c->absorbedIndex != Index(-1);
                                    }),
                     chosen.end());
        group.primaryArrays.clear();
        group.absorbedCount = 0;
      }
    }
    if (chosen.empty() || (!always && !preservedThunk && chosen.size() < 2)) {
      return;
    }
    // Recompute the columns and exact savings over the chosen members only.
    std::vector<Candidate> chosenMembers;
    for (auto* member : chosen) {
      chosenMembers.push_back(*member);
    }
    varying = getVaryingColumns(chosenMembers, numLeaves);
    auto finalCosts = computeColumnCosts(
      group, chosen, varying, !group.secondaryArrays.empty());
    if (!preservedThunk) {
      saving = secArraysBonus;
      for (Index idxInChosen = 0; idxInChosen < chosen.size(); idxInChosen++) {
        auto* member = chosen[idxInChosen];
        Index tableBytes = finalCosts.memberTableBytes[member];
        Index getBytes =
          1 + getLEBSize(U32LEB(globalIndexes[member->global->name]));
        if (member->absorbedIndex != Index(-1) &&
            (!group.primaryArrays.empty() || !group.secondaryArrays.empty())) {
          Index index = member->absorbedIndex;
          Index directUses = member->uses - 1;
          int64_t siteDelta =
            int64_t(callBytes + 1 + getSLEBSize(index)) - int64_t(getBytes);
          int64_t elemBonus = directUses > 0 ? int64_t(getBytes) : 0;
          saving += int64_t(member->initBytes) + int64_t(member->bonus) +
                    elemBonus - int64_t(tableBytes) -
                    int64_t(directUses) * siteDelta;
        } else {
          Index index = idxInChosen % MaxChunkSize;
          int64_t siteDelta =
            int64_t(callBytes + 1 + getSLEBSize(index)) - int64_t(getBytes);
          saving += int64_t(member->initBytes) + int64_t(member->bonus) -
                    int64_t(tableBytes) - int64_t(member->uses) * siteDelta;
        }
      }
    }

    bool allOnceSingleUse =
      !preservedThunk &&
      std::all_of(chosen.begin(), chosen.end(), [](Candidate* c) {
        return c->uses == 1 && c->onceUses == 1;
      });
    // Fixed costs per chunk: the factory function (code, function section
    // entry, type), the cache global and its type (unless cacheless), the
    // fixed template instructions inside the factory, and per table column the
    // global, its type, and the read in the factory.
    Index varyingInitBytes = 0;
    for (auto i : varying) {
      varyingInitBytes += getOperandBytes(chosen[0]->leaves[i]);
    }
    Index fixedTemplateBytes =
      chosen[0]->initBytes > 9 + varyingInitBytes
        ? chosen[0]->initBytes - 9 - varyingInitBytes
        : 0;
    Index numChunks = (chosen.size() + MaxChunkSize - 1) / MaxChunkSize;
    Index baseFixed = allOnceSingleUse ? 16 : 53;
    Index overhead =
      numChunks *
      (baseFixed + fixedTemplateBytes + finalCosts.chunkColumnOverhead);
    if (!always && saving <= int64_t(overhead)) {
      return;
    }

    // Assign the indexes: absorbed array elements first in array order (0..K-1)
    // followed by direct-get members in first-use order.
    std::sort(chosen.begin(), chosen.end(), [](Candidate* a, Candidate* b) {
      bool aAbs = a->absorbedIndex != Index(-1);
      bool bAbs = b->absorbedIndex != Index(-1);
      if (aAbs != bAbs) {
        return aAbs;
      }
      if (aAbs) {
        return a->absorbedIndex < b->absorbedIndex;
      }
      return a->firstUse < b->firstUse;
    });
    for (Index chunk = 0; chunk < numChunks; chunk++) {
      Index begin = chunk * MaxChunkSize;
      Index end = std::min(Index(chosen.size()), begin + MaxChunkSize);
      std::vector<Candidate*> chunkMembers(chosen.begin() + begin,
                                           chosen.begin() + end);
      bool cacheless =
        !preservedThunk &&
        std::all_of(chunkMembers.begin(), chunkMembers.end(), [](Candidate* c) {
          return c->uses == 1 && c->onceUses == 1;
        });
      auto factory =
        makeFactory(group, chunkMembers, varying, cacheless, chunk == 0);
      if (chunk == 0 &&
          (!group.primaryArrays.empty() || !group.secondaryArrays.empty())) {
        group.absorbedFactory = factory;
      }
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
                   bool cacheless,
                   bool isFirstChunk) {
    Builder builder(*module);
    auto type = group.type;
    auto resultType = group.globalType;
    auto& leafFields = group.leafFields;
    bool preservedThunk = isPreservedThunkGroup(group);
    std::string typeName;
    if (auto iter = module->typeNames.find(type);
        iter != module->typeNames.end()) {
      typeName = iter->second.name.toString();
    } else {
      typeName = std::to_string(numFactories);
    }
    numFactories++;
    std::string baseName = "defer$" + typeName;

    // The cache (omitted when every member is single-use inside a once-guard;
    // scalar mutable global when N == 1).
    auto cacheElementType = resultType.with(Nullable);
    Name cacheName;
    Type cacheGlobalType = Type::none;
    bool scalarCache = !cacheless && members.size() == 1;
    if (!cacheless) {
      cacheName = Names::getValidGlobalName(*module, baseName + "$cache");
      if (scalarCache) {
        cacheGlobalType = cacheElementType;
        module->addGlobal(builder.makeGlobal(
          cacheName,
          cacheGlobalType,
          builder.makeRefNull(resultType.getHeapType()),
          Builder::Mutable));
      } else {
        HeapType cacheType(Array(Field(cacheElementType, Mutable)));
        cacheGlobalType = Type(cacheType, NonNullable);
        module->addGlobal(builder.makeGlobal(
          cacheName,
          cacheGlobalType,
          builder.makeArrayNew(cacheType,
                               builder.makeConst(int32_t(members.size()))),
          Builder::Immutable));
      }
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
      Index elemOffset = 0;
      Expression* selectExpr = nullptr;
    };
    struct DataColumnInit {
      Name global;
      HeapType arrType;
      Index offset;
      Index count;
    };
    struct PendingDataCol {
      Index leafIndex;
      Index width;
      HeapType arrType;
      std::vector<char> bytes;
    };
    std::unordered_map<Index, Column> columns;
    std::unordered_map<Index, Index> sameAsCol;
    std::vector<DataColumnInit> dataInits;
    std::vector<PendingDataCol> pendingDataCols;
    std::vector<Index> prevI32Cols;
    std::vector<Index> uniqueVarying;
    Index N = members.size();
    const Index indexLocal = 0;
    bool canUseData = module->features.hasBulkMemory();
    bool hasSecArrays = isFirstChunk && !group.secondaryArrays.empty();
    bool willHaveW1Col = hasSecArrays;
    if (canUseData) {
      std::vector<Index> scanUnique;
      std::vector<Index> scanPrevI32;
      Index combinedPalGainAt8 = 0;
      Index palCountAt8 = 0;
      for (auto i : varying) {
        auto& field = leafFields[i];
        bool isDup = false;
        if (!preservedThunk) {
          for (Index prev : scanUnique) {
            if (leafFields[prev].type == field.type &&
                leafFields[prev].isPacked() == field.isPacked() &&
                (!field.isPacked() ||
                 leafFields[prev].packedType == field.packedType)) {
              bool allEq = true;
              for (Index k = 0; k < N; k++) {
                if (!ExpressionAnalyzer::equal(members[k]->leaves[prev],
                                               members[k]->leaves[i])) {
                  allEq = false;
                  break;
                }
              }
              if (allEq) {
                isDup = true;
                break;
              }
            }
          }
        }
        if (isDup) {
          continue;
        }
        scanUnique.push_back(i);
        if (N >= 7 && field.type == Type::i32 &&
            std::all_of(members.begin(), members.end(), [&](Candidate* m) {
              return m->leaves[i]->is<Const>();
            })) {
          auto evalW = [&](auto getVal) -> Index {
            int64_t mn = std::numeric_limits<int64_t>::max();
            int64_t mx = std::numeric_limits<int64_t>::min();
            for (Index k = 0; k < N; k++) {
              int64_t v = getVal(k);
              mn = std::min(mn, v);
              mx = std::max(mx, v);
            }
            Index w = 4;
            bool s = false;
            HeapType ht;
            classifyI32Range(mn, mx, w, s, ht);
            return w;
          };
          Index w = evalW([&](Index k) {
            int32_t raw = members[k]->leaves[i]->cast<Const>()->value.geti32();
            return getPackedOrI32Value(field, raw);
          });
          if (!field.isPacked()) {
            w = std::min(w, evalW([&](Index k) {
                           int32_t raw = members[k]
                                           ->leaves[i]
                                           ->cast<Const>()
                                           ->value.geti32();
                           return int64_t(int32_t(uint32_t(raw) - uint32_t(k)));
                         }));
            for (Index prev : scanPrevI32) {
              w = std::min(
                w, evalW([&](Index k) {
                  int32_t raw =
                    members[k]->leaves[i]->cast<Const>()->value.geti32();
                  int32_t baseRaw =
                    members[k]->leaves[prev]->cast<Const>()->value.geti32();
                  return int64_t(int32_t(uint32_t(raw) - uint32_t(baseRaw)));
                }));
            }
            scanPrevI32.push_back(i);
          }
          if (w == 1) {
            willHaveW1Col = true;
            break;
          }
          continue;
        }
        if (isPaletteCandidate(field)) {
          std::vector<Expression*> distinct;
          std::vector<std::vector<Index>> distinctIndices;
          bool ok = true;
          Index elemBytesSum = 0;
          for (Index k = 0; k < N; k++) {
            auto* leaf = members[k]->leaves[i];
            elemBytesSum += getOperandBytes(leaf);
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
              distinctIndices.push_back({k});
            } else {
              distinctIndices[idx].push_back(k);
            }
          }
          if (!ok) {
            continue;
          }
          Index D = distinct.size();
          Index effD = getEffectivePaletteCount(distinct);
          Index distinctBytes = 0;
          for (auto* expr : distinct) {
            distinctBytes += getOperandBytes(expr);
          }
          if (D > effD) {
            Index savedPal = 3 * (D - effD);
            distinctBytes =
              distinctBytes > savedPal ? distinctBytes - savedPal : 0;
          }
          Index uncompressedTableCost =
            9 + getLEBSize(U32LEB(N)) + elemBytesSum + 6;
          Index w1PaletteCostAt8 =
            9 + getLEBSize(U32LEB(effD)) + distinctBytes + N + 8;
          Index selectCodeCost = Index(-1);
          if (D >= 2 && D <= 5 && D < N && D == effD &&
              tryPlanSelectChain(distinct,
                                 distinctIndices,
                                 N,
                                 field.type,
                                 indexLocal,
                                 nullptr,
                                 &selectCodeCost) &&
              selectCodeCost <=
                std::min(uncompressedTableCost, w1PaletteCostAt8)) {
            continue;
          }
          if (!preservedThunk) {
            if (9 + 3 * effD + N + 25 < 9 + 3 * N) {
              willHaveW1Col = true;
              break;
            }
            if (9 + 3 * effD + N + 8 < 9 + 3 * N) {
              palCountAt8++;
              combinedPalGainAt8 += (9 + 3 * N) - (9 + 3 * effD + N + 8);
              if (palCountAt8 >= 2 && combinedPalGainAt8 > 17) {
                willHaveW1Col = true;
                break;
              }
            }
          }
        }
      }
    }
    auto findOrAddArrayFixedGlobal = [&](const std::string& hint,
                                         Type globalType,
                                         HeapType arrType,
                                         const std::vector<Expression*>& vals) {
      auto* newInit = builder.makeArrayNewFixed(arrType, vals);
      if (!preservedThunk) {
        for (auto& g : module->globals) {
          if (!g->imported() && !g->mutable_ && g->type == globalType &&
              ExpressionAnalyzer::equal(g->init, newInit)) {
            return g->name;
          }
        }
      }
      auto gName = Names::getValidGlobalName(*module, hint);
      module->addGlobal(
        builder.makeGlobal(gName, globalType, newInit, Builder::Immutable));
      return gName;
    };
    for (auto i : varying) {
      auto& field = leafFields[i];
      Index dupOf = Index(-1);
      if (!preservedThunk) {
        for (Index prev : uniqueVarying) {
          if (leafFields[prev].type == field.type &&
              leafFields[prev].isPacked() == field.isPacked() &&
              (!field.isPacked() ||
               leafFields[prev].packedType == field.packedType)) {
            bool allEq = true;
            for (Index k = 0; k < N; k++) {
              if (!ExpressionAnalyzer::equal(members[k]->leaves[prev],
                                             members[k]->leaves[i])) {
                allEq = false;
                break;
              }
            }
            if (allEq) {
              dupOf = prev;
              break;
            }
          }
        }
      }
      if (dupOf != Index(-1)) {
        sameAsCol[i] = dupOf;
        continue;
      }
      uniqueVarying.push_back(i);
      HeapType tableType(Array(Field(field.type, Immutable)));
      if (field.isPacked()) {
        tableType = HeapType(Array(Field(field.packedType, Immutable)));
      }
      if (canUseData && N >= 7 && field.type == Type::i32 &&
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
        std::vector<char> colBytes;
        colBytes.reserve(N * width);
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
          appendLE(colBytes, raw, width);
        }
        if (width == 1) {
          willHaveW1Col = true;
        }
        pendingDataCols.push_back({i, width, arrType, std::move(colBytes)});
        columns[i] = {Name(),
                      Type(arrType, Nullable),
                      signed_,
                      addIndex,
                      baseColumn,
                      Name(),
                      Type::none,
                      0};
        continue;
      }
      if (isPaletteCandidate(field)) {
        std::vector<Expression*> distinct;
        std::vector<std::vector<Index>> distinctIndices;
        std::vector<uint8_t> indices;
        indices.reserve(N);
        bool ok = true;
        Index elemBytesSum = 0;
        for (Index k = 0; k < N; k++) {
          auto* leaf = members[k]->leaves[i];
          elemBytesSum += getOperandBytes(leaf);
          if (!ok) {
            continue;
          }
          Index idx = 0;
          while (idx < distinct.size() && !sameRefLeaf(distinct[idx], leaf)) {
            idx++;
          }
          if (idx == distinct.size()) {
            if (distinct.size() == 256) {
              ok = false;
              continue;
            }
            distinct.push_back(leaf);
            distinctIndices.push_back({k});
          } else {
            distinctIndices[idx].push_back(k);
          }
          indices.push_back(uint8_t(idx));
        }
        Index D = distinct.size();
        Index effD = ok ? getEffectivePaletteCount(distinct) : D;
        Index distinctBytes = 0;
        for (auto* expr : distinct) {
          distinctBytes += getOperandBytes(expr);
        }
        if (D > effD) {
          Index savedPal = 3 * (D - effD);
          distinctBytes =
            distinctBytes > savedPal ? distinctBytes - savedPal : 0;
        }
        Index uncompressedTableCost =
          9 + getLEBSize(U32LEB(N)) + elemBytesSum + 6;
        Index w1Extra = willHaveW1Col ? 8 : 25;
        Index w1PaletteCost =
          (canUseData && ok)
            ? (9 + getLEBSize(U32LEB(effD)) + distinctBytes + N + w1Extra)
            : Index(-1);
        Expression* selectExpr = nullptr;
        Index selectCodeCost = Index(-1);
        if (ok && D >= 2 && D <= 5 && D < N && D == effD &&
            tryPlanSelectChain(distinct,
                               distinctIndices,
                               N,
                               field.type,
                               indexLocal,
                               &selectExpr,
                               &selectCodeCost) &&
            selectCodeCost <= std::min(uncompressedTableCost, w1PaletteCost)) {
          columns[i] = {Name(),
                        Type::none,
                        false,
                        false,
                        Index(-1),
                        Name(),
                        Type::none,
                        0,
                        selectExpr};
          continue;
        }
        if (canUseData && ok && 9 + 3 * effD + N + w1Extra < 9 + 3 * N) {
          willHaveW1Col = true;
          std::vector<Expression*> palValues;
          palValues.reserve(D);
          for (auto* expr : distinct) {
            palValues.push_back(ExpressionManipulator::copy(expr, *module));
          }
          auto palGlobalType = Type(tableType, NonNullable);
          auto palName = findOrAddArrayFixedGlobal(
            baseName + "$palette$" + std::to_string(i),
            palGlobalType,
            tableType,
            palValues);
          std::vector<char> colBytes;
          colBytes.reserve(N);
          for (uint8_t idx : indices) {
            colBytes.push_back(char(idx));
          }
          auto idxArrType = HeapTypes::getMutI8Array();
          pendingDataCols.push_back({i, 1, idxArrType, std::move(colBytes)});
          columns[i] = {Name(),
                        Type(idxArrType, Nullable),
                        false,
                        false,
                        Index(-1),
                        palName,
                        palGlobalType,
                        0};
          continue;
        }
      }
      std::vector<Expression*> values;
      for (auto* member : members) {
        values.push_back(
          ExpressionManipulator::copy(member->leaves[i], *module));
      }
      auto tableGlobalType = Type(tableType, NonNullable);
      auto tableName =
        findOrAddArrayFixedGlobal(baseName + "$table$" + std::to_string(i),
                                  tableGlobalType,
                                  tableType,
                                  values);
      columns[i] = {tableName,
                    tableGlobalType,
                    false,
                    false,
                    Index(-1),
                    Name(),
                    Type::none,
                    0};
    }

    // Group pending data columns by element width (1, 2, 4) so same-width
    // columns share a single mutable array global and a single array.new_data.
    Name secTableGlobal;
    Type secTableGlobalType = Type::none;
    Index secSubOffset = 0;
    for (Index w : {Index(1), Index(2), Index(4)}) {
      std::vector<PendingDataCol*> groupCols;
      for (auto& pdc : pendingDataCols) {
        if (pdc.width == w) {
          groupCols.push_back(&pdc);
        }
      }
      bool packSecHere = (w == 1 && hasSecArrays);
      if (groupCols.empty() && !packSecHere) {
        continue;
      }
      HeapType arrType =
        !groupCols.empty() ? groupCols[0]->arrType : HeapTypes::getMutI8Array();
      std::string suffix =
        (groupCols.size() == 1 && !packSecHere)
          ? std::to_string(groupCols[0]->leafIndex)
          : ("w" + std::to_string(w));
      auto tableName =
        Names::getValidGlobalName(*module, baseName + "$table$" + suffix);
      auto tableGlobalType = Type(arrType, Nullable);
      module->addGlobal(builder.makeGlobal(tableName,
                                           tableGlobalType,
                                           builder.makeRefNull(arrType),
                                           Builder::Mutable));
      Index startOffset = segmentData.size();
      for (Index c = 0; c < groupCols.size(); c++) {
        auto* pdc = groupCols[c];
        auto& col = columns.at(pdc->leafIndex);
        if (col.paletteGlobal.is()) {
          paletteSlices[col.paletteGlobal].push_back(
            {Index(segmentData.size()), Index(pdc->bytes.size())});
        }
        segmentData.insert(
          segmentData.end(), pdc->bytes.begin(), pdc->bytes.end());
        col.global = tableName;
        col.globalType = tableGlobalType;
        col.elemOffset = c * N;
      }
      Index totalElems = groupCols.size() * N;
      if (packSecHere) {
        secTableGlobal = tableName;
        secTableGlobalType = tableGlobalType;
        secSubOffset = groupCols.empty() ? N : 0;
        std::unordered_map<Name, Index> memberIdxMap;
        for (Index k = 0; k < N; k++) {
          memberIdxMap[members[k]->global->name] = k;
        }
        for (Name secArr : group.secondaryArrays) {
          auto* anf = module->getGlobal(secArr)->init->cast<ArrayNewFixed>();
          Index secOffset = totalElems;
          for (auto* val : anf->values) {
            Name ename = val->cast<GlobalGet>()->name;
            Index targetIdx = memberIdxMap.at(ename);
            assert(targetIdx < 256);
            segmentData.push_back(char(uint8_t(targetIdx)));
          }
          totalElems += anf->values.size();
          group.secondaryArrayInfos.push_back(
            {secArr, secOffset + secSubOffset});
        }
      }
      dataInits.push_back({tableName, arrType, startOffset, totalElems});
    }

    // The factory.
    std::vector<Type> vars;
    Index objectLocal = 1;
    if (!cacheless) {
      vars.push_back(resultType);
    }
    std::vector<Index> baseColsNeeded;
    for (auto i : varying) {
      auto it = columns.find(i);
      if (it != columns.end() && it->second.baseColumn != Index(-1)) {
        Index b = it->second.baseColumn;
        if (std::find(baseColsNeeded.begin(), baseColsNeeded.end(), b) ==
            baseColsNeeded.end()) {
          baseColsNeeded.push_back(b);
        }
      }
    }
    std::sort(baseColsNeeded.begin(), baseColsNeeded.end());
    std::unordered_map<Index, Index> baseColLocals;
    for (Index b : baseColsNeeded) {
      Index loc = 1 + vars.size();
      vars.push_back(Type::i32);
      baseColLocals[b] = loc;
    }
    std::vector<Index> dupColsNeeded;
    for (auto& [_, prev] : sameAsCol) {
      if (std::find(dupColsNeeded.begin(), dupColsNeeded.end(), prev) ==
          dupColsNeeded.end()) {
        dupColsNeeded.push_back(prev);
      }
    }
    std::sort(dupColsNeeded.begin(), dupColsNeeded.end());
    std::unordered_map<Index, Index> dupColLocals;
    for (Index prev : dupColsNeeded) {
      auto it = baseColLocals.find(prev);
      if (it != baseColLocals.end() &&
          !columns.at(prev).paletteGlobal.is() &&
          !columns.at(prev).selectExpr &&
          !columns.at(prev).addIndex &&
          columns.at(prev).baseColumn == Index(-1)) {
        dupColLocals[prev] = it->second;
      } else {
        Index loc = 1 + vars.size();
        vars.push_back(leafFields[prev].type);
        dupColLocals[prev] = loc;
      }
    }

    auto makeElemIndex = [&](Index elemOffset) -> Expression* {
      Expression* idx = builder.makeLocalGet(indexLocal, Type::i32);
      if (elemOffset == 0) {
        return idx;
      }
      return builder.makeBinary(
        AddInt32, idx, builder.makeConst(int32_t(elemOffset)));
    };

    // Rebuild the (nested) allocation from the first member, reading the
    // varying leaves from the tables.
    std::function<Expression*(Index)> readColumn = [&](Index i) -> Expression* {
      auto& col = columns.at(i);
      Expression* val = nullptr;
      if (col.selectExpr) {
        val = col.selectExpr;
      } else if (col.paletteGlobal.is()) {
        auto* idx = builder.makeArrayGet(
          builder.makeGlobalGet(col.global, col.globalType),
          makeElemIndex(col.elemOffset),
          MemoryOrder::Unordered,
          Type::i32,
          false);
        val = builder.makeArrayGet(
          builder.makeGlobalGet(col.paletteGlobal, col.paletteGlobalType),
          idx,
          MemoryOrder::Unordered,
          leafFields[i].type,
          false);
      } else {
        val = builder.makeArrayGet(
          builder.makeGlobalGet(col.global, col.globalType),
          makeElemIndex(col.elemOffset),
          MemoryOrder::Unordered,
          leafFields[i].type,
          col.signed_);
        if (col.addIndex) {
          val = builder.makeBinary(
            AddInt32, val, builder.makeLocalGet(indexLocal, Type::i32));
        } else if (col.baseColumn != Index(-1)) {
          val = builder.makeBinary(
            AddInt32,
            val,
            builder.makeLocalGet(baseColLocals.at(col.baseColumn), Type::i32));
        }
        if (auto it = baseColLocals.find(i); it != baseColLocals.end()) {
          val = builder.makeLocalTee(it->second, val, Type::i32);
        }
      }
      if (auto it = dupColLocals.find(i); it != dupColLocals.end()) {
        auto baseIt = baseColLocals.find(i);
        if (baseIt == baseColLocals.end() || baseIt->second != it->second) {
          val = builder.makeLocalTee(it->second, val, leafFields[i].type);
        }
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
        if (auto it = sameAsCol.find(i); it != sameAsCol.end()) {
          operands.push_back(
            builder.makeLocalGet(dupColLocals.at(it->second),
                                 leafFields[i].type));
        } else if (columns.count(i)) {
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

    Expression* secRemapCheck = nullptr;
    if (secTableGlobal.is()) {
      Expression* tableIdx = builder.makeLocalGet(indexLocal, Type::i32);
      if (secSubOffset != 0) {
        tableIdx = builder.makeBinary(
          SubInt32, tableIdx, builder.makeConst(int32_t(secSubOffset)));
      }
      auto* remapped = builder.makeArrayGet(
        builder.makeGlobalGet(secTableGlobal, secTableGlobalType),
        tableIdx,
        MemoryOrder::Unordered,
        Type::i32,
        false);
      secRemapCheck = builder.makeIf(
        builder.makeBinary(GeUInt32,
                           builder.makeLocalGet(indexLocal, Type::i32),
                           builder.makeConst(int32_t(N))),
        builder.makeLocalSet(indexLocal, remapped));
    }

    auto factoryName = Names::getValidFunctionName(*module, baseName);
    if (cacheless) {
      std::vector<Expression*> stmts;
      if (dataInitCheck) {
        stmts.push_back(dataInitCheck);
      }
      if (secRemapCheck) {
        stmts.push_back(secRemapCheck);
      }
      stmts.push_back(structNew);
      Expression* body =
        stmts.size() == 1 ? stmts[0] : builder.makeBlock(stmts, resultType);
      module->addFunction(builder.makeFunction(
        factoryName, Signature(Type::i32, resultType), std::move(vars), body));
      return factoryName;
    }

    Name label("cached");
    Expression* cacheGet =
      scalarCache
        ? static_cast<Expression*>(
            builder.makeGlobalGet(cacheName, cacheGlobalType))
        : static_cast<Expression*>(builder.makeArrayGet(
            builder.makeGlobalGet(cacheName, cacheGlobalType),
            builder.makeLocalGet(indexLocal, Type::i32),
            MemoryOrder::Unordered,
            cacheElementType));
    auto* brOn = builder.makeBrOn(BrOnNonNull, label, cacheGet);
    auto* tee = builder.makeLocalTee(objectLocal, structNew, resultType);
    Expression* cacheSet =
      scalarCache
        ? static_cast<Expression*>(builder.makeGlobalSet(cacheName, tee))
        : static_cast<Expression*>(builder.makeArraySet(
            builder.makeGlobalGet(cacheName, cacheGlobalType),
            builder.makeLocalGet(indexLocal, Type::i32),
            tee,
            MemoryOrder::Unordered));
    std::vector<Expression*> blockList = {brOn};
    if (dataInitCheck && !secRemapCheck) {
      blockList.push_back(dataInitCheck);
    }
    blockList.push_back(cacheSet);
    blockList.push_back(builder.makeLocalGet(objectLocal, resultType));
    Expression* body = builder.makeBlock(label, blockList, resultType);
    if (secRemapCheck) {
      std::vector<Expression*> outerList;
      if (dataInitCheck) {
        outerList.push_back(dataInitCheck);
      }
      outerList.push_back(secRemapCheck);
      outerList.push_back(body);
      body = builder.makeBlock(outerList, resultType);
    }
    module->addFunction(builder.makeFunction(
      factoryName, Signature(Type::i32, resultType), std::move(vars), body));
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

  static bool hasImmutableSigRefFunc(StructNew* sn) {
    if (!sn || !sn->type.isRef() || !sn->type.getHeapType().isStruct()) {
      return false;
    }
    auto& fields = sn->type.getHeapType().getStruct().fields;
    if (sn->operands.size() != fields.size()) {
      return false;
    }
    for (Index i = 0; i < fields.size(); i++) {
      auto& field = fields[i];
      if (field.mutable_ == Immutable && field.type.isRef() &&
          field.type.getHeapType().isSignature() &&
          sn->operands[i]->is<RefFunc>()) {
        return true;
      }
    }
    return false;
  }

  bool isPreservedThunkGroup(const Group& g) const {
    if (!preserveThunkTypes) {
      return false;
    }
    if (hasImmutableSigField(Type(g.type, NonNullable))) {
      return true;
    }
    for (auto& f : g.leafFields) {
      if (f.mutable_ == Immutable && f.type.isRef() &&
          f.type.getHeapType().isSignature()) {
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

    std::unordered_map<Name, std::vector<std::pair<Function*, If*>>>
      lazyGuardIfs;
    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      for (auto* iff : funcAnalysis.map[func.get()].ifs) {
        Name flag;
        if (!iff->ifFalse && isFlagEqzCond(iff->condition, flag)) {
          flagGuardIfs[flag].push_back({func.get(), iff});
        } else if (!iff->ifFalse) {
          if (auto* rin = iff->condition->dynCast<RefIsNull>()) {
            if (auto* gg = rin->value->dynCast<GlobalGet>()) {
              lazyGuardIfs[gg->name].push_back({func.get(), iff});
            }
          }
        }
      }
    }

    std::unordered_set<Name> onceHelperFuncs;
    std::unordered_map<Name, Function*> flagHelperFunc;
    std::unordered_map<Name, Function*> lazyHelperFunc;
    for (auto& func : module->functions) {
      if (func->imported() || nonCallFuncUses.count(func->name)) {
        continue;
      }
      auto& callers = funcCallers[func->name];
      if (callers.empty()) {
        continue;
      }
      auto* firstSet = getFirstStmt(func->body)->dynCast<GlobalSet>();
      if (firstSet && onceFlags.count(firstSet->name) &&
          funcSets[firstSet->name].size() == 1) {
        Name flag = firstSet->name;
        auto& guards = flagGuardIfs[flag];
        if (guards.size() == callers.size() &&
            guards.size() == funcGets[flag].size()) {
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
            continue;
          }
        }
      }
      // Also recognize helper functions whose every caller is guarded by
      // (if (ref.is_null (global.get $lazy)) (then (call $func))) where $lazy
      // is a 1-set nullable reference global set inside $func.
      for (auto* gs : FindAll<GlobalSet>(func->body).list) {
        Name lazy = gs->name;
        auto* lg = module->getGlobalOrNull(lazy);
        if (!lg || lg->imported() || !lg->mutable_ || !lg->type.isRef() ||
            !lg->type.isNullable() || !lg->init->is<RefNull>() ||
            nonCodeGlobalUses.count(lazy) || initGlobalUses[lazy] != 0 ||
            funcSets[lazy].size() != 1 || gs->value->is<RefNull>()) {
          continue;
        }
        auto& guards = lazyGuardIfs[lazy];
        if (guards.size() != callers.size()) {
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
          lazyHelperFunc[lazy] = func.get();
          break;
        }
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
          if (auto* call = getFirstStmt(iff->ifTrue)->dynCast<Call>()) {
            auto it = lazyHelperFunc.find(lazy);
            if (it != lazyHelperFunc.end() &&
                it->second->name == call->target) {
              return true;
            }
          }
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

    auto isCachedFactoryBlock = [&](Block* b,
                                    bool& isScalarOut,
                                    Index& idxLocalOut) -> bool {
      if (!b->name.is() || b->list.size() < 2) {
        return false;
      }
      auto* brOn = b->list[0]->dynCast<BrOn>();
      if (!brOn || brOn->op != BrOnNonNull || brOn->name != b->name) {
        return false;
      }
      if (auto* gg = brOn->ref->dynCast<GlobalGet>()) {
        Name cname = gg->name;
        auto* cg = module->getGlobalOrNull(cname);
        if (cg && !cg->imported() && cg->mutable_ && cg->type.isRef() &&
            cg->type.isNullable() && cg->init->is<RefNull>() &&
            !nonCodeGlobalUses.count(cname) && initGlobalUses[cname] == 0 &&
            funcGets[cname].size() == 1 && funcSets[cname].size() == 1) {
          Expression* targetSet = *funcSets[cname][0].second;
          for (Index i = 1; i < b->list.size(); i++) {
            for (auto* gs : FindAll<GlobalSet>(b->list[i]).list) {
              if (gs == targetSet && !gs->value->is<RefNull>()) {
                isScalarOut = true;
                idxLocalOut = Index(-1);
                return true;
              }
            }
          }
        }
        return false;
      }
      if (auto* ag = brOn->ref->dynCast<ArrayGet>()) {
        auto* gg = ag->ref->dynCast<GlobalGet>();
        auto* lg = ag->index->dynCast<LocalGet>();
        if (!gg || !lg) {
          return false;
        }
        Name cname = gg->name;
        auto* cg = module->getGlobalOrNull(cname);
        if (cg && !cg->imported() && !cg->mutable_ && cg->init->is<ArrayNew>() &&
            !nonCodeGlobalUses.count(cname) && initGlobalUses[cname] == 0 &&
            funcGets[cname].size() == 2 && funcSets[cname].empty()) {
          for (Index i = 1; i < b->list.size(); i++) {
            for (auto* as : FindAll<ArraySet>(b->list[i]).list) {
              auto* sgg = as->ref->dynCast<GlobalGet>();
              auto* slg = as->index->dynCast<LocalGet>();
              if (sgg && sgg->name == cname && slg &&
                  slg->index == lg->index && !as->value->is<RefNull>()) {
                isScalarOut = false;
                idxLocalOut = lg->index;
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
      std::function<void(Expression*, bool, Index, bool)> walkOnce =
        [&](Expression* curr,
            bool inOnce,
            Index cachedIdxLocal,
            bool inLoopOrTry) {
          if (!curr) {
            return;
          }
          if (curr->is<Loop>() || curr->is<Try>() || curr->is<TryTable>()) {
            inLoopOrTry = true;
          }
          if (!preserveThunkTypes) {
            if (auto* b = curr->dynCast<Block>()) {
              bool isScalar = false;
              Index idxLoc = Index(-1);
              if (isCachedFactoryBlock(b, isScalar, idxLoc)) {
                walkOnce(b->list[0], inOnce, cachedIdxLocal, inLoopOrTry);
                for (Index i = 1; i < b->list.size(); i++) {
                  walkOnce(b->list[i],
                           inOnce || isScalar,
                           isScalar ? cachedIdxLocal : idxLoc,
                           inLoopOrTry);
                }
                return;
              }
            }
          }
          if (auto* iff = curr->dynCast<If>()) {
            walkOnce(iff->condition, inOnce, cachedIdxLocal, inLoopOrTry);
            walkOnce(iff->ifTrue,
                     inOnce || isOnceIf(iff),
                     cachedIdxLocal,
                     inLoopOrTry);
            if (iff->ifFalse) {
              walkOnce(iff->ifFalse, inOnce, cachedIdxLocal, inLoopOrTry);
            }
            return;
          }
          if (auto* ag = curr->dynCast<ArrayGet>()) {
            if (!inLoopOrTry && cachedIdxLocal != Index(-1) &&
                ag->order == MemoryOrder::Unordered) {
              auto* gg = ag->ref->dynCast<GlobalGet>();
              auto* lg = ag->index->dynCast<LocalGet>();
              if (gg && lg && lg->index == cachedIdxLocal) {
                onceGets.insert(gg);
              }
            }
          }
          if (auto* gg = curr->dynCast<GlobalGet>()) {
            if (inOnce && !inLoopOrTry) {
              onceGets.insert(gg);
            }
            return;
          }
          for (auto* ch : ChildIterator(curr)) {
            walkOnce(ch, inOnce, cachedIdxLocal, inLoopOrTry);
          }
        };
      walkOnce(
        func->body, onceHelperFuncs.count(func->name) > 0, Index(-1), false);
    }
  }

  void nullifySoleUsePaletteElements() {
    std::unordered_set<Name> nonCodeUses;
    std::unordered_map<Name, Index> initGlobalUses;
    {
      struct Scanner : public PostWalker<Scanner> {
        std::unordered_set<Name>* nonCode = nullptr;
        std::unordered_map<Name, Index>* initCounts = nullptr;
        void visitGlobalGet(GlobalGet* curr) {
          if (initCounts) {
            (*initCounts)[curr->name]++;
          }
          if (nonCode) {
            nonCode->insert(curr->name);
          }
        }
      } scanner;
      scanner.initCounts = &initGlobalUses;
      for (auto& g : module->globals) {
        if (!g->imported()) {
          scanner.walk(g->init);
        }
      }
      scanner.initCounts = nullptr;
      scanner.nonCode = &nonCodeUses;
      for (auto& seg : module->elementSegments) {
        if (seg->offset) {
          scanner.walk(seg->offset);
        }
        for (auto* item : seg->data) {
          scanner.walk(item);
        }
      }
      for (auto& seg : module->dataSegments) {
        if (seg->offset) {
          scanner.walk(seg->offset);
        }
      }
      for (auto& ex : module->exports) {
        if (ex->kind == ExternalKind::Global) {
          nonCodeUses.insert(*ex->getInternalName());
        }
      }
    }

    struct CodeScanInfo {
      std::unordered_map<Name, Index> gets;
      std::unordered_map<Name, Index> refIsNullGets;
      std::unordered_map<Name, std::vector<GlobalSet*>> globalSets;
      std::unordered_map<Name, std::vector<Expression**>> arrayGets;
      std::unordered_map<Name, std::vector<Name>> palCallTargets;
      std::unordered_map<Name, std::vector<Call*>> palCalls;
    };
    ModuleUtils::ParallelFunctionAnalysis<CodeScanInfo> codeScan(
      *module, [&](Function* func, CodeScanInfo& info) {
        if (func->imported()) {
          return;
        }
        struct Walker : public PostWalker<Walker> {
          CodeScanInfo& info;
          Walker(CodeScanInfo& info) : info(info) {}
          void visitGlobalGet(GlobalGet* curr) { info.gets[curr->name]++; }
          void visitGlobalSet(GlobalSet* curr) {
            info.globalSets[curr->name].push_back(curr);
          }
          void visitRefIsNull(RefIsNull* curr) {
            if (auto* gg = curr->value->dynCast<GlobalGet>()) {
              info.refIsNullGets[gg->name]++;
            }
          }
          void visitArrayGet(ArrayGet* curr) {
            if (curr->order != MemoryOrder::Unordered) {
              return;
            }
            if (auto* gg = curr->ref->dynCast<GlobalGet>()) {
              info.arrayGets[gg->name].push_back(getCurrentPointer());
            }
          }
          void visitCall(Call* curr) {
            if (curr->operands.size() == 2) {
              if (auto* gg = curr->operands[0]->dynCast<GlobalGet>()) {
                info.palCallTargets[gg->name].push_back(curr->target);
                info.palCalls[gg->name].push_back(curr);
              }
            }
          }
        };
        Walker(info).walk(func->body);
      });

    std::unordered_map<Name, Index> codeGlobalGets;
    std::unordered_map<Name, Index> allRefIsNullGets;
    std::unordered_map<Name, std::vector<std::pair<Function*, GlobalSet*>>>
      allGlobalSets;
    std::unordered_map<Name, std::vector<std::pair<Function*, Expression**>>>
      arrayGetSites;
    std::unordered_map<Name, std::vector<Name>> allPalCallTargets;
    std::unordered_map<Name, std::vector<std::pair<Function*, Call*>>>
      allPalCalls;
    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      auto& info = codeScan.map[func.get()];
      for (auto& [name, count] : info.gets) {
        codeGlobalGets[name] += count;
      }
      for (auto& [name, count] : info.refIsNullGets) {
        allRefIsNullGets[name] += count;
      }
      for (auto& [name, sets] : info.globalSets) {
        for (auto* gs : sets) {
          allGlobalSets[name].push_back({func.get(), gs});
        }
      }
      for (auto& [name, sites] : info.arrayGets) {
        for (auto** site : sites) {
          arrayGetSites[name].push_back({func.get(), site});
        }
      }
      for (auto& [name, targets] : info.palCallTargets) {
        auto& dst = allPalCallTargets[name];
        dst.insert(dst.end(), targets.begin(), targets.end());
      }
      for (auto& [name, calls] : info.palCalls) {
        for (auto* c : calls) {
          allPalCalls[name].push_back({func.get(), c});
        }
      }
    }

    computeIndexes();

    struct SoleSlot {
      Global* arrGlobal;
      ArrayNewFixed* anf;
      Index slotIdx;
      Name elemGlobalName;
      StructNew* sn;
      Name existingHelper;
    };
    InsertOrderedMap<HeapType, std::vector<SoleSlot>> byStructType;
    std::unordered_map<HeapType, std::vector<std::pair<Name, StructNew*>>>
      existingPalHelpers;
    for (auto& g : module->globals) {
      if (g->imported() || g->mutable_ || nonCodeUses.count(g->name) ||
          initGlobalUses[g->name] > 0) {
        continue;
      }
      auto* anf = g->init->dynCast<ArrayNewFixed>();
      if (!anf || anf->values.size() < 2) {
        continue;
      }
      Index totalGets = codeGlobalGets[g->name];
      if (totalGets == 0) {
        continue;
      }
      if (!g->type.isRef() || g->type.isNullable() || g->type.isExact()) {
        continue;
      }
      HeapType arrHt = g->type.getHeapType();
      if (!arrHt.isArray() || arrHt.isShared()) {
        continue;
      }
      Field elemField = arrHt.getArray().element;
      if (!elemField.type.isRef() || elemField.type.isExact()) {
        continue;
      }
      HeapType structHt = elemField.type.getHeapType();
      if (!structHt.isStruct() || structHt.isShared()) {
        continue;
      }
      Name existingHelper;
      StructNew* existingTempl = nullptr;
      if (!elemField.type.isNullable() && elemField.mutable_ == Immutable) {
        if (totalGets != arrayGetSites[g->name].size()) {
          continue;
        }
        bool allNonNullGlobalGets = true;
        for (auto* val : anf->values) {
          if (!val->is<GlobalGet>()) {
            allNonNullGlobalGets = false;
            break;
          }
        }
        if (!allNonNullGlobalGets) {
          continue;
        }
      } else if (elemField.type.isNullable() && elemField.mutable_ == Mutable) {
        auto& targets = allPalCallTargets[g->name];
        if (targets.size() != totalGets) {
          continue;
        }
        Name commonTarget = targets[0];
        bool sameTarget = true;
        for (Name t : targets) {
          if (t != commonTarget) {
            sameTarget = false;
            break;
          }
        }
        if (!sameTarget) {
          continue;
        }
        auto* helperFn = module->getFunctionOrNull(commonTarget);
        if (!helperFn || helperFn->imported() || !helperFn->body) {
          continue;
        }
        if (!commonTarget.startsWith("defer$pal$")) {
          if (helperFn->getNumParams() != 2 || helperFn->getNumVars() != 1 ||
              !helperFn->getResults().isRef() ||
              helperFn->getResults().getHeapType() != structHt ||
              FindAll<BrOn>(helperFn->body).list.size() != 1 ||
              FindAll<ArrayGet>(helperFn->body).list.size() != 1 ||
              FindAll<ArraySet>(helperFn->body).list.size() != 1) {
            continue;
          }
        }
        FindAll<StructNew> sns(helperFn->body);
        if (sns.list.size() == 1 && sns.list[0]->type.isRef() &&
            sns.list[0]->type.getHeapType() == structHt) {
          existingTempl = sns.list[0];
        } else if (sns.list.empty()) {
          FindAll<Call> calls(helperFn->body);
          if (calls.list.size() == 1) {
            Call* c = calls.list[0];
            auto* callee = module->getFunctionOrNull(c->target);
            if (callee && !callee->imported() && callee->body) {
              Expression* cBody = callee->body;
              while (auto* b = cBody->dynCast<Block>()) {
                if (b->list.size() == 1) {
                  cBody = b->list[0];
                } else {
                  break;
                }
              }
              if (auto* cSn = cBody->dynCast<StructNew>();
                  cSn && !cSn->desc && cSn->type.isRef() &&
                  cSn->type.getHeapType() == structHt &&
                  cSn->operands.size() == structHt.getStruct().fields.size()) {
                auto* synthSn =
                  ExpressionManipulator::copy(cSn, *module)->cast<StructNew>();
                bool ok = true;
                for (auto*& op : synthSn->operands) {
                  Expression* baseOp = op;
                  while (true) {
                    if (auto* ra = baseOp->dynCast<RefAs>()) {
                      baseOp = ra->value;
                    } else if (auto* rc = baseOp->dynCast<RefCast>()) {
                      baseOp = rc->ref;
                    } else {
                      break;
                    }
                  }
                  if (auto* lg = baseOp->dynCast<LocalGet>()) {
                    if (lg->index < c->operands.size() &&
                        isValidLeaf(c->operands[lg->index])) {
                      op = ExpressionManipulator::copy(c->operands[lg->index],
                                                       *module);
                    } else {
                      ok = false;
                      break;
                    }
                  } else if (!isValidLeaf(op)) {
                    ok = false;
                    break;
                  }
                }
                if (ok) {
                  existingTempl = synthSn;
                }
              }
            }
          }
        }
        if (!existingTempl ||
            (preserveThunkTypes && hasImmutableSigRefFunc(existingTempl))) {
          continue;
        }
        bool validVals = true;
        for (auto* val : anf->values) {
          if (!val->is<GlobalGet>() && !val->is<RefNull>()) {
            validVals = false;
            break;
          }
        }
        if (!validVals) {
          continue;
        }
        existingHelper = commonTarget;
        existingPalHelpers[structHt].push_back({existingHelper, existingTempl});
      } else {
        continue;
      }
      for (Index k = 0; k < anf->values.size(); k++) {
        auto* gg = anf->values[k]->dynCast<GlobalGet>();
        if (!gg) {
          continue;
        }
        Name ename = gg->name;
        if (initGlobalUses[ename] == 0 || codeGlobalGets[ename] != 0 ||
            nonCodeUses.count(ename)) {
          continue;
        }
        auto* eg = module->getGlobalOrNull(ename);
        if (!eg || eg->imported() || eg->mutable_ ||
            eg->type != Type(structHt, NonNullable)) {
          continue;
        }
        auto* sn = eg->init->dynCast<StructNew>();
        if (!sn || sn->desc || !sn->type.isRef() ||
            sn->type.getHeapType() != structHt ||
            sn->operands.size() != structHt.getStruct().fields.size() ||
            (preserveThunkTypes && hasImmutableSigRefFunc(sn))) {
          continue;
        }
        if (existingTempl && !ExpressionAnalyzer::equal(existingTempl, sn)) {
          continue;
        }
        bool validOps = true;
        for (auto* op : sn->operands) {
          if (!isValidLeaf(op)) {
            validOps = false;
            break;
          }
        }
        if (validOps) {
          byStructType[structHt].push_back(
            {g.get(), anf, k, ename, sn, existingHelper});
        }
      }
    }

    Builder builder(*module);
    std::unordered_set<Function*> refinalizeFuncs;
    std::vector<Name> globalsToRemove;
    std::unordered_set<Name> globalsToRemoveSet;

    for (auto& [structHt, slots] : byStructType) {
      // Group slots by identical StructNew initializers for structHt, keeping
      // only elements whose initializer uses are all within that group.
      std::vector<std::vector<SoleSlot>> rawGroups;
      for (auto& slot : slots) {
        bool placed = false;
        for (auto& grp : rawGroups) {
          if (ExpressionAnalyzer::equal(grp[0].sn, slot.sn)) {
            grp.push_back(slot);
            placed = true;
            break;
          }
        }
        if (!placed) {
          rawGroups.push_back({slot});
        }
      }
      std::vector<std::vector<SoleSlot>> groups;
      for (auto& grp : rawGroups) {
        std::unordered_map<Name, Index> grpElemUses;
        for (auto& slot : grp) {
          grpElemUses[slot.elemGlobalName]++;
        }
        std::vector<SoleSlot> validGrp;
        for (auto& slot : grp) {
          if (grpElemUses[slot.elemGlobalName] ==
              initGlobalUses[slot.elemGlobalName]) {
            validGrp.push_back(slot);
          }
        }
        if (!validGrp.empty()) {
          groups.push_back(std::move(validGrp));
        }
      }
      if (groups.empty()) {
        continue;
      }
      std::stable_sort(
        groups.begin(),
        groups.end(),
        [](const std::vector<SoleSlot>& a, const std::vector<SoleSlot>& b) {
          return a.size() > b.size();
        });
      std::unordered_set<Global*> claimedArrays;
      for (auto& bestGroup : groups) {
        bool conflict = false;
        for (auto& slot : bestGroup) {
          if (claimedArrays.count(slot.arrGlobal)) {
            conflict = true;
            break;
          }
        }
        if (conflict) {
          continue;
        }
        StructNew* templ = bestGroup[0].sn;

        // Ensure no operand of templ references any global being removed.
        std::unordered_set<Name> removingSet;
        bool hasNewArray = false;
        Name reusedHelper;
        for (auto& slot : bestGroup) {
          removingSet.insert(slot.elemGlobalName);
          if (slot.existingHelper.is()) {
            reusedHelper = slot.existingHelper;
          } else {
            hasNewArray = true;
          }
        }
        if (!reusedHelper.is()) {
          auto exIt = existingPalHelpers.find(structHt);
          if (exIt != existingPalHelpers.end()) {
            for (auto& [hName, hTempl] : exIt->second) {
              if (ExpressionAnalyzer::equal(hTempl, templ)) {
                reusedHelper = hName;
                break;
              }
            }
          }
        }
        Index totalSavings =
          removingSet.size() * getGlobalEntryBytes(templ) + bestGroup.size();
        Index helperCost = reusedHelper.is() ? 0 : 45;
        if (!always && totalSavings <= helperCost) {
          continue;
        }

        bool selfRef = false;
        for (auto* op : templ->operands) {
          if (auto* gg = op->dynCast<GlobalGet>()) {
            if (removingSet.count(gg->name)) {
              selfRef = true;
              break;
            }
          }
        }
        if (selfRef) {
          continue;
        }

        Type nonNullStructType(structHt, NonNullable);
        Type nullStructType(structHt, Nullable);
        HeapType mutArrHt(Array(Field(nullStructType, Mutable)));
        Type mutArrRefType(mutArrHt, NonNullable);

        Name helperName = reusedHelper;
        if (!helperName.is() && hasNewArray) {
          std::string typeName;
          if (auto iter = module->typeNames.find(structHt);
              iter != module->typeNames.end()) {
            typeName = iter->second.name.toString();
          } else {
            typeName = std::to_string(numFactories);
          }
          numFactories++;
          helperName =
            Names::getValidFunctionName(*module, "defer$pal$" + typeName);
          Name label("cached");
          auto* cacheGet =
            builder.makeArrayGet(builder.makeLocalGet(0, mutArrRefType),
                                 builder.makeLocalGet(1, Type::i32),
                                 MemoryOrder::Unordered,
                                 nullStructType);
          auto* brOn = builder.makeBrOn(BrOnNonNull, label, cacheGet);
          auto* tee = builder.makeLocalTee(
            2, ExpressionManipulator::copy(templ, *module), nonNullStructType);
          auto* cacheSet =
            builder.makeArraySet(builder.makeLocalGet(0, mutArrRefType),
                                 builder.makeLocalGet(1, Type::i32),
                                 tee,
                                 MemoryOrder::Unordered);
          auto* body = builder.makeBlock(
            label,
            {brOn, cacheSet, builder.makeLocalGet(2, nonNullStructType)},
            nonNullStructType);
          module->addFunction(builder.makeFunction(
            helperName,
            Signature({mutArrRefType, Type::i32}, nonNullStructType),
            {nonNullStructType},
            body));
          existingPalHelpers[structHt].push_back({helperName, templ});
        }

        std::unordered_set<Global*> touchedNewArrays;
        for (auto& slot : bestGroup) {
          claimedArrays.insert(slot.arrGlobal);
          slot.anf->values[slot.slotIdx] = builder.makeRefNull(HeapTypes::none);
          if (!slot.existingHelper.is()) {
            touchedNewArrays.insert(slot.arrGlobal);
          }
          if (globalsToRemoveSet.insert(slot.elemGlobalName).second) {
            globalsToRemove.push_back(slot.elemGlobalName);
          }
        }
        for (auto* arrG : touchedNewArrays) {
          arrG->type = mutArrRefType;
          auto* anf = arrG->init->cast<ArrayNewFixed>();
          anf->type = Type(mutArrHt, NonNullable, Exact);
          anf->finalize();
          Type helperRes = module->getFunction(helperName)->getResults();
          for (auto& [fn, site] : arrayGetSites[arrG->name]) {
            auto* ag = (*site)->cast<ArrayGet>();
            ag->ref->cast<GlobalGet>()->type = mutArrRefType;
            Call* rawCall =
              builder.makeCall(helperName, {ag->ref, ag->index}, helperRes);
            allPalCalls[arrG->name].push_back({fn, rawCall});
            Expression* call = rawCall;
            if (helperRes.isNullable()) {
              call = builder.makeRefAs(RefAsNonNull, call);
            }
            *site = call;
            refinalizeFuncs.insert(fn);
          }
        }
      }
    }

    for (auto* fn : refinalizeFuncs) {
      ReFinalize().walkFunctionInModule(fn, module);
    }
    for (Name gname : globalsToRemove) {
      module->removeGlobal(gname);
    }

    auto evalConstI32 = [&](Expression* expr,
                            Function* fn,
                            uint32_t& out) -> bool {
      while (auto* lt = expr->dynCast<LocalSet>()) {
        if (!lt->isTee()) {
          return false;
        }
        expr = lt->value;
      }
      if (auto* c = expr->dynCast<Const>()) {
        if (c->type == Type::i32) {
          out = uint32_t(c->value.geti32());
          return true;
        }
        return false;
      }
      if (auto* gg = expr->dynCast<GlobalGet>()) {
        auto* g = module->getGlobalOrNull(gg->name);
        if (g && !g->imported() && !g->mutable_ && g->init) {
          if (auto* c = g->init->dynCast<Const>(); c && c->type == Type::i32) {
            out = uint32_t(c->value.geti32());
            return true;
          }
        }
        return false;
      }
      if (auto* lg = expr->dynCast<LocalGet>()) {
        if (!fn || lg->index < fn->getNumParams()) {
          return false;
        }
        Const* soleConst = nullptr;
        for (auto* ls : FindAll<LocalSet>(fn->body).list) {
          if (ls->index == lg->index) {
            auto* c = ls->value->dynCast<Const>();
            if (!c || c->type != Type::i32 || soleConst) {
              return false;
            }
            soleConst = c;
          }
        }
        if (soleConst) {
          out = uint32_t(soleConst->value.geti32());
          return true;
        }
      }
      return false;
    };

    auto evalElemOffset = [&](Expression* idxExpr,
                              Function* fn,
                              Index& elemOffsetOut) -> bool {
      if (auto* lg = idxExpr->dynCast<LocalGet>()) {
        if (lg->index == 0) {
          elemOffsetOut = 0;
          return true;
        }
        return false;
      }
      if (auto* bin = idxExpr->dynCast<Binary>()) {
        uint32_t cval = 0;
        if (bin->op == AddInt32) {
          if (auto* lg = bin->left->dynCast<LocalGet>();
              lg && lg->index == 0 && evalConstI32(bin->right, fn, cval)) {
            elemOffsetOut = Index(cval);
            return true;
          }
          if (auto* lg = bin->right->dynCast<LocalGet>();
              lg && lg->index == 0 && evalConstI32(bin->left, fn, cval)) {
            elemOffsetOut = Index(cval);
            return true;
          }
        } else if (bin->op == SubInt32) {
          if (auto* lg = bin->left->dynCast<LocalGet>();
              lg && lg->index == 0 && evalConstI32(bin->right, fn, cval)) {
            elemOffsetOut = Index(uint32_t(-int32_t(cval)));
            return true;
          }
        }
      }
      return false;
    };

    auto getFactoryRowCount = [&](Function* fn, Index& rowCountOut) -> bool {
      for (auto* as : FindAll<ArraySet>(fn->body).list) {
        auto* gg = as->ref->dynCast<GlobalGet>();
        auto* lg = as->index->dynCast<LocalGet>();
        if (!gg || !lg || lg->index != 0) {
          continue;
        }
        auto* cg = module->getGlobalOrNull(gg->name);
        if (!cg || cg->imported() || cg->mutable_ || !cg->init) {
          continue;
        }
        if (auto* an = cg->init->dynCast<ArrayNew>(); an && !an->init) {
          uint32_t n = 0;
          if (evalConstI32(an->size, nullptr, n) && n > 0) {
            rowCountOut = Index(n);
            return true;
          }
        }
      }
      return false;
    };

    // Compact any palette array.new_fixed global whose elements now contain
    // duplicate entries (such as multiple ref.null slots after nullification)
    // when all of its uses are accounted for by paletteSlices or by
    // array.new_data index tables from earlier invocations, and remap the
    // corresponding u8 index slices.
    for (auto& g : module->globals) {
      if (g->imported() || g->mutable_ || !g->init ||
          nonCodeUses.count(g->name) || initGlobalUses[g->name] > 0) {
        continue;
      }
      auto* anf = g->init->dynCast<ArrayNewFixed>();
      if (!anf || anf->values.size() < 2 || anf->values.size() > 256) {
        continue;
      }
      bool allLeaves = true;
      for (auto* val : anf->values) {
        if (!val->is<RefNull>() && !val->is<GlobalGet>() &&
            !val->is<RefFunc>()) {
          allLeaves = false;
          break;
        }
      }
      if (!allLeaves) {
        continue;
      }
      std::vector<Expression*> compacted;
      std::vector<uint8_t> remap(anf->values.size());
      for (Index oldIdx = 0; oldIdx < anf->values.size(); oldIdx++) {
        Expression* val = anf->values[oldIdx];
        Index newIdx = 0;
        while (newIdx < compacted.size() &&
               !sameRefLeaf(compacted[newIdx], val)) {
          newIdx++;
        }
        if (newIdx == compacted.size()) {
          compacted.push_back(val);
        }
        remap[oldIdx] = uint8_t(newIdx);
      }
      if (compacted.size() >= anf->values.size()) {
        continue;
      }
      auto palIt = paletteSlices.find(g->name);
      if (palIt != paletteSlices.end()) {
        if (codeGlobalGets[g->name] != palIt->second.size()) {
          continue;
        }
        anf->values.set(compacted);
        for (auto& slice : palIt->second) {
          for (Index j = 0; j < slice.length; j++) {
            uint8_t oldIdx = uint8_t(segmentData[slice.offset + j]);
            assert(oldIdx < remap.size());
            segmentData[slice.offset + j] = char(remap[oldIdx]);
          }
        }
        continue;
      }

      Index totalGets = codeGlobalGets[g->name];
      if (totalGets == 0) {
        continue;
      }
      std::vector<std::pair<Function*, Expression**>> idxSlots;
      auto& palCalls = allPalCalls[g->name];
      if (palCalls.size() == totalGets) {
        for (auto& [fn, call] : palCalls) {
          idxSlots.push_back({fn, &call->operands[1]});
        }
      } else if (palCalls.empty() &&
                 arrayGetSites[g->name].size() == totalGets) {
        for (auto& [fn, site] : arrayGetSites[g->name]) {
          idxSlots.push_back({fn, &(*site)->cast<ArrayGet>()->index});
        }
      } else {
        continue;
      }

      struct ExternalSlice {
        DataSegment* seg;
        Index offset;
        Index length;
      };
      std::vector<ExternalSlice> extSlices;
      bool slicesOk = true;
      for (auto& [fn, idxSlot] : idxSlots) {
        auto* idxAg = (*idxSlot)->dynCast<ArrayGet>();
        if (!idxAg || idxAg->order != MemoryOrder::Unordered ||
            idxAg->signed_) {
          slicesOk = false;
          break;
        }
        auto* tableGG = idxAg->ref->dynCast<GlobalGet>();
        if (!tableGG) {
          slicesOk = false;
          break;
        }
        Name tname = tableGG->name;
        auto* tg = module->getGlobalOrNull(tname);
        if (!tg || tg->imported() || !tg->mutable_ || !tg->init ||
            !tg->init->is<RefNull>() || nonCodeUses.count(tname) ||
            initGlobalUses[tname] > 0) {
          slicesOk = false;
          break;
        }
        auto& tSets = allGlobalSets[tname];
        if (tSets.size() != 1 || tSets[0].first != fn) {
          slicesOk = false;
          break;
        }
        auto* and_ = tSets[0].second->value->dynCast<ArrayNewData>();
        if (!and_ || and_->segment == segmentName) {
          slicesOk = false;
          break;
        }
        DataSegment* seg = module->getDataSegmentOrNull(and_->segment);
        uint32_t segStartOffset = 0, totalElems = 0;
        if (!seg || !evalConstI32(and_->offset, fn, segStartOffset) ||
            !evalConstI32(and_->size, fn, totalElems)) {
          slicesOk = false;
          break;
        }
        auto& tAgSites = arrayGetSites[tname];
        if (allRefIsNullGets[tname] + tAgSites.size() !=
            codeGlobalGets[tname]) {
          slicesOk = false;
          break;
        }
        for (auto& [agFn, _] : tAgSites) {
          if (agFn != fn) {
            slicesOk = false;
            break;
          }
        }
        if (!slicesOk) {
          break;
        }
        Index elemOffset = 0;
        if (!evalElemOffset(idxAg->index, fn, elemOffset)) {
          slicesOk = false;
          break;
        }
        Index sliceLen = 0;
        if (!getFactoryRowCount(fn, sliceLen)) {
          if (tAgSites.size() == 1 && elemOffset == 0) {
            sliceLen = Index(totalElems);
          } else {
            slicesOk = false;
            break;
          }
        }
        if (sliceLen == 0 ||
            uint64_t(elemOffset) + uint64_t(sliceLen) > uint64_t(totalElems) ||
            uint64_t(segStartOffset) + uint64_t(elemOffset) +
                uint64_t(sliceLen) >
              seg->data.size()) {
          slicesOk = false;
          break;
        }
        for (auto& [_, otherSite] : tAgSites) {
          auto* otherAg = (*otherSite)->cast<ArrayGet>();
          if (otherAg == idxAg) {
            continue;
          }
          Index otherOffset = 0;
          if (!evalElemOffset(otherAg->index, fn, otherOffset) ||
              !(uint64_t(otherOffset) + uint64_t(sliceLen) <=
                  uint64_t(elemOffset) ||
                uint64_t(elemOffset) + uint64_t(sliceLen) <=
                  uint64_t(otherOffset))) {
            slicesOk = false;
            break;
          }
        }
        if (!slicesOk) {
          break;
        }
        Index absOffset = Index(segStartOffset) + elemOffset;
        for (Index j = 0; j < sliceLen; j++) {
          if (uint8_t(seg->data[absOffset + j]) >= anf->values.size()) {
            slicesOk = false;
            break;
          }
        }
        if (!slicesOk) {
          break;
        }
        for (auto& prev : extSlices) {
          if (prev.seg == seg &&
              !(uint64_t(prev.offset) + uint64_t(prev.length) <=
                  uint64_t(absOffset) ||
                uint64_t(absOffset) + uint64_t(sliceLen) <=
                  uint64_t(prev.offset))) {
            slicesOk = false;
            break;
          }
        }
        if (!slicesOk) {
          break;
        }
        extSlices.push_back({seg, absOffset, sliceLen});
      }
      if (slicesOk && extSlices.size() == idxSlots.size()) {
        anf->values.set(compacted);
        for (auto& slice : extSlices) {
          for (Index j = 0; j < slice.length; j++) {
            uint8_t oldIdx = uint8_t(slice.seg->data[slice.offset + j]);
            assert(oldIdx < remap.size());
            slice.seg->data[slice.offset + j] = char(remap[oldIdx]);
          }
        }
      }
    }

    // Convert any array.new_fixed global initializer whose elements are all
    // ref.null into array.new_default.
    for (auto& g : module->globals) {
      if (g->imported() || !g->init) {
        continue;
      }
      auto* anf = g->init->dynCast<ArrayNewFixed>();
      if (!anf || anf->values.empty() || !anf->type.isRef()) {
        continue;
      }
      if (std::all_of(anf->values.begin(),
                      anf->values.end(),
                      [](Expression* v) { return v->is<RefNull>(); })) {
        g->init =
          builder.makeArrayNew(anf->type.getHeapType(),
                               builder.makeConst(int32_t(anf->values.size())));
      }
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

    std::unordered_map<Name, std::vector<std::pair<Function*, If*>>>
      lazyGuardIfs;
    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      for (auto* iff : funcAnalysis.map[func.get()].ifs) {
        Name flag;
        if (!iff->ifFalse && isFlagEqzCond(iff->condition, flag)) {
          flagGuardIfs[flag].push_back({func.get(), iff});
        } else if (!iff->ifFalse) {
          if (auto* rin = iff->condition->dynCast<RefIsNull>()) {
            if (auto* gg = rin->value->dynCast<GlobalGet>()) {
              lazyGuardIfs[gg->name].push_back({func.get(), iff});
            }
          }
        }
      }
    }

    // Identify direct once-helper functions: functions starting with
    // (global.set $flag (i32.const 1)) whose every call is in
    // (if (i32.eqz (global.get $flag)) (then (call $fn))), or functions whose
    // every call is in (if (ref.is_null (global.get $lazy)) (then (call $fn))).
    std::unordered_set<Name> onceHelperFuncs;
    std::unordered_map<Name, Function*> flagHelperFunc;
    std::unordered_map<Name, Function*> lazyHelperFunc;
    for (auto& func : module->functions) {
      if (func->imported() || nonCallFuncUses.count(func->name)) {
        continue;
      }
      auto& callers = funcCallers[func->name];
      if (callers.empty()) {
        continue;
      }
      auto* firstSet = getFirstStmt(func->body)->dynCast<GlobalSet>();
      if (firstSet && onceFlags.count(firstSet->name) &&
          funcSets[firstSet->name].size() == 1) {
        Name flag = firstSet->name;
        auto& guards = flagGuardIfs[flag];
        if (guards.size() == callers.size() &&
            guards.size() == funcGets[flag].size()) {
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
            continue;
          }
        }
      }
      for (auto* gs : FindAll<GlobalSet>(func->body).list) {
        Name lazy = gs->name;
        auto* lg = module->getGlobalOrNull(lazy);
        if (!lg || lg->imported() || !lg->mutable_ || !lg->type.isRef() ||
            !lg->type.isNullable() || !lg->init->is<RefNull>() ||
            nonCodeGlobalUses.count(lazy) || initGlobalUses[lazy] != 0 ||
            funcSets[lazy].size() != 1 || gs->value->is<RefNull>()) {
          continue;
        }
        auto& guards = lazyGuardIfs[lazy];
        if (guards.size() != callers.size()) {
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
          lazyHelperFunc[lazy] = func.get();
          break;
        }
      }
    }

    auto isBrIfOnceBlock = [&](Function* fn, Name& flagOut) -> bool {
      if (!fn || fn->imported() || !fn->body || fn->getNumParams() != 0 ||
          fn->getResults() != Type::none) {
        return false;
      }
      auto* b = fn->body->dynCast<Block>();
      if (!b || b->list.size() < 2) {
        return false;
      }
      Name flag;
      if (auto* br = b->list[0]->dynCast<Break>()) {
        if (br->condition && !br->value && br->name == b->name) {
          if (auto* gg = br->condition->dynCast<GlobalGet>()) {
            flag = gg->name;
          }
        }
      } else if (auto* iff = b->list[0]->dynCast<If>()) {
        if (!iff->ifFalse) {
          if (auto* gg = iff->condition->dynCast<GlobalGet>()) {
            bool exits = iff->ifTrue->is<Return>();
            if (!exits) {
              if (auto* br = iff->ifTrue->dynCast<Break>()) {
                exits = !br->condition && br->name == b->name;
              }
            }
            if (exits) {
              flag = gg->name;
            }
          }
        }
      }
      if (!flag.is() || !onceFlags.count(flag) || funcSets[flag].size() != 1 ||
          funcGets[flag].size() != 1) {
        return false;
      }
      auto* gs = b->list[1]->dynCast<GlobalSet>();
      if (!gs || gs->name != flag) {
        return false;
      }
      flagOut = flag;
      return true;
    };

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
          if (auto* call = getFirstStmt(iff->ifTrue)->dynCast<Call>()) {
            auto it = lazyHelperFunc.find(lazy);
            if (it != lazyHelperFunc.end() &&
                it->second->name == call->target) {
              return true;
            }
          }
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
        Name brFlag;
        walkOnce(&fn->body,
                 onceHelperFuncs.count(fn->name) > 0 ||
                   isBrIfOnceBlock(fn, brFlag),
                 false);
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
          if (onlyArrays) {
            if (auto* anf = g->init->dynCast<ArrayNewFixed>()) {
              if (anf->values.size() >= 2) {
                HeapType commonStruct;
                bool allSameStruct = true;
                for (auto* val : anf->values) {
                  auto* gg = val->dynCast<GlobalGet>();
                  if (!gg || initGlobalUses[gg->name] != 1) {
                    allSameStruct = false;
                    break;
                  }
                  auto* eg = module->getGlobalOrNull(gg->name);
                  if (!eg || eg->imported() || eg->mutable_ ||
                      !eg->init->is<StructNew>()) {
                    allSameStruct = false;
                    break;
                  }
                  HeapType ht = eg->init->type.getHeapType();
                  if (commonStruct == HeapType()) {
                    commonStruct = ht;
                  } else if (commonStruct != ht) {
                    allSameStruct = false;
                    break;
                  }
                }
                if (allSameStruct) {
                  bool isArrayGetRef = false;
                  for (auto* ag : FindAll<ArrayGet>(targetFn->body).list) {
                    if (ag->ref == *getPtr &&
                        ag->order == MemoryOrder::Unordered) {
                      isArrayGetRef = true;
                      break;
                    }
                  }
                  if (isArrayGetRef) {
                    continue;
                  }
                }
              }
            }
          }
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

    // Rebuild funcGets and funcSets helper so subsequent patterns have accurate
    // pointers and counts.
    auto rebuildGetsAndSets = [&]() {
      funcGets.clear();
      funcSets.clear();
      for (auto& func : module->functions) {
        if (func->imported()) {
          continue;
        }
        struct GetSetCollector : public PostWalker<GetSetCollector> {
          Function* fn;
          std::unordered_map<
            Name,
            std::vector<std::pair<Function*, Expression**>>>& funcGets;
          std::unordered_map<
            Name,
            std::vector<std::pair<Function*, Expression**>>>& funcSets;
          GetSetCollector(
            Function* fn,
            std::unordered_map<
              Name,
              std::vector<std::pair<Function*, Expression**>>>& funcGets,
            std::unordered_map<
              Name,
              std::vector<std::pair<Function*, Expression**>>>& funcSets)
            : fn(fn), funcGets(funcGets), funcSets(funcSets) {}
          void visitGlobalGet(GlobalGet* curr) {
            funcGets[curr->name].push_back({fn, getCurrentPointer()});
          }
          void visitGlobalSet(GlobalSet* curr) {
            funcSets[curr->name].push_back({fn, getCurrentPointer()});
          }
        };
        GetSetCollector(func.get(), funcGets, funcSets).walk(func->body);
      }
    };
    rebuildGetsAndSets();

    auto canStmtExitEarly = [&](Expression* s) -> bool {
      if (!FindAll<Return>(s).list.empty() ||
          !FindAll<Throw>(s).list.empty() ||
          !FindAll<Rethrow>(s).list.empty() ||
          !FindAll<Unreachable>(s).list.empty()) {
        return true;
      }
      std::unordered_set<Name> internalLabels;
      for (auto* b : FindAll<Block>(s).list) {
        if (b->name.is()) {
          internalLabels.insert(b->name);
        }
      }
      for (auto* l : FindAll<Loop>(s).list) {
        if (l->name.is()) {
          internalLabels.insert(l->name);
        }
      }
      for (auto* br : FindAll<Break>(s).list) {
        if (!internalLabels.count(br->name)) {
          return true;
        }
      }
      for (auto* sw : FindAll<Switch>(s).list) {
        if (!internalLabels.count(sw->default_)) {
          return true;
        }
        for (Name t : sw->targets) {
          if (!internalLabels.count(t)) {
            return true;
          }
        }
      }
      for (auto* bron : FindAll<BrOn>(s).list) {
        if (!internalLabels.count(bron->name)) {
          return true;
        }
      }
      return false;
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

    // -------------------------------------------------------------------------
    // Pattern 1.5: Collapse single-set alias globals in once-regions
    //   (global.set $g_alias (global.get $g_base)) -> replace gets with $g_base
    // -------------------------------------------------------------------------
    if (!preserveThunkTypes) {
      auto exprContainsPtr = [](Expression* root, Expression** targetPtr) {
        bool found = false;
        std::function<void(Expression**)> scan = [&](Expression** p) {
          if (!p || !*p || found) {
            return;
          }
          if (p == targetPtr) {
            found = true;
            return;
          }
          for (auto*& ch : ChildIterator(*p)) {
            scan(&ch);
          }
        };
        scan(&root);
        return found;
      };
      auto exprContainsNode = [](Expression* root, Expression* targetNode) {
        bool found = false;
        std::function<void(Expression*)> scan = [&](Expression* e) {
          if (!e || found) {
            return;
          }
          if (e == targetNode) {
            found = true;
            return;
          }
          for (auto* ch : ChildIterator(e)) {
            scan(ch);
          }
        };
        scan(root);
        return found;
      };

      bool aliasChanged = false;
      bool iterChanged = true;
      while (iterChanged) {
        iterChanged = false;
        std::vector<Name> aliasToRemove;
        for (auto& g : module->globals) {
          Name aliasName = g->name;
          if (g->imported() || !g->mutable_ || onceFlags.count(aliasName) ||
              nonCodeGlobalUses.count(aliasName) ||
              initGlobalUses[aliasName] > 0) {
            continue;
          }
          bool isDefaultInit =
            g->init->is<RefNull>() ||
            (g->init->is<Const>() && g->init->cast<Const>()->value.isZero());
          if (!isDefaultInit || funcSets[aliasName].size() != 1) {
            continue;
          }
          auto [setterFn, setPtr] = funcSets[aliasName][0];
          auto* gs = (*setPtr)->cast<GlobalSet>();
          auto* baseGet = gs->value->dynCast<GlobalGet>();
          if (!baseGet || baseGet->name == aliasName) {
            continue;
          }
          Name baseName = baseGet->name;
          Global* gBase = module->getGlobalOrNull(baseName);
          if (!gBase || gBase->imported() || gBase->type != g->type) {
            continue;
          }
          if (gBase->mutable_) {
            bool sameDefault =
              (gBase->init->is<RefNull>() && g->init->is<RefNull>()) ||
              (gBase->init->is<Const>() && g->init->is<Const>() &&
               gBase->init->cast<Const>()->value ==
                 g->init->cast<Const>()->value);
            if (!sameDefault || nonCodeGlobalUses.count(baseName) ||
                initGlobalUses[baseName] > 0 ||
                funcSets[baseName].size() != 1) {
              continue;
            }
          }

          // Locate gs inside a once-block of setterFn.
          Block* onceB = nullptr;
          Index aliasIdx = 0;
          std::function<void(Expression*, bool, bool)> findOnceBlock =
            [&](Expression* curr, bool inOnce, bool inLoopOrTry) {
              if (!curr || onceB) {
                return;
              }
              if (curr->is<Loop>() || curr->is<Try>() || curr->is<TryTable>()) {
                inLoopOrTry = true;
              }
              if (auto* iff = curr->dynCast<If>()) {
                findOnceBlock(iff->condition, inOnce, inLoopOrTry);
                findOnceBlock(
                  iff->ifTrue, inOnce || isOnceIf(iff), inLoopOrTry);
                if (iff->ifFalse) {
                  findOnceBlock(iff->ifFalse, inOnce, inLoopOrTry);
                }
                return;
              }
              if (auto* b = curr->dynCast<Block>()) {
                if (inOnce && !inLoopOrTry) {
                  for (Index i = 0; i < b->list.size(); i++) {
                    if (b->list[i] == gs) {
                      onceB = b;
                      aliasIdx = i;
                      return;
                    }
                  }
                }
              }
              for (auto* ch : ChildIterator(curr)) {
                findOnceBlock(ch, inOnce, inLoopOrTry);
              }
            };
          Name brFlag;
          findOnceBlock(setterFn->body,
                        onceHelperFuncs.count(setterFn->name) > 0 ||
                          isBrIfOnceBlock(setterFn, brFlag),
                        false);
          if (!onceB) {
            continue;
          }

          // Check that aliasName is never tested in RefIsNull or EqZInt32.
          bool testedAsGuard = false;
          for (auto& [rFn, _] : funcGets[aliasName]) {
            for (auto* rin : FindAll<RefIsNull>(rFn->body).list) {
              if (auto* gg = rin->value->dynCast<GlobalGet>()) {
                if (gg->name == aliasName) {
                  testedAsGuard = true;
                  break;
                }
              }
            }
            if (testedAsGuard) {
              break;
            }
            for (auto* u : FindAll<Unary>(rFn->body).list) {
              if (u->op == EqZInt32) {
                if (auto* gg = u->value->dynCast<GlobalGet>()) {
                  if (gg->name == aliasName) {
                    testedAsGuard = true;
                    break;
                  }
                }
              }
            }
            if (testedAsGuard) {
              break;
            }
          }
          if (testedAsGuard) {
            continue;
          }

          // Sub-case 1: gBase is set in the exact same onceB before aliasIdx
          // with no calls, early exits, or touches of aliasName in between.
          bool validAlias = false;
          if (gBase->mutable_ && funcSets[baseName][0].first == setterFn) {
            Expression* baseSet = *funcSets[baseName][0].second;
            for (Index i = 0; i < aliasIdx; i++) {
              if (onceB->list[i] == baseSet) {
                bool cleanBetween = true;
                for (Index j = i + 1; j < aliasIdx; j++) {
                  Expression* s = onceB->list[j];
                  if (canStmtExitEarly(s) || !FindAll<Call>(s).list.empty() ||
                      !FindAll<CallRef>(s).list.empty() ||
                      !FindAll<CallIndirect>(s).list.empty()) {
                    cleanBetween = false;
                    break;
                  }
                  for (auto* gg : FindAll<GlobalGet>(s).list) {
                    if (gg->name == aliasName) {
                      cleanBetween = false;
                      break;
                    }
                  }
                }
                if (cleanBetween) {
                  validAlias = true;
                }
                break;
              }
            }
          }

          // Sub-case 2: Every statement in onceB before aliasIdx cannot touch
          // aliasName or exit early, and every reader of aliasName is dominated
          // by setterFn / onceB[aliasIdx].
          if (!validAlias) {
            bool prefixOk = true;
            std::vector<Name> queue;
            std::unordered_set<Name> visited;
            for (Index i = 0; i < aliasIdx; i++) {
              Expression* s = onceB->list[i];
              if (i == 0 && isBrIfOnceBlock(setterFn, brFlag)) {
                continue;
              }
              if (canStmtExitEarly(s) || !FindAll<CallRef>(s).list.empty() ||
                  !FindAll<CallIndirect>(s).list.empty()) {
                prefixOk = false;
                break;
              }
              for (auto* gg : FindAll<GlobalGet>(s).list) {
                if (gg->name == aliasName) {
                  prefixOk = false;
                  break;
                }
              }
              for (auto* c : FindAll<Call>(s).list) {
                if (visited.insert(c->target).second) {
                  queue.push_back(c->target);
                }
              }
            }
            while (prefixOk && !queue.empty()) {
              Name fnName = queue.back();
              queue.pop_back();
              if (fnName == setterFn->name) {
                prefixOk = false;
                break;
              }
              Function* f = module->getFunctionOrNull(fnName);
              if (!f || f->imported()) {
                continue;
              }
              if (funcTouchesGlobal(f, aliasName)) {
                prefixOk = false;
                break;
              }
              for (auto callee : funcCallees[fnName]) {
                if (visited.insert(callee).second) {
                  queue.push_back(callee);
                }
              }
            }

            if (prefixOk) {
              auto isInitStmtForSetter = [&](Expression* s) -> bool {
                if (auto* c = s->dynCast<Call>()) {
                  if (c->target == setterFn->name) {
                    return true;
                  }
                }
                if (auto* iff = s->dynCast<If>()) {
                  if (!iff->ifFalse && isOnceIf(iff)) {
                    if (auto* c = getFirstStmt(iff->ifTrue)->dynCast<Call>()) {
                      if (c == iff->ifTrue && c->target == setterFn->name) {
                        return true;
                      }
                    }
                  }
                }
                return false;
              };

              bool allReadersOk = true;
              for (auto& [rFn, getPtr] : funcGets[aliasName]) {
                bool readerOk = false;
                if (rFn == setterFn) {
                  for (Index j = aliasIdx + 1; j < onceB->list.size(); j++) {
                    if (exprContainsPtr(onceB->list[j], getPtr)) {
                      readerOk = true;
                      break;
                    }
                  }
                } else if (!nonCallFuncUses.count(rFn->name) &&
                           !funcCallers[rFn->name].empty()) {
                  bool allCallersAfter = true;
                  for (auto& [cFn, callNode] : funcCallers[rFn->name]) {
                    if (cFn != setterFn) {
                      allCallersAfter = false;
                      break;
                    }
                    bool foundAfter = false;
                    for (Index j = aliasIdx + 1; j < onceB->list.size(); j++) {
                      if (exprContainsNode(onceB->list[j], callNode)) {
                        foundAfter = true;
                        break;
                      }
                    }
                    if (!foundAfter) {
                      allCallersAfter = false;
                      break;
                    }
                  }
                  if (allCallersAfter) {
                    readerOk = true;
                  }
                }
                if (!readerOk) {
                  for (auto* rb : FindAll<Block>(rFn->body).list) {
                    bool seenInit = false;
                    for (Index j = 0; j < rb->list.size(); j++) {
                      if (seenInit && exprContainsPtr(rb->list[j], getPtr)) {
                        readerOk = true;
                        break;
                      }
                      if (isInitStmtForSetter(rb->list[j])) {
                        seenInit = true;
                      }
                    }
                    if (readerOk) {
                      break;
                    }
                  }
                }
                if (!readerOk) {
                  allReadersOk = false;
                  break;
                }
              }
              if (allReadersOk) {
                validAlias = true;
              }
            }
          }

          if (validAlias) {
            for (auto& [rFn, getPtr] : funcGets[aliasName]) {
              *getPtr = builder.makeGlobalGet(baseName, gBase->type);
              modifiedFuncs.insert(rFn);
            }
            onceB->list.erase(onceB->list.begin() + aliasIdx);
            modifiedFuncs.insert(setterFn);
            aliasToRemove.push_back(aliasName);
            aliasChanged = true;
            iterChanged = true;
            break;
          }
        }
        for (auto name : aliasToRemove) {
          module->removeGlobal(name);
        }
        if (iterChanged) {
          rebuildGetsAndSets();
        }
      }
      (void)aliasChanged;
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
          Name brFlag;
          walkBlocks(fn->body,
                     onceHelperFuncs.count(fn->name) > 0 ||
                       isBrIfOnceBlock(fn, brFlag),
                     false);
        }
        for (auto name : demotedGlobals) {
          module->removeGlobal(name);
        }
      }
    }

    // -------------------------------------------------------------------------
    // Pattern 3: Eliminate redundant i32 once-flag globals, strip dead pure
    // allocation drops in once-regions, and sink caller once-guards.
    // -------------------------------------------------------------------------
    if (!preserveThunkTypes) {
      auto isSelfGuardedOnceFunc = [&](Function* f) -> bool {
        if (!f || f->imported() || !f->body || f->getNumParams() != 0 ||
            f->getResults() != Type::none) {
          return false;
        }
        Name brFlag;
        if (isBrIfOnceBlock(f, brFlag)) {
          return true;
        }
        if (auto* b = f->body->dynCast<Block>()) {
          if (b->list.size() >= 2) {
            if (auto* br = b->list[0]->dynCast<Break>()) {
              if (br->condition && !br->value && br->name == b->name) {
                if (auto* u = br->condition->dynCast<Unary>()) {
                  if (u->op == EqZInt32) {
                    if (auto* rin = u->value->dynCast<RefIsNull>()) {
                      if (rin->value->is<GlobalGet>()) {
                        return true;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        if (auto* iff = f->body->dynCast<If>()) {
          if (!iff->ifFalse) {
            Name flag;
            if (isFlagEqzCond(iff->condition, flag)) {
              if (auto* gs = getFirstStmt(iff->ifTrue)->dynCast<GlobalSet>()) {
                if (gs->name == flag) {
                  return true;
                }
              }
            }
            if (auto* rin = iff->condition->dynCast<RefIsNull>()) {
              if (auto* gg = rin->value->dynCast<GlobalGet>()) {
                Global* lg = module->getGlobalOrNull(gg->name);
                if (lg && !lg->imported() && lg->mutable_ && lg->type.isRef() &&
                    lg->type.isNullable() && lg->init->is<RefNull>()) {
                  for (auto* gs : FindAll<GlobalSet>(iff->ifTrue).list) {
                    if (gs->name == gg->name && !gs->value->is<RefNull>()) {
                      return true;
                    }
                  }
                }
              }
            }
          }
        }
        return false;
      };

      auto callsSetterOnce = [&](Name target, Function* setterFn) -> bool {
        if (target == setterFn->name) {
          return true;
        }
        Function* f = module->getFunctionOrNull(target);
        if (!f || f->imported() || !f->body) {
          return false;
        }
        if (auto* iff = f->body->dynCast<If>()) {
          if (!iff->ifFalse && isOnceIf(iff)) {
            if (auto* sc = getFirstStmt(iff->ifTrue)->dynCast<Call>()) {
              if (sc == iff->ifTrue && sc->target == setterFn->name) {
                return true;
              }
            }
          }
        }
        return false;
      };

      std::function<bool(Expression*, Expression*, Index)> exprMustDeref;
      auto exprMustDerefParam0 = [&](Function* f, Index depth) -> bool {
        if (depth > 3 || !f || f->imported() || !f->body ||
            f->getNumParams() < 1) {
          return false;
        }
        LocalGet lg;
        lg.index = 0;
        lg.type = f->getParams()[0];
        return exprMustDeref(f->body, &lg, depth);
      };
      exprMustDeref = [&](Expression* e,
                          Expression* target,
                          Index depth) -> bool {
        if (!e || !target || depth > 3) {
          return false;
        }
        if (auto* sg = e->dynCast<StructGet>()) {
          return ExpressionAnalyzer::equal(sg->ref, target) ||
                 exprMustDeref(sg->ref, target, depth);
        }
        if (auto* ss = e->dynCast<StructSet>()) {
          return ExpressionAnalyzer::equal(ss->ref, target) ||
                 exprMustDeref(ss->ref, target, depth) ||
                 exprMustDeref(ss->value, target, depth);
        }
        if (auto* ag = e->dynCast<ArrayGet>()) {
          return ExpressionAnalyzer::equal(ag->ref, target) ||
                 exprMustDeref(ag->ref, target, depth) ||
                 exprMustDeref(ag->index, target, depth);
        }
        if (auto* as = e->dynCast<ArraySet>()) {
          return ExpressionAnalyzer::equal(as->ref, target) ||
                 exprMustDeref(as->ref, target, depth) ||
                 exprMustDeref(as->index, target, depth) ||
                 exprMustDeref(as->value, target, depth);
        }
        if (auto* al = e->dynCast<ArrayLen>()) {
          return ExpressionAnalyzer::equal(al->ref, target) ||
                 exprMustDeref(al->ref, target, depth);
        }
        if (auto* ra = e->dynCast<RefAs>()) {
          return (ra->op == RefAsNonNull &&
                  ExpressionAnalyzer::equal(ra->value, target)) ||
                 exprMustDeref(ra->value, target, depth);
        }
        if (auto* u = e->dynCast<Unary>()) {
          return exprMustDeref(u->value, target, depth);
        }
        if (auto* bin = e->dynCast<Binary>()) {
          return exprMustDeref(bin->left, target, depth) ||
                 exprMustDeref(bin->right, target, depth);
        }
        if (auto* ls = e->dynCast<LocalSet>()) {
          return exprMustDeref(ls->value, target, depth);
        }
        if (auto* rc = e->dynCast<RefCast>()) {
          return exprMustDeref(rc->ref, target, depth);
        }
        if (auto* b = e->dynCast<Block>()) {
          return !b->list.empty() && exprMustDeref(b->list[0], target, depth);
        }
        if (auto* c = e->dynCast<Call>()) {
          for (auto* op : c->operands) {
            if (exprMustDeref(op, target, depth)) {
              return true;
            }
          }
          if (!c->operands.empty() &&
              ExpressionAnalyzer::equal(c->operands[0], target)) {
            return exprMustDerefParam0(
              module->getFunctionOrNull(c->target), depth + 1);
          }
        }
        return false;
      };

      std::function<bool(Expression*,
                         Function*,
                         const Block*,
                         Index,
                         Index,
                         std::unordered_set<Name>&,
                         Index)>
        isProvenNonNull = [&](Expression* val,
                              Function* fn,
                              const Block* topB,
                              Index flagIdx,
                              Index tgIdx,
                              std::unordered_set<Name>& visitedFuncs,
                              Index depth) -> bool {
        if (!val || depth > 8) {
          return false;
        }
        if (val->type.isRef() && val->type.isNonNullable()) {
          return true;
        }
        if (val->is<StructNew>() || val->is<ArrayNew>() ||
            val->is<ArrayNewFixed>() || val->is<ArrayNewData>() ||
            val->is<ArrayNewElem>() || val->is<RefFunc>() ||
            val->is<RefI31>() || val->is<StringConst>()) {
          return true;
        }
        if (auto* ra = val->dynCast<RefAs>()) {
          return ra->op == RefAsNonNull;
        }
        if (auto* rc = val->dynCast<RefCast>()) {
          return (rc->type.isRef() && rc->type.isNonNullable()) ||
                 isProvenNonNull(
                   rc->ref, fn, topB, flagIdx, tgIdx, visitedFuncs, depth + 1);
        }
        if (auto* ls = val->dynCast<LocalSet>()) {
          return ls->isTee() &&
                 isProvenNonNull(
                   ls->value, fn, topB, flagIdx, tgIdx, visitedFuncs, depth + 1);
        }
        if (auto* lg = val->dynCast<LocalGet>()) {
          if (lg->index < fn->getNumParams()) {
            return false;
          }
          bool foundSet = false;
          for (auto* ls : FindAll<LocalSet>(fn->body).list) {
            if (ls->index == lg->index) {
              foundSet = true;
              std::unordered_set<Name> subV = visitedFuncs;
              if (!isProvenNonNull(
                    ls->value, fn, topB, flagIdx, tgIdx, subV, depth + 1)) {
                return false;
              }
            }
          }
          return foundSet;
        }
        if (auto* gg = val->dynCast<GlobalGet>()) {
          Global* g = module->getGlobalOrNull(gg->name);
          if (!g || g->imported()) {
            return false;
          }
          if (g->type.isRef() && g->type.isNonNullable()) {
            return true;
          }
          if (!g->mutable_) {
            std::unordered_set<Name> subV = visitedFuncs;
            return isProvenNonNull(
              g->init, fn, nullptr, 0, 0, subV, depth + 1);
          }
          if (topB && !nonCodeGlobalUses.count(g->name) &&
              initGlobalUses[g->name] == 0 && funcSets[g->name].size() == 1) {
            auto [setterFn, setPtr] = funcSets[g->name][0];
            auto* gs = (*setPtr)->cast<GlobalSet>();
            bool calledBefore = false;
            for (Index i = flagIdx + 1; i < tgIdx; i++) {
              if (auto* c = topB->list[i]->dynCast<Call>()) {
                if (callsSetterOnce(c->target, setterFn)) {
                  calledBefore = true;
                  break;
                }
              }
            }
            if (calledBefore) {
              Block* setterB = setterFn->body->dynCast<Block>();
              Index sTgIdx = 0;
              if (setterB) {
                for (Index k = 0; k < setterB->list.size(); k++) {
                  if (setterB->list[k] == gs) {
                    sTgIdx = k;
                    break;
                  }
                }
              }
              std::unordered_set<Name> subV = visitedFuncs;
              return isProvenNonNull(gs->value,
                                     setterFn,
                                     sTgIdx > 0 ? setterB : nullptr,
                                     0,
                                     sTgIdx,
                                     subV,
                                     depth + 1);
            }
          }
          return false;
        }
        if (auto* b = val->dynCast<Block>()) {
          if (b->list.empty()) {
            return false;
          }
          std::unordered_set<Name> subV = visitedFuncs;
          if (!isProvenNonNull(
                b->list.back(), fn, topB, flagIdx, tgIdx, subV, depth + 1)) {
            return false;
          }
          if (b->name.is()) {
            for (auto* br : FindAll<Break>(b).list) {
              if (br->name == b->name) {
                std::unordered_set<Name> brV = visitedFuncs;
                if (!br->value ||
                    !isProvenNonNull(
                      br->value, fn, topB, flagIdx, tgIdx, brV, depth + 1)) {
                  return false;
                }
              }
            }
            for (auto* bron : FindAll<BrOn>(b).list) {
              if (bron->name == b->name &&
                  (bron->op == BrOnNull || bron->op == BrOnCastFail)) {
                return false;
              }
            }
          }
          return true;
        }
        if (auto* iff = val->dynCast<If>()) {
          if (!iff->ifFalse) {
            return false;
          }
          std::unordered_set<Name> subV1 = visitedFuncs;
          std::unordered_set<Name> subV2 = visitedFuncs;
          bool ok1 = isProvenNonNull(
            iff->ifTrue, fn, topB, flagIdx, tgIdx, subV1, depth + 1);
          bool ok2 = isProvenNonNull(
            iff->ifFalse, fn, topB, flagIdx, tgIdx, subV2, depth + 1);
          auto isSelfRecCall = [&](Expression* arm) -> bool {
            if (auto* c = arm->dynCast<Call>()) {
              return !c->isReturn && c->target == fn->name;
            }
            return false;
          };
          if (ok1 && !ok2) {
            if (isSelfRecCall(iff->ifFalse) ||
                (getPassOptions().trapsNeverHappen &&
                 exprMustDeref(iff->condition, iff->ifFalse, 0))) {
              ok2 = true;
            }
          } else if (ok2 && !ok1) {
            if (isSelfRecCall(iff->ifTrue) ||
                (getPassOptions().trapsNeverHappen &&
                 exprMustDeref(iff->condition, iff->ifTrue, 0))) {
              ok1 = true;
            }
          }
          return ok1 && ok2;
        }
        if (auto* sel = val->dynCast<Select>()) {
          std::unordered_set<Name> subV1 = visitedFuncs;
          std::unordered_set<Name> subV2 = visitedFuncs;
          return isProvenNonNull(
                   sel->ifTrue, fn, topB, flagIdx, tgIdx, subV1, depth + 1) &&
                 isProvenNonNull(
                   sel->ifFalse, fn, topB, flagIdx, tgIdx, subV2, depth + 1);
        }
        if (auto* c = val->dynCast<Call>()) {
          if (c->isReturn) {
            return false;
          }
          Function* callee = module->getFunctionOrNull(c->target);
          if (!callee || callee->imported() || !callee->getResults().isRef()) {
            return false;
          }
          if (callee->getResults().isNonNullable()) {
            return true;
          }
          if (!visitedFuncs.insert(callee->name).second) {
            return false;
          }
          for (auto* ret : FindAll<Return>(callee->body).list) {
            std::unordered_set<Name> subV = visitedFuncs;
            if (!ret->value ||
                !isProvenNonNull(
                  ret->value, callee, nullptr, 0, 0, subV, depth + 1)) {
              return false;
            }
          }
          if (callee->body->type != Type::unreachable) {
            std::unordered_set<Name> subV = visitedFuncs;
            if (!isProvenNonNull(
                  callee->body, callee, nullptr, 0, 0, subV, depth + 1)) {
              return false;
            }
          }
          return true;
        }
        return false;
      };

      // Strip dead pure-allocation (drop (call ...)) statements in once-regions
      // under --traps-never-happen before running Pattern 3.
      if (getPassOptions().trapsNeverHappen) {
        struct DropCalleeSummary {
          bool computed = false;
          bool valid = false;
          std::vector<HeapType> mutatedTypes;
        };
        std::unordered_map<Name, DropCalleeSummary> dropCalleeCache;

        auto getDropCalleeSummary =
          [&](Name fnName) -> const DropCalleeSummary& {
          auto& entry = dropCalleeCache[fnName];
          if (entry.computed) {
            return entry;
          }
          entry.computed = true;
          Function* f = module->getFunctionOrNull(fnName);
          if (!f || f->imported() || !f->body ||
              onceHelperFuncs.count(fnName) || isSelfGuardedOnceFunc(f)) {
            return entry;
          }
          struct Checker
            : public PostWalker<Checker, UnifiedExpressionVisitor<Checker>> {
            bool ok = true;
            std::vector<Name> gsets;
            std::vector<HeapType> mutatedTypes;
            void visitExpression(Expression* curr) {
              if (!ok) {
                return;
              }
              if (curr->is<Throw>() || curr->is<Rethrow>() ||
                  curr->is<Unreachable>() || curr->is<CallIndirect>() ||
                  curr->is<Store>() || curr->is<AtomicRMW>() ||
                  curr->is<AtomicCmpxchg>() || curr->is<MemoryCopy>() ||
                  curr->is<MemoryFill>() || curr->is<MemoryInit>() ||
                  curr->is<TableSet>() || curr->is<TableCopy>() ||
                  curr->is<TableFill>() || curr->is<TableGrow>() ||
                  curr->is<TableInit>()) {
                ok = false;
                return;
              }
              if (auto* gs = curr->dynCast<GlobalSet>()) {
                gsets.push_back(gs->name);
              } else if (auto* ss = curr->dynCast<StructSet>()) {
                if (ss->ref->type.isRef()) {
                  mutatedTypes.push_back(ss->ref->type.getHeapType());
                }
              } else if (auto* as = curr->dynCast<ArraySet>()) {
                if (as->ref->type.isRef()) {
                  mutatedTypes.push_back(as->ref->type.getHeapType());
                }
              } else if (auto* ac = curr->dynCast<ArrayCopy>()) {
                if (ac->destRef->type.isRef()) {
                  mutatedTypes.push_back(ac->destRef->type.getHeapType());
                }
              } else if (auto* af = curr->dynCast<ArrayFill>()) {
                if (af->ref->type.isRef()) {
                  mutatedTypes.push_back(af->ref->type.getHeapType());
                }
              }
            }
          } checker;
          checker.walk(f->body);
          if (!checker.ok) {
            return entry;
          }
          for (Name g : checker.gsets) {
            if (onceFlags.count(g)) {
              return entry;
            }
            std::unordered_set<Function*> wFuncs;
            if (auto it = funcSets.find(g); it != funcSets.end()) {
              for (auto& [wFn, _] : it->second) {
                wFuncs.insert(wFn);
              }
            }
            std::unordered_set<Function*> rFuncs;
            if (auto it = funcGets.find(g); it != funcGets.end()) {
              for (auto& [rFn, _] : it->second) {
                rFuncs.insert(rFn);
              }
            }
            bool isPrivate =
              (wFuncs.size() <= 1 && (wFuncs.empty() || wFuncs.count(f)) &&
               rFuncs.size() <= 1 && (rFuncs.empty() || rFuncs.count(f)));
            bool isScratch = (wFuncs.size() >= 2);
            if (!(isPrivate || isScratch)) {
              return entry;
            }
          }
          entry.valid = true;
          entry.mutatedTypes = std::move(checker.mutatedTypes);
          return entry;
        };

        auto isDeadPureOnceDrop = [&](Drop* d, Function* selfFn) -> bool {
          if (!d || !d->value) {
            return false;
          }
          std::unordered_set<Name> visitedNonNull = {selfFn->name};
          if (!isProvenNonNull(
                d->value, selfFn, nullptr, 0, 0, visitedNonNull, 0)) {
            return false;
          }
          struct ExprValidator
            : public PostWalker<ExprValidator,
                                UnifiedExpressionVisitor<ExprValidator>> {
            bool ok = true;
            std::vector<Name> calls;
            std::vector<Name> ggets;
            void visitExpression(Expression* curr) {
              if (!ok) {
                return;
              }
              if (auto* c = curr->dynCast<Call>()) {
                if (c->isReturn) {
                  ok = false;
                  return;
                }
                calls.push_back(c->target);
                return;
              }
              if (auto* gg = curr->dynCast<GlobalGet>()) {
                ggets.push_back(gg->name);
                return;
              }
              if (curr->is<Const>() || curr->is<RefNull>() ||
                  curr->is<RefFunc>() || curr->is<RefI31>() ||
                  curr->is<StringConst>() || curr->is<StructNew>() ||
                  curr->is<ArrayNew>() || curr->is<ArrayNewData>() ||
                  curr->is<ArrayNewElem>() || curr->is<ArrayNewFixed>() ||
                  curr->is<StructGet>() || curr->is<ArrayGet>() ||
                  curr->is<ArrayLen>() || curr->is<RefCast>() ||
                  curr->is<RefTest>() || curr->is<RefAs>() ||
                  curr->is<RefIsNull>() || curr->is<RefEq>() ||
                  curr->is<I31Get>() || curr->is<Unary>() ||
                  curr->is<Binary>() || curr->is<Select>() ||
                  curr->is<Block>() || curr->is<If>()) {
                return;
              }
              ok = false;
            }
          } validator;
          validator.walk(d->value);
          if (!validator.ok) {
            return false;
          }
          if (validator.calls.empty()) {
            return true;
          }
          std::unordered_set<Name> visited;
          std::vector<Name> queue;
          for (Name c : validator.calls) {
            if (visited.insert(c).second) {
              queue.push_back(c);
            }
          }
          std::unordered_set<HeapType> mutatedTypes;
          while (!queue.empty()) {
            if (visited.size() > 128) {
              return false;
            }
            Name fnName = queue.back();
            queue.pop_back();
            if (fnName == selfFn->name) {
              return false;
            }
            const auto& sum = getDropCalleeSummary(fnName);
            if (!sum.valid) {
              return false;
            }
            for (HeapType ht : sum.mutatedTypes) {
              mutatedTypes.insert(ht);
            }
            for (Name callee : funcCallees[fnName]) {
              if (visited.insert(callee).second) {
                queue.push_back(callee);
              }
            }
          }
          for (Name g : validator.ggets) {
            Global* glob = module->getGlobalOrNull(g);
            if (glob && glob->type.isRef()) {
              if (mutatedTypes.count(glob->type.getHeapType())) {
                return false;
              }
            }
          }
          return true;
        };

        bool anyDropRemoved = false;
        for (auto& func : module->functions) {
          if (func->imported() || !func->body) {
            continue;
          }
          Function* fn = func.get();
          std::function<void(Expression*, bool, bool)> walkDropBlocks =
            [&](Expression* curr, bool inOnce, bool inLoopOrTry) {
              if (!curr) {
                return;
              }
              if (curr->is<Loop>() || curr->is<Try>() || curr->is<TryTable>()) {
                inLoopOrTry = true;
              }
              if (auto* iff = curr->dynCast<If>()) {
                walkDropBlocks(iff->condition, inOnce, inLoopOrTry);
                walkDropBlocks(
                  iff->ifTrue, inOnce || isOnceIf(iff), inLoopOrTry);
                if (iff->ifFalse) {
                  walkDropBlocks(iff->ifFalse, inOnce, inLoopOrTry);
                }
                return;
              }
              if (auto* b = curr->dynCast<Block>()) {
                if (inOnce && !inLoopOrTry) {
                  for (Index i = 0; i < b->list.size();) {
                    Expression* s = b->list[i];
                    bool removeStmt = false;
                    if (auto* d = s->dynCast<Drop>()) {
                      if (isDeadPureOnceDrop(d, fn)) {
                        removeStmt = true;
                      }
                    } else if (auto* c = s->dynCast<Call>()) {
                      if (c->operands.empty() && c->type == Type::none) {
                        Function* callee = module->getFunctionOrNull(c->target);
                        if (callee && !callee->imported() &&
                            callee->vars.empty() && callee->body) {
                          if (auto* cd = callee->body->dynCast<Drop>()) {
                            if (isDeadPureOnceDrop(cd, callee)) {
                              removeStmt = true;
                            }
                          }
                        }
                      }
                    }
                    // Do not erase the sole trailing element of a value-typed
                    // block (or leave a block completely empty during walk).
                    if (removeStmt && b->list.size() > 1 &&
                        !(b->type.isConcrete() && i + 1 == b->list.size())) {
                      b->list.erase(b->list.begin() + i);
                      modifiedFuncs.insert(fn);
                      anyDropRemoved = true;
                      continue;
                    }
                    i++;
                  }
                }
              }
              for (auto* ch : ChildIterator(curr)) {
                walkDropBlocks(ch, inOnce, inLoopOrTry);
              }
            };
          Name brFlag;
          walkDropBlocks(fn->body,
                         onceHelperFuncs.count(fn->name) > 0 ||
                           isBrIfOnceBlock(fn, brFlag),
                         false);
        }
        if (anyDropRemoved) {
          rebuildGetsAndSets();
        }
      }

      auto canReachTargetOrFlag = [&](const ExpressionList& stmts,
                                      Index flagIdx,
                                      Index tgIdx,
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
          for (auto* gs : FindAll<GlobalSet>(e).list) {
            if (gs->name == flag || gs->name == tg) {
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
        for (Index i = flagIdx + 1; i < tgIdx; i++) {
          if (scanExpr(stmts[i])) {
            return true;
          }
        }
        if (scanExpr(stmts[tgIdx]->cast<GlobalSet>()->value)) {
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

      auto isEffectivelyEmptyVoidCallee = [&](Name target) -> bool {
        Function* f = module->getFunctionOrNull(target);
        if (!f || f->imported() || !f->body || f->getNumParams() != 0 ||
            f->getResults() != Type::none) {
          return false;
        }
        if (f->body->is<Nop>()) {
          return true;
        }
        Name brFlag;
        if (isBrIfOnceBlock(f, brFlag)) {
          auto* b = f->body->cast<Block>();
          if (b->list.size() == 2) {
            return true;
          }
        }
        return false;
      };

      auto cleanOnceBlock = [&](Block* topB,
                                Name flag,
                                Function* selfFn,
                                If* selfGuardIf,
                                Index startIdx,
                                Index& flagIdxOut) -> bool {
        if (!topB || topB->list.size() <= startIdx) {
          return false;
        }
        Index flagIdx = startIdx;
        while (flagIdx < topB->list.size() &&
               isTrivialLocalInit(topB->list[flagIdx])) {
          flagIdx++;
        }
        if (flagIdx >= topB->list.size()) {
          return false;
        }
        auto* firstSet = topB->list[flagIdx]->dynCast<GlobalSet>();
        if (!firstSet || firstSet->name != flag) {
          return false;
        }
        for (Index i = startIdx; i < topB->list.size(); i++) {
          if (i == flagIdx) {
            continue;
          }
          for (auto* gg : FindAll<GlobalGet>(topB->list[i]).list) {
            if (gg->name == flag) {
              return false;
            }
          }
          for (auto* gs : FindAll<GlobalSet>(topB->list[i]).list) {
            if (gs->name == flag) {
              return false;
            }
          }
        }
        // Strip Nops, calls to emptied once-functions, redundant default-value
        // stores, and direct self-calls before any call has executed.
        bool seenAnyCall = false;
        std::unordered_set<Name> seenSets;
        for (Index i = flagIdx + 1; i < topB->list.size();) {
          Expression* s = topB->list[i];
          if (s->is<Nop>()) {
            topB->list.erase(topB->list.begin() + i);
            modifiedFuncs.insert(selfFn);
            continue;
          }
          if (auto* c = s->dynCast<Call>()) {
            if (c->operands.empty() && c->type == Type::none) {
              if ((selfGuardIf && selfFn->body == selfGuardIf &&
                   c->target == selfFn->name) ||
                  isEffectivelyEmptyVoidCallee(c->target)) {
                topB->list.erase(topB->list.begin() + i);
                modifiedFuncs.insert(selfFn);
                continue;
              }
            }
          }
          if (!seenAnyCall) {
            if (auto* gs = s->dynCast<GlobalSet>()) {
              Name gname = gs->name;
              Global* g = module->getGlobalOrNull(gname);
              if (g && !g->imported() && g->mutable_ && g->init &&
                  !seenSets.count(gname)) {
                bool sameAsInit =
                  (gs->value->is<RefNull>() && g->init->is<RefNull>()) ||
                  (gs->value->is<Const>() && g->init->is<Const>() &&
                   gs->value->cast<Const>()->value ==
                     g->init->cast<Const>()->value);
                if (sameAsInit) {
                  auto& setsVec = funcSets[gname];
                  setsVec.erase(
                    std::remove_if(setsVec.begin(),
                                   setsVec.end(),
                                   [&](const std::pair<Function*, Expression**>&
                                         entry) { return *entry.second == s; }),
                    setsVec.end());
                  topB->list.erase(topB->list.begin() + i);
                  modifiedFuncs.insert(selfFn);
                  continue;
                }
              }
            }
          }
          if (!FindAll<Call>(s).list.empty() ||
              !FindAll<CallRef>(s).list.empty() ||
              !FindAll<CallIndirect>(s).list.empty()) {
            seenAnyCall = true;
          }
          for (auto* gs : FindAll<GlobalSet>(s).list) {
            seenSets.insert(gs->name);
          }
          i++;
        }
        flagIdxOut = flagIdx;
        return true;
      };

      auto allRemainingAreSelfGuardedOnceCalls = [&](Block* topB,
                                                     Index flagIdx) -> bool {
        if (flagIdx + 1 >= topB->list.size()) {
          return false;
        }
        for (Index i = flagIdx + 1; i < topB->list.size(); i++) {
          auto* c = topB->list[i]->dynCast<Call>();
          if (!c || !c->operands.empty() || c->type != Type::none ||
              !isSelfGuardedOnceFunc(module->getFunctionOrNull(c->target))) {
            return false;
          }
        }
        return true;
      };

      auto validateOnceBlock = [&](Block* topB,
                                   Index flagIdx,
                                   Name flag,
                                   Function* selfFn,
                                   Global*& tgGlobOut) -> bool {
        for (Index tgIdx = flagIdx + 1; tgIdx < topB->list.size(); tgIdx++) {
          Expression* stmt = topB->list[tgIdx];
          auto* tgSet = stmt->dynCast<GlobalSet>();
          if (tgSet) {
            Name tg = tgSet->name;
            if (tg != flag && !nonCodeGlobalUses.count(tg) &&
                initGlobalUses[tg] == 0 && funcSets[tg].size() == 1) {
              Global* tgGlob = module->getGlobalOrNull(tg);
              if (tgGlob && !tgGlob->imported() && tgGlob->mutable_ &&
                  tgGlob->type.isRef() && tgGlob->type.isNullable() &&
                  tgGlob->init->is<RefNull>()) {
                if (!canStmtExitEarly(tgSet->value) &&
                    !canReachTargetOrFlag(
                      topB->list, flagIdx, tgIdx, flag, tg, selfFn)) {
                  std::unordered_set<Name> visitedFuncs;
                  visitedFuncs.insert(selfFn->name);
                  if (isProvenNonNull(tgSet->value,
                                      selfFn,
                                      topB,
                                      flagIdx,
                                      tgIdx,
                                      visitedFuncs,
                                      0)) {
                    tgGlobOut = tgGlob;
                    return true;
                  }
                }
              }
            }
          }
          if (canStmtExitEarly(stmt)) {
            return false;
          }
        }
        return false;
      };

      std::unordered_map<Function*,
                         std::unordered_map<Expression*, Expression*>>
        exprReplacements;
      std::unordered_set<Name> unwrappedFlagHelpers;
      std::vector<Name> flagsToRemove;

      for (auto& g : module->globals) {
        Name flag = g->name;
        if (!onceFlags.count(flag) || funcSets[flag].size() != 1 ||
            funcGets[flag].empty()) {
          continue;
        }
        auto& guards = flagGuardIfs[flag];

        // Case 3: br_if-guarded once-function form
        if (guards.empty() && funcGets[flag].size() == 1) {
          Function* fn = funcSets[flag][0].first;
          Name brFlag;
          if (funcGets[flag][0].first == fn && isBrIfOnceBlock(fn, brFlag) &&
              brFlag == flag) {
            Block* topB = fn->body->cast<Block>();
            Index flagIdx = 1;
            if (cleanOnceBlock(topB, flag, fn, nullptr, 1, flagIdx)) {
              if (flagIdx + 1 == topB->list.size()) {
                fn->body = builder.makeNop();
                modifiedFuncs.insert(fn);
                if (!nonCallFuncUses.count(fn->name)) {
                  for (auto& [callerFn, callNode] : funcCallers[fn->name]) {
                    ExpressionManipulator::nop(callNode);
                    modifiedFuncs.insert(callerFn);
                  }
                }
                flagsToRemove.push_back(flag);
                continue;
              }
              if (allRemainingAreSelfGuardedOnceCalls(topB, flagIdx)) {
                topB->list.erase(topB->list.begin(),
                                 topB->list.begin() + flagIdx + 1);
                modifiedFuncs.insert(fn);
                flagsToRemove.push_back(flag);
                continue;
              }
              Global* tgGlob = nullptr;
              if (validateOnceBlock(topB, flagIdx, flag, fn, tgGlob)) {
                topB->list.erase(topB->list.begin() + flagIdx);
                bool labelUsed = false;
                if (topB->name.is()) {
                  for (Index k = 1; k < topB->list.size(); k++) {
                    for (auto* br : FindAll<Break>(topB->list[k]).list) {
                      if (br->name == topB->name) {
                        labelUsed = true;
                        break;
                      }
                    }
                    for (auto* sw : FindAll<Switch>(topB->list[k]).list) {
                      if (sw->default_ == topB->name) {
                        labelUsed = true;
                      }
                      for (Name t : sw->targets) {
                        if (t == topB->name) {
                          labelUsed = true;
                        }
                      }
                    }
                    for (auto* bron : FindAll<BrOn>(topB->list[k]).list) {
                      if (bron->name == topB->name) {
                        labelUsed = true;
                      }
                    }
                  }
                }
                if (!labelUsed) {
                  topB->list.erase(topB->list.begin());
                  topB->name = Name();
                  fn->body = builder.makeIf(
                    builder.makeRefIsNull(
                      builder.makeGlobalGet(tgGlob->name, tgGlob->type)),
                    topB);
                } else {
                  Expression* nonNullCond = builder.makeUnary(
                    EqZInt32,
                    builder.makeRefIsNull(
                      builder.makeGlobalGet(tgGlob->name, tgGlob->type)));
                  if (auto* br = topB->list[0]->dynCast<Break>()) {
                    br->condition = nonNullCond;
                  } else if (auto* iff = topB->list[0]->dynCast<If>()) {
                    iff->condition = nonNullCond;
                  }
                }
                modifiedFuncs.insert(fn);
                flagsToRemove.push_back(flag);
                continue;
              }
            }
          }
          continue;
        }

        if (guards.size() != funcGets[flag].size()) {
          continue;
        }
        // Case 1: onceHelperFunc form
        if (auto it = flagHelperFunc.find(flag); it != flagHelperFunc.end()) {
          Function* helperFn = it->second;
          Block* topB = getEditableBlock(helperFn->body);
          Index flagIdx = 0;
          if (cleanOnceBlock(topB, flag, helperFn, nullptr, 0, flagIdx)) {
            if (flagIdx + 1 == topB->list.size()) {
              for (auto& [callerFn, iff] : guards) {
                Builder(*module).replaceWithIdenticalType(iff);
                modifiedFuncs.insert(callerFn);
              }
              topB->list.erase(topB->list.begin() + flagIdx);
              if (topB->list.empty()) {
                helperFn->body = builder.makeNop();
              }
              modifiedFuncs.insert(helperFn);
              unwrappedFlagHelpers.insert(flag);
              flagsToRemove.push_back(flag);
              continue;
            }
            if (allRemainingAreSelfGuardedOnceCalls(topB, flagIdx)) {
              for (auto& [callerFn, iff] : guards) {
                exprReplacements[callerFn][iff] = iff->ifTrue;
                modifiedFuncs.insert(callerFn);
              }
              topB->list.erase(topB->list.begin() + flagIdx);
              modifiedFuncs.insert(helperFn);
              unwrappedFlagHelpers.insert(flag);
              flagsToRemove.push_back(flag);
              continue;
            }
            Global* tgGlob = nullptr;
            if (validateOnceBlock(topB, flagIdx, flag, helperFn, tgGlob)) {
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
        }
        // Case 2: inline onceIf form (1 get in the same function)
        if (guards.size() == 1) {
          auto [fn, iff] = guards[0];
          if (funcSets[flag][0].first == fn) {
            Block* topB = getEditableBlock(iff->ifTrue);
            Index flagIdx = 0;
            if (cleanOnceBlock(topB, flag, fn, iff, 0, flagIdx)) {
              if (flagIdx + 1 == topB->list.size()) {
                Builder(*module).replaceWithIdenticalType(iff);
                modifiedFuncs.insert(fn);
                flagsToRemove.push_back(flag);
                continue;
              }
              if (allRemainingAreSelfGuardedOnceCalls(topB, flagIdx)) {
                topB->list.erase(topB->list.begin() + flagIdx);
                exprReplacements[fn][iff] = iff->ifTrue;
                modifiedFuncs.insert(fn);
                flagsToRemove.push_back(flag);
                continue;
              }
              Global* tgGlob = nullptr;
              if (validateOnceBlock(topB, flagIdx, flag, fn, tgGlob)) {
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
      }

      // Sink caller once-guards into 0-param void onceHelperFuncs that have
      // >= 2 guarded call sites so every call site becomes a direct (call $fn)
      // and OnceCallOpt can deduplicate them.
      auto sinkGuardsIntoHelper =
        [&](Function* helperFn,
            const std::vector<std::pair<Function*, If*>>& guards) {
          if (!helperFn || helperFn->imported() || !helperFn->body ||
              helperFn->body->is<Nop>() || helperFn->getNumParams() != 0 ||
              helperFn->getResults() != Type::none || guards.size() < 2) {
            return;
          }
          Expression* condCopy =
            ExpressionManipulator::copy(guards[0].second->condition, *module);
          helperFn->body = builder.makeIf(condCopy, helperFn->body);
          modifiedFuncs.insert(helperFn);
          for (auto& [callerFn, iff] : guards) {
            exprReplacements[callerFn][iff] = iff->ifTrue;
            modifiedFuncs.insert(callerFn);
          }
        };

      for (auto& [flag, helperFn] : flagHelperFunc) {
        if (!unwrappedFlagHelpers.count(flag)) {
          sinkGuardsIntoHelper(helperFn, flagGuardIfs[flag]);
        }
      }
      for (auto& [lazy, helperFn] : lazyHelperFunc) {
        sinkGuardsIntoHelper(helperFn, lazyGuardIfs[lazy]);
      }

      for (auto& [fn, repMap] : exprReplacements) {
        struct Replacer
          : public PostWalker<Replacer, UnifiedExpressionVisitor<Replacer>> {
          std::unordered_map<Expression*, Expression*>& repMap;
          Replacer(std::unordered_map<Expression*, Expression*>& repMap)
            : repMap(repMap) {}
          void visitExpression(Expression* curr) {
            auto it = repMap.find(curr);
            if (it != repMap.end()) {
              replaceCurrent(it->second);
            }
          }
        } replacer(repMap);
        replacer.walk(fn->body);
      }

      for (auto flag : flagsToRemove) {
        module->removeGlobal(flag);
      }
    }

    for (auto* fn : modifiedFuncs) {
      ReFinalize().walkFunctionInModule(fn, module);
    }

    if (!onlyArrays) {
      eliminateRedundantOnceCalls();
    }
  }

  void eliminateRedundantOnceCalls() {
    struct OnceCallOpt {
      Module& wasm;
      const PassOptions& options;

      std::unordered_set<Name> onceFuncs;
      std::unordered_map<Name, Name> guardGlobal;
      std::unordered_map<Name, std::unordered_set<Name>> impliedOnce;
      std::unordered_map<Name, std::unordered_set<Name>> exitActiveOnce;
      std::unordered_map<Name, std::unordered_set<Name>> entryActiveOnce;
      std::unordered_map<Name, std::unordered_set<Name>> onceWrittenGlobals;

      std::unordered_set<Name> pureLazyOnce;
      std::unordered_set<Name> safePureOnce;
      std::unordered_map<Name, std::unordered_set<Name>> pureTransitiveGlobals;
      std::unordered_set<Name> allSafePureGlobals;

      std::unordered_set<Name> nonCodeGlobals;
      std::unordered_set<Name> nonCallFuncUses;
      std::unordered_set<Name> resetZeroGlobals;
      std::unordered_map<Name, std::unordered_set<Name>> funcGets;
      std::unordered_map<Name, std::unordered_set<Name>> funcSets;
      std::unordered_map<Name, std::unordered_map<Name, Index>> funcSetCounts;
      std::unordered_map<Name, std::unordered_set<Name>> funcCalls;
      std::unordered_map<Name, bool> pureReadOnlyFunc;
      std::unordered_map<Name, std::unordered_set<Name>> globalWriterFuncs;
      std::unordered_map<Name, std::unordered_set<Name>> globalReaderFuncs;

      OnceCallOpt(Module& wasm, const PassOptions& options)
        : wasm(wasm), options(options) {}

      static bool extractGuardGlobal(Expression* cond, Name& gOut) {
        if (auto* r = cond->dynCast<RefIsNull>()) {
          if (auto* gg = r->value->dynCast<GlobalGet>()) {
            gOut = gg->name;
            return true;
          }
        }
        if (auto* u = cond->dynCast<Unary>()) {
          if (u->op == EqZInt32) {
            if (auto* gg = u->value->dynCast<GlobalGet>()) {
              gOut = gg->name;
              return true;
            }
          }
        }
        return false;
      }

      static bool isZeroOrNullExpr(Expression* e) {
        if (!e) {
          return false;
        }
        while (auto* ls = e->dynCast<LocalSet>()) {
          e = ls->value;
        }
        if (e->is<RefNull>()) {
          return true;
        }
        if (auto* c = e->dynCast<Const>()) {
          return c->value.isZero();
        }
        return false;
      }

      static bool doesNotReturn(Expression* e) {
        if (!e) {
          return false;
        }
        if (e->type == Type::unreachable || e->is<Return>() ||
            e->is<Throw>() || e->is<Rethrow>() || e->is<Unreachable>() ||
            e->is<Switch>()) {
          return true;
        }
        if (auto* br = e->dynCast<Break>()) {
          return !br->condition;
        }
        return false;
      }

      bool isValidGuardGlobal(Name gName) const {
        auto* g = wasm.getGlobalOrNull(gName);
        if (!g || g->imported() || !g->mutable_ || !g->init) {
          return false;
        }
        if (nonCodeGlobals.contains(gName) ||
            resetZeroGlobals.contains(gName)) {
          return false;
        }
        return isZeroOrNullExpr(g->init);
      }

      static bool itemMayExitBlockOrFunc(Expression* item, Name blockName) {
        if (!FindAll<Return>(item).list.empty()) {
          return true;
        }
        if (blockName.is()) {
          for (auto* br : FindAll<Break>(item).list) {
            if (br->name == blockName) {
              return true;
            }
          }
          for (auto* sw : FindAll<Switch>(item).list) {
            if (sw->default_ == blockName) {
              return true;
            }
            for (auto t : sw->targets) {
              if (t == blockName) {
                return true;
              }
            }
          }
        }
        return false;
      }

      static bool collectUnconditionalCalls(Expression* e,
                                            Name blockName,
                                            std::vector<Name>& out) {
        if (!e) {
          return true;
        }
        if (e->is<If>() || e->is<Select>() || e->is<Loop>() || e->is<Try>() ||
            e->is<TryTable>() || e->is<Return>() || e->is<Throw>() ||
            e->is<Rethrow>() || e->is<Unreachable>() || e->is<Break>() ||
            e->is<Switch>() || e->is<BrOn>()) {
          return false;
        }
        if (auto* b = e->dynCast<Block>()) {
          for (auto* item : b->list) {
            if (!collectUnconditionalCalls(item, b->name, out)) {
              return false;
            }
            if (itemMayExitBlockOrFunc(item, b->name) ||
                (blockName.is() && itemMayExitBlockOrFunc(item, blockName))) {
              return false;
            }
          }
          return true;
        }
        for (auto* child : ChildIterator(e)) {
          if (!collectUnconditionalCalls(child, blockName, out)) {
            return false;
          }
        }
        if (auto* c = e->dynCast<Call>()) {
          out.push_back(c->target);
        }
        return true;
      }

      bool
      topLevelSetsGlobal(Expression* body, Name gName, int depth = 0) const {
        if (!body || depth > 1) {
          return false;
        }
        if (auto* gs = body->dynCast<GlobalSet>()) {
          return gs->name == gName;
        }
        if (auto* c = body->dynCast<Call>()) {
          if (auto* callee = wasm.getFunctionOrNull(c->target)) {
            if (!callee->imported() && callee->body) {
              return topLevelSetsGlobal(callee->body, gName, depth + 1);
            }
          }
          return false;
        }
        if (auto* b = body->dynCast<Block>()) {
          for (auto* item : b->list) {
            if (topLevelSetsGlobal(item, gName, depth)) {
              return true;
            }
            if (itemMayExitBlockOrFunc(item, b->name)) {
              return false;
            }
          }
        }
        return false;
      }

      bool isPureInitValue(Expression* val) const {
        struct Validator
          : public PostWalker<Validator, UnifiedExpressionVisitor<Validator>> {
          const OnceCallOpt& parent;
          bool ok = true;
          Validator(const OnceCallOpt& parent) : parent(parent) {}
          void visitExpression(Expression* curr) {
            if (!ok) {
              return;
            }
            if (auto* c = curr->dynCast<Call>()) {
              if (c->isReturn) {
                ok = false;
                return;
              }
              if (c->type == Type::none) {
                if (!c->operands.empty() ||
                    !parent.pureLazyOnce.contains(c->target)) {
                  ok = false;
                }
                return;
              }
              auto* targetFn = parent.wasm.getFunctionOrNull(c->target);
              if (!targetFn || targetFn->imported()) {
                ok = false;
                return;
              }
              auto sIt = parent.funcSets.find(c->target);
              if (sIt != parent.funcSets.end()) {
                for (Name hg : sIt->second) {
                  auto wIt = parent.globalWriterFuncs.find(hg);
                  auto rIt = parent.globalReaderFuncs.find(hg);
                  if ((wIt != parent.globalWriterFuncs.end() &&
                       wIt->second.size() > 1) ||
                      (rIt != parent.globalReaderFuncs.end() &&
                       (rIt->second.size() > 1 ||
                        (rIt->second.size() == 1 &&
                         !rIt->second.contains(c->target))))) {
                    ok = false;
                    return;
                  }
                }
              }
              return;
            }
            if (curr->is<Const>() || curr->is<RefNull>() ||
                curr->is<RefFunc>() || curr->is<RefI31>() ||
                curr->is<StringConst>() || curr->is<GlobalGet>() ||
                curr->is<StructNew>() || curr->is<ArrayNew>() ||
                curr->is<ArrayNewData>() || curr->is<ArrayNewElem>() ||
                curr->is<ArrayNewFixed>() || curr->is<StructGet>() ||
                curr->is<ArrayGet>() || curr->is<ArrayLen>() ||
                curr->is<RefCast>() || curr->is<RefTest>() ||
                curr->is<RefAs>() || curr->is<RefIsNull>() ||
                curr->is<RefEq>() || curr->is<I31Get>() ||
                curr->is<Unary>() || curr->is<Binary>() ||
                curr->is<Select>() || curr->is<Block>() || curr->is<If>()) {
              return;
            }
            ok = false;
          }
        } validator(*this);
        validator.walk(val);
        return validator.ok;
      }

      static void intersectSet(std::unordered_set<Name>& dst,
                               const std::unordered_set<Name>& src) {
        for (auto it = dst.begin(); it != dst.end();) {
          if (!src.contains(*it)) {
            it = dst.erase(it);
          } else {
            ++it;
          }
        }
      }

      std::unordered_set<Name> computeFuncExitOnce(Function* func) const {
        std::unordered_set<Name> empty;
        if (!func || func->imported() || !func->body ||
            !FindAll<TryTable>(func->body).list.empty()) {
          return empty;
        }
        std::unordered_map<Name, std::vector<std::unordered_set<Name>>>
          breakStates;
        std::vector<std::unordered_set<Name>> returnStates;

        std::function<void(Expression*, std::unordered_set<Name>&)> walk =
          [&](Expression* curr, std::unordered_set<Name>& active) {
            if (!curr) {
              return;
            }
            if (auto* c = curr->dynCast<Call>()) {
              for (auto* op : c->operands) {
                walk(op, active);
              }
              auto it = exitActiveOnce.find(c->target);
              if (it != exitActiveOnce.end()) {
                active.insert(it->second.begin(), it->second.end());
              }
              return;
            }
            if (auto* ret = curr->dynCast<Return>()) {
              if (ret->value) {
                walk(ret->value, active);
              }
              returnStates.push_back(active);
              return;
            }
            if (auto* iff = curr->dynCast<If>()) {
              walk(iff->condition, active);
              auto entry = active;
              auto trueSet = entry;
              walk(iff->ifTrue, trueSet);
              bool trueReach = !doesNotReturn(iff->ifTrue);
              if (iff->ifFalse) {
                auto falseSet = entry;
                walk(iff->ifFalse, falseSet);
                bool falseReach = !doesNotReturn(iff->ifFalse);
                if (trueReach && falseReach) {
                  intersectSet(trueSet, falseSet);
                  active = std::move(trueSet);
                } else if (trueReach) {
                  active = std::move(trueSet);
                } else if (falseReach) {
                  active = std::move(falseSet);
                }
              } else {
                active = std::move(entry);
              }
              return;
            }
            if (auto* b = curr->dynCast<Block>()) {
              if (b->name.is()) {
                breakStates[b->name].clear();
              }
              bool fallthroughReach = true;
              for (auto* item : b->list) {
                walk(item, active);
                if (doesNotReturn(item)) {
                  fallthroughReach = false;
                  break;
                }
              }
              if (b->name.is()) {
                auto& bStates = breakStates[b->name];
                if (!bStates.empty()) {
                  if (fallthroughReach) {
                    for (const auto& bs : bStates) {
                      intersectSet(active, bs);
                    }
                  } else {
                    active = bStates[0];
                    for (size_t i = 1; i < bStates.size(); i++) {
                      intersectSet(active, bStates[i]);
                    }
                  }
                }
                breakStates.erase(b->name);
              }
              return;
            }
            if (auto* br = curr->dynCast<Break>()) {
              if (br->value) {
                walk(br->value, active);
              }
              if (br->condition) {
                walk(br->condition, active);
              }
              auto it = breakStates.find(br->name);
              if (it != breakStates.end()) {
                it->second.push_back(active);
              }
              return;
            }
            if (auto* sw = curr->dynCast<Switch>()) {
              if (sw->value) {
                walk(sw->value, active);
              }
              walk(sw->condition, active);
              for (Name t : sw->targets) {
                auto it = breakStates.find(t);
                if (it != breakStates.end()) {
                  it->second.push_back(active);
                }
              }
              auto it = breakStates.find(sw->default_);
              if (it != breakStates.end()) {
                it->second.push_back(active);
              }
              return;
            }
            if (auto* loop = curr->dynCast<Loop>()) {
              auto entry = active;
              auto bodySet = entry;
              walk(loop->body, bodySet);
              active = std::move(entry);
              return;
            }
            if (auto* tryy = curr->dynCast<Try>()) {
              auto entry = active;
              auto bodySet = entry;
              walk(tryy->body, bodySet);
              for (auto* cb : tryy->catchBodies) {
                auto cbSet = entry;
                walk(cb, cbSet);
              }
              active = std::move(entry);
              return;
            }
            for (auto* child : ChildIterator(curr)) {
              walk(child, active);
            }
          };

        std::unordered_set<Name> active;
        walk(func->body, active);
        bool bodyReachesEnd = !doesNotReturn(func->body);
        if (bodyReachesEnd) {
          for (const auto& rs : returnStates) {
            intersectSet(active, rs);
          }
          return active;
        }
        if (!returnStates.empty()) {
          auto res = returnStates[0];
          for (size_t i = 1; i < returnStates.size(); i++) {
            intersectSet(res, returnStates[i]);
          }
          return res;
        }
        return empty;
      }

      void analyze() {
        for (auto& exp : wasm.exports) {
          if (exp->kind == ExternalKind::Global) {
            nonCodeGlobals.insert(*exp->getInternalName());
          } else if (exp->kind == ExternalKind::Function) {
            nonCallFuncUses.insert(*exp->getInternalName());
          }
        }
        if (wasm.start.is()) {
          nonCallFuncUses.insert(wasm.start);
        }
        for (auto& g : wasm.globals) {
          if (!g->imported() && g->init) {
            for (auto* gg : FindAll<GlobalGet>(g->init).list) {
              nonCodeGlobals.insert(gg->name);
            }
            for (auto* rf : FindAll<RefFunc>(g->init).list) {
              nonCallFuncUses.insert(rf->func);
            }
          }
        }
        for (auto& seg : wasm.elementSegments) {
          if (seg->offset) {
            for (auto* gg : FindAll<GlobalGet>(seg->offset).list) {
              nonCodeGlobals.insert(gg->name);
            }
          }
          for (auto* item : seg->data) {
            for (auto* rf : FindAll<RefFunc>(item).list) {
              nonCallFuncUses.insert(rf->func);
            }
          }
        }
        ModuleUtils::iterActiveDataSegments(wasm, [&](DataSegment* seg) {
          for (auto* gg : FindAll<GlobalGet>(seg->offset).list) {
            nonCodeGlobals.insert(gg->name);
          }
        });

        struct FuncSummary {
          std::unordered_set<Name> gets;
          std::unordered_set<Name> sets;
          std::unordered_map<Name, Index> setCounts;
          std::unordered_set<Name> calls;
          std::unordered_set<Name> refFuncs;
          std::unordered_set<Name> resetZero;
          bool pureReadOnly = true;
        };
        ModuleUtils::ParallelFunctionAnalysis<FuncSummary> analysis(
          wasm, [&](Function* func, FuncSummary& summary) {
            if (func->imported() || !func->body) {
              summary.pureReadOnly = false;
              return;
            }
            struct Scanner
              : public PostWalker<Scanner, UnifiedExpressionVisitor<Scanner>> {
              FuncSummary& summary;
              Scanner(FuncSummary& summary) : summary(summary) {}
              void visitExpression(Expression* curr) {
                if (auto* gg = curr->dynCast<GlobalGet>()) {
                  summary.gets.insert(gg->name);
                  return;
                }
                if (auto* gs = curr->dynCast<GlobalSet>()) {
                  summary.sets.insert(gs->name);
                  summary.setCounts[gs->name]++;
                  summary.pureReadOnly = false;
                  if (OnceCallOpt::isZeroOrNullExpr(gs->value)) {
                    summary.resetZero.insert(gs->name);
                  }
                  return;
                }
                if (auto* c = curr->dynCast<Call>()) {
                  summary.calls.insert(c->target);
                  return;
                }
                if (auto* rf = curr->dynCast<RefFunc>()) {
                  summary.refFuncs.insert(rf->func);
                  return;
                }
                if (summary.pureReadOnly) {
                  if (curr->is<CallRef>() || curr->is<CallIndirect>() ||
                      curr->is<Throw>() || curr->is<Rethrow>() ||
                      curr->is<Unreachable>() || curr->is<StructSet>() ||
                      curr->is<ArraySet>() || curr->is<ArrayCopy>() ||
                      curr->is<ArrayFill>() || curr->is<Store>() ||
                      curr->is<AtomicRMW>() || curr->is<AtomicCmpxchg>() ||
                      curr->is<MemoryCopy>() || curr->is<MemoryFill>() ||
                      curr->is<MemoryInit>() || curr->is<TableSet>() ||
                      curr->is<TableCopy>() || curr->is<TableFill>() ||
                      curr->is<TableGrow>() || curr->is<TableInit>()) {
                    summary.pureReadOnly = false;
                  }
                }
              }
            } scanner(summary);
            scanner.walk(func->body);
          });

        for (auto& [func, summary] : analysis.map) {
          if (func->imported()) {
            continue;
          }
          Name fn = func->name;
          for (Name g : summary.gets) {
            globalReaderFuncs[g].insert(fn);
          }
          for (Name g : summary.sets) {
            globalWriterFuncs[g].insert(fn);
          }
          for (Name g : summary.resetZero) {
            resetZeroGlobals.insert(g);
          }
          for (Name rf : summary.refFuncs) {
            nonCallFuncUses.insert(rf);
          }
          funcGets[fn] = std::move(summary.gets);
          funcSets[fn] = std::move(summary.sets);
          funcSetCounts[fn] = std::move(summary.setCounts);
          funcCalls[fn] = std::move(summary.calls);
          pureReadOnlyFunc[fn] = summary.pureReadOnly;
        }

        std::unordered_map<Name, std::vector<Name>> directImplied;
        bool changed = true;
        while (changed) {
          changed = false;
          for (auto& fPtr : wasm.functions) {
            Function* f = fPtr.get();
            if (f->imported() || !f->body || f->getNumParams() != 0 ||
                f->getResults() != Type::none || onceFuncs.contains(f->name)) {
              continue;
            }
            if (auto* b = f->body->dynCast<Block>()) {
              if (b->list.size() >= 2) {
                Name guardG;
                if (auto* br = b->list[0]->dynCast<Break>()) {
                  if (br->condition && !br->value && br->name == b->name) {
                    if (auto* gg = br->condition->dynCast<GlobalGet>()) {
                      guardG = gg->name;
                    }
                  }
                } else if (auto* iff = b->list[0]->dynCast<If>()) {
                  if (!iff->ifFalse) {
                    if (auto* gg = iff->condition->dynCast<GlobalGet>()) {
                      bool exits = iff->ifTrue->is<Return>();
                      if (!exits) {
                        if (auto* br = iff->ifTrue->dynCast<Break>()) {
                          exits = !br->condition && br->name == b->name;
                        }
                      }
                      if (exits) {
                        guardG = gg->name;
                      }
                    }
                  }
                }
                if (guardG.is() && isValidGuardGlobal(guardG)) {
                  auto* gs = b->list[1]->dynCast<GlobalSet>();
                  if (gs && gs->name == guardG) {
                    onceFuncs.insert(f->name);
                    guardGlobal[f->name] = guardG;
                    auto& imp = directImplied[f->name];
                    for (Index i = 1; i < b->list.size(); i++) {
                      if (!collectUnconditionalCalls(
                            b->list[i], b->name, imp) ||
                          itemMayExitBlockOrFunc(b->list[i], b->name)) {
                        break;
                      }
                    }
                    changed = true;
                    continue;
                  }
                }
              }
            }

            If* mainIf = f->body->dynCast<If>();
            std::vector<Name> outerOnceCalls;
            if (!mainIf) {
              if (auto* b = f->body->dynCast<Block>()) {
                bool allValid = true;
                for (auto* item : b->list) {
                  if (auto* iff = item->dynCast<If>()) {
                    if (mainIf) {
                      allValid = false;
                      break;
                    }
                    mainIf = iff;
                  } else if (auto* c = item->dynCast<Call>()) {
                    if (onceFuncs.contains(c->target)) {
                      outerOnceCalls.push_back(c->target);
                    } else {
                      allValid = false;
                      break;
                    }
                  } else {
                    allValid = false;
                    break;
                  }
                }
                if (!allValid) {
                  mainIf = nullptr;
                }
              }
            }
            if (mainIf && !mainIf->ifFalse) {
              Name guardG;
              if (extractGuardGlobal(mainIf->condition, guardG) &&
                  isValidGuardGlobal(guardG) &&
                  topLevelSetsGlobal(mainIf->ifTrue, guardG)) {
                onceFuncs.insert(f->name);
                guardGlobal[f->name] = guardG;
                auto& imp = directImplied[f->name];
                for (Name oc : outerOnceCalls) {
                  imp.push_back(oc);
                }
                collectUnconditionalCalls(mainIf->ifTrue, Name(), imp);
                changed = true;
              }
            }
          }
        }

        if (onceFuncs.empty()) {
          return;
        }

        for (Name fn : onceFuncs) {
          auto& impSet = impliedOnce[fn];
          impSet.insert(fn);
          std::vector<Name> worklist = {fn};
          while (!worklist.empty()) {
            Name cur = worklist.back();
            worklist.pop_back();
            auto dIt = directImplied.find(cur);
            if (dIt != directImplied.end()) {
              for (Name nxt : dIt->second) {
                if (onceFuncs.contains(nxt) && impSet.insert(nxt).second) {
                  worklist.push_back(nxt);
                }
              }
            }
          }
          exitActiveOnce[fn] = impSet;

          auto& written = onceWrittenGlobals[fn];
          std::unordered_set<Name> visitedCallees = {fn};
          std::vector<Name> callWorklist = {fn};
          while (!callWorklist.empty()) {
            Name cur = callWorklist.back();
            callWorklist.pop_back();
            auto sIt = funcSets.find(cur);
            if (sIt != funcSets.end()) {
              written.insert(sIt->second.begin(), sIt->second.end());
            }
            auto cIt = funcCalls.find(cur);
            if (cIt != funcCalls.end()) {
              for (Name callee : cIt->second) {
                if (visitedCallees.insert(callee).second) {
                  callWorklist.push_back(callee);
                }
              }
            }
          }
        }

        // Propagate exitActiveOnce bottom-up across non-once helper functions.
        for (int iter = 0; iter < 3; iter++) {
          bool exitChanged = false;
          for (auto& fPtr : wasm.functions) {
            Function* f = fPtr.get();
            if (f->imported() || !f->body || onceFuncs.contains(f->name)) {
              continue;
            }
            bool callsAnyExitOnce = false;
            for (Name callee : funcCalls[f->name]) {
              if (exitActiveOnce.contains(callee)) {
                callsAnyExitOnce = true;
                break;
              }
            }
            if (!callsAnyExitOnce) {
              continue;
            }
            auto computed = computeFuncExitOnce(f);
            if (!computed.empty()) {
              auto& cur = exitActiveOnce[f->name];
              size_t before = cur.size();
              cur.insert(computed.begin(), computed.end());
              if (cur.size() != before) {
                exitChanged = true;
              }
            }
          }
          if (!exitChanged) {
            break;
          }
        }

        std::unordered_map<Name, std::unordered_set<Name>> pureDirectGlobals;
        std::unordered_map<Name, std::unordered_set<Name>> pureDirectPrereqs;
        changed = true;
        while (changed) {
          changed = false;
          for (Name fn : onceFuncs) {
            if (pureLazyOnce.contains(fn)) {
              continue;
            }
            Function* f = wasm.getFunction(fn);
            if (!f->vars.empty()) {
              continue;
            }
            Name guardG;
            std::vector<Expression*> stmts;
            std::unordered_set<Name> prereqs;

            // Check Shape B (br_if / early-return block) or Shape A (outer
            // pureLazyOnce calls + mainIf).
            bool shapeMatched = false;
            if (auto* b = f->body->dynCast<Block>()) {
              if (b->list.size() >= 3) {
                Name brG;
                if (auto* br = b->list[0]->dynCast<Break>()) {
                  if (br->condition && !br->value && br->name == b->name) {
                    if (auto* gg = br->condition->dynCast<GlobalGet>()) {
                      brG = gg->name;
                    }
                  }
                } else if (auto* iff = b->list[0]->dynCast<If>()) {
                  if (!iff->ifFalse) {
                    if (auto* gg = iff->condition->dynCast<GlobalGet>()) {
                      bool exits = iff->ifTrue->is<Return>();
                      if (!exits) {
                        if (auto* br = iff->ifTrue->dynCast<Break>()) {
                          exits = !br->condition && br->name == b->name;
                        }
                      }
                      if (exits) {
                        brG = gg->name;
                      }
                    }
                  }
                }
                if (brG.is()) {
                  if (auto* gs = b->list[1]->dynCast<GlobalSet>()) {
                    if (gs->name == brG) {
                      guardG = brG;
                      for (Index i = 1; i < b->list.size(); i++) {
                        stmts.push_back(b->list[i]);
                      }
                      shapeMatched = true;
                    }
                  }
                }
              }
            }
            if (!shapeMatched) {
              If* mainIf = f->body->dynCast<If>();
              if (!mainIf) {
                if (auto* b = f->body->dynCast<Block>()) {
                  if (!b->name.is()) {
                    bool allValid = true;
                    for (auto* item : b->list) {
                      if (auto* iff = item->dynCast<If>()) {
                        if (mainIf) {
                          allValid = false;
                          break;
                        }
                        mainIf = iff;
                      } else if (auto* c = item->dynCast<Call>()) {
                        if (c->operands.empty() &&
                            pureLazyOnce.contains(c->target)) {
                          prereqs.insert(c->target);
                        } else {
                          allValid = false;
                          break;
                        }
                      } else {
                        allValid = false;
                        break;
                      }
                    }
                    if (!allValid) {
                      mainIf = nullptr;
                    }
                  }
                }
              }
              if (mainIf && !mainIf->ifFalse &&
                  extractGuardGlobal(mainIf->condition, guardG)) {
                if (auto* tb = mainIf->ifTrue->dynCast<Block>()) {
                  if (!tb->name.is()) {
                    for (auto* item : tb->list) {
                      stmts.push_back(item);
                    }
                    shapeMatched = true;
                  }
                } else {
                  stmts.push_back(mainIf->ifTrue);
                  shapeMatched = true;
                }
              }
            }
            if (!shapeMatched) {
              continue;
            }

            std::unordered_set<Name> writtenGs;
            bool ok = true;
            auto scIt = funcSetCounts.find(fn);
            for (Expression* stmt : stmts) {
              if (auto* c = stmt->dynCast<Call>()) {
                if (!c->operands.empty() || !pureLazyOnce.contains(c->target)) {
                  ok = false;
                  break;
                }
                prereqs.insert(c->target);
              } else if (auto* gs = stmt->dynCast<GlobalSet>()) {
                Name wg = gs->name;
                if (!isValidGuardGlobal(wg)) {
                  ok = false;
                  break;
                }
                auto wIt = globalWriterFuncs.find(wg);
                if (wIt == globalWriterFuncs.end() || wIt->second.size() != 1 ||
                    !wIt->second.contains(fn)) {
                  ok = false;
                  break;
                }
                if (scIt == funcSetCounts.end() ||
                    scIt->second.find(wg)->second != 1) {
                  ok = false;
                  break;
                }
                if (!isPureInitValue(gs->value)) {
                  ok = false;
                  break;
                }
                for (auto* subCall : FindAll<Call>(gs->value).list) {
                  if (subCall->type == Type::none &&
                      pureLazyOnce.contains(subCall->target)) {
                    prereqs.insert(subCall->target);
                  }
                }
                writtenGs.insert(wg);
              } else {
                ok = false;
                break;
              }
            }
            if (ok && writtenGs.contains(guardG)) {
              pureLazyOnce.insert(fn);
              pureDirectGlobals[fn] = std::move(writtenGs);
              pureDirectPrereqs[fn] = std::move(prereqs);
              changed = true;
            }
          }
        }

        if (pureLazyOnce.empty()) {
          return;
        }

        for (Name fn : pureLazyOnce) {
          auto& tg = pureTransitiveGlobals[fn];
          std::unordered_set<Name> visited = {fn};
          std::vector<Name> worklist = {fn};
          while (!worklist.empty()) {
            Name cur = worklist.back();
            worklist.pop_back();
            for (Name g : pureDirectGlobals[cur]) {
              tg.insert(g);
            }
            for (Name p : pureDirectPrereqs[cur]) {
              if (visited.insert(p).second) {
                worklist.push_back(p);
              }
            }
          }
        }

        for (Name fn : pureLazyOnce) {
          const auto& tg = pureTransitiveGlobals[fn];
          bool allReadersProtected = true;
          for (Name g : tg) {
            auto rIt = globalReaderFuncs.find(g);
            if (rIt == globalReaderFuncs.end()) {
              continue;
            }
            for (Name readerName : rIt->second) {
              if (pureLazyOnce.contains(readerName) &&
                  pureTransitiveGlobals[readerName].contains(g)) {
                continue;
              }
              Function* readerFn = wasm.getFunction(readerName);
              if (!verifyReaderDominatesGlobal(readerFn, g)) {
                allReadersProtected = false;
                break;
              }
            }
            if (!allReadersProtected) {
              break;
            }
          }
          if (allReadersProtected) {
            safePureOnce.insert(fn);
          }
        }

        changed = true;
        while (changed) {
          changed = false;
          for (auto it = safePureOnce.begin(); it != safePureOnce.end();) {
            bool ok = true;
            for (Name p : pureDirectPrereqs[*it]) {
              if (!safePureOnce.contains(p)) {
                ok = false;
                break;
              }
            }
            if (!ok) {
              it = safePureOnce.erase(it);
              changed = true;
            } else {
              ++it;
            }
          }
        }

        for (Name fn : safePureOnce) {
          for (Name g : pureTransitiveGlobals[fn]) {
            allSafePureGlobals.insert(g);
          }
        }
      }

      bool verifyReaderDominatesGlobal(Function* readerFn, Name targetG) const {
        if (!readerFn || !readerFn->body) {
          return false;
        }
        std::unordered_map<Name, std::vector<bool>> breakStates;
        bool allDominated = true;

        std::function<void(Expression*, bool&)> walk = [&](Expression* curr,
                                                           bool& isInit) {
          if (!curr || !allDominated) {
            return;
          }
          if (auto* gg = curr->dynCast<GlobalGet>()) {
            if (gg->name == targetG && !isInit) {
              allDominated = false;
            }
            return;
          }
          if (auto* c = curr->dynCast<Call>()) {
            for (auto* op : c->operands) {
              walk(op, isInit);
            }
            auto it = pureTransitiveGlobals.find(c->target);
            if (it != pureTransitiveGlobals.end() &&
                it->second.contains(targetG)) {
              isInit = true;
            } else {
              auto exIt = exitActiveOnce.find(c->target);
              if (exIt != exitActiveOnce.end()) {
                for (Name o : exIt->second) {
                  auto oIt = pureTransitiveGlobals.find(o);
                  if (oIt != pureTransitiveGlobals.end() &&
                      oIt->second.contains(targetG)) {
                    isInit = true;
                    break;
                  }
                }
              }
            }
            return;
          }
          if (auto* iff = curr->dynCast<If>()) {
            walk(iff->condition, isInit);
            bool entry = isInit;
            bool trueInit = entry;
            walk(iff->ifTrue, trueInit);
            bool trueReach = !doesNotReturn(iff->ifTrue);
            if (iff->ifFalse) {
              bool falseInit = entry;
              walk(iff->ifFalse, falseInit);
              bool falseReach = !doesNotReturn(iff->ifFalse);
              if (trueReach && falseReach) {
                isInit = trueInit && falseInit;
              } else if (trueReach) {
                isInit = trueInit;
              } else if (falseReach) {
                isInit = falseInit;
              }
            } else {
              isInit = entry;
            }
            return;
          }
          if (auto* b = curr->dynCast<Block>()) {
            if (b->name.is()) {
              breakStates[b->name].clear();
            }
            bool fallthroughReach = true;
            for (auto* item : b->list) {
              walk(item, isInit);
              if (doesNotReturn(item)) {
                fallthroughReach = false;
                break;
              }
            }
            if (b->name.is()) {
              auto& bStates = breakStates[b->name];
              if (!bStates.empty()) {
                bool joined = fallthroughReach ? isInit : true;
                for (bool bs : bStates) {
                  joined = joined && bs;
                }
                isInit = joined;
              }
              breakStates.erase(b->name);
            }
            return;
          }
          if (auto* br = curr->dynCast<Break>()) {
            if (br->value) {
              walk(br->value, isInit);
            }
            if (br->condition) {
              walk(br->condition, isInit);
            }
            auto it = breakStates.find(br->name);
            if (it != breakStates.end()) {
              it->second.push_back(isInit);
            }
            return;
          }
          if (auto* sw = curr->dynCast<Switch>()) {
            if (sw->value) {
              walk(sw->value, isInit);
            }
            walk(sw->condition, isInit);
            for (Name t : sw->targets) {
              auto it = breakStates.find(t);
              if (it != breakStates.end()) {
                it->second.push_back(isInit);
              }
            }
            auto it = breakStates.find(sw->default_);
            if (it != breakStates.end()) {
              it->second.push_back(isInit);
            }
            return;
          }
          if (auto* loop = curr->dynCast<Loop>()) {
            bool entry = isInit;
            bool bodyInit = entry;
            walk(loop->body, bodyInit);
            isInit = entry;
            return;
          }
          if (auto* tryy = curr->dynCast<Try>()) {
            bool entry = isInit;
            bool bodyInit = entry;
            walk(tryy->body, bodyInit);
            for (auto* cb : tryy->catchBodies) {
              bool cbInit = entry;
              walk(cb, cbInit);
            }
            isInit = entry;
            return;
          }
          for (auto* child : ChildIterator(curr)) {
            walk(child, isInit);
          }
        };

        bool init = false;
        walk(readerFn->body, init);
        return allDominated;
      }

      bool canHoistOverCallsInCond(Expression* cond, Name onceTarget) const {
        struct CondCallScanner
          : public PostWalker<CondCallScanner,
                              UnifiedExpressionVisitor<CondCallScanner>> {
          bool ok = true;
          std::vector<Name> calls;
          void visitExpression(Expression* curr) {
            if (!ok) {
              return;
            }
            if (auto* c = curr->dynCast<Call>()) {
              if (c->isReturn) {
                ok = false;
                return;
              }
              calls.push_back(c->target);
            } else if (curr->is<CallRef>() || curr->is<CallIndirect>()) {
              ok = false;
            }
          }
        } scanner;
        scanner.walk(cond);
        if (!scanner.ok) {
          return false;
        }
        std::unordered_set<Name> visited;
        std::vector<Name> queue;
        for (Name c : scanner.calls) {
          if (visited.insert(c).second) {
            queue.push_back(c);
          }
        }
        auto wIt = onceWrittenGlobals.find(onceTarget);
        while (!queue.empty()) {
          if (visited.size() > 16) {
            return false;
          }
          Name fn = queue.back();
          queue.pop_back();
          auto pIt = pureReadOnlyFunc.find(fn);
          if (pIt == pureReadOnlyFunc.end() || !pIt->second) {
            return false;
          }
          if (wIt != onceWrittenGlobals.end()) {
            auto gIt = funcGets.find(fn);
            if (gIt != funcGets.end()) {
              for (Name g : gIt->second) {
                if (wIt->second.contains(g)) {
                  return false;
                }
              }
            }
          }
          auto cIt = funcCalls.find(fn);
          if (cIt != funcCalls.end()) {
            for (Name callee : cIt->second) {
              if (visited.insert(callee).second) {
                queue.push_back(callee);
              }
            }
          }
        }
        return true;
      }

      void computeEntryActiveOnce() {
        std::unordered_set<Name> hasCallers;
        for (auto& fPtr : wasm.functions) {
          Function* f = fPtr.get();
          if (f->imported() || !f->body) {
            continue;
          }
          for (auto* c : FindAll<Call>(f->body).list) {
            hasCallers.insert(c->target);
          }
        }
        std::unordered_set<Name> rootFuncs = nonCallFuncUses;
        for (auto& fPtr : wasm.functions) {
          if (!fPtr->imported() && fPtr->body &&
              !hasCallers.contains(fPtr->name)) {
            rootFuncs.insert(fPtr->name);
          }
        }

        std::unordered_map<Name, std::unordered_set<Name>> entryActive;
        std::unordered_set<Name> hasEntryState;
        for (Name r : rootFuncs) {
          hasEntryState.insert(r);
          entryActive[r] = {};
        }

        for (int iter = 0; iter < 6; iter++) {
          std::unordered_map<Name, std::unordered_set<Name>> nextEntry;
          std::unordered_set<Name> nextHas;
          for (Name r : rootFuncs) {
            nextHas.insert(r);
            nextEntry[r] = {};
          }

          for (auto& fPtr : wasm.functions) {
            Function* func = fPtr.get();
            if (func->imported() || !func->body ||
                !hasEntryState.contains(func->name)) {
              continue;
            }
            Name selfGuard;
            if (onceFuncs.contains(func->name)) {
              auto gIt = guardGlobal.find(func->name);
              if (gIt != guardGlobal.end()) {
                selfGuard = gIt->second;
              }
            }

            std::unordered_map<Name, std::vector<std::unordered_set<Name>>>
              breakStates;
            std::function<void(Expression*, std::unordered_set<Name>&)> walk =
              [&](Expression* curr, std::unordered_set<Name>& active) {
                if (!curr) {
                  return;
                }
                if (auto* gs = curr->dynCast<GlobalSet>()) {
                  walk(gs->value, active);
                  if (selfGuard.is() && gs->name == selfGuard) {
                    active.insert(func->name);
                  }
                  return;
                }
                if (auto* c = curr->dynCast<Call>()) {
                  for (auto* op : c->operands) {
                    walk(op, active);
                  }
                  if (!nextHas.contains(c->target)) {
                    nextHas.insert(c->target);
                    nextEntry[c->target] = active;
                  } else {
                    intersectSet(nextEntry[c->target], active);
                  }
                  auto exIt = exitActiveOnce.find(c->target);
                  if (exIt != exitActiveOnce.end()) {
                    active.insert(exIt->second.begin(), exIt->second.end());
                  }
                  if (selfGuard.is() && topLevelSetsGlobal(c, selfGuard)) {
                    active.insert(func->name);
                  }
                  return;
                }
                if (auto* iff = curr->dynCast<If>()) {
                  walk(iff->condition, active);
                  auto entry = active;
                  auto trueSet = entry;
                  walk(iff->ifTrue, trueSet);
                  bool trueReach = !doesNotReturn(iff->ifTrue);
                  if (iff->ifFalse) {
                    auto falseSet = entry;
                    walk(iff->ifFalse, falseSet);
                    bool falseReach = !doesNotReturn(iff->ifFalse);
                    if (trueReach && falseReach) {
                      intersectSet(trueSet, falseSet);
                      active = std::move(trueSet);
                    } else if (trueReach) {
                      active = std::move(trueSet);
                    } else if (falseReach) {
                      active = std::move(falseSet);
                    }
                  } else {
                    active = std::move(entry);
                  }
                  return;
                }
                if (auto* b = curr->dynCast<Block>()) {
                  if (b->name.is()) {
                    breakStates[b->name].clear();
                  }
                  bool fallthroughReach = true;
                  for (auto* item : b->list) {
                    walk(item, active);
                    if (doesNotReturn(item)) {
                      fallthroughReach = false;
                      break;
                    }
                  }
                  if (b->name.is()) {
                    auto& bStates = breakStates[b->name];
                    if (!bStates.empty()) {
                      if (fallthroughReach) {
                        for (const auto& bs : bStates) {
                          intersectSet(active, bs);
                        }
                      } else {
                        active = bStates[0];
                        for (size_t i = 1; i < bStates.size(); i++) {
                          intersectSet(active, bStates[i]);
                        }
                      }
                    }
                    breakStates.erase(b->name);
                  }
                  return;
                }
                if (auto* br = curr->dynCast<Break>()) {
                  if (br->value) {
                    walk(br->value, active);
                  }
                  if (br->condition) {
                    walk(br->condition, active);
                  }
                  auto it = breakStates.find(br->name);
                  if (it != breakStates.end()) {
                    it->second.push_back(active);
                  }
                  return;
                }
                if (auto* sw = curr->dynCast<Switch>()) {
                  if (sw->value) {
                    walk(sw->value, active);
                  }
                  walk(sw->condition, active);
                  for (Name t : sw->targets) {
                    auto it = breakStates.find(t);
                    if (it != breakStates.end()) {
                      it->second.push_back(active);
                    }
                  }
                  auto it = breakStates.find(sw->default_);
                  if (it != breakStates.end()) {
                    it->second.push_back(active);
                  }
                  return;
                }
                if (auto* loop = curr->dynCast<Loop>()) {
                  auto entry = active;
                  auto bodySet = entry;
                  walk(loop->body, bodySet);
                  active = std::move(entry);
                  return;
                }
                if (auto* tryy = curr->dynCast<Try>()) {
                  auto entry = active;
                  auto bodySet = entry;
                  walk(tryy->body, bodySet);
                  for (auto* cb : tryy->catchBodies) {
                    auto cbSet = entry;
                    walk(cb, cbSet);
                  }
                  active = std::move(entry);
                  return;
                }
                if (curr->is<TryTable>()) {
                  auto entry = active;
                  for (auto* child : ChildIterator(curr)) {
                    auto childSet = entry;
                    walk(child, childSet);
                  }
                  active = std::move(entry);
                  return;
                }
                for (auto* child : ChildIterator(curr)) {
                  walk(child, active);
                }
              };

            std::unordered_set<Name> active = entryActive[func->name];
            walk(func->body, active);
          }

          if (nextHas == hasEntryState && nextEntry == entryActive) {
            break;
          }
          hasEntryState = std::move(nextHas);
          entryActive = std::move(nextEntry);
        }

        entryActiveOnce = std::move(entryActive);
      }

      bool optimizeFunction(Function* func,
                            bool interproceduralPass = false) const {
        if (!func || func->imported() || !func->body || onceFuncs.empty()) {
          return false;
        }
        bool hasOnceCall = false;
        for (auto* c : FindAll<Call>(func->body).list) {
          if (onceFuncs.contains(c->target)) {
            hasOnceCall = true;
            break;
          }
        }
        if (!hasOnceCall) {
          return false;
        }

        bool anyChanged = false;

        struct State {
          std::unordered_set<Name> activeOnce;
          std::unordered_map<Name, std::unordered_set<Call*>> activeProviders;
        };

        auto joinStates = [](const std::vector<State>& states) -> State {
          State res;
          if (states.empty()) {
            return res;
          }
          res.activeOnce = states[0].activeOnce;
          for (size_t i = 1; i < states.size(); i++) {
            for (auto it = res.activeOnce.begin();
                 it != res.activeOnce.end();) {
              if (!states[i].activeOnce.contains(*it)) {
                it = res.activeOnce.erase(it);
              } else {
                ++it;
              }
            }
          }
          for (const auto& st : states) {
            for (const auto& [g, calls] : st.activeProviders) {
              res.activeProviders[g].insert(calls.begin(), calls.end());
            }
          }
          return res;
        };

        auto runDominatorPass = [&]() -> bool {
          std::unordered_map<Name, std::vector<State>> breakStates;
          std::unordered_set<Call*> dominatedCalls;
          std::unordered_set<Call*> allSafePureCallNodes;
          std::unordered_set<Call*> neededSafePureCallNodes;

          State initialState;
          if (interproceduralPass) {
            auto enIt = entryActiveOnce.find(func->name);
            if (enIt != entryActiveOnce.end()) {
              initialState.activeOnce = enIt->second;
            }
          }
          Name selfGuard;
          if (onceFuncs.contains(func->name)) {
            auto gIt = guardGlobal.find(func->name);
            if (gIt != guardGlobal.end()) {
              selfGuard = gIt->second;
            } else {
              initialState.activeOnce.insert(func->name);
            }
          }

          std::function<void(Expression*, State&)> walk = [&](Expression* curr,
                                                              State& state) {
            if (!curr) {
              return;
            }
            if (auto* gs = curr->dynCast<GlobalSet>()) {
              walk(gs->value, state);
              if (selfGuard.is() && gs->name == selfGuard) {
                state.activeOnce.insert(func->name);
              }
              return;
            }
            if (auto* gg = curr->dynCast<GlobalGet>()) {
              if (allSafePureGlobals.contains(gg->name)) {
                auto pIt = state.activeProviders.find(gg->name);
                if (pIt != state.activeProviders.end()) {
                  neededSafePureCallNodes.insert(pIt->second.begin(),
                                                 pIt->second.end());
                }
              }
              return;
            }
            if (auto* c = curr->dynCast<Call>()) {
              for (auto* op : c->operands) {
                walk(op, state);
              }
              if (onceFuncs.contains(c->target)) {
                if (state.activeOnce.contains(c->target)) {
                  dominatedCalls.insert(c);
                } else {
                  auto exIt = exitActiveOnce.find(c->target);
                  if (exIt != exitActiveOnce.end()) {
                    state.activeOnce.insert(exIt->second.begin(),
                                            exIt->second.end());
                  } else {
                    state.activeOnce.insert(c->target);
                  }
                  if (safePureOnce.contains(c->target)) {
                    allSafePureCallNodes.insert(c);
                    auto tgIt = pureTransitiveGlobals.find(c->target);
                    if (tgIt != pureTransitiveGlobals.end()) {
                      for (Name g : tgIt->second) {
                        auto& prov = state.activeProviders[g];
                        if (prov.empty()) {
                          prov.insert(c);
                        }
                      }
                    }
                  }
                }
              } else {
                auto exIt = exitActiveOnce.find(c->target);
                if (exIt != exitActiveOnce.end()) {
                  state.activeOnce.insert(exIt->second.begin(),
                                          exIt->second.end());
                }
              }
              if (selfGuard.is() && topLevelSetsGlobal(c, selfGuard)) {
                state.activeOnce.insert(func->name);
              }
              return;
            }
            if (auto* iff = curr->dynCast<If>()) {
              walk(iff->condition, state);
              State entry = state;
              State trueState = entry;
              walk(iff->ifTrue, trueState);
              bool trueReach = !doesNotReturn(iff->ifTrue);
              if (iff->ifFalse) {
                State falseState = entry;
                walk(iff->ifFalse, falseState);
                bool falseReach = !doesNotReturn(iff->ifFalse);
                std::vector<State> branches;
                if (trueReach) {
                  branches.push_back(std::move(trueState));
                }
                if (falseReach) {
                  branches.push_back(std::move(falseState));
                }
                if (!branches.empty()) {
                  state = joinStates(branches);
                }
              } else {
                state = std::move(entry);
              }
              return;
            }
            if (auto* b = curr->dynCast<Block>()) {
              if (b->name.is()) {
                breakStates[b->name].clear();
              }
              bool fallthroughReach = true;
              for (auto* item : b->list) {
                walk(item, state);
                if (doesNotReturn(item)) {
                  fallthroughReach = false;
                  break;
                }
              }
              if (b->name.is()) {
                auto& bStates = breakStates[b->name];
                if (!bStates.empty()) {
                  std::vector<State> branches;
                  if (fallthroughReach) {
                    branches.push_back(std::move(state));
                  }
                  for (auto& bs : bStates) {
                    branches.push_back(std::move(bs));
                  }
                  state = joinStates(branches);
                }
                breakStates.erase(b->name);
              }
              return;
            }
            if (auto* br = curr->dynCast<Break>()) {
              if (br->value) {
                walk(br->value, state);
              }
              if (br->condition) {
                walk(br->condition, state);
              }
              auto it = breakStates.find(br->name);
              if (it != breakStates.end()) {
                it->second.push_back(state);
              }
              return;
            }
            if (auto* sw = curr->dynCast<Switch>()) {
              if (sw->value) {
                walk(sw->value, state);
              }
              walk(sw->condition, state);
              for (Name t : sw->targets) {
                auto it = breakStates.find(t);
                if (it != breakStates.end()) {
                  it->second.push_back(state);
                }
              }
              auto it = breakStates.find(sw->default_);
              if (it != breakStates.end()) {
                it->second.push_back(state);
              }
              return;
            }
            if (auto* loop = curr->dynCast<Loop>()) {
              State entry = state;
              State bodyState = entry;
              walk(loop->body, bodyState);
              state = std::move(entry);
              return;
            }
            if (auto* tryy = curr->dynCast<Try>()) {
              State entry = state;
              State bodyState = entry;
              walk(tryy->body, bodyState);
              for (auto* cb : tryy->catchBodies) {
                State cbState = entry;
                walk(cb, cbState);
              }
              state = std::move(entry);
              return;
            }
            if (curr->is<TryTable>()) {
              State entry = state;
              for (auto* child : ChildIterator(curr)) {
                State childState = entry;
                walk(child, childState);
              }
              state = std::move(entry);
              return;
            }
            for (auto* child : ChildIterator(curr)) {
              walk(child, state);
            }
          };

          walk(func->body, initialState);

          bool passChanged = false;
          bool removeUnusedSafePure =
            !interproceduralPass && !pureLazyOnce.contains(func->name);
          for (Call* c : dominatedCalls) {
            ExpressionManipulator::nop(c);
            passChanged = true;
          }
          if (removeUnusedSafePure) {
            for (Call* c : allSafePureCallNodes) {
              if (!neededSafePureCallNodes.contains(c) &&
                  !dominatedCalls.contains(c)) {
                ExpressionManipulator::nop(c);
                passChanged = true;
              }
            }
          }
          return passChanged;
        };

        auto cleanNops = [&]() {
          struct NopCleaner : public PostWalker<NopCleaner> {
            Module& wasm;
            const PassOptions& options;
            NopCleaner(Module& wasm, const PassOptions& options)
              : wasm(wasm), options(options) {}
            void visitBlock(Block* curr) {
              if (curr->list.size() <= 1) {
                if (curr->list.size() == 1 && curr->list[0]->is<Nop>() &&
                    !curr->name.is()) {
                  ExpressionManipulator::nop(curr);
                }
                return;
              }
              curr->list.erase(
                std::remove_if(curr->list.begin(),
                               curr->list.end(),
                               [](Expression* e) { return e->is<Nop>(); }),
                curr->list.end());
              if (curr->list.empty()) {
                if (!curr->name.is()) {
                  ExpressionManipulator::nop(curr);
                } else {
                  curr->list.push_back(Builder(wasm).makeNop());
                }
              }
            }
            void visitIf(If* curr) {
              if (curr->ifFalse && curr->ifFalse->is<Nop>()) {
                curr->ifFalse = nullptr;
              }
              if (!curr->ifFalse && curr->ifTrue->is<Nop>()) {
                EffectAnalyzer condEff(options, wasm, curr->condition);
                if (!condEff.hasSideEffects()) {
                  ExpressionManipulator::nop(curr);
                }
              }
            }
          } cleaner(wasm, options);
          cleaner.walk(func->body);
        };

        if (runDominatorPass()) {
          cleanNops();
          anyChanged = true;
        }

        if (!interproceduralPass) {
          struct IfHoister : public PostWalker<IfHoister> {
            const OnceCallOpt& parent;
            bool changed = false;
            IfHoister(const OnceCallOpt& parent) : parent(parent) {}

            Name getLeadingOnceCall(Expression* branch) const {
              Expression* curr = branch;
              while (auto* b = curr->dynCast<Block>()) {
                if (b->list.empty()) {
                  return Name();
                }
                curr = b->list[0];
              }
              if (auto* c = curr->dynCast<Call>()) {
                if (parent.onceFuncs.contains(c->target)) {
                  return c->target;
                }
              }
              return Name();
            }

            Call* popLeadingOnceCall(Expression*& branch) {
              if (auto* c = branch->dynCast<Call>()) {
                branch = Builder(parent.wasm).makeNop();
                return c;
              }
              auto* b = branch->cast<Block>();
              while (b->list[0]->is<Block>()) {
                b = b->list[0]->cast<Block>();
              }
              auto* c = b->list[0]->cast<Call>();
              if (b->list.size() == 1) {
                b->list[0] = Builder(parent.wasm).makeNop();
              } else {
                b->list.erase(b->list.begin());
              }
              return c;
            }

            void visitIf(If* curr) {
              if (!curr->ifFalse) {
                return;
              }
              std::vector<Call*> hoistedCalls;
              while (true) {
                Name t1 = getLeadingOnceCall(curr->ifTrue);
                if (!t1.is()) {
                  break;
                }
                Name t2 = getLeadingOnceCall(curr->ifFalse);
                if (t1 != t2) {
                  break;
                }
                EffectAnalyzer condEff(
                  parent.options, parent.wasm, curr->condition);
                if (condEff.transfersControlFlow() || condEff.trap ||
                    condEff.throws() || !condEff.globalsWritten.empty() ||
                    condEff.writesMemory || condEff.writesTable ||
                    condEff.writesStruct || condEff.writesArray) {
                  break;
                }
                if (condEff.calls &&
                    !parent.canHoistOverCallsInCond(curr->condition, t1)) {
                  break;
                }
                bool readsWrittenGlobal = false;
                auto wIt = parent.onceWrittenGlobals.find(t1);
                if (wIt != parent.onceWrittenGlobals.end()) {
                  for (const auto& g : condEff.mutableGlobalsRead) {
                    if (wIt->second.contains(g)) {
                      readsWrittenGlobal = true;
                      break;
                    }
                  }
                }
                if (readsWrittenGlobal) {
                  break;
                }
                Call* keptCall = popLeadingOnceCall(curr->ifTrue);
                popLeadingOnceCall(curr->ifFalse);
                hoistedCalls.push_back(keptCall);
              }
              if (!hoistedCalls.empty()) {
                Builder builder(parent.wasm);
                auto* seq = builder.makeBlock();
                for (Call* c : hoistedCalls) {
                  seq->list.push_back(c);
                }
                seq->list.push_back(curr);
                seq->finalize(curr->type);
                replaceCurrent(seq);
                changed = true;
              }
            }
          } hoister(*this);
          hoister.walk(func->body);
          if (hoister.changed) {
            anyChanged = true;
            runDominatorPass();
            cleanNops();
          }
        }

        if (anyChanged) {
          ReFinalize().walkFunctionInModule(func, &wasm);
        }

        return anyChanged;
      }
    };

    OnceCallOpt opt(*module, getPassOptions());
    opt.analyze();
    if (!opt.onceFuncs.empty()) {
      for (auto& func : module->functions) {
        if (!func->imported()) {
          opt.optimizeFunction(func.get(), false);
        }
      }
      opt.computeEntryActiveOnce();
      for (auto& func : module->functions) {
        if (!func->imported()) {
          opt.optimizeFunction(func.get(), true);
        }
      }
    }
  }

  void foldArrayGlobals() {
    std::unordered_set<Name> nonCodeUses;
    std::unordered_map<Name, Index> initGlobalUses;
    {
      struct Scanner : public PostWalker<Scanner> {
        std::unordered_set<Name>* nonCode = nullptr;
        std::unordered_map<Name, Index>* initCounts = nullptr;
        void visitGlobalGet(GlobalGet* curr) {
          if (initCounts) {
            (*initCounts)[curr->name]++;
          }
          if (nonCode) {
            nonCode->insert(curr->name);
          }
        }
      } scanner;
      scanner.initCounts = &initGlobalUses;
      for (auto& g : module->globals) {
        if (!g->imported()) {
          scanner.walk(g->init);
        }
      }
      scanner.initCounts = nullptr;
      scanner.nonCode = &nonCodeUses;
      for (auto& seg : module->elementSegments) {
        if (seg->offset) {
          scanner.walk(seg->offset);
        }
        for (auto* item : seg->data) {
          scanner.walk(item);
        }
      }
      for (auto& seg : module->dataSegments) {
        if (seg->offset) {
          scanner.walk(seg->offset);
        }
      }
      for (auto& ex : module->exports) {
        if (ex->kind == ExternalKind::Global) {
          nonCodeUses.insert(*ex->getInternalName());
        }
      }
    }

    struct CodeScanInfo {
      std::unordered_map<Name, Index> gets;
      std::unordered_map<Name, std::vector<Expression**>> arrayGets;
    };
    ModuleUtils::ParallelFunctionAnalysis<CodeScanInfo> codeScan(
      *module, [&](Function* func, CodeScanInfo& info) {
        if (func->imported()) {
          return;
        }
        struct Walker : public PostWalker<Walker> {
          CodeScanInfo& info;
          Walker(CodeScanInfo& info) : info(info) {}
          void visitGlobalGet(GlobalGet* curr) { info.gets[curr->name]++; }
          void visitArrayGet(ArrayGet* curr) {
            if (curr->order != MemoryOrder::Unordered) {
              return;
            }
            if (auto* gg = curr->ref->dynCast<GlobalGet>()) {
              info.arrayGets[gg->name].push_back(getCurrentPointer());
            }
          }
        };
        Walker(info).walk(func->body);
      });

    std::unordered_map<Name, Index> codeGlobalGets;
    std::unordered_map<Name, std::vector<std::pair<Function*, Expression**>>>
      arrayGetSites;
    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      auto& info = codeScan.map[func.get()];
      for (auto& [name, count] : info.gets) {
        codeGlobalGets[name] += count;
      }
      for (auto& [name, sites] : info.arrayGets) {
        for (auto** site : sites) {
          arrayGetSites[name].push_back({func.get(), site});
        }
      }
    }

    computeIndexes();
    Builder builder(*module);
    std::unordered_set<Function*> refinalizeFuncs;
    std::vector<Name> globalsToRemove;

    std::function<bool(Expression*, bool)> isFoldableElem =
      [&](Expression* e, bool allowStructAlloc) -> bool {
      if (e->is<Const>() || e->is<RefNull>() || e->is<RefFunc>()) {
        return true;
      }
      if (auto* gg = e->dynCast<GlobalGet>()) {
        auto* g = module->getGlobalOrNull(gg->name);
        return g && !g->mutable_;
      }
      if (auto* i31 = e->dynCast<RefI31>()) {
        return i31->value->is<Const>();
      }
      if (allowStructAlloc) {
        if (auto* sn = e->dynCast<StructNew>()) {
          if (sn->desc || sn->type == Type::unreachable) {
            return false;
          }
          for (auto* op : sn->operands) {
            if (!isFoldableElem(op, true)) {
              return false;
            }
          }
          return true;
        }
      }
      return false;
    };

    auto isCachedFactoryFunc = [&](Function* fn) -> bool {
      if (!fn || fn->imported() || !fn->body || fn->getNumParams() != 1 ||
          fn->getParams() != Type::i32) {
        return false;
      }
      auto* b = fn->body->dynCast<Block>();
      if (!b || b->list.size() < 2) {
        return false;
      }
      auto* brOn = b->list[0]->dynCast<BrOn>();
      if (!brOn || brOn->op != BrOnNonNull || brOn->name != b->name) {
        return false;
      }
      auto* ag = brOn->ref->dynCast<ArrayGet>();
      if (!ag || !ag->ref->is<GlobalGet>()) {
        return false;
      }
      auto* lg = ag->index->dynCast<LocalGet>();
      return lg && lg->index == 0;
    };

    auto resolveConstElem = [&](Expression* e) -> Const* {
      if (auto* c = e->dynCast<Const>()) {
        return c;
      }
      if (auto* gg = e->dynCast<GlobalGet>()) {
        auto* cg = module->getGlobalOrNull(gg->name);
        if (cg && !cg->imported() && !cg->mutable_ && cg->init) {
          if (auto* c = cg->init->dynCast<Const>()) {
            return c;
          }
        }
      }
      return nullptr;
    };

    auto sameElem = [&](Expression* a,
                        Expression* b,
                        const Field* packedField = nullptr) -> bool {
      auto* ca = resolveConstElem(a);
      auto* cb = resolveConstElem(b);
      if (ca && cb) {
        if (packedField && packedField->isPacked() &&
            ca->type == Type::i32 && cb->type == Type::i32) {
          uint32_t mask =
            (packedField->packedType == Field::i8) ? 0xffu : 0xffffu;
          return (uint32_t(ca->value.geti32()) & mask) ==
                 (uint32_t(cb->value.geti32()) & mask);
        }
        return ca->value == cb->value;
      }
      return sameRefLeaf(a, b);
    };

    // Prefix/subarray deduplication of read-only array.new_fixed globals.
    std::vector<Global*> roArrayGlobals;
    for (auto& g : module->globals) {
      if (g->imported() || g->mutable_ || nonCodeUses.count(g->name) ||
          initGlobalUses[g->name] > 0) {
        continue;
      }
      if (!onceOnly && paletteSlices.count(g->name)) {
        continue;
      }
      auto* anf = g->init->dynCast<ArrayNewFixed>();
      if (!anf || anf->values.empty() || !g->type.isRef() ||
          g->type.isNullable()) {
        continue;
      }
      HeapType arrHt = g->type.getHeapType();
      if (!arrHt.isArray() || arrHt.isShared()) {
        continue;
      }
      Index totalGets = codeGlobalGets[g->name];
      auto& sites = arrayGetSites[g->name];
      if (totalGets == 0 || totalGets != sites.size()) {
        continue;
      }
      roArrayGlobals.push_back(g.get());
    }
    std::unordered_set<Name> prefixMergedGlobals;
    for (size_t i = 0; i < roArrayGlobals.size(); i++) {
      Global* g1 = roArrayGlobals[i];
      if (prefixMergedGlobals.count(g1->name)) {
        continue;
      }
      auto* anf1 = g1->init->cast<ArrayNewFixed>();
      Field elemField1 = g1->type.getHeapType().getArray().element;
      Index n1 = anf1->values.size();
      Global* bestG2 = nullptr;
      Index bestN2 = 0;
      for (size_t j = 0; j < roArrayGlobals.size(); j++) {
        if (i == j) {
          continue;
        }
        Global* g2 = roArrayGlobals[j];
        if (prefixMergedGlobals.count(g2->name) || g1->type != g2->type) {
          continue;
        }
        auto* anf2 = g2->init->cast<ArrayNewFixed>();
        Index n2 = anf2->values.size();
        if (n1 > n2 || (n1 == n2 && j > i)) {
          continue;
        }
        bool isPrefix = true;
        for (Index k = 0; k < n1; k++) {
          if (!sameElem(anf1->values[k], anf2->values[k], &elemField1)) {
            isPrefix = false;
            break;
          }
        }
        if (isPrefix && (!bestG2 || n2 > bestN2)) {
          bestG2 = g2;
          bestN2 = n2;
        }
      }
      if (bestG2) {
        auto& sites1 = arrayGetSites[g1->name];
        auto& sites2 = arrayGetSites[bestG2->name];
        for (auto& [fn, site] : sites1) {
          auto* ag = (*site)->cast<ArrayGet>();
          ag->ref->cast<GlobalGet>()->name = bestG2->name;
          sites2.push_back({fn, site});
        }
        codeGlobalGets[bestG2->name] += codeGlobalGets[g1->name];
        codeGlobalGets[g1->name] = 0;
        sites1.clear();
        prefixMergedGlobals.insert(g1->name);
        globalsToRemove.push_back(g1->name);
      }
    }

    struct PendingFuncI32Array {
      Global* global;
      Index N;
      Index globalCost;
      int64_t minVal;
      int64_t maxVal;
      std::vector<int32_t> vals;
      std::vector<std::pair<Function*, Expression**>> sites;
    };
    InsertOrderedMap<Function*, std::vector<PendingFuncI32Array>>
      funcI32Arrays;

    for (auto& g : module->globals) {
      if (g->imported() || g->mutable_ || nonCodeUses.count(g->name) ||
          initGlobalUses[g->name] > 0 || prefixMergedGlobals.count(g->name)) {
        continue;
      }
      if (!onceOnly && paletteSlices.count(g->name)) {
        continue;
      }
      auto* anf = g->init->dynCast<ArrayNewFixed>();
      if (!anf || anf->values.empty() || !g->type.isRef() ||
          g->type.isNullable()) {
        continue;
      }
      HeapType arrHt = g->type.getHeapType();
      if (!arrHt.isArray() || arrHt.isShared()) {
        continue;
      }
      Field elemField = arrHt.getArray().element;
      Index totalGets = codeGlobalGets[g->name];
      auto& sites = arrayGetSites[g->name];
      if (totalGets == 0 || totalGets != sites.size()) {
        continue;
      }
      bool canFoldStructAllocs =
        sites.size() == 1 && isCachedFactoryFunc(sites[0].first);
      bool allElemsValid = true;
      bool hasStructAllocElem = false;
      for (auto* v : anf->values) {
        if (!isFoldableElem(v, canFoldStructAllocs)) {
          allElemsValid = false;
          break;
        }
        if (v->is<StructNew>()) {
          hasStructAllocElem = true;
        }
      }
      if (!allElemsValid) {
        continue;
      }
      bool consistentSigned = true;
      bool isSigned =
        sites[0].second ? (*sites[0].second)->cast<ArrayGet>()->signed_ : false;
      Type resType = (*sites[0].second)->type;
      for (auto& [_, site] : sites) {
        auto* ag = (*site)->cast<ArrayGet>();
        if (ag->signed_ != isSigned || ag->type != resType ||
            ag->index->type != Type::i32) {
          consistentSigned = false;
          break;
        }
      }
      if (!consistentSigned) {
        continue;
      }

      Index N = anf->values.size();
      std::vector<Expression*> normVals;
      normVals.reserve(N);
      for (auto* v : anf->values) {
        if (elemField.isPacked()) {
          if (auto* c = resolveConstElem(v)) {
            int32_t raw = c->value.geti32();
            int32_t norm = 0;
            if (elemField.packedType == Field::i8) {
              norm = isSigned ? int32_t(int8_t(uint8_t(raw)))
                              : int32_t(uint8_t(raw));
            } else {
              norm = isSigned ? int32_t(int16_t(uint16_t(raw)))
                              : int32_t(uint16_t(raw));
            }
            normVals.push_back(builder.makeConst(norm));
            continue;
          }
        }
        normVals.push_back(v);
      }

      Index globalCost = 9 + getLEBSize(U32LEB(N));
      for (auto* v : anf->values) {
        globalCost += getOperandBytes(v);
      }
      Index arrayGetOpCost =
        4 + 1 + getLEBSize(U32LEB(globalIndexes[g->name]));

      // Constant-index array.get folding on read-only array.new_fixed globals.
      {
        bool allConstIdx = true;
        Index constCostBefore = globalCost;
        Index constCostAfter = 0;
        for (auto& [_, site] : sites) {
          auto* ag = (*site)->cast<ArrayGet>();
          auto* idxC = resolveConstElem(ag->index);
          if (!idxC || uint32_t(idxC->value.geti32()) >= N) {
            allConstIdx = false;
            break;
          }
          Index k = uint32_t(idxC->value.geti32());
          constCostBefore += arrayGetOpCost + getOperandBytes(ag->index);
          constCostAfter += getOperandBytes(normVals[k]);
        }
        if (allConstIdx && constCostAfter < constCostBefore) {
          for (auto& [fn, site] : sites) {
            auto* ag = (*site)->cast<ArrayGet>();
            Index k = uint32_t(resolveConstElem(ag->index)->value.geti32());
            *site = ExpressionManipulator::copy(normVals[k], *module);
            refinalizeFuncs.insert(fn);
          }
          globalsToRemove.push_back(g->name);
          continue;
        }
        std::vector<std::pair<Function*, Expression**>> remainingSites;
        remainingSites.reserve(sites.size());
        for (auto& [fn, site] : sites) {
          auto* ag = (*site)->cast<ArrayGet>();
          auto* idxC = resolveConstElem(ag->index);
          if (idxC && uint32_t(idxC->value.geti32()) < N) {
            Index k = uint32_t(idxC->value.geti32());
            if (getOperandBytes(normVals[k]) <=
                arrayGetOpCost + getOperandBytes(ag->index)) {
              *site = ExpressionManipulator::copy(normVals[k], *module);
              refinalizeFuncs.insert(fn);
              continue;
            }
          }
          remainingSites.push_back({fn, site});
        }
        if (remainingSites.empty()) {
          globalsToRemove.push_back(g->name);
          continue;
        }
        sites = std::move(remainingSites);
      }

      std::vector<Expression*> distinct;
      std::vector<std::vector<Index>> distinctIndices;
      for (Index k = 0; k < N; k++) {
        Expression* v = normVals[k];
        Index dIdx = 0;
        while (dIdx < distinct.size() && !sameElem(distinct[dIdx], v)) {
          dIdx++;
        }
        if (dIdx == distinct.size()) {
          distinct.push_back(v);
          distinctIndices.push_back({k});
        } else {
          if (getOperandBytes(v) < getOperandBytes(distinct[dIdx])) {
            distinct[dIdx] = v;
          }
          distinctIndices[dIdx].push_back(k);
        }
      }

      Index D = distinct.size();
      Index costBefore = globalCost + sites.size() * arrayGetOpCost;
      Index selectOpBytes = resType.isRef() ? 4 : 1;

      if (D == 1) {
        Expression* val0 = distinct[0];
        Index valBytes = getOperandBytes(val0);
        Index costAfter = 0;
        for (auto& [_, site] : sites) {
          auto* ag = (*site)->cast<ArrayGet>();
          bool needDrop =
            EffectAnalyzer(getPassOptions(), *module, ag->index)
              .hasUnremovableSideEffects();
          costAfter += valBytes + (needDrop ? 1 : 0);
        }
        if (costAfter < costBefore) {
          for (auto& [fn, site] : sites) {
            auto* ag = (*site)->cast<ArrayGet>();
            Expression* repl = ExpressionManipulator::copy(val0, *module);
            if (EffectAnalyzer(getPassOptions(), *module, ag->index)
                  .hasUnremovableSideEffects()) {
              *site = builder.makeSequence(builder.makeDrop(ag->index), repl);
            } else {
              *site = repl;
            }
            refinalizeFuncs.insert(fn);
          }
          globalsToRemove.push_back(g->name);
        }
        continue;
      }

      // Arithmetic-progression and bit-packed i32 array folding.
      bool allI32Consts = false;
      std::vector<int32_t> i32Vals;
      int64_t minI32Val = std::numeric_limits<int64_t>::max();
      int64_t maxI32Val = std::numeric_limits<int64_t>::min();
      if (resType == Type::i32 && N >= 3) {
        allI32Consts = true;
        i32Vals.resize(N);
        for (Index k = 0; k < N; k++) {
          auto* c = resolveConstElem(normVals[k]);
          if (!c || c->type != Type::i32) {
            allI32Consts = false;
            break;
          }
          i32Vals[k] = c->value.geti32();
          minI32Val = std::min(minI32Val, int64_t(i32Vals[k]));
          maxI32Val = std::max(maxI32Val, int64_t(i32Vals[k]));
        }
        if (allI32Consts) {
          int32_t step = int32_t(uint32_t(i32Vals[1]) - uint32_t(i32Vals[0]));
          bool isArithProg = (step == 1 || step == -1);
          for (Index k = 2; isArithProg && k < N; k++) {
            if (int32_t(uint32_t(i32Vals[k]) - uint32_t(i32Vals[k - 1])) !=
                step) {
              isArithProg = false;
            }
          }
          if (isArithProg) {
            int32_t base = i32Vals[0];
            Index extraBytes =
              (step == 1 && base == 0) ? 0 : (2 + getSLEBSize(Index(base)));
            Index costAfter = sites.size() * extraBytes;
            if (costAfter < costBefore) {
              for (auto& [fn, site] : sites) {
                auto* ag = (*site)->cast<ArrayGet>();
                if (step == 1) {
                  *site = (base == 0)
                            ? ag->index
                            : static_cast<Expression*>(builder.makeBinary(
                                AddInt32, ag->index, builder.makeConst(base)));
                } else {
                  *site = builder.makeBinary(
                    SubInt32, builder.makeConst(base), ag->index);
                }
                refinalizeFuncs.insert(fn);
              }
              globalsToRemove.push_back(g->name);
              continue;
            }
          }

          if (N >= 8 && N <= 64) {
            Index chosenBits = 0;
            int32_t offset = 0;
            uint64_t span = uint64_t(maxI32Val - minI32Val);
            for (Index b : {Index(1), Index(2), Index(4), Index(8)}) {
              if (N * b > 64) {
                break;
              }
              uint64_t limit = 1ULL << b;
              if (minI32Val >= 0 && uint64_t(maxI32Val) < limit) {
                chosenBits = b;
                offset = 0;
                break;
              }
              if (span < limit) {
                chosenBits = b;
                offset = int32_t(minI32Val);
                break;
              }
            }
            if (chosenBits != 0) {
              uint64_t packed = 0;
              for (Index k = 0; k < N; k++) {
                uint64_t lane = uint64_t(int64_t(i32Vals[k]) - int64_t(offset));
                packed |= (lane << (k * chosenBits));
              }
              bool useI32 = (N * chosenBits <= 32);
              Index shiftIdxBytes = (chosenBits == 1) ? 0 : 3;
              Index constAndShrBytes =
                useI32 ? (1 + getLEBSize(S32LEB(int32_t(uint32_t(packed)))) + 1)
                       : (1 + getLEBSize(S64LEB(int64_t(packed))) + 3);
              int32_t maskVal = int32_t((1u << chosenBits) - 1u);
              Index maskBytes = 1 + getLEBSize(S32LEB(maskVal)) + 1;
              Index offsetBytes =
                (offset == 0) ? 0 : (1 + getLEBSize(S32LEB(offset)) + 1);
              Index siteExtra =
                shiftIdxBytes + constAndShrBytes + maskBytes + offsetBytes;
              if (sites.size() * siteExtra < costBefore) {
                Index shlAmt =
                  (chosenBits == 2) ? 1 : ((chosenBits == 4) ? 2 : 3);
                for (auto& [fn, site] : sites) {
                  auto* ag = (*site)->cast<ArrayGet>();
                  Expression* shiftExpr = ag->index;
                  if (chosenBits > 1) {
                    shiftExpr = builder.makeBinary(
                      ShlInt32,
                      shiftExpr,
                      builder.makeConst(int32_t(shlAmt)));
                  }
                  Expression* shifted = nullptr;
                  if (useI32) {
                    shifted = builder.makeBinary(
                      ShrUInt32,
                      builder.makeConst(int32_t(uint32_t(packed))),
                      shiftExpr);
                  } else {
                    shifted = builder.makeUnary(
                      WrapInt64,
                      builder.makeBinary(
                        ShrUInt64,
                        builder.makeConst(int64_t(packed)),
                        builder.makeUnary(ExtendUInt32, shiftExpr)));
                  }
                  Expression* res = builder.makeBinary(
                    AndInt32, shifted, builder.makeConst(maskVal));
                  if (offset != 0) {
                    res = builder.makeBinary(
                      AddInt32, res, builder.makeConst(offset));
                  }
                  *site = res;
                  refinalizeFuncs.insert(fn);
                }
                globalsToRemove.push_back(g->name);
                continue;
              }
            }
          }
        }
      }

      bool isExternRefElem =
        elemField.type.isRef() &&
        elemField.type.getHeapType().getBottom() == HeapType::noext;
      if (D == 2 || (D >= 3 && D <= 5 && (D < N || !isExternRefElem))) {
        Index defaultIdx = 0;
        std::vector<std::pair<Index, SelectArmPlan>> steps;
        if (planSelectChainSteps(
              distinct, distinctIndices, N, defaultIdx, steps)) {
          Expression* vDef = distinct[defaultIdx];
          Index armOpBytes =
            hasStructAllocElem ? (selectOpBytes + 1) : selectOpBytes;
          Index baseSiteCost = getOperandBytes(vDef);
          for (auto& [d, p] : steps) {
            baseSiteCost +=
              getOperandBytes(distinct[d]) + p.condExtra + armOpBytes;
          }
          Index M = steps.size();
          Index costAfter = 0;
          for (auto& [_, site] : sites) {
            auto* ag = (*site)->cast<ArrayGet>();
            Index idxExtra =
              (M <= 1) ? 0
                       : (ag->index->is<LocalGet>() ? (2 * (M - 1))
                                                    : (4 + 2 * (M - 1)));
            costAfter += baseSiteCost + idxExtra;
          }
          if (costAfter <= costBefore) {
            for (auto& [fn, site] : sites) {
              auto* ag = (*site)->cast<ArrayGet>();
              Index scratchLoc = Index(-1);
              if (M > 1 && !ag->index->is<LocalGet>()) {
                scratchLoc = Builder::addVar(fn, Type::i32);
              }
              Index teeStep = hasStructAllocElem ? (M - 1) : 0;
              Expression* acc = ExpressionManipulator::copy(vDef, *module);
              for (Index m = 0; m < M; m++) {
                Index d = steps[m].first;
                const SelectArmPlan& p = steps[m].second;
                Expression* idxExpr = nullptr;
                if (M == 1 || ag->index->is<LocalGet>()) {
                  idxExpr =
                    (m == 0) ? ag->index
                             : ExpressionManipulator::copy(ag->index, *module);
                } else {
                  idxExpr =
                    (m == teeStep)
                      ? static_cast<Expression*>(builder.makeLocalTee(
                          scratchLoc, ag->index, Type::i32))
                      : static_cast<Expression*>(
                          builder.makeLocalGet(scratchLoc, Type::i32));
                }
                Expression* cond = buildSelectCond(builder, p, idxExpr);
                Expression* trueExpr =
                  ExpressionManipulator::copy(distinct[d], *module);
                if (hasStructAllocElem) {
                  acc = builder.makeIf(cond, trueExpr, acc, resType);
                } else {
                  acc = builder.makeSelect(cond, trueExpr, acc);
                }
              }
              *site = acc;
              refinalizeFuncs.insert(fn);
            }
            globalsToRemove.push_back(g->name);
            continue;
          }
        }
      }

      // Affine-shift index compaction ((idx + c) >> s) for low-cardinality
      // read-only array.new_fixed globals.
      if (!hasStructAllocElem && D < N) {
        Index chosenS = 0;
        Index chosenC = 0;
        for (Index s : {Index(1), Index(2)}) {
          Index maxC = (1u << s) - 1u;
          for (Index c = 0; c <= maxC; c++) {
            if (((N - 1 + c) >> s) + 1 != D) {
              continue;
            }
            bool match = true;
            for (Index k = 0; k < N; k++) {
              Index mapped = (k + c) >> s;
              if (mapped >= D ||
                  !sameElem(normVals[k], distinct[mapped], &elemField)) {
                match = false;
                break;
              }
            }
            if (match) {
              chosenS = s;
              chosenC = c;
              break;
            }
          }
          if (chosenS != 0) {
            break;
          }
        }
        if (chosenS != 0) {
          Index shrunkCost = 9 + getLEBSize(U32LEB(D));
          for (auto* expr : distinct) {
            shrunkCost += getOperandBytes(expr);
          }
          Index siteExtra = (chosenC == 0) ? 3 : 6;
          if (shrunkCost + sites.size() * siteExtra < globalCost) {
            std::vector<Expression*> newVals;
            newVals.reserve(D);
            for (auto* expr : distinct) {
              newVals.push_back(ExpressionManipulator::copy(expr, *module));
            }
            anf->values.set(newVals);
            anf->finalize();
            for (auto& [fn, site] : sites) {
              auto* ag = (*site)->cast<ArrayGet>();
              Expression* idx = ag->index;
              if (chosenC != 0) {
                idx = builder.makeBinary(
                  AddInt32, idx, builder.makeConst(int32_t(chosenC)));
              }
              ag->index = builder.makeBinary(
                ShrUInt32, idx, builder.makeConst(int32_t(chosenS)));
              refinalizeFuncs.insert(fn);
            }
            continue;
          }
        }
      }

      // Reference palette compression into segmentData for low-cardinality
      // read-only reference array.new_fixed globals.
      if (!onceOnly && !preserveThunkTypes && !segmentData.empty() &&
          module->features.hasBulkMemory() && !hasStructAllocElem &&
          isPaletteCandidate(elemField) && D >= 6 && D <= 256 && D < N) {
        Index palCost = 9 + getLEBSize(U32LEB(D));
        for (auto* expr : distinct) {
          palCost += getOperandBytes(expr);
        }
        if (palCost + N + 8 * sites.size() < globalCost) {
          std::vector<uint8_t> indices(N);
          for (Index d = 0; d < D; d++) {
            for (Index k : distinctIndices[d]) {
              indices[k] = uint8_t(d);
            }
          }
          std::vector<Expression*> newVals;
          newVals.reserve(D);
          for (auto* expr : distinct) {
            newVals.push_back(ExpressionManipulator::copy(expr, *module));
          }
          anf->values.set(newVals);
          anf->finalize();
          Index startOffset = segmentData.size();
          for (uint8_t idx : indices) {
            segmentData.push_back(char(idx));
          }
          auto i8ArrType = HeapTypes::getMutI8Array();
          for (auto& [fn, site] : sites) {
            auto* ag = (*site)->cast<ArrayGet>();
            auto* slice =
              builder.makeArrayNewData(i8ArrType,
                                       segmentName,
                                       builder.makeConst(int32_t(startOffset)),
                                       builder.makeConst(int32_t(N)));
            ag->index = builder.makeArrayGet(
              slice, ag->index, MemoryOrder::Unordered, Type::i32, false);
            refinalizeFuncs.insert(fn);
          }
          continue;
        }
      }

      // Collect remaining single-function i32 constant array.new_fixed globals
      // for grouped lazy-init data segment packing.
      if (!onceOnly && !preserveThunkTypes && !segmentData.empty() &&
          module->features.hasBulkMemory() && allI32Consts && N >= 16) {
        Function* commonFn = sites[0].first;
        bool sameFn = true;
        for (auto& [fn, _] : sites) {
          if (fn != commonFn) {
            sameFn = false;
            break;
          }
        }
        if (sameFn && commonFn) {
          funcI32Arrays[commonFn].push_back(
            {g.get(), N, globalCost, minI32Val, maxI32Val, i32Vals, sites});
        }
      }
    }

    for (auto& [fn, arrs] : funcI32Arrays) {
      Index totalGlobalCost = 0;
      int64_t groupMin = std::numeric_limits<int64_t>::max();
      int64_t groupMax = std::numeric_limits<int64_t>::min();
      for (auto& item : arrs) {
        totalGlobalCost += item.globalCost;
        groupMin = std::min(groupMin, item.minVal);
        groupMax = std::max(groupMax, item.maxVal);
      }
      if (totalGlobalCost < 80) {
        continue;
      }
      Index w = 4;
      bool dummySigned = false;
      HeapType arrType;
      classifyI32Range(groupMin, groupMax, w, dummySigned, arrType);
      if (w == 1) {
        Index startOffset = segmentData.size();
        Index curElemOffset = 0;
        for (auto& item : arrs) {
          for (int32_t v : item.vals) {
            segmentData.push_back(char(uint8_t(v)));
          }
          for (auto& [_, site] : item.sites) {
            auto* ag = (*site)->cast<ArrayGet>();
            auto* slice = builder.makeArrayNewData(
              arrType,
              segmentName,
              builder.makeConst(int32_t(startOffset + curElemOffset)),
              builder.makeConst(int32_t(item.N)));
            *site = builder.makeArrayGet(slice,
                                         ag->index,
                                         MemoryOrder::Unordered,
                                         Type::i32,
                                         item.minVal < 0);
          }
          curElemOffset += item.N;
          globalsToRemove.push_back(item.global->name);
        }
        refinalizeFuncs.insert(fn);
        continue;
      }
      Name tableName = Names::getValidGlobalName(
        *module, "defer$fn$" + fn->name.toString() + "$table");
      Type tableGlobalType(arrType, Nullable);
      module->addGlobal(builder.makeGlobal(tableName,
                                           tableGlobalType,
                                           builder.makeRefNull(arrType),
                                           Builder::Mutable));
      Index startOffset = segmentData.size();
      Index curElemOffset = 0;
      for (auto& item : arrs) {
        for (int32_t v : item.vals) {
          appendLE(segmentData, uint32_t(v), w);
        }
        for (auto& [_, site] : item.sites) {
          auto* ag = (*site)->cast<ArrayGet>();
          Expression* idx = ag->index;
          if (curElemOffset != 0) {
            idx = builder.makeBinary(
              AddInt32, idx, builder.makeConst(int32_t(curElemOffset)));
          }
          *site = builder.makeArrayGet(
            builder.makeGlobalGet(tableName, tableGlobalType),
            idx,
            MemoryOrder::Unordered,
            Type::i32,
            w == 2 ? (item.minVal < 0) : false);
        }
        curElemOffset += item.N;
        globalsToRemove.push_back(item.global->name);
      }
      Expression* initIf = builder.makeIf(
        builder.makeRefIsNull(
          builder.makeGlobalGet(tableName, tableGlobalType)),
        builder.makeGlobalSet(
          tableName,
          builder.makeArrayNewData(arrType,
                                   segmentName,
                                   builder.makeConst(int32_t(startOffset)),
                                   builder.makeConst(int32_t(curElemOffset)))));
      fn->body = builder.makeSequence(initIf, fn->body, fn->body->type);
      refinalizeFuncs.insert(fn);
    }

    for (auto* fn : refinalizeFuncs) {
      ReFinalize().walkFunctionInModule(fn, module);
    }
    for (Name gname : globalsToRemove) {
      module->removeGlobal(gname);
    }

    for (auto& g : module->globals) {
      if (g->imported() || !g->init) {
        continue;
      }
      for (auto** ptr : FindAllPointers<ArrayNewFixed>(g->init).list) {
        auto* anf = (*ptr)->cast<ArrayNewFixed>();
        if (anf->values.empty() || !anf->type.isRef() ||
            !anf->type.getHeapType().isArray()) {
          continue;
        }
        Field elemField = anf->type.getHeapType().getArray().element;
        if (!elemField.type.isDefaultable()) {
          continue;
        }
        bool allDefault = std::all_of(
          anf->values.begin(), anf->values.end(), [&](Expression* v) {
            if (v->is<RefNull>()) {
              return true;
            }
            if (auto* c = resolveConstElem(v)) {
              if (c->value == Literal::makeZero(c->type)) {
                return true;
              }
              if (c->type == Type::i32 && elemField.isPacked() &&
                  getPackedOrI32Value(elemField, c->value.geti32()) == 0) {
                return true;
              }
            }
            return false;
          });
        if (allDefault) {
          *ptr =
            builder.makeArrayNew(anf->type.getHeapType(),
                                 builder.makeConst(int32_t(anf->values.size())));
        }
      }
    }
  }
};

} // anonymous namespace

Pass* createDeferGlobalAllocationsPass() {
  return new DeferGlobalAllocations();
}

} // namespace wasm
