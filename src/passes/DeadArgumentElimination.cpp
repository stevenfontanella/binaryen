/*
 * Copyright 2018 WebAssembly Community Group participants
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
// Optimizes call arguments in a whole-program manner. In particular, this
// removes ones that are not used (dead), but it also does more things:
//
//  * Find functions for whom an argument is always passed the same
//    constant. If so, we can just set that local to that constant
//    in the function.
//  * Find functions that don't use the value passed to an argument.
//    If so, we can avoid even sending and receiving it. (Note how if
//    the previous point was true for an argument, then the second
//    must as well.)
//  * Find return values ("return arguments" ;) that are never used.
//  * Refine the types of arguments, that is make the argument type more
//    specific if all the passed values allow that.
//
// This pass does not depend on flattening, but it may be more effective,
// as then call arguments never have side effects (which we need to
// watch for here).
//

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "ir/branch-utils.h"
#include "ir/lubs.h"
#include "ir/manipulation.h"
#include "ir/names.h"
#include "ir/return-utils.h"
#include "ir/type-updating.h"
#include "ir/utils.h"
#include "param-utils.h"
#include "pass.h"
#include "passes/opt-utils.h"
#include "passes/pass-utils.h"
#include "support/sorted_vector.h"
#include "wasm-builder.h"
#include "wasm.h"

#ifndef DAE_DEBUG
#define DAE_DEBUG 0
#endif

#ifndef DAE_STATS
#define DAE_STATS 0
#endif

#if DAE_STATS

#include <iostream>

#endif // DAE_STATS

namespace wasm {

// Information for a function
struct DAEFunctionInfo {
  // Whether this needs to be recomputed. This begins as true for the first
  // computation, and we reset it every time we touch the function.
  bool stale = true;
  // The unused parameters, if any.
  SortedVector<Index> unusedParams;
  // Maps a function name to the calls going to it.
  std::unordered_map<Name, std::vector<Call*>> calls;
  // Map of all calls that are dropped, to their drops' locations (so that
  // if we can optimize out the drop, we can replace the drop there).
  std::unordered_map<Call*, Expression**> droppedCalls;
  // Whether this function contains any tail calls (including indirect tail
  // calls) and the set of functions this function tail calls. Tail-callers and
  // tail-callees cannot have their dropped returns removed because of the
  // constraint that tail-callees must have the same return type as
  // tail-callers. Indirectly tail called functions are already not optimized
  // because being in a table inhibits DAE. TODO: Allow the removal of dropped
  // returns from tail-callers if their tail-callees can have their returns
  // removed as well.
  bool hasTailCalls = false;
  std::unordered_set<Name> tailCallees;
  // The set of functions that have calls from places that limit what we can do.
  // For now, any call we don't see inhibits our optimizations, but TODO: an
  // export could be worked around by exporting a thunk that adds the parameter.
  //
  // This is built up in parallel in each function, and combined at the end.
  std::unordered_set<Name> hasUnseenCalls;

  // Clears all data, which marks us as stale and in need of recomputation.
  void clear() { *this = DAEFunctionInfo(); }

  void markStale() { stale = true; }
};

using DAEFunctionInfoMap = std::unordered_map<Name, DAEFunctionInfo>;

struct DAEScanner
  : public WalkerPass<PostWalker<DAEScanner, Visitor<DAEScanner>>> {
  bool isFunctionParallel() override { return true; }
  bool modifiesBinaryenIR() override { return false; }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<DAEScanner>(infoMap);
  }

  DAEScanner(DAEFunctionInfoMap* infoMap) : infoMap(infoMap) {}

  // The map of all infos for all functions.
  DAEFunctionInfoMap* infoMap;

  // The info for the function this instance operates on. We stash this as an
  // optimization.
  DAEFunctionInfo* info = nullptr;

  void visitCall(Call* curr) {
    if (!getModule()->getFunction(curr->target)->imported()) {
      info->calls[curr->target].push_back(curr);
    }
    if (curr->isReturn) {
      info->hasTailCalls = true;
      info->tailCallees.insert(curr->target);
    }
  }

  void visitCallIndirect(CallIndirect* curr) {
    if (curr->isReturn) {
      info->hasTailCalls = true;
    }
  }

  void visitCallRef(CallRef* curr) {
    if (curr->isReturn) {
      info->hasTailCalls = true;
    }
  }

  void visitDrop(Drop* curr) {
    if (auto* call = curr->value->dynCast<Call>()) {
      info->droppedCalls[call] = getCurrentPointer();
    }
  }

  void visitRefFunc(RefFunc* curr) {
    // RefFunc may be visited from either a function, in which case |info| was
    // set, or module-level code (in which case we use the null function name in
    // the infoMap).
    auto* currInfo = info ? info : &(*infoMap)[Name()];

    // Treat a ref.func as an unseen call, preventing us from changing the
    // function's type. If we did change it, it could be an observable
    // difference from the outside, if the reference escapes, for example.
    // TODO: look for actual escaping?
    // TODO: create a thunk for external uses that allow internal optimizations
    currInfo->hasUnseenCalls.insert(curr->func);
  }

  // main entry point

  void doWalkFunction(Function* func) {
    // Set the info for this function.
    info = &((*infoMap)[func->name]);

    if (!info->stale) {
      // Nothing changed since last time.
      return;
    }

    // Clear the data, mark us as no longer stale, and recompute everything.
    info->clear();
    info->stale = false;

    auto numParams = func->getNumParams();
    PostWalker<DAEScanner, Visitor<DAEScanner>>::doWalkFunction(func);
    // If there are params, check if they are used.
    // TODO: This work could be avoided if we cannot optimize for other reasons.
    //       That would require deferring this to later and checking that.
    if (numParams > 0) {
      auto usedParams = ParamUtils::getUsedParams(func, getModule());
      for (Index i = 0; i < numParams; i++) {
        if (!usedParams.contains(i)) {
          info->unusedParams.insert(i);
        }
      }
    }
  }
};

struct DAE : public Pass {
  // This pass changes locals and parameters.
  // FIXME DWARF updating does not handle local changes yet.
  bool invalidatesDWARF() override { return true; }

  bool optimize = false;

  Index numFunctions;

  // Map of function names to indexes. This lets us use indexes below for speed.
  std::unordered_map<Name, Index> indexes;

  void run(Module* module) override {
#if DAE_STATS
    Index startParams = 0, startResults = 0;
    for (auto& func : module->functions) {
      startParams += func->getNumParams();
      startResults += func->getResults().size();
    }
#endif // DAE_STATS

    DAEFunctionInfoMap infoMap;
    // Ensure all entries exist so the parallel threads don't modify the data
    // structure.
    for (auto& func : module->functions) {
      infoMap.try_emplace(func->name);
    }
    // The null name represents module-level code (not in a function).
    infoMap.try_emplace(Name());

    numFunctions = module->functions.size();

    for (Index i = 0; i < numFunctions; i++) {
      indexes[module->functions[i]->name] = i;
    }

    // Iterate to convergence.
    while (1) {
      if (!iteration(module, infoMap)) {
        break;
      }
    }

#if DAE_STATS
    Index endParams = 0, endResults = 0;
    for (auto& func : module->functions) {
      endParams += func->getNumParams();
      endResults += func->getResults().size();
    }
    std::cout << "Removed parameters: " << (startParams - endParams) << "\n";
    std::cout << "Removed results: " << (startResults - endResults) << "\n";
#endif // DAE_STATS
  }

  // A count of how many iterations we saw unprofitable removals of parameters.
  // An unprofitable removal is one where we only manage to remove from a single
  // call, that is, from one call target and it has a single call going to it.
  // Such calls are not very interesting, as when there is a single call like
  // that then inlining will handle it anyhow, in most cases, and inlining
  // does so far more efficiently in situations of call chains:
  //
  //   a -> b -> c -> d
  //
  // Imagine we remove a param from d, and so we remove it from the call in c.
  // If c received that as a parameter, and only ever used it to call d, then
  // now we can remove a param from c, and from the call in b, and so forth -
  // requiring a full iteration each time to find the small amount of progress.
  // (Inlining, otoh, will inline b into a, then c into a, and d into a,
  // efficiently.)
  Index unprofitableRemovalIters = 0;

  bool iteration(Module* module, DAEFunctionInfoMap& infoMap) {
    allDroppedCalls.clear();

#if DAE_DEBUG
    // Enable this path to mark all contents as stale at the start of each
    // iteration, which can be used to check for staleness bugs (that is, bugs
    // where something should have been marked stale, but wasn't). Note, though,
    // that staleness bugs can easily cause serious issues with validation (e.g.
    // if data is stale we may miss that there is an additional caller, that
    // prevents refining argument types etc.), so this may not be terribly
    // helpful.
    if (getenv("ALWAYS_MARK_STALE")) {
      for (auto& [_, info] : infoMap) {
        info.markStale();
      }
    }
#endif

    DAEScanner scanner(&infoMap);
    scanner.walkModuleCode(module);
    // Scan all the functions.
    scanner.run(getPassRunner(), module);

    // Combine all the info from the scan.
    std::vector<std::vector<Call*>> allCalls(numFunctions);
    std::vector<bool> tailCallees(numFunctions);
    std::vector<bool> hasUnseenCalls(numFunctions);
    // For each function, the functions that call it. This is used to propagate
    // changes, e.g. if we remove a return value from a function, the calls
    // might benefit from optimization, and if we refine a return type, the
    // callers must be refinalized. This must be computed in each iteration
    // rather than once, as optimizations can add new direct calls (e.g., by
    // turning a call_ref of a ref.func into a call). As |info.calls| is keyed
    // by callee, each caller appears at most once per callee.
    std::vector<std::vector<Name>> callers(numFunctions);

    for (auto& [func, info] : infoMap) {
      for (auto& [name, calls] : info.calls) {
        auto index = indexes[name];
        auto& allCallsToName = allCalls[index];
        allCallsToName.insert(allCallsToName.end(), calls.begin(), calls.end());
        callers[index].push_back(func);
      }
      for (auto& callee : info.tailCallees) {
        tailCallees[indexes[callee]] = true;
      }
      for (auto& [call, dropp] : info.droppedCalls) {
        allDroppedCalls[call] = dropp;
      }
      for (auto& name : info.hasUnseenCalls) {
        hasUnseenCalls[indexes[name]] = true;
      }
    }
    // Exports are considered unseen calls.
    for (auto& curr : module->exports) {
      if (curr->kind == ExternalKind::Function) {
        hasUnseenCalls[indexes[*curr->getInternalName()]] = true;
      }
    }

    // Track which functions we changed that are worth re-optimizing at the end.
    std::unordered_set<Function*> worthOptimizing;

    // The callers of functions whose return types we refined. The types of the
    // calls changed, so these functions must be refinalized. The more refined
    // types may also enable optimizations there (e.g., a cast may become
    // statically known to succeed), so we optimize them at the end, together
    // with |worthOptimizing|. We keep them apart from |worthOptimizing| as that
    // set also affects which transformations we do in this iteration. If this
    // is not empty, we iterate again, as the refined types may allow further
    // refinements.
    std::unordered_set<Function*> refinedCallers;

    // If we find that localizing call arguments can help (by moving their
    // effects outside, so ParamUtils::removeParameters can handle them), then
    // we do that at the end and perform another cycle. It is simpler to just do
    // another cycle than to track the locations of calls, which is tricky as
    // localization might move a call (if a call happens to be another call's
    // param). In practice it is rare to find call arguments we want to remove,
    // and even more rare to find effects get in the way, so this should not
    // cause much overhead.
    //
    // This set tracks the functions for whom calls to it should be modified.
    std::unordered_set<Name> callTargetsToLocalize;

    // As we optimize, we mark things as stale.
    auto markStale = [&](Name func) {
      // We only ever mark functions stale (not the global scope, which we never
      // modify). An attempt to modify the global scope, identified by a null
      // function name, is a logic bug.
      assert(func.is());
      infoMap[func].markStale();
    };
    auto markCallersStale = [&](Index index) {
      for (auto caller : callers[index]) {
        markStale(caller);
      }
    };

    // We now have a mapping of all call sites for each function, and can look
    // for optimization opportunities.
    for (Index index = 0; index < numFunctions; index++) {
      auto* func = module->functions[index].get();
      if (func->imported()) {
        continue;
      }
      // We can only optimize if we see all the calls and can modify them.
      if (hasUnseenCalls[index]) {
        continue;
      }
      auto& calls = allCalls[index];
      if (calls.empty()) {
        // Nothing calls this, so it is not worth optimizing.
        continue;
      }
      // Refine argument types before doing anything else. This does not
      // affect whether an argument is used or not, it just refines the type
      // where possible.
      auto name = func->name;
      if (refineArgumentTypes(func, calls, module, infoMap[name])) {
        worthOptimizing.insert(func);
        markStale(func->name);
      }
      // Refine return types as well.
      if (refineReturnTypes(func, calls, module)) {
        markStale(name);
        markCallersStale(index);
        for (auto caller : callers[index]) {
          refinedCallers.insert(module->getFunction(caller));
        }
      }
      auto optimizedIndexes =
        ParamUtils::applyConstantValues({func}, calls, {}, module);
      for (auto i : optimizedIndexes) {
        // Mark it as unused, which we know it now is (no point to re-scan just
        // for that).
        infoMap[name].unusedParams.insert(i);
      }
      if (!optimizedIndexes.empty()) {
        markStale(func->name);
      }
    }
    if (!refinedCallers.empty()) {
      // Changing a call expression's return type can propagate out to its
      // parents, and so we must refinalize the callers.
      PassUtils::FilteredPassRunner runner(
        module, refinedCallers, getPassRunner()->options);
      runner.setIsNested(true);
      runner.add(std::make_unique<ReFinalize>());
      runner.run();
    }
    // We now know which parameters are unused, and can potentially remove them.
    // Only do so if we didn't run into unprofitable removals - if so, leave
    // any further removals for other invocations of this pass. (This avoids us
    // getting stuck in long unprofitable call chains as mentioned in the
    // comment earlier; note that we do process one unprofitable iteration
    // before giving up here, so we do make progress at least.)
    if (!unprofitableRemovalIters) {
      Index removals = 0;
      Index singleCallerRemovals = 0;
      for (Index index = 0; index < numFunctions; index++) {
        auto* func = module->functions[index].get();
        if (func->imported()) {
          continue;
        }
        if (hasUnseenCalls[index]) {
          continue;
        }
        auto numParams = func->getNumParams();
        if (numParams == 0) {
          continue;
        }
        auto& calls = allCalls[index];
        if (calls.empty()) {
          continue;
        }
        auto name = func->name;
        auto [removedIndexes, outcome] =
          ParamUtils::removeParameters({func},
                                       infoMap[name].unusedParams,
                                       calls,
                                       {},
                                       module,
                                       getPassRunner());
        if (!removedIndexes.empty()) {
          // Success!
          worthOptimizing.insert(func);
          markStale(name);
          markCallersStale(index);
          if (calls.size() == 1) {
            singleCallerRemovals++;
          }
          removals++;
        }
        if (outcome == ParamUtils::RemovalOutcome::Failure) {
          callTargetsToLocalize.insert(name);
        }
      }
      if (removals == 1 && singleCallerRemovals == 1 &&
          callTargetsToLocalize.empty()) {
        // We only removed parameters from one function, and it had a single
        // caller, and we don't have other pending actions (call targets we
        // need to localize), so this was unprofitable as mentioned earlier.
        unprofitableRemovalIters++;
      }
    }
    // We can also tell which calls have all their return values dropped. Note
    // that we can't do this if we changed anything so far, as we may have
    // modified allCalls (we can't modify a call site twice in one iteration,
    // once to remove a param, once to drop the return value).
    if (worthOptimizing.empty()) {
      for (Index index = 0; index < numFunctions; index++) {
        auto& func = module->functions[index];
        if (func->imported()) {
          continue;
        }
        if (func->getResults() == Type::none) {
          continue;
        }
        if (hasUnseenCalls[index]) {
          continue;
        }
        auto name = func->name;
        if (infoMap[name].hasTailCalls) {
          continue;
        }
        if (tailCallees[index]) {
          continue;
        }
        auto& calls = allCalls[index];
        if (calls.empty()) {
          continue;
        }
        bool allDropped =
          std::all_of(calls.begin(), calls.end(), [&](Call* call) {
            return allDroppedCalls.contains(call);
          });
        if (!allDropped) {
          continue;
        }
        if (removeReturnValue(func.get(), calls, module)) {
          // We should optimize the callers.
          for (auto caller : callers[index]) {
            worthOptimizing.insert(module->getFunction(caller));
          }
        }
        // TODO Removing a drop may also open optimization opportunities in the
        // callers.
        worthOptimizing.insert(func.get());
        markStale(name);
        markCallersStale(index);
      }
      if (worthOptimizing.empty() && refinedCallers.empty() &&
          callTargetsToLocalize.empty()) {
        splitDroppedReturnTails(module,
                                infoMap,
                                allCalls,
                                callers,
                                tailCallees,
                                hasUnseenCalls,
                                worthOptimizing);
      }
    }
    if (!callTargetsToLocalize.empty()) {
      ParamUtils::localizeCallsTo(
        callTargetsToLocalize, *module, getPassRunner(), [&](Function* func) {
          markStale(func->name);
        });
    }
    if (optimize) {
      // This happens after the ReFinalize above, so the callers of functions
      // with refined results see the new types.
      worthOptimizing.insert(refinedCallers.begin(), refinedCallers.end());
      if (!worthOptimizing.empty()) {
        OptUtils::optimizeAfterInlining(
          worthOptimizing, module, getPassRunner());
      }
    }

    return !worthOptimizing.empty() || !refinedCallers.empty() ||
           !callTargetsToLocalize.empty();
  }

private:
  std::unordered_map<Call*, Expression**> allDroppedCalls;

  int getPureReturnSize(Expression* curr,
                        Function* func,
                        const std::vector<bool>& hasSet,
                        Module* module) {
    auto numParams = func->getNumParams();
    if (auto* get = curr->dynCast<LocalGet>()) {
      return (get->index < numParams && !hasSet[get->index]) ? 2 : 0;
    }
    if (curr->is<Const>() || curr->is<RefNull>()) {
      return 2;
    }
    if (curr->is<RefFunc>()) {
      return 3;
    }
    if (auto* gget = curr->dynCast<GlobalGet>()) {
      auto* g = module->getGlobalOrNull(gget->name);
      return (g && !g->mutable_) ? 3 : 0;
    }
    if (getPassOptions().trapsNeverHappen) {
      if (auto* rc = curr->dynCast<RefCast>()) {
        if (!rc->desc) {
          if (auto* get = rc->ref->dynCast<LocalGet>()) {
            if (get->index < numParams && !hasSet[get->index]) {
              return 6;
            }
          }
        }
      }
    }
    return 0;
  }

  std::vector<bool> getParamSets(Function* func) {
    auto numParams = func->getNumParams();
    std::vector<bool> hasSet(numParams, false);
    if (numParams == 0 || !func->body) {
      return hasSet;
    }
    struct SetScanner : public PostWalker<SetScanner> {
      Index numParams;
      std::vector<bool>& hasSet;
      SetScanner(Index numParams, std::vector<bool>& hasSet)
        : numParams(numParams), hasSet(hasSet) {}
      void visitLocalSet(LocalSet* curr) {
        if (curr->index < numParams) {
          hasSet[curr->index] = true;
        }
      }
    } scanner(numParams, hasSet);
    scanner.walk(func->body);
    return hasSet;
  }

  Name getWrappedVoidCallee(Function* func, Module* module) {
    if (!func || func->imported() || !func->vars.empty()) {
      return Name();
    }
    auto* block = func->body->dynCast<Block>();
    if (!block || block->list.size() != 2) {
      return Name();
    }
    if (block->name.is() &&
        BranchUtils::BranchSeeker::has(block, block->name)) {
      return Name();
    }
    auto* call = block->list[0]->dynCast<Call>();
    if (!call || call->isReturn || call->type != Type::none) {
      return Name();
    }
    auto numParams = func->getNumParams();
    if (call->operands.size() != numParams) {
      return Name();
    }
    for (Index i = 0; i < numParams; i++) {
      auto* get = call->operands[i]->dynCast<LocalGet>();
      if (!get || get->index != i) {
        return Name();
      }
    }
    std::vector<bool> hasSet(numParams, false);
    if (getPureReturnSize(block->list[1], func, hasSet, module) <= 0) {
      return Name();
    }
    auto* targetFunc = module->getFunctionOrNull(call->target);
    if (!targetFunc || targetFunc->getResults() != Type::none ||
        targetFunc->getParams() != func->getParams()) {
      return Name();
    }
    return call->target;
  }

  void splitDroppedReturnTails(Module* module,
                               DAEFunctionInfoMap& infoMap,
                               const std::vector<std::vector<Call*>>& allCalls,
                               const std::vector<std::vector<Name>>& callers,
                               const std::vector<bool>& tailCallees,
                               const std::vector<bool>& hasUnseenCalls,
                               std::unordered_set<Function*>& worthOptimizing) {
    Index origNumFunctions = numFunctions;
    auto markStale = [&](Name func) {
      assert(func.is());
      infoMap[func].markStale();
    };
    auto markCallersStale = [&](Index index) {
      for (auto caller : callers[index]) {
        markStale(caller);
      }
    };

    // Pass 0 splits Shape A functions (whose body is a Block ending directly in
    // a pure value expression). Pass 1 redirects dropped calls to existing void
    // wrappers and splits Shape B functions (whose tail calls a Shape A
    // function and forwards or casts its pure return value).
    for (int pass = 0; pass < 2; pass++) {
      for (Index index = 0; index < origNumFunctions; index++) {
        auto* func = module->functions[index].get();
        if (func->imported() || !func->getResults().isSingle()) {
          continue;
        }
        auto name = func->name;
        if (infoMap[name].hasTailCalls || tailCallees[index]) {
          continue;
        }
        if (infoMap[name].calls.contains(name)) {
          continue;
        }
        auto& calls = allCalls[index];
        if (calls.empty()) {
          continue;
        }
        std::vector<Call*> droppedCalls;
        Index usedCallCount = 0;
        for (auto* call : calls) {
          if (allDroppedCalls.contains(call)) {
            droppedCalls.push_back(call);
          } else {
            usedCallCount++;
          }
        }
        if (droppedCalls.empty()) {
          continue;
        }
        if (usedCallCount == 0 && !hasUnseenCalls[index]) {
          continue;
        }

        // Rule 1: If func is already a 0-var wrapper around a void callee with
        // a pure return value, redirect all dropped calls directly to the void
        // callee at zero wrapper cost.
        if (Name voidTarget = getWrappedVoidCallee(func, module);
            voidTarget.is()) {
          if (indexes[voidTarget] >= 16384 && index < 16384 &&
              droppedCalls.size() < 2) {
            continue;
          }
          for (auto* call : droppedCalls) {
            auto iter = allDroppedCalls.find(call);
            if (iter == allDroppedCalls.end()) {
              continue;
            }
            Expression** location = iter->second;
            *location = call;
            call->target = voidTarget;
            if (call->type != Type::unreachable) {
              call->type = Type::none;
            }
            allDroppedCalls.erase(iter);
          }
          for (auto caller : callers[index]) {
            worthOptimizing.insert(module->getFunction(caller));
          }
          markStale(name);
          markCallersStale(index);
          continue;
        }

        // Rule 2: Void-Tail Splitting. Only create new wrapper functions when
        // shrinking or running standalone DAE (not in -O3 dae-optimizing where
        // subsequent inlining passes could re-inline the wrapper).
        if (optimize && getPassOptions().shrinkLevel == 0) {
          continue;
        }
        if (droppedCalls.size() < 4) {
          continue;
        }
        if (ReturnUtils::getInfo(func->body).hasReturn) {
          continue;
        }

        auto numParams = func->getNumParams();
        if (pass == 0) {
          // Shape A: func->body is an untargeted Block whose elements before
          // the final pure return value all have Type::none.
          auto* block = func->body->dynCast<Block>();
          if (!block || block->list.size() < 2) {
            continue;
          }
          if (block->name.is() &&
              BranchUtils::BranchSeeker::has(block, block->name)) {
            continue;
          }
          bool priorAllVoid = true;
          for (Index i = 0; i + 1 < block->list.size(); i++) {
            if (block->list[i]->type != Type::none) {
              priorAllVoid = false;
              break;
            }
          }
          if (!priorAllVoid) {
            continue;
          }
          auto hasSet = getParamSets(func);
          Expression* tail = block->list.back();
          int retSize = getPureReturnSize(tail, func, hasSet, module);
          if (retSize <= 0) {
            continue;
          }
          int wrapperCost = (usedCallCount == 1 && !hasUnseenCalls[index])
                              ? 1
                              : int(6 + 2 * numParams + retSize);
          int saved = int(droppedCalls.size()) - wrapperCost;
          if (saved < 8) {
            continue;
          }

          Builder builder(*module);
          Name voidName = Names::getValidFunctionName(
            *module, func->name.toString() + "_void");
          auto voidFunc =
            builder.makeFunction(voidName,
                                 Signature(func->getParams(), Type::none),
                                 std::move(func->vars));
          func->vars.clear();

          block->list.pop_back();
          block->type = Type::none;
          voidFunc->body = block;

          std::vector<Expression*> args;
          args.reserve(numParams);
          for (Index i = 0; i < numParams; i++) {
            args.push_back(builder.makeLocalGet(i, func->getLocalType(i)));
          }
          auto* voidCall = builder.makeCall(voidName, args, Type::none);
          func->body = builder.makeSequence(voidCall, tail);

          infoMap.try_emplace(voidName);
          indexes[voidName] = numFunctions++;
          Function* voidFuncPtr = module->addFunction(std::move(voidFunc));

          for (auto* call : droppedCalls) {
            auto iter = allDroppedCalls.find(call);
            assert(iter != allDroppedCalls.end());
            Expression** location = iter->second;
            *location = call;
            call->target = voidName;
            if (call->type != Type::unreachable) {
              call->type = Type::none;
            }
            allDroppedCalls.erase(iter);
          }
          for (auto caller : callers[index]) {
            worthOptimizing.insert(module->getFunction(caller));
          }
          worthOptimizing.insert(voidFuncPtr);
          markStale(name);
          markStale(voidName);
          markCallersStale(index);
        } else {
          // Shape B: func's tail is a non-tail Call (or under TNH a RefCast of
          // a non-tail Call) to a Shape A function G whose pure return value
          // maps to a pure return value in func.
          Block* block = func->body->dynCast<Block>();
          Expression** tailPtr = nullptr;
          if (block) {
            if (block->list.size() < 2) {
              continue;
            }
            if (block->name.is() &&
                BranchUtils::BranchSeeker::has(block, block->name)) {
              continue;
            }
            bool priorAllVoid = true;
            for (Index i = 0; i + 1 < block->list.size(); i++) {
              if (block->list[i]->type != Type::none) {
                priorAllVoid = false;
                break;
              }
            }
            if (!priorAllVoid) {
              continue;
            }
            tailPtr = &block->list.back();
          } else {
            tailPtr = &func->body;
          }

          Expression* tail = *tailPtr;
          RefCast* rc = nullptr;
          Call* innerCall = tail->dynCast<Call>();
          if (!innerCall && getPassOptions().trapsNeverHappen) {
            rc = tail->dynCast<RefCast>();
            if (rc && !rc->desc) {
              innerCall = rc->ref->dynCast<Call>();
            }
          }
          if (!innerCall || innerCall->isReturn) {
            continue;
          }
          Function* G = module->getFunctionOrNull(innerCall->target);
          if (!G || G->imported() || G == func || !G->getResults().isSingle()) {
            continue;
          }
          if (ReturnUtils::getInfo(G->body).hasReturn) {
            continue;
          }
          auto* gBlock = G->body->dynCast<Block>();
          if (!gBlock || gBlock->list.size() < 2) {
            continue;
          }
          if (gBlock->name.is() &&
              BranchUtils::BranchSeeker::has(gBlock, gBlock->name)) {
            continue;
          }
          bool gPriorAllVoid = true;
          for (Index i = 0; i + 1 < gBlock->list.size(); i++) {
            if (gBlock->list[i]->type != Type::none) {
              gPriorAllVoid = false;
              break;
            }
          }
          if (!gPriorAllVoid) {
            continue;
          }
          auto gHasSet = getParamSets(G);
          Expression* gTail = gBlock->list.back();
          if (getPureReturnSize(gTail, G, gHasSet, module) <= 0) {
            continue;
          }

          Expression* candRet = nullptr;
          if (auto* gGet = gTail->dynCast<LocalGet>()) {
            if (gGet->index < innerCall->operands.size()) {
              candRet = innerCall->operands[gGet->index];
            }
          } else if (gTail->is<Const>() || gTail->is<RefNull>() ||
                     gTail->is<RefFunc>() || gTail->is<GlobalGet>()) {
            candRet = gTail;
          }
          if (!candRet) {
            continue;
          }

          auto hasSet = getParamSets(func);
          if (getPureReturnSize(candRet, func, hasSet, module) <= 0) {
            continue;
          }

          Builder builder(*module);
          Expression* retVal = ExpressionManipulator::copy(candRet, *module);
          if (rc && !Type::isSubType(retVal->type, rc->type)) {
            retVal = builder.makeRefCast(retVal, rc->type);
          }
          if (!Type::isSubType(retVal->type, func->getResults())) {
            continue;
          }
          int retSize = getPureReturnSize(retVal, func, hasSet, module);
          if (retSize <= 0) {
            continue;
          }

          Name voidG = getWrappedVoidCallee(G, module);
          int extraDropCost = voidG.is() ? 0 : 1;
          int wrapperCost = ((usedCallCount == 1 && !hasUnseenCalls[index])
                               ? 1
                               : int(6 + 2 * numParams + retSize)) +
                            extraDropCost;
          int saved = int(droppedCalls.size()) - wrapperCost;
          if (saved < 8) {
            continue;
          }

          Name voidName = Names::getValidFunctionName(
            *module, func->name.toString() + "_void");
          auto voidFunc =
            builder.makeFunction(voidName,
                                 Signature(func->getParams(), Type::none),
                                 std::move(func->vars));
          func->vars.clear();

          if (voidG.is()) {
            innerCall->target = voidG;
            if (innerCall->type != Type::unreachable) {
              innerCall->type = Type::none;
            }
            *tailPtr = innerCall;
          } else {
            *tailPtr = builder.makeDrop(innerCall);
          }
          if (block) {
            block->type = Type::none;
          }
          voidFunc->body = func->body;

          std::vector<Expression*> args;
          args.reserve(numParams);
          for (Index i = 0; i < numParams; i++) {
            args.push_back(builder.makeLocalGet(i, func->getLocalType(i)));
          }
          auto* voidCall = builder.makeCall(voidName, args, Type::none);
          func->body = builder.makeSequence(voidCall, retVal);

          infoMap.try_emplace(voidName);
          indexes[voidName] = numFunctions++;
          Function* voidFuncPtr = module->addFunction(std::move(voidFunc));

          for (auto* call : droppedCalls) {
            auto iter = allDroppedCalls.find(call);
            assert(iter != allDroppedCalls.end());
            Expression** location = iter->second;
            *location = call;
            call->target = voidName;
            if (call->type != Type::unreachable) {
              call->type = Type::none;
            }
            allDroppedCalls.erase(iter);
          }
          for (auto caller : callers[index]) {
            worthOptimizing.insert(module->getFunction(caller));
          }
          worthOptimizing.insert(voidFuncPtr);
          markStale(name);
          markStale(voidName);
          markCallersStale(index);
        }
      }
    }
  }

  // Returns `true` if the caller should be optimized.
  bool
  removeReturnValue(Function* func, std::vector<Call*>& calls, Module* module) {
    // If the result type is uninhabitable, then the caller knows the call will
    // never return. That useful information would be lost if we did nothing
    // else when removing the return value, but we will insert an `unreachable`
    // after the call in the caller to preserve the optimization effect. TODO:
    // Do this for more complicated uninhabitable types such as non-nullable
    // references to structs with non-nullable reference cycles.
    bool wasReturnUninhabitable =
      func->getResults().isNull() && func->getResults().isNonNullable();
    func->setResults(Type::none);
    // Remove the drops on the calls. Note that we must do this before updating
    // returns in ReturnUpdater, as there may be recursive calls of this
    // function to itself. So we first use the information in allDroppedCalls
    // before the ReturnUpdater potentially invalidates that information as it
    // modifies the function.
    for (auto* call : calls) {
      auto iter = allDroppedCalls.find(call);
      assert(iter != allDroppedCalls.end());
      Expression** location = iter->second;
      if (wasReturnUninhabitable) {
        Builder builder(*module);
        *location = builder.makeSequence(call, builder.makeUnreachable());
      } else {
        *location = call;
      }
      // Update the call's type.
      if (call->type != Type::unreachable) {
        call->type = Type::none;
      }
    }
    // Remove any return values.
    ReturnUtils::removeReturns(func, *module);
    // It's definitely worth optimizing the caller after inserting the
    // unreachable.
    return wasReturnUninhabitable;
  }

  // Given a function and all the calls to it, see if we can refine the type of
  // its arguments. If we only pass in a subtype, we may as well refine the type
  // to that.
  //
  // This assumes that the function has no calls aside from |calls|, that is, it
  // is not exported or called from the table or by reference.
  bool refineArgumentTypes(Function* func,
                           const std::vector<Call*>& calls,
                           Module* module,
                           const DAEFunctionInfo& info) {
    if (!module->features.hasGC()) {
      return false;
    }
    auto numParams = func->getNumParams();
    std::vector<Type> newParamTypes;
    newParamTypes.reserve(numParams);
    std::vector<LUBFinder> lubs(numParams);
    for (Index i = 0; i < numParams; i++) {
      auto originalType = func->getLocalType(i);
      // If the parameter type is not a reference, there is nothing to refine.
      // And if it is unused, also do nothing, as we can leave it to the other
      // parts of this pass to optimize it properly, which avoids having to
      // think about corner cases involving refining the type of an unused
      // param (in particular, unused params are turned into locals, which means
      // we'd need to think about defaultability etc.).
      if (!originalType.isRef() || info.unusedParams.has(i)) {
        newParamTypes.push_back(originalType);
        continue;
      }
      auto& lub = lubs[i];
      for (auto* call : calls) {
        auto* operand = call->operands[i];
        lub.note(operand->type);
        if (lub.getLUB() == originalType) {
          // We failed to refine this parameter to anything more specific.
          break;
        }
      }

      // Nothing is sent here at all; leave such optimizations to DCE.
      if (!lub.noted()) {
        return false;
      }
      newParamTypes.push_back(lub.getLUB());
    }

    // Check if we are able to optimize here before we do the work to scan the
    // function body.
    auto newParams = Type(newParamTypes);
    if (newParams == func->getParams()) {
      return false;
    }

    // We can do this!
    TypeUpdating::updateParamTypes(func, newParamTypes, *module);

    // Update the function's type.
    func->setParams(newParams);

    return true;
  }

  // See if the types returned from a function allow us to define a more refined
  // return type for it. If so, we can update it and all calls going to it.
  //
  // This assumes that the function has no calls aside from |calls|, that is, it
  // is not exported or called from the table or by reference. Exports should be
  // fine, as should indirect calls in principle, but VMs will need to support
  // function subtyping in indirect calls. TODO: relax this when possible
  //
  // Returns whether we optimized.
  //
  // TODO: We may be missing a global optimum here, as e.g. if a function calls
  //       itself and returns that value, then we would not do any change here,
  //       as one of the return values is exactly what it already is. Similar
  //       unoptimality can happen with multiple functions, more local code in
  //       the middle, etc.
  bool refineReturnTypes(Function* func,
                         const std::vector<Call*>& calls,
                         Module* module) {
    auto lub = LUB::getResultsLUB(func, *module);
    Type newType;
    if (lub.noted()) {
      newType = lub.getLUB();
    } else {
      // No value can ever be returned: the function never returns normally
      // (it throws, traps, or loops forever), and neither do the functions it
      // tail-calls. If the result is a single reference, refine it to the
      // uninhabitable bottom type of its hierarchy, so the callers can see that
      // the call never returns. (Callers that tail-call this function will in
      // turn note this type as their possible result.)
      auto results = func->getResults();
      if (!module->features.hasGC() || !results.isRef()) {
        return false;
      }
      newType = Type(results.getHeapType().getBottom(), NonNullable);
    }
    if (newType != func->getResults()) {
      func->setResults(newType);
      for (auto* call : calls) {
        if (call->type != Type::unreachable) {
          call->type = newType;
        }
      }
      return true;
    }
    return false;
  }
};

Pass* createDAEPass() { return new DAE(); }

Pass* createDAEOptimizingPass() {
  auto* ret = new DAE();
  ret->optimize = true;
  return ret;
}

} // namespace wasm
