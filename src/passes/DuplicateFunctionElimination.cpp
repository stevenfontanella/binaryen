/*
 * Copyright 2016 WebAssembly Community Group participants
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
// Removes duplicate functions. That can happen due to C++ templates,
// and also due to types being different at the source level, but
// identical when finally lowered into concrete wasm code.
//

#include "ir/find_all.h"
#include "ir/function-utils.h"
#include "ir/hashed.h"
#include "ir/intrinsics.h"
#include "ir/module-utils.h"
#include "ir/subtypes.h"
#include "ir/type-updating.h"
#include "ir/utils.h"
#include "opt-utils.h"
#include "pass.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

struct DuplicateFunctionElimination : public Pass {
  // FIXME Merge DWARF info
  bool invalidatesDWARF() override { return true; }

  // This pass merges functions and may narrow non-escaped target signatures.
  bool requiresNonNullableLocalFixups() override { return false; }

  Name getIdentityThunkTarget(Module* module, Function* func) {
    if (func->imported() || func->getNumVars() > 0 || func->noFullInline ||
        func->noPartialInline || !func->body) {
      return Name();
    }
    Expression* body = func->body;
    while (auto* block = body->dynCast<Block>()) {
      if (!block->name.is() && block->list.size() == 1) {
        body = block->list[0];
      } else {
        break;
      }
    }
    if (auto* ret = body->dynCast<Return>()) {
      if (!ret->value) {
        return Name();
      }
      body = ret->value;
    }
    auto* call = body->dynCast<Call>();
    if (!call || call->target == func->name ||
        Intrinsics(*module).isCallWithoutEffects(call)) {
      return Name();
    }
    Index numParams = func->getNumParams();
    if (call->operands.size() != numParams) {
      return Name();
    }
    for (Index i = 0; i < numParams; ++i) {
      auto* get = call->operands[i]->dynCast<LocalGet>();
      if (!get || get->index != i) {
        return Name();
      }
    }
    return call->target;
  }

  void unifyPublicPrivateSignatures(Module* module) {
    if (!module->features.hasGC()) {
      return;
    }
    auto& options = getPassOptions();
    auto typeInfo = ModuleUtils::collectHeapTypeInfo(
      *module,
      options.worldMode,
      ModuleUtils::TypeInclusion::UsedIRTypes,
      ModuleUtils::VisibilityHandling::FindVisibility);
    std::vector<HeapType> allTypes;
    allTypes.reserve(typeInfo.size());
    for (auto& [type, _] : typeInfo) {
      allTypes.push_back(type);
    }
    SubTypes subTypes(allTypes);

    std::unordered_map<Signature, HeapType> publicSigToHT;
    for (auto& [type, info] : typeInfo) {
      if (info.visibility == ModuleUtils::Visibility::Public &&
          type.isSignature() && !type.isOpen() &&
          !type.getDeclaredSuperType() &&
          subTypes.getImmediateSubTypes(type).empty() && !type.isShared() &&
          !type.getDescriptorType() && !type.getDescribedType()) {
        publicSigToHT.try_emplace(type.getSignature(), type);
      }
    }
    if (publicSigToHT.empty()) {
      return;
    }

    struct FuncTypeScan {
      std::vector<HeapType> castTypes;
      std::vector<bool> usedParams;
    };
    ModuleUtils::ParallelFunctionAnalysis<FuncTypeScan> scanner(
      *module, [&](Function* func, FuncTypeScan& scan) {
        if (func->imported()) {
          return;
        }
        Index numParams = func->getNumParams();
        scan.usedParams.assign(numParams, false);
        for (auto* get : FindAll<LocalGet>(func->body).list) {
          if (get->index < numParams) {
            scan.usedParams[get->index] = true;
          }
        }
        for (auto* rt : FindAll<RefTest>(func->body).list) {
          if (rt->castType.isRef()) {
            scan.castTypes.push_back(rt->castType.getHeapType());
          }
        }
        for (auto* br : FindAll<BrOn>(func->body).list) {
          if ((br->op == BrOnCast || br->op == BrOnCastFail) &&
              br->castType.isRef()) {
            scan.castTypes.push_back(br->castType.getHeapType());
          }
        }
        if (!options.trapsNeverHappen ||
            options.worldMode != WorldMode::Closed) {
          for (auto* rc : FindAll<RefCast>(func->body).list) {
            if (rc->type.isRef()) {
              scan.castTypes.push_back(rc->type.getHeapType());
            }
          }
          for (auto* ci : FindAll<CallIndirect>(func->body).list) {
            scan.castTypes.push_back(ci->heapType);
          }
        }
        if (options.worldMode != WorldMode::Closed) {
          for (auto* cr : FindAll<CallRef>(func->body).list) {
            if (cr->target->type.isSignature()) {
              scan.castTypes.push_back(cr->target->type.getHeapType());
            }
          }
        }
      });

    std::unordered_set<HeapType> castTypes;
    std::unordered_map<HeapType, std::vector<Function*>> sigFuncs;
    for (auto& fn : module->functions) {
      if (fn->imported()) {
        continue;
      }
      Function* func = fn.get();
      sigFuncs[func->type.getHeapType()].push_back(func);
      for (auto ht : scanner.map[func].castTypes) {
        castTypes.insert(ht);
      }
    }

    // Track whether unifying a private signature with a public one directly
    // unlocks an identity thunk fold.
    std::unordered_set<HeapType> unlocksThunk;
    for (auto& fn : module->functions) {
      if (fn->imported()) {
        continue;
      }
      Name targetName = getIdentityThunkTarget(module, fn.get());
      if (!targetName.is()) {
        continue;
      }
      Function* target = module->getFunction(targetName);
      HeapType h1 = fn->type.getHeapType();
      HeapType h2 = target->type.getHeapType();
      if (h1 != h2 && h1.getSignature() == h2.getSignature()) {
        unlocksThunk.insert(h1);
        unlocksThunk.insert(h2);
      }
    }

    std::unordered_map<HeapType, HeapType> typeReplacements;
    for (auto& [type, info] : typeInfo) {
      if (info.visibility != ModuleUtils::Visibility::Private ||
          !type.isSignature() || type.getDeclaredSuperType() ||
          !subTypes.getImmediateSubTypes(type).empty() || type.isShared() ||
          type.getDescriptorType() || type.getDescribedType() ||
          castTypes.count(type)) {
        continue;
      }
      Signature sig = type.getSignature();
      auto pubIt = publicSigToHT.find(sig);
      if (pubIt == publicSigToHT.end() || castTypes.count(pubIt->second)) {
        continue;
      }
      if (!unlocksThunk.count(type)) {
        // Avoid merging into a public signature if any parameter is unused
        // across all defined functions of `type`, so SignaturePruning can still
        // prune it.
        const auto& funcs = sigFuncs[type];
        if (!funcs.empty() && sig.params.size() > 0) {
          bool allParamsUsed = true;
          for (Index i = 0; i < sig.params.size(); ++i) {
            bool paramUsed = false;
            for (auto* f : funcs) {
              if (i < scanner.map[f].usedParams.size() &&
                  scanner.map[f].usedParams[i]) {
                paramUsed = true;
                break;
              }
            }
            if (!paramUsed) {
              allParamsUsed = false;
              break;
            }
          }
          if (!allParamsUsed) {
            continue;
          }
        }
        // When shrinkLevel == 0 (e.g. early -O3 passes before SignatureRefining
        // and RelaxNullability), do not merge signatures containing non-basic
        // or non-nullable reference types unless it unlocks a thunk fold.
        if (options.shrinkLevel == 0) {
          bool hasRefinableRef = false;
          for (Type t : sig.params) {
            if (t.isRef() && (!t.getHeapType().isBasic() || t.isNonNullable())) {
              hasRefinableRef = true;
              break;
            }
          }
          for (Type t : sig.results) {
            if (t.isRef() && (!t.getHeapType().isBasic() || t.isNonNullable())) {
              hasRefinableRef = true;
              break;
            }
          }
          if (hasRefinableRef) {
            continue;
          }
        }
      }
      typeReplacements[type] = pubIt->second;
    }

    if (typeReplacements.empty()) {
      return;
    }

    bool referencedInHeapType = false;
    auto checkTypeRef = [&](Type t) {
      for (Type elem : t) {
        if (elem.isRef() && typeReplacements.count(elem.getHeapType())) {
          referencedInHeapType = true;
        }
      }
    };
    for (auto ht : allTypes) {
      if (ht.isStruct()) {
        for (auto& f : ht.getStruct().fields) {
          checkTypeRef(f.type);
        }
      } else if (ht.isArray()) {
        checkTypeRef(ht.getArray().element.type);
      } else if (ht.isSignature()) {
        checkTypeRef(ht.getSignature().params);
        checkTypeRef(ht.getSignature().results);
      } else if (ht.isContinuation()) {
        if (typeReplacements.count(ht.getContinuation().type)) {
          referencedInHeapType = true;
        }
      }
      if (referencedInHeapType) {
        break;
      }
    }

    if (referencedInHeapType && options.worldMode != WorldMode::Closed) {
      return;
    }

    if (referencedInHeapType) {
      TypeMapper(*module, typeReplacements, options.worldMode).map();
      return;
    }

    // Direct replacement when none of the replaced signature HeapTypes are
    // nested inside other HeapType definitions (avoids rebuilding unrelated
    // singleton rec groups).
    auto mapType = [&](auto& self, Type t) -> Type {
      if (t.isRef()) {
        auto it = typeReplacements.find(t.getHeapType());
        if (it != typeReplacements.end()) {
          return t.with(it->second);
        }
        return t;
      }
      if (t.isTuple()) {
        Tuple tuple = t.getTuple();
        bool any = false;
        for (auto& elem : tuple) {
          Type mapped = self(self, elem);
          if (mapped != elem) {
            elem = mapped;
            any = true;
          }
        }
        return any ? Type(tuple) : t;
      }
      return t;
    };

    if (!module->indirectCallEffects.empty()) {
      for (auto& [privHT, pubHT] : typeReplacements) {
        module->indirectCallEffects.erase(privHT);
        module->indirectCallEffects.erase(pubHT);
      }
    }

    for (auto& fn : module->functions) {
      fn->type = mapType(mapType, fn->type);
      for (auto& v : fn->vars) {
        v = mapType(mapType, v);
      }
    }
    for (auto& g : module->globals) {
      g->type = mapType(mapType, g->type);
      if (!g->imported() && g->init) {
        for (auto* rf : FindAll<RefFunc>(g->init).list) {
          rf->type = mapType(mapType, rf->type);
        }
      }
    }
    for (auto& seg : module->elementSegments) {
      seg->type = mapType(mapType, seg->type);
      for (auto* expr : seg->data) {
        for (auto* rf : FindAll<RefFunc>(expr).list) {
          rf->type = mapType(mapType, rf->type);
        }
      }
    }

    struct ExprTypeUpdater
      : public PostWalker<ExprTypeUpdater,
                          UnifiedExpressionVisitor<ExprTypeUpdater>> {
      decltype(mapType)& mapT;
      Function* func;
      ExprTypeUpdater(decltype(mapType)& mapT, Function* func)
        : mapT(mapT), func(func) {}
      void visitExpression(Expression* curr) {
        if (auto* get = curr->dynCast<LocalGet>()) {
          curr->type = func->getLocalType(get->index);
          return;
        }
        if (auto* set = curr->dynCast<LocalSet>()) {
          if (set->type != Type::none && set->type != Type::unreachable) {
            curr->type = func->getLocalType(set->index);
          }
          return;
        }
        curr->type = mapT(mapT, curr->type);
        if (auto* rn = curr->dynCast<RefNull>()) {
          rn->type = mapT(mapT, rn->type);
        } else if (auto* rf = curr->dynCast<RefFunc>()) {
          rf->type = mapT(mapT, rf->type);
        } else if (auto* rc = curr->dynCast<RefCast>()) {
          rc->type = mapT(mapT, rc->type);
        } else if (auto* ci = curr->dynCast<CallIndirect>()) {
          auto it = mapT(mapT, Type(ci->heapType, Nullable));
          ci->heapType = it.getHeapType();
        }
      }
    };

    ModuleUtils::ParallelFunctionAnalysis<int, Mutable>(
      *module, [&](Function* func, int&) {
        if (func->imported()) {
          return;
        }
        ExprTypeUpdater updater(mapType, func);
        updater.walk(func->body);
        ReFinalize().walkFunctionInModule(func, module);
      });
  }

  bool foldIdentityThunks(Module* module) {
    bool anyChanged = false;

    // Step 2: Fold same-signature 0-local identity forwarding thunks
    // (thunk->type == target->type) to fixpoint.
    while (true) {
      std::map<Name, Name> replacements;
      std::unordered_set<Name> duplicates;
      for (auto& fn : module->functions) {
        Function* thunk = fn.get();
        Name targetName = getIdentityThunkTarget(module, thunk);
        if (!targetName.is()) {
          continue;
        }
        Function* target = module->getFunction(targetName);
        if (thunk->type != target->type) {
          continue;
        }
        // Resolve target through existing replacements and guard against cycles.
        Name finalTarget = targetName;
        std::unordered_set<Name> visited = {thunk->name};
        bool cycle = false;
        while (true) {
          if (!visited.insert(finalTarget).second) {
            cycle = true;
            break;
          }
          auto it = replacements.find(finalTarget);
          if (it == replacements.end()) {
            break;
          }
          finalTarget = it->second;
        }
        if (cycle || finalTarget == thunk->name) {
          continue;
        }
        for (auto& [_, to] : replacements) {
          if (to == thunk->name) {
            to = finalTarget;
          }
        }
        replacements[thunk->name] = finalTarget;
        duplicates.insert(thunk->name);
      }

      if (replacements.empty()) {
        break;
      }
      module->removeFunctions(
        [&](Function* func) { return duplicates.contains(func->name); });
      OptUtils::replaceFunctions(getPassRunner(), *module, replacements);
      anyChanged = true;
    }

    // Collect escape status and direct call sites for Steps 3 & 4.
    std::unordered_set<Name> escapedFuncs;
    if (module->start.is()) {
      escapedFuncs.insert(module->start);
    }
    for (auto& exp : module->exports) {
      if (exp->kind == ExternalKind::Function) {
        escapedFuncs.insert(*exp->getInternalName());
      }
    }
    for (auto func : Intrinsics(*module).getJSCalledFunctions()) {
      escapedFuncs.insert(func);
    }
    for (auto& fn : module->functions) {
      if (fn->imported()) {
        escapedFuncs.insert(fn->name);
      }
    }
    for (auto& g : module->globals) {
      if (!g->imported() && g->init) {
        for (auto* rf : FindAll<RefFunc>(g->init).list) {
          escapedFuncs.insert(rf->func);
        }
      }
    }
    for (auto& seg : module->elementSegments) {
      for (auto* expr : seg->data) {
        for (auto* rf : FindAll<RefFunc>(expr).list) {
          escapedFuncs.insert(rf->func);
        }
      }
    }

    struct FuncCallScan {
      std::vector<Name> refFuncs;
      std::vector<Call*> calls;
    };
    ModuleUtils::ParallelFunctionAnalysis<FuncCallScan> callScanner(
      *module, [&](Function* func, FuncCallScan& scan) {
        if (func->imported()) {
          return;
        }
        for (auto* rf : FindAll<RefFunc>(func->body).list) {
          scan.refFuncs.push_back(rf->func);
        }
        for (auto* call : FindAll<Call>(func->body).list) {
          if (!Intrinsics(*module).isCallWithoutEffects(call)) {
            scan.calls.push_back(call);
          }
        }
      });

    std::unordered_map<Name, std::vector<std::pair<Function*, Call*>>> callsTo;
    for (auto& fn : module->functions) {
      if (fn->imported()) {
        continue;
      }
      Function* caller = fn.get();
      auto& scan = callScanner.map[caller];
      for (auto name : scan.refFuncs) {
        escapedFuncs.insert(name);
      }
      for (auto* call : scan.calls) {
        callsTo[call->target].push_back({caller, call});
      }
    }

    // Step 3: Direct-call bypassing for subtype-compatible identity thunks
    // (Type::isSubType(thunk->getParams(), target->getParams()) &&
    //  Type::isSubType(target->getResults(), thunk->getResults())).
    std::unordered_set<Function*> callersToRefinalize;
    std::unordered_set<Name> deadThunks;
    for (auto& fn : module->functions) {
      Function* thunk = fn.get();
      Name targetName = getIdentityThunkTarget(module, thunk);
      if (!targetName.is()) {
        continue;
      }
      Function* target = module->getFunction(targetName);
      if (thunk->type == target->type) {
        continue;
      }
      if (!Type::isSubType(thunk->getParams(), target->getParams()) ||
          !Type::isSubType(target->getResults(), thunk->getResults())) {
        continue;
      }
      auto callIt = callsTo.find(thunk->name);
      if (callIt != callsTo.end() && !callIt->second.empty()) {
        Type targetRes = target->getResults();
        bool resChanged = targetRes != thunk->getResults();
        for (auto& [caller, call] : callIt->second) {
          if (caller == thunk) {
            continue;
          }
          call->target = target->name;
          callsTo[target->name].push_back({caller, call});
          if (resChanged && call->type != Type::unreachable) {
            call->type = targetRes;
            callersToRefinalize.insert(caller);
          }
          anyChanged = true;
        }
        callIt->second.clear();
      }
      if (!escapedFuncs.count(thunk->name)) {
        deadThunks.insert(thunk->name);
      }
    }

    if (!deadThunks.empty()) {
      for (auto& fn : module->functions) {
        if (deadThunks.count(fn->name)) {
          callersToRefinalize.erase(fn.get());
          Name targetName = getIdentityThunkTarget(module, fn.get());
          if (targetName.is()) {
            auto& vec = callsTo[targetName];
            vec.erase(
              std::remove_if(vec.begin(),
                             vec.end(),
                             [&](const std::pair<Function*, Call*>& p) {
                               return p.first == fn.get();
                             }),
              vec.end());
          }
        }
      }
      module->removeFunctions(
        [&](Function* func) { return deadThunks.contains(func->name); });
      anyChanged = true;
    }

    for (auto* caller : callersToRefinalize) {
      ReFinalize().walkFunctionInModule(caller, module);
    }

    // Step 4: Target signature narrowing for escaped identity thunks calling
    // non-escaped targets whose body and direct callers already satisfy
    // thunk->type.
    std::map<Name, Name> narrowedReplacements;
    std::unordered_set<Name> narrowedDuplicates;
    std::unordered_set<Function*> narrowedCallersToRefinalize;
    for (auto& fn : module->functions) {
      Function* thunk = fn.get();
      if (!escapedFuncs.count(thunk->name)) {
        continue;
      }
      auto thunkCallsIt = callsTo.find(thunk->name);
      if (thunkCallsIt != callsTo.end() && !thunkCallsIt->second.empty()) {
        continue;
      }
      Name targetName = getIdentityThunkTarget(module, thunk);
      if (!targetName.is()) {
        continue;
      }
      Function* target = module->getFunction(targetName);
      if (target->imported() || escapedFuncs.count(target->name)) {
        // If target was already narrowed to the exact same type by an earlier
        // thunk in this loop, we can fold this thunk into target too.
        if (!target->imported() && target->type == thunk->type &&
            !narrowedDuplicates.count(target->name)) {
          narrowedReplacements[thunk->name] = target->name;
          narrowedDuplicates.insert(thunk->name);
        }
        continue;
      }
      Signature newSig = thunk->getSig();
      Signature oldSig = target->getSig();
      if (newSig.params.size() != oldSig.params.size()) {
        continue;
      }
      Index numParams = newSig.params.size();
      bool compatible = true;
      for (Index i = 0; i < numParams; ++i) {
        if (!Type::isSubType(newSig.params[i], oldSig.params[i])) {
          compatible = false;
          break;
        }
      }
      if (!compatible) {
        continue;
      }
      if (newSig.results != oldSig.results &&
          !Type::isSubType(newSig.results, oldSig.results)) {
        continue;
      }

      // Check LocalSets to narrowed parameters inside target->body.
      for (auto* set : FindAll<LocalSet>(target->body).list) {
        if (set->index < numParams &&
            newSig.params[set->index] != oldSig.params[set->index]) {
          if (!Type::isSubType(set->value->type, newSig.params[set->index])) {
            compatible = false;
            break;
          }
        }
      }
      if (!compatible) {
        continue;
      }

      // Check target's return values against newSig.results.
      if (newSig.results != oldSig.results) {
        if (!Type::isSubType(target->body->type, newSig.results)) {
          continue;
        }
        for (auto* ret : FindAll<Return>(target->body).list) {
          if (ret->value && !Type::isSubType(ret->value->type, newSig.results)) {
            compatible = false;
            break;
          }
        }
        if (!compatible) {
          continue;
        }
      }

      // Check all direct callers of target (excluding thunk itself).
      Index castsNeeded = 0;
      auto targetCallsIt = callsTo.find(target->name);
      if (targetCallsIt != callsTo.end()) {
        for (auto& [caller, call] : targetCallsIt->second) {
          if (caller == thunk || narrowedDuplicates.count(caller->name)) {
            continue;
          }
          if (call->operands.size() != numParams) {
            compatible = false;
            break;
          }
          for (Index i = 0; i < numParams; ++i) {
            if (newSig.params[i] != oldSig.params[i] &&
                !Type::isSubType(call->operands[i]->type, newSig.params[i])) {
              if (getPassOptions().trapsNeverHappen &&
                  newSig.params[i].isRef() && !newSig.params[i].isExact() &&
                  call->operands[i]->type.isRef() &&
                  Type::isSubType(newSig.params[i], call->operands[i]->type) &&
                  castsNeeded < 2) {
                castsNeeded++;
              } else {
                compatible = false;
                break;
              }
            }
          }
          if (!compatible) {
            break;
          }
        }
      }
      if (!compatible) {
        continue;
      }

      if (castsNeeded > 0 && targetCallsIt != callsTo.end()) {
        Builder builder(*module);
        for (auto& [caller, call] : targetCallsIt->second) {
          if (caller == thunk || narrowedDuplicates.count(caller->name)) {
            continue;
          }
          for (Index i = 0; i < numParams; ++i) {
            if (newSig.params[i] != oldSig.params[i] &&
                !Type::isSubType(call->operands[i]->type, newSig.params[i])) {
              call->operands[i] =
                builder.makeRefCast(call->operands[i], newSig.params[i]);
              narrowedCallersToRefinalize.insert(caller);
            }
          }
        }
      }

      // Narrow target->type to thunk->type and fold thunk into target.
      target->type = thunk->type;
      for (auto* get : FindAll<LocalGet>(target->body).list) {
        if (get->index < numParams) {
          get->type = target->getLocalType(get->index);
        }
      }
      for (auto* set : FindAll<LocalSet>(target->body).list) {
        if (set->index < numParams && set->type != Type::none &&
            set->type != Type::unreachable) {
          set->type = target->getLocalType(set->index);
        }
      }
      ReFinalize().walkFunctionInModule(target, module);

      if (newSig.results != oldSig.results && targetCallsIt != callsTo.end()) {
        for (auto& [caller, call] : targetCallsIt->second) {
          if (caller == thunk || narrowedDuplicates.count(caller->name)) {
            continue;
          }
          if (call->type != Type::unreachable) {
            call->type = newSig.results;
            narrowedCallersToRefinalize.insert(caller);
          }
        }
      }

      escapedFuncs.insert(target->name);
      narrowedReplacements[thunk->name] = target->name;
      narrowedDuplicates.insert(thunk->name);
    }

    if (!narrowedReplacements.empty()) {
      for (auto& fn : module->functions) {
        if (narrowedDuplicates.count(fn->name)) {
          narrowedCallersToRefinalize.erase(fn.get());
        }
      }
      for (auto* caller : narrowedCallersToRefinalize) {
        ReFinalize().walkFunctionInModule(caller, module);
      }
      module->removeFunctions([&](Function* func) {
        return narrowedDuplicates.contains(func->name);
      });
      OptUtils::replaceFunctions(
        getPassRunner(), *module, narrowedReplacements);
      anyChanged = true;
    }

    return anyChanged;
  }

  void run(Module* module) override {
    // Multiple iterations may be necessary: A and B may be identical only after
    // we see the functions C1 and C2 that they call are in fact identical.
    // Rarely, such "chains" can be very long, so we limit how many we do.
    auto& options = getPassOptions();
    bool isFinalPassStage = true;
    if (auto* runner = getPassRunner()) {
      bool seenSelf = false;
      for (const auto& p : runner->passes) {
        if (p.get() == this) {
          seenSelf = true;
          continue;
        }
        if (!seenSelf) {
          continue;
        }
        if (p->name == "duplicate-function-elimination" ||
            p->name == "inlining" || p->name == "inlining-optimizing" ||
            p->name == "dae" || p->name == "dae-optimizing" ||
            p->name == "signature-pruning" || p->name == "signature-refining" ||
            p->name == "relax-nullability" || p->name == "outline-expressions" ||
            p->name == "optimize-j2cl" || p->name == "gufa" ||
            p->name == "cfp-reftest" || p->name == "cfp") {
          isFinalPassStage = false;
          break;
        }
      }
    }
    bool foldThunks =
      hasArgument("duplicate-function-elimination") ||
      hasArgument("duplicate-function-elimination-fold-thunks") ||
      (isFinalPassStage &&
       (options.shrinkLevel >= 1 ||
        (options.optimizeLevel >= 3 && getPassRunner() &&
         getPassRunner()->passes.size() == 1)));
    if (foldThunks) {
      unifyPublicPrivateSignatures(module);
    }
    Index limit;
    if (options.optimizeLevel >= 3 || options.shrinkLevel >= 1) {
      limit = module->functions.size(); // no limit
    } else if (options.optimizeLevel >= 2) {
      // 10 passes usually does most of the work, as this is typically
      // logarithmic
      limit = 10;
    } else {
      limit = 1;
    }
    while (limit > 0) {
      limit--;
      bool changed = false;
      if (foldThunks && foldIdentityThunks(module)) {
        changed = true;
      }
      // Hash all the functions
      auto hashes = FunctionHasher::createMap(module);
      auto literalHashes = FunctionHasher::createMap(module);
      FunctionHasher(&hashes, &literalHashes).run(getPassRunner(), module);
      // Find hash-equal groups
      std::map<uint32_t, std::vector<Function*>> hashGroups;
      ModuleUtils::iterDefinedFunctions(*module, [&](Function* func) {
        uint32_t h = hashes[func];
        hashGroups[h].push_back(func);
        uint32_t litH = literalHashes[func];
        if (litH != uint32_t(hash(0)) && litH != h) {
          hashGroups[litH].push_back(func);
        }
      });
      // Find actually equal functions and prepare to replace them
      std::map<Name, Name> replacements;
      std::unordered_set<Name> duplicates;
      for (auto& [_, group] : hashGroups) {
        Index size = group.size();
        if (size == 1) {
          continue;
        }
        // The groups should be fairly small, and even if a group is large we
        // should have almost all of them identical, so we should not hit actual
        // O(N^2) here unless the hash is quite poor.
        for (Index i = 0; i < size - 1; i++) {
          auto* first = group[i];
          if (duplicates.contains(first->name)) {
            continue;
          }
          for (Index j = i + 1; j < size; j++) {
            auto* second = group[j];
            if (second == first || duplicates.contains(second->name)) {
              continue;
            }
            if (FunctionUtils::equal(first, second)) {
              // great, we can replace the second with the first!
              replacements[second->name] = first->name;
              duplicates.insert(second->name);
            }
          }
        }
      }
      // perform replacements
      if (replacements.size() > 0) {
        // remove the duplicates
        module->removeFunctions(
          [&](Function* func) { return duplicates.contains(func->name); });
        OptUtils::replaceFunctions(getPassRunner(), *module, replacements);
        changed = true;
      }
      if (!changed) {
        break;
      }
    }
  }
};

Pass* createDuplicateFunctionEliminationPass() {
  return new DuplicateFunctionElimination();
}

} // namespace wasm
