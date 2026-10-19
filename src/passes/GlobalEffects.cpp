/*
 * Copyright 2022 WebAssembly Community Group participants
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
// Handle the computation of global effects. The effects are stored on the
// PassOptions structure; see more details there.
//

#include "ir/bits.h"
#include "ir/effects.h"
#include "ir/find_all.h"
#include "ir/intrinsics.h"
#include "ir/module-utils.h"
#include "ir/subtypes.h"
#include "pass.h"
#include "passes/string-utils.h"
#include "support/graph_traversal.h"
#include "support/strongly_connected_components.h"
#include "support/utilities.h"
#include "wasm.h"

namespace wasm {

namespace {

struct FuncInfo {
  // Effects in this function. nullopt means that we don't know what effects
  // this function has, so we conservatively assume all effects.
  // Nullopt cases won't be copied to Function::effects.
  std::optional<EffectAnalyzer> effects;

  // Directly-called functions from this function.
  std::unordered_set<Name> calledFunctions;

  // Types that are targets of indirect calls.
  std::unordered_set<HeapType> indirectCalledTypes;
};

// Only funcs that are referenced may be the target of an indirect call. A
// function is referenced if:
// - It appears in a ref.func expression (this includes `elem` statements
// because of how our IR is represented).
// - It's exported, because it may flow back to us as a reference.
//
// If a function doesn't meet any of these criteria, it can't be the target of
// an indirect call and we don't need to include its effects in indirect calls.
std::unordered_set<Function*> getReferencedFuncs(Module& module,
                                                 PassRunner& passRunner) {
  struct AddressedFuncsWalker : WalkerPass<PostWalker<AddressedFuncsWalker>> {
    // For each function, which functions are referenced in its body.
    // The key for `nullptr` contains references that are not in a function
    // (e.g. `elem` segments).
    std::unordered_map<Function*, std::unordered_set<Function*>>&
      allReferencedFuncs;
    // Points to `allReferencedFuncs`.
    std::unordered_set<Function*>* referencedFuncs = nullptr;

    AddressedFuncsWalker(
      std::unordered_map<Function*, std::unordered_set<Function*>>&
        allReferencedFuncs)
      : allReferencedFuncs(allReferencedFuncs),
        referencedFuncs(&allReferencedFuncs[nullptr]) {}

    std::unique_ptr<Pass> create() override {
      return std::make_unique<AddressedFuncsWalker>(allReferencedFuncs);
    }

    bool isFunctionParallel() override { return true; }

    bool modifiesBinaryenIR() override { return false; }

    void doWalkFunction(Function* func) {
      referencedFuncs = &allReferencedFuncs.at(func);
      walk(func->body);
    }

    void visitRefFunc(RefFunc* refFunc) {
      referencedFuncs->insert(getModule()->getFunction(refFunc->func));
    }
  };

  std::unordered_map<Function*, std::unordered_set<Function*>>
    allReferencedFuncs;
  for (auto& func : module.functions) {
    allReferencedFuncs[func.get()];
  }

  AddressedFuncsWalker walker(allReferencedFuncs);
  walker.run(&passRunner, &module);
  walker.runOnModuleCode(&passRunner, &module);

  std::unordered_set<Function*> mergedReferencedFuncs;
  for (auto& [_, referencedFuncs] : allReferencedFuncs) {
    mergedReferencedFuncs.merge(referencedFuncs);
  }

  for (const auto& export_ : module.exports) {
    if (export_->kind != ExternalKind::Function) {
      continue;
    }

    mergedReferencedFuncs.insert(
      module.getFunction(*export_->getInternalName()));
  }

  return mergedReferencedFuncs;
}

std::map<Function*, FuncInfo> analyzeFuncs(Module& module,
                                           const PassOptions& passOptions) {
  ModuleUtils::ParallelFunctionAnalysis<FuncInfo> analysis(
    module, [&](Function* func, FuncInfo& funcInfo) {
      if (func->imported()) {
        if (func->module == WasmStringsModule) {
          auto setImplicitTrap = [&](EffectAnalyzer& eff) {
            if (!passOptions.ignoreImplicitTraps) {
              eff.implicitTrap = true;
              eff.trap = true;
            }
          };
          if (func->base == "equals" || func->base == "test") {
            funcInfo.effects.emplace(passOptions, module);
            funcInfo.effects->isIdempotent = true;
            funcInfo.effects->onlyTouchesInitGlobals = true;
            funcInfo.effects->returnsOneBit = true;
            return;
          }
          if (func->base == "compare" || func->base == "length" ||
              func->base == "charCodeAt" || func->base == "codePointAt" ||
              func->base == "cast") {
            funcInfo.effects.emplace(passOptions, module);
            setImplicitTrap(*funcInfo.effects);
            funcInfo.effects->isIdempotent = true;
            funcInfo.effects->onlyTouchesInitGlobals = true;
            return;
          }
          if (func->base == "fromCodePoint" || func->base == "fromCharCode" ||
              func->base == "concat" || func->base == "substring") {
            funcInfo.effects.emplace(passOptions, module);
            setImplicitTrap(*funcInfo.effects);
            funcInfo.effects->generative = true;
            return;
          }
          if (func->base == "fromCharCodeArray") {
            funcInfo.effects.emplace(passOptions, module);
            setImplicitTrap(*funcInfo.effects);
            funcInfo.effects->readsMutableArray = true;
            funcInfo.effects->generative = true;
            return;
          }
          if (func->base == "intoCharCodeArray") {
            funcInfo.effects.emplace(passOptions, module);
            setImplicitTrap(*funcInfo.effects);
            funcInfo.effects->writesArray = true;
            return;
          }
        }
        // Imports can do almost anything, so we need to assume the worst
        // anyhow, which is the same as not specifying any effects for them in
        // the map (which we do by not setting funcInfo.effects).
        //
        // TODO: We can be more precise here since imports can't mutate
        // globals/tables/memories that aren't imported or exported.
        return;
      }

      // Gather the effects.
      funcInfo.effects.emplace(passOptions, module, func);

      if (funcInfo.effects->calls) {
        // There are calls in this function, which we will analyze in detail.
        // Clear the |calls| field first, and we'll handle calls of all sorts
        // below.
        funcInfo.effects->calls = false;

        // Clear throws, suspends, and generative as well, as we are
        // "forgetting" calls right now, and want to forget their throwing,
        // suspending, and generative effects as well. If we see something else
        // that throws, suspends, or allocates, below, then we'll note that
        // there.
        funcInfo.effects->throws_ = false;
        funcInfo.effects->suspends = false;
        funcInfo.effects->generative = false;

        struct CallScanner
          : public PostWalker<CallScanner,
                              UnifiedExpressionVisitor<CallScanner>> {
          Module& wasm;
          const PassOptions& options;
          FuncInfo& funcInfo;

          CallScanner(Module& wasm,
                      const PassOptions& options,
                      FuncInfo& funcInfo)
            : wasm(wasm), options(options), funcInfo(funcInfo) {}

          void visitExpression(Expression* curr) {
            ShallowEffectAnalyzer effects(options, wasm, curr);
            if (auto* call = curr->dynCast<Call>()) {
              // Note the direct call.
              funcInfo.calledFunctions.insert(call->target);
            } else if (effects.calls &&
                       options.worldMode == WorldMode::Closed) {
              HeapType type;
              if (auto* callRef = curr->dynCast<CallRef>()) {
                // call_ref on unreachable does not have a call effect,
                // so this must be a HeapType.
                type = callRef->target->type.getHeapType();
              } else if (auto* callIndirect = curr->dynCast<CallIndirect>()) {
                type = callIndirect->heapType;
              } else {
                funcInfo.effects = std::nullopt;
                return;
              }

              funcInfo.indirectCalledTypes.insert(type);
            } else if (effects.calls) {
              assert(options.worldMode == WorldMode::Open);
              funcInfo.effects = std::nullopt;
            } else {
              // No call here, but update throwing, suspending, and generative
              // if we see it. (Only do so, however, if we have effects; if we
              // cleared it - see before - then we assume the worst anyhow, and
              // have nothing to update.)
              if (effects.throws_ && funcInfo.effects) {
                funcInfo.effects->throws_ = true;
              }
              if (effects.suspends && funcInfo.effects) {
                funcInfo.effects->suspends = true;
              }
              if (effects.generative && funcInfo.effects) {
                funcInfo.effects->generative = true;
              }
            }
          }
        };
        CallScanner scanner(module, passOptions, funcInfo);
        scanner.walkFunction(func);
      }
    });

  return std::move(analysis.map);
}

using CallGraphNode = std::variant<Function*, HeapType>;

// Call graph for indirect and direct calls.
//
// key (caller) -> value (callee)
// Function  -> Function : direct call
// Function  -> HeapType : indirect call to the given HeapType
// HeapType  -> Function : The function `callee` has the type `caller`. The
//                         HeapType may essentially 'call' any of its
//                         potential implementations.
// HeapType  -> HeapType : `callee` is a subtype of `caller`. A call_ref
//                         could target any subtype of the ref, so we need to
//                         aggregate effects of subtypes of the target type.
//
// If we're running in an open world, we only include Function -> Function
// edges, and don't compute effects for indirect calls, conservatively assuming
// the worst.
using CallGraph =
  std::unordered_map<CallGraphNode, std::unordered_set<CallGraphNode>>;

CallGraph buildCallGraph(const Module& module,
                         const std::map<Function*, FuncInfo>& funcInfos,
                         const std::unordered_set<Function*>& referencedFuncs,
                         WorldMode worldMode) {
  CallGraph callGraph;
  if (worldMode == WorldMode::Open) {
    for (const auto& [caller, callerInfo] : funcInfos) {
      auto& callees = callGraph[caller];

      // Function -> Function
      for (Name calleeFunction : callerInfo.calledFunctions) {
        callees.insert(module.getFunction(calleeFunction));
      }
    }

    return callGraph;
  }

  std::unordered_set<HeapType> allFunctionTypes;
  for (const auto& [caller, callerInfo] : funcInfos) {
    auto& callees = callGraph[caller];

    // Function -> Function
    for (Name calleeFunction : callerInfo.calledFunctions) {
      callees.insert(module.getFunction(calleeFunction));
    }

    // Function -> Type
    allFunctionTypes.insert(caller->type.getHeapType());
    for (HeapType calleeType : callerInfo.indirectCalledTypes) {
      callees.insert(calleeType);

      // Add the key to ensure the lookup doesn't fail for indirect calls to
      // uninhabited types.
      callGraph[calleeType];
    }

    // Type -> Function
    if (referencedFuncs.contains(caller)) {
      callGraph[caller->type.getHeapType()].insert(caller);
    }
  }

  // Type -> Type
  // Do a DFS up the type hierarchy for all function implementations.
  // We are essentially walking up each supertype chain and adding edges from
  // super -> subtype, but doing it via DFS to avoid repeated work.
  Graph superTypeGraph(allFunctionTypes.begin(),
                       allFunctionTypes.end(),
                       [&callGraph](const auto& push, HeapType t) {
                         // Not needed except that during lookup we expect the
                         // key to exist.
                         callGraph[t];

                         if (auto super = t.getDeclaredSuperType()) {
                           callGraph[*super].insert(t);
                           push(*super);
                         }
                       });
  (void)superTypeGraph.traverseDepthFirst();

  // Add Type -> Function edges to account for inexact imports. For (ref.func)
  // on a *defined* function, we know its exact type and can add a single
  // Type -> Function edge in the graph (done above). We know that indirect
  // calls to strict subtypes of the function can't reach the function.
  //
  // OTOH for inexactly imported functions, they may be downcasted to a subtype.
  // To account for this, add Type -> Function edges to all subtypes for
  // inexactly imported functions.
  SubTypes subtypes(module);
  ModuleUtils::iterImportedFunctions(module, [&](Function* func) {
    if (func->type.isExact()) {
      return;
    }
    if (!referencedFuncs.contains(func)) {
      return;
    }

    subtypes.iterSubTypes(func->type.getHeapType(), [&](auto subtype, int _) {
      callGraph[subtype].insert(func);
      return true;
    });
  });

  return callGraph;
}

constexpr auto UnknownEffects = nullptr;

static bool SubtreeTargetsName(Expression* root, Name target) {
  if (!target.is() || !root) {
    return false;
  }
  struct Finder : public PostWalker<Finder> {
    Name target;
    bool found = false;
    Finder(Name target) : target(target) {}
    void visitBreak(Break* curr) {
      if (curr->name == target) {
        found = true;
      }
    }
    void visitSwitch(Switch* curr) {
      if (curr->default_ == target) {
        found = true;
        return;
      }
      for (auto t : curr->targets) {
        if (t == target) {
          found = true;
          return;
        }
      }
    }
    void visitBrOn(BrOn* curr) {
      if (curr->name == target) {
        found = true;
      }
    }
    void visitTryTable(TryTable* curr) {
      for (auto t : curr->catchDests) {
        if (t == target) {
          found = true;
          return;
        }
      }
    }
  } finder(target);
  finder.walk(root);
  return finder.found;
}

static bool exprDoesNotReturn(Expression* e, const Module& module) {
  if (!e) {
    return false;
  }
  if (e->type == Type::unreachable) {
    return true;
  }
  if (auto* b = e->dynCast<Block>()) {
    for (auto* elem : b->list) {
      if (SubtreeTargetsName(elem, b->name)) {
        return false;
      }
      if (exprDoesNotReturn(elem, module)) {
        return true;
      }
    }
    return false;
  }
  if (auto* iff = e->dynCast<If>()) {
    if (exprDoesNotReturn(iff->condition, module)) {
      return true;
    }
    return iff->ifFalse && exprDoesNotReturn(iff->ifTrue, module) &&
           exprDoesNotReturn(iff->ifFalse, module);
  }
  if (auto* d = e->dynCast<Drop>()) {
    return exprDoesNotReturn(d->value, module);
  }
  if (auto* ls = e->dynCast<LocalSet>()) {
    return exprDoesNotReturn(ls->value, module);
  }
  if (auto* call = e->dynCast<Call>()) {
    if (auto* target = module.getFunctionOrNull(call->target)) {
      if (target->effects && target->effects->mustNotReturn) {
        return true;
      }
    }
    for (auto* op : call->operands) {
      if (exprDoesNotReturn(op, module)) {
        return true;
      }
    }
    return false;
  }
  return false;
}

// Merges effects from another connected component (const EffectAnalyzer*) or a
// function (std::optional<EffectAnalyzer>&).
template<typename EffectAnalyzerPtr>
void mergeMaybeEffects(std::shared_ptr<EffectAnalyzer>& dest,
                       const EffectAnalyzerPtr& src) {
  if (dest == UnknownEffects) {
    return;
  }
  if (!src) {
    dest = UnknownEffects;
    return;
  }

  dest->mergeIn(*src);
}

// Propagate effects from callees to callers transitively and populate direct
// and indirect call effects. e.g. if A -> B -> C (A calls B which calls C),
// then B inherits effects from C and A inherits effects from both B and C.
//
// Generate SCC for the call graph, then traverse it in reverse topological
// order processing each callee before its callers. When traversing:
// - Merge all of the effects of functions within the CC
// - Also merge the (already computed) effects of each callee CC
// - Add trap effects for potentially recursive call chains
void propagateEffects(
  const Module& module,
  const PassOptions& passOptions,
  std::map<Function*, FuncInfo>& funcInfos,
  std::unordered_map<HeapType, std::shared_ptr<const EffectAnalyzer>>&
    indirectCallEffects,
  const CallGraph& callGraph) {
  // We only care about Functions that are roots, not types.
  // A type would be a root if a function exists with that type, but no-one
  // indirect calls the type.
  std::vector<CallGraphNode> funcNodes;
  for (const auto& [node, _] : callGraph) {
    if (std::holds_alternative<Function*>(node)) {
      funcNodes.push_back(node);
    }
  }

  struct CallGraphSCCs
    : SCCs<std::vector<CallGraphNode>::iterator, CallGraphSCCs> {

    const std::map<Function*, FuncInfo>& funcInfos;
    const CallGraph& callGraph;
    const Module& module;

    CallGraphSCCs(std::vector<CallGraphNode>& nodes,
                  const std::map<Function*, FuncInfo>& funcInfos,
                  const CallGraph& callGraph,
                  const Module& module)
      : SCCs<std::vector<CallGraphNode>::iterator, CallGraphSCCs>(nodes.begin(),
                                                                  nodes.end()),
        funcInfos(funcInfos), callGraph(callGraph), module(module) {}

    void pushChildren(CallGraphNode node) {
      for (CallGraphNode callee : callGraph.at(node)) {
        push(callee);
      }
    }
  };
  CallGraphSCCs sccs(funcNodes, funcInfos, callGraph, module);

  std::unordered_map<Name, Index> mutableGlobalFuncCount;
  bool hasUnknownFuncEffects = false;
  for (const auto& [fn, info] : funcInfos) {
    if (!fn->imported() && !info.effects) {
      hasUnknownFuncEffects = true;
    }
    if (info.effects) {
      for (Name g : info.effects->mutableGlobalsRead) {
        mutableGlobalFuncCount[g]++;
      }
      for (Name g : info.effects->globalsWritten) {
        if (!info.effects->mutableGlobalsRead.contains(g)) {
          mutableGlobalFuncCount[g]++;
        }
      }
    }
  }
  for (const auto& ex : module.exports) {
    if (ex->kind == ExternalKind::Global) {
      mutableGlobalFuncCount[*ex->getInternalName()] += 2;
    }
  }

  auto unwrapRefAsNonNull = [](Expression* e) -> Expression* {
    while (auto* ra = e->dynCast<RefAs>()) {
      if (ra->op == RefAsNonNull) {
        e = ra->value;
      } else {
        break;
      }
    }
    return e;
  };

  auto extractArrayGetOperands =
    [&](Expression* e, Expression*& refOut, Expression*& idxOut) -> bool {
    e = unwrapRefAsNonNull(e);
    if (auto* ag = e->dynCast<ArrayGet>()) {
      refOut = ag->ref;
      idxOut = ag->index;
      return true;
    }
    if (auto* call = e->dynCast<Call>()) {
      if (call->operands.size() == 2) {
        auto* target = module.getFunctionOrNull(call->target);
        if (target && !target->imported() && target->getNumParams() == 2 &&
            target->body) {
          Expression* tBody = unwrapRefAsNonNull(target->body);
          if (auto* ag = tBody->dynCast<ArrayGet>()) {
            auto* p0 = unwrapRefAsNonNull(ag->ref)->dynCast<LocalGet>();
            auto* p1 = ag->index->dynCast<LocalGet>();
            if (p0 && p1 && p0->index == 0 && p1->index == 1) {
              refOut = call->operands[0];
              idxOut = call->operands[1];
              return true;
            }
          }
        }
      }
    }
    return false;
  };

  auto isCachedFactoryFunc =
    [&](Function* f, std::unordered_set<Name>& restoredGlobals) -> bool {
    if (!f->getResults().isRef() || !f->body) {
      return false;
    }
    Expression* body = unwrapRefAsNonNull(f->body);
    auto* b = body->dynCast<Block>();
    if (!b || b->list.empty()) {
      return false;
    }
    if (!b->name.is()) {
      if (f->getNumParams() == 0 && b->list.size() == 2) {
        auto* iff = b->list[0]->dynCast<If>();
        auto* retGlobal = unwrapRefAsNonNull(b->list[1])->dynCast<GlobalGet>();
        if (iff && !iff->ifFalse && retGlobal) {
          if (auto* rn = iff->condition->dynCast<RefIsNull>()) {
            if (auto* condGlobal = rn->value->dynCast<GlobalGet>()) {
              auto* cg = module.getGlobalOrNull(retGlobal->name);
              if (condGlobal->name == retGlobal->name && cg &&
                  !cg->imported() && cg->mutable_ && cg->init->is<RefNull>() &&
                  mutableGlobalFuncCount[retGlobal->name] <= 2) {
                return true;
              }
            }
          }
        }
      }
      Expression* last = unwrapRefAsNonNull(b->list.back());
      auto* inner = last->dynCast<Block>();
      if (!inner) {
        return false;
      }
      for (Index i = 0; i + 1 < b->list.size(); i++) {
        if (!b->list[i]->is<If>()) {
          return false;
        }
      }
      b = inner;
    }
    if (!b->name.is() || b->list.size() < 2) {
      return false;
    }
    auto* brOn = b->list[0]->dynCast<BrOn>();
    if (!brOn || brOn->op != BrOnNonNull || brOn->name != b->name) {
      return false;
    }
    Expression* retExpr = unwrapRefAsNonNull(b->list.back());
    auto* retGet = retExpr->dynCast<LocalGet>();
    Expression* setStmt = b->list[b->list.size() - 2];
    if (f->getNumParams() == 1 && f->getParams() == Type::i32) {
      if (!retGet) {
        return false;
      }
      Expression* agRef = nullptr;
      Expression* agIdx = nullptr;
      auto* as = setStmt->dynCast<ArraySet>();
      if (!extractArrayGetOperands(brOn->ref, agRef, agIdx) || !as) {
        return false;
      }
      auto* gg1 = unwrapRefAsNonNull(agRef)->dynCast<GlobalGet>();
      auto* lg1 = agIdx->dynCast<LocalGet>();
      auto* gg2 = unwrapRefAsNonNull(as->ref)->dynCast<GlobalGet>();
      auto* lg2 = as->index->dynCast<LocalGet>();
      auto* lt = as->value->dynCast<LocalSet>();
      if (!gg1 || !lg1 || !gg2 || !lg2 || !lt || !lt->isTee()) {
        return false;
      }
      auto* cg = module.getGlobalOrNull(gg1->name);
      return cg && !cg->imported() && !cg->mutable_ &&
             cg->init->is<ArrayNew>() && gg1->name == gg2->name &&
             lg1->index == 0 && lg2->index == 0 && lt->index == retGet->index;
    }
    if (f->getNumParams() == 0) {
      auto* gg1 = brOn->ref->dynCast<GlobalGet>();
      if (!gg1) {
        return false;
      }
      auto* cg = module.getGlobalOrNull(gg1->name);
      if (!cg || cg->imported() || !cg->mutable_ || !cg->init->is<RefNull>() ||
          mutableGlobalFuncCount[gg1->name] != 1) {
        return false;
      }
      if (retGet) {
        auto* gs = setStmt->dynCast<GlobalSet>();
        if (!gs || gg1->name != gs->name) {
          return false;
        }
        auto* lt = gs->value->dynCast<LocalSet>();
        return lt && lt->isTee() && lt->index == retGet->index;
      }
      if (auto* retGlobal = retExpr->dynCast<GlobalGet>()) {
        if (retGlobal->name != gg1->name) {
          return false;
        }
        Index gsIdx = 0;
        for (Index i = 1; i + 1 < b->list.size(); i++) {
          if (auto* gs = b->list[i]->dynCast<GlobalSet>()) {
            if (gs->name == gg1->name) {
              gsIdx = i;
            }
          }
        }
        if (gsIdx == 0) {
          return false;
        }
        for (Index k = gsIdx + 1; k + 1 < b->list.size(); k++) {
          auto* restoreGs = b->list[k]->dynCast<GlobalSet>();
          auto* restoreLg =
            restoreGs ? restoreGs->value->dynCast<LocalGet>() : nullptr;
          if (!restoreGs || !restoreLg) {
            return false;
          }
          bool foundSave = false;
          for (Index j = 1; j < gsIdx; j++) {
            if (auto* saveLs = b->list[j]->dynCast<LocalSet>()) {
              if (saveLs->index == restoreLg->index) {
                if (auto* saveGg = saveLs->value->dynCast<GlobalGet>()) {
                  if (saveGg->name == restoreGs->name) {
                    foundSave = true;
                    break;
                  }
                }
              }
            }
          }
          if (!foundSave) {
            return false;
          }
          restoredGlobals.insert(restoreGs->name);
        }
        return true;
      }
      return false;
    }
    if (f->getNumParams() == 2 && b->list.size() >= 3) {
      if (!retGet) {
        return false;
      }
      for (Index i = 1; i + 2 < b->list.size(); i++) {
        if (!b->list[i]->is<If>()) {
          return false;
        }
      }
      Expression* agRef = nullptr;
      Expression* agIdx = nullptr;
      auto* as = setStmt->dynCast<ArraySet>();
      if (!extractArrayGetOperands(brOn->ref, agRef, agIdx) || !as) {
        return false;
      }
      auto* lgArr1 = unwrapRefAsNonNull(agRef)->dynCast<LocalGet>();
      auto* lgIdx1 = agIdx->dynCast<LocalGet>();
      auto* lgArr2 = unwrapRefAsNonNull(as->ref)->dynCast<LocalGet>();
      auto* lgIdx2 = as->index->dynCast<LocalGet>();
      auto* lt = as->value->dynCast<LocalSet>();
      return lgArr1 && lgIdx1 && lgArr2 && lgIdx2 && lt && lt->isTee() &&
             lgArr1->index == 0 && lgIdx1->index == 1 && lgArr2->index == 0 &&
             lgIdx2->index == 1 && lt->index == retGet->index;
    }
    return false;
  };

  auto extractGuardGlobal = [](Expression* cond, Name& gOut) -> bool {
    if (auto* rin = cond->dynCast<RefIsNull>()) {
      if (auto* gg = rin->value->dynCast<GlobalGet>()) {
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
  };

  auto isLazyInitOrOnceFunc = [&](Function* f,
                                  const EffectAnalyzer& eff) -> bool {
    if (f->getNumParams() != 0 || !f->body) {
      return false;
    }
    if (f->getResults().isConcrete()) {
      Expression* body = unwrapRefAsNonNull(f->body);
      auto* b = body->dynCast<Block>();
      if (!b || b->list.size() != 2) {
        return false;
      }
      auto* iff = b->list[0]->dynCast<If>();
      auto* retGG = unwrapRefAsNonNull(b->list[1])->dynCast<GlobalGet>();
      Name guardG;
      if (iff && !iff->ifFalse && retGG &&
          extractGuardGlobal(iff->condition, guardG)) {
        auto* g1 = module.getGlobalOrNull(guardG);
        auto* g2 = module.getGlobalOrNull(retGG->name);
        if (g1 && !g1->imported() && g1->mutable_ && g2 && !g2->imported() &&
            g2->mutable_ && eff.globalsWritten.contains(guardG) &&
            eff.globalsWritten.contains(retGG->name)) {
          return true;
        }
      }
      return false;
    }
    if (f->getResults() == Type::none) {
      if (auto* iff = f->body->dynCast<If>()) {
        Name guardG;
        if (!iff->ifFalse && extractGuardGlobal(iff->condition, guardG)) {
          auto* g = module.getGlobalOrNull(guardG);
          if (g && !g->imported() && g->mutable_ &&
              eff.globalsWritten.contains(guardG)) {
            return true;
          }
        }
      }
      if (auto* b = f->body->dynCast<Block>()) {
        if (b->list.size() >= 2) {
          auto* gs = b->list[1]->dynCast<GlobalSet>();
          auto* c = gs ? gs->value->dynCast<Const>() : nullptr;
          if (gs && c && !c->value.isZero()) {
            if (auto* iff = b->list[0]->dynCast<If>()) {
              if (!iff->ifFalse) {
                auto* gg = iff->condition->dynCast<GlobalGet>();
                bool exits = iff->ifTrue->is<Return>();
                if (!exits) {
                  if (auto* br = iff->ifTrue->dynCast<Break>()) {
                    exits = !br->condition && br->name == b->name;
                  }
                }
                if (gg && exits && gg->name == gs->name) {
                  return true;
                }
              }
            } else if (auto* br = b->list[0]->dynCast<Break>()) {
              if (br->condition && !br->value && br->name == b->name) {
                auto* gg = br->condition->dynCast<GlobalGet>();
                if (gg && gg->name == gs->name) {
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

  std::vector<std::shared_ptr<EffectAnalyzer>> componentEffects;
  // Points to an index in componentEffects
  std::unordered_map<CallGraphNode, Index> nodeComponents;

  for (auto ccIterator : sccs) {
    auto& ccEffects = componentEffects.emplace_back(
      std::make_shared<EffectAnalyzer>(passOptions, module));
    std::vector<CallGraphNode> cc(ccIterator.begin(), ccIterator.end());

    std::vector<Function*> ccFuncs;
    for (CallGraphNode node : cc) {
      nodeComponents.emplace(node, componentEffects.size() - 1);
      if (auto** func = std::get_if<Function*>(&node)) {
        ccFuncs.push_back(*func);
      }
    }

    std::unordered_set<int> calleeSccs;
    for (CallGraphNode caller : cc) {
      for (CallGraphNode callee : callGraph.at(caller)) {
        calleeSccs.insert(nodeComponents.at(callee));
      }
    }

    // Merge in effects from callees
    for (int calleeScc : calleeSccs) {
      const auto& calleeComponentEffects = componentEffects.at(calleeScc);
      mergeMaybeEffects(ccEffects, calleeComponentEffects.get());
    }

    // Add trap effects for potential cycles.
    if (cc.size() > 1) {
      if (ccEffects != UnknownEffects) {
        ccEffects->trap = true;
      }
    } else if (ccFuncs.size() == 1) {
      // It's possible for a CC to only contain 1 type, but that is not a
      // cycle in the call graph.
      auto* func = ccFuncs[0];
      if (funcInfos.at(func).calledFunctions.contains(func->name)) {
        if (ccEffects != UnknownEffects) {
          ccEffects->trap = true;
        }
      }
    } else if (ccFuncs.empty() && calleeSccs.empty()) {
      // This node came from an indirect call to an uninhabited type.
      // This CC must consist of exactly one type, because an uninhabited type
      // can't make any indirect calls to other types.
      //
      // Since the type is uninhabited, this call must trap.
      assert(cc.size() == 1);
      ccEffects->trap = true;
    }

    // Aggregate effects within this CC
    if (ccEffects) {
      for (Function* f : ccFuncs) {
        const auto& effects = funcInfos.at(f).effects;
        mergeMaybeEffects(ccEffects, effects);
      }
    }

    if (ccEffects && cc.size() == 1 && ccFuncs.size() == 1) {
      Function* f = ccFuncs[0];
      if (f->imported()) {
        if (const auto& eff = funcInfos.at(f).effects) {
          ccEffects->isIdempotent = eff->isIdempotent;
          ccEffects->onlyTouchesInitGlobals = eff->onlyTouchesInitGlobals;
          ccEffects->returnsOneBit = eff->returnsOneBit;
        }
      } else if (f->body && !funcInfos.at(f).calledFunctions.contains(f->name)) {
        struct ReturnOrTailCallFinder
          : public PostWalker<ReturnOrTailCallFinder> {
          bool found = false;
          bool hasIndirectTailCall = false;
          std::vector<Expression*> returnValues;
          std::vector<Name> tailCallTargets;
          void visitReturn(Return* curr) {
            found = true;
            if (curr->value && curr->value->type == Type::i32) {
              returnValues.push_back(curr->value);
            }
          }
          void visitCall(Call* curr) {
            if (curr->isReturn) {
              found = true;
              tailCallTargets.push_back(curr->target);
            }
          }
          void visitCallIndirect(CallIndirect* curr) {
            if (curr->isReturn) {
              found = true;
              hasIndirectTailCall = true;
            }
          }
          void visitCallRef(CallRef* curr) {
            if (curr->isReturn) {
              found = true;
              hasIndirectTailCall = true;
            }
          }
        } finder;
        finder.walk(f->body);
        if (!finder.found && exprDoesNotReturn(f->body, module)) {
          ccEffects->mustNotReturn = true;
        }
        if (f->getResults() == Type::i32 && !finder.hasIndirectTailCall) {
          if (f->body->type == Type::i32) {
            finder.returnValues.push_back(f->body);
          }
          if (!finder.returnValues.empty() || !finder.tailCallTargets.empty()) {
            bool allOneBit = true;
            for (Name target : finder.tailCallTargets) {
              auto* callee = module.getFunctionOrNull(target);
              if (!callee || !callee->effects ||
                  !callee->effects->returnsOneBit) {
                allOneBit = false;
                break;
              }
            }
            if (allOneBit) {
              struct CallBitProvider : Bits::DummyLocalInfoProvider {
                const Module& module;
                CallBitProvider(const Module& module) : module(module) {}
                Index getMaxBitsForCall(Call* call) {
                  if (call->type == Type::i32) {
                    if (auto* callee = module.getFunctionOrNull(call->target)) {
                      if (callee->effects && callee->effects->returnsOneBit) {
                        return 1;
                      }
                    }
                  }
                  return call->type == Type::i64 ? 64 : 32;
                }
              } provider(module);
              for (auto* val : finder.returnValues) {
                if (Bits::getMaxBits(val, &provider) > 1) {
                  allOneBit = false;
                  break;
                }
              }
            }
            if (allOneBit) {
              ccEffects->returnsOneBit = true;
            }
          }
        }
      }
    }

    if (passOptions.shrinkLevel > 0 && ccEffects && cc.size() == 1 &&
        ccFuncs.size() == 1) {
      Function* f = ccFuncs[0];
      if (!f->imported() &&
          !funcInfos.at(f).calledFunctions.contains(f->name)) {
        if (Intrinsics::getAnnotations(f).idempotent) {
          ccEffects->generative = false;
          ccEffects->isIdempotent = true;
        }
        bool onlyArrayOrGlobalWritesOrColdLoop =
          (!ccEffects->writesMemory || module.memories.empty()) &&
          !ccEffects->writesSharedMemory && !ccEffects->writesTable &&
          !ccEffects->writesStruct && !ccEffects->writesSharedStruct &&
          !ccEffects->writesSharedArray && !ccEffects->throws() &&
          !ccEffects->suspends && !ccEffects->calls &&
          !ccEffects->danglingPop && !ccEffects->hasSynchronization();
        bool onlyArrayOrGlobalWrites = onlyArrayOrGlobalWritesOrColdLoop &&
                                       !ccEffects->writesMemory &&
                                       !ccEffects->mayNotReturn;
        std::unordered_set<Name> restoredGlobals;
        if (onlyArrayOrGlobalWrites && !ccEffects->hasNonTrapSideEffects() &&
            !ccEffects->generative && !ccEffects->readsMutableGlobalState()) {
          ccEffects->isIdempotent = true;
          ccEffects->onlyTouchesInitGlobals = true;
        } else if (onlyArrayOrGlobalWritesOrColdLoop &&
                   isCachedFactoryFunc(f, restoredGlobals)) {
          bool allPrivateGlobals = !hasUnknownFuncEffects;
          auto isPrivateOrLazyGlobal = [&](Name g) {
            if (mutableGlobalFuncCount[g] == 1 ||
                restoredGlobals.contains(g)) {
              return true;
            }
            auto* glob = module.getGlobalOrNull(g);
            if (!glob || glob->imported() || !glob->mutable_ || !glob->init) {
              return false;
            }
            if (glob->init->is<RefNull>()) {
              return true;
            }
            const auto& localEffects = funcInfos.at(f).effects;
            return glob->init->is<Const>() && localEffects &&
                   !localEffects->globalsWritten.contains(g) &&
                   !localEffects->mutableGlobalsRead.contains(g);
          };
          for (Name g : ccEffects->globalsWritten) {
            if (!isPrivateOrLazyGlobal(g)) {
              allPrivateGlobals = false;
              break;
            }
          }
          for (Name g : ccEffects->mutableGlobalsRead) {
            if (!isPrivateOrLazyGlobal(g)) {
              allPrivateGlobals = false;
              break;
            }
          }
          if (allPrivateGlobals && f->getNumParams() <= 2) {
            ccEffects->writesArray = false;
            ccEffects->readsMutableArray = false;
            if (module.memories.empty()) {
              ccEffects->writesMemory = false;
            }
            ccEffects->mayNotReturn = false;
            ccEffects->globalsWritten.clear();
            ccEffects->mutableGlobalsRead.clear();
          }
          ccEffects->generative = false;
          ccEffects->isIdempotent = true;
          ccEffects->onlyTouchesInitGlobals = true;
        } else if (onlyArrayOrGlobalWrites &&
                   isLazyInitOrOnceFunc(f, *ccEffects)) {
          ccEffects->generative = false;
          ccEffects->isIdempotent = true;
          ccEffects->onlyTouchesInitGlobals = true;
        }
      }
    }

    // Assign each function's effects to its CC effects.
    for (auto node : cc) {
      std::visit(overloaded{[&](HeapType type) {
                              if (ccEffects) {
                                indirectCallEffects[type] = ccEffects;
                              }
                            },
                            [&](Function* f) { f->effects = ccEffects; }},
                 node);
    }
  }
}

struct GenerateGlobalEffects : public Pass {
  void run(Module* module) override {
    for (auto& func : module->functions) {
      func->effects.reset();
    }
    module->indirectCallEffects.clear();

    std::map<Function*, FuncInfo> funcInfos =
      analyzeFuncs(*module, getPassOptions());

    auto referencedFuncs = getReferencedFuncs(*module, *getPassRunner());

    auto callGraph = buildCallGraph(
      *module, funcInfos, referencedFuncs, getPassOptions().worldMode);

    propagateEffects(*module,
                     getPassOptions(),
                     funcInfos,
                     module->indirectCallEffects,
                     callGraph);
  }
};

struct DiscardGlobalEffects : public Pass {
  void run(Module* module) override {
    for (auto& func : module->functions) {
      func->effects.reset();
    }
    module->indirectCallEffects.clear();
  }
};

} // namespace

Pass* createGenerateGlobalEffectsPass() { return new GenerateGlobalEffects(); }

Pass* createDiscardGlobalEffectsPass() { return new DiscardGlobalEffects(); }

} // namespace wasm
