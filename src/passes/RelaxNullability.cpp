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
// Relaxes nullability and widens reference supertypes for code size, assuming
// traps never happen.
//
// Non-nullable and overly specific reference types carry a cost in the binary:
// every value that flows into a non-nullable or narrower parameter or local
// needs a ref.as_non_null or ref.cast, locals that differ in nullability or
// subtype cannot share a declaration group or coalesce, and function types or
// covariant bridge wrappers that only differ in parameter subtypes or redundant
// casts cannot be merged. When traps never happen a ref.as_non_null or ref.cast
// can never fail, so it only serves the type system, and we are free to remove
// it wherever the consumer of its value accepts a wider or nullable reference.
//
// This pass therefore
//
//  1. makes every non-nullable reference parameter of every private function
//     type nullable and widens it to the highest declared supertype that
//     satisfies all consumers across functions sharing that signature
//     (results are unchanged, as callers often rely on them);
//  2. makes every reference local nullable and widens it to the highest
//     declared supertype that satisfies all consumers in the function; and
//  3. walks each function and, using the type constraints the parent imposes
//     on each child (ChildTyper, with struct.get/struct.set demanding the
//     highest defining struct ancestor), removes ref.as_non_null and ref.cast
//     where the parent's demanded type is already satisfied and inserts a
//     ref.as_non_null where the parent still demands a non-nullable value.
//
// Public (exported/imported) function types, tag types and types with
// signature subtyping are left alone. The pass requires --traps-never-happen
// and does nothing without it. Follow it with --optimize-instructions and
// --coalesce-locals to fold the copies `tee $a (ref.as_non_null (get $b))`
// that become plain copies, and with --type-merging to merge the function
// types that now differ only in results.
//

#include <unordered_set>

#include "ir/child-typer.h"
#include "ir/effects.h"
#include "ir/find_all.h"
#include "ir/intrinsics.h"
#include "ir/local-structural-dominance.h"
#include "ir/module-utils.h"
#include "ir/properties.h"
#include "ir/subtypes.h"
#include "ir/type-updating.h"
#include "ir/utils.h"
#include "pass.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

Type makeNullable(Type type) {
  if (type.isRef() && type.isNonNullable()) {
    return type.with(Nullable);
  }
  return type;
}

HeapType getRootAncestor(HeapType ht) {
  while (auto super = ht.getDeclaredSuperType()) {
    ht = *super;
  }
  return ht;
}

// Return the highest declared ancestor of `orig` that is a subtype of both
// `curAllowed` (an ancestor of `orig`) and `reqHT`.
HeapType tightenAncestor(HeapType orig, HeapType curAllowed, HeapType reqHT) {
  if (HeapType::isSubType(curAllowed, reqHT)) {
    return curAllowed;
  }
  HeapType best = orig;
  for (HeapType anc = orig;;) {
    auto super = anc.getDeclaredSuperType();
    if (!super || !HeapType::isSubType(*super, curAllowed) ||
        !HeapType::isSubType(*super, reqHT)) {
      break;
    }
    best = *super;
    anc = *super;
  }
  return best;
}

// Return the highest declared struct ancestor of `ht` that defines `fields[index]`
// identically.
HeapType getHighestDefiningAncestor(HeapType ht, Index index) {
  if (!ht.isStruct()) {
    return ht;
  }
  const auto& fields = ht.getStruct().fields;
  if (index >= fields.size()) {
    return ht;
  }
  const auto& field = fields[index];
  while (auto super = ht.getDeclaredSuperType()) {
    if (!super->isStruct()) {
      break;
    }
    const auto& superFields = super->getStruct().fields;
    if (index < superFields.size() && superFields[index] == field) {
      ht = *super;
    } else {
      break;
    }
  }
  return ht;
}

// Return the highest declared struct ancestor of `ht` that has `fields[index]`.
HeapType getHighestFieldAncestor(HeapType ht, Index index) {
  while (auto super = ht.getDeclaredSuperType()) {
    if (super->isStruct() && index < super->getStruct().fields.size()) {
      ht = *super;
    } else {
      break;
    }
  }
  return ht;
}

using SigParamHTMap =
  std::unordered_map<HeapType, std::vector<std::optional<HeapType>>>;
using FuncParamHTMap =
  std::unordered_map<Function*, std::vector<std::optional<HeapType>>>;

struct CallReturnDemand {
  HeapType ht;
  bool acceptsNull = true;
};
using FuncResultDemandMap = std::unordered_map<Function*, CallReturnDemand>;

// Computes for each local in `func` the highest declared supertype of its
// current type that satisfies all consumers in `func`, and optionally records
// return-value demands on direct calls and expression demands on local gets/tees.
struct DemandAnalyzer : public ChildTyper<DemandAnalyzer> {
  const SigParamHTMap* sigParamHTs;
  const FuncParamHTMap* funcParamHTs;
  const FuncResultDemandMap* funcResultDemands;
  const std::unordered_set<HeapType>* candidateSigSet;
  bool useSigParamsForSet;
  Index numParams;
  std::vector<bool> isCandidate;
  std::vector<HeapType> origHT;
  std::vector<HeapType> rootHT;
  std::vector<HeapType> curHT;
  std::vector<HeapType> nextHT;
  std::unordered_map<Name, HeapType> labelTypes;
  std::unordered_map<Name, bool> labelAcceptsNull;
  std::optional<HeapType> bodyReq;
  bool bodyAcceptsNull = true;
  std::unordered_map<Name, CallReturnDemand> callReturnDemands;
  std::unordered_map<HeapType, CallReturnDemand> callRefReturnDemands;
  std::unordered_map<Expression*, std::optional<HeapType>>* exprDemands =
    nullptr;

  DemandAnalyzer(
    Module& wasm,
    Function* func,
    const SigParamHTMap* sigParamHTs,
    bool allowParams,
    bool useSigParamsForSet = false,
    const FuncParamHTMap* funcParamHTs = nullptr,
    const FuncResultDemandMap* funcResultDemands = nullptr,
    const std::unordered_set<HeapType>* candidateSigSet = nullptr,
    std::unordered_map<Expression*, std::optional<HeapType>>* exprDemands =
      nullptr,
    bool allowTopForVars = false)
    : ChildTyper(wasm, func), sigParamHTs(sigParamHTs),
      funcParamHTs(funcParamHTs), funcResultDemands(funcResultDemands),
      candidateSigSet(candidateSigSet), useSigParamsForSet(useSigParamsForSet),
      numParams(func->getNumParams()), exprDemands(exprDemands) {
    Index numLocals = func->getNumLocals();
    isCandidate.assign(numLocals, false);
    origHT.resize(numLocals);
    rootHT.resize(numLocals);
    curHT.resize(numLocals);
    nextHT.resize(numLocals);
    for (Index i = 0; i < numLocals; ++i) {
      Type t = func->getLocalType(i);
      if (t.isRef()) {
        HeapType ht = t.getHeapType();
        origHT[i] = ht;
        if ((allowParams || i >= numParams) && !t.isExact() &&
            (ht.getDeclaredSuperType() ||
             (allowTopForVars && i >= numParams && ht.isStruct()))) {
          isCandidate[i] = true;
          rootHT[i] =
            (allowTopForVars && i >= numParams && !ht.getDeclaredSuperType())
              ? ht.getTop()
              : getRootAncestor(ht);
          curHT[i] = rootHT[i];
          nextHT[i] = rootHT[i];
        } else {
          rootHT[i] = ht;
          curHT[i] = ht;
          nextHT[i] = ht;
        }
      }
    }
  }

  std::vector<HeapType> analyze() {
    bool anyCandidate = false;
    for (bool c : isCandidate) {
      if (c) {
        anyCandidate = true;
        break;
      }
    }
    auto results = func->getResults();
    bodyReq = std::nullopt;
    bodyAcceptsNull = true;
    if (funcResultDemands) {
      auto it = funcResultDemands->find(func);
      if (it != funcResultDemands->end()) {
        bodyReq = it->second.ht;
        bodyAcceptsNull = it->second.acceptsNull;
      }
    }
    if (!bodyReq && results.isRef()) {
      bodyReq = results.getHeapType();
      bodyAcceptsNull = results.isNullable();
    }
    if (!anyCandidate) {
      if (funcResultDemands || exprDemands) {
        callReturnDemands.clear();
        callRefReturnDemands.clear();
        labelTypes.clear();
        labelAcceptsNull.clear();
        walk(func->body, bodyReq, bodyAcceptsNull);
      }
      return curHT;
    }
    for (int iter = 0; iter < 3; ++iter) {
      nextHT = rootHT;
      callReturnDemands.clear();
      callRefReturnDemands.clear();
      labelTypes.clear();
      labelAcceptsNull.clear();
      walk(func->body, bodyReq, bodyAcceptsNull);
      if (nextHT == curHT) {
        break;
      }
      curHT = nextHT;
    }
    return curHT;
  }

  void requireLocal(Index v, HeapType reqHT) {
    if (!isCandidate[v]) {
      return;
    }
    nextHT[v] = tightenAncestor(origHT[v], nextHT[v], reqHT);
  }

  void note(Expression** childp, Constraints constraints) {
    auto* child = *childp;
    if (constraints.size() == 1) {
      if (auto* type = std::get_if<Type>(&constraints[0])) {
        if (type->isRef() && !type->isExact()) {
          walk(child, type->getHeapType(), type->isNullable());
          return;
        } else if (type->isRef() && type->isExact()) {
          walk(child,
               child->type.isRef()
                 ? std::optional<HeapType>(child->type.getHeapType())
                 : std::nullopt,
               type->isNullable());
          return;
        } else if (!type->isRef()) {
          walk(child, std::nullopt, false);
          return;
        }
      } else if (auto* ref = std::get_if<VarRef>(&constraints[0])) {
        bool nullOk = false;
        if (auto* null = std::get_if<Nullability>(&ref->null)) {
          nullOk = (*null == Nullable);
        }
        walk(child,
             child->type.isRef()
               ? std::optional<HeapType>(child->type.getHeapType())
               : std::nullopt,
             nullOk);
        return;
      }
    }
    if (child->type.isRef()) {
      walk(child, child->type.getHeapType(), false);
    } else {
      walk(child, std::nullopt, false);
    }
  }

  Type getLabelType(Name label) { WASM_UNREACHABLE("unexpected branch"); }

  void noteUnknown() {}

  void visitStructGet(StructGet* curr) {
    if (!curr->ref->type.isStruct()) {
      walk(curr->ref, std::nullopt, false);
      return;
    }
    auto anc =
      getHighestDefiningAncestor(curr->ref->type.getHeapType(), curr->index);
    ChildTyper<DemandAnalyzer>::visitStructGet(curr, anc);
  }

  void visitStructSet(StructSet* curr) {
    if (!curr->ref->type.isStruct()) {
      walk(curr->ref, std::nullopt, false);
      walk(curr->value, std::nullopt, false);
      return;
    }
    auto anc =
      getHighestDefiningAncestor(curr->ref->type.getHeapType(), curr->index);
    ChildTyper<DemandAnalyzer>::visitStructSet(curr, anc);
  }

  void visitRefIsNull(RefIsNull* curr) {
    if (curr->value->type.isRef()) {
      walk(curr->value, curr->value->type.getHeapType().getTop(), true);
    } else {
      walk(curr->value, std::nullopt, false);
    }
  }

  void visitRefEq(RefEq* curr) {
    if (curr->left->type.isRef()) {
      walk(curr->left,
           HeapTypes::eq.getBasic(
             curr->left->type.getHeapType().getShared()),
           true);
    } else {
      walk(curr->left, std::nullopt, false);
    }
    if (curr->right->type.isRef()) {
      walk(curr->right,
           HeapTypes::eq.getBasic(
             curr->right->type.getHeapType().getShared()),
           true);
    } else {
      walk(curr->right, std::nullopt, false);
    }
  }

  void visitDrop(Drop* curr) {
    if (curr->value->type.isRef()) {
      walk(curr->value, curr->value->type.getHeapType().getTop(), true);
    } else {
      walk(curr->value, std::nullopt, false);
    }
  }

  void visitRefTest(RefTest* curr) {
    if (curr->castType.isExact() && !wasm.features.hasCustomDescriptors()) {
      if (curr->ref->type.isRef()) {
        walk(curr->ref, curr->ref->type.getHeapType(), true);
      } else {
        walk(curr->ref, std::nullopt, false);
      }
      return;
    }
    ChildTyper<DemandAnalyzer>::visitRefTest(curr);
  }

  void walk(Expression* curr,
            std::optional<HeapType> reqHT,
            bool acceptsNull = true) {
    if (auto* get = curr->dynCast<LocalGet>()) {
      std::optional<HeapType> effReq = reqHT;
      if (!effReq && get->type.isRef()) {
        effReq = get->type.getHeapType();
      }
      if (effReq) {
        requireLocal(get->index, *effReq);
      }
      if (exprDemands) {
        (*exprDemands)[get] = effReq;
      }
      return;
    }
    if (auto* set = curr->dynCast<LocalSet>()) {
      std::optional<HeapType> valReq;
      bool localAcceptsNull = true;
      Type localType = func->getLocalType(set->index);
      if (localType.isRef()) {
        valReq = curHT[set->index];
        if (set->index < numParams) {
          bool paramRelaxed =
            (funcParamHTs && funcParamHTs->count(func)) ||
            (sigParamHTs && sigParamHTs->count(func->type.getHeapType())) ||
            (candidateSigSet &&
             candidateSigSet->count(func->type.getHeapType()));
          if (!paramRelaxed) {
            localAcceptsNull = localType.isNullable();
          }
          if (useSigParamsForSet) {
            bool found = false;
            if (funcParamHTs) {
              auto it = funcParamHTs->find(func);
              if (it != funcParamHTs->end() && set->index < it->second.size() &&
                  it->second[set->index]) {
                valReq = *it->second[set->index];
                found = true;
              }
            }
            if (!found && sigParamHTs) {
              auto it = sigParamHTs->find(func->type.getHeapType());
              if (it != sigParamHTs->end() && set->index < it->second.size() &&
                  it->second[set->index]) {
                valReq = *it->second[set->index];
              }
            }
          }
        }
      }
      bool valAcceptsNull = localAcceptsNull;
      if (set->isTee() && set->type.isRef()) {
        HeapType teeReq = reqHT ? *reqHT : set->type.getHeapType();
        requireLocal(set->index, teeReq);
        if (exprDemands) {
          (*exprDemands)[set] = teeReq;
        }
        if (valReq && !HeapType::isSubType(*valReq, teeReq)) {
          valReq = tightenAncestor(origHT[set->index], *valReq, teeReq);
        }
        valAcceptsNull = localAcceptsNull && acceptsNull;
      }
      walk(set->value, valReq, valAcceptsNull);
      return;
    }
    if (auto* as = curr->dynCast<RefAs>()) {
      if (as->op == RefAsNonNull) {
        walk(as->value,
             reqHT ? reqHT
                   : (as->type.isRef()
                        ? std::optional<HeapType>(as->type.getHeapType())
                        : std::nullopt),
             true);
        return;
      }
    }
    if (auto* cast = curr->dynCast<RefCast>()) {
      if (cast->desc) {
        walk(cast->desc,
             cast->desc->type.isRef()
               ? std::optional<HeapType>(cast->desc->type.getHeapType())
               : std::nullopt,
             cast->desc->type.isNullable());
      }
      if (!cast->ref->type.isRef()) {
        walk(cast->ref, std::nullopt, false);
        return;
      }
      if (cast->type.isExact() && !wasm.features.hasCustomDescriptors()) {
        walk(cast->ref, cast->ref->type.getHeapType(), true);
        return;
      }
      HeapType top = cast->ref->type.getHeapType().getTop();
      if (!cast->desc && reqHT && !cast->ref->type.isNull() &&
          HeapType::isSubType(cast->ref->type.getHeapType(), *reqHT)) {
        walk(cast->ref, *reqHT, acceptsNull);
      } else {
        walk(cast->ref, top, true);
      }
      return;
    }
    if (auto* call = curr->dynCast<Call>()) {
      if (Intrinsics(wasm).isCallWithoutEffects(call)) {
        for (auto* op : call->operands) {
          walk(op,
               op->type.isRef()
                 ? std::optional<HeapType>(op->type.getHeapType())
                 : std::nullopt,
               op->type.isRef() && op->type.isNullable());
        }
        return;
      }
      auto* callee = wasm.getFunction(call->target);
      if (funcResultDemands && call->type.isRef() && !call->isReturn) {
        auto origRes = callee->getResults();
        if (origRes.isRef() && !origRes.isExact() && !origRes.isTuple()) {
          HeapType origResHT = origRes.getHeapType();
          HeapType rReq = reqHT ? *reqHT : origResHT;
          bool rNull = reqHT ? acceptsNull : origRes.isNullable();
          auto [it, inserted] = callReturnDemands.try_emplace(
            call->target,
            CallReturnDemand{getRootAncestor(origResHT), true});
          it->second.ht = tightenAncestor(origResHT, it->second.ht, rReq);
          if (!rNull) {
            it->second.acceptsNull = false;
          }
        }
      }
      auto params = callee->getParams();
      const std::vector<std::optional<HeapType>>* curParams = nullptr;
      bool calleeRelaxed = false;
      if (funcParamHTs) {
        auto it = funcParamHTs->find(callee);
        if (it != funcParamHTs->end()) {
          curParams = &it->second;
          calleeRelaxed = true;
        }
      }
      if (!curParams && sigParamHTs) {
        auto it = sigParamHTs->find(callee->type.getHeapType());
        if (it != sigParamHTs->end()) {
          curParams = &it->second;
          calleeRelaxed = true;
        }
      }
      if (!calleeRelaxed && candidateSigSet &&
          candidateSigSet->count(callee->type.getHeapType())) {
        calleeRelaxed = true;
      }
      for (size_t i = 0; i < call->operands.size(); ++i) {
        std::optional<HeapType> argReq;
        if (curParams && i < curParams->size()) {
          argReq = (*curParams)[i];
        } else if (i < params.size() && params[i].isRef()) {
          argReq = params[i].getHeapType();
        }
        bool argNull =
          calleeRelaxed || (i < params.size() && params[i].isNullable());
        walk(call->operands[i], argReq, argNull);
      }
      return;
    }
    if (auto* callRef = curr->dynCast<CallRef>()) {
      if (!callRef->target->type.isSignature()) {
        for (auto* op : callRef->operands) {
          walk(op,
               op->type.isRef()
                 ? std::optional<HeapType>(op->type.getHeapType())
                 : std::nullopt,
               false);
        }
        walk(callRef->target, std::nullopt, false);
        return;
      }
      HeapType sigHT = callRef->target->type.getHeapType();
      if (funcResultDemands && callRef->type.isRef() && !callRef->isReturn) {
        auto origRes = sigHT.getSignature().results;
        if (origRes.isRef() && !origRes.isExact() && !origRes.isTuple()) {
          HeapType origResHT = origRes.getHeapType();
          HeapType rReq = reqHT ? *reqHT : origResHT;
          bool rNull = reqHT ? acceptsNull : origRes.isNullable();
          auto [it, inserted] = callRefReturnDemands.try_emplace(
            sigHT, CallReturnDemand{getRootAncestor(origResHT), true});
          it->second.ht = tightenAncestor(origResHT, it->second.ht, rReq);
          if (!rNull) {
            it->second.acceptsNull = false;
          }
        }
      }
      auto params = sigHT.getSignature().params;
      const std::vector<std::optional<HeapType>>* curParams = nullptr;
      bool sigRelaxed =
        candidateSigSet && candidateSigSet->count(sigHT) != 0;
      if (sigParamHTs) {
        auto it = sigParamHTs->find(sigHT);
        if (it != sigParamHTs->end()) {
          curParams = &it->second;
          sigRelaxed = true;
        }
      }
      for (size_t i = 0; i < callRef->operands.size(); ++i) {
        std::optional<HeapType> argReq;
        if (curParams && i < curParams->size()) {
          argReq = (*curParams)[i];
        } else if (i < params.size() && params[i].isRef()) {
          argReq = params[i].getHeapType();
        }
        bool argNull =
          sigRelaxed || (i < params.size() && params[i].isNullable());
        walk(callRef->operands[i], argReq, argNull);
      }
      walk(callRef->target, sigHT, true);
      return;
    }
    if (auto* block = curr->dynCast<Block>()) {
      if (block->list.empty()) {
        return;
      }
      std::optional<HeapType> tailReq;
      bool tailNull = false;
      if (block->type.isRef()) {
        if (block->type.isExact()) {
          tailReq = block->type.getHeapType();
          tailNull = false;
        } else {
          tailReq = reqHT ? *reqHT : block->type.getHeapType();
          tailNull = reqHT ? acceptsNull : block->type.isNullable();
        }
      }
      std::optional<HeapType> prevLabelType;
      bool prevLabelNull = false;
      if (block->name.is()) {
        auto it = labelTypes.find(block->name);
        if (it != labelTypes.end()) {
          prevLabelType = it->second;
          prevLabelNull = labelAcceptsNull[block->name];
        }
        if (tailReq) {
          labelTypes[block->name] = *tailReq;
          labelAcceptsNull[block->name] = tailNull;
        } else {
          labelTypes.erase(block->name);
          labelAcceptsNull.erase(block->name);
        }
      }
      for (size_t i = 0; i + 1 < block->list.size(); ++i) {
        walk(block->list[i], std::nullopt, false);
      }
      if (!block->type.isRef()) {
        walk(block->list.back(), std::nullopt, false);
      } else {
        walk(block->list.back(), tailReq, tailNull);
      }
      if (block->name.is()) {
        if (prevLabelType) {
          labelTypes[block->name] = *prevLabelType;
          labelAcceptsNull[block->name] = prevLabelNull;
        } else {
          labelTypes.erase(block->name);
          labelAcceptsNull.erase(block->name);
        }
      }
      return;
    }
    if (auto* iff = curr->dynCast<If>()) {
      walk(iff->condition, std::nullopt, false);
      std::optional<HeapType> armReq;
      bool armNull = false;
      if (iff->type.isRef()) {
        armReq = reqHT ? reqHT
                       : std::optional<HeapType>(iff->type.getHeapType());
        armNull = reqHT ? acceptsNull : iff->type.isNullable();
      }
      walk(iff->ifTrue, armReq, armNull);
      if (iff->ifFalse) {
        walk(iff->ifFalse, armReq, armNull);
      }
      return;
    }
    if (auto* loop = curr->dynCast<Loop>()) {
      std::optional<HeapType> loopReq;
      bool loopNull = false;
      if (loop->type.isRef()) {
        if (loop->type.isExact()) {
          loopReq = loop->type.getHeapType();
          loopNull = false;
        } else {
          loopReq =
            reqHT ? reqHT : std::optional<HeapType>(loop->type.getHeapType());
          loopNull = reqHT ? acceptsNull : loop->type.isNullable();
        }
      }
      walk(loop->body, loopReq, loopNull);
      return;
    }
    if (auto* try_ = curr->dynCast<Try>()) {
      std::optional<HeapType> tryReq;
      bool tryNull = false;
      if (try_->type.isRef()) {
        if (try_->type.isExact()) {
          tryReq = try_->type.getHeapType();
          tryNull = false;
        } else {
          tryReq =
            reqHT ? reqHT : std::optional<HeapType>(try_->type.getHeapType());
          tryNull = reqHT ? acceptsNull : try_->type.isNullable();
        }
      }
      walk(try_->body, tryReq, tryNull);
      for (auto* catchBody : try_->catchBodies) {
        walk(catchBody, tryReq, tryNull);
      }
      return;
    }
    if (auto* tryTable = curr->dynCast<TryTable>()) {
      std::optional<HeapType> tryReq;
      bool tryNull = false;
      if (tryTable->type.isRef()) {
        if (tryTable->type.isExact()) {
          tryReq = tryTable->type.getHeapType();
          tryNull = false;
        } else {
          tryReq = reqHT
                     ? reqHT
                     : std::optional<HeapType>(tryTable->type.getHeapType());
          tryNull = reqHT ? acceptsNull : tryTable->type.isNullable();
        }
      }
      walk(tryTable->body, tryReq, tryNull);
      return;
    }
    if (auto* br = curr->dynCast<Break>()) {
      if (br->condition) {
        walk(br->condition, std::nullopt, false);
      }
      if (br->value) {
        std::optional<HeapType> valReq;
        bool valNull = false;
        if (auto it = labelTypes.find(br->name); it != labelTypes.end()) {
          valReq = it->second;
          valNull = labelAcceptsNull[br->name];
        } else if (br->value->type.isRef()) {
          valReq = br->value->type.getHeapType();
        }
        if (br->condition && br->type.isRef()) {
          HeapType flowReq = reqHT ? *reqHT : br->type.getHeapType();
          bool flowNull = reqHT ? acceptsNull : br->type.isNullable();
          if (!valReq || HeapType::isSubType(flowReq, *valReq)) {
            valReq = flowReq;
          }
          valNull = valNull && flowNull;
        }
        walk(br->value, valReq, valNull);
      }
      return;
    }
    if (auto* sw = curr->dynCast<Switch>()) {
      walk(sw->condition, std::nullopt, false);
      if (sw->value) {
        walk(sw->value,
             sw->value->type.isRef()
               ? std::optional<HeapType>(sw->value->type.getHeapType())
               : std::nullopt,
             false);
      }
      return;
    }
    if (auto* brOn = curr->dynCast<BrOn>()) {
      if (brOn->desc) {
        walk(brOn->desc,
             brOn->desc->type.isRef()
               ? std::optional<HeapType>(brOn->desc->type.getHeapType())
               : std::nullopt,
             brOn->desc->type.isNullable());
      }
      if (!brOn->ref->type.isRef()) {
        walk(brOn->ref, std::nullopt, false);
        return;
      }
      std::optional<HeapType> refReq = brOn->ref->type.getHeapType();
      bool refNull = brOn->ref->type.isNullable();
      if (brOn->op == BrOnNull) {
        if (!brOn->ref->type.isExact() && reqHT) {
          refReq = reqHT;
        }
        refNull = true;
      } else if (brOn->op == BrOnNonNull) {
        if (!brOn->ref->type.isExact()) {
          if (auto it = labelTypes.find(brOn->name); it != labelTypes.end()) {
            refReq = it->second;
          }
        }
        refNull = true;
      } else if (!brOn->desc && !brOn->ref->type.isExact()) {
        if (brOn->op == BrOnCast) {
          refNull = refNull || brOn->castType.isNullable() || acceptsNull;
        } else if (brOn->op == BrOnCastFail) {
          auto it = labelAcceptsNull.find(brOn->name);
          bool targetNull = it != labelAcceptsNull.end() && it->second;
          refNull = refNull || (brOn->castType.isNullable() ? acceptsNull
                                                            : targetNull);
        }
      }
      walk(brOn->ref, refReq, refNull);
      return;
    }
    if (auto* sel = curr->dynCast<Select>()) {
      walk(sel->condition, std::nullopt, false);
      std::optional<HeapType> armReq;
      bool armNull = false;
      if (sel->type.isRef()) {
        armReq = reqHT ? reqHT
                       : std::optional<HeapType>(sel->type.getHeapType());
        armNull = reqHT ? acceptsNull : sel->type.isNullable();
      }
      walk(sel->ifTrue, armReq, armNull);
      walk(sel->ifFalse, armReq, armNull);
      return;
    }
    if (auto* ret = curr->dynCast<Return>()) {
      if (ret->value) {
        walk(ret->value, bodyReq, bodyAcceptsNull);
      }
      return;
    }
    visit(curr);
  }
};

// Per-function fix-up: relax and widen the vars, then remove or add
// ref.as_non_null and strip redundant ref.casts according to what each parent
// demands of its children.
struct FunctionRelaxer
  : public WalkerPass<
      PostWalker<FunctionRelaxer, UnifiedExpressionVisitor<FunctionRelaxer>>> {
  bool widenTypes;
  bool allowTopForVars;

  FunctionRelaxer(bool widenTypes = true, bool allowTopForVars = false)
    : widenTypes(widenTypes), allowTopForVars(allowTopForVars) {}

  bool isFunctionParallel() override { return true; }

  // We only make locals more nullable / wider, which cannot break validation.
  bool requiresNonNullableLocalFixups() override { return false; }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<FunctionRelaxer>(widenTypes, allowTopForVars);
  }

  bool stripTrailingCasts(Expression** exprp, Type demanded, bool stripRefAs) {
    if (!demanded.isRef() || demanded.isExact()) {
      return false;
    }
    if (auto* block = (*exprp)->dynCast<Block>()) {
      if (!block->name.is() && !block->list.empty()) {
        if (stripTrailingCasts(&block->list.back(), demanded, stripRefAs)) {
          block->finalize();
          return true;
        }
        return false;
      }
      if (block->name.is() && !block->list.empty() && block->type.isRef() &&
          !block->type.isExact()) {
        struct BranchFinder : public PostWalker<BranchFinder> {
          Name target;
          bool canModify = true;
          std::vector<Break*> allBreaks;
          std::unordered_set<Break*> droppedBreaks;
          std::vector<Break*> breaks;
          std::vector<BrOn*> brOnNonNulls;
          void visitBreak(Break* curr) {
            if (curr->name == target) {
              allBreaks.push_back(curr);
            }
          }
          void visitDrop(Drop* curr) {
            if (auto* br = curr->value->dynCast<Break>()) {
              if (br->name == target && br->value) {
                droppedBreaks.insert(br);
              }
            }
          }
          void visitSwitch(Switch* curr) {
            if (curr->default_ == target) {
              canModify = false;
              return;
            }
            for (auto t : curr->targets) {
              if (t == target) {
                canModify = false;
                return;
              }
            }
          }
          void visitBrOn(BrOn* curr) {
            if (curr->name == target) {
              if (curr->op == BrOnNonNull) {
                brOnNonNulls.push_back(curr);
              }
              canModify = false;
            }
          }
        } finder;
        finder.target = block->name;
        finder.walk(*exprp);
        for (auto* br : finder.allBreaks) {
          if (!br->value) {
            finder.canModify = false;
          } else if (br->condition && !finder.droppedBreaks.count(br)) {
            if (!br->value->type.isNull()) {
              finder.canModify = false;
            }
          } else {
            finder.breaks.push_back(br);
          }
        }
        Nullability armNull =
          demanded.isNullable() ? Nullable : block->type.getNullability();
        Type armDemanded(demanded.getHeapType(), armNull);
        bool anyChanged = false;
        if (block->list.back()->type.isRef()) {
          anyChanged |= stripTrailingCasts(
            &block->list.back(), armDemanded, stripRefAs && armNull == Nullable);
        }
        for (auto* br : finder.breaks) {
          if (br->value->type.isRef()) {
            if (stripTrailingCasts(
                  &br->value, armDemanded, stripRefAs && armNull == Nullable)) {
              br->finalize();
              anyChanged = true;
            }
          }
        }
        for (auto* brOn : finder.brOnNonNulls) {
          if (brOn->ref->type.isRef()) {
            if (stripTrailingCasts(&brOn->ref,
                                   Type(armDemanded.getHeapType(), Nullable),
                                   true)) {
              brOn->finalize();
              anyChanged = true;
            }
          }
        }
        if (anyChanged) {
          block->finalize();
          if (demanded.isNonNullable() && block->type.isRef() &&
              block->type.isNullable()) {
            *exprp = Builder(*getModule()).makeRefAs(RefAsNonNull, *exprp);
          }
          return true;
        }

        if (finder.canModify) {
          // Also check if >= 2 exits have RefCast to block's HeapType that can
          // be sunk to a single RefCast outside the block.
          HeapType blockHT = block->type.getHeapType();
          std::vector<Expression**> exits;
          if (block->list.back()->type != Type::unreachable) {
            exits.push_back(&block->list.back());
          }
          for (auto* br : finder.breaks) {
            if (br->value->type != Type::unreachable) {
              exits.push_back(&br->value);
            }
          }
          if (exits.size() >= 2) {
            size_t castExits = 0;
            bool allCompatible = true;
            Nullability outerNull =
              demanded.isNonNullable() ? NonNullable : block->type.getNullability();
            for (auto* exitp : exits) {
              Expression** tailp = exitp;
              while (auto* nb = (*tailp)->dynCast<Block>()) {
                if (!nb->name.is() && !nb->list.empty()) {
                  tailp = &nb->list.back();
                } else {
                  break;
                }
              }
              if (auto* cast = (*tailp)->dynCast<RefCast>()) {
                if (!cast->desc && !cast->type.isExact() &&
                    cast->type.getHeapType() == blockHT &&
                    cast->ref->type.isRef() &&
                    cast->ref->type.getHeapType().getTop() ==
                      blockHT.getTop() &&
                    (getPassOptions().trapsNeverHappen ||
                     cast->type.getNullability() == outerNull)) {
                  castExits++;
                  continue;
                }
              }
              if (!Type::isSubType((*tailp)->type, Type(blockHT, Nullable))) {
                allCompatible = false;
                break;
              }
            }
            if (allCompatible && castExits >= 2) {
              for (auto* exitp : exits) {
                std::vector<Block*> unwrapBlocks;
                Expression** tailp = exitp;
                while (auto* nb = (*tailp)->dynCast<Block>()) {
                  if (!nb->name.is() && !nb->list.empty()) {
                    unwrapBlocks.push_back(nb);
                    tailp = &nb->list.back();
                  } else {
                    break;
                  }
                }
                if (auto* cast = (*tailp)->dynCast<RefCast>()) {
                  if (!cast->desc && !cast->type.isExact() &&
                      cast->type.getHeapType() == blockHT &&
                      cast->ref->type.isRef()) {
                    *tailp = cast->ref;
                    for (auto it = unwrapBlocks.rbegin();
                         it != unwrapBlocks.rend();
                         ++it) {
                      (*it)->finalize();
                    }
                  }
                }
              }
              for (auto* br : finder.breaks) {
                br->finalize();
              }
              block->finalize();
              *exprp = Builder(*getModule())
                         .makeRefCast(*exprp, Type(blockHT, outerNull));
              return true;
            }
          }
        }
        return false;
      }
    }
    if (auto* loop = (*exprp)->dynCast<Loop>()) {
      if (loop->type.isRef() && !loop->type.isExact()) {
        Nullability armNull =
          demanded.isNullable() ? Nullable : loop->type.getNullability();
        Type armDemanded(demanded.getHeapType(), armNull);
        if (stripTrailingCasts(
              &loop->body, armDemanded, stripRefAs && armNull == Nullable)) {
          loop->finalize();
          return true;
        }
        return false;
      }
    }
    if (auto* iff = (*exprp)->dynCast<If>()) {
      if (iff->ifFalse && iff->type.isRef()) {
        Nullability armNull =
          demanded.isNullable() ? Nullable : iff->type.getNullability();
        Type armDemanded(demanded.getHeapType(), armNull);
        bool c1 = stripTrailingCasts(
          &iff->ifTrue, armDemanded, stripRefAs && armNull == Nullable);
        bool c2 = stripTrailingCasts(
          &iff->ifFalse, armDemanded, stripRefAs && armNull == Nullable);
        if (c1 || c2) {
          iff->finalize();
          return true;
        }
        return false;
      }
    }
    if (auto* sel = (*exprp)->dynCast<Select>()) {
      if (sel->type.isRef()) {
        Nullability armNull =
          demanded.isNullable() ? Nullable : sel->type.getNullability();
        Type armDemanded(demanded.getHeapType(), armNull);
        bool c1 = stripTrailingCasts(
          &sel->ifTrue, armDemanded, stripRefAs && armNull == Nullable);
        bool c2 = stripTrailingCasts(
          &sel->ifFalse, armDemanded, stripRefAs && armNull == Nullable);
        if (c1 || c2) {
          sel->finalize();
          return true;
        }
        return false;
      }
    }
    if (auto* brOn = (*exprp)->dynCast<BrOn>()) {
      if (brOn->op == BrOnNull && brOn->type.isRef() &&
          !brOn->type.isExact()) {
        if (stripTrailingCasts(
              &brOn->ref, Type(demanded.getHeapType(), Nullable), true)) {
          brOn->finalize();
          return true;
        }
        return false;
      }
    }
    auto demandedHT = demanded.getHeapType();
    bool acceptsNull = demanded.isNullable();
    bool demandsNonNull = demanded.isNonNullable();
    bool stripped = false;
    while (true) {
      if (auto* as = (*exprp)->dynCast<RefAs>()) {
        if (as->op == RefAsNonNull &&
            ((stripRefAs && acceptsNull) ||
             (as->value->is<RefCast>() && !as->value->cast<RefCast>()->desc &&
              as->value->cast<RefCast>()->ref->type.isRef() &&
              !as->value->cast<RefCast>()->ref->type.isNull() &&
              HeapType::isSubType(
                as->value->cast<RefCast>()->ref->type.getHeapType(),
                demandedHT)))) {
          *exprp = as->value;
          stripped = true;
          continue;
        }
      } else if (auto* cast = (*exprp)->dynCast<RefCast>()) {
        if (!cast->desc && cast->ref->type.isRef() &&
            !cast->ref->type.isNull() &&
            HeapType::isSubType(cast->ref->type.getHeapType(), demandedHT)) {
          *exprp = cast->ref;
          stripped = true;
          continue;
        }
      }
      break;
    }
    if (auto* cast = (*exprp)->dynCast<RefCast>()) {
      if (!cast->desc && !cast->type.isExact() && !demandedHT.isBasic() &&
          cast->ref->type.isRef() && !cast->ref->type.isNull() &&
          HeapType::isSubType(cast->type.getHeapType(), demandedHT)) {
        Nullability targetNull =
          (stripRefAs && acceptsNull) ? Nullable : cast->type.getNullability();
        Type oldType = cast->type;
        cast->type = Type(demandedHT, targetNull);
        cast->finalize();
        if (cast->type != oldType) {
          stripped = true;
        }
      }
    }
    if (stripped && demandsNonNull && (*exprp)->type.isRef() &&
        (*exprp)->type.isNullable()) {
      if (auto* cast = (*exprp)->dynCast<RefCast>()) {
        cast->type = cast->type.with(NonNullable);
      } else {
        *exprp = Builder(*getModule()).makeRefAs(RefAsNonNull, *exprp);
      }
    }
    return stripped;
  }

  // Un-alias cast-temporary locals `local.set/tee $a (ref.cast $T (local.get $b))`
  // where `$b` is an unmodified parameter with a wider type than `$a`: redirect
  // supertype-compatible reads of `$a` to `$b`, and if `<= 1` narrow read of `$a`
  // remains, fold the `ref.cast` directly into that read site and eliminate `$a`.
  bool unaliasCastLocals(Function* func, const std::vector<LocalSet*>& sets) {
    Index numParams = func->getNumParams();
    Index numLocals = func->getNumLocals();
    if (numLocals <= numParams) {
      return false;
    }

    std::vector<Index> writeCount(numLocals, 0);
    std::vector<LocalSet*> singleSet(numLocals, nullptr);
    for (auto* set : sets) {
      writeCount[set->index]++;
      singleSet[set->index] = set;
    }

    struct CastAlias {
      Index a;
      Index b;
      LocalSet* set;
      Expression** valPtr;
      RefCast* cast;
    };
    std::vector<CastAlias> candidates;
    for (Index a = numParams; a < numLocals; ++a) {
      if (writeCount[a] != 1) {
        continue;
      }
      Type aType = func->getLocalType(a);
      if (!aType.isRef() || aType.isExact()) {
        continue;
      }
      auto* set = singleSet[a];
      Expression** valPtr = &set->value;
      while (auto* block = (*valPtr)->dynCast<Block>()) {
        if (!block->name.is() && !block->list.empty()) {
          valPtr = &block->list.back();
        } else {
          break;
        }
      }
      Expression* val = *valPtr;
      while (auto* as = val->dynCast<RefAs>()) {
        if (as->op == RefAsNonNull) {
          val = as->value;
        } else {
          break;
        }
      }
      auto* cast = val->dynCast<RefCast>();
      if (!cast || cast->desc || !cast->type.isRef() || cast->type.isExact()) {
        continue;
      }
      Expression* src = cast->ref;
      while (auto* as = src->dynCast<RefAs>()) {
        if (as->op == RefAsNonNull) {
          src = as->value;
        } else {
          break;
        }
      }
      auto* srcGet = src->dynCast<LocalGet>();
      if (!srcGet) {
        continue;
      }
      Index b = srcGet->index;
      if (b == a ||
          (b < numParams ? writeCount[b] != 0 : writeCount[b] != 1)) {
        continue;
      }
      Type bType = func->getLocalType(b);
      if (!bType.isRef() || bType.isExact()) {
        continue;
      }
      HeapType aHT = aType.getHeapType();
      HeapType bHT = bType.getHeapType();
      if (aHT == bHT || !HeapType::isSubType(aHT, bHT)) {
        continue;
      }
      candidates.push_back({a, b, set, valPtr, cast});
    }

    if (candidates.empty()) {
      return false;
    }

    LocalStructuralDominance dom(
      func, *getModule(), LocalStructuralDominance::All);
    std::vector<CastAlias> domCandidates;
    for (auto& cand : candidates) {
      if (!dom.nonDominatingIndices.count(cand.a) &&
          (cand.b < numParams || !dom.nonDominatingIndices.count(cand.b))) {
        domCandidates.push_back(cand);
      }
    }
    if (domCandidates.empty()) {
      return false;
    }

    std::unordered_map<Expression*, std::optional<HeapType>> exprDemands;
    DemandAnalyzer analyzer(*getModule(),
                            func,
                            nullptr,
                            false,
                            false,
                            nullptr,
                            nullptr,
                            nullptr,
                            &exprDemands);
    analyzer.analyze();

    struct PtrCollector : public PostWalker<PtrCollector> {
      std::unordered_map<Index, std::vector<Expression**>> getPtrs;
      std::unordered_map<LocalSet*, Expression**> setPtrs;
      void visitLocalGet(LocalGet* curr) {
        getPtrs[curr->index].push_back(getCurrentPointer());
      }
      void visitLocalSet(LocalSet* curr) {
        setPtrs[curr] = getCurrentPointer();
      }
    } collector;
    collector.walk(func->body);

    Builder builder(*getModule());
    bool anyModified = false;
    std::unordered_set<Index> modifiedAs;
    std::unordered_set<Index> usedBs;
    for (auto& cand : domCandidates) {
      if (modifiedAs.count(cand.a) || modifiedAs.count(cand.b) ||
          usedBs.count(cand.a)) {
        continue;
      }
      auto setIt = collector.setPtrs.find(cand.set);
      if (setIt == collector.setPtrs.end()) {
        continue;
      }
      Expression** setPtr = setIt->second;
      Type bType = func->getLocalType(cand.b);
      HeapType bHT = bType.getHeapType();

      std::vector<Expression**> wideGets;
      std::vector<Expression**> narrowGets;
      for (auto* getPtr : collector.getPtrs[cand.a]) {
        auto* get = (*getPtr)->cast<LocalGet>();
        auto demIt = exprDemands.find(get);
        if (demIt != exprDemands.end() && demIt->second &&
            HeapType::isSubType(bHT, *demIt->second)) {
          wideGets.push_back(getPtr);
        } else {
          narrowGets.push_back(getPtr);
        }
      }

      bool teeOk = false;
      if (cand.set->isTee()) {
        auto demIt = exprDemands.find(cand.set);
        if (demIt != exprDemands.end() && demIt->second &&
            HeapType::isSubType(bHT, *demIt->second)) {
          teeOk = true;
        }
      }

      size_t remTee = (cand.set->isTee() && !teeOk) ? 1 : 0;
      size_t totalRem = remTee + narrowGets.size();
      bool canFoldRem =
        totalRem <= 1 &&
        (cand.valPtr == &cand.set->value ||
         getPassOptions().trapsNeverHappen || narrowGets.empty());

      if (!teeOk && wideGets.empty() && !canFoldRem) {
        continue;
      }

      // Redirect all supertype-compatible gets of $a directly to $b.
      for (auto* getPtr : wideGets) {
        *getPtr = builder.makeLocalGet(cand.b, bType);
      }
      anyModified = true;
      modifiedAs.insert(cand.a);
      usedBs.insert(cand.b);

      if (canFoldRem) {
        cand.cast->ref = builder.makeLocalGet(cand.b, bType);
        cand.cast->type =
          cand.cast->type.with(bType.isNonNullable() ? NonNullable : Nullable);
        if (remTee == 1) {
          *cand.valPtr = cand.cast;
          *setPtr = cand.set->value;
        } else if (narrowGets.size() == 1) {
          *narrowGets[0] = cand.cast;
          if (cand.set->isTee()) {
            *cand.valPtr = builder.makeLocalGet(cand.b, bType);
            *setPtr = cand.set->value;
          } else {
            *cand.valPtr = builder.makeNop();
            *setPtr = cand.set->value;
          }
        } else {
          if (cand.set->isTee()) {
            *cand.valPtr = builder.makeLocalGet(cand.b, bType);
            *setPtr = cand.set->value;
          } else {
            *cand.valPtr = builder.makeNop();
            *setPtr = cand.set->value;
          }
        }
      }
    }
    return anyModified;
  }

  // Sink RefCasts across multiple LocalSet/LocalTee assignments to the same
  // local `$a` when widening `$a` to the LUB of the uncast operands requires
  // strictly fewer casts on narrow reads than the casts removed on writes.
  bool sinkMultiSetCastLocals(Function* func,
                              const std::vector<LocalSet*>& sets) {
    Index numParams = func->getNumParams();
    Index numLocals = func->getNumLocals();
    if (numLocals <= numParams) {
      return false;
    }

    std::vector<std::vector<LocalSet*>> localSets(numLocals);
    for (auto* set : sets) {
      if (set->index >= numParams) {
        localSets[set->index].push_back(set);
      }
    }

    struct CastWrite {
      LocalSet* set;
      Expression** valPtr;
      RefCast* cast;
    };
    struct MultiCastCandidate {
      Index a;
      HeapType aHT;
      HeapType lubHT;
      std::vector<CastWrite> castWrites;
    };
    std::vector<MultiCastCandidate> candidates;

    for (Index a = numParams; a < numLocals; ++a) {
      if (localSets[a].size() < 2) {
        continue;
      }
      Type aType = func->getLocalType(a);
      if (!aType.isRef() || aType.isExact()) {
        continue;
      }
      HeapType aHT = aType.getHeapType();
      std::vector<CastWrite> castWrites;
      bool allValid = true;
      for (auto* set : localSets[a]) {
        Expression** valPtr = &set->value;
        while (auto* block = (*valPtr)->dynCast<Block>()) {
          if (!block->name.is() && !block->list.empty()) {
            valPtr = &block->list.back();
          } else {
            break;
          }
        }
        Expression* val = *valPtr;
        if (val->type.isNull()) {
          continue;
        }
        auto* cast = val->dynCast<RefCast>();
        if (cast && !cast->desc && !cast->type.isExact() &&
            cast->type.getHeapType() == aHT && cast->ref->type.isRef() &&
            !cast->ref->type.isNull() &&
            cast->ref->type.getHeapType().getTop() == aHT.getTop()) {
          castWrites.push_back({set, valPtr, cast});
        } else {
          allValid = false;
          break;
        }
      }
      if (!allValid || castWrites.size() < 2) {
        continue;
      }
      HeapType lubHT = castWrites[0].cast->ref->type.getHeapType();
      bool validLub = true;
      for (size_t i = 1; i < castWrites.size(); ++i) {
        auto nextLub = HeapType::getLeastUpperBound(
          lubHT, castWrites[i].cast->ref->type.getHeapType());
        if (!nextLub) {
          validLub = false;
          break;
        }
        lubHT = *nextLub;
      }
      if (!validLub || lubHT == aHT || !HeapType::isSubType(aHT, lubHT)) {
        continue;
      }
      candidates.push_back({a, aHT, lubHT, std::move(castWrites)});
    }

    if (candidates.empty()) {
      return false;
    }

    std::unordered_map<Expression*, std::optional<HeapType>> exprDemands;
    DemandAnalyzer analyzer(*getModule(),
                            func,
                            nullptr,
                            false,
                            false,
                            nullptr,
                            nullptr,
                            nullptr,
                            &exprDemands);
    analyzer.analyze();

    struct PtrCollector : public PostWalker<PtrCollector> {
      std::unordered_map<Index, std::vector<Expression**>> getPtrs;
      std::unordered_map<LocalSet*, Expression**> setPtrs;
      void visitLocalGet(LocalGet* curr) {
        getPtrs[curr->index].push_back(getCurrentPointer());
      }
      void visitLocalSet(LocalSet* curr) {
        setPtrs[curr] = getCurrentPointer();
      }
    } collector;
    collector.walk(func->body);

    Builder builder(*getModule());
    bool anyModified = false;
    for (auto& cand : candidates) {
      std::vector<Expression**> narrowGets;
      for (auto* getPtr : collector.getPtrs[cand.a]) {
        auto* get = (*getPtr)->cast<LocalGet>();
        auto demIt = exprDemands.find(get);
        if (demIt == exprDemands.end() || !demIt->second ||
            !HeapType::isSubType(cand.lubHT, *demIt->second)) {
          narrowGets.push_back(getPtr);
        }
      }
      std::vector<CastWrite> narrowTees;
      bool missingSetPtr = false;
      for (auto& cw : cand.castWrites) {
        if (!collector.setPtrs.count(cw.set)) {
          missingSetPtr = true;
          break;
        }
        if (cw.set->isTee()) {
          auto demIt = exprDemands.find(cw.set);
          if (demIt == exprDemands.end() || !demIt->second ||
              !HeapType::isSubType(cand.lubHT, *demIt->second)) {
            narrowTees.push_back(cw);
          }
        }
      }
      if (missingSetPtr) {
        continue;
      }
      if (narrowGets.size() + narrowTees.size() >= cand.castWrites.size()) {
        continue;
      }

      Type wideType(cand.lubHT, Nullable);
      func->vars[cand.a - numParams] = wideType;
      for (auto& cw : cand.castWrites) {
        *cw.valPtr = cw.cast->ref;
      }
      for (auto& cw : narrowTees) {
        cw.set->type = wideType;
        *collector.setPtrs[cw.set] = builder.makeRefCast(
          cw.set, Type(cand.aHT, cw.cast->type.getNullability()));
      }
      for (auto* getPtr : narrowGets) {
        (*getPtr)->type = wideType;
        *getPtr = builder.makeRefCast(*getPtr, Type(cand.aHT, Nullable));
      }
      anyModified = true;
    }
    return anyModified;
  }

  void doWalkFunction(Function* func) {
    auto gets = FindAll<LocalGet>(func->body).list;
    auto sets = FindAll<LocalSet>(func->body).list;
    auto refreshLocals = [&]() {
      for (auto* get : gets) {
        get->type = func->getLocalType(get->index);
      }
      for (auto* set : sets) {
        if (set->isTee() && set->type != Type::unreachable) {
          set->type = func->getLocalType(set->index);
        }
      }
      ReFinalize().walkFunctionInModule(func, getModule());
    };

    // Update parameter local.get/local.tee types from signature relaxation
    // before analyzing body local demands.
    refreshLocals();

    if (widenTypes) {
      bool hasRefVars = false;
      for (auto& var : func->vars) {
        if (var.isRef()) {
          hasRefVars = true;
          break;
        }
      }
      if (hasRefVars) {
        DemandAnalyzer analyzer(*getModule(),
                                func,
                                nullptr,
                                false,
                                false,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                allowTopForVars);
        auto allowedHTs = analyzer.analyze();
        Index numParams = func->getNumParams();
        for (Index i = 0; i < func->vars.size(); ++i) {
          auto& var = func->vars[i];
          var = makeNullable(var);
          if (var.isRef() && !var.isExact() &&
              allowedHTs[numParams + i] != var.getHeapType()) {
            var = Type(allowedHTs[numParams + i], Nullable);
          }
        }
        refreshLocals();
        for (int iter = 0; iter < 4 && unaliasCastLocals(func, sets); ++iter) {
          gets = FindAll<LocalGet>(func->body).list;
          sets = FindAll<LocalSet>(func->body).list;
          refreshLocals();
        }
        if (sinkMultiSetCastLocals(func, sets)) {
          gets = FindAll<LocalGet>(func->body).list;
          sets = FindAll<LocalSet>(func->body).list;
          refreshLocals();
        }
      }
    } else {
      for (auto& var : func->vars) {
        var = makeNullable(var);
      }
      refreshLocals();
    }

    Super::doWalkFunction(func);

    // The implicit return.
    auto results = func->getResults();
    if (results.isRef()) {
      if (widenTypes) {
        stripTrailingCasts(&func->body, results, results.isNullable());
      } else if (results.isNullable()) {
        while (auto* as = func->body->dynCast<RefAs>()) {
          if (as->op != RefAsNonNull) {
            break;
          }
          func->body = as->value;
        }
      }
      if (results.isNonNullable() && func->body->type.isRef() &&
          func->body->type.isNullable()) {
        if (auto* cast = func->body->dynCast<RefCast>()) {
          cast->type = cast->type.with(NonNullable);
        } else {
          func->body = Builder(*getModule()).makeRefAs(RefAsNonNull, func->body);
        }
      }
    }
    ReFinalize().walkFunctionInModule(func, getModule());
  }

  struct Typer : public ChildTyper<Typer> {
    FunctionRelaxer& parent;
    bool changed = false;

    Typer(FunctionRelaxer& parent)
      : ChildTyper(*parent.getModule(), parent.getFunction()), parent(parent) {}

    using ChildTyper<Typer>::note;

    void visitCall(Call* curr) {
      if (Intrinsics(wasm).isCallWithoutEffects(curr)) {
        return;
      }
      ChildTyper<Typer>::visitCall(curr);
    }

    void visitStructGet(StructGet* curr) {
      if (!curr->ref->type.isStruct()) {
        noteUnknown();
        return;
      }
      if (!parent.widenTypes) {
        ChildTyper<Typer>::visitStructGet(curr);
        return;
      }
      auto anc =
        getHighestDefiningAncestor(curr->ref->type.getHeapType(), curr->index);
      ChildTyper<Typer>::visitStructGet(curr, anc);
    }

    void visitStructSet(StructSet* curr) {
      if (!curr->ref->type.isStruct()) {
        noteUnknown();
        return;
      }
      if (!parent.widenTypes) {
        ChildTyper<Typer>::visitStructSet(curr);
        return;
      }
      auto anc =
        getHighestDefiningAncestor(curr->ref->type.getHeapType(), curr->index);
      ChildTyper<Typer>::visitStructSet(curr, anc);
    }

    void visitRefIsNull(RefIsNull* curr) {
      if (parent.widenTypes && curr->value->type.isRef()) {
        note(&curr->value,
             Type(curr->value->type.getHeapType().getTop(), Nullable));
      }
    }

    void visitRefEq(RefEq* curr) {
      if (!parent.widenTypes) {
        return;
      }
      if (curr->left->type.isRef()) {
        note(&curr->left,
             Type(HeapTypes::eq.getBasic(
                    curr->left->type.getHeapType().getShared()),
                  Nullable));
      }
      if (curr->right->type.isRef()) {
        note(&curr->right,
             Type(HeapTypes::eq.getBasic(
                    curr->right->type.getHeapType().getShared()),
                  Nullable));
      }
    }

    void visitDrop(Drop* curr) {
      if (parent.widenTypes && curr->value->type.isRef()) {
        note(&curr->value,
             Type(curr->value->type.getHeapType().getTop(), Nullable));
      }
    }

    void visitRefTest(RefTest* curr) {
      if (curr->castType.isExact() && !wasm.features.hasCustomDescriptors()) {
        if (curr->ref->type.isRef()) {
          note(&curr->ref,
               Type(curr->ref->type.getHeapType(),
                    Nullable,
                    curr->ref->type.getExactness()));
        }
        return;
      }
      ChildTyper<Typer>::visitRefTest(curr);
    }

    void visitRefCast(RefCast* curr) {
      if (curr->type.isExact() && !wasm.features.hasCustomDescriptors()) {
        if (curr->ref->type.isRef()) {
          note(&curr->ref,
               Type(curr->ref->type.getHeapType(),
                    Nullable,
                    curr->ref->type.getExactness()));
        }
        if (curr->desc) {
          auto desc = curr->type.getHeapType().getDescriptorType();
          assert(desc);
          note(&curr->desc, Type(*desc, Nullable, curr->type.getExactness()));
        }
        return;
      }
      ChildTyper<Typer>::visitRefCast(curr);
    }

    void note(Expression** childp, Constraints constraints) {
      if (constraints.size() != 1) {
        // TODO: tuples
        return;
      }
      if (!(*childp)->type.isRef()) {
        return;
      }
      bool demandsNonNull = false;
      bool acceptsNull = false;
      std::optional<HeapType> demandedHT;
      auto& constraint = constraints[0];
      if (auto* type = std::get_if<Type>(&constraint)) {
        if (type->isRef()) {
          demandsNonNull = type->isNonNullable();
          acceptsNull = type->isNullable();
          if (parent.widenTypes && !type->isExact()) {
            demandedHT = type->getHeapType();
          }
        }
      } else if (auto* ref = std::get_if<VarRef>(&constraint)) {
        if (auto* null = std::get_if<Nullability>(&ref->null)) {
          demandsNonNull = *null == NonNullable;
          acceptsNull = *null == Nullable;
        }
      }
      if (!demandsNonNull && !acceptsNull) {
        return;
      }
      if (demandedHT) {
        if (parent.stripTrailingCasts(
              childp,
              Type(*demandedHT, demandsNonNull ? NonNullable : Nullable),
              acceptsNull)) {
          changed = true;
        }
      } else if (acceptsNull) {
        while (auto* as = (*childp)->dynCast<RefAs>()) {
          if (as->op != RefAsNonNull) {
            break;
          }
          *childp = as->value;
          changed = true;
        }
      }
      if (demandsNonNull && (*childp)->type.isRef() &&
          (*childp)->type.isNullable()) {
        if (auto* cast = (*childp)->dynCast<RefCast>()) {
          cast->type = cast->type.with(NonNullable);
        } else {
          *childp = Builder(*parent.getModule()).makeRefAs(RefAsNonNull, *childp);
        }
        changed = true;
      }
    }

    Type getLabelType(Name label) { WASM_UNREACHABLE("unexpected branch"); }

    void noteUnknown() {}
  };

  void visitExpression(Expression* curr) {
    // Control flow structures and branches just forward values; their types
    // follow from their children and are handled by refinalization.
    if (Properties::isControlFlowStructure(curr) ||
        Properties::isBranch(curr)) {
      return;
    }
    Typer typer(*this);
    typer.visit(curr);
    if (typer.changed) {
      ReFinalizeNode().visit(curr);
    }
  }
};

struct ConsumerSite {
  Expression** loc = nullptr;
  LocalGet* get = nullptr;
  bool isWrappedInNonNull = false;
  bool isFixedConsumer = false;
  bool isSelfReturn = false;
  HeapType acceptedHT;
  Nullability acceptedNull = NonNullable;
  Index visitOrder = 0;
  Index domEndOrder = 0;
};

struct CallConsumerSite {
  Function* caller = nullptr;
  Expression** loc = nullptr;
  Call* call = nullptr;
  bool isWrappedInNonNull = false;
  bool isFixedConsumer = false;
  HeapType acceptedHT;
  Nullability acceptedNull = NonNullable;
};

struct FuncConsumerSummary {
  std::vector<Index> localWriteCounts;
  std::vector<std::vector<ConsumerSite>> paramSites;
  std::vector<CallConsumerSite> callSites;
  std::vector<std::pair<Function*, Call*>> allCalls;
  std::vector<Expression**> returnValueLocs;
};

void scanFunctionConsumers(Module& wasm,
                           Function* func,
                           FuncConsumerSummary& out) {
  Index numParams = func->getNumParams();
  Index numLocals = func->getNumLocals();
  out.localWriteCounts.assign(numLocals, 0);
  out.paramSites.assign(numParams, {});

  auto recordChild = [&](Expression** childp,
                         bool isFixed,
                         bool isSelfRet,
                         HeapType accHT,
                         Nullability accNull) {
    Expression* curr = *childp;
    bool wrappedNonNull = false;
    if (auto* as = curr->dynCast<RefAs>()) {
      if (as->op == RefAsNonNull) {
        wrappedNonNull = true;
        curr = as->value;
        accNull = Nullable;
      }
    }
    if (auto* get = curr->dynCast<LocalGet>()) {
      if (get->index < numParams && get->type.isRef()) {
        ConsumerSite site;
        site.loc = childp;
        site.get = get;
        site.isWrappedInNonNull = wrappedNonNull;
        site.isFixedConsumer = isFixed;
        site.isSelfReturn = isSelfRet;
        site.acceptedHT =
          isFixed ? accHT : get->type.getHeapType();
        site.acceptedNull =
          isFixed ? accNull : get->type.getNullability();
        out.paramSites[get->index].push_back(site);
      }
    } else if (auto* call = curr->dynCast<Call>()) {
      if (call->type.isRef() && !call->isReturn) {
        CallConsumerSite csite;
        csite.caller = func;
        csite.loc = childp;
        csite.call = call;
        csite.isWrappedInNonNull = wrappedNonNull;
        csite.isFixedConsumer = isFixed;
        csite.acceptedHT =
          isFixed ? accHT : call->type.getHeapType();
        csite.acceptedNull =
          isFixed ? accNull : call->type.getNullability();
        out.callSites.push_back(csite);
      }
    }
  };

  struct Scanner : public PostWalker<Scanner, UnifiedExpressionVisitor<Scanner>> {
    Module& wasm;
    Function* func;
    FuncConsumerSummary& out;
    decltype(recordChild)& record;
    std::unordered_set<Expression*> handled;
    std::vector<Block*> blockStack;
    std::unordered_map<Block*, Type> blockDemands;

    Scanner(Module& wasm,
            Function* func,
            FuncConsumerSummary& out,
            decltype(recordChild)& record)
      : wasm(wasm), func(func), out(out), record(record) {}

    void propagateBlockDemand(Expression* e, Type t) {
      while (auto* b = e->dynCast<Block>()) {
        if (!b->type.isRef() || b->type.isExact()) {
          break;
        }
        if (b->name.is()) {
          blockDemands[b] = t;
        }
        if (b->list.empty()) {
          break;
        }
        e = b->list.back();
      }
    }

    static void doEnterBlock(Scanner* self, Expression** currp) {
      self->blockStack.push_back((*currp)->cast<Block>());
    }

    static void scan(Scanner* self, Expression** currp) {
      if (auto* set = (*currp)->dynCast<LocalSet>()) {
        if (!set->isTee()) {
          Type lt = self->func->getLocalType(set->index);
          if (lt.isRef() && !lt.isExact()) {
            self->propagateBlockDemand(set->value, lt);
          }
        }
      }
      PostWalker<Scanner, UnifiedExpressionVisitor<Scanner>>::scan(self, currp);
      if (auto* block = (*currp)->dynCast<Block>()) {
        if (block->name.is()) {
          self->pushTask(Scanner::doEnterBlock, currp);
        }
      }
    }

    Type getTargetBlockType(Name name) const {
      for (auto it = blockStack.rbegin(); it != blockStack.rend(); ++it) {
        if ((*it)->name == name) {
          auto dIt = blockDemands.find(*it);
          return dIt != blockDemands.end() ? dIt->second : (*it)->type;
        }
      }
      return Type::none;
    }

    void markAndRecord(Expression** childp,
                       bool isFixed,
                       bool isSelfRet,
                       HeapType accHT,
                       Nullability accNull) {
      while (auto* block = (*childp)->dynCast<Block>()) {
        if (!block->name.is() && !block->list.empty()) {
          childp = &block->list.back();
        } else {
          break;
        }
      }
      Expression* curr = *childp;
      if (handled.count(curr)) {
        return;
      }
      if (auto* brOn = curr->dynCast<BrOn>()) {
        if (brOn->op == BrOnNull && brOn->ref->type.isRef() &&
            !brOn->ref->type.isExact()) {
          handled.insert(brOn);
          markAndRecord(&brOn->ref, isFixed, false, accHT, Nullable);
          return;
        }
      }
      if (auto* iff = curr->dynCast<If>()) {
        if (iff->ifFalse && iff->type.isRef()) {
          handled.insert(iff);
          Nullability armNull =
            accNull == Nullable ? Nullable : iff->type.getNullability();
          if (iff->ifTrue->type.isRef()) {
            markAndRecord(&iff->ifTrue, isFixed, isSelfRet, accHT, armNull);
          }
          if (iff->ifFalse->type.isRef()) {
            markAndRecord(&iff->ifFalse, isFixed, isSelfRet, accHT, armNull);
          }
          return;
        }
      }
      if (auto* sel = curr->dynCast<Select>()) {
        if (sel->type.isRef()) {
          handled.insert(sel);
          Nullability armNull =
            accNull == Nullable ? Nullable : sel->type.getNullability();
          if (sel->ifTrue->type.isRef()) {
            markAndRecord(&sel->ifTrue, isFixed, isSelfRet, accHT, armNull);
          }
          if (sel->ifFalse->type.isRef()) {
            markAndRecord(&sel->ifFalse, isFixed, isSelfRet, accHT, armNull);
          }
          return;
        }
      }
      if (isSelfRet &&
          std::find(out.returnValueLocs.begin(),
                    out.returnValueLocs.end(),
                    childp) == out.returnValueLocs.end()) {
        out.returnValueLocs.push_back(childp);
      }
      handled.insert(curr);
      if (auto* as = curr->dynCast<RefAs>()) {
        if (as->op == RefAsNonNull) {
          handled.insert(as->value);
        }
      }
      record(childp, isFixed, isSelfRet, accHT, accNull);
    }

    void visitExpression(Expression* curr) {
      if (auto* set = curr->dynCast<LocalSet>()) {
        out.localWriteCounts[set->index]++;
        if (!set->isTee()) {
          Type lt = func->getLocalType(set->index);
          if (lt.isRef() && !lt.isExact()) {
            markAndRecord(
              &set->value, true, false, lt.getHeapType(), lt.getNullability());
          }
        }
      } else if (auto* drop = curr->dynCast<Drop>()) {
        if (auto* br = drop->value->dynCast<Break>()) {
          if (br->condition && br->value && br->value->type.isRef()) {
            if (Type bt = getTargetBlockType(br->name);
                bt.isRef() && !bt.isExact()) {
              markAndRecord(
                &br->value, true, false, bt.getHeapType(), bt.getNullability());
            }
          }
        } else if (drop->value->type.isRef()) {
          markAndRecord(&drop->value,
                        true,
                        false,
                        drop->value->type.getHeapType().getTop(),
                        Nullable);
        }
      } else if (auto* block = curr->dynCast<Block>()) {
        if (block->name.is()) {
          blockStack.pop_back();
        }
        if (!block->list.empty() && block->type.isRef() &&
            !block->type.isExact() && block->list.back()->type.isRef()) {
          auto dIt = blockDemands.find(block);
          Type bt = dIt != blockDemands.end() ? dIt->second : block->type;
          markAndRecord(&block->list.back(),
                        true,
                        false,
                        bt.getHeapType(),
                        bt.getNullability());
        }
      } else if (auto* br = curr->dynCast<Break>()) {
        if (!br->condition && br->value && br->value->type.isRef()) {
          if (Type bt = getTargetBlockType(br->name);
              bt.isRef() && !bt.isExact()) {
            markAndRecord(
              &br->value, true, false, bt.getHeapType(), bt.getNullability());
          }
        }
      } else if (auto* brOn = curr->dynCast<BrOn>()) {
        if (brOn->op == BrOnNonNull && brOn->ref->type.isRef() &&
            !brOn->ref->type.isExact()) {
          if (Type bt = getTargetBlockType(brOn->name);
              bt.isRef() && !bt.isExact()) {
            markAndRecord(
              &brOn->ref, true, false, bt.getHeapType(), Nullable);
          }
        }
      } else if (auto* alen = curr->dynCast<ArrayLen>()) {
        if (alen->ref->type.isRef()) {
          markAndRecord(&alen->ref,
                        true,
                        false,
                        HeapTypes::array.getBasic(
                          alen->ref->type.getHeapType().getShared()),
                        Nullable);
        }
      } else if (auto* aget = curr->dynCast<ArrayGet>()) {
        if (aget->ref->type.isArray()) {
          markAndRecord(
            &aget->ref, true, false, aget->ref->type.getHeapType(), Nullable);
        }
      } else if (auto* aset = curr->dynCast<ArraySet>()) {
        if (aset->ref->type.isArray()) {
          HeapType arrHT = aset->ref->type.getHeapType();
          markAndRecord(&aset->ref, true, false, arrHT, Nullable);
          Type et = arrHT.getArray().element.type;
          if (et.isRef() && !et.isExact()) {
            markAndRecord(
              &aset->value, true, false, et.getHeapType(), et.getNullability());
          }
        }
      } else if (auto* anew = curr->dynCast<ArrayNew>()) {
        if (!anew->isWithDefault() && anew->type.isArray()) {
          Type et = anew->type.getHeapType().getArray().element.type;
          if (et.isRef() && !et.isExact()) {
            markAndRecord(
              &anew->init, true, false, et.getHeapType(), et.getNullability());
          }
        }
      } else if (auto* afixed = curr->dynCast<ArrayNewFixed>()) {
        if (afixed->type.isArray()) {
          Type et = afixed->type.getHeapType().getArray().element.type;
          if (et.isRef() && !et.isExact()) {
            for (auto& val : afixed->values) {
              markAndRecord(
                &val, true, false, et.getHeapType(), et.getNullability());
            }
          }
        }
      } else if (auto* isNull = curr->dynCast<RefIsNull>()) {
        if (isNull->value->type.isRef()) {
          markAndRecord(&isNull->value,
                        true,
                        false,
                        isNull->value->type.getHeapType().getTop(),
                        Nullable);
        }
      } else if (auto* eq = curr->dynCast<RefEq>()) {
        if (eq->left->type.isRef()) {
          markAndRecord(&eq->left,
                        true,
                        false,
                        HeapTypes::eq.getBasic(
                          eq->left->type.getHeapType().getShared()),
                        Nullable);
        }
        if (eq->right->type.isRef()) {
          markAndRecord(&eq->right,
                        true,
                        false,
                        HeapTypes::eq.getBasic(
                          eq->right->type.getHeapType().getShared()),
                        Nullable);
        }
      } else if (auto* cast = curr->dynCast<RefCast>()) {
        if (!cast->desc && cast->ref->type.isRef() &&
            (!cast->type.isExact() || wasm.features.hasCustomDescriptors())) {
          markAndRecord(&cast->ref,
                        true,
                        false,
                        cast->ref->type.getHeapType().getTop(),
                        Nullable);
        }
      } else if (auto* test = curr->dynCast<RefTest>()) {
        if (test->ref->type.isRef() &&
            (!test->castType.isExact() ||
             wasm.features.hasCustomDescriptors())) {
          markAndRecord(&test->ref,
                        true,
                        false,
                        test->ref->type.getHeapType().getTop(),
                        Nullable);
        }
      } else if (auto* sget = curr->dynCast<StructGet>()) {
        if (sget->ref->type.isStruct()) {
          auto anc = getHighestDefiningAncestor(sget->ref->type.getHeapType(),
                                                sget->index);
          markAndRecord(&sget->ref, true, false, anc, Nullable);
        }
      } else if (auto* sset = curr->dynCast<StructSet>()) {
        if (sset->ref->type.isStruct()) {
          auto anc = getHighestDefiningAncestor(sset->ref->type.getHeapType(),
                                                sset->index);
          markAndRecord(&sset->ref, true, false, anc, Nullable);
          const auto& fields =
            sset->ref->type.getHeapType().getStruct().fields;
          if (sset->index < fields.size()) {
            Type ft = fields[sset->index].type;
            if (ft.isRef() && !ft.isExact()) {
              markAndRecord(&sset->value,
                            true,
                            false,
                            ft.getHeapType(),
                            ft.getNullability());
            }
          }
        }
      } else if (auto* snew = curr->dynCast<StructNew>()) {
        if (!snew->isWithDefault() && snew->type.isStruct()) {
          const auto& fields = snew->type.getHeapType().getStruct().fields;
          for (size_t i = 0;
               i < snew->operands.size() && i < fields.size();
               ++i) {
            Type ft = fields[i].type;
            if (ft.isRef() && !ft.isExact()) {
              markAndRecord(&snew->operands[i],
                            true,
                            false,
                            ft.getHeapType(),
                            ft.getNullability());
            }
          }
        }
      } else if (auto* gset = curr->dynCast<GlobalSet>()) {
        Type gt = wasm.getGlobal(gset->name)->type;
        if (gt.isRef() && !gt.isExact()) {
          markAndRecord(
            &gset->value, true, false, gt.getHeapType(), gt.getNullability());
        }
      } else if (auto* call = curr->dynCast<Call>()) {
        out.allCalls.push_back({func, call});
        auto params = wasm.getFunction(call->target)->getParams();
        for (size_t i = 0; i < call->operands.size() && i < params.size();
             ++i) {
          Type pt = params[i];
          if (pt.isRef() && !pt.isExact()) {
            markAndRecord(&call->operands[i],
                          true,
                          false,
                          pt.getHeapType(),
                          pt.getNullability());
          }
        }
      } else if (auto* callRef = curr->dynCast<CallRef>()) {
        if (callRef->target->type.isSignature()) {
          auto params =
            callRef->target->type.getHeapType().getSignature().params;
          for (size_t i = 0;
               i < callRef->operands.size() && i < params.size();
               ++i) {
            Type pt = params[i];
            if (pt.isRef() && !pt.isExact()) {
              markAndRecord(&callRef->operands[i],
                            true,
                            false,
                            pt.getHeapType(),
                            pt.getNullability());
            }
          }
        }
      } else if (auto* ret = curr->dynCast<Return>()) {
        Type rt = func->getResults();
        if (ret->value && rt.isRef() && !rt.isExact()) {
          markAndRecord(
            &ret->value, true, true, rt.getHeapType(), rt.getNullability());
        }
      }
    }
  } scanner(wasm, func, out, recordChild);

  Type rt = func->getResults();
  if (rt.isRef() && !rt.isExact()) {
    scanner.propagateBlockDemand(func->body, rt);
    scanner.markAndRecord(
      &func->body, true, true, rt.getHeapType(), rt.getNullability());
  }
  scanner.walk(func->body);

  // Second pass to record any remaining unhandled LocalGet / Call locations
  // as non-fixed consumers (which require their current type).
  struct FallbackScanner : public PostWalker<FallbackScanner> {
    Scanner& prev;
    FallbackScanner(Scanner& prev) : prev(prev) {}
    void visitLocalGet(LocalGet* curr) {
      if (!prev.handled.count(curr) && curr->index < prev.func->getNumParams() &&
          curr->type.isRef()) {
        prev.record(getCurrentPointer(),
                    false,
                    false,
                    curr->type.getHeapType(),
                    curr->type.getNullability());
      }
    }
    void visitCall(Call* curr) {
      if (!prev.handled.count(curr) && curr->type.isRef() && !curr->isReturn) {
        prev.record(getCurrentPointer(),
                    false,
                    false,
                    curr->type.getHeapType(),
                    curr->type.getNullability());
      }
    }
  } fallback(scanner);
  fallback.walk(func->body);

  bool needDomOrder = false;
  for (Index i = 0; i < numParams; ++i) {
    if (out.localWriteCounts[i] == 0 && out.paramSites[i].size() >= 2) {
      needDomOrder = true;
      break;
    }
  }
  if (needDomOrder) {
    struct DomOrderScanner : public PostWalker<DomOrderScanner> {
      Index numParams;
      Index curOrder = 0;
      struct ActiveScope {
        Index id;
        Name blockName;
        bool isConditional;
        bool hasPreBranch;
      };
      std::vector<ActiveScope> scopeStack;
      std::vector<Index> scopeEndOrder;
      std::unordered_map<LocalGet*, std::pair<Index, Index>> getDomRecord;

      DomOrderScanner(Index numParams) : numParams(numParams) {
        scopeEndOrder.push_back(0);
        scopeStack.push_back({0, Name(), false, false});
      }

      void beginScope(Name blockName, bool isConditional) {
        Index id = scopeEndOrder.size();
        scopeEndOrder.push_back(0);
        scopeStack.push_back({id, blockName, isConditional, false});
      }

      static void doBeginCondScope(DomOrderScanner* self, Expression** currp) {
        self->beginScope(Name(), true);
      }

      static void doBeginBlockScope(DomOrderScanner* self, Expression** currp) {
        self->beginScope((*currp)->cast<Block>()->name, false);
      }

      static void doEndScope(DomOrderScanner* self, Expression** currp) {
        self->scopeEndOrder[self->scopeStack.back().id] = self->curOrder;
        self->scopeStack.pop_back();
      }

      void markBranchTarget(Name target) {
        if (!target.is()) {
          return;
        }
        for (auto it = scopeStack.rbegin(); it != scopeStack.rend(); ++it) {
          if (it->blockName == target) {
            it->hasPreBranch = true;
            break;
          }
        }
      }

      void visitLocalGet(LocalGet* curr) {
        if (curr->index < numParams && curr->type.isRef()) {
          Index order = ++curOrder;
          Index domId = 0;
          for (auto it = scopeStack.rbegin(); it != scopeStack.rend(); ++it) {
            if (it->isConditional || it->hasPreBranch) {
              domId = it->id;
              break;
            }
          }
          getDomRecord[curr] = {order, domId};
        }
      }

      void visitBreak(Break* curr) { markBranchTarget(curr->name); }

      void visitSwitch(Switch* curr) {
        for (auto target : curr->targets) {
          markBranchTarget(target);
        }
        markBranchTarget(curr->default_);
      }

      void visitBrOn(BrOn* curr) { markBranchTarget(curr->name); }

      static void scan(DomOrderScanner* self, Expression** currp) {
        Expression* curr = *currp;
        switch (curr->_id) {
          case Expression::Id::BlockId: {
            auto* block = curr->cast<Block>();
            if (block->name.is()) {
              self->pushTask(DomOrderScanner::doEndScope, currp);
            }
            auto& list = block->list;
            for (int i = int(list.size()) - 1; i >= 0; i--) {
              self->pushTask(DomOrderScanner::scan, &list[i]);
            }
            if (block->name.is()) {
              doBeginBlockScope(self, currp);
            }
            return;
          }
          case Expression::Id::IfId: {
            auto* iff = curr->cast<If>();
            if (iff->ifFalse) {
              self->pushTask(DomOrderScanner::doEndScope, currp);
              self->pushTask(DomOrderScanner::scan, &iff->ifFalse);
              self->pushTask(DomOrderScanner::doBeginCondScope, currp);
            }
            self->pushTask(DomOrderScanner::doEndScope, currp);
            self->pushTask(DomOrderScanner::scan, &iff->ifTrue);
            self->pushTask(DomOrderScanner::doBeginCondScope, currp);
            self->pushTask(DomOrderScanner::scan, &iff->condition);
            return;
          }
          case Expression::Id::LoopId: {
            auto* loop = curr->cast<Loop>();
            self->pushTask(DomOrderScanner::doEndScope, currp);
            self->pushTask(DomOrderScanner::scan, &loop->body);
            doBeginCondScope(self, currp);
            return;
          }
          case Expression::Id::TryId: {
            auto* try_ = curr->cast<Try>();
            auto& list = try_->catchBodies;
            for (int i = int(list.size()) - 1; i >= 0; i--) {
              self->pushTask(DomOrderScanner::doEndScope, currp);
              self->pushTask(DomOrderScanner::scan, &list[i]);
              self->pushTask(DomOrderScanner::doBeginCondScope, currp);
            }
            self->pushTask(DomOrderScanner::doEndScope, currp);
            self->pushTask(DomOrderScanner::scan, &try_->body);
            doBeginCondScope(self, currp);
            return;
          }
          case Expression::Id::TryTableId: {
            auto* tryTable = curr->cast<TryTable>();
            for (auto dest : tryTable->catchDests) {
              self->markBranchTarget(dest);
            }
            self->pushTask(DomOrderScanner::doEndScope, currp);
            self->pushTask(DomOrderScanner::scan, &tryTable->body);
            doBeginCondScope(self, currp);
            return;
          }
          default: {
            PostWalker<DomOrderScanner>::scan(self, currp);
            return;
          }
        }
      }
    } domScanner(numParams);
    domScanner.walk(func->body);
    domScanner.scopeEndOrder[0] = domScanner.curOrder;
    for (Index i = 0; i < numParams; ++i) {
      if (out.localWriteCounts[i] == 0 && out.paramSites[i].size() >= 2) {
        for (auto& site : out.paramSites[i]) {
          auto it = domScanner.getDomRecord.find(site.get);
          if (it != domScanner.getDomRecord.end()) {
            site.visitOrder = it->second.first;
            site.domEndOrder = domScanner.scopeEndOrder[it->second.second];
          }
        }
        std::sort(out.paramSites[i].begin(),
                  out.paramSites[i].end(),
                  [](const ConsumerSite& a, const ConsumerSite& b) {
                    return a.visitOrder < b.visitOrder;
                  });
      }
    }
  }
}

struct FieldKey {
  HeapType anc;
  Index index;
  bool operator==(const FieldKey& o) const {
    return anc == o.anc && index == o.index;
  }
};

struct FieldKeyHash {
  size_t operator()(const FieldKey& k) const {
    auto h = std::hash<HeapType>{}(k.anc);
    rehash(h, k.index);
    return h;
  }
};

struct FieldDemandState {
  HeapType baseHT;
  HeapType allowedHT;
  bool canWidenHT = true;
  bool acceptsNull = true;
  bool pinned = false;

  void require(HeapType reqHT, bool reqNull) {
    if (pinned) {
      return;
    }
    if (!reqNull) {
      acceptsNull = false;
    }
    if (!canWidenHT) {
      return;
    }
    if (!HeapType::isSubType(baseHT, reqHT)) {
      canWidenHT = false;
      allowedHT = baseHT;
    } else if (HeapType::isSubType(reqHT, allowedHT)) {
      allowedHT = reqHT;
    }
  }

  void mergeIn(const FieldDemandState& o) {
    if (o.pinned) {
      pin();
      return;
    }
    if (!o.acceptsNull) {
      acceptsNull = false;
    }
    if (!o.canWidenHT) {
      canWidenHT = false;
      allowedHT = baseHT;
    } else if (canWidenHT) {
      require(o.allowedHT, true);
    }
  }

  void pin() {
    pinned = true;
    canWidenHT = false;
    allowedHT = baseHT;
    acceptsNull = false;
  }
};

struct FuncFieldDemandSummary {
  struct FieldReq {
    FieldKey key;
    HeapType reqHT;
    bool reqNull;
  };
  struct ArrayReq {
    HeapType arrHT;
    HeapType reqHT;
    bool reqNull;
  };
  struct GlobalReq {
    Name name;
    HeapType reqHT;
    bool reqNull;
  };
  std::vector<FieldReq> fieldReqs;
  std::vector<FieldReq> immFieldReqs;
  std::vector<FieldKey> pinnedFields;
  std::vector<FieldKey> pinnedImmFields;
  std::vector<ArrayReq> arrayReqs;
  std::vector<HeapType> pinnedArrays;
  std::vector<HeapType> castStructs;
  std::vector<HeapType> exactCastStructs;
  std::unordered_map<HeapType, Index> structTypeUses;
  std::unordered_map<HeapType, std::vector<StructNew*>> structNews;
  std::vector<GlobalReq> pureGlobalReqs;
  std::unordered_map<Name, Index> totalGlobalGets;
  std::unordered_map<Name, Index> validArrayGlobalGets;
};

struct FieldAndArrayDemandWalker
  : public ChildTyper<FieldAndArrayDemandWalker> {
  const std::unordered_set<Name>& pureGlobalCandidates;
  const std::unordered_set<FieldKey, FieldKeyHash>& nullRelaxableFields;
  const std::unordered_set<HeapType>& nullRelaxableArrays;
  FuncFieldDemandSummary& out;
  std::unordered_map<Name, HeapType> labelTypes;
  std::unordered_map<Name, bool> labelAcceptsNull;
  std::optional<HeapType> bodyReq;
  bool bodyAcceptsNull = true;

  FieldAndArrayDemandWalker(
    Module& wasm,
    Function* func,
    const std::unordered_set<Name>& pureGlobalCandidates,
    const std::unordered_set<FieldKey, FieldKeyHash>& nullRelaxableFields,
    const std::unordered_set<HeapType>& nullRelaxableArrays,
    FuncFieldDemandSummary& out)
    : ChildTyper(wasm, func), pureGlobalCandidates(pureGlobalCandidates),
      nullRelaxableFields(nullRelaxableFields),
      nullRelaxableArrays(nullRelaxableArrays), out(out) {
    if (func) {
      auto results = func->getResults();
      if (results.isRef()) {
        bodyReq = results.getHeapType();
        bodyAcceptsNull = results.isNullable();
      }
    }
  }

  bool isNullRelaxableStructField(HeapType uHT, Index fi) const {
    if (func == nullptr || !uHT.isStruct()) {
      return false;
    }
    const auto& fields = uHT.getStruct().fields;
    if (fi >= fields.size()) {
      return false;
    }
    const auto& f = fields[fi];
    if (f.isPacked() || !f.type.isRef() || f.type.isExact() ||
        f.type.isNull()) {
      return false;
    }
    FieldKey key = f.mutable_ == Mutable
                     ? FieldKey{getHighestFieldAncestor(uHT, fi), fi}
                     : FieldKey{uHT, fi};
    return nullRelaxableFields.count(key) != 0;
  }

  bool isNullRelaxableArrayElem(HeapType arrHT) const {
    return func != nullptr && arrHT.isArray() &&
           nullRelaxableArrays.count(arrHT) != 0;
  }

  using ChildTyper<FieldAndArrayDemandWalker>::note;

  void note(Expression** childp, Constraints constraints) {
    auto* child = *childp;
    if (constraints.size() == 1) {
      if (auto* type = std::get_if<Type>(&constraints[0])) {
        if (type->isRef() && !type->isExact()) {
          walk(child, type->getHeapType(), type->isNullable());
          return;
        } else if (type->isRef() && type->isExact()) {
          walk(child,
               child->type.isRef()
                 ? std::optional<HeapType>(child->type.getHeapType())
                 : std::nullopt,
               false);
          return;
        } else if (!type->isRef()) {
          walk(child, std::nullopt, false);
          return;
        }
      } else if (auto* ref = std::get_if<VarRef>(&constraints[0])) {
        bool nullOk = false;
        if (auto* null = std::get_if<Nullability>(&ref->null)) {
          nullOk = (*null == Nullable);
        }
        walk(child,
             child->type.isRef()
               ? std::optional<HeapType>(child->type.getHeapType())
               : std::nullopt,
             nullOk);
        return;
      }
    }
    if (child->type.isRef()) {
      walk(child, child->type.getHeapType(), false);
    } else {
      walk(child, std::nullopt, false);
    }
  }

  Type getLabelType(Name label) { WASM_UNREACHABLE("unexpected branch"); }

  void noteUnknown() {}

  void visitStructNew(StructNew* curr) {
    if (!curr->type.isStruct()) {
      return;
    }
    HeapType uHT = curr->type.getHeapType();
    out.structTypeUses[uHT]++;
    out.structNews[uHT].push_back(curr);
    if (!curr->isWithDefault()) {
      const auto& fields = uHT.getStruct().fields;
      assert(fields.size() == curr->operands.size());
      for (size_t i = 0; i < fields.size(); ++i) {
        Type ft = fields[i].type;
        if (isNullRelaxableStructField(uHT, i)) {
          ft = Type(ft.getHeapType(), Nullable);
        }
        note(&curr->operands[i], ft);
      }
    }
    if (auto desc = uHT.getDescriptorType()) {
      note(&curr->desc, Type(*desc, NonNullable, Exact));
    }
  }

  void visitStructSet(StructSet* curr) {
    if (!curr->ref->type.isStruct()) {
      walk(curr->ref, std::nullopt, false);
      walk(curr->value, std::nullopt, false);
      return;
    }
    HeapType uHT = curr->ref->type.getHeapType();
    out.structTypeUses[uHT]++;
    auto anc = getHighestDefiningAncestor(uHT, curr->index);
    const auto& fields = anc.getStruct().fields;
    assert(curr->index < fields.size());
    note(&curr->ref, Type(anc, Nullable));
    Type ft = fields[curr->index].type;
    if (isNullRelaxableStructField(uHT, curr->index)) {
      ft = Type(ft.getHeapType(), Nullable);
    }
    note(&curr->value, ft);
  }

  void visitArrayNew(ArrayNew* curr) {
    if (!curr->isWithDefault()) {
      if (!curr->type.isArray()) {
        return;
      }
      HeapType arrHT = curr->type.getHeapType();
      Type et = arrHT.getArray().element.type;
      if (isNullRelaxableArrayElem(arrHT)) {
        et = Type(et.getHeapType(), Nullable);
      }
      note(&curr->init, et);
    }
    note(&curr->size, Type::i32);
  }

  void visitArrayNewFixed(ArrayNewFixed* curr) {
    if (!curr->type.isArray()) {
      return;
    }
    HeapType arrHT = curr->type.getHeapType();
    Type et = arrHT.getArray().element.type;
    if (isNullRelaxableArrayElem(arrHT)) {
      et = Type(et.getHeapType(), Nullable);
    }
    for (auto& expr : curr->values) {
      note(&expr, et);
    }
  }

  void visitArraySet(ArraySet* curr) {
    if (!curr->ref->type.isArray()) {
      walk(curr->ref, std::nullopt, false);
      walk(curr->index, std::nullopt, false);
      walk(curr->value, std::nullopt, false);
      return;
    }
    HeapType arrHT = curr->ref->type.getHeapType();
    Type et = arrHT.getArray().element.type;
    if (isNullRelaxableArrayElem(arrHT)) {
      et = Type(et.getHeapType(), Nullable);
    }
    note(&curr->ref, Type(arrHT, Nullable));
    if (auto* gget = curr->ref->dynCast<GlobalGet>()) {
      if (pureGlobalCandidates.count(gget->name) &&
          arrHT.getArray().element.mutable_ == Mutable) {
        out.validArrayGlobalGets[gget->name]++;
      }
    }
    note(&curr->index, Type::i32);
    note(&curr->value, et);
  }

  void visitArrayFill(ArrayFill* curr) {
    if (!curr->ref->type.isArray()) {
      walk(curr->ref, std::nullopt, false);
      walk(curr->index, std::nullopt, false);
      walk(curr->value, std::nullopt, false);
      walk(curr->size, std::nullopt, false);
      return;
    }
    HeapType arrHT = curr->ref->type.getHeapType();
    Type et = arrHT.getArray().element.type;
    if (isNullRelaxableArrayElem(arrHT)) {
      et = Type(et.getHeapType(), Nullable);
    }
    note(&curr->ref, Type(arrHT, Nullable));
    note(&curr->index, Type::i32);
    note(&curr->value, et);
    note(&curr->size, Type::i32);
  }

  void visitRefIsNull(RefIsNull* curr) {
    if (curr->value->type.isRef()) {
      walk(curr->value, curr->value->type.getHeapType().getTop(), true);
    } else {
      walk(curr->value, std::nullopt, false);
    }
  }

  void visitRefEq(RefEq* curr) {
    if (curr->left->type.isRef()) {
      walk(curr->left,
           HeapTypes::eq.getBasic(curr->left->type.getHeapType().getShared()),
           true);
    } else {
      walk(curr->left, std::nullopt, false);
    }
    if (curr->right->type.isRef()) {
      walk(curr->right,
           HeapTypes::eq.getBasic(curr->right->type.getHeapType().getShared()),
           true);
    } else {
      walk(curr->right, std::nullopt, false);
    }
  }

  void visitDrop(Drop* curr) {
    if (curr->value->type.isRef()) {
      walk(curr->value, curr->value->type.getHeapType().getTop(), true);
    } else {
      walk(curr->value, std::nullopt, false);
    }
  }

  void visitRefTest(RefTest* curr) {
    if (curr->castType.isArray()) {
      out.pinnedArrays.push_back(curr->castType.getHeapType());
    } else if (curr->castType.isStruct()) {
      HeapType cHT = curr->castType.getHeapType();
      out.castStructs.push_back(cHT);
      out.structTypeUses[cHT]++;
      if (curr->castType.isExact()) {
        out.exactCastStructs.push_back(cHT);
      }
    }
    if (curr->castType.isExact() && !wasm.features.hasCustomDescriptors()) {
      if (curr->ref->type.isRef()) {
        walk(curr->ref, curr->ref->type.getHeapType(), true);
      } else {
        walk(curr->ref, std::nullopt, false);
      }
      return;
    }
    ChildTyper<FieldAndArrayDemandWalker>::visitRefTest(curr);
  }

  void walk(Expression* curr,
            std::optional<HeapType> reqHT,
            bool acceptsNull = true) {
    if (auto* sget = curr->dynCast<StructGet>()) {
      if (sget->ref->type.isStruct()) {
        HeapType uHT = sget->ref->type.getHeapType();
        out.structTypeUses[uHT]++;
        const auto& field = uHT.getStruct().fields[sget->index];
        if (sget->type.isRef()) {
          HeapType effHT = reqHT ? *reqHT : sget->type.getHeapType();
          bool effNull = reqHT ? acceptsNull : sget->type.isNullable();
          if (field.mutable_ == Mutable) {
            HeapType anc = getHighestFieldAncestor(uHT, sget->index);
            out.fieldReqs.push_back({{anc, sget->index}, effHT, effNull});
            if (anc == uHT) {
              out.immFieldReqs.push_back({{uHT, sget->index}, effHT, effNull});
            }
          } else {
            out.immFieldReqs.push_back({{uHT, sget->index}, effHT, effNull});
          }
          if (field.mutable_ == Mutable) {
            auto defAnc = getHighestDefiningAncestor(uHT, sget->index);
            walk(sget->ref, defAnc, true);
          } else {
            walk(sget->ref, uHT, true);
          }
        } else {
          auto defAnc = getHighestDefiningAncestor(uHT, sget->index);
          walk(sget->ref, defAnc, true);
        }
      } else {
        walk(sget->ref, std::nullopt, false);
      }
      return;
    }
    if (auto* aget = curr->dynCast<ArrayGet>()) {
      if (aget->type.isRef() && aget->ref->type.isArray()) {
        HeapType arrHT = aget->ref->type.getHeapType();
        HeapType effHT = reqHT ? *reqHT : aget->type.getHeapType();
        bool effNull = reqHT ? acceptsNull : aget->type.isNullable();
        if (auto* gget = aget->ref->dynCast<GlobalGet>()) {
          if (pureGlobalCandidates.count(gget->name)) {
            out.validArrayGlobalGets[gget->name]++;
            out.pureGlobalReqs.push_back({gget->name, effHT, effNull});
          } else {
            out.arrayReqs.push_back({arrHT, effHT, effNull});
          }
        } else {
          out.arrayReqs.push_back({arrHT, effHT, effNull});
        }
      }
      if (aget->ref->type.isArray()) {
        walk(aget->ref, aget->ref->type.getHeapType(), true);
      } else {
        walk(aget->ref, std::nullopt, false);
      }
      walk(aget->index, std::nullopt, false);
      return;
    }
    if (auto* alen = curr->dynCast<ArrayLen>()) {
      if (auto* gget = alen->ref->dynCast<GlobalGet>()) {
        if (pureGlobalCandidates.count(gget->name)) {
          out.validArrayGlobalGets[gget->name]++;
        }
      }
      if (alen->ref->type.isRef()) {
        walk(alen->ref, HeapType::array, true);
      } else {
        walk(alen->ref, std::nullopt, false);
      }
      return;
    }
    if (auto* gget = curr->dynCast<GlobalGet>()) {
      if (pureGlobalCandidates.count(gget->name)) {
        out.totalGlobalGets[gget->name]++;
      }
      return;
    }
    if (auto* acopy = curr->dynCast<ArrayCopy>()) {
      if (acopy->destRef->type.isArray() && acopy->srcRef->type.isArray()) {
        HeapType destHT = acopy->destRef->type.getHeapType();
        HeapType srcHT = acopy->srcRef->type.getHeapType();
        if (destHT != srcHT) {
          out.pinnedArrays.push_back(destHT);
          out.pinnedArrays.push_back(srcHT);
        }
      }
      ChildTyper<FieldAndArrayDemandWalker>::visitArrayCopy(acopy);
      return;
    }
    if (auto* armw = curr->dynCast<ArrayRMW>()) {
      if (armw->ref->type.isArray()) {
        out.pinnedArrays.push_back(armw->ref->type.getHeapType());
      }
      ChildTyper<FieldAndArrayDemandWalker>::visitArrayRMW(armw);
      return;
    }
    if (auto* acx = curr->dynCast<ArrayCmpxchg>()) {
      if (acx->ref->type.isArray()) {
        out.pinnedArrays.push_back(acx->ref->type.getHeapType());
      }
      ChildTyper<FieldAndArrayDemandWalker>::visitArrayCmpxchg(acx);
      return;
    }
    if (auto* srmw = curr->dynCast<StructRMW>()) {
      if (srmw->ref->type.isStruct()) {
        out.pinnedFields.push_back(
          {getHighestFieldAncestor(srmw->ref->type.getHeapType(), srmw->index),
           srmw->index});
      }
      ChildTyper<FieldAndArrayDemandWalker>::visitStructRMW(srmw);
      return;
    }
    if (auto* scx = curr->dynCast<StructCmpxchg>()) {
      if (scx->ref->type.isStruct()) {
        out.pinnedFields.push_back(
          {getHighestFieldAncestor(scx->ref->type.getHeapType(), scx->index),
           scx->index});
      }
      ChildTyper<FieldAndArrayDemandWalker>::visitStructCmpxchg(scx);
      return;
    }
    if (auto* swait = curr->dynCast<StructWait>()) {
      if (swait->ref->type.isStruct()) {
        out.pinnedFields.push_back(
          {getHighestFieldAncestor(swait->ref->type.getHeapType(),
                                   swait->index),
           swait->index});
      }
      ChildTyper<FieldAndArrayDemandWalker>::visitStructWait(swait);
      return;
    }
    if (curr->is<LocalGet>()) {
      return;
    }
    if (auto* set = curr->dynCast<LocalSet>()) {
      Type localType = func->getLocalType(set->index);
      if (localType.isRef() && !localType.isExact()) {
        walk(set->value, localType.getHeapType(), localType.isNullable());
      } else {
        walk(set->value,
             localType.isRef()
               ? std::optional<HeapType>(localType.getHeapType())
               : std::nullopt,
             false);
      }
      return;
    }
    if (auto* as = curr->dynCast<RefAs>()) {
      if (as->op == RefAsNonNull) {
        walk(as->value,
             reqHT ? reqHT
                   : (as->type.isRef()
                        ? std::optional<HeapType>(as->type.getHeapType())
                        : std::nullopt),
             true);
        return;
      }
    }
    if (auto* cast = curr->dynCast<RefCast>()) {
      if (cast->type.isStruct()) {
        HeapType cHT = cast->type.getHeapType();
        out.structTypeUses[cHT]++;
        if (cast->type.isExact()) {
          out.exactCastStructs.push_back(cHT);
        }
      } else if (cast->type.isArray() && cast->type.isExact()) {
        out.pinnedArrays.push_back(cast->type.getHeapType());
      }
      if (cast->desc) {
        walk(cast->desc,
             cast->desc->type.isRef()
               ? std::optional<HeapType>(cast->desc->type.getHeapType())
               : std::nullopt,
             cast->desc->type.isNullable());
      }
      if (!cast->ref->type.isRef()) {
        walk(cast->ref, std::nullopt, false);
        return;
      }
      if ((cast->type.isExact() && !wasm.features.hasCustomDescriptors()) ||
          cast->desc) {
        walk(cast->ref, cast->ref->type.getHeapType(), true);
        return;
      }
      walk(cast->ref, cast->ref->type.getHeapType().getTop(), true);
      return;
    }
    if (auto* call = curr->dynCast<Call>()) {
      if (Intrinsics(wasm).isCallWithoutEffects(call)) {
        for (auto* op : call->operands) {
          walk(op,
               op->type.isRef()
                 ? std::optional<HeapType>(op->type.getHeapType())
                 : std::nullopt,
               op->type.isRef() && op->type.isNullable());
        }
        return;
      }
      auto* callee = wasm.getFunction(call->target);
      auto params = callee->getParams();
      for (size_t i = 0; i < call->operands.size(); ++i) {
        std::optional<HeapType> argReq;
        bool argNull = false;
        if (i < params.size() && params[i].isRef()) {
          argReq = params[i].getHeapType();
          argNull = !params[i].isExact() && params[i].isNullable();
        }
        walk(call->operands[i], argReq, argNull);
      }
      return;
    }
    if (auto* callRef = curr->dynCast<CallRef>()) {
      if (!callRef->target->type.isSignature()) {
        for (auto* op : callRef->operands) {
          walk(op,
               op->type.isRef()
                 ? std::optional<HeapType>(op->type.getHeapType())
                 : std::nullopt,
               false);
        }
        walk(callRef->target, std::nullopt, false);
        return;
      }
      HeapType sigHT = callRef->target->type.getHeapType();
      auto params = sigHT.getSignature().params;
      for (size_t i = 0; i < callRef->operands.size(); ++i) {
        std::optional<HeapType> argReq;
        bool argNull = false;
        if (i < params.size() && params[i].isRef()) {
          argReq = params[i].getHeapType();
          argNull = !params[i].isExact() && params[i].isNullable();
        }
        walk(callRef->operands[i], argReq, argNull);
      }
      walk(callRef->target, sigHT, true);
      return;
    }
    if (auto* block = curr->dynCast<Block>()) {
      if (block->list.empty()) {
        return;
      }
      std::optional<HeapType> tailReq;
      bool tailNull = false;
      if (block->type.isRef()) {
        if (block->type.isExact()) {
          tailReq = block->type.getHeapType();
          tailNull = false;
        } else {
          tailReq = reqHT ? *reqHT : block->type.getHeapType();
          tailNull = reqHT ? acceptsNull : block->type.isNullable();
        }
      }
      std::optional<HeapType> prevLabelType;
      bool prevLabelNull = false;
      if (block->name.is()) {
        auto it = labelTypes.find(block->name);
        if (it != labelTypes.end()) {
          prevLabelType = it->second;
          prevLabelNull = labelAcceptsNull[block->name];
        }
        if (tailReq) {
          labelTypes[block->name] = *tailReq;
          labelAcceptsNull[block->name] = tailNull;
        } else {
          labelTypes.erase(block->name);
          labelAcceptsNull.erase(block->name);
        }
      }
      for (size_t i = 0; i + 1 < block->list.size(); ++i) {
        walk(block->list[i], std::nullopt, false);
      }
      if (!block->type.isRef()) {
        walk(block->list.back(), std::nullopt, false);
      } else {
        walk(block->list.back(), tailReq, tailNull);
      }
      if (block->name.is()) {
        if (prevLabelType) {
          labelTypes[block->name] = *prevLabelType;
          labelAcceptsNull[block->name] = prevLabelNull;
        } else {
          labelTypes.erase(block->name);
          labelAcceptsNull.erase(block->name);
        }
      }
      return;
    }
    if (auto* iff = curr->dynCast<If>()) {
      walk(iff->condition, std::nullopt, false);
      std::optional<HeapType> armReq;
      bool armNull = false;
      if (iff->type.isRef()) {
        if (iff->type.isExact()) {
          armReq = iff->type.getHeapType();
          armNull = false;
        } else {
          armReq = reqHT ? reqHT
                         : std::optional<HeapType>(iff->type.getHeapType());
          armNull = reqHT ? acceptsNull : iff->type.isNullable();
        }
      }
      walk(iff->ifTrue, armReq, armNull);
      if (iff->ifFalse) {
        walk(iff->ifFalse, armReq, armNull);
      }
      return;
    }
    if (auto* loop = curr->dynCast<Loop>()) {
      std::optional<HeapType> loopReq;
      bool loopNull = false;
      if (loop->type.isRef()) {
        if (loop->type.isExact()) {
          loopReq = loop->type.getHeapType();
          loopNull = false;
        } else {
          loopReq =
            reqHT ? reqHT : std::optional<HeapType>(loop->type.getHeapType());
          loopNull = reqHT ? acceptsNull : loop->type.isNullable();
        }
      }
      walk(loop->body, loopReq, loopNull);
      return;
    }
    if (auto* try_ = curr->dynCast<Try>()) {
      std::optional<HeapType> tryReq;
      bool tryNull = false;
      if (try_->type.isRef()) {
        if (try_->type.isExact()) {
          tryReq = try_->type.getHeapType();
          tryNull = false;
        } else {
          tryReq =
            reqHT ? reqHT : std::optional<HeapType>(try_->type.getHeapType());
          tryNull = reqHT ? acceptsNull : try_->type.isNullable();
        }
      }
      walk(try_->body, tryReq, tryNull);
      for (auto* catchBody : try_->catchBodies) {
        walk(catchBody, tryReq, tryNull);
      }
      return;
    }
    if (auto* tryTable = curr->dynCast<TryTable>()) {
      std::optional<HeapType> tryReq;
      bool tryNull = false;
      if (tryTable->type.isRef()) {
        if (tryTable->type.isExact()) {
          tryReq = tryTable->type.getHeapType();
          tryNull = false;
        } else {
          tryReq = reqHT
                     ? reqHT
                     : std::optional<HeapType>(tryTable->type.getHeapType());
          tryNull = reqHT ? acceptsNull : tryTable->type.isNullable();
        }
      }
      walk(tryTable->body, tryReq, tryNull);
      return;
    }
    if (auto* br = curr->dynCast<Break>()) {
      if (br->condition) {
        walk(br->condition, std::nullopt, false);
      }
      if (br->value) {
        std::optional<HeapType> valReq;
        bool valNull = false;
        if (auto it = labelTypes.find(br->name); it != labelTypes.end()) {
          valReq = it->second;
          valNull = labelAcceptsNull[br->name];
        } else if (br->value->type.isRef()) {
          valReq = br->value->type.getHeapType();
        }
        if (br->condition && br->type.isRef()) {
          HeapType flowReq = reqHT ? *reqHT : br->type.getHeapType();
          bool flowNull = reqHT ? acceptsNull : br->type.isNullable();
          if (!valReq || HeapType::isSubType(flowReq, *valReq)) {
            valReq = flowReq;
          }
          valNull = valNull && flowNull;
        }
        walk(br->value, valReq, valNull);
      }
      return;
    }
    if (auto* sw = curr->dynCast<Switch>()) {
      walk(sw->condition, std::nullopt, false);
      if (sw->value) {
        walk(sw->value,
             sw->value->type.isRef()
               ? std::optional<HeapType>(sw->value->type.getHeapType())
               : std::nullopt,
             false);
      }
      return;
    }
    if (auto* brOn = curr->dynCast<BrOn>()) {
      if (brOn->op == BrOnCast || brOn->op == BrOnCastFail) {
        if (brOn->castType.isArray()) {
          out.pinnedArrays.push_back(brOn->castType.getHeapType());
        } else if (brOn->castType.isStruct()) {
          HeapType cHT = brOn->castType.getHeapType();
          out.castStructs.push_back(cHT);
          out.structTypeUses[cHT]++;
          if (brOn->castType.isExact()) {
            out.exactCastStructs.push_back(cHT);
          }
        }
      }
      if (brOn->desc) {
        walk(brOn->desc,
             brOn->desc->type.isRef()
               ? std::optional<HeapType>(brOn->desc->type.getHeapType())
               : std::nullopt,
             brOn->desc->type.isNullable());
      }
      if (!brOn->ref->type.isRef()) {
        walk(brOn->ref, std::nullopt, false);
        return;
      }
      std::optional<HeapType> refReq = brOn->ref->type.getHeapType();
      bool refNull = brOn->ref->type.isNullable();
      if (brOn->op == BrOnNull) {
        if (!brOn->ref->type.isExact() && reqHT) {
          refReq = reqHT;
        }
        refNull = true;
      } else if (brOn->op == BrOnNonNull) {
        if (!brOn->ref->type.isExact()) {
          if (auto it = labelTypes.find(brOn->name); it != labelTypes.end()) {
            refReq = it->second;
          }
        }
        refNull = true;
      } else if (!brOn->desc && !brOn->ref->type.isExact()) {
        if (brOn->op == BrOnCast) {
          refNull = refNull || brOn->castType.isNullable() || acceptsNull;
        } else if (brOn->op == BrOnCastFail) {
          auto it = labelAcceptsNull.find(brOn->name);
          bool targetNull = it != labelAcceptsNull.end() && it->second;
          refNull = refNull || (brOn->castType.isNullable() ? acceptsNull
                                                            : targetNull);
        }
      }
      walk(brOn->ref, refReq, refNull);
      return;
    }
    if (auto* sel = curr->dynCast<Select>()) {
      walk(sel->condition, std::nullopt, false);
      std::optional<HeapType> armReq;
      bool armNull = false;
      if (sel->type.isRef()) {
        if (sel->type.isExact()) {
          armReq = sel->type.getHeapType();
          armNull = false;
        } else {
          armReq = reqHT ? reqHT
                         : std::optional<HeapType>(sel->type.getHeapType());
          armNull = reqHT ? acceptsNull : sel->type.isNullable();
        }
      }
      walk(sel->ifTrue, armReq, armNull);
      walk(sel->ifFalse, armReq, armNull);
      return;
    }
    if (auto* ret = curr->dynCast<Return>()) {
      if (ret->value) {
        walk(ret->value, bodyReq, bodyAcceptsNull);
      }
      return;
    }
    visit(curr);
  }
};

struct RelaxNullability : public Pass {
  // Signatures and locals change, and all casts are revisited.
  bool invalidatesDWARF() override { return true; }

  void generalizeSignatures(Module* module,
                            const std::unordered_set<Name>& escapedFuncs,
                            const std::unordered_set<Name>& hasReturnCall,
                            const std::unordered_set<HeapType>& fixed,
                            const std::unordered_set<HeapType>& sigUsedAsRef,
                            bool widenFields = false) {
    auto allTypes = ModuleUtils::collectHeapTypes(*module);
    SubTypes subTypes(allTypes);

    std::unordered_map<Signature, HeapType> sigToHT;
    std::unordered_set<HeapType> pinnedHTs;
    for (auto type : allTypes) {
      if (!type.isSignature()) {
        continue;
      }
      if (fixed.count(type) || sigUsedAsRef.count(type) ||
          type.getDeclaredSuperType() ||
          !subTypes.getImmediateSubTypes(type).empty() || type.isShared()) {
        pinnedHTs.insert(type);
      }
      if (!type.getDeclaredSuperType() &&
          subTypes.getImmediateSubTypes(type).empty() && !type.isShared()) {
        sigToHT.try_emplace(type.getSignature(), type);
      }
    }

    std::unordered_map<HeapType, std::vector<Function*>> htFuncs;
    std::vector<HeapType> htOrder;
    std::unordered_set<Signature> existingSigs;
    for (auto type : pinnedHTs) {
      if (type.isSignature()) {
        existingSigs.insert(type.getSignature());
      }
    }
    for (auto& func : module->functions) {
      HeapType ht = func->type.getHeapType();
      if (func->imported() || escapedFuncs.count(func->name) ||
          hasReturnCall.count(func->name)) {
        pinnedHTs.insert(ht);
        existingSigs.insert(func->getSig());
      } else {
        auto& flist = htFuncs[ht];
        if (flist.empty()) {
          htOrder.push_back(ht);
        }
        flist.push_back(func.get());
      }
    }

    ModuleUtils::ParallelFunctionAnalysis<FuncConsumerSummary> summaries(
      *module, [&](Function* func, FuncConsumerSummary& summary) {
        if (!func->imported()) {
          scanFunctionConsumers(*module, func, summary);
        }
      });

    std::unordered_map<Name, std::vector<CallConsumerSite>> callsTo;
    std::unordered_map<Name, std::vector<std::pair<Function*, Call*>>>
      allCallsTo;
    for (auto& func : module->functions) {
      if (func->imported()) {
        continue;
      }
      auto& summary = summaries.map[func.get()];
      for (auto& csite : summary.callSites) {
        callsTo[csite.call->target].push_back(csite);
      }
      for (auto& ac : summary.allCalls) {
        allCallsTo[ac.second->target].push_back(ac);
      }
    }

    // Also count all functions per signature so we know if a signature is shared.
    std::unordered_map<Signature, size_t> sigFuncCount;
    for (auto& func : module->functions) {
      sigFuncCount[func->getSig()]++;
    }

    std::string modeArg = getArgumentOrDefault("relax-nullability", "");
    bool mergeSignatures = hasArgument("relax-nullability-merge-signatures") ||
                           modeArg == "merge-signatures" ||
                           modeArg == "cache-param-casts";
    bool cacheParamCasts =
      modeArg == "cache-param-casts" ||
      hasArgument("relax-nullability-cache-param-casts") ||
      !getPassOptions().optimizeLevel;

    std::optional<HeapType> primaryStructRoot;
    size_t maxStructRootSubs = 0;
    for (auto type : allTypes) {
      if (type.isStruct() && !type.getDeclaredSuperType()) {
        size_t numSubs = subTypes.getImmediateSubTypes(type).size();
        if (numSubs > maxStructRootSubs) {
          maxStructRootSubs = numSubs;
          primaryStructRoot = type;
        }
      }
    }

    // Pre-scan unpinned sole-group DGA cache array types `(array (mut (ref null
    // $T)))` whose `ArrayGet`/`ArraySet` users all belong to a single signature
    // group `ht` and whose `ArrayGet`s flow directly to the function return.
    std::unordered_map<Type, HeapType> existingMutArrayHT;
    std::unordered_map<HeapType, std::vector<HeapType>> coupledArraysForSig;
    std::unordered_map<HeapType, int> arrGlobalCount;
    std::unordered_map<HeapType, int> arrFuncCount;
    std::unordered_map<HeapType, std::vector<HeapType>> arrCastInnerHTs;
    if (widenFields) {
      std::unordered_set<HeapType> candArrayHTs;
      for (auto type : allTypes) {
        if (type.isArray() && !type.isShared() && !type.getDescriptorType() &&
            !type.getDescribedType() && !type.getDeclaredSuperType() &&
            subTypes.getImmediateSubTypes(type).empty()) {
          const auto& elem = type.getArray().element;
          if (!elem.isPacked() && elem.mutable_ == Mutable &&
              elem.type.isRef() && !elem.type.isExact()) {
            existingMutArrayHT.try_emplace(elem.type, type);
            if (!fixed.count(type) && elem.type.isNullable() &&
                !elem.type.isNull()) {
              candArrayHTs.insert(type);
            }
          }
        }
      }
      if (!candArrayHTs.empty()) {
        std::unordered_set<Name> exportedGlobals;
        for (auto& exp : module->exports) {
          if (exp->kind == ExternalKind::Global) {
            exportedGlobals.insert(*exp->getInternalName());
          }
        }
        for (auto& g : module->globals) {
          if (!g->imported() && !exportedGlobals.count(g->name) &&
              g->type.isArray() && candArrayHTs.count(g->type.getHeapType())) {
            arrGlobalCount[g->type.getHeapType()]++;
          }
        }
        struct FuncArrScan {
          std::vector<HeapType> pinned;
          std::vector<HeapType> gets;
          std::vector<HeapType> sets;
          std::vector<std::pair<HeapType, HeapType>> castInners;
        };
        ModuleUtils::ParallelFunctionAnalysis<FuncArrScan> arrScans(
          *module, [&](Function* func, FuncArrScan& scan) {
            if (func->imported()) {
              return;
            }
            std::unordered_set<Name> retBlockNames;
            Expression* cur = func->body;
            while (auto* b = cur->dynCast<Block>()) {
              if (b->name.is()) {
                retBlockNames.insert(b->name);
              }
              if (!b->list.empty()) {
                cur = b->list.back();
              } else {
                break;
              }
            }
            std::unordered_set<Expression*> retTailGets;
            for (auto* tailp : summaries.map[func].returnValueLocs) {
              retTailGets.insert(*tailp);
            }
            struct Scanner : public PostWalker<Scanner> {
              const std::unordered_set<HeapType>& candArrayHTs;
              const std::unordered_set<Name>& retBlockNames;
              std::unordered_set<Expression*>& retTailGets;
              Function* func;
              FuncArrScan& out;
              std::unordered_set<HeapType> seenGets;
              std::unordered_set<HeapType> seenSets;
              std::unordered_map<Type, std::vector<HeapType>> localCastInners;

              Scanner(const std::unordered_set<HeapType>& candArrayHTs,
                      const std::unordered_set<Name>& retBlockNames,
                      std::unordered_set<Expression*>& retTailGets,
                      Function* func,
                      FuncArrScan& out)
                : candArrayHTs(candArrayHTs), retBlockNames(retBlockNames),
                  retTailGets(retTailGets), func(func), out(out) {}

              void visitBrOn(BrOn* curr) {
                if (curr->op == BrOnNonNull &&
                    retBlockNames.count(curr->name)) {
                  retTailGets.insert(curr->ref);
                }
                if ((curr->op == BrOnCast || curr->op == BrOnCastFail) &&
                    curr->castType.isArray() &&
                    candArrayHTs.count(curr->castType.getHeapType())) {
                  out.pinned.push_back(curr->castType.getHeapType());
                }
              }
              void visitRefTest(RefTest* curr) {
                if (curr->castType.isArray() &&
                    candArrayHTs.count(curr->castType.getHeapType())) {
                  out.pinned.push_back(curr->castType.getHeapType());
                }
              }
              void visitArrayCopy(ArrayCopy* curr) {
                if (curr->destRef->type.isArray() &&
                    curr->srcRef->type.isArray()) {
                  HeapType dHT = curr->destRef->type.getHeapType();
                  HeapType sHT = curr->srcRef->type.getHeapType();
                  if (dHT != sHT) {
                    if (candArrayHTs.count(dHT)) {
                      out.pinned.push_back(dHT);
                    }
                    if (candArrayHTs.count(sHT)) {
                      out.pinned.push_back(sHT);
                    }
                  }
                }
              }
              void visitArrayRMW(ArrayRMW* curr) {
                if (curr->ref->type.isArray() &&
                    candArrayHTs.count(curr->ref->type.getHeapType())) {
                  out.pinned.push_back(curr->ref->type.getHeapType());
                }
              }
              void visitArrayCmpxchg(ArrayCmpxchg* curr) {
                if (curr->ref->type.isArray() &&
                    candArrayHTs.count(curr->ref->type.getHeapType())) {
                  out.pinned.push_back(curr->ref->type.getHeapType());
                }
              }
              void visitLocalSet(LocalSet* curr) {
                Type lt = func->getLocalType(curr->index);
                if (!lt.isRef()) {
                  return;
                }
                Expression* val = curr->value;
                while (auto* as = val->dynCast<RefAs>()) {
                  if (as->op == RefAsNonNull) {
                    val = as->value;
                  } else {
                    break;
                  }
                }
                if (auto* cast = val->dynCast<RefCast>()) {
                  if (!cast->desc && cast->type.isRef() &&
                      cast->type.getHeapType() == lt.getHeapType() &&
                      cast->ref->type.isRef() && !cast->ref->type.isNull()) {
                    localCastInners[lt.with(Nullable)].push_back(
                      cast->ref->type.getHeapType());
                  }
                }
              }
              void visitArrayGet(ArrayGet* curr) {
                if (curr->ref->type.isArray()) {
                  HeapType arrHT = curr->ref->type.getHeapType();
                  if (candArrayHTs.count(arrHT)) {
                    seenGets.insert(arrHT);
                  }
                }
              }
              void visitArraySet(ArraySet* curr) {
                if (curr->ref->type.isArray()) {
                  HeapType arrHT = curr->ref->type.getHeapType();
                  if (candArrayHTs.count(arrHT)) {
                    seenSets.insert(arrHT);
                  }
                }
              }
              void visitArrayFill(ArrayFill* curr) {
                if (curr->ref->type.isArray()) {
                  HeapType arrHT = curr->ref->type.getHeapType();
                  if (candArrayHTs.count(arrHT)) {
                    seenSets.insert(arrHT);
                  }
                }
              }
            } scanner(candArrayHTs, retBlockNames, retTailGets, func, scan);
            scanner.walk(func->body);
            // Verify every ArrayGet of a seen candidate array flows to return.
            for (auto* aget : FindAll<ArrayGet>(func->body).list) {
              if (aget->ref->type.isArray()) {
                HeapType arrHT = aget->ref->type.getHeapType();
                if (candArrayHTs.count(arrHT) && !retTailGets.count(aget)) {
                  scan.pinned.push_back(arrHT);
                }
              }
            }
            for (auto arrHT : scanner.seenGets) {
              scan.gets.push_back(arrHT);
            }
            for (auto arrHT : scanner.seenSets) {
              scan.sets.push_back(arrHT);
              Type et = arrHT.getArray().element.type;
              auto cIt = scanner.localCastInners.find(et);
              if (cIt != scanner.localCastInners.end()) {
                for (HeapType innerHT : cIt->second) {
                  scan.castInners.push_back({arrHT, innerHT});
                }
              }
            }
          });

        std::unordered_set<HeapType> pinnedArrs;
        std::unordered_map<HeapType, std::unordered_set<Function*>> arrGetFuncs;
        std::unordered_map<HeapType, std::unordered_set<Function*>> arrSetFuncs;
        for (auto& fn : module->functions) {
          if (fn->imported()) {
            continue;
          }
          auto& scan = arrScans.map[fn.get()];
          for (auto arrHT : scan.pinned) {
            pinnedArrs.insert(arrHT);
          }
          for (auto arrHT : scan.gets) {
            arrGetFuncs[arrHT].insert(fn.get());
          }
          for (auto arrHT : scan.sets) {
            arrSetFuncs[arrHT].insert(fn.get());
          }
          for (auto& [arrHT, innerHT] : scan.castInners) {
            arrCastInnerHTs[arrHT].push_back(innerHT);
          }
        }
        for (auto arrHT : candArrayHTs) {
          if (pinnedArrs.count(arrHT)) {
            continue;
          }
          auto& gFuncs = arrGetFuncs[arrHT];
          auto& sFuncs = arrSetFuncs[arrHT];
          if (gFuncs.empty() || gFuncs != sFuncs) {
            continue;
          }
          Function* firstF = *gFuncs.begin();
          HeapType sigHT = firstF->type.getHeapType();
          if (pinnedHTs.count(sigHT) ||
              sigHT.getSignature().results != arrHT.getArray().element.type) {
            continue;
          }
          auto& flist = htFuncs[sigHT];
          bool allInFlist = true;
          for (auto* gf : gFuncs) {
            if (gf->type.getHeapType() != sigHT ||
                std::find(flist.begin(), flist.end(), gf) == flist.end()) {
              allInFlist = false;
              break;
            }
          }
          if (allInFlist) {
            arrFuncCount[arrHT] = int(gFuncs.size());
            coupledArraysForSig[sigHT].push_back(arrHT);
          }
        }
      }
    }

    auto getTypeBytes = [&](Type t) -> int {
      if (t == Type::none) {
        return 0;
      }
      if (t.isBasic()) {
        return 1;
      }
      if (t.isRef()) {
        HeapType ht = t.getHeapType();
        if (ht.isBasic()) {
          return (t.isNullable() && !ht.isShared()) ? 1 : 2;
        }
        return getRootAncestor(ht) == ht ? 2 : 3;
      }
      return 2;
    };

    auto getSigDefBytes = [&](const Signature& sig) -> int {
      int bytes = 3;
      for (Type p : sig.params) {
        bytes += getTypeBytes(p);
      }
      if (sig.results != Type::none) {
        bytes += getTypeBytes(sig.results);
      }
      return bytes;
    };

    auto addCanonCandidates = [&](const Signature& origSig,
                                  const std::vector<Function*>& flist,
                                  std::vector<Signature>& out) {
      std::vector<Type> origParams(origSig.params.begin(),
                                   origSig.params.end());
      Type origRes = origSig.results;
      std::vector<Type> resChoices = {origRes};
      if (origRes.isRef() && !origRes.isExact()) {
        Type nullRes = Type(origRes.getHeapType(), Nullable);
        Type rootRes = Type(getRootAncestor(origRes.getHeapType()), Nullable);
        HeapType eqHT =
          HeapTypes::eq.getBasic(origRes.getHeapType().getShared());
        Type topRes = Type(origRes.getHeapType().getTop(), Nullable);
        if (nullRes != origRes) {
          resChoices.push_back(nullRes);
        }
        if (rootRes != nullRes) {
          resChoices.push_back(rootRes);
        }
        if (origRes.getHeapType().isBasic() &&
            HeapType::isSubType(origRes.getHeapType(), eqHT)) {
          Type eqRes(eqHT, Nullable);
          if (eqRes != rootRes) {
            resChoices.push_back(eqRes);
          }
        }
        if (topRes != rootRes) {
          resChoices.push_back(topRes);
        }
      }

      std::vector<Type> rootParams = origParams;
      std::vector<Type> topParams = origParams;
      std::vector<Type> unsubTopParams = origParams;
      std::vector<Type> objRootOtherTopParams = origParams;
      std::vector<bool> isWrittenParam(origParams.size(), false);
      bool anyWrittenParam = false;
      for (size_t i = 0; i < origParams.size(); ++i) {
        for (auto* f : flist) {
          if (summaries.map[f].localWriteCounts[i] > 0) {
            isWrittenParam[i] = true;
            anyWrittenParam = true;
            break;
          }
        }
        Type p = origParams[i];
        if (p.isRef() && !p.isExact()) {
          HeapType rHT = getRootAncestor(p.getHeapType());
          HeapType tHT = p.getHeapType().getTop();
          rootParams[i] = Type(rHT, Nullable);
          topParams[i] = Type(tHT, Nullable);
          unsubTopParams[i] =
            (!rHT.isBasic() && subTypes.getImmediateSubTypes(rHT).empty())
              ? Type(tHT, Nullable)
              : Type(rHT, Nullable);
          objRootOtherTopParams[i] =
            (primaryStructRoot && rHT == *primaryStructRoot)
              ? Type(rHT, Nullable)
              : Type(tHT, Nullable);
        }
      }

      std::vector<Type> paramChoices;
      auto addParamChoice = [&](Type pt) {
        if (std::find(paramChoices.begin(), paramChoices.end(), pt) ==
            paramChoices.end()) {
          paramChoices.push_back(pt);
        }
      };
      addParamChoice(Type(rootParams));
      addParamChoice(origSig.params);
      addParamChoice(Type(topParams));
      addParamChoice(Type(unsubTopParams));
      if (cacheParamCasts) {
        addParamChoice(Type(objRootOtherTopParams));
      }
      if (anyWrittenParam) {
        auto addKeepWritten = [&](std::vector<Type> vec) {
          for (size_t i = 0; i < origParams.size(); ++i) {
            if (isWrittenParam[i]) {
              vec[i] = origParams[i];
            }
          }
          addParamChoice(Type(vec));
        };
        addKeepWritten(rootParams);
        addKeepWritten(topParams);
        addKeepWritten(unsubTopParams);
        if (cacheParamCasts) {
          addKeepWritten(objRootOtherTopParams);
        }
      }
      for (size_t i = 0; i < origParams.size(); ++i) {
        Type p = origParams[i];
        if (p.isRef() && !p.isExact()) {
          Type tType(p.getHeapType().getTop(), Nullable);
          if (cacheParamCasts) {
            if (rootParams[i] != p && origParams.size() > 1) {
              std::vector<Type> singleRoot = origParams;
              singleRoot[i] = rootParams[i];
              addParamChoice(Type(singleRoot));
              std::vector<Type> rootKeepSingle = rootParams;
              rootKeepSingle[i] = p;
              addParamChoice(Type(rootKeepSingle));
            }
            if (objRootOtherTopParams[i] != p && origParams.size() > 1) {
              std::vector<Type> objRootKeepSingle = objRootOtherTopParams;
              objRootKeepSingle[i] = p;
              addParamChoice(Type(objRootKeepSingle));
            }
            if (tType != objRootOtherTopParams[i]) {
              std::vector<Type> objRootSingleTop = objRootOtherTopParams;
              objRootSingleTop[i] = tType;
              addParamChoice(Type(objRootSingleTop));
            }
            if (auto superHT = p.getHeapType().getDeclaredSuperType()) {
              Type sType(*superHT, Nullable);
              if (sType != p && sType != rootParams[i]) {
                std::vector<Type> singleSuper = origParams;
                singleSuper[i] = sType;
                addParamChoice(Type(singleSuper));
              }
            }
          }
          if (tType != p) {
            std::vector<Type> singleTop = origParams;
            singleTop[i] = tType;
            addParamChoice(Type(singleTop));
          }
          if (tType != rootParams[i]) {
            std::vector<Type> rootSingleTop = rootParams;
            rootSingleTop[i] = tType;
            addParamChoice(Type(rootSingleTop));
          }
        }
      }

      for (Type pt : paramChoices) {
        for (Type rt : resChoices) {
          Signature cand(pt, rt);
          if (std::find(out.begin(), out.end(), cand) == out.end()) {
            out.push_back(cand);
          }
        }
      }
    };

    Builder builder(*module);
    std::unordered_set<Function*> modifiedFuncs;
    std::unordered_set<Function*> generalizedFuncs;
    std::unordered_set<HeapType> usedTargetHTs;
    std::unordered_map<Expression*, Type> strippedCastDemands;
    GlobalTypeRewriter::SignatureUpdates genUpdates;
    std::vector<std::pair<Function*, Expression*>> pendingPrologueSets;

    auto getEffectiveType = [&](auto& self, Expression* expr) -> Type {
      while (auto* block = expr->dynCast<Block>()) {
        if (!block->name.is() && !block->list.empty()) {
          expr = block->list.back();
        } else {
          break;
        }
      }
      if (auto* as = expr->dynCast<RefAs>()) {
        if (as->op == RefAsNonNull) {
          Type innerT = self(self, as->value);
          return innerT.isRef() ? innerT.with(NonNullable) : innerT;
        }
      }
      if (auto* iff = expr->dynCast<If>()) {
        if (iff->ifFalse && iff->type.isRef()) {
          return Type::getLeastUpperBound(self(self, iff->ifTrue),
                                          self(self, iff->ifFalse));
        }
      }
      if (auto* sel = expr->dynCast<Select>()) {
        if (sel->type.isRef()) {
          return Type::getLeastUpperBound(self(self, sel->ifTrue),
                                          self(self, sel->ifFalse));
        }
      }
      return expr->type;
    };

    auto recordStrippedCastDemand =
      [&](auto& self, Expression* expr, Type reqType) -> void {
      while (auto* block = expr->dynCast<Block>()) {
        if (!block->name.is() && !block->list.empty()) {
          expr = block->list.back();
        } else {
          break;
        }
      }
      if (auto* as = expr->dynCast<RefAs>()) {
        if (as->op == RefAsNonNull) {
          self(self, as->value, reqType.with(Nullable));
          return;
        }
      }
      if (auto* iff = expr->dynCast<If>()) {
        if (iff->ifFalse && iff->type.isRef()) {
          Nullability armNull =
            reqType.isNullable() ? Nullable : iff->type.getNullability();
          Type armReq(reqType.getHeapType(), armNull);
          if (iff->ifTrue->type.isRef()) {
            self(self, iff->ifTrue, armReq);
          }
          if (iff->ifFalse->type.isRef()) {
            self(self, iff->ifFalse, armReq);
          }
          return;
        }
      }
      if (auto* sel = expr->dynCast<Select>()) {
        if (sel->type.isRef()) {
          Nullability armNull =
            reqType.isNullable() ? Nullable : sel->type.getNullability();
          Type armReq(reqType.getHeapType(), armNull);
          if (sel->ifTrue->type.isRef()) {
            self(self, sel->ifTrue, armReq);
          }
          if (sel->ifFalse->type.isRef()) {
            self(self, sel->ifFalse, armReq);
          }
          return;
        }
      }
      auto sdIt = strippedCastDemands.find(expr);
      if (sdIt == strippedCastDemands.end()) {
        strippedCastDemands[expr] = reqType;
      } else {
        Nullability combNull =
          (reqType.isNonNullable() || sdIt->second.isNonNullable())
            ? NonNullable
            : Nullable;
        HeapType combHT = sdIt->second.getHeapType();
        if (HeapType::isSubType(reqType.getHeapType(), combHT)) {
          combHT = reqType.getHeapType();
        } else if (!HeapType::isSubType(combHT, reqType.getHeapType())) {
          combHT = expr->type.getHeapType();
        }
        strippedCastDemands[expr] = Type(combHT, combNull);
      }
    };

    struct CandEval {
      Signature candSig;
      int origDefBytes;
      int candDefBytes;
      int castsSavedBytes;
      int castsAddedBytes;
      int memberNet;
    };

    auto evalGroupCandidates = [&](HeapType ht,
                                   const std::vector<Function*>& flist,
                                   bool canMutateHTInPlace,
                                   std::vector<CandEval>& outEvals) {
      Signature origSig = ht.getSignature();
      if (origSig.params.isTuple() && origSig.params.size() > 16) {
        return;
      }
      if (origSig.results.isTuple()) {
        return;
      }

      std::vector<Type> origParams(origSig.params.begin(),
                                   origSig.params.end());
      Type origRes = origSig.results;
      int sigDefBytes = getSigDefBytes(origSig);

      std::vector<Signature> candidates;
      addCanonCandidates(origSig, flist, candidates);
      {
        std::vector<Type> selLubParams = origParams;
        std::vector<Type> selRootParams = origParams;
        bool anySel = false;
        for (size_t i = 0; i < origParams.size(); ++i) {
          Type p = origParams[i];
          if (!p.isRef() || p.isExact()) {
            continue;
          }
          HeapType lubHT = p.getHeapType();
          int castCallers = 0;
          bool validLub = true;
          for (auto* f : flist) {
            auto callIt = allCallsTo.find(f->name);
            if (callIt == allCallsTo.end()) {
              continue;
            }
            for (auto& [caller, call] : callIt->second) {
              if (i >= call->operands.size()) {
                continue;
              }
              Expression* arg = call->operands[i];
              if (auto* cast = arg->dynCast<RefCast>()) {
                Type effCastRefT = getEffectiveType(getEffectiveType, cast->ref);
                if (!cast->desc && effCastRefT.isRef() &&
                    !effCastRefT.isNull() &&
                    effCastRefT.getHeapType().getTop() == lubHT.getTop()) {
                  if (auto nextLub = HeapType::getLeastUpperBound(
                        lubHT, effCastRefT.getHeapType())) {
                    lubHT = *nextLub;
                    castCallers++;
                    continue;
                  }
                }
              }
              Type effArgT = getEffectiveType(getEffectiveType, arg);
              if (effArgT.isRef() && !effArgT.isNull()) {
                if (auto nextLub = HeapType::getLeastUpperBound(
                      lubHT, effArgT.getHeapType())) {
                  lubHT = *nextLub;
                } else {
                  validLub = false;
                }
              }
            }
          }
          if (castCallers > 0 && validLub) {
            selLubParams[i] = Type(lubHT, Nullable);
            selRootParams[i] = Type(getRootAncestor(p.getHeapType()), Nullable);
            anySel = true;
            if (origParams.size() > 1) {
              std::vector<Type> singleLub = origParams;
              singleLub[i] = selLubParams[i];
              candidates.push_back(Signature(Type(singleLub), origRes));
              std::vector<Type> singleRoot = origParams;
              singleRoot[i] = selRootParams[i];
              if (singleRoot != singleLub) {
                candidates.push_back(Signature(Type(singleRoot), origRes));
              }
            }
          }
        }
        if (anySel) {
          candidates.push_back(Signature(Type(selLubParams), origRes));
          if (selRootParams != selLubParams) {
            candidates.push_back(Signature(Type(selRootParams), origRes));
          }
        }
      }
      {
        // Basic-type (`any` / `eq` / `func`) zero-cast generalization candidate.
        std::vector<Type> basicParams = origParams;
        for (size_t i = 0; i < basicParams.size(); ++i) {
          Type p = basicParams[i];
          if (!p.isRef() || p.isExact() ||
              p.getHeapType() == p.getHeapType().getTop()) {
            continue;
          }
          HeapType topHT = p.getHeapType().getTop();
          HeapType eqHT =
            HeapTypes::eq.getBasic(p.getHeapType().getShared());
          bool canTop = true;
          bool canEq = HeapType::isSubType(p.getHeapType(), eqHT);
          for (auto* f : flist) {
            auto& sum = summaries.map[f];
            if (sum.localWriteCounts[i] > 0) {
              canTop = false;
              canEq = false;
              break;
            }
            for (auto& site : sum.paramSites[i]) {
              auto sdIt = strippedCastDemands.find(site.get);
              if (sdIt != strippedCastDemands.end()) {
                if (!Type::isSubType(Type(topHT, Nullable), sdIt->second)) {
                  canTop = false;
                }
                if (!canEq ||
                    !Type::isSubType(Type(eqHT, Nullable), sdIt->second)) {
                  canEq = false;
                }
                continue;
              }
              if (!site.isFixedConsumer || site.isSelfReturn) {
                canTop = false;
                canEq = false;
                break;
              }
              if (!HeapType::isSubType(topHT, site.acceptedHT)) {
                canTop = false;
              }
              if (!canEq || !HeapType::isSubType(eqHT, site.acceptedHT)) {
                canEq = false;
              }
            }
          }
          if (canTop) {
            basicParams[i] = Type(topHT, Nullable);
          } else if (canEq) {
            basicParams[i] = Type(eqHT, Nullable);
          }
        }
        Type basicRes = origRes;
        if (origRes.isRef() && !origRes.isExact() &&
            origRes.getHeapType() != origRes.getHeapType().getTop()) {
          HeapType topHT = origRes.getHeapType().getTop();
          HeapType eqHT =
            HeapTypes::eq.getBasic(origRes.getHeapType().getShared());
          bool canTop = true;
          bool canEq = HeapType::isSubType(origRes.getHeapType(), eqHT);
          for (auto* f : flist) {
            auto it = callsTo.find(f->name);
            if (it == callsTo.end() || it->second.empty()) {
              canTop = false;
              canEq = false;
              break;
            }
            for (auto& csite : it->second) {
              auto sdIt = strippedCastDemands.find(csite.call);
              if (sdIt != strippedCastDemands.end()) {
                if (!Type::isSubType(Type(topHT, Nullable), sdIt->second)) {
                  canTop = false;
                }
                if (!canEq ||
                    !Type::isSubType(Type(eqHT, Nullable), sdIt->second)) {
                  canEq = false;
                }
                continue;
              }
              if (!csite.isFixedConsumer ||
                  csite.acceptedNull == NonNullable) {
                canTop = false;
                canEq = false;
                break;
              }
              if (!HeapType::isSubType(topHT, csite.acceptedHT)) {
                canTop = false;
              }
              if (!canEq || !HeapType::isSubType(eqHT, csite.acceptedHT)) {
                canEq = false;
              }
            }
          }
          if (canTop) {
            basicRes = Type(topHT, Nullable);
          } else if (canEq) {
            basicRes = Type(eqHT, Nullable);
          }
        }
        Signature basicSig(Type(basicParams), basicRes);
        if (basicSig != origSig) {
          candidates.push_back(basicSig);
        }
        if (basicRes != origRes) {
          candidates.push_back(Signature(origSig.params, basicRes));
        }
        if (Type(basicParams) != origSig.params) {
          candidates.push_back(Signature(Type(basicParams), origRes));
        }
      }

      std::unordered_set<Signature> seenCands;
      for (const auto& candSig : candidates) {
        if (candSig == origSig || !seenCands.insert(candSig).second) {
          continue;
        }
        bool alreadyExists = sigToHT.count(candSig) != 0;
        if (!canMutateHTInPlace && !alreadyExists) {
          continue;
        }

        std::vector<Type> candParams(candSig.params.begin(),
                                     candSig.params.end());
        Type candRes = candSig.results;

        int castsAddedBytes = 0;
        int castsSavedBytes = 0;
        bool possible = true;

        int totalNarrowGets = 0;
        int totalNarrowCalls = 0;
        for (auto* f : flist) {
          auto& sum = summaries.map[f];
          for (size_t i = 0; i < origParams.size(); ++i) {
            if (candParams[i] == origParams[i]) {
              continue;
            }
            if (!Type::isSubType(origParams[i], candParams[i])) {
              possible = false;
              break;
            }
            if (sum.localWriteCounts[i] > 0) {
              if (!cacheParamCasts || !origParams[i].isNullable()) {
                possible = false;
                break;
              }
              if (!sum.paramSites[i].empty()) {
                castsAddedBytes += 8;
                totalNarrowGets = std::max(totalNarrowGets, 1);
              }
            } else {
              int narrowGets = 0;
              const ConsumerSite* firstNarrow = nullptr;
              const ConsumerSite* lastNarrow = nullptr;
              for (auto& site : sum.paramSites[i]) {
                auto sdIt = strippedCastDemands.find(site.get);
                if (sdIt != strippedCastDemands.end()) {
                  if (!Type::isSubType(candParams[i], sdIt->second)) {
                    possible = false;
                    break;
                  }
                  continue;
                }
                Type accType =
                  site.isSelfReturn
                    ? candRes
                    : Type(site.acceptedHT, site.acceptedNull);
                if (!site.isFixedConsumer ||
                    !Type::isSubType(candParams[i], accType)) {
                  narrowGets++;
                  if (!firstNarrow) {
                    firstNarrow = &site;
                  }
                  lastNarrow = &site;
                }
              }
              bool canCacheInLocal =
                cacheParamCasts && narrowGets >= 2 && origParams[i].isNullable();
              if (!possible || (!canCacheInLocal && narrowGets > 8)) {
                possible = false;
                break;
              }
              totalNarrowGets =
                std::max(totalNarrowGets, canCacheInLocal ? 1 : narrowGets);
              if (canCacheInLocal) {
                bool canTee =
                  firstNarrow && lastNarrow &&
                  lastNarrow->visitOrder <= firstNarrow->domEndOrder;
                castsAddedBytes += canTee ? 6 : 7;
              } else {
                castsAddedBytes += narrowGets * 3;
              }
            }
            auto callIt = allCallsTo.find(f->name);
            if (callIt != allCallsTo.end()) {
              for (auto& [caller, call] : callIt->second) {
                if (i < call->operands.size()) {
                  Expression* arg = call->operands[i];
                  if (auto* cast = arg->dynCast<RefCast>()) {
                    Type effCastRefT =
                      getEffectiveType(getEffectiveType, cast->ref);
                    if (!cast->desc && effCastRefT.isRef() &&
                        !effCastRefT.isNull() &&
                        Type::isSubType(effCastRefT, candParams[i])) {
                      castsSavedBytes += 3;
                    }
                  }
                }
              }
            }
          }
          if (!possible) {
            break;
          }
          if (candRes != origRes) {
            auto callIt = callsTo.find(f->name);
            if (callIt == callsTo.end() || callIt->second.empty() ||
                !Type::isSubType(origRes, candRes)) {
              possible = false;
              break;
            }
            int narrowCalls = 0;
            int callCastBytes =
              (candRes.getHeapType() != origRes.getHeapType()) ? 3 : 1;
            for (auto& csite : callIt->second) {
              auto sdIt = strippedCastDemands.find(csite.call);
              if (sdIt != strippedCastDemands.end()) {
                if (!Type::isSubType(candRes, sdIt->second)) {
                  possible = false;
                  break;
                }
                continue;
              }
              Type accType(csite.acceptedHT, csite.acceptedNull);
              if (!csite.isFixedConsumer ||
                  !Type::isSubType(candRes, accType)) {
                narrowCalls++;
                castsAddedBytes += callCastBytes;
              }
            }
            if (!possible || narrowCalls > 8) {
              possible = false;
              break;
            }
            totalNarrowCalls = std::max(totalNarrowCalls, narrowCalls);
            for (auto* tailp : sum.returnValueLocs) {
              Expression* tail = *tailp;
              if (auto* as = tail->dynCast<RefAs>()) {
                if (as->op == RefAsNonNull && candRes.isNullable()) {
                  castsSavedBytes += 1;
                  tail = as->value;
                }
              }
              if (auto* cast = tail->dynCast<RefCast>()) {
                Expression* refInner = cast->ref;
                if (auto* as2 = refInner->dynCast<RefAs>()) {
                  if (as2->op == RefAsNonNull) {
                    refInner = as2->value;
                  }
                }
                Type effRefType = getEffectiveType(getEffectiveType, cast->ref);
                if (auto* lg = refInner->dynCast<LocalGet>()) {
                  if (lg->index < candParams.size() &&
                      sum.localWriteCounts[lg->index] == 0 &&
                      candParams[lg->index] != origParams[lg->index]) {
                    effRefType = (refInner != cast->ref)
                                   ? candParams[lg->index].with(NonNullable)
                                   : candParams[lg->index];
                  }
                }
                if (!cast->desc && effRefType.isRef() &&
                    !effRefType.isNull() &&
                    HeapType::isSubType(effRefType.getHeapType(),
                                        candRes.getHeapType())) {
                  castsSavedBytes +=
                    (candRes.isNonNullable() && effRefType.isNullable())
                      ? 2
                      : 3;
                }
              }
            }
          }
        }

        if (!possible) {
          continue;
        }
        if (widenFields && canMutateHTInPlace && candRes != origRes &&
            candRes.isNullable()) {
          auto cIt = coupledArraysForSig.find(ht);
          if (cIt != coupledArraysForSig.end()) {
            for (HeapType arrHT : cIt->second) {
              auto mIt = existingMutArrayHT.find(candRes);
              if (mIt != existingMutArrayHT.end() && mIt->second != arrHT) {
                int numGlobals = arrGlobalCount[arrHT];
                int numFuncs = arrFuncCount[arrHT];
                int arrBonus = 5 + 2 * numFuncs;
                if (primaryStructRoot &&
                    candRes == Type(*primaryStructRoot, Nullable)) {
                  arrBonus += 4 * numGlobals + 4 * numFuncs;
                } else if (!candRes.getHeapType().isBasic()) {
                  arrBonus += 2 * numGlobals + 2 * numFuncs;
                } else {
                  arrBonus += 2 * numGlobals;
                }
                for (HeapType innerHT : arrCastInnerHTs[arrHT]) {
                  if (HeapType::isSubType(innerHT, candRes.getHeapType())) {
                    arrBonus += 3;
                  }
                }
                castsSavedBytes += arrBonus;
              }
            }
          }
        }
        int candDefBytes = getSigDefBytes(candSig);
        if (widenFields && canMutateHTInPlace && castsAddedBytes == 0 &&
            castsSavedBytes == 0) {
          bool isBasicSubtypeWiden = true;
          for (size_t i = 0; i < origParams.size(); ++i) {
            if (candParams[i] != origParams[i] &&
                !origParams[i].getHeapType().isBasic()) {
              isBasicSubtypeWiden = false;
              break;
            }
          }
          if (candRes != origRes && !origRes.getHeapType().isBasic() &&
              !(widenFields && candParams == origParams &&
                !origRes.getHeapType().getDeclaredSuperType() &&
                !origRes.getHeapType().isOpen() &&
                subTypes.getImmediateSubTypes(origRes.getHeapType()).empty())) {
            isBasicSubtypeWiden = false;
          }
          if (isBasicSubtypeWiden) {
            castsSavedBytes = 1;
          }
        }
        if (castsSavedBytes <= castsAddedBytes && !mergeSignatures &&
            (totalNarrowGets > 1 || totalNarrowCalls > 1 ||
             castsAddedBytes > castsSavedBytes + (alreadyExists ? 4 : 0))) {
          continue;
        }
        int memberNet = (canMutateHTInPlace ? sigDefBytes : 0) +
                        castsSavedBytes - castsAddedBytes;
        outEvals.push_back({candSig,
                            sigDefBytes,
                            candDefBytes,
                            castsSavedBytes,
                            castsAddedBytes,
                            memberNet});
      }
    };

    // Group-level new-signature cohort verification across all eligible
    // whole-HeapType groups: each group contributes only to its single selected
    // candidate, and a new signature (!sigToHT.count(candSig)) is approved only
    // when >= 2 groups select it and sum(memberNet) - candDefBytes > 0.
    std::vector<HeapType> eligibleHTs;
    std::unordered_map<HeapType, std::vector<CandEval>> preEvals;
    std::unordered_map<Signature, int> activeNewSigCounts;
    for (auto ht : htOrder) {
      auto& flist = htFuncs[ht];
      if (pinnedHTs.count(ht) || flist.empty()) {
        continue;
      }
      Signature origSig = ht.getSignature();
      if (sigFuncCount[origSig] != flist.size() || origSig.results.isTuple() ||
          (origSig.params.isTuple() && origSig.params.size() > 16)) {
        continue;
      }
      auto& evals = preEvals[ht];
      evalGroupCandidates(ht, flist, true, evals);
      if (!evals.empty()) {
        eligibleHTs.push_back(ht);
        for (const auto& ce : evals) {
          if (!sigToHT.count(ce.candSig) &&
              (ce.memberNet > 0 || ce.castsSavedBytes > ce.castsAddedBytes)) {
            activeNewSigCounts[ce.candSig]++;
          }
        }
      }
    }
    for (auto it = activeNewSigCounts.begin();
         it != activeNewSigCounts.end();) {
      if (it->second < 2) {
        it = activeNewSigCounts.erase(it);
      } else {
        ++it;
      }
    }

    std::unordered_map<Signature, std::vector<HeapType>> newSigCohorts;
    std::unordered_map<HeapType, Signature> assignedNewSig;
    while (true) {
      newSigCohorts.clear();
      assignedNewSig.clear();
      std::unordered_map<Signature, int> newSigSumMemberNet;
      std::unordered_map<Signature, int> newSigDefCost;

      for (auto ht : eligibleHTs) {
        Signature bestCand = ht.getSignature();
        int bestScore = 0;
        int bestMemberNet = 0;
        for (const auto& ce : preEvals[ht]) {
          bool alreadyExists = sigToHT.count(ce.candSig) != 0;
          int score = 0;
          if (alreadyExists) {
            if (ce.memberNet > 0) {
              score = ce.memberNet * 2 + 1;
            }
          } else {
            auto cntIt = activeNewSigCounts.find(ce.candSig);
            if (cntIt != activeNewSigCounts.end() && ce.memberNet > 0) {
              int k = std::max(2, cntIt->second);
              score = std::max(1, ce.memberNet * 2 - (ce.candDefBytes * 2) / k);
            } else if (ce.castsSavedBytes > ce.castsAddedBytes) {
              score = (ce.castsSavedBytes - ce.castsAddedBytes) * 2;
            }
          }
          if (score > bestScore) {
            bestScore = score;
            bestCand = ce.candSig;
            bestMemberNet = ce.memberNet;
          }
        }
        if (bestScore > 0 && !sigToHT.count(bestCand) &&
            activeNewSigCounts.count(bestCand)) {
          newSigCohorts[bestCand].push_back(ht);
          assignedNewSig[ht] = bestCand;
          newSigSumMemberNet[bestCand] += bestMemberNet;
          newSigDefCost[bestCand] = getSigDefBytes(bestCand);
        }
      }

      bool changed = false;
      std::unordered_map<Signature, int> nextCounts;
      for (const auto& [sig, prevCnt] : activeNewSigCounts) {
        auto cIt = newSigCohorts.find(sig);
        if (cIt != newSigCohorts.end() && cIt->second.size() >= 2 &&
            newSigSumMemberNet[sig] - newSigDefCost[sig] > 0) {
          nextCounts[sig] = int(cIt->second.size());
          if (nextCounts[sig] != prevCnt) {
            changed = true;
          }
        } else {
          changed = true;
        }
      }
      activeNewSigCounts = std::move(nextCounts);
      if (!changed) {
        break;
      }
    }

    auto tryGeneralizeGroup = [&](HeapType ht,
                                  const std::vector<Function*>& flist,
                                  bool canMutateHTInPlace) {
      if (usedTargetHTs.count(ht)) {
        canMutateHTInPlace = false;
      }
      Signature origSig = ht.getSignature();
      std::vector<CandEval> evals;
      evalGroupCandidates(ht, flist, canMutateHTInPlace, evals);
      if (evals.empty()) {
        return;
      }

      Signature bestSig = origSig;
      int bestScore = 0;
      auto assignedIt = assignedNewSig.find(ht);

      for (const auto& ce : evals) {
        bool alreadyExists = sigToHT.count(ce.candSig) != 0;
        int score = 0;
        if (alreadyExists) {
          int net = (canMutateHTInPlace ? ce.origDefBytes : 0) +
                    ce.castsSavedBytes - ce.castsAddedBytes;
          if (net > 0) {
            score = net * 2 +
                    ((assignedIt != assignedNewSig.end() &&
                      assignedIt->second == ce.candSig)
                       ? 2
                       : 1);
          }
        } else if (canMutateHTInPlace) {
          auto cIt = newSigCohorts.find(ce.candSig);
          bool hasActiveCohort = false;
          int k = 2;
          if (cIt != newSigCohorts.end() && ce.memberNet > 0) {
            int ungenCount = 0;
            for (auto otherHT : cIt->second) {
              if (!usedTargetHTs.count(otherHT) &&
                  !htFuncs[otherHT].empty() &&
                  !generalizedFuncs.count(htFuncs[otherHT][0])) {
                ungenCount++;
              }
            }
            if (ungenCount >= 2) {
              hasActiveCohort = true;
              k = std::max(2, int(cIt->second.size()));
            }
          }
          if (hasActiveCohort) {
            score =
              std::max(1, ce.memberNet * 2 - (ce.candDefBytes * 2) / k) +
              ((assignedIt != assignedNewSig.end() &&
                assignedIt->second == ce.candSig)
                 ? 1
                 : 0);
          } else if (ce.castsSavedBytes > ce.castsAddedBytes) {
            score = (ce.castsSavedBytes - ce.castsAddedBytes) * 2;
          }
        }
        if (score > bestScore) {
          bestScore = score;
          bestSig = ce.candSig;
        }
      }

      if (bestSig == origSig || bestScore <= 0) {
        return;
      }

      std::vector<Type> origParams(origSig.params.begin(),
                                   origSig.params.end());
      Type origRes = origSig.results;

      HeapType targetHT = ht;
      auto sigIt = sigToHT.find(bestSig);
      if (sigIt != sigToHT.end()) {
        targetHT = sigIt->second;
        usedTargetHTs.insert(targetHT);
      } else {
        targetHT = ht;
        sigToHT.erase(origSig);
        sigToHT.emplace(bestSig, ht);
        usedTargetHTs.insert(ht);
        genUpdates[ht] = bestSig;
      }
      existingSigs.insert(bestSig);

      std::vector<Type> targetParams(bestSig.params.begin(),
                                     bestSig.params.end());
      Type targetRes = bestSig.results;

      // Step A: Update all widened parameter gets and call result sites first
      // so `getEffectiveType` sees the updated types before any caller/return
      // casts are stripped in Step B.
      for (auto* f : flist) {
        modifiedFuncs.insert(f);
        generalizedFuncs.insert(f);
        auto& sum = summaries.map[f];
        for (size_t i = 0; i < origParams.size(); ++i) {
          if (targetParams[i] == origParams[i]) {
            continue;
          }
          if (sum.localWriteCounts[i] > 0) {
            Index scratchLocal = Builder::addVar(f, origParams[i]);
            struct LocalRewriter : public PostWalker<LocalRewriter> {
              Index oldIdx;
              Index newIdx;
              LocalRewriter(Index oldIdx, Index newIdx)
                : oldIdx(oldIdx), newIdx(newIdx) {}
              void visitLocalGet(LocalGet* curr) {
                if (curr->index == oldIdx) {
                  curr->index = newIdx;
                }
              }
              void visitLocalSet(LocalSet* curr) {
                if (curr->index == oldIdx) {
                  curr->index = newIdx;
                }
              }
            } rewriter(i, scratchLocal);
            rewriter.walk(f->body);
            if (!sum.paramSites[i].empty()) {
              Expression* castExpr = builder.makeRefCast(
                builder.makeLocalGet(i, targetParams[i]), origParams[i]);
              pendingPrologueSets.push_back(
                {f, builder.makeLocalSet(scratchLocal, castExpr)});
            }
            continue;
          }
          std::vector<ConsumerSite*> narrowSites;
          for (auto& site : sum.paramSites[i]) {
            site.get->type = targetParams[i];
            if (strippedCastDemands.count(site.get)) {
              if (site.isWrappedInNonNull) {
                (*site.loc)->type = targetParams[i].with(NonNullable);
              }
              continue;
            }
            Type accType =
              site.isSelfReturn
                ? targetRes
                : Type(site.acceptedHT, site.acceptedNull);
            if (site.isFixedConsumer &&
                Type::isSubType(targetParams[i], accType)) {
              if (site.isWrappedInNonNull) {
                (*site.loc)->type = targetParams[i].with(NonNullable);
              }
            } else {
              narrowSites.push_back(&site);
            }
          }
          if (narrowSites.size() == 1 ||
              (narrowSites.size() >= 2 &&
               (!cacheParamCasts || !origParams[i].isNullable()))) {
            for (auto* sitePtr : narrowSites) {
              auto& site = *sitePtr;
              Type castTarget = site.isWrappedInNonNull
                                  ? origParams[i].with(NonNullable)
                                  : origParams[i];
              *site.loc = builder.makeRefCast(site.get, castTarget);
            }
          } else if (narrowSites.size() >= 2) {
            Index scratchLocal = Builder::addVar(f, origParams[i]);
            bool canTee =
              narrowSites.back()->visitOrder <= narrowSites.front()->domEndOrder;
            size_t startIdx = 0;
            if (canTee) {
              auto& first = *narrowSites[0];
              Expression* castExpr =
                builder.makeRefCast(first.get, origParams[i]);
              Expression* teeExpr =
                builder.makeLocalTee(scratchLocal, castExpr, origParams[i]);
              if (first.isWrappedInNonNull) {
                (*first.loc)->cast<RefAs>()->value = teeExpr;
              } else {
                *first.loc = teeExpr;
              }
              startIdx = 1;
            } else {
              Expression* castExpr = builder.makeRefCast(
                builder.makeLocalGet(i, targetParams[i]), origParams[i]);
              pendingPrologueSets.push_back(
                {f, builder.makeLocalSet(scratchLocal, castExpr)});
            }
            for (size_t k = startIdx; k < narrowSites.size(); ++k) {
              auto& site = *narrowSites[k];
              Expression* getScratch =
                builder.makeLocalGet(scratchLocal, origParams[i]);
              if (site.isWrappedInNonNull) {
                (*site.loc)->cast<RefAs>()->value = getScratch;
              } else {
                *site.loc = getScratch;
              }
            }
          }
        }

        if (targetRes != origRes) {
          auto callIt = callsTo.find(f->name);
          if (callIt != callsTo.end()) {
            for (auto& csite : callIt->second) {
              modifiedFuncs.insert(csite.caller);
              if (csite.call->type != Type::unreachable) {
                csite.call->type = targetRes;
              }
              if (strippedCastDemands.count(csite.call)) {
                if (csite.isWrappedInNonNull) {
                  (*csite.loc)->type = targetRes.with(NonNullable);
                }
                continue;
              }
              Type accType(csite.acceptedHT, csite.acceptedNull);
              if (csite.isFixedConsumer &&
                  Type::isSubType(targetRes, accType)) {
                if (csite.isWrappedInNonNull) {
                  (*csite.loc)->type = targetRes.with(NonNullable);
                }
              } else {
                if (targetRes.getHeapType() != origRes.getHeapType()) {
                  Type castTarget = csite.isWrappedInNonNull
                                      ? origRes.with(NonNullable)
                                      : origRes;
                  *csite.loc = builder.makeRefCast(csite.call, castTarget);
                } else {
                  *csite.loc = builder.makeRefAs(RefAsNonNull, csite.call);
                }
              }
            }
          }
        }
      }

      // Step B: Strip redundant casts on caller arguments and return values,
      // and propagate stripped cast demands through `Block`/`If`/`Select`.
      for (auto* f : flist) {
        auto& sum = summaries.map[f];
        for (size_t i = 0; i < origParams.size(); ++i) {
          if (targetParams[i] == origParams[i]) {
            continue;
          }
          auto callIt = allCallsTo.find(f->name);
          if (callIt != allCallsTo.end()) {
            for (auto& [caller, call] : callIt->second) {
              if (i < call->operands.size()) {
                Expression* arg = call->operands[i];
                if (auto* cast = arg->dynCast<RefCast>()) {
                  Type effCastRefT =
                    getEffectiveType(getEffectiveType, cast->ref);
                  if (!cast->desc && effCastRefT.isRef() &&
                      !effCastRefT.isNull() &&
                      Type::isSubType(effCastRefT, targetParams[i])) {
                    call->operands[i] = cast->ref;
                    modifiedFuncs.insert(caller);
                    recordStrippedCastDemand(
                      recordStrippedCastDemand, cast->ref, targetParams[i]);
                  }
                }
              }
            }
          }
        }

        if (targetRes != origRes) {
          for (auto* tailp : sum.returnValueLocs) {
            bool stripped = false;
            if (auto* as = (*tailp)->dynCast<RefAs>()) {
              if (as->op == RefAsNonNull && targetRes.isNullable()) {
                *tailp = as->value;
                stripped = true;
              }
            }
            if (auto* cast = (*tailp)->dynCast<RefCast>()) {
              Type effCastRefT =
                getEffectiveType(getEffectiveType, cast->ref);
              if (!cast->desc && effCastRefT.isRef() &&
                  !effCastRefT.isNull() &&
                  HeapType::isSubType(effCastRefT.getHeapType(),
                                      targetRes.getHeapType())) {
                *tailp = cast->ref;
                if (targetRes.isNonNullable() && effCastRefT.isNullable()) {
                  *tailp = builder.makeRefAs(RefAsNonNull, *tailp);
                }
                stripped = true;
              }
            }
            if (stripped) {
              recordStrippedCastDemand(
                recordStrippedCastDemand, *tailp, targetRes);
            }
          }
        }

        f->type = f->type.with(targetHT);
      }
    };

    // First pass: whole-HeapType groups where `ht` is unpinned and not shared
    // with escaped functions.
    for (auto ht : htOrder) {
      auto& flist = htFuncs[ht];
      if (pinnedHTs.count(ht) || flist.empty()) {
        continue;
      }
      Signature origSig = ht.getSignature();
      if (sigFuncCount[origSig] != flist.size()) {
        continue;
      }
      tryGeneralizeGroup(ht, flist, true);
    }

    // Second pass: per-function fallback for any remaining non-escaped function
    // where switching to an already-existing signature saves casts in the code
    // section.
    for (auto ht : htOrder) {
      auto& flist = htFuncs[ht];
      for (auto* f : flist) {
        if (!generalizedFuncs.count(f)) {
          tryGeneralizeGroup(ht, {f}, false);
        }
      }
    }

    for (auto& [f, prologueSet] : pendingPrologueSets) {
      f->body = builder.makeSequence(prologueSet, f->body);
    }

    GlobalTypeRewriter::updateSignatures(
      genUpdates, *module, getPassOptions().worldMode);

    for (auto& fn : module->functions) {
      if (modifiedFuncs.count(fn.get())) {
        ReFinalize().walkFunctionInModule(fn.get(), module);
      }
    }
  }

  bool widenFieldsAndArrays(Module* module,
                            bool allowSiblingCastWiden = false) {
    std::unordered_set<HeapType> fixed;
    for (auto type :
         ModuleUtils::getPublicHeapTypes(*module, getPassOptions().worldMode)) {
      fixed.insert(type);
    }
    for (auto& tag : module->tags) {
      fixed.insert(tag->type);
      for (Type t : tag->params()) {
        if (t.isRef()) {
          fixed.insert(t.getHeapType());
        }
      }
      for (Type t : tag->results()) {
        if (t.isRef()) {
          fixed.insert(t.getHeapType());
        }
      }
    }
    for (auto func : Intrinsics(*module).getJSCalledFunctions()) {
      fixed.insert(module->getFunction(func)->type.getHeapType());
    }

    std::unordered_set<Name> exportedGlobals;
    for (auto& exp : module->exports) {
      if (exp->kind == ExternalKind::Global) {
        exportedGlobals.insert(*exp->getInternalName());
      }
    }

    auto allTypes = ModuleUtils::collectHeapTypes(*module);
    SubTypes subTypes(allTypes);

    std::unordered_map<FieldKey, FieldDemandState, FieldKeyHash>
      fieldCandidates;
    std::unordered_set<FieldKey, FieldKeyHash> nullRelaxableFields;
    bool hasImmCandidate = false;
    for (auto A : allTypes) {
      if (!A.isStruct()) {
        continue;
      }
      const auto& aFields = A.getStruct().fields;
      for (Index fi = 0; fi < aFields.size(); ++fi) {
        if (aFields[fi].mutable_ == Immutable && !aFields[fi].isPacked() &&
            aFields[fi].type.isRef() && !aFields[fi].type.isExact() &&
            !fixed.count(A) && !A.isShared() && !A.getDescriptorType() &&
            !A.getDescribedType()) {
          hasImmCandidate = true;
          if (!aFields[fi].type.isNullable() && !aFields[fi].type.isNull()) {
            nullRelaxableFields.insert(FieldKey{A, fi});
          }
          continue;
        }
        if (aFields[fi].mutable_ != Mutable ||
            getHighestFieldAncestor(A, fi) != A) {
          continue;
        }
        const auto& family = subTypes.getSubTypes(A);
        bool valid = true;
        HeapType baseHT = aFields[fi].type.isRef()
                            ? aFields[fi].type.getHeapType()
                            : HeapType(HeapType::any);
        for (auto U : family) {
          if (fixed.count(U) || U.isShared() || U.getDescriptorType() ||
              U.getDescribedType()) {
            valid = false;
            break;
          }
          const auto& uFields = U.getStruct().fields;
          if (fi >= uFields.size()) {
            valid = false;
            break;
          }
          const auto& f = uFields[fi];
          if (f.isPacked() || !f.type.isRef() || f.type.isExact() ||
              f.type.isNull() || f.mutable_ != aFields[fi].mutable_) {
            valid = false;
            break;
          }
          HeapType uHT = f.type.getHeapType();
          if (!HeapType::isSubType(uHT, baseHT) ||
              uHT.getTop() != baseHT.getTop()) {
            valid = false;
            break;
          }
        }
        if (valid) {
          fieldCandidates.emplace(
            FieldKey{A, fi},
            FieldDemandState{baseHT, baseHT.getTop(), true, true, false});
          if (!aFields[fi].type.isNullable()) {
            nullRelaxableFields.insert(FieldKey{A, fi});
          }
        }
      }
    }

    std::unordered_map<HeapType, FieldDemandState> arrayCandidates;
    std::unordered_set<HeapType> nullRelaxableArrays;
    for (auto arrHT : allTypes) {
      if (!arrHT.isArray() || fixed.count(arrHT) || arrHT.isShared() ||
          arrHT.getDescriptorType() || arrHT.getDescribedType()) {
        continue;
      }
      if (arrHT.getDeclaredSuperType() ||
          !subTypes.getImmediateSubTypes(arrHT).empty()) {
        continue;
      }
      const auto& elem = arrHT.getArray().element;
      if (elem.isPacked() || !elem.type.isRef() || elem.type.isExact() ||
          elem.type.isNull()) {
        continue;
      }
      HeapType baseHT = elem.type.getHeapType();
      arrayCandidates.emplace(
        arrHT, FieldDemandState{baseHT, baseHT.getTop(), true, true, false});
      if (!elem.type.isNullable()) {
        nullRelaxableArrays.insert(arrHT);
      }
    }

    bool cacheParamCasts =
      hasArgument("relax-nullability-cache-param-casts") ||
      getArgumentOrDefault("relax-nullability", "") == "cache-param-casts" ||
      !getPassOptions().optimizeLevel;

    if (fieldCandidates.empty() && !hasImmCandidate && arrayCandidates.empty() &&
        !cacheParamCasts) {
      return false;
    }

    std::unordered_set<Name> pureGlobalCandidateSet;
    std::unordered_map<Name, FieldDemandState> pureGlobalCandidates;
    for (auto& g : module->globals) {
      if (g->imported() || g->mutable_ || exportedGlobals.count(g->name) ||
          !g->init || !g->type.isArray() || g->type.isExact()) {
        continue;
      }
      if (auto* anew = g->init->dynCast<ArrayNewFixed>()) {
        if (!anew->type.isArray()) {
          continue;
        }
        HeapType arrHT = anew->type.getHeapType();
        if (g->type.getHeapType() != arrHT || !arrayCandidates.count(arrHT) ||
            arrHT.getArray().element.mutable_ != Immutable) {
          continue;
        }
        HeapType baseHT = arrHT.getArray().element.type.getHeapType();
        pureGlobalCandidateSet.insert(g->name);
        pureGlobalCandidates.emplace(
          g->name,
          FieldDemandState{baseHT, baseHT.getTop(), true, true, false});
      } else if (auto* anew = g->init->dynCast<ArrayNew>()) {
        if (!anew->type.isArray() ||
            (!anew->isWithDefault() &&
             (!anew->init || !anew->init->type.isNull()))) {
          continue;
        }
        HeapType arrHT = anew->type.getHeapType();
        if (g->type.getHeapType() != arrHT || !arrayCandidates.count(arrHT) ||
            arrHT.getArray().element.mutable_ != Mutable ||
            !arrHT.getArray().element.type.isNullable()) {
          continue;
        }
        HeapType baseHT = arrHT.getArray().element.type.getHeapType();
        pureGlobalCandidateSet.insert(g->name);
        pureGlobalCandidates.emplace(
          g->name,
          FieldDemandState{baseHT, baseHT.getTop(), true, true, false});
      }
    }

    using StructUpdateMap =
      std::unordered_map<HeapType, std::vector<std::pair<Index, Type>>>;
    using ArrayUpdateMap = std::unordered_map<HeapType, Type>;
    StructUpdateMap structUpdates;
    ArrayUpdateMap arrayUpdates;
    std::unordered_map<Name, Index> totalGlobalGets;
    std::unordered_map<Name, Index> validArrayGlobalGets;
    std::unordered_set<Name> pureGlobalHasGet;
    std::unordered_map<FieldKey,
                       std::vector<std::pair<HeapType, bool>>,
                       FieldKeyHash>
      immFieldDemands;
    std::unordered_map<FieldKey, Type, FieldKeyHash> finalImmFieldTypes;
    std::unordered_map<HeapType, std::vector<std::pair<HeapType, bool>>>
      arrayGetDemands;
    std::unordered_set<HeapType> castStructs;
    std::unordered_set<HeapType> exactCastStructs;
    std::unordered_map<HeapType, Index> structTypeUses;
    std::unordered_map<HeapType, std::vector<StructNew*>> allStructNews;
    std::unordered_set<FieldKey, FieldKeyHash> allPinnedFields;

    while (true) {
      for (auto& [_, state] : fieldCandidates) {
        state = FieldDemandState{
          state.baseHT, state.baseHT.getTop(), true, true, false};
      }
      for (auto& [_, state] : arrayCandidates) {
        state = FieldDemandState{
          state.baseHT, state.baseHT.getTop(), true, true, false};
      }
      for (auto& [_, state] : pureGlobalCandidates) {
        state = FieldDemandState{
          state.baseHT, state.baseHT.getTop(), true, true, false};
      }
      structUpdates.clear();
      arrayUpdates.clear();
      totalGlobalGets.clear();
      validArrayGlobalGets.clear();
      pureGlobalHasGet.clear();
      immFieldDemands.clear();
      finalImmFieldTypes.clear();
      arrayGetDemands.clear();
      castStructs.clear();
      exactCastStructs.clear();
      structTypeUses.clear();
      allStructNews.clear();
      allPinnedFields.clear();

      ModuleUtils::ParallelFunctionAnalysis<FuncFieldDemandSummary> summaries(
        *module, [&](Function* func, FuncFieldDemandSummary& summary) {
          if (!func->imported()) {
            FieldAndArrayDemandWalker walker(*module,
                                             func,
                                             pureGlobalCandidateSet,
                                             nullRelaxableFields,
                                             nullRelaxableArrays,
                                             summary);
            walker.walk(func->body, walker.bodyReq, walker.bodyAcceptsNull);
          }
        });

      FuncFieldDemandSummary moduleRootSummary;
      {
        FieldAndArrayDemandWalker rootWalker(*module,
                                             nullptr,
                                             pureGlobalCandidateSet,
                                             nullRelaxableFields,
                                             nullRelaxableArrays,
                                             moduleRootSummary);
        for (auto& g : module->globals) {
          if (!g->imported() && g->init) {
            rootWalker.walk(g->init,
                            g->type.isRef()
                              ? std::optional<HeapType>(g->type.getHeapType())
                              : std::nullopt,
                            g->type.isRef() && !g->type.isExact() &&
                              g->type.isNullable());
          }
        }
        for (auto& seg : module->elementSegments) {
          for (auto* expr : seg->data) {
            rootWalker.walk(expr,
                            seg->type.isRef()
                              ? std::optional<HeapType>(seg->type.getHeapType())
                              : std::nullopt,
                            seg->type.isRef() && !seg->type.isExact() &&
                              seg->type.isNullable());
          }
        }
      }

      auto mergeSummary = [&](const FuncFieldDemandSummary& sum) {
        for (const auto& k : sum.pinnedFields) {
          allPinnedFields.insert(k);
          if (auto it = fieldCandidates.find(k); it != fieldCandidates.end()) {
            it->second.pin();
          }
        }
        for (const auto& [ht, news] : sum.structNews) {
          auto& dst = allStructNews[ht];
          dst.insert(dst.end(), news.begin(), news.end());
        }
        for (const auto& r : sum.fieldReqs) {
          if (auto it = fieldCandidates.find(r.key);
              it != fieldCandidates.end()) {
            it->second.require(r.reqHT, r.reqNull);
          }
        }
        for (const auto& r : sum.immFieldReqs) {
          immFieldDemands[r.key].push_back({r.reqHT, r.reqNull});
        }
        for (const auto& arrHT : sum.pinnedArrays) {
          if (auto it = arrayCandidates.find(arrHT);
              it != arrayCandidates.end()) {
            it->second.pin();
          }
        }
        for (const auto& r : sum.arrayReqs) {
          arrayGetDemands[r.arrHT].push_back({r.reqHT, r.reqNull});
          if (auto it = arrayCandidates.find(r.arrHT);
              it != arrayCandidates.end()) {
            it->second.require(r.reqHT, r.reqNull);
          }
        }
        for (const auto& r : sum.pureGlobalReqs) {
          pureGlobalHasGet.insert(r.name);
          HeapType gArrHT = module->getGlobal(r.name)->type.getHeapType();
          arrayGetDemands[gArrHT].push_back({r.reqHT, r.reqNull});
          if (auto it = pureGlobalCandidates.find(r.name);
              it != pureGlobalCandidates.end()) {
            it->second.require(r.reqHT, r.reqNull);
          }
        }
        for (const auto& [name, count] : sum.totalGlobalGets) {
          totalGlobalGets[name] += count;
        }
        for (const auto& [name, count] : sum.validArrayGlobalGets) {
          validArrayGlobalGets[name] += count;
        }
        for (auto ht : sum.castStructs) {
          castStructs.insert(ht);
        }
        for (auto ht : sum.exactCastStructs) {
          exactCastStructs.insert(ht);
        }
        for (const auto& [ht, count] : sum.structTypeUses) {
          structTypeUses[ht] += count;
        }
      };

      for (auto& fn : module->functions) {
        if (!fn->imported()) {
          mergeSummary(summaries.map[fn.get()]);
        }
      }
      mergeSummary(moduleRootSummary);

      for (const auto& [gName, gState] : pureGlobalCandidates) {
        HeapType arrHT = module->getGlobal(gName)->type.getHeapType();
        if (auto it = arrayCandidates.find(arrHT);
            it != arrayCandidates.end()) {
          if (pureGlobalHasGet.count(gName)) {
            it->second.mergeIn(gState);
          }
        }
      }

      for (const auto& [key, state] : fieldCandidates) {
        if (state.pinned) {
          continue;
        }
        const auto& family = subTypes.getSubTypes(key.anc);
        Type aOldFT = key.anc.getStruct().fields[key.index].type;
        bool canUnifyToAllowed =
          state.canWidenHT && (state.acceptsNull || !aOldFT.isNullable());
        for (auto U : family) {
          Type oldFT = U.getStruct().fields[key.index].type;
          Nullability newNull =
            (oldFT.isNullable() || state.acceptsNull) ? Nullable : NonNullable;
          HeapType newHT =
            canUnifyToAllowed ? state.allowedHT : oldFT.getHeapType();
          Type newFT(newHT, newNull);
          if (newFT != oldFT) {
            structUpdates[U].push_back({key.index, newFT});
          }
        }
      }

      if (hasImmCandidate) {
        auto subTypesFirst = subTypes.getSubTypesFirstSort();
        for (auto it = subTypesFirst.rbegin(); it != subTypesFirst.rend();
             ++it) {
          auto U = *it;
          if (!U.isStruct()) {
            continue;
          }
          const auto& uFields = U.getStruct().fields;
          auto super = U.getDeclaredSuperType();
          bool uFixed = fixed.count(U) || U.isShared() ||
                        U.getDescriptorType() || U.getDescribedType();
          for (Index fi = 0; fi < uFields.size(); ++fi) {
            const auto& f = uFields[fi];
            if (f.mutable_ != Immutable || f.isPacked() || !f.type.isRef()) {
              continue;
            }
            Type oldFT = f.type;
            if (uFixed || oldFT.isExact()) {
              finalImmFieldTypes[FieldKey{U, fi}] = oldFT;
              continue;
            }
            bool hasParentField = super && super->isStruct() &&
                                  fi < super->getStruct().fields.size();
            Type boundFT = oldFT;
            if (hasParentField) {
              auto pIt = finalImmFieldTypes.find(FieldKey{*super, fi});
              boundFT = pIt != finalImmFieldTypes.end()
                          ? pIt->second
                          : super->getStruct().fields[fi].type;
            } else if (!oldFT.isNull()) {
              boundFT = Type(oldFT.getHeapType().getTop(), Nullable);
            } else {
              finalImmFieldTypes[FieldKey{U, fi}] = oldFT;
              continue;
            }
            if (boundFT.isExact() || boundFT.isNull()) {
              finalImmFieldTypes[FieldKey{U, fi}] = oldFT;
              continue;
            }
            HeapType allowedHT = boundFT.getHeapType();
            bool acceptsNull = boundFT.isNullable();
            if (auto demIt = immFieldDemands.find(FieldKey{U, fi});
                demIt != immFieldDemands.end()) {
              for (const auto& [reqHT, reqNull] : demIt->second) {
                if (!reqNull) {
                  acceptsNull = false;
                }
                if (HeapType::isSubType(allowedHT, reqHT)) {
                  continue;
                }
                if (oldFT.isNull()) {
                  if (HeapType::isSubType(reqHT, allowedHT) &&
                      !reqHT.isBottom()) {
                    allowedHT = reqHT;
                  } else {
                    allowedHT = oldFT.getHeapType();
                  }
                } else {
                  HeapType oldHT = oldFT.getHeapType();
                  if (HeapType::isSubType(reqHT, allowedHT) &&
                      HeapType::isSubType(oldHT, reqHT)) {
                    allowedHT = reqHT;
                  } else {
                    allowedHT = tightenAncestor(
                      oldHT,
                      HeapType::isSubType(oldHT, allowedHT) ? allowedHT : oldHT,
                      reqHT);
                  }
                }
              }
            }
            Type newFT = oldFT;
            if (oldFT.isNull()) {
              if (acceptsNull && !allowedHT.isBottom()) {
                newFT = Type(allowedHT, Nullable);
              }
            } else {
              Nullability newNull =
                (oldFT.isNullable() || acceptsNull) ? Nullable : NonNullable;
              newFT = Type(allowedHT, newNull);
            }
            if (!Type::isSubType(oldFT, newFT) ||
                !Type::isSubType(newFT, boundFT)) {
              newFT = oldFT;
            }
            finalImmFieldTypes[FieldKey{U, fi}] = newFT;
            if (newFT != oldFT) {
              structUpdates[U].push_back({fi, newFT});
            }
          }
        }
      }

      for (const auto& [arrHT, state] : arrayCandidates) {
        if (state.pinned) {
          continue;
        }
        Type oldET = arrHT.getArray().element.type;
        Nullability newNull =
          (oldET.isNullable() || state.acceptsNull) ? Nullable : NonNullable;
        HeapType newHT =
          state.canWidenHT ? state.allowedHT : oldET.getHeapType();
        Type newET(newHT, newNull);
        if (newET != oldET) {
          arrayUpdates[arrHT] = newET;
        }
      }

      bool shrunk = false;
      for (auto it = nullRelaxableFields.begin();
           it != nullRelaxableFields.end();) {
        bool becameNullable = false;
        if (auto fIt = fieldCandidates.find(*it);
            fIt != fieldCandidates.end()) {
          becameNullable = !fIt->second.pinned && fIt->second.acceptsNull;
        } else if (auto iIt = finalImmFieldTypes.find(*it);
                   iIt != finalImmFieldTypes.end()) {
          becameNullable = iIt->second.isNullable();
        }
        if (!becameNullable) {
          it = nullRelaxableFields.erase(it);
          shrunk = true;
        } else {
          ++it;
        }
      }
      for (auto it = nullRelaxableArrays.begin();
           it != nullRelaxableArrays.end();) {
        auto aIt = arrayCandidates.find(*it);
        bool becameNullable = aIt != arrayCandidates.end() &&
                              !aIt->second.pinned && aIt->second.acceptsNull;
        if (!becameNullable) {
          it = nullRelaxableArrays.erase(it);
          shrunk = true;
        } else {
          ++it;
        }
      }
      if (!shrunk) {
        break;
      }
    }

    std::unordered_set<Type> mutableElemTypes;
    for (auto ht : allTypes) {
      if (ht.isArray() && !ht.isShared() && !ht.getDescriptorType() &&
          !ht.getDescribedType() && !ht.getDeclaredSuperType() &&
          subTypes.getImmediateSubTypes(ht).empty()) {
        const auto& elem = ht.getArray().element;
        if (!elem.isPacked() && elem.mutable_ == Mutable &&
            elem.type.isRef() && !elem.type.isExact()) {
          auto uIt = arrayUpdates.find(ht);
          mutableElemTypes.insert(uIt != arrayUpdates.end() ? uIt->second
                                                            : elem.type);
        }
      }
    }

    auto noteStructTypeUse = [&](Type t, bool countUse) {
      for (Type elem : t) {
        if (elem.isRef() && elem.getHeapType().isStruct()) {
          if (countUse) {
            structTypeUses[elem.getHeapType()]++;
          }
          if (elem.isExact()) {
            exactCastStructs.insert(elem.getHeapType());
          }
        }
      }
    };
    for (auto ht : allTypes) {
      if (ht.isStruct()) {
        for (const auto& f : ht.getStruct().fields) {
          noteStructTypeUse(f.type, false);
        }
      } else if (ht.isArray()) {
        noteStructTypeUse(ht.getArray().element.type, true);
      } else if (ht.isSignature()) {
        noteStructTypeUse(ht.getSignature().params, true);
        noteStructTypeUse(ht.getSignature().results, true);
      }
    }
    for (auto& g : module->globals) {
      noteStructTypeUse(g->type, true);
    }
    for (auto& fn : module->functions) {
      if (!fn->imported() && !fn->vars.empty()) {
        Type prev = Type::none;
        for (Type v : fn->vars) {
          if (v != prev) {
            noteStructTypeUse(v, true);
            prev = v;
          }
        }
      }
    }

    auto typeCounts = ModuleUtils::collectHeapTypeInfo(
      *module,
      getPassOptions().worldMode,
      ModuleUtils::TypeInclusion::BinaryTypes);
    std::unordered_map<HeapType, double> typeWeights;
    std::vector<HeapType> rankedTypes;
    rankedTypes.reserve(allTypes.size());
    for (auto ht : allTypes) {
      typeWeights[ht] = typeCounts[ht].useCount;
      rankedTypes.push_back(ht);
    }
    for (auto ht : subTypes.getSubTypesFirstSort()) {
      if (auto sup = ht.getDeclaredSuperType()) {
        if (typeWeights.count(*sup)) {
          typeWeights[*sup] =
            std::max(typeWeights[*sup], typeWeights[ht] + 0.5);
        }
      }
    }
    std::stable_sort(rankedTypes.begin(),
                     rankedTypes.end(),
                     [&](const HeapType& a, const HeapType& b) {
                       return typeWeights[a] > typeWeights[b];
                     });
    std::unordered_map<HeapType, Index> typeIndex;
    for (Index i = 0; i < rankedTypes.size(); ++i) {
      typeIndex[rankedTypes[i]] = i;
    }

    auto getLebSavings = [&](HeapType M, Index cIdx) -> int {
      Index mIdx = typeIndex[M];
      auto it = typeCounts.find(M);
      if (it == typeCounts.end()) {
        return (cIdx < 64 && mIdx >= 64) ? int(structTypeUses[M]) : 0;
      }
      const auto& tc = it->second;
      int selfRef = 0;
      for (HeapType child : M.getReferencedHeapTypes()) {
        if (child == M) {
          selfRef = 1;
          break;
        }
      }
      int s33Uses = std::max(0, int(tc.s33UseCount) - selfRef);
      int u32Uses = int(tc.u32UseCount);
      int save = 0;
      if (cIdx < 64 && mIdx >= 64) {
        save += s33Uses;
      }
      if (cIdx < 128 && mIdx >= 128) {
        save += u32Uses;
      }
      return save;
    };

    auto isCleanPrivateStruct = [&](HeapType W) -> bool {
      return W.isStruct() && !fixed.count(W) && !W.isShared() &&
             !W.getDescriptorType() && !W.getDescribedType() &&
             !castStructs.count(W) && !exactCastStructs.count(W);
    };

    auto getFinalFieldType = [&](HeapType S, Index fi) -> Type {
      const auto& f = S.getStruct().fields[fi];
      if (auto iIt = finalImmFieldTypes.find(FieldKey{S, fi});
          iIt != finalImmFieldTypes.end()) {
        return iIt->second;
      }
      if (auto uIt = structUpdates.find(S); uIt != structUpdates.end()) {
        for (const auto& [ufi, uFT] : uIt->second) {
          if (ufi == fi) {
            return uFT;
          }
        }
      }
      return f.type;
    };

    std::unordered_map<HeapType, HeapType> mergedIntoSuper;
    std::unordered_set<HeapType> structOpenUpdates;
    std::unordered_map<HeapType, std::unordered_set<Index>> structMutUpdates;

    auto mapMergedHT = [&](HeapType ht) -> HeapType {
      while (true) {
        auto it = mergedIntoSuper.find(ht);
        if (it == mergedIntoSuper.end()) {
          break;
        }
        ht = it->second;
      }
      return ht;
    };

    auto mapMergedType = [&](Type t) -> Type {
      if (t.isRef()) {
        HeapType ht = t.getHeapType();
        HeapType mHT = mapMergedHT(ht);
        if (mHT != ht) {
          return Type(mHT, t.getNullability(), t.getExactness());
        }
      }
      return t;
    };

    auto recordMergedIntoSuper = [&](HeapType W) {
      auto super = W.getDeclaredSuperType();
      if (!super) {
        return;
      }
      mergedIntoSuper[W] = mapMergedHT(*super);
      if (cacheParamCasts && !W.isOpen()) {
        structOpenUpdates.insert(W);
      }
    };

    auto onlyDiffersAt = [&](HeapType W, Index diffFi) -> bool {
      if (!isCleanPrivateStruct(W)) {
        return false;
      }
      auto super = W.getDeclaredSuperType();
      if (!super || !super->isStruct()) {
        return false;
      }
      const auto& wFields = W.getStruct().fields;
      const auto& sFields = super->getStruct().fields;
      if (wFields.size() != sFields.size()) {
        return false;
      }
      for (Index k = 0; k < wFields.size(); ++k) {
        if (wFields[k].mutable_ != sFields[k].mutable_ ||
            wFields[k].packedType != sFields[k].packedType) {
          return false;
        }
        if (k == diffFi) {
          continue;
        }
        if (mapMergedType(getFinalFieldType(W, k)) !=
            mapMergedType(getFinalFieldType(*super, k))) {
          return false;
        }
      }
      return true;
    };

    if (cacheParamCasts) {
      auto subTypesFirst = subTypes.getSubTypesFirstSort();
      for (auto it = subTypesFirst.rbegin(); it != subTypesFirst.rend(); ++it) {
        HeapType W = *it;
        if (onlyDiffersAt(W, Index(-1))) {
          recordMergedIntoSuper(W);
        }
      }
    }

    std::unordered_map<FieldKey, Type, FieldKeyHash> castWidenedImmFields;
    auto getExactTypeBytes = [&](Type t) -> int {
      if (!t.isRef()) {
        return 1;
      }
      HeapType ht = t.getHeapType();
      if (ht.isBasic()) {
        return (t.isNullable() && !ht.isShared()) ? 1 : 2;
      }
      auto it = typeIndex.find(ht);
      return (it != typeIndex.end() && it->second < 64) ? 2 : 3;
    };

    auto getStructDefBytes = [&](HeapType W) -> int {
      auto super = W.getDeclaredSuperType();
      int bytes = super ? (5 + (typeIndex[*super] >= 64 ? 1 : 0)) : 2;
      const auto& fields = W.getStruct().fields;
      for (Index k = 0; k < fields.size(); ++k) {
        bytes += getExactTypeBytes(getFinalFieldType(W, k)) + 1;
      }
      return bytes;
    };

    auto countNarrowGets = [&](const std::vector<HeapType>& ws,
                               Index fi,
                               Type boundFT) -> int {
      int narrowGets = 0;
      for (auto W : ws) {
        if (auto demIt = immFieldDemands.find(FieldKey{W, fi});
            demIt != immFieldDemands.end()) {
          for (const auto& [reqHT, reqNull] : demIt->second) {
            if (!HeapType::isSubType(boundFT.getHeapType(), reqHT) ||
                (boundFT.isNullable() && !reqNull)) {
              narrowGets++;
            }
          }
        }
      }
      return narrowGets;
    };

    if (hasImmCandidate) {
      auto getTypeBytes = [&](Type t) -> int {
        if (!t.isRef()) {
          return 1;
        }
        HeapType ht = t.getHeapType();
        if (ht.isBasic()) {
          return (t.isNullable() && !ht.isShared()) ? 1 : 2;
        }
        if (!ht.getDeclaredSuperType() &&
            !subTypes.getImmediateSubTypes(ht).empty()) {
          return 2;
        }
        return 3;
      };

      std::unordered_map<HeapType, int> totalStructFieldHTCount;
      for (auto S : allTypes) {
        if (!S.isStruct()) {
          continue;
        }
        const auto& sFields = S.getStruct().fields;
        for (Index fi = 0; fi < sFields.size(); ++fi) {
          Type curFT = getFinalFieldType(S, fi);
          if (curFT.isRef()) {
            totalStructFieldHTCount[curFT.getHeapType()]++;
            if (S != curFT.getHeapType() && curFT.isStruct()) {
              structTypeUses[curFT.getHeapType()]++;
            }
          }
        }
      }

      std::unordered_set<Type> existingTargetMutElemTypes;
      for (auto ht : allTypes) {
        if (ht.isArray() && !ht.isShared() && !ht.getDescriptorType() &&
            !ht.getDescribedType() && !ht.getDeclaredSuperType() &&
            subTypes.getImmediateSubTypes(ht).empty()) {
          const auto& elem = ht.getArray().element;
          if (!elem.isPacked() && elem.mutable_ == Mutable &&
              elem.type.isRef() && !elem.type.isExact() &&
              !arrayUpdates.count(ht)) {
            existingTargetMutElemTypes.insert(elem.type);
          }
        }
      }

      std::unordered_set<HeapType> pureGlobalArrHTs;
      for (const auto& [gName, gState] : pureGlobalCandidates) {
        if (!gState.pinned && pureGlobalHasGet.count(gName) &&
            totalGlobalGets[gName] == validArrayGlobalGets[gName]) {
          pureGlobalArrHTs.insert(module->getGlobal(gName)->type.getHeapType());
        }
      }

      auto hasCoupledMutArray = [&](Type oldFT,
                                    Type boundFT,
                                    int groupStructCount) -> bool {
        if (!oldFT.isStruct() || !existingTargetMutElemTypes.count(boundFT)) {
          return false;
        }
        auto cntIt = totalStructFieldHTCount.find(oldFT.getHeapType());
        if (cntIt == totalStructFieldHTCount.end() ||
            cntIt->second != groupStructCount) {
          return false;
        }
        for (const auto& [arrHT, state] : arrayCandidates) {
          if (!state.pinned && arrHT.getArray().element.mutable_ == Mutable &&
              arrHT.getArray().element.type == oldFT &&
              !arrayUpdates.count(arrHT) && pureGlobalArrHTs.count(arrHT)) {
            return true;
          }
        }
        return false;
      };

      auto hasCoupledSelfMutArray = [&](HeapType W, HeapType S) -> bool {
        Type wRef(W, Nullable);
        Type sRef(S, Nullable);
        if (!existingTargetMutElemTypes.count(sRef)) {
          return false;
        }
        for (const auto& [arrHT, state] : arrayCandidates) {
          if (!state.pinned && arrHT.getArray().element.mutable_ == Mutable &&
              arrHT.getArray().element.type == wRef &&
              !arrayUpdates.count(arrHT)) {
            return true;
          }
        }
        return false;
      };

      std::unordered_map<HeapType, int> coupledHolderBonus;
      std::unordered_map<HeapType, std::vector<HeapType>> coupledHolders;
      if (cacheParamCasts) {
        for (auto H : allTypes) {
          if (!isCleanPrivateStruct(H) || mergedIntoSuper.count(H)) {
            continue;
          }
          auto sH = H.getDeclaredSuperType();
          if (!sH || !sH->isStruct()) {
            continue;
          }
          const auto& hFields = H.getStruct().fields;
          if (hFields.size() != sH->getStruct().fields.size()) {
            continue;
          }
          for (Index hFi = 0; hFi < hFields.size(); ++hFi) {
            if (!onlyDiffersAt(H, hFi)) {
              continue;
            }
            Type hFT = getFinalFieldType(H, hFi);
            Type sFT = getFinalFieldType(*sH, hFi);
            if (hFT.isStruct() && sFT.isStruct() &&
                hFT.getNullability() == sFT.getNullability()) {
              HeapType V = hFT.getHeapType();
              HeapType sV = sFT.getHeapType();
              if (V != sV && isCleanPrivateStruct(V) &&
                  V.getDeclaredSuperType() &&
                  mapMergedHT(*V.getDeclaredSuperType()) == mapMergedHT(sV)) {
                HeapType effSH = mapMergedHT(*sH);
                int hBonus = getStructDefBytes(H) +
                             ((typeIndex[effSH] < 64 && typeIndex[H] >= 64)
                                ? int(structTypeUses[H])
                                : 0);
                coupledHolderBonus[V] += hBonus;
                coupledHolders[V].push_back(H);
              }
            }
            break;
          }
        }
      }

      auto applyGroupWiden = [&](const std::vector<HeapType>& ws,
                                 Index fi,
                                 Type curFT,
                                 Type boundFT) {
        for (auto W : ws) {
          bool mergesToSuper = cacheParamCasts && onlyDiffersAt(W, fi);
          finalImmFieldTypes[FieldKey{W, fi}] = boundFT;
          castWidenedImmFields[FieldKey{W, fi}] = curFT;
          auto& vec = structUpdates[W];
          bool found = false;
          for (auto& [ufi, uFT] : vec) {
            if (ufi == fi) {
              uFT = boundFT;
              found = true;
              break;
            }
          }
          if (!found && boundFT != W.getStruct().fields[fi].type) {
            vec.push_back({fi, boundFT});
          }
          if (mergesToSuper) {
            recordMergedIntoSuper(W);
            if (auto cIt = coupledHolders.find(W);
                cIt != coupledHolders.end()) {
              for (auto H : cIt->second) {
                recordMergedIntoSuper(H);
              }
            }
          }
        }
      };

      int numPhases = cacheParamCasts ? 3 : 1;
      auto ancTypes = allTypes;
      if (cacheParamCasts) {
        auto subTypesFirst = subTypes.getSubTypesFirstSort();
        ancTypes.assign(subTypesFirst.rbegin(), subTypesFirst.rend());
      }
      for (int phase = 0; phase < numPhases; ++phase) {
        for (auto anc : ancTypes) {
          if (!anc.isStruct()) {
            continue;
          }
          const auto& aFields = anc.getStruct().fields;
          for (Index fi = 0; fi < aFields.size(); ++fi) {
            if (aFields[fi].mutable_ != Immutable) {
              continue;
            }
            if (phase < 2) {
              if (getHighestFieldAncestor(anc, fi) != anc) {
                continue;
              }
            } else {
              if (getHighestFieldAncestor(anc, fi) == anc) {
                continue;
              }
              auto ancSup = anc.getDeclaredSuperType();
              if (!ancSup || !ancSup->isStruct() ||
                  fi >= ancSup->getStruct().fields.size() ||
                  getFinalFieldType(anc, fi) ==
                    getFinalFieldType(*ancSup, fi)) {
                continue;
              }
            }
            auto ancIt = finalImmFieldTypes.find(FieldKey{anc, fi});
            if (ancIt == finalImmFieldTypes.end()) {
              continue;
            }
            Type ancFT = ancIt->second;
            if (ancFT.isExact() || ancFT.isNull()) {
              continue;
            }
            if (cacheParamCasts && phase < 2) {
              bool isBasicAnc = ancFT.getHeapType().isBasic();
              if ((phase == 0) != isBasicAnc) {
                continue;
              }
            }
            const auto& family = subTypes.getSubTypes(anc);
            std::unordered_map<Type, std::vector<HeapType>> groups;
            for (auto W : family) {
              if (W == anc || castWidenedImmFields.count(FieldKey{W, fi})) {
                continue;
              }
              bool wFixed = fixed.count(W) || W.isShared() ||
                            W.getDescriptorType() || W.getDescribedType();
              Type origFT = W.getStruct().fields[fi].type;
              if (wFixed || origFT.isExact() || origFT.isNull() ||
                  (!cacheParamCasts && origFT.getHeapType().isSignature())) {
                continue;
              }
              auto wIt = finalImmFieldTypes.find(FieldKey{W, fi});
              if (wIt == finalImmFieldTypes.end()) {
                continue;
              }
              Type curFT = wIt->second;
              if (cacheParamCasts &&
                  mapMergedType(curFT).getHeapType() ==
                    mapMergedType(ancFT).getHeapType()) {
                if (onlyDiffersAt(W, fi)) {
                  recordMergedIntoSuper(W);
                }
                continue;
              }
              if (curFT.getHeapType() == ancFT.getHeapType() ||
                  !Type::isSubType(curFT, ancFT)) {
                continue;
              }
              groups[curFT].push_back(W);
            }

            if (!groups.empty()) {
              Type boundFT = ancFT;
              if (cacheParamCasts) {
                for (auto it = groups.begin(); it != groups.end();) {
                  Type curFT = it->first;
                  auto& ws = it->second;
                  if (!hasCoupledMutArray(curFT, boundFT, int(ws.size()))) {
                    ws.erase(
                      std::remove_if(
                        ws.begin(),
                        ws.end(),
                        [&](HeapType W) {
                          if (!subTypes.getImmediateSubTypes(W).empty() ||
                              onlyDiffersAt(W, fi)) {
                            return false;
                          }
                          int wDef =
                            getTypeBytes(curFT) - getTypeBytes(boundFT);
                          int wGets = countNarrowGets({W}, fi, boundFT);
                          return wDef - wGets * 3 <= 0;
                        }),
                      ws.end());
                  }
                  if (ws.empty()) {
                    it = groups.erase(it);
                  } else {
                    ++it;
                  }
                }
              }
              if (groups.empty()) {
                continue;
              }
              std::unordered_map<Type, Type> parentGroup;
              std::unordered_map<Type, std::vector<Type>> childrenGroups;
              std::unordered_set<Type> invalidGroups;
              for (const auto& [curFT, ws] : groups) {
                for (auto W : ws) {
                  HeapType p = *W.getDeclaredSuperType();
                  auto pIt = finalImmFieldTypes.find(FieldKey{p, fi});
                  Type pFT = pIt != finalImmFieldTypes.end()
                               ? pIt->second
                               : p.getStruct().fields[fi].type;
                  if (pFT != boundFT && pFT != curFT) {
                    if (!groups.count(pFT) ||
                        (parentGroup.count(curFT) &&
                         parentGroup[curFT] != pFT)) {
                      invalidGroups.insert(curFT);
                    } else if (!parentGroup.count(curFT)) {
                      parentGroup[curFT] = pFT;
                      childrenGroups[pFT].push_back(curFT);
                    }
                  }
                }
              }
              std::unordered_map<Type, int> groupSelfNet;
              for (const auto& [curFT, ws] : groups) {
                int defSavings = 0;
                for (auto W : ws) {
                  if (cacheParamCasts && onlyDiffersAt(W, fi)) {
                    HeapType effSuper = mapMergedHT(*W.getDeclaredSuperType());
                    defSavings += getStructDefBytes(W);
                    if (typeIndex[effSuper] < 64 && typeIndex[W] >= 64) {
                      defSavings += int(structTypeUses[W]);
                    }
                    defSavings += coupledHolderBonus[W];
                    if (hasCoupledSelfMutArray(W, effSuper)) {
                      defSavings += 21;
                    }
                  } else {
                    defSavings += getTypeBytes(curFT) - getTypeBytes(boundFT);
                  }
                }
                int narrowGets = countNarrowGets(ws, fi, boundFT);
                int arrBonus =
                  hasCoupledMutArray(curFT, boundFT, int(ws.size())) ? 21 : 0;
                groupSelfNet[curFT] = defSavings + arrBonus - narrowGets * 3;
              }
              std::unordered_map<Type, int> groupBestNet;
              auto computeDP = [&](auto& self, Type ft) -> int {
                if (invalidGroups.count(ft)) {
                  groupBestNet[ft] = -1000000;
                  return -1000000;
                }
                int total = groupSelfNet[ft];
                for (Type childFT : childrenGroups[ft]) {
                  int childBest = self(self, childFT);
                  if (childBest > 0) {
                    total += childBest;
                  }
                }
                groupBestNet[ft] = total;
                return total;
              };
              for (const auto& [ft, _] : groups) {
                if (!parentGroup.count(ft)) {
                  computeDP(computeDP, ft);
                }
              }
              auto selectDP = [&](auto& self, Type ft) -> void {
                if (groupBestNet[ft] <= 0) {
                  return;
                }
                applyGroupWiden(groups[ft], fi, ft, boundFT);
                for (Type childFT : childrenGroups[ft]) {
                  if (groupBestNet[childFT] > 0) {
                    self(self, childFT);
                  }
                }
              };
              for (const auto& [ft, _] : groups) {
                if (!parentGroup.count(ft)) {
                  selectDP(selectDP, ft);
                }
              }
            } else if (phase == 0 || !cacheParamCasts) {
              auto ancSuper = anc.getDeclaredSuperType();
              bool ancHasParentField =
                ancSuper && ancSuper->isStruct() &&
                fi < ancSuper->getStruct().fields.size();
              if (!ancHasParentField && !ancFT.getHeapType().isSignature()) {
                Type boundFT(ancFT.getHeapType().getTop(), Nullable);
                if (ancFT.getHeapType() != boundFT.getHeapType() &&
                    Type::isSubType(ancFT, boundFT)) {
                  bool allValid = true;
                  for (auto W : family) {
                    bool wFixed = fixed.count(W) || W.isShared() ||
                                  W.getDescriptorType() || W.getDescribedType();
                    Type origFT = W.getStruct().fields[fi].type;
                    auto wIt = finalImmFieldTypes.find(FieldKey{W, fi});
                    if (wFixed || origFT.isExact() || origFT.isNull() ||
                        wIt == finalImmFieldTypes.end() ||
                        wIt->second != ancFT) {
                      allValid = false;
                      break;
                    }
                  }
                  if (allValid &&
                      hasCoupledMutArray(ancFT, boundFT, int(family.size()))) {
                    int defSavings =
                      int(family.size()) *
                      (getTypeBytes(ancFT) - getTypeBytes(boundFT));
                    int narrowGets = countNarrowGets(family, fi, boundFT);
                    if (defSavings + 21 - narrowGets * 3 > 0) {
                      applyGroupWiden(family, fi, ancFT, boundFT);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    struct StructFieldLayoutUpdate {
      FieldList newFields;
      std::vector<Index> oldToNew;
      std::vector<Index> newToOld;
    };
    std::unordered_map<HeapType, StructFieldLayoutUpdate> structLayoutUpdates;
    std::unordered_set<HeapType> collapsedParents;
    std::unordered_set<HeapType> usedInStepC;

    auto getDefaultConstBytes = [](Type t) -> int {
      if (t == Type::i32 || t == Type::i64) {
        return 2;
      }
      if (t == Type::f32) {
        return 5;
      }
      if (t == Type::f64) {
        return 9;
      }
      if (t.isRef() && t.isNullable()) {
        return 2;
      }
      return -1;
    };

    auto getNonDefaultStructNewCount = [&](HeapType S) -> int {
      auto it = allStructNews.find(S);
      if (it == allStructNews.end()) {
        return 0;
      }
      int cnt = 0;
      for (auto* snew : it->second) {
        if (!snew->isWithDefault()) {
          cnt++;
        }
      }
      return cnt;
    };

    auto hasPinnedField = [&](HeapType S) -> bool {
      const auto& fields = S.getStruct().fields;
      for (Index fi = 0; fi < fields.size(); ++fi) {
        HeapType anc = getHighestFieldAncestor(S, fi);
        if (allPinnedFields.count(FieldKey{anc, fi}) ||
            allPinnedFields.count(FieldKey{S, fi})) {
          return true;
        }
      }
      return false;
    };

    if (cacheParamCasts) {
      struct StructShapeKey {
        std::optional<HeapType> super;
        std::vector<std::pair<Type, Field::PackedType>> fields;
        bool operator==(const StructShapeKey& o) const {
          return super == o.super && fields == o.fields;
        }
      };
      struct StructShapeKeyHash {
        size_t operator()(const StructShapeKey& k) const {
          size_t h =
            k.super ? std::hash<HeapType>{}(*k.super) : size_t(0x9e3779b9);
          for (const auto& [ft, pt] : k.fields) {
            h ^= std::hash<Type>{}(ft) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= size_t(pt) + 0x9e3779b9 + (h << 6) + (h >> 2);
          }
          return h;
        }
      };

      auto getEffMut = [&](HeapType T, Index fi) -> Mutability {
        if (T.getStruct().fields[fi].mutable_ == Mutable) {
          return Mutable;
        }
        auto it = structMutUpdates.find(T);
        if (it != structMutUpdates.end() && it->second.count(fi)) {
          return Mutable;
        }
        return Immutable;
      };

      auto getPostStepCNumFields = [&](HeapType T) -> Index {
        auto lIt = structLayoutUpdates.find(T);
        if (lIt != structLayoutUpdates.end()) {
          return Index(lIt->second.newFields.size());
        }
        return Index(T.getStruct().fields.size());
      };

      auto getPostStepCField = [&](HeapType T, Index newFi) -> Field {
        auto lIt = structLayoutUpdates.find(T);
        if (lIt != structLayoutUpdates.end()) {
          return lIt->second.newFields[newFi];
        }
        Field f = T.getStruct().fields[newFi];
        f.type = getFinalFieldType(T, newFi);
        f.mutable_ = getEffMut(T, newFi);
        return f;
      };

      auto canMakeMut = [&](HeapType T, Index fi) -> bool {
        HeapType root = getHighestFieldAncestor(T, fi);
        if (root != T) {
          return false;
        }
        if (auto fIt = fieldCandidates.find(FieldKey{root, fi});
            fIt != fieldCandidates.end() && fIt->second.pinned) {
          return false;
        }
        if (allPinnedFields.count(FieldKey{root, fi})) {
          return false;
        }
        Field rootF = getPostStepCField(root, fi);
        for (auto V : subTypes.getSubTypes(root)) {
          if (fixed.count(V) || V.isShared() || V.getDescriptorType() ||
              V.getDescribedType()) {
            return false;
          }
          Index vFi = fi;
          if (auto vIt = structLayoutUpdates.find(V);
              vIt != structLayoutUpdates.end()) {
            if (fi >= vIt->second.oldToNew.size() ||
                vIt->second.oldToNew[fi] != fi) {
              return false;
            }
          }
          Field vF = getPostStepCField(V, vFi);
          if (vF.type != rootF.type || vF.packedType != rootF.packedType) {
            return false;
          }
        }
        return true;
      };

      auto canMakePostStepCMut = [&](HeapType T, Index newFi) -> bool {
        auto lIt = structLayoutUpdates.find(T);
        if (lIt != structLayoutUpdates.end()) {
          Index oldFi = lIt->second.newToOld[newFi];
          if (oldFi == Index(-1)) {
            return true;
          }
          if (getHighestFieldAncestor(T, oldFi) == T &&
              subTypes.getImmediateSubTypes(T).empty()) {
            auto fIt = fieldCandidates.find(FieldKey{T, oldFi});
            return (fIt == fieldCandidates.end() || !fIt->second.pinned) &&
                   !allPinnedFields.count(FieldKey{T, oldFi});
          }
          return canMakeMut(T, oldFi);
        }
        if (getHighestFieldAncestor(T, newFi) == T &&
            subTypes.getImmediateSubTypes(T).empty()) {
          auto fIt = fieldCandidates.find(FieldKey{T, newFi});
          return (fIt == fieldCandidates.end() || !fIt->second.pinned) &&
                 !allPinnedFields.count(FieldKey{T, newFi});
        }
        return canMakeMut(T, newFi);
      };

      auto setRootSubtreeMut = [&](HeapType root, Index oldFi) {
        for (auto V : subTypes.getSubTypes(root)) {
          structMutUpdates[V].insert(oldFi);
          if (auto vIt = structLayoutUpdates.find(V);
              vIt != structLayoutUpdates.end()) {
            if (oldFi < vIt->second.oldToNew.size()) {
              Index vNewFi = vIt->second.oldToNew[oldFi];
              if (vNewFi < vIt->second.newFields.size()) {
                vIt->second.newFields[vNewFi].mutable_ = Mutable;
              }
            }
          }
        }
      };

      auto setPostStepCMut = [&](HeapType T, Index newFi) {
        auto lIt = structLayoutUpdates.find(T);
        if (lIt != structLayoutUpdates.end()) {
          lIt->second.newFields[newFi].mutable_ = Mutable;
          Index oldFi = lIt->second.newToOld[newFi];
          if (oldFi != Index(-1) && getHighestFieldAncestor(T, oldFi) != T) {
            HeapType root = getHighestFieldAncestor(T, oldFi);
            setRootSubtreeMut(root, oldFi);
          }
          return;
        }
        HeapType root = getHighestFieldAncestor(T, newFi);
        setRootSubtreeMut(root, newFi);
      };

      auto subTypesFirst = subTypes.getSubTypesFirstSort();

      if (allowSiblingCastWiden) {
        // Component 3: Single-Child Uncast Parent/Child Collapsing (P -> C).
        // When an uncast non-leaf struct P has a single immediate subtype C, C
        // is an uncast leaf struct with len(C) > len(P), and P's prefix fields
        // 0..len(P)-1 match C's inherited prefix (after optional covariant
        // widening on C), append C's extra fields len(P)..len(C)-1 to P and pad
        // struct.new $P with default constants so TypeMerging::merge(Supertypes)
        // collapses C into P.
        for (auto it = subTypesFirst.rbegin(); it != subTypesFirst.rend();
             ++it) {
          HeapType P = *it;
          if (!isCleanPrivateStruct(P) || mergedIntoSuper.count(P) ||
              collapsedParents.count(P) || hasPinnedField(P)) {
            continue;
          }
          const auto& immSubs = subTypes.getImmediateSubTypes(P);
          if (immSubs.size() != 1) {
            continue;
          }
          HeapType C = immSubs[0];
          if (!isCleanPrivateStruct(C) || mergedIntoSuper.count(C) ||
              collapsedParents.count(C) ||
              !subTypes.getImmediateSubTypes(C).empty() || hasPinnedField(C)) {
            continue;
          }
          const auto& pFields = P.getStruct().fields;
          const auto& cFields = C.getStruct().fields;
          if (cFields.size() <= pFields.size()) {
            continue;
          }
          bool ok = true;
          int castCost = 0;
          std::vector<Index> widenChildFields;
          std::vector<Index> makeMutPrefixFields;
          for (Index fi = 0; fi < pFields.size(); ++fi) {
            if (pFields[fi].packedType != cFields[fi].packedType) {
              ok = false;
              break;
            }
            Mutability pMut = getEffMut(P, fi);
            Mutability cMut = getEffMut(C, fi);
            if (pMut != cMut) {
              if (pMut == Immutable && canMakeMut(P, fi)) {
                makeMutPrefixFields.push_back(fi);
              } else {
                ok = false;
                break;
              }
            }
            Type pRealFT = getFinalFieldType(P, fi);
            Type cRealFT = getFinalFieldType(C, fi);
            if (pRealFT != cRealFT) {
              if (cFields[fi].mutable_ != Immutable || cMut != Immutable ||
                  pMut != Immutable || cFields[fi].isPacked() ||
                  !cRealFT.isRef() || !cRealFT.isNullable() ||
                  cRealFT.isExact() || cRealFT.isNull() ||
                  castWidenedImmFields.count(FieldKey{C, fi}) ||
                  pRealFT.isExact() || pRealFT.isNull() ||
                  !Type::isSubType(cRealFT, pRealFT)) {
                ok = false;
                break;
              }
              int narrowGets = countNarrowGets({C}, fi, pRealFT);
              int castBytes =
                (typeIndex[cRealFT.getHeapType()] < 64) ? 3 : 4;
              castCost += narrowGets * castBytes;
              widenChildFields.push_back(fi);
            }
          }
          if (!ok) {
            continue;
          }
          int padBytesPerNew = 0;
          for (Index fi = pFields.size(); fi < cFields.size(); ++fi) {
            Type cRealFT = getFinalFieldType(C, fi);
            int cb = getDefaultConstBytes(cRealFT);
            if (cb < 0) {
              ok = false;
              break;
            }
            padBytesPerNew += cb;
          }
          if (!ok) {
            continue;
          }
          int pNews = getNonDefaultStructNewCount(P);
          int padCost = pNews * padBytesPerNew;
          int defSave = getStructDefBytes(P);
          Index cIdx = std::min(typeIndex[P], typeIndex[C]);
          int lebSave = getLebSavings(P, cIdx) + getLebSavings(C, cIdx);
          int net = defSave + lebSave - padCost - castCost;
          if (net <= 0) {
            continue;
          }
          for (Index fi : makeMutPrefixFields) {
            HeapType root = getHighestFieldAncestor(P, fi);
            setRootSubtreeMut(root, fi);
          }
          for (Index fi : widenChildFields) {
            Type pRealFT = getFinalFieldType(P, fi);
            finalImmFieldTypes[FieldKey{C, fi}] = pRealFT;
            castWidenedImmFields[FieldKey{C, fi}] = cFields[fi].type;
            auto& vec = structUpdates[C];
            bool found = false;
            for (auto& [ufi, uFT] : vec) {
              if (ufi == fi) {
                uFT = pRealFT;
                found = true;
                break;
              }
            }
            if (!found && pRealFT != cFields[fi].type) {
              vec.push_back({fi, pRealFT});
            }
          }
          StructFieldLayoutUpdate layout;
          layout.oldToNew.resize(pFields.size());
          layout.newToOld.assign(cFields.size(), Index(-1));
          layout.newFields.resize(cFields.size());
          for (Index fi = 0; fi < pFields.size(); ++fi) {
            layout.oldToNew[fi] = fi;
            layout.newToOld[fi] = fi;
            Field f = pFields[fi];
            f.type = getFinalFieldType(P, fi);
            f.mutable_ = getEffMut(P, fi);
            layout.newFields[fi] = f;
          }
          for (Index fi = pFields.size(); fi < cFields.size(); ++fi) {
            Field f = cFields[fi];
            f.type = getFinalFieldType(C, fi);
            f.mutable_ = getEffMut(C, fi);
            layout.newFields[fi] = f;
          }
          structLayoutUpdates[P] = std::move(layout);
          collapsedParents.insert(P);
          recordMergedIntoSuper(C);
        }

        // Step C + Component 2: Sibling-targeted cast-inserting field widening,
        // mutability unification, and canonical non-inherited field permutation
        // (fi >= supLen) on uncast leaf structs.
        auto coarseFieldLess =
          [](const std::pair<Type, Field::PackedType>& a,
             const std::pair<Type, Field::PackedType>& b) {
            if (a.first != b.first) {
              return a.first.getID() < b.first.getID();
            }
            return a.second < b.second;
          };

        std::vector<StructShapeKey> skelOrder;
        std::unordered_map<StructShapeKey,
                           std::vector<HeapType>,
                           StructShapeKeyHash>
          skelGroups;
        for (auto it = subTypesFirst.rbegin(); it != subTypesFirst.rend();
             ++it) {
          HeapType T = *it;
          if (!isCleanPrivateStruct(T) || mergedIntoSuper.count(T) ||
              collapsedParents.count(T) ||
              !subTypes.getImmediateSubTypes(T).empty()) {
            continue;
          }
          StructShapeKey skel;
          if (auto s = T.getDeclaredSuperType()) {
            skel.super = mapMergedHT(*s);
          }
          Index supLen =
            (skel.super && skel.super->isStruct())
              ? Index(skel.super->getStruct().fields.size())
              : 0;
          const auto& tFields = T.getStruct().fields;
          skel.fields.reserve(tFields.size());
          for (Index fi = 0; fi < tFields.size(); ++fi) {
            Type curFT = mapMergedType(getFinalFieldType(T, fi));
            Field::PackedType pt = tFields[fi].packedType;
            Type coarseFT = (!pt && curFT.isRef() && curFT.isNullable() &&
                             !curFT.isExact() && !curFT.isNull())
                              ? Type(curFT.getHeapType().getTop(), Nullable)
                              : curFT;
            skel.fields.push_back({coarseFT, pt});
          }
          if (skel.fields.size() > supLen) {
            std::sort(skel.fields.begin() + supLen,
                      skel.fields.end(),
                      coarseFieldLess);
          }
          auto& vec = skelGroups[skel];
          if (vec.empty()) {
            skelOrder.push_back(skel);
          }
          vec.push_back(T);
        }

        using CandField = std::pair<Type, Field::PackedType>;
        using CandShape = std::vector<CandField>;

        auto getShapeDefBytes = [&](std::optional<HeapType> sup,
                                    const CandShape& fts) -> int {
          int bytes = sup ? (5 + (typeIndex[*sup] >= 64 ? 1 : 0)) : 2;
          for (const auto& [ft, _] : fts) {
            bytes += getExactTypeBytes(ft) + 1;
          }
          return bytes;
        };

        for (const auto& skel : skelOrder) {
          const auto& members = skelGroups[skel];
          if (members.size() < 2) {
            continue;
          }
          Index numF = skel.fields.size();
          Index supLen =
            (skel.super && skel.super->isStruct())
              ? Index(skel.super->getStruct().fields.size())
              : 0;

          std::vector<CandShape> mCurShapes(members.size());
          std::vector<CandShape> mCoarse(members.size());
          std::vector<std::vector<Index>> canonOrder(members.size());
          std::vector<std::vector<bool>> canSwap(
            members.size(), std::vector<bool>(numF * numF, false));
          std::vector<bool> canPermuteToCanon(members.size(), true);
          std::vector<CandShape> candShapes;
          auto addCandShape = [&](const CandShape& s) -> Index {
            for (Index i = 0; i < candShapes.size(); ++i) {
              if (candShapes[i] == s) {
                return i;
              }
            }
            candShapes.push_back(s);
            return Index(candShapes.size() - 1);
          };

          std::vector<Index> mCurShapeIdx(members.size());
          std::optional<HeapType> commonObjRoot;
          for (size_t mi = 0; mi < members.size(); ++mi) {
            HeapType M = members[mi];
            const auto& mFields = M.getStruct().fields;
            mCurShapes[mi].resize(numF);
            mCoarse[mi].resize(numF);
            canonOrder[mi].resize(numF);
            for (Index fi = 0; fi < numF; ++fi) {
              Type ft = mapMergedType(getFinalFieldType(M, fi));
              Field::PackedType pt = mFields[fi].packedType;
              mCurShapes[mi][fi] = {ft, pt};
              Type coarseFT = (!pt && ft.isRef() && ft.isNullable() &&
                               !ft.isExact() && !ft.isNull())
                                ? Type(ft.getHeapType().getTop(), Nullable)
                                : ft;
              mCoarse[mi][fi] = {coarseFT, pt};
              canonOrder[mi][fi] = fi;
              if (!commonObjRoot && fi >= supLen && ft.isStruct()) {
                HeapType rHT = getRootAncestor(ft.getHeapType());
                if (!rHT.isBasic() &&
                    !subTypes.getImmediateSubTypes(rHT).empty()) {
                  commonObjRoot = rHT;
                }
              }
            }
            if (numF > supLen) {
              std::stable_sort(
                canonOrder[mi].begin() + supLen,
                canonOrder[mi].end(),
                [&](Index a, Index b) {
                  return coarseFieldLess(mCoarse[mi][a], mCoarse[mi][b]);
                });
            }
            if (numF - supLen >= 2 && numF - supLen <= 12 &&
                !hasPinnedField(M)) {
              for (Index a = supLen; a < numF; ++a) {
                for (Index b = a + 1; b < numF; ++b) {
                  canSwap[mi][a * numF + b] = true;
                }
              }
              auto nIt = allStructNews.find(M);
              if (nIt != allStructNews.end()) {
                for (auto* snew : nIt->second) {
                  if (snew->isWithDefault() ||
                      snew->operands.size() != numF) {
                    continue;
                  }
                  std::vector<EffectAnalyzer> eas;
                  eas.reserve(numF - supLen);
                  for (Index fi = supLen; fi < numF; ++fi) {
                    eas.emplace_back(
                      getPassOptions(), *module, snew->operands[fi]);
                  }
                  for (Index a = supLen; a < numF; ++a) {
                    for (Index b = a + 1; b < numF; ++b) {
                      if (canSwap[mi][a * numF + b]) {
                        const auto& eaA = eas[a - supLen];
                        const auto& eaB = eas[b - supLen];
                        if (eaA.orderedBefore(eaB) || eaB.orderedBefore(eaA)) {
                          canSwap[mi][a * numF + b] = false;
                        }
                      }
                    }
                  }
                }
              }
            }
            for (Index c1 = supLen; c1 < numF; ++c1) {
              for (Index c2 = c1 + 1; c2 < numF; ++c2) {
                Index old1 = canonOrder[mi][c1];
                Index old2 = canonOrder[mi][c2];
                if (old1 > old2 && !canSwap[mi][old2 * numF + old1]) {
                  canPermuteToCanon[mi] = false;
                  break;
                }
              }
              if (!canPermuteToCanon[mi]) {
                break;
              }
            }
            mCurShapeIdx[mi] = addCandShape(mCurShapes[mi]);
          }

          std::vector<bool> canonVarying(numF, false);
          std::vector<bool> canonHasObjRootAt(numF, false);
          std::vector<bool> canonAllObjRootAt(numF, commonObjRoot.has_value());
          for (Index cPos = 0; cPos < numF; ++cPos) {
            CandField firstCF = mCurShapes[0][canonOrder[0][cPos]];
            for (size_t mi = 0; mi < members.size(); ++mi) {
              CandField cf = mCurShapes[mi][canonOrder[mi][cPos]];
              if (cf != firstCF) {
                canonVarying[cPos] = true;
              }
              if (commonObjRoot &&
                  Type::isSubType(cf.first, Type(*commonObjRoot, Nullable))) {
                canonHasObjRootAt[cPos] = true;
              } else {
                canonAllObjRootAt[cPos] = false;
              }
            }
          }

          for (size_t mi = 0; mi < members.size(); ++mi) {
            std::vector<Index> oldToCanonPos(numF);
            for (Index cPos = 0; cPos < numF; ++cPos) {
              oldToCanonPos[canonOrder[mi][cPos]] = cPos;
            }
            CandShape memObjShape(numF);
            CandShape memObjOrTopShape(numF);
            CandShape memTopShape(numF);
            for (Index fi = 0; fi < numF; ++fi) {
              Index cPos = oldToCanonPos[fi];
              Field::PackedType pt = mCurShapes[mi][fi].second;
              if (!canonVarying[cPos]) {
                memObjShape[fi] = mCurShapes[mi][fi];
                memObjOrTopShape[fi] = mCurShapes[mi][fi];
                memTopShape[fi] = mCurShapes[mi][fi];
              } else if (fi < supLen) {
                Type sFT = mapMergedType(getFinalFieldType(*skel.super, fi));
                memObjShape[fi] = {sFT, pt};
                memObjOrTopShape[fi] = {sFT, pt};
                memTopShape[fi] = {sFT, pt};
              } else {
                Type ft = mCurShapes[mi][fi].first;
                Type objFT = ft;
                if (ft.isStruct()) {
                  HeapType rHT = getRootAncestor(ft.getHeapType());
                  if (!rHT.isBasic() &&
                      !subTypes.getImmediateSubTypes(rHT).empty()) {
                    objFT = Type(rHT, Nullable);
                  }
                }
                Type objOrTopFT =
                  (commonObjRoot && ft.isRef() && ft.isNullable() &&
                   !ft.isExact() && !ft.isNull() &&
                   HeapType::isSubType(ft.getHeapType(), *commonObjRoot))
                    ? Type(*commonObjRoot, Nullable)
                    : mCoarse[mi][fi].first;
                memObjShape[fi] = {objFT, pt};
                memObjOrTopShape[fi] = {objOrTopFT, pt};
                memTopShape[fi] = {mCoarse[mi][fi].first, pt};
              }
            }
            addCandShape(memObjShape);
            addCandShape(memObjOrTopShape);
            addCandShape(memTopShape);
            if (canPermuteToCanon[mi]) {
              CandShape canonCur(numF);
              CandShape canonObj(numF);
              CandShape canonObjOrTop(numF);
              CandShape canonTop(numF);
              for (Index cPos = 0; cPos < numF; ++cPos) {
                Index oldFi = canonOrder[mi][cPos];
                canonCur[cPos] = mCurShapes[mi][oldFi];
                canonObj[cPos] = memObjShape[oldFi];
                canonObjOrTop[cPos] = memObjOrTopShape[oldFi];
                canonTop[cPos] = memTopShape[oldFi];
              }
              addCandShape(canonCur);
              addCandShape(canonObj);
              addCandShape(canonObjOrTop);
              addCandShape(canonTop);
            }
          }

          Type anyRefT(HeapType::any, Nullable);
          for (int useObjRoot = 0; useObjRoot < 3; ++useObjRoot) {
            if (useObjRoot && !commonObjRoot) {
              continue;
            }
            for (int onlyVarying = 0; onlyVarying < 2; ++onlyVarying) {
              CandShape synth(numF);
              for (Index cPos = 0; cPos < numF; ++cPos) {
                Field::PackedType pt = skel.fields[cPos].second;
                if (cPos < supLen) {
                  Type ft = canonVarying[cPos]
                              ? mapMergedType(
                                  getFinalFieldType(*skel.super, cPos))
                              : mCurShapes[0][cPos].first;
                  synth[cPos] = {ft, pt};
                } else if (onlyVarying && !canonVarying[cPos]) {
                  synth[cPos] = mCurShapes[0][canonOrder[0][cPos]];
                } else if (skel.fields[cPos].first == anyRefT) {
                  bool useRoot = (useObjRoot == 1 && canonHasObjRootAt[cPos]) ||
                                 (useObjRoot == 2 && canonAllObjRootAt[cPos]);
                  Type ft = useRoot ? Type(*commonObjRoot, Nullable) : anyRefT;
                  synth[cPos] = {ft, pt};
                } else {
                  synth[cPos] = skel.fields[cPos];
                }
              }
              for (size_t mi = 0; mi < members.size(); ++mi) {
                CandShape memSynth(numF);
                for (Index cPos = 0; cPos < numF; ++cPos) {
                  memSynth[canonOrder[mi][cPos]] = synth[cPos];
                }
                addCandShape(memSynth);
              }
              addCandShape(synth);
            }
          }

          if (candShapes.size() <= 1) {
            continue;
          }

          std::vector<int> sDefBytes(candShapes.size());
          for (size_t si = 0; si < candShapes.size(); ++si) {
            sDefBytes[si] = getShapeDefBytes(skel.super, candShapes[si]);
          }

          struct ReachEntry {
            Index si;
            int castCost;
            std::vector<Index> oldToNew;
            std::vector<Index> newToOld;
            bool isIdentityPerm;
          };

          std::vector<Index> idPerm(numF);
          for (Index fi = 0; fi < numF; ++fi) {
            idPerm[fi] = fi;
          }

          // Precompute reachability, best valid non-inherited field
          // permutation, and cast cost for each member -> shape.
          std::vector<std::vector<ReachEntry>> reach(members.size());
          for (size_t mi = 0; mi < members.size(); ++mi) {
            HeapType M = members[mi];
            reach[mi].push_back({mCurShapeIdx[mi], 0, idPerm, idPerm, true});
            if (!subTypes.getImmediateSubTypes(M).empty()) {
              continue;
            }
            const auto& mFields = M.getStruct().fields;
            std::vector<bool> canWidenFi(numF, false);
            std::vector<Type> realFTs(numF);
            for (Index fi = 0; fi < numF; ++fi) {
              realFTs[fi] = getFinalFieldType(M, fi);
              bool mutOk = false;
              if (mFields[fi].mutable_ == Immutable &&
                  getEffMut(M, fi) == Immutable) {
                mutOk = true;
              } else if (fi >= supLen && getHighestFieldAncestor(M, fi) == M) {
                auto fIt = fieldCandidates.find(FieldKey{M, fi});
                mutOk = (fIt != fieldCandidates.end() && !fIt->second.pinned);
              }
              if (mutOk && !mFields[fi].isPacked() && realFTs[fi].isRef() &&
                  realFTs[fi].isNullable() && !realFTs[fi].isExact() &&
                  !realFTs[fi].isNull() &&
                  !castWidenedImmFields.count(FieldKey{M, fi})) {
                canWidenFi[fi] = true;
              }
            }

            auto fieldMatchCost = [&](Index oldFi,
                                      const CandField& tCF) -> int {
              if (mCurShapes[mi][oldFi].second != tCF.second) {
                return -1;
              }
              if (mCurShapes[mi][oldFi].first == tCF.first) {
                return 0;
              }
              if (!canWidenFi[oldFi] ||
                  !Type::isSubType(realFTs[oldFi], tCF.first) ||
                  tCF.first.isExact() || tCF.first.isNull()) {
                return -1;
              }
              if (oldFi < supLen &&
                  (!Type::isSubType(tCF.first,
                                    getFinalFieldType(*skel.super, oldFi)) ||
                   (M.getDeclaredSuperType() &&
                    !Type::isSubType(
                      tCF.first,
                      getFinalFieldType(*M.getDeclaredSuperType(), oldFi))))) {
                return -1;
              }
              int narrowGets = countNarrowGets({M}, oldFi, tCF.first);
              int castBytes =
                (typeIndex[realFTs[oldFi].getHeapType()] < 64) ? 3 : 4;
              return narrowGets * castBytes;
            };

            for (Index si = 0; si < candShapes.size(); ++si) {
              if (si == mCurShapeIdx[mi]) {
                continue;
              }
              const auto& targetS = candShapes[si];
              bool inhOk = true;
              int inhCastCost = 0;
              for (Index fi = 0; fi < supLen; ++fi) {
                int c = fieldMatchCost(fi, targetS[fi]);
                if (c < 0) {
                  inhOk = false;
                  break;
                }
                inhCastCost += c;
              }
              if (!inhOk) {
                continue;
              }

              // First check if identity permutation works; also search general
              // bucket-matched permutation if non-inherited fields exist.
              std::vector<Index> oldToNew(numF, Index(-1));
              std::vector<Index> newToOld(numF, Index(-1));
              for (Index fi = 0; fi < supLen; ++fi) {
                oldToNew[fi] = fi;
                newToOld[fi] = fi;
              }

              // Group non-inherited indices by coarse CandField.
              struct Bucket {
                CandField coarse;
                std::vector<Index> olds;
                std::vector<Index> news;
              };
              std::vector<Bucket> buckets;
              for (Index oldFi = supLen; oldFi < numF; ++oldFi) {
                CandField cf = mCoarse[mi][oldFi];
                bool found = false;
                for (auto& b : buckets) {
                  if (b.coarse == cf) {
                    b.olds.push_back(oldFi);
                    found = true;
                    break;
                  }
                }
                if (!found) {
                  buckets.push_back({cf, {oldFi}, {}});
                }
              }
              bool bucketsMatch = true;
              for (Index newFi = supLen; newFi < numF; ++newFi) {
                Type ft = targetS[newFi].first;
                Field::PackedType pt = targetS[newFi].second;
                Type coarseFT = (!pt && ft.isRef() && ft.isNullable() &&
                                 !ft.isExact() && !ft.isNull())
                                  ? Type(ft.getHeapType().getTop(), Nullable)
                                  : ft;
                CandField cf{coarseFT, pt};
                bool found = false;
                for (auto& b : buckets) {
                  if (b.coarse == cf) {
                    b.news.push_back(newFi);
                    found = true;
                    break;
                  }
                }
                if (!found) {
                  bucketsMatch = false;
                  break;
                }
              }
              if (!bucketsMatch) {
                continue;
              }
              for (const auto& b : buckets) {
                if (b.olds.size() != b.news.size()) {
                  bucketsMatch = false;
                  break;
                }
              }
              if (!bucketsMatch) {
                continue;
              }

              int totalCastCost = inhCastCost;
              std::vector<size_t> heteroBucketIndices;
              for (size_t bi = 0; bi < buckets.size(); ++bi) {
                const auto& b = buckets[bi];
                size_t k = b.olds.size();
                bool homoTarget = true;
                for (size_t s = 1; s < k; ++s) {
                  if (targetS[b.news[s]] != targetS[b.news[0]]) {
                    homoTarget = false;
                    break;
                  }
                }
                if (homoTarget || k == 1 || k > 7) {
                  for (size_t r = 0; r < k; ++r) {
                    int c = fieldMatchCost(b.olds[r], targetS[b.news[r]]);
                    if (c < 0) {
                      bucketsMatch = false;
                      break;
                    }
                    oldToNew[b.olds[r]] = b.news[r];
                    newToOld[b.news[r]] = b.olds[r];
                    totalCastCost += c;
                  }
                  if (!bucketsMatch) {
                    break;
                  }
                } else {
                  heteroBucketIndices.push_back(bi);
                }
              }
              if (!bucketsMatch) {
                continue;
              }

              // Verify canSwap among already-locked pairs.
              for (Index a = supLen; a < numF; ++a) {
                if (oldToNew[a] == Index(-1)) {
                  continue;
                }
                for (Index b = a + 1; b < numF; ++b) {
                  if (oldToNew[b] == Index(-1)) {
                    continue;
                  }
                  if (oldToNew[a] > oldToNew[b] && !canSwap[mi][a * numF + b]) {
                    bucketsMatch = false;
                    break;
                  }
                }
                if (!bucketsMatch) {
                  break;
                }
              }
              if (!bucketsMatch) {
                continue;
              }

              for (size_t bi : heteroBucketIndices) {
                const auto& b = buckets[bi];
                size_t k = b.olds.size();
                std::vector<std::vector<int>> bCost(k, std::vector<int>(k, -1));
                for (size_t r = 0; r < k; ++r) {
                  for (size_t s = 0; s < k; ++s) {
                    bCost[r][s] = fieldMatchCost(b.olds[r], targetS[b.news[s]]);
                  }
                }
                int bestBCost = 1000000000;
                int bestBInvs = 1000000000;
                std::vector<size_t> bestAssign;
                std::vector<size_t> curAssign(k, 0);

                auto solveBucket = [&](auto& self,
                                       size_t r,
                                       uint32_t usedMask,
                                       int curCost,
                                       int curInvs) -> void {
                  if (curCost > bestBCost ||
                      (curCost == bestBCost && curInvs >= bestBInvs)) {
                    return;
                  }
                  if (r == k) {
                    bestBCost = curCost;
                    bestBInvs = curInvs;
                    bestAssign = curAssign;
                    return;
                  }
                  Index oldR = b.olds[r];
                  for (size_t s = 0; s < k; ++s) {
                    if ((usedMask & (1u << s)) || bCost[r][s] < 0) {
                      continue;
                    }
                    Index newS = b.news[s];
                    bool swapOk = true;
                    int addInvs = (oldR != newS) ? 1 : 0;
                    for (Index otherOld = supLen; otherOld < numF; ++otherOld) {
                      Index otherNew = oldToNew[otherOld];
                      if (otherNew == Index(-1)) {
                        continue;
                      }
                      if (otherOld < oldR && otherNew > newS) {
                        if (!canSwap[mi][otherOld * numF + oldR]) {
                          swapOk = false;
                          break;
                        }
                        addInvs++;
                      } else if (oldR < otherOld && newS > otherNew) {
                        if (!canSwap[mi][oldR * numF + otherOld]) {
                          swapOk = false;
                          break;
                        }
                        addInvs++;
                      }
                    }
                    if (!swapOk) {
                      continue;
                    }
                    curAssign[r] = s;
                    oldToNew[oldR] = newS;
                    self(self,
                         r + 1,
                         usedMask | (1u << s),
                         curCost + bCost[r][s],
                         curInvs + addInvs);
                    oldToNew[oldR] = Index(-1);
                  }
                };
                solveBucket(solveBucket, 0, 0, 0, 0);
                if (bestAssign.empty()) {
                  bucketsMatch = false;
                  break;
                }
                for (size_t r = 0; r < k; ++r) {
                  Index oldR = b.olds[r];
                  Index newS = b.news[bestAssign[r]];
                  oldToNew[oldR] = newS;
                  newToOld[newS] = oldR;
                }
                totalCastCost += bestBCost;
              }
              if (!bucketsMatch) {
                continue;
              }

              bool isId = true;
              for (Index fi = supLen; fi < numF; ++fi) {
                if (oldToNew[fi] != fi) {
                  isId = false;
                  break;
                }
              }
              reach[mi].push_back({si,
                                   totalCastCost,
                                   std::move(oldToNew),
                                   std::move(newToOld),
                                   isId});
            }
          }

          struct ReachMember {
            size_t mi;
            int castCost;
            bool stationary;
            Index tIdx;
            std::vector<Index> oldToNew;
            std::vector<Index> newToOld;
            bool isIdentityPerm;
          };

          std::vector<bool> usedMember(members.size(), false);
          while (true) {
            int bestClusterNet = 0;
            int bestIdPermCount = -1;
            Index bestSi = Index(-1);
            std::vector<ReachMember> bestCluster;
            std::vector<bool> bestReqMut;

            for (Index si = 0; si < candShapes.size(); ++si) {
              const auto& targetS = candShapes[si];
              std::vector<ReachMember> rMembers;
              for (size_t mi = 0; mi < members.size(); ++mi) {
                if (usedMember[mi]) {
                  continue;
                }
                for (const auto& re : reach[mi]) {
                  if (re.si == si) {
                    rMembers.push_back(
                      {mi,
                       re.castCost,
                       mCurShapeIdx[mi] == si && re.isIdentityPerm,
                       typeIndex[members[mi]],
                       re.oldToNew,
                       re.newToOld,
                       re.isIdentityPerm});
                    break;
                  }
                }
              }
              if (rMembers.size() < 2) {
                continue;
              }
              std::stable_sort(
                rMembers.begin(),
                rMembers.end(),
                [](const ReachMember& a, const ReachMember& b) {
                  if (a.stationary != b.stationary) {
                    return a.stationary > b.stationary;
                  }
                  return a.tIdx < b.tIdx;
                });

              struct MutSubGroup {
                std::vector<ReachMember> rms;
                std::vector<bool> reqMut;
                std::vector<bool> allowMut;
              };
              std::vector<MutSubGroup> mSubGroups;
              for (const auto& rm : rMembers) {
                size_t mi = rm.mi;
                HeapType M = members[mi];
                std::vector<bool> reqM(numF, false);
                std::vector<bool> allowM(numF, false);
                for (Index newFi = 0; newFi < numF; ++newFi) {
                  Index oldFi = rm.newToOld[newFi];
                  bool isMut = (getEffMut(M, oldFi) == Mutable);
                  reqM[newFi] = isMut;
                  if (mCurShapes[mi][oldFi] == targetS[newFi]) {
                    allowM[newFi] = isMut || canMakeMut(M, oldFi);
                  } else {
                    allowM[newFi] =
                      isMut || (getHighestFieldAncestor(M, oldFi) == M &&
                                !allPinnedFields.count(FieldKey{M, oldFi}));
                  }
                }
                bool placed = false;
                for (auto& msg : mSubGroups) {
                  bool compat = true;
                  for (Index fi = 0; fi < numF; ++fi) {
                    bool cReq = msg.reqMut[fi] || reqM[fi];
                    bool cAllow = msg.allowMut[fi] && allowM[fi];
                    if (cReq && !cAllow) {
                      compat = false;
                      break;
                    }
                  }
                  if (compat) {
                    msg.rms.push_back(rm);
                    for (Index fi = 0; fi < numF; ++fi) {
                      msg.reqMut[fi] = msg.reqMut[fi] || reqM[fi];
                      msg.allowMut[fi] = msg.allowMut[fi] && allowM[fi];
                    }
                    placed = true;
                    break;
                  }
                }
                if (!placed) {
                  mSubGroups.push_back(
                    {{rm}, std::move(reqM), std::move(allowM)});
                }
              }

              std::vector<size_t> origShapeTotalCount(candShapes.size(), 0);
              for (size_t mi = 0; mi < members.size(); ++mi) {
                origShapeTotalCount[mCurShapeIdx[mi]]++;
              }

              for (auto& msg : mSubGroups) {
                auto cluster = msg.rms;
                while (cluster.size() >= 2) {
                  Index cIdx = Index(-1);
                  std::vector<size_t> origShapeInCluster(candShapes.size(), 0);
                  std::vector<int> origShapeNet(candShapes.size(), 0);
                  for (const auto& rm : cluster) {
                    cIdx = std::min(cIdx, rm.tIdx);
                    origShapeInCluster[mCurShapeIdx[rm.mi]]++;
                  }
                  for (const auto& rm : cluster) {
                    if (!rm.stationary) {
                      Index origSi = mCurShapeIdx[rm.mi];
                      origShapeNet[origSi] +=
                        getLebSavings(members[rm.mi], cIdx) - rm.castCost;
                    }
                  }
                  for (Index origSi = 0; origSi < candShapes.size(); ++origSi) {
                    if (origSi != si && origShapeInCluster[origSi] > 0 &&
                        origShapeInCluster[origSi] ==
                          origShapeTotalCount[origSi]) {
                      origShapeNet[origSi] += sDefBytes[origSi];
                    }
                  }
                  bool pruned = false;
                  for (auto it = cluster.begin(); it != cluster.end();) {
                    if (!it->stationary) {
                      Index origSi = mCurShapeIdx[it->mi];
                      int mNet = getLebSavings(members[it->mi], cIdx) -
                                 it->castCost;
                      if (origShapeInCluster[origSi] ==
                          origShapeTotalCount[origSi]) {
                        if (origShapeNet[origSi] <= 0) {
                          it = cluster.erase(it);
                          pruned = true;
                          continue;
                        }
                      } else if (mNet <= 0) {
                        it = cluster.erase(it);
                        pruned = true;
                        continue;
                      }
                    }
                    ++it;
                  }
                  if (!pruned) {
                    break;
                  }
                }
                if (cluster.size() < 2) {
                  continue;
                }
                bool anyWidened = false;
                bool hasStationary = false;
                int idPermCount = 0;
                Index cIdx = Index(-1);
                std::vector<size_t> origShapeInCluster(candShapes.size(), 0);
                for (const auto& rm : cluster) {
                  if (rm.stationary) {
                    hasStationary = true;
                  } else {
                    anyWidened = true;
                  }
                  if (rm.isIdentityPerm) {
                    idPermCount++;
                  }
                  cIdx = std::min(cIdx, rm.tIdx);
                  origShapeInCluster[mCurShapeIdx[rm.mi]]++;
                }
                if (!anyWidened) {
                  continue;
                }
                int sgNet = 0;
                for (const auto& rm : cluster) {
                  if (!rm.stationary) {
                    HeapType M = members[rm.mi];
                    sgNet += getLebSavings(M, cIdx) - rm.castCost;
                  }
                }
                for (Index origSi = 0; origSi < candShapes.size(); ++origSi) {
                  if (origSi != si && origShapeInCluster[origSi] > 0 &&
                      origShapeInCluster[origSi] ==
                        origShapeTotalCount[origSi]) {
                    sgNet += sDefBytes[origSi];
                  }
                }
                if (!hasStationary) {
                  sgNet -= sDefBytes[si];
                }
                if (sgNet > bestClusterNet ||
                    (sgNet > 0 && sgNet == bestClusterNet &&
                     idPermCount > bestIdPermCount)) {
                  bestClusterNet = sgNet;
                  bestIdPermCount = idPermCount;
                  bestSi = si;
                  bestCluster = cluster;
                  std::vector<bool> cReqMut(numF, false);
                  for (const auto& rm : cluster) {
                    for (Index newFi = 0; newFi < numF; ++newFi) {
                      Index oldFi = rm.newToOld[newFi];
                      if (getEffMut(members[rm.mi], oldFi) == Mutable) {
                        cReqMut[newFi] = true;
                      }
                    }
                  }
                  bestReqMut = std::move(cReqMut);
                }
              }
            }

            if (bestClusterNet <= 0) {
              break;
            }
            const auto& targetS = candShapes[bestSi];
            for (const auto& rm : bestCluster) {
              size_t mi = rm.mi;
              usedMember[mi] = true;
              HeapType M = members[mi];
              usedInStepC.insert(M);
              if (rm.isIdentityPerm) {
                for (Index fi = 0; fi < numF; ++fi) {
                  if (bestReqMut[fi] && getEffMut(M, fi) == Immutable) {
                    HeapType root = getHighestFieldAncestor(M, fi);
                    setRootSubtreeMut(root, fi);
                  }
                  if (mCurShapes[mi][fi] == targetS[fi]) {
                    continue;
                  }
                  Type targetFT = targetS[fi].first;
                  finalImmFieldTypes[FieldKey{M, fi}] = targetFT;
                  castWidenedImmFields[FieldKey{M, fi}] =
                    M.getStruct().fields[fi].type;
                  auto& vec = structUpdates[M];
                  bool found = false;
                  for (auto& [ufi, uFT] : vec) {
                    if (ufi == fi) {
                      uFT = targetFT;
                      found = true;
                      break;
                    }
                  }
                  if (!found && targetFT != M.getStruct().fields[fi].type) {
                    vec.push_back({fi, targetFT});
                  }
                }
              } else {
                for (Index fi = 0; fi < supLen; ++fi) {
                  if (bestReqMut[fi] && getEffMut(M, fi) == Immutable) {
                    HeapType root = getHighestFieldAncestor(M, fi);
                    setRootSubtreeMut(root, fi);
                  }
                }
                StructFieldLayoutUpdate layout;
                layout.oldToNew = rm.oldToNew;
                layout.newToOld = rm.newToOld;
                layout.newFields.resize(numF);
                const auto& mFields = M.getStruct().fields;
                for (Index newFi = 0; newFi < numF; ++newFi) {
                  Index oldFi = rm.newToOld[newFi];
                  Field f = mFields[oldFi];
                  f.mutable_ =
                    (bestReqMut[newFi] || getEffMut(M, oldFi) == Mutable)
                      ? Mutable
                      : Immutable;
                  if (mCurShapes[mi][oldFi] == targetS[newFi]) {
                    f.type = getFinalFieldType(M, oldFi);
                  } else {
                    f.type = targetS[newFi].first;
                    castWidenedImmFields[FieldKey{M, oldFi}] =
                      mFields[oldFi].type;
                  }
                  layout.newFields[newFi] = f;
                }
                structLayoutUpdates[M] = std::move(layout);
              }
            }
          }
        }
      }

      // Component 1: Order-Preserving Subsequence Field Padding (1..3 extra
      // fields, ZERO operand reordering) for remaining singleton uncast leaf
      // structs.
      if (allowSiblingCastWiden) {
        struct PaddingTargetGroup {
          std::optional<HeapType> super;
          Index supLen = 0;
          std::vector<HeapType> members;
          FieldList rawFields;
          FieldList fields;
          std::vector<bool> reqMut;
          std::vector<bool> allowMut;
          Index cIdx = Index(-1);
          bool wasPadded = false;
          bool hasPaddedChild = false;
          int nonInhI32 = 0;
          int nonInhI64 = 0;
          int nonInhF32 = 0;
          int nonInhF64 = 0;
          int nonInhRef = 0;
        };

        std::vector<StructShapeKey> padShapeOrder;
        std::unordered_map<StructShapeKey,
                           std::vector<HeapType>,
                           StructShapeKeyHash>
          padShapeGroups;
        for (auto it = subTypesFirst.rbegin(); it != subTypesFirst.rend();
             ++it) {
          HeapType T = *it;
          if (!isCleanPrivateStruct(T) || mergedIntoSuper.count(T) ||
              collapsedParents.count(T) ||
              !subTypes.getImmediateSubTypes(T).empty()) {
            continue;
          }
          StructShapeKey key;
          if (auto s = T.getDeclaredSuperType()) {
            key.super = mapMergedHT(*s);
          }
          Index numF = getPostStepCNumFields(T);
          key.fields.reserve(numF);
          for (Index fi = 0; fi < numF; ++fi) {
            Field f = getPostStepCField(T, fi);
            key.fields.push_back({mapMergedType(f.type), f.packedType});
          }
          auto& vec = padShapeGroups[key];
          if (vec.empty()) {
            padShapeOrder.push_back(key);
          }
          vec.push_back(T);
        }

        std::vector<PaddingTargetGroup> tGroups;
        for (const auto& key : padShapeOrder) {
          const auto& members = padShapeGroups[key];
          Index numF = key.fields.size();
          Index supLen =
            (key.super && key.super->isStruct())
              ? Index(key.super->getStruct().fields.size())
              : 0;
          for (auto T : members) {
            std::vector<bool> tReq(numF, false);
            std::vector<bool> tAllow(numF, false);
            for (Index fi = 0; fi < numF; ++fi) {
              Field f = getPostStepCField(T, fi);
              bool isMut = (f.mutable_ == Mutable);
              tReq[fi] = isMut;
              tAllow[fi] = isMut || canMakePostStepCMut(T, fi);
            }
            bool placed = false;
            for (auto& tg : tGroups) {
              if (tg.super != key.super || tg.fields.size() != numF) {
                continue;
              }
              bool sameShape = true;
              for (Index fi = 0; fi < numF; ++fi) {
                if (tg.fields[fi].type != key.fields[fi].first ||
                    tg.fields[fi].packedType != key.fields[fi].second) {
                  sameShape = false;
                  break;
                }
                bool cReq = tg.reqMut[fi] || tReq[fi];
                bool cAllow = tg.allowMut[fi] && tAllow[fi];
                if (cReq && !cAllow) {
                  sameShape = false;
                  break;
                }
              }
              if (sameShape) {
                tg.members.push_back(T);
                tg.cIdx = std::min(tg.cIdx, typeIndex[T]);
                for (Index fi = 0; fi < numF; ++fi) {
                  tg.reqMut[fi] = tg.reqMut[fi] || tReq[fi];
                  tg.allowMut[fi] = tg.allowMut[fi] && tAllow[fi];
                }
                placed = true;
                break;
              }
            }
            if (!placed) {
              PaddingTargetGroup tg;
              tg.super = key.super;
              tg.supLen = supLen;
              tg.members = {T};
              tg.rawFields.resize(numF);
              tg.fields.resize(numF);
              for (Index fi = 0; fi < numF; ++fi) {
                Field rf = getPostStepCField(T, fi);
                tg.rawFields[fi] = rf;
                Field mf = rf;
                mf.type = key.fields[fi].first;
                tg.fields[fi] = mf;
                if (fi >= supLen) {
                  Type ft = mf.type;
                  if (ft == Type::i32) {
                    tg.nonInhI32++;
                  } else if (ft == Type::i64) {
                    tg.nonInhI64++;
                  } else if (ft == Type::f32) {
                    tg.nonInhF32++;
                  } else if (ft == Type::f64) {
                    tg.nonInhF64++;
                  } else if (ft.isRef()) {
                    tg.nonInhRef++;
                  }
                }
              }
              tg.reqMut = std::move(tReq);
              tg.allowMut = std::move(tAllow);
              tg.cIdx = typeIndex[T];
              tGroups.push_back(std::move(tg));
            }
          }
        }

        std::unordered_map<
          std::optional<HeapType>,
          std::unordered_map<Index, std::vector<size_t>>>
          groupsBySuperAndLen;
        std::vector<size_t> candGroupIndices;
        for (size_t gi = 0; gi < tGroups.size(); ++gi) {
          const auto& tg = tGroups[gi];
          groupsBySuperAndLen[tg.super][Index(tg.fields.size())].push_back(gi);
          if (tg.members.size() == 1) {
            HeapType M = tg.members[0];
            if (!usedInStepC.count(M) && !structLayoutUpdates.count(M) &&
                !hasPinnedField(M)) {
              candGroupIndices.push_back(gi);
            }
          }
        }

        std::stable_sort(candGroupIndices.begin(),
                         candGroupIndices.end(),
                         [&](size_t a, size_t b) {
                           size_t lenA = tGroups[a].fields.size();
                           size_t lenB = tGroups[b].fields.size();
                           if (lenA != lenB) {
                             return lenA > lenB;
                           }
                           return tGroups[a].cIdx > tGroups[b].cIdx;
                         });

        for (size_t mGi : candGroupIndices) {
          auto& mGroup = tGroups[mGi];
          if (mGroup.wasPadded || mGroup.hasPaddedChild ||
              mGroup.members.size() != 1) {
            continue;
          }
          HeapType M = mGroup.members[0];
          const auto& mFields = M.getStruct().fields;
          Index lenM = mFields.size();
          Index supLen = mGroup.supLen;
          if (lenM < supLen) {
            continue;
          }
          Index K = lenM - supLen;
          int mNews = getNonDefaultStructNewCount(M);
          int defSave = getStructDefBytes(M);

          std::vector<bool> canWidenFi(lenM, false);
          std::vector<Type> realFTs(lenM);
          for (Index fi = 0; fi < lenM; ++fi) {
            realFTs[fi] = getFinalFieldType(M, fi);
            bool mutOk = false;
            if (mFields[fi].mutable_ == Immutable &&
                getEffMut(M, fi) == Immutable) {
              mutOk = true;
            } else if (fi >= supLen && getHighestFieldAncestor(M, fi) == M) {
              auto fIt = fieldCandidates.find(FieldKey{M, fi});
              mutOk = (fIt != fieldCandidates.end() && !fIt->second.pinned);
            }
            if (mutOk && !mFields[fi].isPacked() && realFTs[fi].isRef() &&
                realFTs[fi].isNullable() && !realFTs[fi].isExact() &&
                !realFTs[fi].isNull() &&
                !castWidenedImmFields.count(FieldKey{M, fi})) {
              canWidenFi[fi] = true;
            }
          }

          int bestNet = 0;
          size_t bestTGi = size_t(-1);
          std::vector<Index> bestOldToNew;
          std::vector<Index> bestNewToOld;

          std::vector<bool> canSwapM;
          bool computedCanSwapM = false;
          bool anyCanSwapM = false;
          auto ensureCanSwapM = [&]() {
            if (computedCanSwapM) {
              return;
            }
            computedCanSwapM = true;
            if (K < 2 || K > 8) {
              return;
            }
            canSwapM.assign(lenM * lenM, false);
            for (Index a = supLen; a < lenM; ++a) {
              for (Index b = a + 1; b < lenM; ++b) {
                canSwapM[a * lenM + b] = true;
              }
            }
            auto nIt = allStructNews.find(M);
            if (nIt != allStructNews.end()) {
              for (auto* snew : nIt->second) {
                if (snew->isWithDefault() || snew->operands.size() != lenM) {
                  continue;
                }
                std::vector<EffectAnalyzer> eas;
                eas.reserve(K);
                for (Index fi = supLen; fi < lenM; ++fi) {
                  eas.emplace_back(
                    getPassOptions(), *module, snew->operands[fi]);
                }
                for (Index a = supLen; a < lenM; ++a) {
                  for (Index b = a + 1; b < lenM; ++b) {
                    if (canSwapM[a * lenM + b]) {
                      const auto& eaA = eas[a - supLen];
                      const auto& eaB = eas[b - supLen];
                      if (eaA.orderedBefore(eaB) || eaB.orderedBefore(eaA)) {
                        canSwapM[a * lenM + b] = false;
                      }
                    }
                  }
                }
              }
            }
            for (Index a = supLen; a < lenM; ++a) {
              for (Index b = a + 1; b < lenM; ++b) {
                if (canSwapM[a * lenM + b]) {
                  anyCanSwapM = true;
                  return;
                }
              }
            }
          };

          auto evalNonInhMatchCost = [&]( const PaddingTargetGroup& tGroup,
                                          Index oldIdx,
                                          Index newIdx) -> int {
            if (mFields[oldIdx].packedType != tGroup.fields[newIdx].packedType) {
              return -1;
            }
            bool mReq = mGroup.reqMut[oldIdx];
            bool mAllow = mGroup.allowMut[oldIdx];
            Type mFT = mGroup.fields[oldIdx].type;
            Type tFT = tGroup.fields[newIdx].type;
            if (mFT != tFT && canWidenFi[oldIdx]) {
              mAllow = true;
            }
            if ((mReq || tGroup.reqMut[newIdx]) &&
                !(mAllow && tGroup.allowMut[newIdx])) {
              return -1;
            }
            if (mFT == tFT) {
              return 0;
            }
            if (canWidenFi[oldIdx] && !tFT.isExact() && !tFT.isNull() &&
                Type::isSubType(realFTs[oldIdx], tFT)) {
              int narrowGets = countNarrowGets({M}, oldIdx, tFT);
              int castBytes =
                (typeIndex[realFTs[oldIdx].getHeapType()] < 64) ? 3 : 4;
              return narrowGets * castBytes;
            }
            return -1;
          };

          auto supIt = groupsBySuperAndLen.find(mGroup.super);
          if (supIt == groupsBySuperAndLen.end()) {
            continue;
          }
          for (Index diff = 1; diff <= 3; ++diff) {
            Index lenT = lenM + diff;
            auto lenIt = supIt->second.find(lenT);
            if (lenIt == supIt->second.end()) {
              continue;
            }
            Index N = lenT - supLen;
            Index D = diff;
            for (size_t tGi : lenIt->second) {
              const auto& tGroup = tGroups[tGi];
              if (tGroup.wasPadded) {
                continue;
              }
              if (mGroup.nonInhI32 > tGroup.nonInhI32 ||
                  mGroup.nonInhI64 > tGroup.nonInhI64 ||
                  mGroup.nonInhF32 > tGroup.nonInhF32 ||
                  mGroup.nonInhF64 > tGroup.nonInhF64 ||
                  mGroup.nonInhRef > tGroup.nonInhRef) {
                continue;
              }

              // Check inherited fields 0..supLen-1.
              bool inhOk = true;
              int inhCastCost = 0;
              for (Index fi = 0; fi < supLen; ++fi) {
                if (mFields[fi].packedType != tGroup.fields[fi].packedType) {
                  inhOk = false;
                  break;
                }
                bool mReq = mGroup.reqMut[fi];
                bool mAllow = mGroup.allowMut[fi];
                if ((mReq || tGroup.reqMut[fi]) &&
                    !(mAllow && tGroup.allowMut[fi])) {
                  inhOk = false;
                  break;
                }
                Type mFT = mGroup.fields[fi].type;
                Type tFT = tGroup.fields[fi].type;
                if (mFT != tFT) {
                  if (!canWidenFi[fi] || tFT.isExact() || tFT.isNull() ||
                      !Type::isSubType(realFTs[fi], tFT) ||
                      !Type::isSubType(tFT,
                                       getFinalFieldType(*mGroup.super, fi)) ||
                      (M.getDeclaredSuperType() &&
                       !Type::isSubType(
                         tFT,
                         getFinalFieldType(*M.getDeclaredSuperType(), fi)))) {
                    inhOk = false;
                    break;
                  }
                  int narrowGets = countNarrowGets({M}, fi, tFT);
                  int castBytes =
                    (typeIndex[realFTs[fi].getHeapType()] < 64) ? 3 : 4;
                  inhCastCost += narrowGets * castBytes;
                }
              }
              if (!inhOk) {
                continue;
              }

              // DP over (i in 0..K, d in 0..D) where j = i + d in 0..N.
              const int INF = 1000000000;
              std::vector<std::array<int, 4>> dp(
                K + 1, {INF, INF, INF, INF});
              // parentMove[i][d]: 1 = came from Match (i-1, d), 2 = came from
              // Pad (i, d-1).
              std::vector<std::array<uint8_t, 4>> parentMove(
                K + 1, {0, 0, 0, 0});
              dp[0][0] = 0;

              for (Index i = 0; i <= K; ++i) {
                for (Index d = 0; d <= D; ++d) {
                  int cur = dp[i][d];
                  if (cur >= INF) {
                    continue;
                  }
                  Index j = i + d;
                  // Option 1: Pad T[supLen + j].
                  if (d < D && j < N) {
                    Index newIdx = supLen + j;
                    int cb = getDefaultConstBytes(tGroup.fields[newIdx].type);
                    if (cb >= 0) {
                      int padStep = mNews * cb;
                      if (cur + padStep < dp[i][d + 1]) {
                        dp[i][d + 1] = cur + padStep;
                        parentMove[i][d + 1] = 2;
                      }
                    }
                  }
                  // Option 2: Match M[supLen + i] -> T[supLen + j].
                  if (i < K && j < N) {
                    Index oldIdx = supLen + i;
                    Index newIdx = supLen + j;
                    int mCost = evalNonInhMatchCost(tGroup, oldIdx, newIdx);
                    if (mCost >= 0 && cur + mCost < dp[i + 1][d]) {
                      dp[i + 1][d] = cur + mCost;
                      parentMove[i + 1][d] = 1;
                    }
                  }
                }
              }

              int bestNonInhCost = dp[K][D];
              std::vector<Index> permAssign;
              if (bestNonInhCost > 0 && K >= 2 && K <= 8) {
                ensureCanSwapM();
                if (anyCanSwapM) {
                  std::vector<int> padCostJ(N, -1);
                  int nonPadCount = 0;
                  for (Index j = 0; j < N; ++j) {
                    int cb =
                      getDefaultConstBytes(tGroup.fields[supLen + j].type);
                    if (cb >= 0) {
                      padCostJ[j] = mNews * cb;
                    } else {
                      nonPadCount++;
                    }
                  }
                  if (nonPadCount <= int(K)) {
                    std::vector<std::vector<std::pair<Index, int>>> compatJ(K);
                    bool allHaveCompat = true;
                    for (Index i = 0; i < K; ++i) {
                      for (Index j = 0; j < N; ++j) {
                        int mc =
                          evalNonInhMatchCost(tGroup, supLen + i, supLen + j);
                        if (mc >= 0) {
                          compatJ[i].push_back({j, mc});
                        }
                      }
                      if (compatJ[i].empty()) {
                        allHaveCompat = false;
                        break;
                      }
                    }
                    if (allHaveCompat) {
                      std::vector<Index> curAssign(K, 0);
                      int bestInvs = INF;
                      auto solvePermPad = [&](auto& self,
                                              Index i,
                                              uint32_t usedMask,
                                              int curMatchCost,
                                              int curInvs) -> void {
                        if (curMatchCost >= bestNonInhCost) {
                          return;
                        }
                        if (i == K) {
                          int padCost = 0;
                          for (Index j = 0; j < N; ++j) {
                            if (!(usedMask & (1u << j))) {
                              if (padCostJ[j] < 0) {
                                return;
                              }
                              padCost += padCostJ[j];
                            }
                          }
                          int tot = curMatchCost + padCost;
                          if (tot < bestNonInhCost ||
                              (tot == bestNonInhCost && curInvs < bestInvs)) {
                            bestNonInhCost = tot;
                            bestInvs = curInvs;
                            permAssign = curAssign;
                          }
                          return;
                        }
                        Index oldI = supLen + i;
                        for (const auto& [j, mc] : compatJ[i]) {
                          if (usedMask & (1u << j)) {
                            continue;
                          }
                          if (curMatchCost + mc >= bestNonInhCost) {
                            continue;
                          }
                          bool swapOk = true;
                          int addInvs = 0;
                          for (Index prevI = 0; prevI < i; ++prevI) {
                            if (curAssign[prevI] > j) {
                              if (!canSwapM[(supLen + prevI) * lenM + oldI]) {
                                swapOk = false;
                                break;
                              }
                              addInvs++;
                            }
                          }
                          if (!swapOk) {
                            continue;
                          }
                          curAssign[i] = j;
                          self(self,
                               i + 1,
                               usedMask | (1u << j),
                               curMatchCost + mc,
                               curInvs + addInvs);
                        }
                      };
                      solvePermPad(solvePermPad, 0, 0, 0, 0);
                    }
                  }
                }
              }

              if (bestNonInhCost >= INF) {
                continue;
              }
              int totalCost = inhCastCost + bestNonInhCost;
              Index cIdx = std::min(typeIndex[M], tGroup.cIdx);
              int lebSave = getLebSavings(M, cIdx);
              if (cIdx < tGroup.cIdx) {
                for (auto U : tGroup.members) {
                  lebSave += getLebSavings(U, cIdx);
                }
              }
              int net = defSave + lebSave - totalCost;
              if (net > bestNet) {
                bestNet = net;
                bestTGi = tGi;
                bestOldToNew.assign(lenM, Index(-1));
                bestNewToOld.assign(lenT, Index(-1));
                for (Index fi = 0; fi < supLen; ++fi) {
                  bestOldToNew[fi] = fi;
                  bestNewToOld[fi] = fi;
                }
                if (!permAssign.empty()) {
                  for (Index i = 0; i < K; ++i) {
                    Index oldIdx = supLen + i;
                    Index newIdx = supLen + permAssign[i];
                    bestOldToNew[oldIdx] = newIdx;
                    bestNewToOld[newIdx] = oldIdx;
                  }
                } else {
                  Index ci = K;
                  Index cd = D;
                  while (ci > 0 || cd > 0) {
                    uint8_t mv = parentMove[ci][cd];
                    assert(mv == 1 || mv == 2);
                    if (mv == 1) {
                      ci--;
                      Index oldIdx = supLen + ci;
                      Index newIdx = supLen + ci + cd;
                      bestOldToNew[oldIdx] = newIdx;
                      bestNewToOld[newIdx] = oldIdx;
                    } else {
                      cd--;
                    }
                  }
                }
              }
            }
          }

          if (bestNet <= 0) {
            continue;
          }

          auto& bestTGroup = tGroups[bestTGi];
          mGroup.wasPadded = true;
          bestTGroup.hasPaddedChild = true;
          Index lenT = bestTGroup.fields.size();

          for (Index fi = 0; fi < supLen; ++fi) {
            if (bestTGroup.reqMut[fi] && !mGroup.reqMut[fi]) {
              setPostStepCMut(M, fi);
            }
          }
          for (Index newIdx = 0; newIdx < lenT; ++newIdx) {
            Index oldIdx = bestNewToOld[newIdx];
            bool mReq = (oldIdx != Index(-1)) && mGroup.reqMut[oldIdx];
            bool mAllow = (oldIdx == Index(-1)) || mGroup.allowMut[oldIdx] ||
                          (mGroup.fields[oldIdx].type !=
                             bestTGroup.fields[newIdx].type &&
                           canWidenFi[oldIdx]);
            if (mReq && !bestTGroup.reqMut[newIdx]) {
              for (auto U : bestTGroup.members) {
                setPostStepCMut(U, newIdx);
              }
              bestTGroup.reqMut[newIdx] = true;
            }
            bestTGroup.allowMut[newIdx] =
              bestTGroup.allowMut[newIdx] && mAllow;
          }

          StructFieldLayoutUpdate layout;
          layout.oldToNew = std::move(bestOldToNew);
          layout.newToOld = std::move(bestNewToOld);
          layout.newFields.resize(lenT);
          for (Index newIdx = 0; newIdx < lenT; ++newIdx) {
            Index oldIdx = layout.newToOld[newIdx];
            Field f = bestTGroup.rawFields[newIdx];
            f.mutable_ = bestTGroup.reqMut[newIdx] ? Mutable : Immutable;
            if (oldIdx != Index(-1)) {
              Type mFT = mGroup.fields[oldIdx].type;
              Type tFT = bestTGroup.fields[newIdx].type;
              if (mFT == tFT) {
                f.type = getFinalFieldType(M, oldIdx);
              } else {
                f.type = tFT;
                castWidenedImmFields[FieldKey{M, oldIdx}] =
                  mFields[oldIdx].type;
              }
            }
            layout.newFields[newIdx] = f;
          }
          structLayoutUpdates[M] = std::move(layout);
          bestTGroup.members.push_back(M);
          bestTGroup.cIdx = std::min(bestTGroup.cIdx, typeIndex[M]);
        }
      }

      std::vector<StructShapeKey> shapeGroupOrder;
      std::unordered_map<StructShapeKey,
                         std::vector<HeapType>,
                         StructShapeKeyHash>
        shapeGroups;
      for (auto it = subTypesFirst.rbegin(); it != subTypesFirst.rend(); ++it) {
        HeapType T = *it;
        if (!isCleanPrivateStruct(T) || mergedIntoSuper.count(T) ||
            collapsedParents.count(T)) {
          continue;
        }
        StructShapeKey key;
        if (auto s = T.getDeclaredSuperType()) {
          key.super = mapMergedHT(*s);
        }
        Index numF = getPostStepCNumFields(T);
        key.fields.reserve(numF);
        for (Index fi = 0; fi < numF; ++fi) {
          Field f = getPostStepCField(T, fi);
          key.fields.push_back({mapMergedType(f.type), f.packedType});
        }
        auto& vec = shapeGroups[key];
        if (vec.empty()) {
          shapeGroupOrder.push_back(key);
        }
        vec.push_back(T);
      }

      struct SubGroup {
        std::vector<HeapType> types;
        std::vector<Mutability> targetMut;
      };

      for (const auto& key : shapeGroupOrder) {
        const auto& members = shapeGroups[key];
        if (members.size() < 2) {
          continue;
        }
        Index numF = key.fields.size();
        std::vector<SubGroup> subGroups;
        for (auto T : members) {
          std::vector<Mutability> tMut(numF);
          for (Index fi = 0; fi < numF; ++fi) {
            tMut[fi] = getPostStepCField(T, fi).mutable_;
          }
          bool placed = false;
          for (auto& sg : subGroups) {
            std::vector<Mutability> combMut(numF);
            for (Index fi = 0; fi < numF; ++fi) {
              combMut[fi] =
                (sg.targetMut[fi] == Mutable || tMut[fi] == Mutable)
                  ? Mutable
                  : Immutable;
            }
            bool ok = true;
            for (Index fi = 0; fi < numF; ++fi) {
              if (combMut[fi] == Mutable) {
                if (tMut[fi] == Immutable && !canMakePostStepCMut(T, fi)) {
                  ok = false;
                  break;
                }
                for (auto U : sg.types) {
                  if (getPostStepCField(U, fi).mutable_ == Immutable &&
                      !canMakePostStepCMut(U, fi)) {
                    ok = false;
                    break;
                  }
                }
                if (!ok) {
                  break;
                }
              }
            }
            if (ok) {
              sg.types.push_back(T);
              sg.targetMut = std::move(combMut);
              placed = true;
              break;
            }
          }
          if (!placed) {
            subGroups.push_back({{T}, std::move(tMut)});
          }
        }

        for (const auto& sg : subGroups) {
          if (sg.types.size() < 2) {
            continue;
          }
          bool anyOpen = false;
          for (auto U : sg.types) {
            if (U.isOpen() || structOpenUpdates.count(U)) {
              anyOpen = true;
              break;
            }
          }
          for (auto U : sg.types) {
            if (anyOpen && !U.isOpen()) {
              structOpenUpdates.insert(U);
            }
            for (Index fi = 0; fi < numF; ++fi) {
              if (sg.targetMut[fi] == Mutable &&
                  getPostStepCField(U, fi).mutable_ == Immutable) {
                setPostStepCMut(U, fi);
              }
            }
          }
        }
      }
    }

    std::unordered_set<HeapType> arrayMutUpdates;
    for (const auto& [arrHT, state] : arrayCandidates) {
      if (state.pinned || arrHT.getArray().element.mutable_ != Immutable) {
        continue;
      }
      auto uIt = arrayUpdates.find(arrHT);
      Type finalET = uIt != arrayUpdates.end()
                       ? uIt->second
                       : arrHT.getArray().element.type;
      if (mutableElemTypes.count(finalET)) {
        arrayMutUpdates.insert(arrHT);
      } else if (uIt != arrayUpdates.end()) {
        Type oldET = arrHT.getArray().element.type;
        Nullability newNull = finalET.getNullability();
        Type rootET(getRootAncestor(oldET.getHeapType()), newNull);
        Type eqET(
          HeapTypes::eq.getBasic(oldET.getHeapType().getShared()), newNull);
        if (Type::isSubType(oldET, rootET) &&
            Type::isSubType(rootET, finalET) &&
            mutableElemTypes.count(rootET)) {
          uIt->second = rootET;
          arrayMutUpdates.insert(arrHT);
        } else if (Type::isSubType(oldET, eqET) &&
                   Type::isSubType(eqET, finalET) &&
                   mutableElemTypes.count(eqET)) {
          uIt->second = eqET;
          arrayMutUpdates.insert(arrHT);
        }
      }
    }

    std::unordered_map<HeapType, Type> castWidenedArrays;
    if (allowSiblingCastWiden && !arrayCandidates.empty()) {
      auto countNarrowArrayGets = [&](HeapType A, Type targetET) -> int {
        int narrow = 0;
        if (auto it = arrayGetDemands.find(A); it != arrayGetDemands.end()) {
          for (const auto& [reqHT, reqNull] : it->second) {
            if (!HeapType::isSubType(targetET.getHeapType(), reqHT) ||
                (targetET.isNullable() && !reqNull)) {
              narrow++;
            }
          }
        }
        return narrow;
      };

      struct ArrayTargetKey {
        Type elemType;
        Mutability mut;
        bool operator==(const ArrayTargetKey& o) const {
          return elemType == o.elemType && mut == o.mut;
        }
      };
      struct ArrayTargetKeyHash {
        size_t operator()(const ArrayTargetKey& k) const {
          return std::hash<Type>{}(k.elemType) ^ (size_t(k.mut) << 1);
        }
      };
      std::unordered_map<ArrayTargetKey, Index, ArrayTargetKeyHash>
        canonArrayIdx;
      for (const auto& [arrHT, state] : arrayCandidates) {
        if (state.pinned) {
          continue;
        }
        Type curET = arrayUpdates.count(arrHT)
                       ? arrayUpdates[arrHT]
                       : arrHT.getArray().element.type;
        Mutability curMut =
          (arrHT.getArray().element.mutable_ == Mutable ||
           arrayMutUpdates.count(arrHT))
            ? Mutable
            : Immutable;
        ArrayTargetKey key{curET, curMut};
        auto it = canonArrayIdx.find(key);
        if (it == canonArrayIdx.end() || typeIndex[arrHT] < it->second) {
          canonArrayIdx[key] = typeIndex[arrHT];
        }
      }

      for (auto arrHT : allTypes) {
        auto cIt = arrayCandidates.find(arrHT);
        if (cIt == arrayCandidates.end() || cIt->second.pinned) {
          continue;
        }
        Type curET = arrayUpdates.count(arrHT)
                       ? arrayUpdates[arrHT]
                       : arrHT.getArray().element.type;
        if (!curET.isRef() || !curET.isNullable() || curET.isExact() ||
            curET.isNull() || curET.getHeapType().isBasic()) {
          continue;
        }
        Mutability curMut =
          (arrHT.getArray().element.mutable_ == Mutable ||
           arrayMutUpdates.count(arrHT))
            ? Mutable
            : Immutable;
        int aDefBytes = 3 + getExactTypeBytes(curET);
        int castBytes = (typeIndex[curET.getHeapType()] < 64) ? 3 : 4;

        int bestNet = 0;
        Type bestET;
        Mutability bestMut = curMut;
        for (const auto& [tKey, cIdx] : canonArrayIdx) {
          if (tKey.elemType == curET ||
              !Type::isSubType(curET, tKey.elemType)) {
            continue;
          }
          if (tKey.mut != Mutable && !tKey.elemType.getHeapType().isBasic()) {
            continue;
          }
          if (curMut == Mutable && tKey.mut != Mutable) {
            continue;
          }
          int lebSave = getLebSavings(arrHT, cIdx);
          int narrowGets = countNarrowArrayGets(arrHT, tKey.elemType);
          int net = aDefBytes + lebSave - narrowGets * castBytes;
          if (net > bestNet) {
            bestNet = net;
            bestET = tKey.elemType;
            bestMut = tKey.mut;
          }
        }
        if (bestNet > 0) {
          arrayUpdates[arrHT] = bestET;
          if (bestMut == Mutable &&
              arrHT.getArray().element.mutable_ == Immutable) {
            arrayMutUpdates.insert(arrHT);
          }
          castWidenedArrays[arrHT] = arrHT.getArray().element.type;
        }
      }
    }

    bool anyModified = false;
    GlobalTypeRewriter::TypeMap oldToNew;
    if (!structUpdates.empty() || !arrayUpdates.empty() ||
        !arrayMutUpdates.empty() || !structMutUpdates.empty() ||
        !structOpenUpdates.empty() || !structLayoutUpdates.empty()) {
      class FieldAndArrayRewriter : public GlobalTypeRewriter {
        const StructUpdateMap& structUpdates;
        const ArrayUpdateMap& arrayUpdates;
        const std::unordered_set<HeapType>& arrayMutUpdates;
        const std::unordered_map<HeapType, std::unordered_set<Index>>&
          structMutUpdates;
        const std::unordered_set<HeapType>& structOpenUpdates;
        const std::unordered_map<HeapType, StructFieldLayoutUpdate>&
          structLayoutUpdates;

      public:
        TypeMap& outMap;

        FieldAndArrayRewriter(
          Module& wasm,
          const StructUpdateMap& structUpdates,
          const ArrayUpdateMap& arrayUpdates,
          const std::unordered_set<HeapType>& arrayMutUpdates,
          const std::unordered_map<HeapType, std::unordered_set<Index>>&
            structMutUpdates,
          const std::unordered_set<HeapType>& structOpenUpdates,
          const std::unordered_map<HeapType, StructFieldLayoutUpdate>&
            structLayoutUpdates,
          WorldMode worldMode,
          TypeMap& outMap)
          : GlobalTypeRewriter(wasm, worldMode), structUpdates(structUpdates),
            arrayUpdates(arrayUpdates), arrayMutUpdates(arrayMutUpdates),
            structMutUpdates(structMutUpdates),
            structOpenUpdates(structOpenUpdates),
            structLayoutUpdates(structLayoutUpdates), outMap(outMap) {
          outMap = rebuildTypes(getSortedTypes(getPrivatePredecessors()));
          mapTypes(outMap);
        }

        void modifyStruct(HeapType oldType, Struct& struct_) override {
          auto lIt = structLayoutUpdates.find(oldType);
          if (lIt != structLayoutUpdates.end()) {
            struct_.fields = lIt->second.newFields;
            for (auto& f : struct_.fields) {
              f.type = getTempType(f.type);
            }
            auto mIt = structMutUpdates.find(oldType);
            if (mIt != structMutUpdates.end()) {
              for (Index oldFi : mIt->second) {
                if (oldFi < lIt->second.oldToNew.size()) {
                  Index newFi = lIt->second.oldToNew[oldFi];
                  if (newFi < struct_.fields.size()) {
                    struct_.fields[newFi].mutable_ = Mutable;
                  }
                }
              }
            }
            return;
          }
          auto it = structUpdates.find(oldType);
          if (it != structUpdates.end()) {
            for (const auto& [fi, newType] : it->second) {
              struct_.fields[fi].type = getTempType(newType);
            }
          }
          auto mIt = structMutUpdates.find(oldType);
          if (mIt != structMutUpdates.end()) {
            for (Index fi : mIt->second) {
              struct_.fields[fi].mutable_ = Mutable;
            }
          }
        }

        void modifyArray(HeapType oldType, Array& array) override {
          auto it = arrayUpdates.find(oldType);
          if (it != arrayUpdates.end()) {
            array.element.type = getTempType(it->second);
          }
          if (arrayMutUpdates.count(oldType)) {
            array.element.mutable_ = Mutable;
          }
        }

        void modifyTypeBuilderEntry(TypeBuilder& typeBuilder,
                                    Index i,
                                    HeapType oldType) override {
          if (structOpenUpdates.count(oldType)) {
            typeBuilder[i].setOpen(true);
          }
        }
      } rewriter(*module,
                 structUpdates,
                 arrayUpdates,
                 arrayMutUpdates,
                 structMutUpdates,
                 structOpenUpdates,
                 structLayoutUpdates,
                 getPassOptions().worldMode,
                 oldToNew);
      anyModified = true;
    }

    auto mapNewHT = [&](HeapType ht) {
      if (auto it = oldToNew.find(ht); it != oldToNew.end()) {
        return it->second;
      }
      return ht;
    };

    std::unordered_map<HeapType, StructFieldLayoutUpdate> mappedLayoutUpdates;
    for (const auto& [oldHT, layout] : structLayoutUpdates) {
      mappedLayoutUpdates[mapNewHT(oldHT)] = layout;
    }

    std::unordered_map<FieldKey, Type, FieldKeyHash> mappedCastWidened;
    for (const auto& [k, oldFT] : castWidenedImmFields) {
      HeapType newU = mapNewHT(k.anc);
      Index newIdx = k.index;
      if (auto lIt = structLayoutUpdates.find(k.anc);
          lIt != structLayoutUpdates.end()) {
        assert(k.index < lIt->second.oldToNew.size());
        newIdx = lIt->second.oldToNew[k.index];
      }
      HeapType newOldHT = oldFT.getHeapType().isBasic()
                            ? oldFT.getHeapType()
                            : mapNewHT(oldFT.getHeapType());
      mappedCastWidened[FieldKey{newU, newIdx}] =
        Type(newOldHT, oldFT.getNullability());
    }

    std::unordered_map<HeapType, Type> mappedCastWidenedArrays;
    for (const auto& [arrHT, oldET] : castWidenedArrays) {
      HeapType newArrHT = mapNewHT(arrHT);
      HeapType newOldHT = oldET.getHeapType().isBasic()
                            ? oldET.getHeapType()
                            : mapNewHT(oldET.getHeapType());
      mappedCastWidenedArrays[newArrHT] =
        Type(newOldHT, oldET.getNullability());
    }

    // Check if any pure array global can switch to an existing wider or
    // canonical mutable/immutable array HeapType.
    std::unordered_set<HeapType> newlyMutatedArrHTs;
    for (auto ht : arrayMutUpdates) {
      newlyMutatedArrHTs.insert(mapNewHT(ht));
    }
    auto curTypes = ModuleUtils::collectHeapTypes(*module);
    SubTypes curSubTypes(curTypes);
    std::unordered_map<Type, HeapType> existingMutArrayHT;
    std::unordered_map<Type, HeapType> existingImmArrayHT;
    for (auto ht : curTypes) {
      if (ht.isArray() && !ht.isShared() && !ht.getDescriptorType() &&
          !ht.getDescribedType() && !ht.getDeclaredSuperType() &&
          curSubTypes.getImmediateSubTypes(ht).empty()) {
        const auto& elem = ht.getArray().element;
        if (!elem.isPacked() && elem.mutable_ == Mutable &&
            elem.type.isRef() && !elem.type.isExact() &&
            !newlyMutatedArrHTs.count(ht)) {
          existingMutArrayHT.try_emplace(elem.type, ht);
        }
      }
    }
    for (auto ht : curTypes) {
      if (ht.isArray() && !ht.isShared() && !ht.getDescriptorType() &&
          !ht.getDescribedType() && !ht.getDeclaredSuperType() &&
          curSubTypes.getImmediateSubTypes(ht).empty()) {
        const auto& elem = ht.getArray().element;
        if (!elem.isPacked() && elem.type.isRef() && !elem.type.isExact()) {
          if (elem.mutable_ == Mutable) {
            existingMutArrayHT.try_emplace(elem.type, ht);
          } else {
            existingImmArrayHT.try_emplace(elem.type, ht);
          }
        }
      }
    }

    std::unordered_set<Name> retargetedGlobals;
    for (const auto& [gName, gState] : pureGlobalCandidates) {
      if (gState.pinned || !pureGlobalHasGet.count(gName) ||
          totalGlobalGets[gName] != validArrayGlobalGets[gName]) {
        continue;
      }
      Global* g = module->getGlobal(gName);
      HeapType curArrHT = g->type.getHeapType();
      if (mappedCastWidenedArrays.count(curArrHT)) {
        continue;
      }
      Type curET = curArrHT.getArray().element.type;
      bool isImmGlobal = g->init->is<ArrayNewFixed>();
      Nullability wantNull =
        (curET.isNullable() || gState.acceptsNull) ? Nullable : NonNullable;
      HeapType wantHT =
        gState.canWidenHT ? mapNewHT(gState.allowedHT) : curET.getHeapType();
      Type wantET(wantHT, wantNull);
      if (!Type::isSubType(curET, wantET)) {
        continue;
      }
      auto findTargetArr = [&](Type et) -> std::optional<HeapType> {
        auto mIt = existingMutArrayHT.find(et);
        if (mIt != existingMutArrayHT.end()) {
          return mIt->second;
        }
        if (isImmGlobal) {
          auto iIt = existingImmArrayHT.find(et);
          if (iIt != existingImmArrayHT.end()) {
            return iIt->second;
          }
        }
        return std::nullopt;
      };
      std::optional<HeapType> targetArrHT;
      if (wantHT.isBasic() && !curET.getHeapType().isBasic()) {
        Type rootET(getRootAncestor(curET.getHeapType()), wantNull);
        if (Type::isSubType(curET, rootET) &&
            Type::isSubType(rootET, wantET)) {
          targetArrHT = findTargetArr(rootET);
        }
        if (!targetArrHT) {
          Type eqET(
            HeapTypes::eq.getBasic(curET.getHeapType().getShared()), wantNull);
          if (Type::isSubType(curET, eqET) &&
              Type::isSubType(eqET, wantET)) {
            targetArrHT = findTargetArr(eqET);
          }
        }
      }
      if (!targetArrHT) {
        targetArrHT = findTargetArr(wantET);
      }
      if (targetArrHT && *targetArrHT != curArrHT &&
          (wantET != curET ||
           existingMutArrayHT.find(curET) != existingMutArrayHT.end())) {
        g->type = g->type.with(*targetArrHT);
        if (auto* anew = g->init->dynCast<ArrayNewFixed>()) {
          anew->type = anew->type.with(*targetArrHT);
        } else if (auto* anew = g->init->dynCast<ArrayNew>()) {
          anew->type = anew->type.with(*targetArrHT);
        }
        retargetedGlobals.insert(gName);
        anyModified = true;
      }
    }

    if (anyModified) {
      auto makeDefaultFieldConst = [](Module& wasm, Type ft) -> Expression* {
        Builder builder(wasm);
        if (ft == Type::i32) {
          return builder.makeConst(Literal(int32_t(0)));
        }
        if (ft == Type::i64) {
          return builder.makeConst(Literal(int64_t(0)));
        }
        if (ft == Type::f32) {
          return builder.makeConst(Literal(float(0.0f)));
        }
        if (ft == Type::f64) {
          return builder.makeConst(Literal(double(0.0)));
        }
        if (ft.isRef() && ft.isNullable()) {
          return builder.makeRefNull(ft.getHeapType().getBottom());
        }
        WASM_UNREACHABLE("unexpected padded field type");
      };

      struct CastInserter : public PostWalker<CastInserter> {
        Module* module;
        const std::unordered_map<FieldKey, Type, FieldKeyHash>&
          mappedCastWidened;
        const std::unordered_map<HeapType, Type>& mappedCastWidenedArrays;
        const std::unordered_map<HeapType, StructFieldLayoutUpdate>&
          mappedLayoutUpdates;
        decltype(makeDefaultFieldConst)& makeDefault;
        CastInserter(
          Module* module,
          const std::unordered_map<FieldKey, Type, FieldKeyHash>&
            mappedCastWidened,
          const std::unordered_map<HeapType, Type>&
            mappedCastWidenedArrays,
          const std::unordered_map<HeapType, StructFieldLayoutUpdate>&
            mappedLayoutUpdates,
          decltype(makeDefaultFieldConst)& makeDefault)
          : module(module), mappedCastWidened(mappedCastWidened),
            mappedCastWidenedArrays(mappedCastWidenedArrays),
            mappedLayoutUpdates(mappedLayoutUpdates),
            makeDefault(makeDefault) {}
        void visitStructNew(StructNew* curr) {
          if (curr->isWithDefault() || !curr->type.isStruct()) {
            return;
          }
          HeapType uHT = curr->type.getHeapType();
          auto lIt = mappedLayoutUpdates.find(uHT);
          if (lIt == mappedLayoutUpdates.end()) {
            return;
          }
          const auto& layout = lIt->second;
          const auto& newFields = uHT.getStruct().fields;
          Index newNumF = newFields.size();
          std::vector<Expression*> oldOps(curr->operands.begin(),
                                          curr->operands.end());
          curr->operands.resize(newNumF);
          for (Index newIdx = 0; newIdx < newNumF; ++newIdx) {
            Index oldIdx = layout.newToOld[newIdx];
            if (oldIdx != Index(-1)) {
              assert(oldIdx < oldOps.size());
              curr->operands[newIdx] = oldOps[oldIdx];
            } else {
              curr->operands[newIdx] =
                makeDefault(*module, newFields[newIdx].type);
            }
          }
        }
        void visitStructGet(StructGet* curr) {
          if (!curr->ref->type.isStruct()) {
            return;
          }
          HeapType uHT = curr->ref->type.getHeapType();
          if (auto lIt = mappedLayoutUpdates.find(uHT);
              lIt != mappedLayoutUpdates.end()) {
            assert(curr->index < lIt->second.oldToNew.size());
            curr->index = lIt->second.oldToNew[curr->index];
          }
          if (curr->type == Type::unreachable) {
            return;
          }
          auto it = mappedCastWidened.find(FieldKey{uHT, curr->index});
          if (it != mappedCastWidened.end()) {
            curr->type = uHT.getStruct().fields[curr->index].type;
            replaceCurrent(Builder(*module).makeRefCast(curr, it->second));
          }
        }
        void visitStructSet(StructSet* curr) {
          if (!curr->ref->type.isStruct()) {
            return;
          }
          HeapType uHT = curr->ref->type.getHeapType();
          if (auto lIt = mappedLayoutUpdates.find(uHT);
              lIt != mappedLayoutUpdates.end()) {
            assert(curr->index < lIt->second.oldToNew.size());
            curr->index = lIt->second.oldToNew[curr->index];
          }
        }
        void visitArrayGet(ArrayGet* curr) {
          if (curr->type == Type::unreachable ||
              !curr->ref->type.isArray()) {
            return;
          }
          HeapType arrHT = curr->ref->type.getHeapType();
          auto it = mappedCastWidenedArrays.find(arrHT);
          if (it != mappedCastWidenedArrays.end()) {
            curr->type = arrHT.getArray().element.type;
            replaceCurrent(Builder(*module).makeRefCast(curr, it->second));
          }
        }
      };

      bool needInserter = !mappedCastWidened.empty() ||
                          !mappedCastWidenedArrays.empty() ||
                          !mappedLayoutUpdates.empty();
      ModuleUtils::ParallelFunctionAnalysis<int, Mutable>(
        *module, [&](Function* func, int&) {
          if (func->imported()) {
            return;
          }
          if (!retargetedGlobals.empty()) {
            for (auto* gget : FindAll<GlobalGet>(func->body).list) {
              if (retargetedGlobals.count(gget->name)) {
                gget->type = module->getGlobal(gget->name)->type;
              }
            }
          }
          if (needInserter) {
            CastInserter inserter(module,
                                  mappedCastWidened,
                                  mappedCastWidenedArrays,
                                  mappedLayoutUpdates,
                                  makeDefaultFieldConst);
            inserter.walk(func->body);
          }
          ReFinalize().walkFunctionInModule(func, module);
        });
      if (needInserter) {
        for (auto& g : module->globals) {
          if (!g->imported() && g->init) {
            CastInserter inserter(module,
                                  mappedCastWidened,
                                  mappedCastWidenedArrays,
                                  mappedLayoutUpdates,
                                  makeDefaultFieldConst);
            inserter.walk(g->init);
          }
        }
      }
    }

    return anyModified;
  }

  void collectModuleEscapeAndSigInfo(
    Module* module,
    const std::vector<HeapType>& types,
    std::unordered_set<HeapType>& fixed,
    std::unordered_map<HeapType, std::vector<Function*>>& sigFuncs,
    std::unordered_set<Name>& escapedFuncs,
    std::unordered_map<Function*, std::vector<Call*>>& callsTo,
    std::unordered_set<Name>& hasReturnCall,
    std::unordered_set<HeapType>& sigUsedAsRef) {
    fixed.clear();
    for (auto type :
         ModuleUtils::getPublicHeapTypes(*module, getPassOptions().worldMode)) {
      fixed.insert(type);
    }
    for (auto& tag : module->tags) {
      fixed.insert(tag->type);
    }
    for (auto func : Intrinsics(*module).getJSCalledFunctions()) {
      fixed.insert(module->getFunction(func)->type.getHeapType());
    }

    sigFuncs.clear();
    for (auto& func : module->functions) {
      if (!func->imported()) {
        sigFuncs[func->type.getHeapType()].push_back(func.get());
      }
    }

    escapedFuncs.clear();
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
    for (auto& func : module->functions) {
      if (func->imported()) {
        escapedFuncs.insert(func->name);
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
      std::vector<HeapType> callRefSigs;
      bool hasTailCall = false;
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
          scan.calls.push_back(call);
          if (call->isReturn) {
            scan.hasTailCall = true;
          }
        }
        for (auto* ci : FindAll<CallIndirect>(func->body).list) {
          if (ci->isReturn) {
            scan.hasTailCall = true;
          }
        }
        for (auto* cr : FindAll<CallRef>(func->body).list) {
          if (cr->target->type.isSignature()) {
            scan.callRefSigs.push_back(cr->target->type.getHeapType());
          }
          if (cr->isReturn) {
            scan.hasTailCall = true;
          }
        }
      });

    callsTo.clear();
    hasReturnCall.clear();
    sigUsedAsRef.clear();
    for (auto& fn : module->functions) {
      if (fn->imported()) {
        continue;
      }
      auto* func = fn.get();
      auto& scan = callScanner.map[func];
      if (scan.hasTailCall) {
        hasReturnCall.insert(func->name);
      }
      for (auto name : scan.refFuncs) {
        escapedFuncs.insert(name);
      }
      for (auto sigHT : scan.callRefSigs) {
        sigUsedAsRef.insert(sigHT);
      }
      for (auto* call : scan.calls) {
        auto* callee = module->getFunction(call->target);
        callsTo[callee].push_back(call);
        if (call->isReturn) {
          hasReturnCall.insert(call->target);
        }
      }
    }

    // Any signature referenced as a value type (e.g. `(ref null $sig)` in a
    // param, local, global, or struct field) that has no `ref.func` in the
    // module (such as in unit tests) treats its functions as escaped so
    // `call_ref $sig` stays synchronized with them.
    auto noteValueType = [&](Type t) {
      for (auto elem : t) {
        if (elem.isRef() && elem.getHeapType().isSignature()) {
          sigUsedAsRef.insert(elem.getHeapType());
        }
      }
    };
    for (auto type : types) {
      if (type.isSignature()) {
        noteValueType(type.getSignature().params);
        noteValueType(type.getSignature().results);
      } else if (type.isStruct()) {
        for (auto& f : type.getStruct().fields) {
          noteValueType(f.type);
        }
      } else if (type.isArray()) {
        noteValueType(type.getArray().element.type);
      }
    }
    for (auto& g : module->globals) {
      noteValueType(g->type);
    }
    for (auto& func : module->functions) {
      for (auto v : func->vars) {
        noteValueType(v);
      }
    }
    for (auto sigHT : sigUsedAsRef) {
      auto it = sigFuncs.find(sigHT);
      if (it != sigFuncs.end()) {
        bool anyEscaped = false;
        for (auto* f : it->second) {
          if (escapedFuncs.count(f->name)) {
            anyEscaped = true;
            break;
          }
        }
        if (!anyEscaped) {
          for (auto* f : it->second) {
            escapedFuncs.insert(f->name);
          }
        }
      }
    }
  }

  void relaxSignaturesAndLocals(Module* module, bool widenTypes) {
    auto types = ModuleUtils::collectHeapTypes(*module);
    SubTypes subTypes(types);

    std::unordered_set<HeapType> fixed;
    std::unordered_map<HeapType, std::vector<Function*>> sigFuncs;
    std::unordered_set<Name> escapedFuncs;
    std::unordered_map<Function*, std::vector<Call*>> callsTo;
    std::unordered_set<Name> hasReturnCall;
    std::unordered_set<HeapType> sigUsedAsRef;
    collectModuleEscapeAndSigInfo(module,
                                  types,
                                  fixed,
                                  sigFuncs,
                                  escapedFuncs,
                                  callsTo,
                                  hasReturnCall,
                                  sigUsedAsRef);

    std::vector<HeapType> candidateSigs;
    std::unordered_set<HeapType> candidateSigSet;
    SigParamHTMap curSigParamHTs;
    for (auto type : types) {
      if (!type.isSignature() || fixed.count(type)) {
        continue;
      }
      // Keep it simple: do not touch types that take part in subtyping.
      if (type.getDeclaredSuperType() ||
          !subTypes.getImmediateSubTypes(type).empty()) {
        continue;
      }
      candidateSigs.push_back(type);
      candidateSigSet.insert(type);
      if (widenTypes && !sigFuncs[type].empty()) {
        auto sig = type.getSignature();
        std::vector<std::optional<HeapType>> paramHTs(sig.params.size());
        for (size_t i = 0; i < sig.params.size(); ++i) {
          if (sig.params[i].isRef() && !sig.params[i].isExact()) {
            paramHTs[i] = sig.params[i].getHeapType();
          }
        }
        curSigParamHTs[type] = std::move(paramHTs);
      }
    }

    FuncParamHTMap curFuncParamHTs;
    FuncResultDemandMap curFuncResultDemands;
    std::unordered_map<HeapType, std::vector<CallRef*>> callRefsTo;
    std::unordered_set<HeapType> pinnedResultSigs;
    if (widenTypes) {
      std::unordered_set<Name> nonCallRefEscaped;
      if (module->start.is()) {
        nonCallRefEscaped.insert(module->start);
      }
      for (auto& exp : module->exports) {
        if (exp->kind == ExternalKind::Function) {
          nonCallRefEscaped.insert(*exp->getInternalName());
        }
      }
      for (auto func : Intrinsics(*module).getJSCalledFunctions()) {
        nonCallRefEscaped.insert(func);
      }
      for (auto& seg : module->elementSegments) {
        for (auto* expr : seg->data) {
          for (auto* rf : FindAll<RefFunc>(expr).list) {
            nonCallRefEscaped.insert(rf->func);
          }
        }
      }
      struct CallRefScan {
        std::vector<CallRef*> callRefs;
        std::vector<HeapType> pinnedSigs;
      };
      ModuleUtils::ParallelFunctionAnalysis<CallRefScan> crScanner(
        *module, [&](Function* func, CallRefScan& scan) {
          if (func->imported()) {
            return;
          }
          for (auto* ci : FindAll<CallIndirect>(func->body).list) {
            scan.pinnedSigs.push_back(ci->heapType);
          }
          for (auto* call : FindAll<Call>(func->body).list) {
            if (Intrinsics(*module).isCallWithoutEffects(call) &&
                !call->operands.empty() &&
                call->operands.back()->type.isSignature()) {
              scan.pinnedSigs.push_back(
                call->operands.back()->type.getHeapType());
            }
          }
          for (auto* cr : FindAll<CallRef>(func->body).list) {
            if (cr->target->type.isSignature()) {
              HeapType sigHT = cr->target->type.getHeapType();
              if (cr->isReturn) {
                scan.pinnedSigs.push_back(sigHT);
              } else {
                scan.callRefs.push_back(cr);
              }
            }
          }
        });
      for (auto& fn : module->functions) {
        if (fn->imported()) {
          continue;
        }
        auto& scan = crScanner.map[fn.get()];
        for (auto sigHT : scan.pinnedSigs) {
          pinnedResultSigs.insert(sigHT);
        }
        for (auto* cr : scan.callRefs) {
          callRefsTo[cr->target->type.getHeapType()].push_back(cr);
        }
      }
      for (auto type : candidateSigs) {
        if (sigUsedAsRef.count(type) && callRefsTo[type].empty()) {
          pinnedResultSigs.insert(type);
        }
        for (auto* f : sigFuncs[type]) {
          if (nonCallRefEscaped.count(f->name) ||
              hasReturnCall.count(f->name)) {
            pinnedResultSigs.insert(type);
            break;
          }
        }
      }

      for (auto& func : module->functions) {
        if (func->imported()) {
          continue;
        }
        HeapType sigHT = func->type.getHeapType();
        if (!escapedFuncs.count(func->name)) {
          auto params = func->getParams();
          std::vector<std::optional<HeapType>> paramHTs(params.size());
          for (size_t i = 0; i < params.size(); ++i) {
            if (params[i].isRef() && !params[i].isExact()) {
              paramHTs[i] = params[i].getHeapType();
            }
          }
          curFuncParamHTs[func.get()] = std::move(paramHTs);
        }

        auto origRes = func->getResults();
        bool canRelaxRes =
          !escapedFuncs.count(func->name) ||
          (candidateSigSet.count(sigHT) && !pinnedResultSigs.count(sigHT));
        bool hasResultConsumerOrUnreachableStub =
          !callsTo[func.get()].empty() || !callRefsTo[sigHT].empty() ||
          (func->body && func->body->is<Unreachable>() &&
           escapedFuncs.count(func->name) && !sigUsedAsRef.count(sigHT));
        if (canRelaxRes && hasResultConsumerOrUnreachableStub &&
            !hasReturnCall.count(func->name) && origRes.isRef() &&
            !origRes.isExact() && !origRes.isTuple()) {
          curFuncResultDemands[func.get()] = CallReturnDemand{
            getRootAncestor(origRes.getHeapType()), true};
        }
      }

      // Propagate parameter and return demands across direct and indirect calls
      // for up to 4 rounds, followed by verification passes until convergence.
      for (int round = 0; round < 16; ++round) {
        bool verifyOnly = (round >= 4);
        struct FuncAnalysisResult {
          std::vector<HeapType> allowedHTs;
          std::unordered_map<Name, CallReturnDemand> callDemands;
          std::unordered_map<HeapType, CallReturnDemand> callRefDemands;
        };
        ModuleUtils::ParallelFunctionAnalysis<FuncAnalysisResult> analysis(
          *module, [&](Function* func, FuncAnalysisResult& res) {
            if (func->imported()) {
              return;
            }
            bool allowParams =
              curFuncParamHTs.count(func) != 0 ||
              curSigParamHTs.count(func->type.getHeapType()) != 0;
            DemandAnalyzer analyzer(*module,
                                    func,
                                    &curSigParamHTs,
                                    allowParams,
                                    round > 0,
                                    &curFuncParamHTs,
                                    &curFuncResultDemands,
                                    &candidateSigSet);
            res.allowedHTs = analyzer.analyze();
            res.callDemands = std::move(analyzer.callReturnDemands);
            res.callRefDemands = std::move(analyzer.callRefReturnDemands);
          });

        bool anyChanged = false;
        for (auto& [f, paramHTs] : curFuncParamHTs) {
          auto params = f->getParams();
          const auto& fAllowed = analysis.map[f].allowedHTs;
          for (size_t i = 0; i < params.size(); ++i) {
            Type param = params[i];
            if (!param.isRef() || param.isExact() ||
                !param.getHeapType().getDeclaredSuperType()) {
              continue;
            }
            HeapType orig = param.getHeapType();
            HeapType best = verifyOnly ? *paramHTs[i] : getRootAncestor(orig);
            if (i < fAllowed.size()) {
              best = tightenAncestor(orig, best, fAllowed[i]);
            } else {
              best = orig;
            }
            if (paramHTs[i] != best) {
              paramHTs[i] = best;
              anyChanged = true;
            }
          }
        }

        for (auto type : candidateSigs) {
          auto it = curSigParamHTs.find(type);
          if (it == curSigParamHTs.end()) {
            continue;
          }
          const auto& funcs = sigFuncs[type];
          bool hasEscapedInSig = false;
          for (auto* f : funcs) {
            if (escapedFuncs.count(f->name)) {
              hasEscapedInSig = true;
              break;
            }
          }
          auto sig = type.getSignature();
          for (size_t i = 0; i < sig.params.size(); ++i) {
            Type param = sig.params[i];
            if (!param.isRef() || param.isExact() ||
                !param.getHeapType().getDeclaredSuperType()) {
              continue;
            }
            HeapType orig = param.getHeapType();
            HeapType best =
              verifyOnly ? *it->second[i] : getRootAncestor(orig);
            for (auto* f : funcs) {
              if (hasEscapedInSig && !escapedFuncs.count(f->name)) {
                continue;
              }
              const auto& fAllowed = analysis.map[f].allowedHTs;
              if (i < fAllowed.size()) {
                best = tightenAncestor(orig, best, fAllowed[i]);
              } else {
                best = orig;
              }
            }
            if (it->second[i] != best) {
              it->second[i] = best;
              anyChanged = true;
            }
          }
        }

        std::unordered_map<Function*, CallReturnDemand> nextResDemands;
        for (auto& [f, curDem] : curFuncResultDemands) {
          HeapType origResHT = f->getResults().getHeapType();
          nextResDemands[f] = CallReturnDemand{
            verifyOnly ? curDem.ht : getRootAncestor(origResHT),
            verifyOnly ? curDem.acceptsNull : true};
        }
        for (auto& fn : module->functions) {
          if (fn->imported()) {
            continue;
          }
          auto& res = analysis.map[fn.get()];
          for (auto& [calleeName, cDem] : res.callDemands) {
            auto* callee = module->getFunction(calleeName);
            auto it = nextResDemands.find(callee);
            if (it != nextResDemands.end()) {
              HeapType origResHT = callee->getResults().getHeapType();
              it->second.ht =
                tightenAncestor(origResHT, it->second.ht, cDem.ht);
              if (!cDem.acceptsNull) {
                it->second.acceptsNull = false;
              }
            }
          }
          for (auto& [sigHT, cDem] : res.callRefDemands) {
            auto sfIt = sigFuncs.find(sigHT);
            if (sfIt != sigFuncs.end()) {
              for (auto* callee : sfIt->second) {
                auto it = nextResDemands.find(callee);
                if (it != nextResDemands.end()) {
                  HeapType origResHT = callee->getResults().getHeapType();
                  it->second.ht =
                    tightenAncestor(origResHT, it->second.ht, cDem.ht);
                  if (!cDem.acceptsNull) {
                    it->second.acceptsNull = false;
                  }
                }
              }
            }
          }
        }
        for (auto type : candidateSigs) {
          auto origRes = type.getSignature().results;
          if (pinnedResultSigs.count(type) || sigFuncs[type].size() <= 1 ||
              !origRes.isRef() || origRes.isExact() || origRes.isTuple()) {
            continue;
          }
          bool hasEscaped = sigUsedAsRef.count(type) != 0;
          if (!hasEscaped) {
            for (auto* f : sigFuncs[type]) {
              if (escapedFuncs.count(f->name)) {
                hasEscaped = true;
                break;
              }
            }
          }
          if (!hasEscaped) {
            continue;
          }
          HeapType origResHT = origRes.getHeapType();
          HeapType commonHT = getRootAncestor(origResHT);
          bool commonNull = true;
          bool allInMap = true;
          for (auto* f : sigFuncs[type]) {
            auto it = nextResDemands.find(f);
            if (it == nextResDemands.end()) {
              allInMap = false;
              break;
            }
            commonHT = tightenAncestor(origResHT, commonHT, it->second.ht);
            if (!it->second.acceptsNull) {
              commonNull = false;
            }
          }
          if (allInMap) {
            for (auto* f : sigFuncs[type]) {
              nextResDemands[f] = CallReturnDemand{commonHT, commonNull};
            }
          }
        }
        for (auto& [f, curDem] : curFuncResultDemands) {
          auto& nextDem = nextResDemands[f];
          if (curDem.ht != nextDem.ht ||
              curDem.acceptsNull != nextDem.acceptsNull) {
            curDem = nextDem;
            anyChanged = true;
          }
        }

        if (!anyChanged) {
          if (verifyOnly) {
            break;
          }
          // Jump straight to verification pass.
          round = 3;
        }
      }
    }

    std::unordered_set<Function*> resultChangedFuncs;
    std::vector<HeapType> resultChangedSigs;
    GlobalTypeRewriter::SignatureUpdates updates;
    std::unordered_map<Signature, HeapType> targetSigToHT;
    for (auto type : candidateSigs) {
      auto sig = type.getSignature();
      std::vector<Type> params;
      bool changed = false;
      auto it = curSigParamHTs.find(type);
      for (size_t i = 0; i < sig.params.size(); ++i) {
        auto param = sig.params[i];
        auto relaxed = makeNullable(param);
        if (widenTypes && it != curSigParamHTs.end() && it->second[i] &&
            *it->second[i] != param.getHeapType()) {
          relaxed = Type(*it->second[i], Nullable);
        }
        changed = changed || relaxed != param;
        params.push_back(relaxed);
      }
      Type results = sig.results;
      if (widenTypes && !pinnedResultSigs.count(type) &&
          !sigFuncs[type].empty()) {
        std::optional<Type> commonRes;
        bool allAgree = true;
        for (auto* f : sigFuncs[type]) {
          Type fRes = sig.results;
          auto resIt = curFuncResultDemands.find(f);
          if (resIt != curFuncResultDemands.end()) {
            Nullability resNull =
              (sig.results.isNullable() || resIt->second.acceptsNull)
                ? Nullable
                : NonNullable;
            fRes = Type(resIt->second.ht, resNull);
          } else {
            allAgree = false;
            break;
          }
          if (!commonRes) {
            commonRes = fRes;
          } else if (*commonRes != fRes) {
            allAgree = false;
            break;
          }
        }
        if (allAgree && commonRes && *commonRes != results) {
          results = *commonRes;
          changed = true;
          for (auto* f : sigFuncs[type]) {
            resultChangedFuncs.insert(f);
          }
          resultChangedSigs.push_back(type);
        }
      }
      Signature finalSig(Type(params), results);
      targetSigToHT.try_emplace(finalSig, type);
      if (changed) {
        updates[type] = finalSig;
      }
    }

    if (widenTypes) {
      for (auto& func : module->functions) {
        Function* f = func.get();
        auto pIt = curFuncParamHTs.find(f);
        if (pIt == curFuncParamHTs.end()) {
          continue;
        }
        auto& paramHTs = pIt->second;
        auto sig = f->getSig();
        std::vector<Type> newParams;
        newParams.reserve(sig.params.size());
        for (size_t i = 0; i < sig.params.size(); ++i) {
          Type param = sig.params[i];
          Type relaxed = makeNullable(param);
          if (i < paramHTs.size() && paramHTs[i]) {
            relaxed = Type(*paramHTs[i], Nullable);
          }
          newParams.push_back(relaxed);
        }
        Type newResults = sig.results;
        auto resIt = curFuncResultDemands.find(f);
        if (resIt != curFuncResultDemands.end()) {
          Nullability resNull =
            (sig.results.isNullable() || resIt->second.acceptsNull)
              ? Nullable
              : NonNullable;
          newResults = Type(resIt->second.ht, resNull);
        }
        if (newResults != sig.results) {
          resultChangedFuncs.insert(f);
        }
        Signature newSig(Type(newParams), newResults);
        HeapType oldHT = f->type.getHeapType();
        if (candidateSigSet.count(oldHT)) {
          Signature htTargetSig =
            updates.count(oldHT) ? updates[oldHT] : oldHT.getSignature();
          if (newSig == htTargetSig) {
            continue;
          }
        } else if (newSig == sig && !oldHT.getDeclaredSuperType() &&
                   subTypes.getImmediateSubTypes(oldHT).empty()) {
          continue;
        }
        auto it = targetSigToHT.find(newSig);
        if (it != targetSigToHT.end()) {
          f->type = f->type.with(it->second);
        } else {
          HeapType freshHT(newSig);
          targetSigToHT.emplace(newSig, freshHT);
          f->type = f->type.with(freshHT);
          updates.try_emplace(freshHT, newSig);
        }
      }
    }

    GlobalTypeRewriter::updateSignatures(
      updates, *module, getPassOptions().worldMode);

    for (auto* f : resultChangedFuncs) {
      Type finalRes = f->getResults();
      for (auto* call : callsTo[f]) {
        if (call->type != Type::unreachable) {
          call->type = finalRes;
        }
      }
    }
    for (auto sigHT : resultChangedSigs) {
      if (!sigFuncs[sigHT].empty()) {
        Type finalRes = sigFuncs[sigHT][0]->getResults();
        for (auto* cr : callRefsTo[sigHT]) {
          if (cr->type != Type::unreachable) {
            cr->type = finalRes;
          }
        }
      }
    }

    // 2 + 3. Relax locals, un-alias cast temporaries, and fix up casts in
    // every function.
    std::string modeArg = getArgumentOrDefault("relax-nullability", "");
    bool widenFields =
      widenTypes &&
      getPassOptions().worldMode == WorldMode::Closed &&
      (hasArgument("relax-nullability-widen-fields") ||
       hasArgument("relax-nullability-merge-signatures") ||
       modeArg == "widen-fields" || modeArg == "merge-signatures" ||
       modeArg == "cache-param-casts");
    FunctionRelaxer(widenTypes, widenFields).run(getPassRunner(), module);
  }

  void runGeneralizeSignatures(Module* module, bool widenFields) {
    auto types = ModuleUtils::collectHeapTypes(*module);
    std::unordered_set<HeapType> fixed;
    std::unordered_map<HeapType, std::vector<Function*>> sigFuncs;
    std::unordered_set<Name> escapedFuncs;
    std::unordered_map<Function*, std::vector<Call*>> callsTo;
    std::unordered_set<Name> hasReturnCall;
    std::unordered_set<HeapType> sigUsedAsRef;
    collectModuleEscapeAndSigInfo(module,
                                  types,
                                  fixed,
                                  sigFuncs,
                                  escapedFuncs,
                                  callsTo,
                                  hasReturnCall,
                                  sigUsedAsRef);
    generalizeSignatures(
      module, escapedFuncs, hasReturnCall, fixed, sigUsedAsRef, widenFields);
  }

  void run(Module* module) override {
    if (!getPassOptions().trapsNeverHappen) {
      // Without this assumption removing casts changes behavior.
      return;
    }

    if (getPassOptions().shrinkLevel >= 2) {
      PassRunner runner(getPassRunner());
      runner.setIsNested(true);
      runner.add("ssa-nomerge");
      runner.run();
    }

    std::string modeArg = getArgumentOrDefault("relax-nullability", "");
    bool widenTypes =
      getPassOptions().shrinkLevel >= 2 || !getPassOptions().optimizeLevel;
    bool widenFields =
      widenTypes &&
      getPassOptions().worldMode == WorldMode::Closed &&
      (hasArgument("relax-nullability-widen-fields") ||
       hasArgument("relax-nullability-merge-signatures") ||
       modeArg == "widen-fields" || modeArg == "merge-signatures" ||
       modeArg == "cache-param-casts");

    // 1 + 2 + 3. Relax and widen function signatures, locals, and casts.
    relaxSignaturesAndLocals(module, widenTypes);

    // 4. Widen private struct reference fields and array globals if enabled.
    if (widenFields && widenFieldsAndArrays(module)) {
      relaxSignaturesAndLocals(module, widenTypes);
    }

    // 5. Whole-signature reference generalization for remaining private
    // signatures where profitable.
    if (widenTypes) {
      runGeneralizeSignatures(module, widenFields);
    }

    // 6. Second pass of field/array widening after signature generalization.
    if (widenFields && widenFieldsAndArrays(module)) {
      relaxSignaturesAndLocals(module, widenTypes);
      widenFieldsAndArrays(module);
      FunctionRelaxer(widenTypes, widenFields).run(getPassRunner(), module);
      runGeneralizeSignatures(module, widenFields);
      if (widenFieldsAndArrays(module)) {
        FunctionRelaxer(widenTypes, widenFields).run(getPassRunner(), module);
      }
    }

    // 7. Final sibling-targeted cast-inserting field widening and mutability
    // unification before TypeMerging, followed by signature/result convergence.
    bool cacheParamCasts =
      hasArgument("relax-nullability-cache-param-casts") ||
      modeArg == "cache-param-casts" || !getPassOptions().optimizeLevel;
    if (widenFields && cacheParamCasts &&
        widenFieldsAndArrays(module, /*allowSiblingCastWiden=*/true)) {
      relaxSignaturesAndLocals(module, widenTypes);
      runGeneralizeSignatures(module, widenFields);
      FunctionRelaxer(widenTypes, widenFields).run(getPassRunner(), module);
    }
  }
};

} // anonymous namespace

Pass* createRelaxNullabilityPass() { return new RelaxNullability(); }

} // namespace wasm
