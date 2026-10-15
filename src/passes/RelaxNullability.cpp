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
#include "ir/find_all.h"
#include "ir/intrinsics.h"
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

using SigParamHTMap =
  std::unordered_map<HeapType, std::vector<std::optional<HeapType>>>;

// Computes for each local in `func` the highest declared supertype of its
// current type that satisfies all consumers in `func`.
struct DemandAnalyzer : public ChildTyper<DemandAnalyzer> {
  const SigParamHTMap* sigParamHTs;
  bool useSigParamsForSet;
  Index numParams;
  std::vector<bool> isCandidate;
  std::vector<HeapType> origHT;
  std::vector<HeapType> rootHT;
  std::vector<HeapType> curHT;
  std::vector<HeapType> nextHT;
  std::unordered_map<Name, HeapType> labelTypes;

  DemandAnalyzer(Module& wasm,
                 Function* func,
                 const SigParamHTMap* sigParamHTs,
                 bool allowParams,
                 bool useSigParamsForSet = false)
    : ChildTyper(wasm, func), sigParamHTs(sigParamHTs),
      useSigParamsForSet(useSigParamsForSet),
      numParams(func->getNumParams()) {
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
            ht.getDeclaredSuperType()) {
          isCandidate[i] = true;
          rootHT[i] = getRootAncestor(ht);
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
    if (!anyCandidate) {
      return curHT;
    }
    auto results = func->getResults();
    std::optional<HeapType> bodyReq;
    if (results.isRef()) {
      bodyReq = results.getHeapType();
    }
    for (int iter = 0; iter < 3; ++iter) {
      nextHT = rootHT;
      labelTypes.clear();
      walk(func->body, bodyReq);
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
          walk(child, type->getHeapType());
          return;
        } else if (!type->isRef()) {
          walk(child, std::nullopt);
          return;
        }
      }
    }
    if (child->type.isRef()) {
      walk(child, child->type.getHeapType());
    } else {
      walk(child, std::nullopt);
    }
  }

  Type getLabelType(Name label) { WASM_UNREACHABLE("unexpected branch"); }

  void noteUnknown() {}

  void visitStructGet(StructGet* curr) {
    if (!curr->ref->type.isStruct()) {
      walk(curr->ref, std::nullopt);
      return;
    }
    auto anc =
      getHighestDefiningAncestor(curr->ref->type.getHeapType(), curr->index);
    ChildTyper<DemandAnalyzer>::visitStructGet(curr, anc);
  }

  void visitStructSet(StructSet* curr) {
    if (!curr->ref->type.isStruct()) {
      walk(curr->ref, std::nullopt);
      walk(curr->value, std::nullopt);
      return;
    }
    auto anc =
      getHighestDefiningAncestor(curr->ref->type.getHeapType(), curr->index);
    ChildTyper<DemandAnalyzer>::visitStructSet(curr, anc);
  }

  void visitRefIsNull(RefIsNull* curr) {
    if (curr->value->type.isRef()) {
      walk(curr->value, curr->value->type.getHeapType().getTop());
    } else {
      walk(curr->value, std::nullopt);
    }
  }

  void visitRefEq(RefEq* curr) {
    if (curr->left->type.isRef()) {
      walk(curr->left,
           HeapTypes::eq.getBasic(
             curr->left->type.getHeapType().getShared()));
    } else {
      walk(curr->left, std::nullopt);
    }
    if (curr->right->type.isRef()) {
      walk(curr->right,
           HeapTypes::eq.getBasic(
             curr->right->type.getHeapType().getShared()));
    } else {
      walk(curr->right, std::nullopt);
    }
  }

  void visitDrop(Drop* curr) {
    if (curr->value->type.isRef()) {
      walk(curr->value, curr->value->type.getHeapType().getTop());
    } else {
      walk(curr->value, std::nullopt);
    }
  }

  void walk(Expression* curr, std::optional<HeapType> reqHT) {
    if (auto* get = curr->dynCast<LocalGet>()) {
      if (reqHT) {
        requireLocal(get->index, *reqHT);
      } else if (get->type.isRef()) {
        requireLocal(get->index, get->type.getHeapType());
      }
      return;
    }
    if (auto* set = curr->dynCast<LocalSet>()) {
      std::optional<HeapType> valReq;
      if (func->getLocalType(set->index).isRef()) {
        valReq = curHT[set->index];
        if (useSigParamsForSet && set->index < numParams && sigParamHTs) {
          auto it = sigParamHTs->find(func->type.getHeapType());
          if (it != sigParamHTs->end() && set->index < it->second.size() &&
              it->second[set->index]) {
            valReq = *it->second[set->index];
          }
        }
      }
      if (set->isTee() && set->type.isRef()) {
        HeapType teeReq = reqHT ? *reqHT : set->type.getHeapType();
        requireLocal(set->index, teeReq);
        if (valReq && !HeapType::isSubType(*valReq, teeReq)) {
          valReq = tightenAncestor(origHT[set->index], *valReq, teeReq);
        }
      }
      walk(set->value, valReq);
      return;
    }
    if (auto* as = curr->dynCast<RefAs>()) {
      if (as->op == RefAsNonNull) {
        walk(as->value,
             reqHT ? reqHT
                   : (as->type.isRef()
                        ? std::optional<HeapType>(as->type.getHeapType())
                        : std::nullopt));
        return;
      }
    }
    if (auto* cast = curr->dynCast<RefCast>()) {
      if (cast->desc) {
        walk(cast->desc,
             cast->desc->type.isRef()
               ? std::optional<HeapType>(cast->desc->type.getHeapType())
               : std::nullopt);
      }
      if (!cast->ref->type.isRef()) {
        walk(cast->ref, std::nullopt);
        return;
      }
      HeapType top = cast->ref->type.getHeapType().getTop();
      if (!cast->desc && reqHT && !cast->ref->type.isNull() &&
          HeapType::isSubType(cast->ref->type.getHeapType(), *reqHT)) {
        walk(cast->ref, *reqHT);
      } else {
        walk(cast->ref, top);
      }
      return;
    }
    if (auto* call = curr->dynCast<Call>()) {
      auto* callee = wasm.getFunction(call->target);
      auto params = callee->getParams();
      const std::vector<std::optional<HeapType>>* curParams = nullptr;
      if (sigParamHTs) {
        auto it = sigParamHTs->find(callee->type.getHeapType());
        if (it != sigParamHTs->end()) {
          curParams = &it->second;
        }
      }
      for (size_t i = 0; i < call->operands.size(); ++i) {
        std::optional<HeapType> argReq;
        if (curParams && i < curParams->size()) {
          argReq = (*curParams)[i];
        } else if (i < params.size() && params[i].isRef()) {
          argReq = params[i].getHeapType();
        }
        walk(call->operands[i], argReq);
      }
      return;
    }
    if (auto* callRef = curr->dynCast<CallRef>()) {
      if (!callRef->target->type.isSignature()) {
        for (auto* op : callRef->operands) {
          walk(op,
               op->type.isRef()
                 ? std::optional<HeapType>(op->type.getHeapType())
                 : std::nullopt);
        }
        walk(callRef->target, std::nullopt);
        return;
      }
      HeapType sigHT = callRef->target->type.getHeapType();
      auto params = sigHT.getSignature().params;
      const std::vector<std::optional<HeapType>>* curParams = nullptr;
      if (sigParamHTs) {
        auto it = sigParamHTs->find(sigHT);
        if (it != sigParamHTs->end()) {
          curParams = &it->second;
        }
      }
      for (size_t i = 0; i < callRef->operands.size(); ++i) {
        std::optional<HeapType> argReq;
        if (curParams && i < curParams->size()) {
          argReq = (*curParams)[i];
        } else if (i < params.size() && params[i].isRef()) {
          argReq = params[i].getHeapType();
        }
        walk(callRef->operands[i], argReq);
      }
      walk(callRef->target, sigHT);
      return;
    }
    if (auto* block = curr->dynCast<Block>()) {
      if (block->list.empty()) {
        return;
      }
      std::optional<HeapType> tailReq = reqHT;
      if (block->name.is()) {
        tailReq = block->type.isRef()
                    ? std::optional<HeapType>(block->type.getHeapType())
                    : std::nullopt;
        if (tailReq) {
          labelTypes[block->name] = *tailReq;
        }
      }
      for (size_t i = 0; i + 1 < block->list.size(); ++i) {
        walk(block->list[i], std::nullopt);
      }
      walk(block->list.back(), tailReq);
      return;
    }
    if (auto* iff = curr->dynCast<If>()) {
      walk(iff->condition, std::nullopt);
      std::optional<HeapType> armReq =
        reqHT ? reqHT
              : (iff->type.isRef()
                   ? std::optional<HeapType>(iff->type.getHeapType())
                   : std::nullopt);
      walk(iff->ifTrue, armReq);
      if (iff->ifFalse) {
        walk(iff->ifFalse, armReq);
      }
      return;
    }
    if (auto* loop = curr->dynCast<Loop>()) {
      std::optional<HeapType> bodyReq =
        loop->type.isRef() ? std::optional<HeapType>(loop->type.getHeapType())
                           : std::nullopt;
      walk(loop->body, bodyReq);
      return;
    }
    if (auto* try_ = curr->dynCast<Try>()) {
      std::optional<HeapType> bodyReq =
        try_->type.isRef() ? std::optional<HeapType>(try_->type.getHeapType())
                           : std::nullopt;
      walk(try_->body, bodyReq);
      for (auto* catchBody : try_->catchBodies) {
        walk(catchBody, bodyReq);
      }
      return;
    }
    if (auto* tryTable = curr->dynCast<TryTable>()) {
      std::optional<HeapType> bodyReq =
        tryTable->type.isRef()
          ? std::optional<HeapType>(tryTable->type.getHeapType())
          : std::nullopt;
      walk(tryTable->body, bodyReq);
      return;
    }
    if (auto* br = curr->dynCast<Break>()) {
      if (br->condition) {
        walk(br->condition, std::nullopt);
      }
      if (br->value) {
        std::optional<HeapType> valReq;
        if (auto it = labelTypes.find(br->name); it != labelTypes.end()) {
          valReq = it->second;
        } else if (br->value->type.isRef()) {
          valReq = br->value->type.getHeapType();
        }
        walk(br->value, valReq);
      }
      return;
    }
    if (auto* sw = curr->dynCast<Switch>()) {
      walk(sw->condition, std::nullopt);
      if (sw->value) {
        walk(sw->value,
             sw->value->type.isRef()
               ? std::optional<HeapType>(sw->value->type.getHeapType())
               : std::nullopt);
      }
      return;
    }
    if (auto* brOn = curr->dynCast<BrOn>()) {
      if (brOn->desc) {
        walk(brOn->desc,
             brOn->desc->type.isRef()
               ? std::optional<HeapType>(brOn->desc->type.getHeapType())
               : std::nullopt);
      }
      walk(brOn->ref,
           brOn->ref->type.isRef()
             ? std::optional<HeapType>(brOn->ref->type.getHeapType())
             : std::nullopt);
      return;
    }
    if (auto* sel = curr->dynCast<Select>()) {
      walk(sel->condition, std::nullopt);
      std::optional<HeapType> armReq =
        reqHT ? reqHT
              : (sel->type.isRef()
                   ? std::optional<HeapType>(sel->type.getHeapType())
                   : std::nullopt);
      walk(sel->ifTrue, armReq);
      walk(sel->ifFalse, armReq);
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

  FunctionRelaxer(bool widenTypes = true) : widenTypes(widenTypes) {}

  bool isFunctionParallel() override { return true; }

  // We only make locals more nullable / wider, which cannot break validation.
  bool requiresNonNullableLocalFixups() override { return false; }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<FunctionRelaxer>(widenTypes);
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
    }
    if (auto* iff = (*exprp)->dynCast<If>()) {
      if (iff->ifFalse && iff->type.isRef()) {
        Type armDemanded(demanded.getHeapType(), iff->type.getNullability());
        bool c1 = stripTrailingCasts(&iff->ifTrue, armDemanded, false);
        bool c2 = stripTrailingCasts(&iff->ifFalse, armDemanded, false);
        if (c1 || c2) {
          iff->finalize();
          return true;
        }
        return false;
      }
    }
    if (auto* sel = (*exprp)->dynCast<Select>()) {
      if (sel->type.isRef()) {
        Type armDemanded(demanded.getHeapType(), sel->type.getNullability());
        bool c1 = stripTrailingCasts(&sel->ifTrue, armDemanded, false);
        bool c2 = stripTrailingCasts(&sel->ifFalse, armDemanded, false);
        if (c1 || c2) {
          sel->finalize();
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
    if (stripped && demandsNonNull && (*exprp)->type.isRef() &&
        (*exprp)->type.isNullable()) {
      *exprp = Builder(*getModule()).makeRefAs(RefAsNonNull, *exprp);
    }
    return stripped;
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
        DemandAnalyzer analyzer(*getModule(), func, nullptr, false);
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
        func->body = Builder(*getModule()).makeRefAs(RefAsNonNull, func->body);
      }
    }
  }

  struct Typer : public ChildTyper<Typer> {
    FunctionRelaxer& parent;
    bool changed = false;

    Typer(FunctionRelaxer& parent)
      : ChildTyper(*parent.getModule(), parent.getFunction()), parent(parent) {}

    using ChildTyper<Typer>::note;

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
        *childp = Builder(*parent.getModule()).makeRefAs(RefAsNonNull, *childp);
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

struct RelaxNullability : public Pass {
  // Signatures and locals change, and all casts are revisited.
  bool invalidatesDWARF() override { return true; }

  void run(Module* module) override {
    if (!getPassOptions().trapsNeverHappen) {
      // Without this assumption removing casts changes behavior.
      return;
    }

    bool widenTypes =
      getPassOptions().shrinkLevel >= 2 || !getPassOptions().optimizeLevel;

    // 1. Relax and widen the parameters of private function types.
    std::unordered_set<HeapType> fixed;
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

    auto types = ModuleUtils::collectHeapTypes(*module);
    SubTypes subTypes(types);

    std::unordered_map<HeapType, std::vector<Function*>> sigFuncs;
    if (widenTypes) {
      for (auto& func : module->functions) {
        if (!func->imported()) {
          sigFuncs[func->type.getHeapType()].push_back(func.get());
        }
      }
    }

    std::vector<HeapType> candidateSigs;
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
      if (widenTypes && !sigFuncs[type].empty()) {
        auto sig = type.getSignature();
        std::vector<std::optional<HeapType>> paramHTs(sig.params.size());
        for (size_t i = 0; i < sig.params.size(); ++i) {
          if (sig.params[i].isRef()) {
            paramHTs[i] = sig.params[i].getHeapType();
          }
        }
        curSigParamHTs[type] = std::move(paramHTs);
      }
    }

    if (widenTypes) {
      // Propagate parameter demands across direct and indirect calls for up to 4
      // rounds, followed by 1 verification pass with shared signature parameter
      // types applied to parameter local.sets.
      for (int round = 0; round < 5; ++round) {
        bool verifyOnly = (round == 4);
        ModuleUtils::ParallelFunctionAnalysis<std::vector<HeapType>> analysis(
          *module, [&](Function* func, std::vector<HeapType>& allowedHTs) {
            if (func->imported()) {
              return;
            }
            bool allowParams =
              curSigParamHTs.count(func->type.getHeapType()) != 0;
            DemandAnalyzer analyzer(
              *module, func, &curSigParamHTs, allowParams, verifyOnly);
            allowedHTs = analyzer.analyze();
          });

        bool anyChanged = false;
        for (auto type : candidateSigs) {
          auto it = curSigParamHTs.find(type);
          if (it == curSigParamHTs.end()) {
            continue;
          }
          const auto& funcs = sigFuncs[type];
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
              const auto& fAllowed = analysis.map[f];
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
        if (!anyChanged && !verifyOnly) {
          // Jump straight to verification pass.
          round = 3;
        }
      }
    }

    GlobalTypeRewriter::SignatureUpdates updates;
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
      if (changed) {
        updates[type] = Signature(Type(params), sig.results);
      }
    }
    GlobalTypeRewriter::updateSignatures(
      updates, *module, getPassOptions().worldMode);

    // 2 + 3. Relax locals and fix up casts in every function.
    FunctionRelaxer(widenTypes).run(getPassRunner(), module);
  }
};

} // anonymous namespace

Pass* createRelaxNullabilityPass() { return new RelaxNullability(); }

} // namespace wasm
