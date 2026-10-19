/*
 * Copyright 2024 WebAssembly Community Group participants
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
// Optimizes heap (GC) stores.
//
// TODO: Add dead store elimination / load forwarding here.
//

#include "cfg/cfg-traversal.h"
#include "ir/effects.h"
#include "ir/iteration.h"
#include "ir/local-graph.h"
#include "ir/manipulation.h"
#include "ir/module-utils.h"
#include "ir/properties.h"
#include "ir/utils.h"
#include "pass.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

static bool isDefaultFieldValue(Expression* val, Type fieldType) {
  if (!val || val->type == Type::unreachable) {
    return false;
  }
  if (fieldType.isRef()) {
    return val->is<RefNull>();
  }
  if (auto* c = val->dynCast<Const>()) {
    return c->value == Literal::makeZero(fieldType);
  }
  return false;
}

static bool isSameConstantOrDefaultValue(Expression* a,
                                         Expression* b,
                                         Type fieldType) {
  if (!a || !b || a->type == Type::unreachable ||
      b->type == Type::unreachable) {
    return false;
  }
  if (isDefaultFieldValue(a, fieldType) && isDefaultFieldValue(b, fieldType)) {
    return true;
  }
  if (auto* ca = a->dynCast<Const>()) {
    if (auto* cb = b->dynCast<Const>()) {
      return ca->value == cb->value;
    }
  }
  if (auto* rfa = a->dynCast<RefFunc>()) {
    if (auto* rfb = b->dynCast<RefFunc>()) {
      return rfa->func == rfb->func;
    }
  }
  return false;
}

struct AllocHelperInfo {
  Function* func = nullptr;
  HeapType ht;
  bool isWithDefault = false;
  uint64_t defaultFieldsMask = 0;
  std::vector<int> holeParamIndex;
  std::vector<Expression*> constOperands;
  std::optional<EffectAnalyzer> snEffects;
};

enum class EdgeResultUsage {
  Dropped,
  TailReturned,
  Other,
};

struct CallEdge {
  Name callee;
  Index calleeParam;
  EdgeResultUsage usage;
};

struct ParamSummary {
  bool modifiesLocal = false;
  bool escapes = false;
  bool mayReturnParam = false;
  bool alwaysReturnsParam = false;
  uint64_t nonDefaultWrittenFields = 0;
  std::vector<CallEdge> callEdges;
};

struct FreshInfo {
  bool unescaped = false;
  uint64_t defaultFields = 0;
};

struct LocalState {
  bool unreachable = false;
  std::vector<bool> unescaped;
  std::vector<uint64_t> defaultFields;
  std::vector<uint64_t> localVersion;

  LocalState() = default;
  explicit LocalState(Index numLocals)
    : unescaped(numLocals, false), defaultFields(numLocals, 0),
      localVersion(numLocals, 0) {}
};

static void mergeState(LocalState& dst, const LocalState& src) {
  if (src.unreachable) {
    for (Index i = 0; i < dst.localVersion.size() && i < src.localVersion.size();
         i++) {
      dst.localVersion[i] = std::max(dst.localVersion[i], src.localVersion[i]);
    }
    return;
  }
  if (dst.unreachable) {
    auto oldVers = dst.localVersion;
    dst = src;
    for (Index i = 0; i < dst.localVersion.size() && i < oldVers.size(); i++) {
      dst.localVersion[i] = std::max(dst.localVersion[i], oldVers[i]);
    }
    return;
  }
  for (Index i = 0; i < dst.unescaped.size() && i < src.unescaped.size(); i++) {
    dst.unescaped[i] = dst.unescaped[i] && src.unescaped[i];
    dst.defaultFields[i] =
      dst.unescaped[i] ? (dst.defaultFields[i] & src.defaultFields[i]) : 0;
    dst.localVersion[i] = std::max(dst.localVersion[i], src.localVersion[i]);
  }
}

struct ModuleStoreInfo {
  bool trapsNeverHappen = false;

  std::unordered_set<Name> escapedFuncs;
  std::unordered_map<Name, AllocHelperInfo> allocHelpers;
  std::unordered_set<Name> castHelpers;
  std::unordered_set<Name> funcsWithReturn;
  std::unordered_map<Name, Index> expectedCallSiteCount;
  std::unordered_map<Name, std::vector<ParamSummary>> paramSummaries;

  // Per-function entry state for parameters:
  std::unordered_map<Name, std::vector<bool>> entryIsUnescaped;
  std::unordered_map<Name, std::vector<uint64_t>> entryDefaultFields;

  // Per-function return state:
  std::unordered_map<Name, bool> returnIsFresh;
  std::unordered_map<Name, uint64_t> returnDefaultFields;

  // Scratch state during call-site analysis iteration:
  struct PendingCallTargetInfo {
    Index visitedCalls = 0;
    std::vector<bool> unescaped;
    std::vector<uint64_t> defaultFields;
  };
  std::unordered_map<Name, PendingCallTargetInfo> pendingCalls;

  ModuleStoreInfo() = default;

  void clear() {
    escapedFuncs.clear();
    allocHelpers.clear();
    castHelpers.clear();
    funcsWithReturn.clear();
    expectedCallSiteCount.clear();
    paramSummaries.clear();
    entryIsUnescaped.clear();
    entryDefaultFields.clear();
    returnIsFresh.clear();
    returnDefaultFields.clear();
    pendingCalls.clear();
  }

  const AllocHelperInfo* getAllocHelper(Name name) const {
    auto it = allocHelpers.find(name);
    return it != allocHelpers.end() ? &it->second : nullptr;
  }

  const ParamSummary* getParamSummary(Name name, Index paramIdx) const {
    auto it = paramSummaries.find(name);
    if (it == paramSummaries.end() || paramIdx >= it->second.size()) {
      return nullptr;
    }
    return &it->second[paramIdx];
  }

  bool isReturnFresh(Name name) const {
    auto it = returnIsFresh.find(name);
    return it != returnIsFresh.end() && it->second;
  }

  uint64_t getReturnDefaultFields(Name name) const {
    auto it = returnDefaultFields.find(name);
    return it != returnDefaultFields.end() ? it->second : 0;
  }

  uint64_t getAllocCallDefaultMask(Call* call,
                                   const AllocHelperInfo& helper) const {
    uint64_t mask = helper.defaultFieldsMask;
    auto& fields = helper.ht.getStruct().fields;
    Index limit = std::min<Index>(fields.size(), 64);
    for (Index idx = 0; idx < limit; idx++) {
      int k = helper.holeParamIndex[idx];
      if (k >= 0 && Index(k) < call->operands.size() &&
          isDefaultFieldValue(call->operands[k], fields[idx].type)) {
        mask |= (uint64_t(1) << idx);
      }
    }
    return mask;
  }

  Expression* unwrapCasts(Expression* expr,
                          bool allowTrappingCasts = true) const {
    while (expr) {
      if (auto* rc = expr->dynCast<RefCast>()) {
        if (!rc->desc && allowTrappingCasts) {
          expr = rc->ref;
          continue;
        }
      }
      if (auto* ra = expr->dynCast<RefAs>()) {
        if (ra->op == RefAsNonNull) {
          expr = ra->value;
          continue;
        }
      }
      if (auto* c = expr->dynCast<Call>()) {
        if (allowTrappingCasts && c->operands.size() == 1 &&
            castHelpers.count(c->target)) {
          expr = c->operands[0];
          continue;
        }
      }
      break;
    }
    return expr;
  }

  void analyzeExprForLocal(Expression* expr,
                           Index localIdx,
                           bool inTailReturn,
                           bool inDropped,
                           ParamSummary& sum) const {
    if (!expr) {
      return;
    }
    if (auto* lg = expr->dynCast<LocalGet>()) {
      if (lg->index == localIdx) {
        if (inTailReturn) {
          sum.mayReturnParam = true;
        } else if (!inDropped) {
          sum.escapes = true;
        }
      }
      return;
    }
    if (auto* ls = expr->dynCast<LocalSet>()) {
      if (ls->index == localIdx) {
        Expression* innerVal = unwrapCasts(ls->value, trapsNeverHappen);
        if (auto* lg = innerVal->dynCast<LocalGet>()) {
          if (lg->index == localIdx) {
            if (ls->isTee() && !inDropped) {
              if (inTailReturn) {
                sum.mayReturnParam = true;
              } else {
                sum.escapes = true;
              }
            }
            return;
          }
        }
        sum.modifiesLocal = true;
        sum.escapes = true;
      }
      analyzeExprForLocal(ls->value, localIdx, false, false, sum);
      return;
    }
    if (auto* sg = expr->dynCast<StructGet>()) {
      Expression* innerRef = unwrapCasts(sg->ref, trapsNeverHappen);
      if (auto* lg = innerRef->dynCast<LocalGet>()) {
        if (lg->index == localIdx) {
          return;
        }
      }
      analyzeExprForLocal(sg->ref, localIdx, false, false, sum);
      return;
    }
    if (auto* rn = expr->dynCast<RefIsNull>()) {
      Expression* inner = unwrapCasts(rn->value, trapsNeverHappen);
      if (auto* lg = inner->dynCast<LocalGet>()) {
        if (lg->index == localIdx) {
          return;
        }
      }
      analyzeExprForLocal(rn->value, localIdx, false, false, sum);
      return;
    }
    if (auto* rt = expr->dynCast<RefTest>()) {
      Expression* inner = unwrapCasts(rt->ref, trapsNeverHappen);
      if (auto* lg = inner->dynCast<LocalGet>()) {
        if (lg->index == localIdx) {
          return;
        }
      }
      analyzeExprForLocal(rt->ref, localIdx, false, false, sum);
      return;
    }
    if (expr->is<RefCast>() || expr->is<RefAs>()) {
      Expression* inner = unwrapCasts(expr, trapsNeverHappen);
      if (auto* lg = inner->dynCast<LocalGet>()) {
        if (lg->index == localIdx) {
          if (inTailReturn) {
            sum.mayReturnParam = true;
          } else if (!inDropped) {
            sum.escapes = true;
          }
          return;
        }
      }
    }
    if (auto* drop = expr->dynCast<Drop>()) {
      analyzeExprForLocal(drop->value, localIdx, false, true, sum);
      return;
    }
    if (auto* ret = expr->dynCast<Return>()) {
      if (ret->value) {
        analyzeExprForLocal(ret->value, localIdx, true, false, sum);
      }
      return;
    }
    if (auto* ss = expr->dynCast<StructSet>()) {
      Expression* innerRef = unwrapCasts(ss->ref, trapsNeverHappen);
      if (auto* lg = innerRef->dynCast<LocalGet>()) {
        if (lg->index == localIdx) {
          if (ss->ref->type != Type::unreachable && !ss->ref->type.isNull() &&
              ss->ref->type.isRef()) {
            auto& fields = ss->ref->type.getHeapType().getStruct().fields;
            if (ss->index < fields.size() && ss->index < 64) {
              if (!isDefaultFieldValue(ss->value, fields[ss->index].type)) {
                sum.nonDefaultWrittenFields |= (uint64_t(1) << ss->index);
              }
            } else {
              sum.nonDefaultWrittenFields = ~uint64_t(0);
            }
          } else {
            sum.nonDefaultWrittenFields = ~uint64_t(0);
          }
        }
      } else {
        analyzeExprForLocal(ss->ref, localIdx, false, false, sum);
      }
      analyzeExprForLocal(ss->value, localIdx, false, false, sum);
      return;
    }
    if (auto* call = expr->dynCast<Call>()) {
      if (trapsNeverHappen && call->operands.size() == 1 &&
          castHelpers.count(call->target)) {
        Expression* inner = unwrapCasts(call, true);
        if (auto* lg = inner->dynCast<LocalGet>()) {
          if (lg->index == localIdx) {
            if (inTailReturn) {
              sum.mayReturnParam = true;
            } else if (!inDropped) {
              sum.escapes = true;
            }
            return;
          }
        }
      }
      Index matchCount = 0;
      for (Index m = 0; m < call->operands.size(); m++) {
        Expression* inner =
          unwrapCasts(call->operands[m], trapsNeverHappen);
        if (auto* lg = inner->dynCast<LocalGet>()) {
          if (lg->index == localIdx) {
            matchCount++;
          }
        }
      }
      EdgeResultUsage usage = inDropped      ? EdgeResultUsage::Dropped
                              : inTailReturn ? EdgeResultUsage::TailReturned
                                             : EdgeResultUsage::Other;
      for (Index m = 0; m < call->operands.size(); m++) {
        Expression* inner =
          unwrapCasts(call->operands[m], trapsNeverHappen);
        if (auto* lg = inner->dynCast<LocalGet>()) {
          if (lg->index == localIdx) {
            if (matchCount > 1) {
              sum.escapes = true;
            } else {
              sum.callEdges.push_back({call->target, m, usage});
            }
          }
          continue;
        }
        analyzeExprForLocal(call->operands[m], localIdx, false, false, sum);
      }
      return;
    }
    if (auto* blk = expr->dynCast<Block>()) {
      for (Index i = 0; i < blk->list.size(); i++) {
        bool isLast = (i + 1 == blk->list.size());
        analyzeExprForLocal(blk->list[i],
                            localIdx,
                            (isLast && !blk->name.is()) ? inTailReturn : false,
                            isLast ? inDropped : false,
                            sum);
      }
      return;
    }
    if (auto* iff = expr->dynCast<If>()) {
      analyzeExprForLocal(iff->condition, localIdx, false, false, sum);
      analyzeExprForLocal(iff->ifTrue, localIdx, inTailReturn, inDropped, sum);
      if (iff->ifFalse) {
        analyzeExprForLocal(
          iff->ifFalse, localIdx, inTailReturn, inDropped, sum);
      }
      return;
    }
    if (auto* lp = expr->dynCast<Loop>()) {
      analyzeExprForLocal(lp->body, localIdx, inTailReturn, inDropped, sum);
      return;
    }
    for (Expression* child : ChildIterator(expr)) {
      analyzeExprForLocal(child, localIdx, false, false, sum);
    }
  }

  void resolveCallEdges(ParamSummary& sum) const {
    for (const auto& edge : sum.callEdges) {
      const ParamSummary* targetSum =
        getParamSummary(edge.callee, edge.calleeParam);
      if (!targetSum) {
        sum.escapes = true;
        continue;
      }
      if (targetSum->escapes) {
        sum.escapes = true;
      }
      if (targetSum->mayReturnParam) {
        if (edge.usage == EdgeResultUsage::TailReturned) {
          sum.mayReturnParam = true;
        } else if (edge.usage == EdgeResultUsage::Other) {
          sum.escapes = true;
        }
      }
      sum.nonDefaultWrittenFields |= targetSum->nonDefaultWrittenFields;
    }
  }

  void recordCallArgument(Name callee, Index paramIdx, FreshInfo argFresh) {
    auto it = pendingCalls.find(callee);
    if (it == pendingCalls.end()) {
      return;
    }
    auto& info = it->second;
    if (paramIdx < info.unescaped.size()) {
      if (!argFresh.unescaped) {
        info.unescaped[paramIdx] = false;
        info.defaultFields[paramIdx] = 0;
      } else if (info.unescaped[paramIdx]) {
        info.defaultFields[paramIdx] &= argFresh.defaultFields;
      }
    }
  }

  void recordCallVisited(Name callee) {
    auto it = pendingCalls.find(callee);
    if (it != pendingCalls.end()) {
      it->second.visitedCalls++;
    }
  }

  void analyze(const PassOptions& options,
               Module& module,
               bool isNested = false);
};

enum class WalkMode {
  AnalyzeCallSites,
  Optimize,
};

struct CallSiteObs {
  Name callee;
  SmallVector<FreshInfo, 4> args;
};

struct FreshDefaultStoreWalker {
  ModuleStoreInfo& moduleInfo;
  const PassOptions& options;
  Module& module;
  Function* func;
  WalkMode mode;
  std::vector<CallSiteObs>* outCallObs = nullptr;
  bool optimizedAny = false;
  std::unordered_map<Name, LocalState> branchExitStates;

  FreshDefaultStoreWalker(ModuleStoreInfo& moduleInfo,
                          const PassOptions& options,
                          Module& module,
                          Function* func,
                          WalkMode mode,
                          std::vector<CallSiteObs>* outCallObs = nullptr)
    : moduleInfo(moduleInfo), options(options), module(module), func(func),
      mode(mode), outCallObs(outCallObs) {}

  FreshInfo run() {
    if (!func->body) {
      return {false, 0};
    }
    Index numLocals = func->getNumLocals();
    LocalState state(numLocals);
    auto itUnesc = moduleInfo.entryIsUnescaped.find(func->name);
    auto itDef = moduleInfo.entryDefaultFields.find(func->name);
    if (itUnesc != moduleInfo.entryIsUnescaped.end() &&
        itDef != moduleInfo.entryDefaultFields.end()) {
      for (Index p = 0;
           p < func->getNumParams() && p < itUnesc->second.size();
           p++) {
        if (itUnesc->second[p]) {
          state.unescaped[p] = true;
          state.defaultFields[p] = itDef->second[p];
        }
      }
    }
    return walkExpr(&func->body,
                    state,
                    /*resultDropped=*/false,
                    /*isTailReturn=*/true,
                    /*assignedToLocal=*/Index(-1));
  }

  void walkNonEscapingRef(Expression** slotp,
                          LocalState& state,
                          Index& outLocal,
                          FreshInfo& outFresh) {
    outLocal = Index(-1);
    outFresh = {false, 0};
    Expression* inner =
      moduleInfo.unwrapCasts(*slotp, options.trapsNeverHappen);
    if (auto* lg = inner->dynCast<LocalGet>()) {
      outLocal = lg->index;
      if (!state.unreachable && lg->index < state.unescaped.size() &&
          state.unescaped[lg->index]) {
        outFresh = {true, state.defaultFields[lg->index]};
      }
      return;
    }
    if (auto* tee = inner->dynCast<LocalSet>()) {
      if (tee->isTee()) {
        if (*slotp == inner) {
          walkExpr(slotp,
                   state,
                   /*resultDropped=*/true,
                   /*isTailReturn=*/false,
                   /*assignedToLocal=*/Index(-1));
        } else {
          walkExpr(&inner,
                   state,
                   /*resultDropped=*/true,
                   /*isTailReturn=*/false,
                   /*assignedToLocal=*/Index(-1));
        }
        outLocal = tee->index;
        if (!state.unreachable && tee->index < state.unescaped.size() &&
            state.unescaped[tee->index]) {
          outFresh = {true, state.defaultFields[tee->index]};
        }
        return;
      }
    }
    outFresh = walkExpr(slotp, state);
  }

  FreshInfo walkExpr(Expression** exprp,
                     LocalState& state,
                     bool resultDropped = false,
                     bool isTailReturn = false,
                     Index assignedToLocal = Index(-1)) {
    Expression* expr = *exprp;
    if (!expr) {
      return {false, 0};
    }
    if (expr->is<Nop>() || expr->is<Const>() || expr->is<RefNull>() ||
        expr->is<RefFunc>() || expr->is<GlobalGet>() ||
        expr->is<StringConst>()) {
      return {false, 0};
    }
    if (expr->is<Unreachable>()) {
      state.unreachable = true;
      return {false, 0};
    }
    if (auto* lg = expr->dynCast<LocalGet>()) {
      if (resultDropped) {
        return {false, 0};
      }
      if (!state.unreachable && isTailReturn &&
          lg->index >= func->getNumParams() &&
          lg->index < state.unescaped.size() && state.unescaped[lg->index]) {
        FreshInfo ret = {true, state.defaultFields[lg->index]};
        state.unescaped[lg->index] = false;
        state.defaultFields[lg->index] = 0;
        return ret;
      }
      if (lg->index < state.unescaped.size()) {
        state.unescaped[lg->index] = false;
        state.defaultFields[lg->index] = 0;
      }
      return {false, 0};
    }
    if (auto* ls = expr->dynCast<LocalSet>()) {
      Expression* innerVal =
        moduleInfo.unwrapCasts(ls->value, options.trapsNeverHappen);
      if (auto* lg = innerVal->dynCast<LocalGet>()) {
        if (lg->index == ls->index) {
          if (ls->isTee() && !resultDropped) {
            if (!state.unreachable && isTailReturn &&
                ls->index >= func->getNumParams() &&
                ls->index < state.unescaped.size() &&
                state.unescaped[ls->index]) {
              FreshInfo ret = {true, state.defaultFields[ls->index]};
              state.unescaped[ls->index] = false;
              state.defaultFields[ls->index] = 0;
              return ret;
            }
            if (ls->index < state.unescaped.size()) {
              state.unescaped[ls->index] = false;
              state.defaultFields[ls->index] = 0;
            }
          }
          return {false, 0};
        }
      }
      FreshInfo valFresh = walkExpr(&ls->value,
                                    state,
                                    /*resultDropped=*/false,
                                    /*isTailReturn=*/false,
                                    /*assignedToLocal=*/ls->index);
      if (ls->index < state.localVersion.size()) {
        state.localVersion[ls->index]++;
      }
      if (!ls->isTee() || resultDropped) {
        if (!state.unreachable && ls->index < state.unescaped.size()) {
          state.unescaped[ls->index] = valFresh.unescaped;
          state.defaultFields[ls->index] =
            valFresh.unescaped ? valFresh.defaultFields : 0;
        }
        return {false, 0};
      }
      if (!state.unreachable && isTailReturn &&
          ls->index >= func->getNumParams() && valFresh.unescaped) {
        if (ls->index < state.unescaped.size()) {
          state.unescaped[ls->index] = false;
          state.defaultFields[ls->index] = 0;
        }
        return valFresh;
      }
      if (ls->index < state.unescaped.size()) {
        state.unescaped[ls->index] = false;
        state.defaultFields[ls->index] = 0;
      }
      return {false, 0};
    }
    if (auto* sg = expr->dynCast<StructGet>()) {
      Index refLocal = Index(-1);
      FreshInfo refFresh;
      walkNonEscapingRef(&sg->ref, state, refLocal, refFresh);
      return {false, 0};
    }
    if (auto* rn = expr->dynCast<RefIsNull>()) {
      Index refLocal = Index(-1);
      FreshInfo refFresh;
      walkNonEscapingRef(&rn->value, state, refLocal, refFresh);
      return {false, 0};
    }
    if (auto* rt = expr->dynCast<RefTest>()) {
      Index refLocal = Index(-1);
      FreshInfo refFresh;
      walkNonEscapingRef(&rt->ref, state, refLocal, refFresh);
      return {false, 0};
    }
    if (auto* rc = expr->dynCast<RefCast>()) {
      if (rc->desc) {
        walkExpr(&rc->ref, state);
        walkExpr(&rc->desc, state);
        return {false, 0};
      }
      FreshInfo inner =
        walkExpr(&rc->ref, state, resultDropped, isTailReturn, assignedToLocal);
      return options.trapsNeverHappen ? inner : FreshInfo{false, 0};
    }
    if (auto* ra = expr->dynCast<RefAs>()) {
      FreshInfo inner = walkExpr(
        &ra->value, state, resultDropped, isTailReturn, assignedToLocal);
      return (ra->op == RefAsNonNull) ? inner : FreshInfo{false, 0};
    }
    if (auto* drop = expr->dynCast<Drop>()) {
      Expression* inner =
        moduleInfo.unwrapCasts(drop->value, options.trapsNeverHappen);
      if (inner->is<Call>()) {
        walkExpr(&drop->value, state, /*resultDropped=*/true);
      } else {
        Index refLocal = Index(-1);
        FreshInfo refFresh;
        walkNonEscapingRef(&drop->value, state, refLocal, refFresh);
      }
      return {false, 0};
    }
    if (auto* sn = expr->dynCast<StructNew>()) {
      for (auto*& op : sn->operands) {
        walkExpr(&op, state);
      }
      if (sn->desc) {
        walkExpr(&sn->desc, state);
        return {false, 0};
      }
      if (state.unreachable || sn->type == Type::unreachable ||
          !sn->type.isRef() || !sn->type.getHeapType().isStruct()) {
        return {false, 0};
      }
      uint64_t mask = 0;
      auto& fields = sn->type.getHeapType().getStruct().fields;
      Index limit = std::min<Index>(fields.size(), 64);
      for (Index idx = 0; idx < limit; idx++) {
        if (sn->isWithDefault() ||
            isDefaultFieldValue(sn->operands[idx], fields[idx].type)) {
          mask |= (uint64_t(1) << idx);
        }
      }
      return {true, mask};
    }
    if (auto* ss = expr->dynCast<StructSet>()) {
      Index refLocal = Index(-1);
      FreshInfo refFresh;
      walkNonEscapingRef(&ss->ref, state, refLocal, refFresh);
      uint64_t refVer =
        (refLocal != Index(-1) && refLocal < state.localVersion.size())
          ? state.localVersion[refLocal]
          : 0;
      walkExpr(&ss->value, state);
      if (!state.unreachable && refLocal != Index(-1) &&
          refLocal < state.unescaped.size() && state.unescaped[refLocal] &&
          state.localVersion[refLocal] == refVer &&
          ss->type != Type::unreachable && ss->ref->type != Type::unreachable &&
          !ss->ref->type.isNull() && ss->ref->type.isRef()) {
        auto& fields = ss->ref->type.getHeapType().getStruct().fields;
        if (ss->index < fields.size() && ss->index < 64 && !ss->isAtomic()) {
          Type fieldType = fields[ss->index].type;
          uint64_t bit = uint64_t(1) << ss->index;
          if (isDefaultFieldValue(ss->value, fieldType)) {
            if (state.defaultFields[refLocal] & bit) {
              if (mode == WalkMode::Optimize) {
                Expression* innerRef =
                  moduleInfo.unwrapCasts(ss->ref, options.trapsNeverHappen);
                if (innerRef == ss->ref || options.trapsNeverHappen) {
                  if (innerRef->is<LocalGet>()) {
                    ExpressionManipulator::nop(ss);
                    optimizedAny = true;
                  } else if (auto* tee = innerRef->dynCast<LocalSet>()) {
                    tee->makeSet();
                    *exprp = tee;
                    optimizedAny = true;
                  }
                }
              }
            }
          } else {
            state.defaultFields[refLocal] &= ~bit;
          }
        } else if (ss->index < 64) {
          state.defaultFields[refLocal] &= ~(uint64_t(1) << ss->index);
        }
      }
      if (ss->type == Type::unreachable) {
        state.unreachable = true;
      }
      return {false, 0};
    }
    if (auto* call = expr->dynCast<Call>()) {
      if (const AllocHelperInfo* helper =
            moduleInfo.getAllocHelper(call->target)) {
        for (auto*& op : call->operands) {
          walkExpr(&op, state);
        }
        if (mode == WalkMode::AnalyzeCallSites) {
          moduleInfo.recordCallVisited(call->target);
        }
        if (state.unreachable || call->type == Type::unreachable) {
          state.unreachable = true;
          return {false, 0};
        }
        return {true, moduleInfo.getAllocCallDefaultMask(call, *helper)};
      }
      if (options.trapsNeverHappen && call->operands.size() == 1 &&
          moduleInfo.castHelpers.count(call->target)) {
        if (mode == WalkMode::AnalyzeCallSites) {
          moduleInfo.recordCallVisited(call->target);
        }
        FreshInfo inner = walkExpr(&call->operands[0],
                                   state,
                                   resultDropped,
                                   isTailReturn,
                                   assignedToLocal);
        if (call->type == Type::unreachable) {
          state.unreachable = true;
          return {false, 0};
        }
        return inner;
      }

      Index numOps = call->operands.size();
      SmallVector<Index, 4> opLocal(numOps);
      SmallVector<FreshInfo, 4> opRvalueFresh(numOps);
      SmallVector<uint64_t, 4> opVer(numOps);
      for (Index m = 0; m < numOps; m++) {
        opLocal[m] = Index(-1);
        opVer[m] = 0;
        walkNonEscapingRef(
          &call->operands[m], state, opLocal[m], opRvalueFresh[m]);
        if (opLocal[m] != Index(-1) &&
            opLocal[m] < state.localVersion.size()) {
          opVer[m] = state.localVersion[opLocal[m]];
        }
      }

      SmallVector<Index, 4> localUseCount(numOps);
      for (Index m = 0; m < numOps; m++) {
        localUseCount[m] = 0;
        if (opLocal[m] != Index(-1)) {
          for (Index k = 0; k < numOps; k++) {
            if (opLocal[k] == opLocal[m]) {
              localUseCount[m]++;
            }
          }
        }
      }

      for (Index m = 0; m < numOps; m++) {
        if (opLocal[m] != Index(-1)) {
          Index x = opLocal[m];
          if (x >= state.unescaped.size() || state.localVersion[x] != opVer[m]) {
            opRvalueFresh[m] = {false, 0};
            opLocal[m] = Index(-1);
          } else if (localUseCount[m] > 1) {
            state.unescaped[x] = false;
            state.defaultFields[x] = 0;
            opRvalueFresh[m] = {false, 0};
          } else {
            bool isUnesc = !state.unreachable && state.unescaped[x];
            opRvalueFresh[m] = {isUnesc,
                                isUnesc ? state.defaultFields[x] : 0};
          }
        } else if (state.unreachable) {
          opRvalueFresh[m] = {false, 0};
        }
      }

      if (mode == WalkMode::AnalyzeCallSites) {
        if (outCallObs) {
          outCallObs->push_back({call->target, opRvalueFresh});
        } else {
          moduleInfo.recordCallVisited(call->target);
          for (Index m = 0; m < numOps; m++) {
            moduleInfo.recordCallArgument(call->target, m, opRvalueFresh[m]);
          }
        }
      }

      FreshInfo callResultFresh = {false, 0};
      if (!state.unreachable && moduleInfo.isReturnFresh(call->target)) {
        callResultFresh = {true,
                           moduleInfo.getReturnDefaultFields(call->target)};
      }

      for (Index m = 0; m < numOps; m++) {
        const ParamSummary* sum =
          moduleInfo.getParamSummary(call->target, m);
        Index x = opLocal[m];
        if (x != Index(-1) && x < state.unescaped.size() &&
            state.unescaped[x]) {
          bool canStayUnescaped =
            sum && !sum->escapes &&
            (!sum->mayReturnParam || resultDropped || isTailReturn ||
             assignedToLocal == x);
          if (canStayUnescaped) {
            state.defaultFields[x] &= ~sum->nonDefaultWrittenFields;
            if (!state.unreachable && sum->alwaysReturnsParam &&
                (isTailReturn || assignedToLocal == x)) {
              callResultFresh = {true, state.defaultFields[x]};
            }
          } else {
            state.unescaped[x] = false;
            state.defaultFields[x] = 0;
          }
        } else if (x == Index(-1) && !state.unreachable &&
                   opRvalueFresh[m].unescaped) {
          if (sum && !sum->escapes && sum->alwaysReturnsParam) {
            callResultFresh = {
              true,
              opRvalueFresh[m].defaultFields & ~sum->nonDefaultWrittenFields};
          }
        }
      }

      if (call->type == Type::unreachable) {
        state.unreachable = true;
        return {false, 0};
      }
      return callResultFresh;
    }
    if (auto* blk = expr->dynCast<Block>()) {
      FreshInfo lastFresh = {false, 0};
      for (Index i = 0; i < blk->list.size(); i++) {
        bool isLast = (i + 1 == blk->list.size());
        lastFresh = walkExpr(&blk->list[i],
                             state,
                             isLast ? resultDropped : false,
                             (isLast && !blk->name.is()) ? isTailReturn : false,
                             (isLast && !blk->name.is()) ? assignedToLocal
                                                         : Index(-1));
      }
      if (blk->name.is()) {
        auto it = branchExitStates.find(blk->name);
        if (it != branchExitStates.end()) {
          mergeState(state, it->second);
          branchExitStates.erase(it);
          lastFresh = {false, 0};
        }
      }
      return lastFresh;
    }
    if (auto* iff = expr->dynCast<If>()) {
      walkExpr(&iff->condition, state);
      LocalState thenState = state;
      LocalState elseState = state;
      FreshInfo thenFresh = walkExpr(&iff->ifTrue,
                                     thenState,
                                     resultDropped,
                                     isTailReturn,
                                     assignedToLocal);
      FreshInfo elseFresh = {false, 0};
      if (iff->ifFalse) {
        elseFresh = walkExpr(&iff->ifFalse,
                             elseState,
                             resultDropped,
                             isTailReturn,
                             assignedToLocal);
      }
      state = std::move(thenState);
      mergeState(state, elseState);
      if (!state.unreachable && iff->ifFalse && thenFresh.unescaped &&
          elseFresh.unescaped) {
        return {true, thenFresh.defaultFields & elseFresh.defaultFields};
      }
      return {false, 0};
    }
    if (auto* lp = expr->dynCast<Loop>()) {
      EffectAnalyzer loopEff(options, module, lp->body);
      for (Index w : loopEff.localsWritten) {
        if (w < state.unescaped.size()) {
          state.unescaped[w] = false;
          state.defaultFields[w] = 0;
          state.localVersion[w]++;
        }
      }
      for (Index x = 0; x < state.unescaped.size(); x++) {
        if (!state.unescaped[x]) {
          continue;
        }
        ParamSummary loopSum;
        moduleInfo.analyzeExprForLocal(lp->body, x, false, false, loopSum);
        moduleInfo.resolveCallEdges(loopSum);
        if (loopSum.modifiesLocal || loopSum.escapes ||
            loopSum.mayReturnParam) {
          state.unescaped[x] = false;
          state.defaultFields[x] = 0;
        } else {
          state.defaultFields[x] &= ~loopSum.nonDefaultWrittenFields;
        }
      }
      FreshInfo bodyFresh = walkExpr(
        &lp->body, state, resultDropped, isTailReturn, assignedToLocal);
      if (lp->name.is()) {
        branchExitStates.erase(lp->name);
      }
      return bodyFresh;
    }
    if (auto* br = expr->dynCast<Break>()) {
      if (br->value) {
        walkExpr(&br->value, state);
      }
      if (br->condition) {
        walkExpr(&br->condition, state);
      }
      if (br->name.is()) {
        auto it = branchExitStates.find(br->name);
        if (it == branchExitStates.end()) {
          branchExitStates.emplace(br->name, state);
        } else {
          mergeState(it->second, state);
        }
      }
      if (!br->condition) {
        state.unreachable = true;
      }
      return {false, 0};
    }
    if (auto* brOn = expr->dynCast<BrOn>()) {
      walkExpr(&brOn->ref, state);
      if (brOn->desc) {
        walkExpr(&brOn->desc, state);
      }
      if (brOn->name.is()) {
        auto it = branchExitStates.find(brOn->name);
        if (it == branchExitStates.end()) {
          branchExitStates.emplace(brOn->name, state);
        } else {
          mergeState(it->second, state);
        }
      }
      return {false, 0};
    }
    if (auto* sw = expr->dynCast<Switch>()) {
      if (sw->value) {
        walkExpr(&sw->value, state);
      }
      walkExpr(&sw->condition, state);
      for (Name target : sw->targets) {
        auto it = branchExitStates.find(target);
        if (it == branchExitStates.end()) {
          branchExitStates.emplace(target, state);
        } else {
          mergeState(it->second, state);
        }
      }
      if (sw->default_.is()) {
        auto it = branchExitStates.find(sw->default_);
        if (it == branchExitStates.end()) {
          branchExitStates.emplace(sw->default_, state);
        } else {
          mergeState(it->second, state);
        }
      }
      state.unreachable = true;
      return {false, 0};
    }
    if (auto* ret = expr->dynCast<Return>()) {
      if (ret->value) {
        walkExpr(&ret->value, state);
      }
      state.unreachable = true;
      return {false, 0};
    }
    if (expr->is<Try>() || expr->is<TryTable>()) {
      for (Index i = 0; i < state.unescaped.size(); i++) {
        state.unescaped[i] = false;
        state.defaultFields[i] = 0;
        state.localVersion[i]++;
      }
      for (auto*& child : ChildIterator(expr)) {
        walkExpr(&child, state);
        for (Index i = 0; i < state.unescaped.size(); i++) {
          state.unescaped[i] = false;
          state.defaultFields[i] = 0;
          state.localVersion[i]++;
        }
      }
      return {false, 0};
    }

    for (auto*& child : ChildIterator(expr)) {
      walkExpr(&child, state);
    }
    if (expr->type == Type::unreachable) {
      state.unreachable = true;
    }
    return {false, 0};
  }
};

void ModuleStoreInfo::analyze(const PassOptions& options,
                              Module& module,
                              bool isNested) {
  clear();
  trapsNeverHappen = options.trapsNeverHappen;

  // 1. Identify single-StructNew allocation helpers and 1-param cast helpers.
  // This only inspects function root nodes and is fast enough to run even in
  // nested FilteredPassRunner invocations.
  for (auto& f : module.functions) {
    if (f->imported() || !f->body) {
      continue;
    }
    if (auto* sn = f->body->dynCast<StructNew>()) {
      if (sn->type != Type::unreachable && sn->type.isRef() &&
          sn->type.getHeapType().isStruct() && !sn->desc) {
        AllocHelperInfo helper;
        helper.func = f.get();
        helper.ht = sn->type.getHeapType();
        helper.isWithDefault = sn->isWithDefault();
        auto& fields = helper.ht.getStruct().fields;
        helper.holeParamIndex.assign(fields.size(), -1);
        helper.constOperands.assign(fields.size(), nullptr);

        EffectAnalyzer snEff(options, module, sn);
        std::vector<Index> paramReads(f->getNumParams(), 0);
        struct ParamReadCounter : public PostWalker<ParamReadCounter> {
          std::vector<Index>& paramReads;
          ParamReadCounter(std::vector<Index>& paramReads)
            : paramReads(paramReads) {}
          void visitLocalGet(LocalGet* curr) {
            if (curr->index < paramReads.size()) {
              paramReads[curr->index]++;
            }
          }
        };
        ParamReadCounter(paramReads).walk(f->body);

        for (Index idx = 0; idx < fields.size(); idx++) {
          if (sn->isWithDefault()) {
            if (idx < 64) {
              helper.defaultFieldsMask |= (uint64_t(1) << idx);
            }
          } else {
            Expression* op = sn->operands[idx];
            if (idx < 64 && isDefaultFieldValue(op, fields[idx].type)) {
              helper.defaultFieldsMask |= (uint64_t(1) << idx);
            }
            if (op->is<Const>() || op->is<RefNull>() || op->is<RefFunc>()) {
              helper.constOperands[idx] =
                ExpressionManipulator::copy(op, module);
            }
            if (snEff.localsWritten.empty()) {
              if (auto* lg = op->dynCast<LocalGet>()) {
                if (lg->index < f->getNumParams() &&
                    paramReads[lg->index] == 1) {
                  helper.holeParamIndex[idx] = int(lg->index);
                }
              }
            }
          }
        }
        snEff.localsRead.clear();
        snEff.localsWritten.clear();
        helper.snEffects.emplace(std::move(snEff));
        allocHelpers.emplace(f->name, std::move(helper));
        continue;
      }
    }

    if (f->getNumParams() == 1 && f->getResults().isRef()) {
      Expression* cur = f->body;
      bool peeled = false;
      while (cur) {
        if (auto* rc = cur->dynCast<RefCast>()) {
          if (!rc->desc) {
            cur = rc->ref;
            peeled = true;
            continue;
          }
        }
        if (auto* ra = cur->dynCast<RefAs>()) {
          if (ra->op == RefAsNonNull) {
            cur = ra->value;
            peeled = true;
            continue;
          }
        }
        break;
      }
      if (peeled) {
        if (auto* lg = cur->dynCast<LocalGet>()) {
          if (lg->index == 0) {
            castHelpers.insert(f->name);
          }
        }
      }
    }
  }

  if (isNested) {
    return;
  }

  // 2. Collect escaped functions, call counts, and functions with Return in
  // parallel across all functions.
  for (auto& exp : module.exports) {
    if (exp->kind == ExternalKind::Function) {
      escapedFuncs.insert(*exp->getInternalName());
    }
  }
  if (module.start.is()) {
    escapedFuncs.insert(module.start);
  }

  struct ModuleRefScanner : public PostWalker<ModuleRefScanner> {
    ModuleStoreInfo& info;
    ModuleRefScanner(ModuleStoreInfo& info) : info(info) {}
    void visitRefFunc(RefFunc* curr) { info.escapedFuncs.insert(curr->func); }
  } moduleScanner(*this);
  for (auto& g : module.globals) {
    if (!g->imported() && g->init) {
      moduleScanner.walk(g->init);
    }
  }
  for (auto& seg : module.elementSegments) {
    for (auto* init : seg->data) {
      moduleScanner.walk(init);
    }
  }

  struct FuncScanData {
    SmallVector<Name, 2> escaped;
    SmallVector<Name, 8> calls;
    bool hasReturn = false;
  };
  ModuleUtils::ParallelFunctionAnalysis<FuncScanData> scanAnalysis(
    module, [&](Function* f, FuncScanData& data) {
      if (f->imported() || !f->body) {
        return;
      }
      struct FuncScanner : public PostWalker<FuncScanner> {
        FuncScanData& data;
        FuncScanner(FuncScanData& data) : data(data) {}
        void visitRefFunc(RefFunc* curr) { data.escaped.push_back(curr->func); }
        void visitCall(Call* curr) { data.calls.push_back(curr->target); }
        void visitReturn(Return* curr) { data.hasReturn = true; }
      } fs(data);
      fs.walk(f->body);
    });

  for (auto& f : module.functions) {
    if (f->imported() || !f->body) {
      continue;
    }
    auto& data = scanAnalysis.map[f.get()];
    for (Name esc : data.escaped) {
      escapedFuncs.insert(esc);
    }
    for (Name callee : data.calls) {
      expectedCallSiteCount[callee]++;
    }
    if (data.hasReturn) {
      funcsWithReturn.insert(f->name);
    }
  }

  // 3. Build and propagate ParamSummary for each defined function parameter in
  // parallel.
  ModuleUtils::ParallelFunctionAnalysis<std::vector<ParamSummary>> sumAnalysis(
    module, [&](Function* f, std::vector<ParamSummary>& vec) {
      if (f->imported() || !f->body) {
        return;
      }
      Index numParams = f->getNumParams();
      vec.resize(numParams);
      for (Index p = 0; p < numParams; p++) {
        if (!f->getLocalType(p).isRef()) {
          vec[p].escapes = true;
          continue;
        }
        analyzeExprForLocal(f->body,
                            p,
                            /*inTailReturn=*/true,
                            /*inDropped=*/false,
                            vec[p]);
      }
    });
  for (auto& f : module.functions) {
    if (f->imported() || !f->body) {
      continue;
    }
    paramSummaries.emplace(f->name, std::move(sumAnalysis.map[f.get()]));
  }

  for (int iter = 0; iter < 16; iter++) {
    bool changed = false;
    for (auto& f : module.functions) {
      if (f->imported() || !f->body) {
        continue;
      }
      auto& vec = paramSummaries[f->name];
      for (Index p = 0; p < vec.size(); p++) {
        auto& sum = vec[p];
        bool oldEsc = sum.escapes;
        bool oldRet = sum.mayReturnParam;
        uint64_t oldFields = sum.nonDefaultWrittenFields;
        resolveCallEdges(sum);
        if (sum.escapes != oldEsc || sum.mayReturnParam != oldRet ||
            sum.nonDefaultWrittenFields != oldFields) {
          changed = true;
        }
      }
    }
    if (!changed) {
      break;
    }
  }

  for (auto& f : module.functions) {
    if (f->imported() || !f->body || funcsWithReturn.count(f->name)) {
      continue;
    }
    auto& vec = paramSummaries[f->name];
    Expression* tail = f->body;
    while (auto* blk = tail->dynCast<Block>()) {
      if (blk->name.is() || blk->list.empty()) {
        break;
      }
      tail = blk->list.back();
    }
    tail = unwrapCasts(tail, options.trapsNeverHappen);
    if (auto* lg = tail->dynCast<LocalGet>()) {
      if (lg->index < vec.size() && !vec[lg->index].modifiesLocal &&
          !vec[lg->index].escapes) {
        vec[lg->index].alwaysReturnsParam = true;
      }
    }
  }

  // 4. Interprocedural fixed-point propagation for entryDefaultFields and
  // returnDefaultFields in parallel.
  for (auto& f : module.functions) {
    if (f->imported() || !f->body) {
      continue;
    }
    entryIsUnescaped[f->name].assign(f->getNumParams(), false);
    entryDefaultFields[f->name].assign(f->getNumParams(), 0);
    returnIsFresh[f->name] = false;
    returnDefaultFields[f->name] = 0;
  }

  for (int iter = 0; iter < 10; iter++) {
    pendingCalls.clear();
    for (auto& f : module.functions) {
      if (f->imported() || !f->body || escapedFuncs.count(f->name)) {
        continue;
      }
      auto itCount = expectedCallSiteCount.find(f->name);
      if (itCount == expectedCallSiteCount.end() || itCount->second == 0) {
        continue;
      }
      Index numParams = f->getNumParams();
      if (numParams == 0) {
        continue;
      }
      auto& sumVec = paramSummaries[f->name];
      PendingCallTargetInfo info;
      info.unescaped.assign(numParams, false);
      info.defaultFields.assign(numParams, 0);
      bool anyCandidateParam = false;
      for (Index p = 0; p < numParams; p++) {
        if (f->getLocalType(p).isRef() && !sumVec[p].modifiesLocal) {
          info.unescaped[p] = true;
          info.defaultFields[p] = ~uint64_t(0);
          anyCandidateParam = true;
        }
      }
      if (anyCandidateParam) {
        pendingCalls.emplace(f->name, std::move(info));
      }
    }

    struct Step4FuncObs {
      FreshInfo retFresh = {false, 0};
      std::vector<CallSiteObs> callObs;
    };
    ModuleUtils::ParallelFunctionAnalysis<Step4FuncObs> step4Analysis(
      module, [&](Function* f, Step4FuncObs& obs) {
        if (f->imported() || !f->body) {
          return;
        }
        bool mayReturnFresh =
          f->getResults().isRef() && !funcsWithReturn.count(f->name);
        bool callsPending = false;
        if (!mayReturnFresh) {
          for (Name callee : scanAnalysis.map.at(f).calls) {
            if (pendingCalls.count(callee)) {
              callsPending = true;
              break;
            }
          }
          if (!callsPending) {
            return;
          }
        }
        FreshDefaultStoreWalker walker(*this,
                                       options,
                                       module,
                                       f,
                                       WalkMode::AnalyzeCallSites,
                                       &obs.callObs);
        obs.retFresh = walker.run();
      });

    std::unordered_map<Name, bool> nextReturnIsFresh;
    std::unordered_map<Name, uint64_t> nextReturnDefaultFields;
    for (auto& f : module.functions) {
      if (f->imported() || !f->body) {
        continue;
      }
      auto& obs = step4Analysis.map[f.get()];
      for (const auto& co : obs.callObs) {
        recordCallVisited(co.callee);
        for (Index m = 0; m < co.args.size(); m++) {
          recordCallArgument(co.callee, m, co.args[m]);
        }
      }
      if (!funcsWithReturn.count(f->name) && obs.retFresh.unescaped) {
        nextReturnIsFresh[f->name] = true;
        nextReturnDefaultFields[f->name] = obs.retFresh.defaultFields;
      } else {
        nextReturnIsFresh[f->name] = false;
        nextReturnDefaultFields[f->name] = 0;
      }
    }

    bool changed = false;
    for (auto& f : module.functions) {
      if (f->imported() || !f->body) {
        continue;
      }
      bool rf = nextReturnIsFresh[f->name];
      uint64_t rdf = nextReturnDefaultFields[f->name];
      if (returnIsFresh[f->name] != rf || returnDefaultFields[f->name] != rdf) {
        returnIsFresh[f->name] = rf;
        returnDefaultFields[f->name] = rdf;
        changed = true;
      }

      Index numParams = f->getNumParams();
      std::vector<bool> nextUnesc(numParams, false);
      std::vector<uint64_t> nextDef(numParams, 0);
      auto itPend = pendingCalls.find(f->name);
      if (itPend != pendingCalls.end() &&
          itPend->second.visitedCalls == expectedCallSiteCount[f->name]) {
        for (Index p = 0; p < numParams; p++) {
          if (itPend->second.unescaped[p]) {
            nextUnesc[p] = true;
            nextDef[p] = itPend->second.defaultFields[p];
          }
        }
      }
      if (entryIsUnescaped[f->name] != nextUnesc ||
          entryDefaultFields[f->name] != nextDef) {
        entryIsUnescaped[f->name] = std::move(nextUnesc);
        entryDefaultFields[f->name] = std::move(nextDef);
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
  }
  pendingCalls.clear();
}

// In each basic block we will store the relevant heap store operations and
// other actions that matter to our analysis.
struct Info {
  std::vector<Expression**> actions;
};

struct HeapStoreOptimization
  : public WalkerPass<
      CFGWalker<HeapStoreOptimization, Visitor<HeapStoreOptimization>, Info>> {
  std::shared_ptr<ModuleStoreInfo> moduleInfo;

  HeapStoreOptimization()
    : moduleInfo(std::make_shared<ModuleStoreInfo>()) {}
  HeapStoreOptimization(std::shared_ptr<ModuleStoreInfo> moduleInfo)
    : moduleInfo(std::move(moduleInfo)) {}

  bool isFunctionParallel() override { return true; }

  // Locals are not modified here.
  bool requiresNonNullableLocalFixups() override { return false; }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<HeapStoreOptimization>(moduleInfo);
  }

  bool hasPrepareToRun() override { return true; }

  void prepareToRun(Module* module) override {
    if (moduleInfo) {
      moduleInfo->analyze(getPassOptions(), *module);
    }
  }

  // Branches outside of the function can be ignored, as we only look at local
  // state in the function. (This may need to change if we do more general dead
  // store elimination.)
  bool ignoreBranchesOutsideOfFunc = true;
  bool needsRefinalize = false;
  bool funcHasCatch = false;

  // Store struct.sets and blocks, as we can find patterns among those.
  void addAction() {
    if (currBasicBlock) {
      currBasicBlock->contents.actions.push_back(getCurrentPointer());
    }
  }
  void visitStructSet(StructSet* curr) { addAction(); }
  void visitBlock(Block* curr) { addAction(); }
  void visitTry(Try* curr) { funcHasCatch = true; }
  void visitTryTable(TryTable* curr) { funcHasCatch = true; }

  bool hasInFuncControlFlow(const EffectAnalyzer& eff) const {
    return eff.hasExternalBreakTargets() || (eff.throws() && funcHasCatch);
  }

  void visitFunction(Function* curr) {
    // Now that the walk is complete and we have a CFG, find things to optimize.
    for (auto& block : basicBlocks) {
      for (auto** currp : block->contents.actions) {
        auto* act = *currp;
        if (auto* set = act->dynCast<StructSet>()) {
          optimizeStructSet(set, currp);
        } else if (auto* blk = act->dynCast<Block>()) {
          optimizeBlock(blk);
        } else {
          WASM_UNREACHABLE("bad action");
        }
      }
    }

    if (moduleInfo) {
      FreshDefaultStoreWalker freshWalker(
        *moduleInfo, getPassOptions(), *getModule(), curr, WalkMode::Optimize);
      freshWalker.run();
    }

    if (getPassOptions().shrinkLevel > 0) {
      struct BlockStoreLoadOptimizer
        : public PostWalker<BlockStoreLoadOptimizer> {
        HeapStoreOptimization& parent;
        BlockStoreLoadOptimizer(HeapStoreOptimization& parent)
          : parent(parent) {}
        void visitBlock(Block* curr) {
          parent.optimizeBlockStoresAndLoads(curr);
        }
      };
      BlockStoreLoadOptimizer(*this).walk(curr->body);
    }
    if (needsRefinalize) {
      ReFinalize().walkFunctionInModule(curr, getModule());
    }
    needsRefinalize = false;
    funcHasCatch = false;
  }

  // Optimize a struct.set. Receives also a pointer to where it is referred to,
  // so we can replace it (which we do if we optimize).
  void optimizeStructSet(StructSet* curr, Expression** currp) {
    // If our reference is a tee of a struct.new (or single-StructNew allocation
    // helper call), we may be able to fold the stored value into the new
    // itself:
    //
    //  (struct.set (local.tee $x (struct.new X Y Z)) X')
    // =>
    //  (local.set $x (struct.new X' Y Z))
    //
    if (auto* tee = curr->ref->dynCast<LocalSet>()) {
      if (auto* new_ = tee->value->dynCast<StructNew>()) {
        if (optimizeSubsequentStructSet(new_, curr, tee)) {
          // Success, so we do not need the struct.set any more, and the tee
          // can just be a set instead of us.
          tee->makeSet();
          *currp = tee;
        }
      } else if (auto* call = tee->value->dynCast<Call>()) {
        if (const auto* helper =
              moduleInfo ? moduleInfo->getAllocHelper(call->target) : nullptr) {
          if (optimizeSubsequentAllocCallStructSet(call, *helper, curr, tee)) {
            tee->makeSet();
            *currp = tee;
          }
        }
      }
    }
  }

  // Check whether every LocalGet of `refLocal` inside `expr` is the `ref`
  // operand of a `StructGet` on `expectedHT`, and collect the set of fields
  // read on `refLocal`.
  bool checkOnlyStructGetsOnRef(Expression* expr,
                                Index refLocal,
                                HeapType expectedHT,
                                SmallSet<Index, 4>& fieldsRead,
                                std::vector<Expression**>& selfGetPtrs) {
    struct Scanner : public PostWalker<Scanner> {
      Index refLocal;
      HeapType expectedHT;
      Index totalGets = 0;
      SmallSet<Index, 4>& fieldsRead;
      std::vector<Expression**>& selfGetPtrs;
      Scanner(Index refLocal,
              HeapType expectedHT,
              SmallSet<Index, 4>& fieldsRead,
              std::vector<Expression**>& selfGetPtrs)
        : refLocal(refLocal), expectedHT(expectedHT), fieldsRead(fieldsRead),
          selfGetPtrs(selfGetPtrs) {}
      void visitLocalGet(LocalGet* curr) {
        if (curr->index == refLocal) {
          totalGets++;
        }
      }
      void visitStructGet(StructGet* curr) {
        if (auto* lg = curr->ref->dynCast<LocalGet>()) {
          if (lg->index == refLocal && curr->ref->type.isRef() &&
              curr->ref->type.getHeapType() == expectedHT) {
            fieldsRead.insert(curr->index);
            selfGetPtrs.push_back(getCurrentPointer());
          }
        }
      }
    };
    Scanner scanner(refLocal, expectedHT, fieldsRead, selfGetPtrs);
    scanner.walk(expr);
    return scanner.totalGets == selfGetPtrs.size();
  }

  // Similar to the above with struct.set whose reference is a tee of a new, we
  // can do the same for subsequent sets in a list:
  //
  //  (local.set $x (struct.new X Y Z))
  //  (struct.set (local.get $x) X')
  // =>
  //  (local.set $x (struct.new X' Y Z))
  //
  // We also handle other struct.sets immediately after this one. If the
  // instruction following the new is not a struct.set we push the new down if
  // possible.
  void optimizeBlock(Block* curr) {
    auto& list = curr->list;

    for (Index i = 0; i < list.size(); i++) {
      LocalSet* localSet = list[i]->dynCast<LocalSet>();
      StructSet* firstSet = nullptr;
      if (!localSet) {
        firstSet = list[i]->dynCast<StructSet>();
        if (firstSet) {
          localSet = firstSet->ref->dynCast<LocalSet>();
        }
      }
      if (!localSet) {
        continue;
      }
      auto* new_ = localSet->value->dynCast<StructNew>();
      Call* allocCall = nullptr;
      const AllocHelperInfo* allocHelper = nullptr;
      if (!new_) {
        allocCall = localSet->value->dynCast<Call>();
        if (allocCall && moduleInfo) {
          allocHelper = moduleInfo->getAllocHelper(allocCall->target);
        }
      }
      if ((!new_ || new_->type == Type::unreachable) &&
          (!allocHelper || allocCall->type == Type::unreachable)) {
        continue;
      }

      HeapType newHT = new_ ? new_->type.getHeapType() : allocHelper->ht;
      struct InterveningSet {
        StructSet* set;
        EffectAnalyzer valEffects;
        SmallSet<Index, 4> selfFieldsRead;
      };
      std::vector<InterveningSet> intervening;

      if (firstSet) {
        auto valEff = effects(firstSet->value);
        SmallSet<Index, 4> fieldsRead;
        std::vector<Expression**> selfGetPtrs;
        if (hasInFuncControlFlow(valEff) ||
            valEff.localsWritten.contains(localSet->index) ||
            !checkOnlyStructGetsOnRef(
              firstSet->value, localSet->index, newHT, fieldsRead, selfGetPtrs)) {
          continue;
        }
        intervening.push_back(
          {firstSet, std::move(valEff), std::move(fieldsRead)});
      }

      // This local.set (or struct.set with local.tee) of a struct.new looks
      // good. Find struct.sets after it to optimize.
      Index localSetIndex = i;
      for (Index j = localSetIndex + 1; j < list.size(); j++) {
        if (!intervening.empty() && list[j]->is<Nop>()) {
          continue;
        }

        // Check that the next instruction is a struct.set on the same local as
        // the struct.new.
        auto* structSet = list[j]->dynCast<StructSet>();
        auto* localGet =
          structSet ? structSet->ref->dynCast<LocalGet>() : nullptr;
        if (!structSet || !localGet || localGet->index != localSet->index ||
            !structSet->ref->type.isRef() ||
            structSet->ref->type.getHeapType() != newHT) {
          // Any time the pattern no longer matches, we try to push the
          // struct.new further down but if it is not possible we stop
          // optimizing possible struct.sets for this struct.new.
          if (intervening.empty() && trySwap(list, localSetIndex, j)) {
            // Update the index and continue to try again.
            localSetIndex = j;
            continue;
          }
          break;
        }

        // Check if we can fold structSet past any intervening un-folded sets.
        bool canPassIntervening = true;
        auto setValEff = effects(structSet->value);
        SmallSet<Index, 4> currSelfFieldsRead;
        std::vector<Expression**> currSelfGetPtrs;
        bool onlySelfGets = checkOnlyStructGetsOnRef(structSet->value,
                                                     localSet->index,
                                                     newHT,
                                                     currSelfFieldsRead,
                                                     currSelfGetPtrs);
        if (!intervening.empty()) {
          if (hasInFuncControlFlow(setValEff) || !onlySelfGets ||
              setValEff.localsWritten.contains(localSet->index)) {
            canPassIntervening = false;
          } else {
            for (auto& prev : intervening) {
              if (prev.set->index == structSet->index ||
                  prev.selfFieldsRead.contains(structSet->index) ||
                  currSelfFieldsRead.contains(prev.set->index) ||
                  prev.valEffects.orderedBefore(setValEff)) {
                canPassIntervening = false;
                break;
              }
            }
          }
        }

        // The pattern matches, try to optimize.
        bool optimized = false;
        if (canPassIntervening) {
          if (new_) {
            optimized = optimizeSubsequentStructSet(new_, structSet, localSet);
          } else {
            optimized = optimizeSubsequentAllocCallStructSet(
              allocCall, *allocHelper, structSet, localSet);
          }
        }
        if (optimized) {
          // Success. Replace the set with a nop, and continue to perhaps
          // optimize more.
          ExpressionManipulator::nop(structSet);
        } else {
          // If we could not fold this structSet, see if subsequent struct.sets
          // on the same fresh allocation can still fold past it.
          if (!hasInFuncControlFlow(setValEff) && onlySelfGets &&
              !setValEff.localsWritten.contains(localSet->index)) {
            intervening.push_back({structSet,
                                   std::move(setValEff),
                                   std::move(currSelfFieldsRead)});
            continue;
          }
          break;
        }
      }
    }
  }

  // Helper function for optimizeHeapStores. Tries pushing the struct.new at
  // index i down to index j, swapping it with the instruction already at j, so
  // that it is closer to (potential) later struct.sets.
  bool trySwap(ExpressionList& list, Index i, Index j) {
    if (j == list.size() - 1) {
      // There is no reason to swap with the last element of the list as it
      // won't match the pattern because there won't be anything after. This
      // also avoids swapping an instruction that does not leave anything in the
      // stack by one that could leave something, and that which would be
      // incorrect.
      return false;
    }

    if (auto* ls = list[j]->dynCast<LocalSet>()) {
      if (ls->value->is<StructNew>()) {
        // Don't swap two struct.new instructions to avoid going back and forth.
        return false;
      }
      if (auto* call = ls->value->dynCast<Call>()) {
        if (moduleInfo && moduleInfo->getAllocHelper(call->target)) {
          return false;
        }
      }
    }
    // Check if the two expressions can be swapped safely considering their
    // effects.
    auto firstEffects = effects(list[i]);
    auto secondEffects = effects(list[j]);
    if (firstEffects.orderedBefore(secondEffects)) {
      return false;
    }

    std::swap(list[i], list[j]);
    return true;
  }

  Expression* makeSelfGetReplacement(Expression* sourceOp,
                                     bool isWithDefault,
                                     StructGet* get,
                                     const Field& field) {
    Builder builder(*getModule());
    Expression* rep = nullptr;
    if (isWithDefault) {
      rep = builder.makeConstantExpression(Literal::makeZero(field.type));
    } else if (auto* tee2 = sourceOp->dynCast<LocalSet>()) {
      rep = builder.makeLocalGet(tee2->index, tee2->type);
    } else {
      rep = ExpressionManipulator::copy(sourceOp, *getModule());
    }
    if (field.type == Type::i32 && field.packedType != Field::NotPacked) {
      if (auto* c = rep->dynCast<Const>()) {
        int32_t v = c->value.geti32();
        if (field.packedType == Field::i8) {
          v = get->signed_ ? int32_t(int8_t(v)) : int32_t(uint8_t(v));
        } else {
          v = get->signed_ ? int32_t(int16_t(v)) : int32_t(uint16_t(v));
        }
        c->value = Literal(v);
      } else if (field.packedType == Field::i8) {
        rep = get->signed_
                ? static_cast<Expression*>(
                    builder.makeUnary(ExtendS8Int32, rep))
                : static_cast<Expression*>(builder.makeBinary(
                    AndInt32, rep, builder.makeConst(int32_t(0xff))));
      } else {
        rep = get->signed_
                ? static_cast<Expression*>(
                    builder.makeUnary(ExtendS16Int32, rep))
                : static_cast<Expression*>(builder.makeBinary(
                    AndInt32, rep, builder.makeConst(int32_t(0xffff))));
      }
    }
    if (rep->type != get->type) {
      needsRefinalize = true;
    }
    return rep;
  }

  // Given a struct.new and a struct.set that occurs right after it, and that
  // applies to the same data, try to apply the set during the new. This can be
  // either with a nested tee:
  //
  //  (struct.set
  //    (local.tee $x (struct.new X Y Z))
  //    X'
  //  )
  // =>
  //  (local.set $x (struct.new X' Y Z))
  //
  // or without:
  //
  //  (local.set $x (struct.new X Y Z))
  //  (struct.set (local.get $x) X')
  // =>
  //  (local.set $x (struct.new X' Y Z))
  //
  // Returns true if we succeeded.
  bool optimizeSubsequentStructSet(StructNew* new_,
                                   StructSet* set,
                                   LocalSet* localSet) {
    // Leave unreachable code for DCE, to avoid updating types here.
    if (new_->type == Type::unreachable || set->type == Type::unreachable) {
      return false;
    }

    auto index = set->index;
    auto& operands = new_->operands;
    auto refLocalIndex = localSet->index;
    auto newHT = new_->type.getHeapType();
    auto& fields = newHT.getStruct().fields;

    bool isRedundantDefaultStore = false;
    if (new_->isWithDefault()) {
      if (isDefaultFieldValue(set->value, fields[index].type)) {
        isRedundantDefaultStore = true;
      } else if (getPassOptions().shrinkLevel > 0 && fields.size() > 3) {
        // Materializing > 3 default operands in struct.new bloats code size.
        return false;
      }
    }

    // Check for effects that prevent us moving the struct.set's value (X' in
    // the function comment) into its new position in the struct.new. First, it
    // must be ok to move it past the local.set (otherwise, it might read from
    // memory using that local, and depend on the struct.new having already
    // occurred; or, if it writes to that local, then it would cross another
    // write).
    auto setValueEffects = effects(set->value);
    if (setValueEffects.localsWritten.contains(refLocalIndex)) {
      return false;
    }

    std::vector<Expression**> selfGetPtrs;
    if (setValueEffects.localsRead.contains(refLocalIndex)) {
      SmallSet<Index, 4> fieldsRead;
      if (!checkOnlyStructGetsOnRef(
            set->value, refLocalIndex, newHT, fieldsRead, selfGetPtrs) ||
          selfGetPtrs.empty()) {
        return false;
      }
      // Verify every self-get reads a field whose initial value in `new_` can
      // be substituted into `set->value`.
      for (auto** getp : selfGetPtrs) {
        auto* get = (*getp)->cast<StructGet>();
        Index f2 = get->index;
        if (new_->isWithDefault()) {
          continue;
        }
        Expression* op2 = operands[f2];
        if (op2->is<Const>() || op2->is<RefNull>() || op2->is<RefFunc>()) {
          continue;
        }
        if (auto* lg = op2->dynCast<LocalGet>()) {
          if (lg->index == refLocalIndex ||
              setValueEffects.localsWritten.contains(lg->index)) {
            return false;
          }
          bool writtenInNew = false;
          for (Index k = 0; k < operands.size(); k++) {
            if (k != f2 &&
                effects(operands[k]).localsWritten.contains(lg->index)) {
              writtenInNew = true;
              break;
            }
          }
          if (writtenInNew ||
              (new_->desc &&
               effects(new_->desc).localsWritten.contains(lg->index))) {
            return false;
          }
          continue;
        }
        if (auto* gg = op2->dynCast<GlobalGet>()) {
          if (setValueEffects.globalsWritten.contains(gg->name)) {
            return false;
          }
          bool writtenInNew = false;
          for (Index k = 0; k < operands.size(); k++) {
            if (k != f2 &&
                effects(operands[k]).globalsWritten.contains(gg->name)) {
              writtenInNew = true;
              break;
            }
          }
          if (writtenInNew ||
              (new_->desc &&
               effects(new_->desc).globalsWritten.contains(gg->name))) {
            return false;
          }
          continue;
        }
        if (auto* tee2 = op2->dynCast<LocalSet>()) {
          if (tee2->isTee() && f2 < index && tee2->index != refLocalIndex &&
              !setValueEffects.localsWritten.contains(tee2->index)) {
            bool writtenAfter = false;
            for (Index k = f2 + 1; k < operands.size(); k++) {
              if (effects(operands[k]).localsWritten.contains(tee2->index)) {
                writtenAfter = true;
                break;
              }
            }
            if (!writtenAfter) {
              continue;
            }
          }
        }
        return false;
      }
      setValueEffects.localsRead.erase(refLocalIndex);
    }

    // We must move the set's value past indexes greater than it (Y and Z in
    // the example in the comment on this function). If this is not with_default
    // then we must check for effects.
    // TODO When this function is called repeatedly in a sequence this can
    //      become quadratic - perhaps we should memoize (though, struct sizes
    //      tend to not be ridiculously large).
    if (!new_->isWithDefault()) {
      for (Index i = index + 1; i < operands.size(); i++) {
        auto operandEffects = effects(operands[i]);
        if (operandEffects.orderedBefore(setValueEffects)) {
          // TODO: we could use locals to reorder everything
          return false;
        }
      }
    }

    // Similarly, we must move the set's value past the allocation's descriptor,
    // if it exists.
    if (new_->desc) {
      auto descEffects = effects(new_->desc);
      if (descEffects.orderedBefore(setValueEffects)) {
        // TODO: we could use locals to reorder everything
        return false;
      }
    }

    // We also cannot reorder if the struct.new itself has effects (which it can
    // in the case of a descriptor) that interact with X' (from the comment
    // above), as e.g. a struct.new trap happens before effects in X', but after
    // the optimization X' would happen first.
    ShallowEffectAnalyzer structNewEffects(
      getPassOptions(), *getModule(), new_);
    if (structNewEffects.orderedBefore(setValueEffects)) {
      return false;
    }

    // We must also be careful of branches out from the value that skip the
    // local.set, see below.
    if (canSkipLocalSet(set, setValueEffects, localSet)) {
      return false;
    }

    if (isRedundantDefaultStore) {
      return true;
    }

    // We can optimize here!
    Builder builder(*getModule());

    // Perform any self-get substitutions in set->value before overwriting
    // operands[index].
    if (!selfGetPtrs.empty()) {
      for (auto** getp : selfGetPtrs) {
        auto* get = (*getp)->cast<StructGet>();
        Index f2 = get->index;
        *getp = makeSelfGetReplacement(
          new_->isWithDefault() ? nullptr : operands[f2],
          new_->isWithDefault(),
          get,
          fields[f2]);
      }
    }

    // If this was with_default then we add default values now. That does
    // increase code size in some cases (if there are many values, and few sets
    // that get removed), but in general this optimization is worth it.
    if (new_->isWithDefault()) {
      for (auto& field : fields) {
        auto zero = Literal::makeZero(field.type);
        operands.push_back(builder.makeConstantExpression(zero));
      }
    }

    // See if we need to keep the old value.
    if (effects(operands[index]).hasUnremovableSideEffects()) {
      operands[index] =
        builder.makeSequence(builder.makeDrop(operands[index]), set->value);
    } else {
      operands[index] = set->value;
    }

    return true;
  }

  bool optimizeSubsequentAllocCallStructSet(Call* call,
                                            const AllocHelperInfo& helper,
                                            StructSet* set,
                                            LocalSet* localSet) {
    if (call->type == Type::unreachable || set->type == Type::unreachable ||
        !set->ref->type.isRef() ||
        set->ref->type.getHeapType() != helper.ht) {
      return false;
    }

    Index index = set->index;
    auto& fields = helper.ht.getStruct().fields;
    if (index >= fields.size() || index >= helper.holeParamIndex.size()) {
      return false;
    }
    int holeParam = helper.holeParamIndex[index];
    Type fieldType = fields[index].type;

    bool isRedundantStore = false;
    if (helper.isWithDefault) {
      isRedundantStore = isDefaultFieldValue(set->value, fieldType);
    } else if (holeParam >= 0 && Index(holeParam) < call->operands.size()) {
      isRedundantStore = isSameConstantOrDefaultValue(
        call->operands[holeParam], set->value, fieldType);
    } else if (index < helper.constOperands.size() &&
               helper.constOperands[index]) {
      isRedundantStore = isSameConstantOrDefaultValue(
        helper.constOperands[index], set->value, fieldType);
    }

    if (isRedundantStore) {
      return true;
    }
    if (holeParam < 0 || Index(holeParam) >= call->operands.size()) {
      return false;
    }
    if (!Type::isSubType(set->value->type,
                         helper.func->getLocalType(Index(holeParam)))) {
      return false;
    }

    auto refLocalIndex = localSet->index;
    auto setValueEffects = effects(set->value);
    if (setValueEffects.localsWritten.contains(refLocalIndex)) {
      return false;
    }

    std::vector<Expression**> selfGetPtrs;
    if (setValueEffects.localsRead.contains(refLocalIndex)) {
      SmallSet<Index, 4> fieldsRead;
      if (!checkOnlyStructGetsOnRef(
            set->value, refLocalIndex, helper.ht, fieldsRead, selfGetPtrs) ||
          selfGetPtrs.empty()) {
        return false;
      }
      for (auto** getp : selfGetPtrs) {
        auto* get = (*getp)->cast<StructGet>();
        Index f2 = get->index;
        if (f2 >= fields.size()) {
          return false;
        }
        if (helper.isWithDefault) {
          continue;
        }
        int hole2 = helper.holeParamIndex[f2];
        Expression* op2 =
          (hole2 >= 0 && Index(hole2) < call->operands.size())
            ? call->operands[hole2]
            : (f2 < helper.constOperands.size() ? helper.constOperands[f2]
                                                : nullptr);
        if (!op2) {
          return false;
        }
        if (op2->is<Const>() || op2->is<RefNull>() || op2->is<RefFunc>()) {
          continue;
        }
        if (hole2 >= 0) {
          if (auto* lg = op2->dynCast<LocalGet>()) {
            if (lg->index == refLocalIndex ||
                setValueEffects.localsWritten.contains(lg->index)) {
              return false;
            }
            bool writtenInCall = false;
            for (Index k = 0; k < call->operands.size(); k++) {
              if (k != Index(hole2) &&
                  effects(call->operands[k])
                    .localsWritten.contains(lg->index)) {
                writtenInCall = true;
                break;
              }
            }
            if (!writtenInCall) {
              continue;
            }
          }
        }
        return false;
      }
      setValueEffects.localsRead.erase(refLocalIndex);
    }

    for (Index k = Index(holeParam) + 1; k < call->operands.size(); k++) {
      if (effects(call->operands[k]).orderedBefore(setValueEffects)) {
        return false;
      }
    }

    if (helper.snEffects && helper.snEffects->orderedBefore(setValueEffects)) {
      return false;
    }

    if (canSkipLocalSet(set, setValueEffects, localSet)) {
      return false;
    }

    Builder builder(*getModule());
    if (!selfGetPtrs.empty()) {
      for (auto** getp : selfGetPtrs) {
        auto* get = (*getp)->cast<StructGet>();
        Index f2 = get->index;
        int hole2 = helper.holeParamIndex[f2];
        Expression* op2 =
          helper.isWithDefault
            ? nullptr
            : ((hole2 >= 0 && Index(hole2) < call->operands.size())
                 ? call->operands[hole2]
                 : helper.constOperands[f2]);
        *getp =
          makeSelfGetReplacement(op2, helper.isWithDefault, get, fields[f2]);
      }
    }

    Index hp = Index(holeParam);
    if (effects(call->operands[hp]).hasUnremovableSideEffects()) {
      call->operands[hp] =
        builder.makeSequence(builder.makeDrop(call->operands[hp]), set->value);
    } else {
      call->operands[hp] = set->value;
    }
    return true;
  }

  bool exprReadsStructField(Expression* root,
                            HeapType targetHT,
                            Index targetField,
                            int depth = 0) {
    struct Checker : public PostWalker<Checker> {
      HeapStoreOptimization& parent;
      HeapType targetHT;
      Index targetField;
      int depth;
      bool found = false;
      Checker(HeapStoreOptimization& parent,
              HeapType targetHT,
              Index targetField,
              int d)
        : parent(parent), targetHT(targetHT), targetField(targetField),
          depth(d) {}
      void visitStructGet(StructGet* curr) {
        if (found || curr->ref->type == Type::unreachable ||
            curr->ref->type.isNull()) {
          return;
        }
        if (curr->index == targetField &&
            (HeapType::isSubType(curr->ref->type.getHeapType(), targetHT) ||
             HeapType::isSubType(targetHT, curr->ref->type.getHeapType()))) {
          found = true;
        }
      }
      void visitCall(Call* curr) {
        if (found) {
          return;
        }
        ShallowEffectAnalyzer eff(
          parent.getPassOptions(), *parent.getModule(), curr);
        if (eff.readsMutableStruct) {
          found = true;
        }
      }
      void visitCallIndirect(CallIndirect* curr) { found = true; }
      void visitCallRef(CallRef* curr) { found = true; }
    };
    Checker checker(*this, targetHT, targetField, depth);
    checker.walk(root);
    return checker.found;
  }

  bool exprWritesStructField(Expression* root,
                             HeapType targetHT,
                             Index targetField,
                             int depth = 0) {
    struct Checker : public PostWalker<Checker> {
      HeapStoreOptimization& parent;
      HeapType targetHT;
      Index targetField;
      int depth;
      bool found = false;
      Checker(HeapStoreOptimization& parent,
              HeapType targetHT,
              Index targetField,
              int d)
        : parent(parent), targetHT(targetHT), targetField(targetField),
          depth(d) {}
      void visitStructSet(StructSet* curr) {
        if (found || curr->ref->type == Type::unreachable ||
            curr->ref->type.isNull()) {
          return;
        }
        if (curr->index == targetField &&
            (HeapType::isSubType(curr->ref->type.getHeapType(), targetHT) ||
             HeapType::isSubType(targetHT, curr->ref->type.getHeapType()))) {
          found = true;
        }
      }
      void visitCall(Call* curr) {
        if (found) {
          return;
        }
        ShallowEffectAnalyzer eff(
          parent.getPassOptions(), *parent.getModule(), curr);
        if (eff.writesStruct) {
          found = true;
        }
      }
      void visitCallIndirect(CallIndirect* curr) { found = true; }
      void visitCallRef(CallRef* curr) { found = true; }
    };
    Checker checker(*this, targetHT, targetField, depth);
    checker.walk(root);
    return checker.found;
  }

  bool isProvablyNonNull(Expression* expr,
                         const SmallSet<Index, 8>& nonNullLocals) {
    while (auto* blk = expr->dynCast<Block>()) {
      if (blk->name.is() || blk->list.empty()) {
        break;
      }
      expr = blk->list.back();
    }
    if (expr->type.isNonNullable()) {
      return true;
    }
    if (expr->is<StructNew>() || expr->is<ArrayNew>() ||
        expr->is<ArrayNewFixed>() || expr->is<ArrayNewData>() ||
        expr->is<RefFunc>() || expr->is<StringConst>()) {
      return true;
    }
    if (auto* ls = expr->dynCast<LocalSet>()) {
      if (ls->isTee()) {
        return isProvablyNonNull(ls->value, nonNullLocals);
      }
    }
    if (auto* lg = expr->dynCast<LocalGet>()) {
      return nonNullLocals.contains(lg->index);
    }
    return false;
  }

  Expression* getForwardableValue(Expression* val,
                                  const Field& field,
                                  Index& depLocal,
                                  Name& depGlobal) {
    depLocal = Index(-1);
    depGlobal = Name();
    if (field.packedType != Field::NotPacked) {
      if (auto* c = val->dynCast<Const>()) {
        return c;
      }
      return nullptr;
    }
    if (auto* lg = val->dynCast<LocalGet>()) {
      depLocal = lg->index;
      return lg;
    }
    if (auto* ls = val->dynCast<LocalSet>()) {
      if (ls->isTee()) {
        depLocal = ls->index;
        return ls;
      }
    }
    if (auto* gg = val->dynCast<GlobalGet>()) {
      depGlobal = gg->name;
      return gg;
    }
    if (val->is<RefNull>() || val->is<RefFunc>()) {
      return val;
    }
    if (auto* c = val->dynCast<Const>()) {
      if (c->type == Type::i32) {
        int32_t v = c->value.geti32();
        if (v >= -8192 && v <= 8191) {
          return c;
        }
      }
    }
    return nullptr;
  }

  Expression* makeForwardedReplacement(Expression* fwdSource,
                                       StructGet* get,
                                       const Field& field) {
    Builder builder(*getModule());
    Expression* rep = nullptr;
    if (auto* ls = fwdSource->dynCast<LocalSet>()) {
      rep = builder.makeLocalGet(ls->index, ls->type);
    } else {
      rep = ExpressionManipulator::copy(fwdSource, *getModule());
    }
    if (field.type == Type::i32 && field.packedType != Field::NotPacked) {
      auto* c = rep->cast<Const>();
      int32_t v = c->value.geti32();
      if (field.packedType == Field::i8) {
        v = get->signed_ ? int32_t(int8_t(v)) : int32_t(uint8_t(v));
      } else {
        v = get->signed_ ? int32_t(int16_t(v)) : int32_t(uint16_t(v));
      }
      c->value = Literal(v);
    }
    if (rep->type != get->type) {
      needsRefinalize = true;
    }
    return rep;
  }

  // Walk `exprp` in WebAssembly evaluation order, forwarding `fwdSource` (if
  // non-null) to matching `(struct.get $T f (local.get $refLocal))` and folding
  // `(ref.is_null (struct.get $T f (local.get $refLocal)))` when `valIsNonNull`
  // is true. Returns false as soon as an instruction might clobber `(ht, f)`,
  // `refLocal`, `depLocal`, `depGlobal`, or transfer control flow.
  bool forwardStoreInExpr(Expression** exprp,
                          Index refLocal,
                          HeapType ht,
                          Index fieldIdx,
                          const Field& field,
                          Expression* fwdSource,
                          Index depLocal,
                          Name depGlobal,
                          bool valIsNonNull) {
    Expression* expr = *exprp;
    if (!expr || expr->type == Type::unreachable) {
      return false;
    }
    if (expr->is<Nop>() || expr->is<LocalGet>() || expr->is<Const>() ||
        expr->is<RefNull>() || expr->is<RefFunc>() || expr->is<GlobalGet>() ||
        expr->is<StringConst>()) {
      return true;
    }
    if (auto* get = expr->dynCast<StructGet>()) {
      if (!forwardStoreInExpr(&get->ref,
                              refLocal,
                              ht,
                              fieldIdx,
                              field,
                              fwdSource,
                              depLocal,
                              depGlobal,
                              valIsNonNull)) {
        return false;
      }
      if (fwdSource && get->index == fieldIdx &&
          get->ref->is<LocalGet>() &&
          get->ref->cast<LocalGet>()->index == refLocal &&
          get->ref->type.isRef() && get->ref->type.getHeapType() == ht) {
        *exprp = makeForwardedReplacement(fwdSource, get, field);
      }
      return true;
    }
    if (auto* rn = expr->dynCast<RefIsNull>()) {
      if (valIsNonNull) {
        if (auto* get = rn->value->dynCast<StructGet>()) {
          if (get->index == fieldIdx && get->ref->is<LocalGet>() &&
              get->ref->cast<LocalGet>()->index == refLocal &&
              get->ref->type.isRef() && get->ref->type.getHeapType() == ht) {
            Builder builder(*getModule());
            *exprp = builder.makeConst(int32_t(0));
            return true;
          }
        }
      }
      return forwardStoreInExpr(&rn->value,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull);
    }
    if (auto* u = expr->dynCast<Unary>()) {
      return forwardStoreInExpr(&u->value,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull);
    }
    if (auto* b = expr->dynCast<Binary>()) {
      return forwardStoreInExpr(&b->left,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull) &&
             forwardStoreInExpr(&b->right,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull);
    }
    if (auto* re = expr->dynCast<RefEq>()) {
      return forwardStoreInExpr(&re->left,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull) &&
             forwardStoreInExpr(&re->right,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull);
    }
    if (auto* rc = expr->dynCast<RefCast>()) {
      return forwardStoreInExpr(&rc->ref,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull);
    }
    if (auto* rt = expr->dynCast<RefTest>()) {
      return forwardStoreInExpr(&rt->ref,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull);
    }
    if (auto* ra = expr->dynCast<RefAs>()) {
      return forwardStoreInExpr(&ra->value,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull);
    }
    if (auto* al = expr->dynCast<ArrayLen>()) {
      return forwardStoreInExpr(&al->ref,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull);
    }
    if (auto* sel = expr->dynCast<Select>()) {
      return forwardStoreInExpr(&sel->ifTrue,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull) &&
             forwardStoreInExpr(&sel->ifFalse,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull) &&
             forwardStoreInExpr(&sel->condition,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull);
    }
    if (auto* drop = expr->dynCast<Drop>()) {
      return forwardStoreInExpr(&drop->value,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull);
    }
    if (auto* ret = expr->dynCast<Return>()) {
      if (ret->value) {
        forwardStoreInExpr(&ret->value,
                           refLocal,
                           ht,
                           fieldIdx,
                           field,
                           fwdSource,
                           depLocal,
                           depGlobal,
                           valIsNonNull);
      }
      return false;
    }
    if (auto* ls = expr->dynCast<LocalSet>()) {
      if (!forwardStoreInExpr(&ls->value,
                              refLocal,
                              ht,
                              fieldIdx,
                              field,
                              fwdSource,
                              depLocal,
                              depGlobal,
                              valIsNonNull)) {
        return false;
      }
      if (ls->index == refLocal || ls->index == depLocal) {
        return false;
      }
      return true;
    }
    if (auto* gs = expr->dynCast<GlobalSet>()) {
      if (!forwardStoreInExpr(&gs->value,
                              refLocal,
                              ht,
                              fieldIdx,
                              field,
                              fwdSource,
                              depLocal,
                              depGlobal,
                              valIsNonNull)) {
        return false;
      }
      if (depGlobal.is() && gs->name == depGlobal) {
        return false;
      }
      return true;
    }
    if (auto* ss = expr->dynCast<StructSet>()) {
      if (!forwardStoreInExpr(&ss->ref,
                              refLocal,
                              ht,
                              fieldIdx,
                              field,
                              fwdSource,
                              depLocal,
                              depGlobal,
                              valIsNonNull) ||
          !forwardStoreInExpr(&ss->value,
                              refLocal,
                              ht,
                              fieldIdx,
                              field,
                              fwdSource,
                              depLocal,
                              depGlobal,
                              valIsNonNull)) {
        return false;
      }
      if (ss->ref->type == Type::unreachable || ss->ref->type.isNull()) {
        return false;
      }
      if (ss->index == fieldIdx &&
          (HeapType::isSubType(ss->ref->type.getHeapType(), ht) ||
           HeapType::isSubType(ht, ss->ref->type.getHeapType()))) {
        return false;
      }
      return true;
    }
    if (auto* sn = expr->dynCast<StructNew>()) {
      for (auto*& op : sn->operands) {
        if (!forwardStoreInExpr(&op,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull)) {
          return false;
        }
      }
      if (sn->desc && !forwardStoreInExpr(&sn->desc,
                                          refLocal,
                                          ht,
                                          fieldIdx,
                                          field,
                                          fwdSource,
                                          depLocal,
                                          depGlobal,
                                          valIsNonNull)) {
        return false;
      }
      return true;
    }
    if (auto* call = expr->dynCast<Call>()) {
      for (auto*& op : call->operands) {
        if (!forwardStoreInExpr(&op,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull)) {
          return false;
        }
      }
      ShallowEffectAnalyzer callEff(getPassOptions(), *getModule(), call);
      if (hasInFuncControlFlow(callEff)) {
        return false;
      }
      if (depGlobal.is() && callEff.globalsWritten.contains(depGlobal)) {
        return false;
      }
      if (callEff.writesStruct && exprWritesStructField(call, ht, fieldIdx)) {
        return false;
      }
      return true;
    }
    if (auto* iff = expr->dynCast<If>()) {
      if (!forwardStoreInExpr(&iff->condition,
                              refLocal,
                              ht,
                              fieldIdx,
                              field,
                              fwdSource,
                              depLocal,
                              depGlobal,
                              valIsNonNull)) {
        return false;
      }
      if (auto* c = iff->condition->dynCast<Const>()) {
        if (c->value.geti32() == 0 && !iff->ifFalse) {
          ExpressionManipulator::nop(iff);
          return true;
        }
      }
      bool okTrue = forwardStoreInExpr(&iff->ifTrue,
                                       refLocal,
                                       ht,
                                       fieldIdx,
                                       field,
                                       fwdSource,
                                       depLocal,
                                       depGlobal,
                                       valIsNonNull);
      bool okFalse = !iff->ifFalse ||
                     forwardStoreInExpr(&iff->ifFalse,
                                        refLocal,
                                        ht,
                                        fieldIdx,
                                        field,
                                        fwdSource,
                                        depLocal,
                                        depGlobal,
                                        valIsNonNull);
      return okTrue && okFalse;
    }
    if (auto* blk = expr->dynCast<Block>()) {
      if (blk->name.is()) {
        return false;
      }
      for (auto*& item : blk->list) {
        if (!forwardStoreInExpr(&item,
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  Expression** findLeftmostSelfGet(Expression** exprp,
                                   Index refLocal,
                                   HeapType ht,
                                   Index fieldIdx) {
    Expression* expr = *exprp;
    if (!expr || expr->type == Type::unreachable) {
      return nullptr;
    }
    if (auto* get = expr->dynCast<StructGet>()) {
      if (get->index == fieldIdx && get->ref->is<LocalGet>() &&
          get->ref->cast<LocalGet>()->index == refLocal &&
          get->ref->type.isRef() && get->ref->type.getHeapType() == ht) {
        return exprp;
      }
      return findLeftmostSelfGet(&get->ref, refLocal, ht, fieldIdx);
    }
    if (auto* u = expr->dynCast<Unary>()) {
      return findLeftmostSelfGet(&u->value, refLocal, ht, fieldIdx);
    }
    if (auto* b = expr->dynCast<Binary>()) {
      return findLeftmostSelfGet(&b->left, refLocal, ht, fieldIdx);
    }
    if (auto* s = expr->dynCast<Select>()) {
      return findLeftmostSelfGet(&s->ifTrue, refLocal, ht, fieldIdx);
    }
    if (auto* c = expr->dynCast<Call>()) {
      if (!c->operands.empty()) {
        return findLeftmostSelfGet(&c->operands[0], refLocal, ht, fieldIdx);
      }
    }
    return nullptr;
  }

  void optimizeBlockStoresAndLoads(Block* curr) {
    auto& list = curr->list;
    SmallSet<Index, 8> nonNullLocals;

    for (Index i = 0; i < list.size(); i++) {
      Expression* stmt = list[i];
      if (stmt->is<Nop>()) {
        continue;
      }
      if (auto* ls = stmt->dynCast<LocalSet>()) {
        if (isProvablyNonNull(ls->value, nonNullLocals)) {
          nonNullLocals.insert(ls->index);
        } else {
          nonNullLocals.erase(ls->index);
        }
        continue;
      }

      auto* set = stmt->dynCast<StructSet>();
      if (!set || set->type == Type::unreachable ||
          set->ref->type == Type::unreachable || set->ref->type.isNull()) {
        auto eff = effects(stmt);
        for (Index w : eff.localsWritten) {
          nonNullLocals.erase(w);
        }
        continue;
      }

      Index refLocal = Index(-1);
      if (auto* lg = set->ref->dynCast<LocalGet>()) {
        refLocal = lg->index;
      } else if (auto* tee = set->ref->dynCast<LocalSet>()) {
        refLocal = tee->index;
      }
      if (refLocal == Index(-1)) {
        auto eff = effects(stmt);
        for (Index w : eff.localsWritten) {
          nonNullLocals.erase(w);
        }
        continue;
      }

      if (getPassOptions().trapsNeverHappen) {
        nonNullLocals.insert(refLocal);
      }
      auto setValEff = effects(set->value);
      for (Index w : setValEff.localsWritten) {
        nonNullLocals.erase(w);
      }
      if (auto* tee = set->value->dynCast<LocalSet>()) {
        if (isProvablyNonNull(tee->value, nonNullLocals)) {
          nonNullLocals.insert(tee->index);
        }
      }

      HeapType ht = set->ref->type.getHeapType();
      Index fieldIdx = set->index;
      auto& field = ht.getStruct().fields[fieldIdx];

      // 1. Check for adjacent dead overwritten struct.set $T f (local.get $x)
      if (set->ref->is<LocalGet>()) {
        Index nextIdx = i + 1;
        while (nextIdx < list.size() && list[nextIdx]->is<Nop>()) {
          nextIdx++;
        }
        if (nextIdx < list.size()) {
          if (auto* nextSet = list[nextIdx]->dynCast<StructSet>()) {
            auto* nextLg = nextSet->ref->dynCast<LocalGet>();
            if (nextLg && nextLg->index == refLocal &&
                nextSet->index == fieldIdx &&
                nextSet->type != Type::unreachable &&
                nextSet->ref->type.isRef() &&
                nextSet->ref->type.getHeapType() == ht) {
              auto val2Eff = effects(nextSet->value);
              if (!setValEff.transfersControlFlow() &&
                  !val2Eff.localsWritten.contains(refLocal) &&
                  !val2Eff.transfersControlFlow() &&
                  (getPassOptions().trapsNeverHappen ||
                   set->ref->type.isNonNullable() ||
                   !val2Eff.hasUnremovableSideEffects()) &&
                  !exprWritesStructField(nextSet->value, ht, fieldIdx)) {
                if (!exprReadsStructField(nextSet->value, ht, fieldIdx)) {
                  if (!setValEff.hasUnremovableSideEffects()) {
                    ExpressionManipulator::nop(set);
                  } else {
                    Builder builder(*getModule());
                    list[i] = builder.makeDrop(set->value);
                  }
                  continue;
                } else if (field.packedType == Field::NotPacked &&
                           !setValEff.localsWritten.contains(refLocal) &&
                           !setValEff.transfersControlFlow()) {
                  if (auto** slot = findLeftmostSelfGet(
                        &nextSet->value, refLocal, ht, fieldIdx)) {
                    Expression* origGet = *slot;
                    *slot = set->ref;
                    if (!exprReadsStructField(nextSet->value, ht, fieldIdx)) {
                      if (set->value->type != origGet->type) {
                        needsRefinalize = true;
                      }
                      *slot = set->value;
                      ExpressionManipulator::nop(set);
                      continue;
                    }
                    *slot = origGet;
                  }
                }
              }
            }
          }
        }
      }

      // 2. Intra-block store-to-load forwarding & non-null guard folding.
      Index depLocal = Index(-1);
      Name depGlobal;
      Expression* fwdSource =
        getForwardableValue(set->value, field, depLocal, depGlobal);
      bool valIsNonNull = isProvablyNonNull(set->value, nonNullLocals);
      if ((depLocal != Index(-1) && setValEff.localsWritten.contains(depLocal) &&
           !set->value->is<LocalSet>()) ||
          depLocal == refLocal) {
        fwdSource = nullptr;
      }

      if (!fwdSource && !valIsNonNull) {
        continue;
      }

      for (Index j = i + 1; j < list.size(); j++) {
        if (list[j]->is<Nop>()) {
          continue;
        }
        if (!forwardStoreInExpr(&list[j],
                                refLocal,
                                ht,
                                fieldIdx,
                                field,
                                fwdSource,
                                depLocal,
                                depGlobal,
                                valIsNonNull)) {
          break;
        }
      }
    }
  }

  // We must be careful of branches that skip the local.set. For example:
  //
  //  (block $out
  //    (local.set $x (struct.new X Y Z))
  //    (struct.set (local.get $x) (..br $out..))  ;; X' here has a br
  //  )
  //  ..local.get $x..
  //
  // Note how we do the local.set first. Imagine we optimized to this:
  //
  //  (block $out
  //    (local.set $x (struct.new (..br $out..) Y Z))
  //  )
  //  ..local.get $x..
  //
  // Now the br happens first, skipping the local.set entirely, and the use
  // later down will not get the proper value. This is the problem we must
  // detect here.
  //
  // We are given a struct.set, the computed effects of its value (the caller
  // already has those, so this is an optimization to avoid recomputation), and
  // the local.set.
  bool canSkipLocalSet(StructSet* set,
                       const EffectAnalyzer& setValueEffects,
                       LocalSet* localSet) {
    // First, if the set's value cannot branch at all, then we have no problem.
    if (!setValueEffects.transfersControlFlow()) {
      return false;
    }

    // We may branch, so do an analysis using a LocalGraph. We will check
    // whether it is valid to move the local.set to where the struct.set is, so
    // we provide StructSet as the query class.
    //
    // It is valid to reuse the LocalGraph many times because the optimization
    // that we do in this pass does not generate new, dangerous control flow. We
    // only optimize if moving a LocalSet is valid, and that does not invalidate
    // any other one.
    if (!localGraph) {
      localGraph.emplace(getFunction(), getModule(), StructSet::SpecificId);
    }
    auto badGets = localGraph->canMoveSet(localSet, set);
    if (badGets.size() == 0) {
      // No problems at all.
      return false;
    }
    // It is ok to have a local.get in the struct.set itself, as if we optimize
    // then that local.get goes away anyhow, that is,
    //
    //   (local.set $x (struct.new X Y Z))
    //   (struct.set (local.get $x) (X'))
    //  =>
    //   (local.set $x (struct.new X' Y Z))
    //
    // Both the local.get and the struct.set are removed.
    if (badGets.size() >= 2) {
      return true;
    }
    return *badGets.begin() != set->ref;
  }

  EffectAnalyzer effects(Expression* expr) {
    return EffectAnalyzer(getPassOptions(), *getModule(), expr);
  }

private:
  // A local graph that is constructed the first time we need it.
  std::optional<LazyLocalGraph> localGraph;
};

} // anonymous namespace

Pass* createHeapStoreOptimizationPass() { return new HeapStoreOptimization(); }

} // namespace wasm
