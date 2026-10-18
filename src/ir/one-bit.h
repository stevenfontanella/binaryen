/*
 * Copyright 2026 WebAssembly Community Group participants
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

#ifndef wasm_ir_one_bit_h
#define wasm_ir_one_bit_h

#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ir/bits.h"
#include "ir/branch-utils.h"
#include "ir/effects.h"
#include "ir/intrinsics.h"
#include "ir/iteration.h"
#include "ir/module-utils.h"
#include "ir/subtypes.h"
#include "pass.h"
#include "passes/string-utils.h"
#include "support/small_vector.h"
#include "wasm-traversal.h"
#include "wasm.h"

namespace wasm {

inline bool isI32FuncRefType(Type t) {
  return t.isSignature() && t.getHeapType().getSignature().results == Type::i32;
}

inline bool isTrackedOneBitType(Type t) {
  return t.isInteger() || isI32FuncRefType(t);
}

struct ReachingDefInfo {
  LocalGet* get;
  SmallVector<Expression*, 4> defs;
  bool paramReaches;
};

// Compute reaching LocalSet definitions for each LocalGet in a function across
// structured control flow (Block, If, Loop, Try), allowing 1-bit boolean
// inference across coalesced locals whose live ranges share a local index.
inline void
computeReachingDefs(Function* func,
                    const std::vector<std::vector<Expression*>>& localWrites,
                    std::vector<ReachingDefInfo>& reachingDefs) {
  if (!func->body) {
    return;
  }
  Index numLocals = func->getNumLocals();
  if (numLocals == 0) {
    return;
  }
  std::vector<int32_t> localToTracked(numLocals, -1);
  Index numTracked = 0;
  for (Index i = 0; i < numLocals; i++) {
    if (isTrackedOneBitType(func->getLocalType(i)) &&
        i < localWrites.size() &&
        (localWrites[i].size() >= 2 ||
         (func->isParam(i) && !localWrites[i].empty()))) {
      localToTracked[i] = int32_t(numTracked++);
    }
  }
  if (numTracked == 0 || numTracked > 256) {
    return;
  }

  static constexpr size_t kMaxDefsPerLocal = 8;

  struct LocalState {
    SmallVector<Expression*, 4> defs;
    uint8_t paramReaches = 0;
    uint8_t unbounded = 0;

    void setSingleDef(Expression* val) {
      defs.clear();
      if (val && val->type != Type::unreachable) {
        defs.push_back(val);
        paramReaches = 0;
        unbounded = 0;
      } else {
        paramReaches = 0;
        unbounded = 1;
      }
    }

    void addDef(Expression* val) {
      if (unbounded) {
        return;
      }
      if (!val || val->type == Type::unreachable) {
        defs.clear();
        unbounded = 1;
        return;
      }
      for (auto* existing : defs) {
        if (existing == val) {
          return;
        }
      }
      if (defs.size() >= kMaxDefsPerLocal) {
        defs.clear();
        unbounded = 1;
        return;
      }
      defs.push_back(val);
    }

    void mergeFrom(const LocalState& other) {
      if (unbounded) {
        return;
      }
      if (other.unbounded) {
        defs.clear();
        unbounded = 1;
        return;
      }
      paramReaches |= other.paramReaches;
      for (auto* val : other.defs) {
        addDef(val);
        if (unbounded) {
          return;
        }
      }
    }
  };

  struct State {
    std::vector<LocalState> locals;
    bool reachable = true;

    State() = default;
    explicit State(Index n) : locals(n), reachable(true) {}

    void mergeFrom(const State& other) {
      if (!other.reachable) {
        return;
      }
      if (!reachable) {
        locals = other.locals;
        reachable = true;
        return;
      }
      for (size_t i = 0; i < locals.size(); i++) {
        locals[i].mergeFrom(other.locals[i]);
      }
    }
  } state(numTracked);

  for (Index i = 0; i < numLocals; i++) {
    int32_t t = localToTracked[i];
    if (t >= 0 && func->isParam(i)) {
      state.locals[t].paramReaches = 1;
    }
  }

  struct BlockFrame {
    Name name;
    State branchState;
    bool hasBranch = false;
  };
  std::vector<BlockFrame> blockFrames;

  struct CollectSets : PostWalker<CollectSets> {
    const std::vector<int32_t>& localToTracked;
    std::vector<std::pair<int32_t, Expression*>>& sets;
    CollectSets(const std::vector<int32_t>& localToTracked,
                std::vector<std::pair<int32_t, Expression*>>& sets)
      : localToTracked(localToTracked), sets(sets) {}
    void visitLocalSet(LocalSet* curr) {
      if (curr->index < localToTracked.size()) {
        int32_t t = localToTracked[curr->index];
        if (t >= 0 && curr->value->type != Type::unreachable) {
          sets.emplace_back(t, curr->value);
        }
      }
    }
  };

  std::function<void(Expression*)> walk = [&](Expression* curr) {
    if (!curr || !state.reachable) {
      return;
    }
    if (auto* lg = curr->dynCast<LocalGet>()) {
      if (lg->index < numLocals) {
        int32_t t = localToTracked[lg->index];
        if (t >= 0) {
          const auto& ls = state.locals[t];
          if (!ls.unbounded &&
              (ls.defs.size() < localWrites[lg->index].size() ||
               (func->isParam(lg->index) && !ls.paramReaches))) {
            reachingDefs.push_back({lg, ls.defs, ls.paramReaches != 0});
          }
        }
      }
      return;
    }
    if (auto* ls = curr->dynCast<LocalSet>()) {
      walk(ls->value);
      if (!state.reachable || ls->value->type == Type::unreachable) {
        state.reachable = false;
        return;
      }
      if (ls->index < numLocals) {
        int32_t t = localToTracked[ls->index];
        if (t >= 0) {
          state.locals[t].setSingleDef(ls->value);
        }
      }
      return;
    }
    if (auto* blk = curr->dynCast<Block>()) {
      if (blk->name.is()) {
        blockFrames.push_back({blk->name, State(numTracked), false});
        for (auto* item : blk->list) {
          walk(item);
          if (!state.reachable || item->type == Type::unreachable) {
            state.reachable = false;
            break;
          }
        }
        auto frame = std::move(blockFrames.back());
        blockFrames.pop_back();
        if (frame.hasBranch) {
          if (state.reachable) {
            state.mergeFrom(frame.branchState);
          } else {
            state = std::move(frame.branchState);
            state.reachable = true;
          }
        }
      } else {
        for (auto* item : blk->list) {
          walk(item);
          if (!state.reachable || item->type == Type::unreachable) {
            state.reachable = false;
            break;
          }
        }
      }
      if (blk->type == Type::unreachable) {
        state.reachable = false;
      }
      return;
    }
    if (auto* iff = curr->dynCast<If>()) {
      walk(iff->condition);
      if (!state.reachable || iff->condition->type == Type::unreachable) {
        state.reachable = false;
        return;
      }
      State preState = state;
      walk(iff->ifTrue);
      if (iff->ifTrue->type == Type::unreachable) {
        state.reachable = false;
      }
      State trueState = std::move(state);
      state = std::move(preState);
      if (iff->ifFalse) {
        walk(iff->ifFalse);
        if (iff->ifFalse->type == Type::unreachable) {
          state.reachable = false;
        }
        if (trueState.reachable && state.reachable) {
          state.mergeFrom(trueState);
        } else if (trueState.reachable) {
          state = std::move(trueState);
        }
      } else {
        if (trueState.reachable) {
          state.mergeFrom(trueState);
        }
      }
      if (iff->type == Type::unreachable) {
        state.reachable = false;
      }
      return;
    }
    if (curr->is<Loop>() || curr->is<Try>() || curr->is<TryTable>()) {
      std::vector<std::pair<int32_t, Expression*>> innerSets;
      CollectSets(localToTracked, innerSets).walk(curr);
      for (auto& [t, val] : innerSets) {
        state.locals[t].addDef(val);
      }
      State loopTryState = state;
      for (auto* child : ChildIterator(curr)) {
        state = loopTryState;
        walk(child);
        if (state.reachable) {
          loopTryState.mergeFrom(state);
        }
      }
      state = std::move(loopTryState);
      state.reachable = curr->type != Type::unreachable;
      return;
    }
    for (auto* child : ChildIterator(curr)) {
      walk(child);
      if (!state.reachable || child->type == Type::unreachable) {
        state.reachable = false;
        return;
      }
    }
    if ((curr->is<Break>() || curr->is<Switch>() || curr->is<BrOn>() ||
         curr->is<TryTable>()) &&
        BranchUtils::isBranchReachable(curr)) {
      BranchUtils::operateOnScopeNameUses(curr, [&](Name& name) {
        for (int i = int(blockFrames.size()) - 1; i >= 0; i--) {
          if (blockFrames[i].name == name) {
            if (blockFrames[i].hasBranch) {
              blockFrames[i].branchState.mergeFrom(state);
            } else {
              blockFrames[i].branchState = state;
              blockFrames[i].hasBranch = true;
            }
            break;
          }
        }
      });
    }
    if (curr->type == Type::unreachable) {
      state.reachable = false;
    }
  };

  walk(func->body);
}

// Closed-world module-wide 1-bit boolean value inference (greatest fixed point
// over struct fields, globals, function returns, function parameters, locals,
// reaching definitions, named blocks, and i32-returning function references).
struct ModuleOneBitInfo {
  std::unordered_map<HeapType, std::vector<bool>> oneBitStructFields;
  std::unordered_set<Name> oneBitGlobals;
  std::unordered_set<Name> oneBitArrayGlobals;
  std::unordered_set<Name> oneBitReturns;
  std::unordered_map<Name, std::vector<bool>> oneBitParams;
  std::unordered_map<Name, std::vector<bool>> oneBitLocals;
  std::unordered_set<LocalGet*> oneBitLocalGets;
  std::unordered_set<Block*> oneBitBlocks;
  std::unordered_map<Name, Name> negatedOneBitCallTwin;
  std::unordered_map<Name, Name> negatedBoolCallTwin;

  bool isOneBitStructField(HeapType ht, Index idx) const {
    auto it = oneBitStructFields.find(ht);
    return it != oneBitStructFields.end() && idx < it->second.size() &&
           it->second[idx];
  }

  bool isOneBitGlobal(Name name) const { return oneBitGlobals.contains(name); }

  bool isOneBitArrayGlobal(Name name) const {
    return oneBitArrayGlobals.contains(name);
  }

  bool isOneBitArrayGet(ArrayGet* ag) const {
    if (!ag) {
      return false;
    }
    Expression* ref = ag->ref;
    while (ref) {
      if (auto* ra = ref->dynCast<RefAs>(); ra && ra->op == RefAsNonNull) {
        ref = ra->value;
      } else if (auto* rc = ref->dynCast<RefCast>()) {
        ref = rc->ref;
      } else {
        break;
      }
    }
    if (auto* gg = ref ? ref->dynCast<GlobalGet>() : nullptr) {
      return oneBitArrayGlobals.contains(gg->name);
    }
    return false;
  }

  Name getNegatedOneBitCallTwin(Name target,
                                Name callerFunc = Name(),
                                bool mustBeOneBit = true) const {
    const auto& map =
      mustBeOneBit ? negatedOneBitCallTwin : negatedBoolCallTwin;
    auto it = map.find(target);
    if (it != map.end() && it->second != callerFunc) {
      return it->second;
    }
    return Name();
  }

  Name getNegatedOneBitCallTwin(Call* call,
                                Module& wasm,
                                Name callerFunc = Name(),
                                bool mustBeOneBit = true) const {
    if (!call || call->isReturn || call->type != Type::i32) {
      return Name();
    }
    Name twin =
      getNegatedOneBitCallTwin(call->target, callerFunc, mustBeOneBit);
    if (!twin.is()) {
      return Name();
    }
    auto* twinFunc = wasm.getFunctionOrNull(twin);
    if (!twinFunc || twinFunc->getResults() != Type::i32 ||
        twinFunc->getNumParams() != call->operands.size()) {
      return Name();
    }
    for (Index i = 0; i < call->operands.size(); i++) {
      Type argType = call->operands[i]->type;
      if (argType != Type::unreachable &&
          !Type::isSubType(argType, twinFunc->getLocalType(i))) {
        return Name();
      }
    }
    return twin;
  }

  bool isOneBitReturn(Name funcName) const {
    return oneBitReturns.contains(funcName);
  }

  bool isOneBitParam(Name funcName, Index paramIdx) const {
    auto it = oneBitParams.find(funcName);
    return it != oneBitParams.end() && paramIdx < it->second.size() &&
           it->second[paramIdx];
  }

  bool isOneBitLocal(Name funcName, Index localIdx) const {
    auto it = oneBitLocals.find(funcName);
    return it != oneBitLocals.end() && localIdx < it->second.size() &&
           it->second[localIdx];
  }

  bool isOneBitLocalGet(LocalGet* get) const {
    return oneBitLocalGets.contains(get);
  }

  bool isOneBitBlock(Block* block) const {
    return oneBitBlocks.contains(block);
  }

  bool isOneBitFuncRefExpr(Expression* curr,
                           const std::vector<bool>* currFuncLocals = nullptr,
                           Name funcName = Name(),
                           int depth = 0) const {
    if (!curr || depth > 8) {
      return false;
    }
    if (curr->is<RefNull>() || curr->type.isNull()) {
      return true;
    }
    if (!isI32FuncRefType(curr->type)) {
      return false;
    }
    if (auto* rf = curr->dynCast<RefFunc>()) {
      return isOneBitReturn(rf->func);
    }
    if (auto* lg = curr->dynCast<LocalGet>()) {
      if (currFuncLocals && lg->index < currFuncLocals->size() &&
          (*currFuncLocals)[lg->index]) {
        return true;
      }
      if (!funcName.isNull() && isOneBitLocal(funcName, lg->index)) {
        return true;
      }
      return isOneBitLocalGet(lg);
    }
    if (auto* ls = curr->dynCast<LocalSet>()) {
      return isOneBitFuncRefExpr(
        ls->value, currFuncLocals, funcName, depth + 1);
    }
    if (auto* gg = curr->dynCast<GlobalGet>()) {
      return isOneBitGlobal(gg->name);
    }
    if (auto* sg = curr->dynCast<StructGet>()) {
      return sg->ref->type.isStruct() &&
             isOneBitStructField(sg->ref->type.getHeapType(), sg->index);
    }
    if (auto* ag = curr->dynCast<ArrayGet>()) {
      return isOneBitArrayGet(ag);
    }
    if (auto* call = curr->dynCast<Call>()) {
      return isOneBitReturn(call->target);
    }
    if (auto* cr = curr->dynCast<CallRef>()) {
      return isOneBitFuncRefExpr(
        cr->target, currFuncLocals, funcName, depth + 1);
    }
    if (auto* ra = curr->dynCast<RefAs>()) {
      return ra->op == RefAsNonNull &&
             isOneBitFuncRefExpr(
               ra->value, currFuncLocals, funcName, depth + 1);
    }
    if (auto* rc = curr->dynCast<RefCast>()) {
      return isOneBitFuncRefExpr(rc->ref, currFuncLocals, funcName, depth + 1);
    }
    if (auto* bo = curr->dynCast<BrOn>()) {
      return (bo->op == BrOnNull || bo->op == BrOnCastFail) &&
             isOneBitFuncRefExpr(bo->ref, currFuncLocals, funcName, depth + 1);
    }
    if (auto* blk = curr->dynCast<Block>()) {
      if (!blk->name.is()) {
        return !blk->list.empty() &&
               isOneBitFuncRefExpr(
                 blk->list.back(), currFuncLocals, funcName, depth + 1);
      }
      return isOneBitBlock(blk);
    }
    if (auto* sel = curr->dynCast<Select>()) {
      return isOneBitFuncRefExpr(
               sel->ifTrue, currFuncLocals, funcName, depth + 1) &&
             isOneBitFuncRefExpr(
               sel->ifFalse, currFuncLocals, funcName, depth + 1);
    }
    if (auto* iff = curr->dynCast<If>()) {
      return iff->ifFalse &&
             (iff->ifTrue->type == Type::unreachable ||
              isOneBitFuncRefExpr(
                iff->ifTrue, currFuncLocals, funcName, depth + 1)) &&
             (iff->ifFalse->type == Type::unreachable ||
              isOneBitFuncRefExpr(
                iff->ifFalse, currFuncLocals, funcName, depth + 1));
    }
    return false;
  }

  void compute(Module& wasm, const PassOptions& options) {
    oneBitStructFields.clear();
    oneBitGlobals.clear();
    oneBitArrayGlobals.clear();
    oneBitReturns.clear();
    oneBitParams.clear();
    oneBitLocals.clear();
    oneBitLocalGets.clear();
    oneBitBlocks.clear();
    negatedOneBitCallTwin.clear();
    negatedBoolCallTwin.clear();

    bool isClosedWorld = options.worldMode == WorldMode::Closed;
    bool hasOneBitArrayGlobalCandidate = false;
    if (!isClosedWorld && wasm.features.hasGC()) {
      for (auto& g : wasm.globals) {
        if (!g->imported() && !g->mutable_ && g->init && g->type.isArray()) {
          auto arr = g->type.getHeapType().getArray();
          if (arr.element.mutable_ == Immutable &&
              isI32FuncRefType(arr.element.type) &&
              g->init->is<ArrayNewFixed>()) {
            hasOneBitArrayGlobalCandidate = true;
            break;
          }
        }
      }
    }

    if (!isClosedWorld && !hasOneBitArrayGlobalCandidate) {
      for (auto& f : wasm.functions) {
        if (f->imported()) {
          if (f->module == WasmStringsModule &&
              (f->base == "equals" || f->base == "test") &&
              f->getResults() == Type::i32) {
            oneBitReturns.insert(f->name);
          }
        } else if (f->getResults() == Type::i32 && f->effects &&
                   f->effects->returnsOneBit) {
          oneBitReturns.insert(f->name);
        }
      }
      return;
    }

    std::optional<SubTypes> subTypes;
    if (wasm.features.hasGC()) {
      subTypes.emplace(wasm);
      for (HeapType ht : subTypes->types) {
        if (ht.isStruct()) {
          const auto& fields = ht.getStruct().fields;
          auto& vec = oneBitStructFields[ht];
          vec.resize(fields.size(), false);
          for (Index i = 0; i < fields.size(); i++) {
            vec[i] = isClosedWorld
                       ? isTrackedOneBitType(fields[i].type)
                       : (fields[i].mutable_ == Immutable &&
                          isI32FuncRefType(fields[i].type));
          }
        }
      }
    }

    struct FuncSummary {
      std::vector<std::vector<Expression*>> localWrites;
      std::vector<ReachingDefInfo> reachingDefs;
      std::unordered_map<Block*, std::vector<Expression*>> blockValues;
      std::unordered_set<Block*> disqualifiedBlocks;
      std::vector<Expression*> returnValues;
      std::vector<Name> tailCallTargets;
      std::vector<CallRef*> tailCallRefs;
      bool hasIndirectTailCall = false;
      std::vector<std::pair<Name, Expression*>> globalWrites;
      std::vector<std::pair<std::pair<HeapType, Index>, Expression*>>
        structNewWrites;
      std::vector<std::pair<std::pair<HeapType, Index>, Expression*>>
        structSetWrites;
      std::vector<std::pair<HeapType, Index>> structDisqualifiedFields;
      std::vector<Call*> directCalls;
      std::vector<CallRef*> callRefs;
      std::vector<Name> refFuncs;
    };

    struct SummaryScanner
      : public ExpressionStackWalker<SummaryScanner,
                                     UnifiedExpressionVisitor<SummaryScanner>> {
      FuncSummary& summary;
      Function* func;
      SummaryScanner(FuncSummary& summary, Function* func = nullptr)
        : summary(summary), func(func) {}

      void visitExpression(Expression* curr) {
        if ((curr->is<Break>() || curr->is<Switch>() || curr->is<BrOn>() ||
             curr->is<TryTable>()) &&
            BranchUtils::isBranchReachable(curr)) {
          BranchUtils::operateOnScopeNameUsesAndSentValues(
            curr, [&](Name& name, Expression* val) {
              if (auto* target = findBreakTarget(name)) {
                if (auto* blk = target->dynCast<Block>();
                    blk && blk->name.is() && isTrackedOneBitType(blk->type)) {
                  if (!val) {
                    summary.disqualifiedBlocks.insert(blk);
                  } else if (val->type != Type::unreachable) {
                    summary.blockValues[blk].push_back(val);
                  }
                }
              }
            });
        }
        if (auto* blk = curr->dynCast<Block>()) {
          if (blk->name.is() && isTrackedOneBitType(blk->type)) {
            auto& vec = summary.blockValues[blk];
            if (blk->list.empty()) {
              summary.disqualifiedBlocks.insert(blk);
            } else if (blk->list.back()->type != Type::unreachable) {
              vec.push_back(blk->list.back());
            }
          }
        } else if (auto* ls = curr->dynCast<LocalSet>()) {
          if (func && ls->index < summary.localWrites.size() &&
              isTrackedOneBitType(func->getLocalType(ls->index)) &&
              ls->value->type != Type::unreachable) {
            summary.localWrites[ls->index].push_back(ls->value);
          }
        } else if (auto* gs = curr->dynCast<GlobalSet>()) {
          if (isTrackedOneBitType(gs->value->type) ||
              gs->value->type.isNull()) {
            summary.globalWrites.emplace_back(gs->name, gs->value);
          }
        } else if (auto* ret = curr->dynCast<Return>()) {
          if (func && isTrackedOneBitType(func->getResults()) && ret->value &&
              ret->value->type != Type::unreachable) {
            summary.returnValues.push_back(ret->value);
          }
        } else if (auto* call = curr->dynCast<Call>()) {
          bool reachable = true;
          for (auto* op : call->operands) {
            if (op->type == Type::unreachable) {
              reachable = false;
              break;
            }
          }
          if (reachable) {
            summary.directCalls.push_back(call);
            if (call->isReturn) {
              summary.tailCallTargets.push_back(call->target);
            }
          }
        } else if (auto* ci = curr->dynCast<CallIndirect>()) {
          if (ci->isReturn) {
            summary.hasIndirectTailCall = true;
          }
        } else if (auto* cr = curr->dynCast<CallRef>()) {
          bool reachable = cr->target->type != Type::unreachable;
          for (auto* op : cr->operands) {
            if (op->type == Type::unreachable) {
              reachable = false;
              break;
            }
          }
          if (reachable) {
            if (cr->target->type.isSignature()) {
              summary.callRefs.push_back(cr);
            }
            if (cr->isReturn) {
              summary.tailCallRefs.push_back(cr);
            }
          }
        } else if (auto* rf = curr->dynCast<RefFunc>()) {
          summary.refFuncs.push_back(rf->func);
        } else if (auto* sn = curr->dynCast<StructNew>()) {
          if (sn->type.isStruct() && !sn->isWithDefault()) {
            HeapType ht = sn->type.getHeapType();
            const auto& fields = ht.getStruct().fields;
            for (Index i = 0; i < fields.size() && i < sn->operands.size();
                 i++) {
              if (isTrackedOneBitType(fields[i].type) &&
                  sn->operands[i]->type != Type::unreachable) {
                summary.structNewWrites.push_back({{ht, i}, sn->operands[i]});
              }
            }
          }
        } else if (auto* ss = curr->dynCast<StructSet>()) {
          if (ss->ref->type.isStruct() &&
              ss->value->type != Type::unreachable) {
            HeapType ht = ss->ref->type.getHeapType();
            const auto& fields = ht.getStruct().fields;
            if (ss->index < fields.size() &&
                isTrackedOneBitType(fields[ss->index].type)) {
              summary.structSetWrites.push_back({{ht, ss->index}, ss->value});
            }
          }
        } else if (auto* rmw = curr->dynCast<StructRMW>()) {
          if (rmw->ref->type.isStruct()) {
            summary.structDisqualifiedFields.push_back(
              {rmw->ref->type.getHeapType(), rmw->index});
          }
        } else if (auto* cmpx = curr->dynCast<StructCmpxchg>()) {
          if (cmpx->ref->type.isStruct()) {
            summary.structDisqualifiedFields.push_back(
              {cmpx->ref->type.getHeapType(), cmpx->index});
          }
        }
      }
    };

    ModuleUtils::ParallelFunctionAnalysis<FuncSummary> analysis(
      wasm, [&](Function* func, FuncSummary& summary) {
        if (func->imported() || !func->body) {
          return;
        }
        summary.localWrites.resize(func->getNumLocals());
        SummaryScanner scanner(summary, func);
        scanner.walk(func->body);
        if (isTrackedOneBitType(func->getResults()) &&
            func->body->type != Type::unreachable) {
          summary.returnValues.push_back(func->body);
        }
        computeReachingDefs(func, summary.localWrites, summary.reachingDefs);
      });

    FuncSummary moduleSummary;
    {
      SummaryScanner scanner(moduleSummary);
      for (auto& g : wasm.globals) {
        if (!g->imported() && g->init) {
          scanner.walk(g->init);
        }
      }
      for (auto& seg : wasm.elementSegments) {
        for (auto* e : seg->data) {
          scanner.walk(e);
        }
      }
    }

    std::unordered_set<Name> exportedOrStartFuncs;
    std::unordered_set<Name> refFuncTaken;
    std::unordered_set<Name> exportedGlobals;
    for (auto& ex : wasm.exports) {
      if (ex->kind == ExternalKind::Function) {
        exportedOrStartFuncs.insert(*ex->getInternalName());
      } else if (ex->kind == ExternalKind::Global) {
        exportedGlobals.insert(*ex->getInternalName());
      }
    }
    if (wasm.start.is()) {
      exportedOrStartFuncs.insert(wasm.start);
    }
    for (Name rf : moduleSummary.refFuncs) {
      refFuncTaken.insert(rf);
    }

    std::unordered_set<HeapType> unoptimizableSigTypes;
    bool canInferIndirectParams =
      wasm.features.hasGC() && wasm.tables.empty() && subTypes.has_value();
    if (canInferIndirectParams) {
      for (auto type :
           ModuleUtils::getPublicHeapTypes(wasm, options.worldMode)) {
        if (type.isFunction()) {
          unoptimizableSigTypes.insert(type);
        }
      }
      for (auto& tag : wasm.tags) {
        unoptimizableSigTypes.insert(tag->type);
      }
      if (wasm.features.hasStackSwitching()) {
        for (auto type : subTypes->types) {
          if (type.isContinuation()) {
            unoptimizableSigTypes.insert(type.getContinuation().type);
          }
        }
      }
      for (auto func : Intrinsics(wasm).getJSCalledFunctions()) {
        unoptimizableSigTypes.insert(wasm.getFunction(func)->type.getHeapType());
      }
    }

    std::unordered_map<Name, Index> directCallCount;
    std::unordered_set<HeapType> sigHasCallRef;
    for (auto& [func, fs] : analysis.map) {
      for (Name rf : fs.refFuncs) {
        refFuncTaken.insert(rf);
      }
      for (Call* call : fs.directCalls) {
        directCallCount[call->target]++;
        if (canInferIndirectParams &&
            Intrinsics(wasm).isCallWithoutEffects(call) &&
            !call->operands.empty() &&
            call->operands.back()->type.isFunction()) {
          unoptimizableSigTypes.insert(
            call->operands.back()->type.getHeapType());
        }
      }
      if (canInferIndirectParams) {
        for (CallRef* cr : fs.callRefs) {
          HeapType sigHt = cr->target->type.getHeapType();
          if (sigHasCallRef.insert(sigHt).second) {
            subTypes->iterSubTypes(sigHt, [&](HeapType subHt, int) {
              sigHasCallRef.insert(subHt);
              return true;
            });
          }
        }
      }
    }
    if (canInferIndirectParams && !unoptimizableSigTypes.empty()) {
      std::vector<HeapType> seeds(unoptimizableSigTypes.begin(),
                                  unoptimizableSigTypes.end());
      for (HeapType ht : seeds) {
        subTypes->iterSubTypes(ht, [&](HeapType subHt, int) {
          unoptimizableSigTypes.insert(subHt);
          return true;
        });
      }
    }

    for (auto& g : wasm.globals) {
      if (!g->imported() && g->init) {
        if ((isClosedWorld ? isTrackedOneBitType(g->type)
                           : isI32FuncRefType(g->type)) &&
            (!g->mutable_ ||
             (isClosedWorld && !exportedGlobals.contains(g->name)))) {
          oneBitGlobals.insert(g->name);
        } else if (!g->mutable_ && g->type.isArray()) {
          auto elem = g->type.getHeapType().getArray().element;
          if (elem.mutable_ == Immutable && isTrackedOneBitType(elem.type) &&
              (g->init->is<ArrayNewFixed>() || g->init->is<ArrayNew>())) {
            oneBitArrayGlobals.insert(g->name);
          }
        }
      }
    }

    std::unordered_map<HeapType, std::vector<Name>> refFuncsBySig;
    std::unordered_map<HeapType, std::vector<bool>> oneBitSigParams;
    if (isClosedWorld && canInferIndirectParams) {
      for (HeapType ht : subTypes->types) {
        if (ht.isSignature()) {
          auto params = ht.getSignature().params;
          auto& vec = oneBitSigParams[ht];
          vec.resize(params.size(), false);
          for (Index i = 0; i < params.size(); i++) {
            vec[i] = isTrackedOneBitType(params[i]);
          }
        }
      }
    }

    for (auto& f : wasm.functions) {
      if (f->imported()) {
        if (f->module == WasmStringsModule &&
            (f->base == "equals" || f->base == "test") &&
            f->getResults() == Type::i32) {
          oneBitReturns.insert(f->name);
        }
        continue;
      }
      if (!f->body) {
        continue;
      }
      auto& fs = analysis.map[f.get()];
      if (isTrackedOneBitType(f->getResults()) && !fs.hasIndirectTailCall &&
          (isClosedWorld || refFuncTaken.contains(f->name))) {
        oneBitReturns.insert(f->name);
      }
      auto& params = oneBitParams[f->name];
      params.resize(f->getNumParams(), false);
      auto& locals = oneBitLocals[f->name];
      locals.resize(f->getNumLocals(), false);
      if (!isClosedWorld) {
        continue;
      }
      bool isExported = exportedOrStartFuncs.contains(f->name);
      bool isRefTaken = refFuncTaken.contains(f->name);
      HeapType fSig = f->type.getHeapType();
      bool canInferParams = false;
      if (!isExported) {
        if (!isRefTaken) {
          canInferParams = directCallCount[f->name] > 0;
        } else if (canInferIndirectParams &&
                   !unoptimizableSigTypes.contains(fSig)) {
          canInferParams =
            directCallCount[f->name] > 0 || sigHasCallRef.contains(fSig);
          if (canInferParams) {
            refFuncsBySig[fSig].push_back(f->name);
          }
        }
      }
      for (Index p = 0; p < f->getNumParams(); p++) {
        if (canInferParams && isTrackedOneBitType(f->getLocalType(p))) {
          params[p] = true;
        }
      }
      for (Index i = 0; i < f->getNumLocals(); i++) {
        if (isTrackedOneBitType(f->getLocalType(i)) &&
            (!f->isParam(i) || canInferParams)) {
          locals[i] = true;
        }
      }
      for (auto& rd : fs.reachingDefs) {
        oneBitLocalGets.insert(rd.get);
      }
      for (auto& [blk, vals] : fs.blockValues) {
        if (!fs.disqualifiedBlocks.contains(blk) && !vals.empty()) {
          oneBitBlocks.insert(blk);
        }
      }
    }

    bool changed = false;
    auto disqualifyStructFieldAndSupers = [&](HeapType ht, Index idx) {
      std::optional<HeapType> curr = ht;
      while (curr && curr->isStruct()) {
        auto it = oneBitStructFields.find(*curr);
        if (it == oneBitStructFields.end() || idx >= it->second.size()) {
          break;
        }
        if (it->second[idx]) {
          it->second[idx] = false;
          changed = true;
        }
        curr = curr->getDeclaredSuperType();
      }
    };

    std::unordered_set<std::pair<HeapType, Index>> disqualifiedStructSetFields;
    auto disqualifyStructFieldSubtreeAndSupers = [&](HeapType ht, Index idx) {
      if (!subTypes || !disqualifiedStructSetFields.insert({ht, idx}).second) {
        return;
      }
      disqualifyStructFieldAndSupers(ht, idx);
      subTypes->iterSubTypes(ht, [&](HeapType subHt, int) {
        auto it = oneBitStructFields.find(subHt);
        if (it != oneBitStructFields.end() && idx < it->second.size() &&
            it->second[idx]) {
          it->second[idx] = false;
          changed = true;
        }
        return true;
      });
    };

    auto disqualifySigParam = [&](HeapType sigHt, Index p) {
      if (!subTypes) {
        return;
      }
      auto it = oneBitSigParams.find(sigHt);
      if (it == oneBitSigParams.end() || p >= it->second.size() ||
          !it->second[p]) {
        return;
      }
      subTypes->iterSubTypes(sigHt, [&](HeapType subHt, int) {
        auto subIt = oneBitSigParams.find(subHt);
        if (subIt != oneBitSigParams.end() && p < subIt->second.size() &&
            subIt->second[p]) {
          subIt->second[p] = false;
          auto fnIt = refFuncsBySig.find(subHt);
          if (fnIt != refFuncsBySig.end()) {
            for (Name fnName : fnIt->second) {
              auto& fnParams = oneBitParams[fnName];
              if (p < fnParams.size() && fnParams[p]) {
                fnParams[p] = false;
                changed = true;
              }
              auto& fnLocals = oneBitLocals[fnName];
              if (p < fnLocals.size() && fnLocals[p]) {
                fnLocals[p] = false;
                changed = true;
              }
            }
          }
        }
        return true;
      });
    };

    for (auto& [ht, idx] : moduleSummary.structDisqualifiedFields) {
      disqualifyStructFieldSubtreeAndSupers(ht, idx);
    }
    for (auto& [func, fs] : analysis.map) {
      for (auto& [ht, idx] : fs.structDisqualifiedFields) {
        disqualifyStructFieldSubtreeAndSupers(ht, idx);
      }
    }

    struct FPBitProvider : Bits::DummyLocalInfoProvider {
      const ModuleOneBitInfo& info;
      const std::vector<bool>* currFuncLocals = nullptr;
      FPBitProvider(const ModuleOneBitInfo& info) : info(info) {}

      Index getMaxBitsForLocal(LocalGet* get) {
        if ((currFuncLocals && get->index < currFuncLocals->size() &&
             (*currFuncLocals)[get->index]) ||
            info.isOneBitLocalGet(get)) {
          return 1;
        }
        return get->type == Type::i64 ? 64 : 32;
      }
      Index getMaxBitsForBlock(Block* block) {
        if (info.isOneBitBlock(block)) {
          return 1;
        }
        return block->type == Type::i64 ? 64 : 32;
      }
      Index getMaxBitsForStructGet(StructGet* get) {
        if (get->ref->type.isStruct() &&
            info.isOneBitStructField(get->ref->type.getHeapType(),
                                     get->index)) {
          return 1;
        }
        return get->type == Type::i64 ? 64 : 32;
      }
      Index getMaxBitsForCall(Call* call) {
        if (call->type == Type::i32 && info.isOneBitReturn(call->target)) {
          return 1;
        }
        return call->type == Type::i64 ? 64 : 32;
      }
      Index getMaxBitsForCallRef(CallRef* call) {
        if (call->type == Type::i32 &&
            info.isOneBitFuncRefExpr(call->target, currFuncLocals)) {
          return 1;
        }
        return call->type == Type::i64 ? 64 : 32;
      }
      Index getMaxBitsForGlobalGet(GlobalGet* get) {
        if (info.isOneBitGlobal(get->name)) {
          return 1;
        }
        return get->type == Type::i64 ? 64 : 32;
      }
      Index getMaxBitsForArrayGet(ArrayGet* get) {
        if (info.isOneBitArrayGet(get)) {
          return 1;
        }
        return get->type == Type::i64 ? 64 : 32;
      }
    } provider(*this);

    auto isOneBitExpr = [&](Expression* val) -> bool {
      if (val->type.isInteger()) {
        return Bits::getMaxBits(val, &provider) <= 1;
      }
      return isOneBitFuncRefExpr(val, provider.currFuncLocals);
    };

    changed = true;
    while (changed) {
      changed = false;
      provider.currFuncLocals = nullptr;

      for (auto& g : wasm.globals) {
        if (oneBitGlobals.contains(g->name) && !isOneBitExpr(g->init)) {
          oneBitGlobals.erase(g->name);
          changed = true;
        }
        if (oneBitArrayGlobals.contains(g->name)) {
          bool ok = true;
          if (auto* anf = g->init->dynCast<ArrayNewFixed>()) {
            for (auto* val : anf->values) {
              if (!val->type.isNull() && !isOneBitExpr(val)) {
                ok = false;
                break;
              }
            }
          } else if (auto* an = g->init->dynCast<ArrayNew>()) {
            if (!an->isWithDefault() && !an->init->type.isNull() &&
                !isOneBitExpr(an->init)) {
              ok = false;
            }
          } else {
            ok = false;
          }
          if (!ok) {
            oneBitArrayGlobals.erase(g->name);
            changed = true;
          }
        }
      }
      for (auto& [field, val] : moduleSummary.structNewWrites) {
        if (isOneBitStructField(field.first, field.second) &&
            !isOneBitExpr(val)) {
          disqualifyStructFieldAndSupers(field.first, field.second);
        }
      }
      for (auto& [field, val] : moduleSummary.structSetWrites) {
        if (!disqualifiedStructSetFields.contains(field) &&
            !isOneBitExpr(val)) {
          disqualifyStructFieldSubtreeAndSupers(field.first, field.second);
        }
      }

      for (auto& f : wasm.functions) {
        if (f->imported() || !f->body) {
          continue;
        }
        auto& fs = analysis.map[f.get()];
        auto& params = oneBitParams[f->name];
        auto& locals = oneBitLocals[f->name];
        provider.currFuncLocals = &locals;

        for (Index i = 0; i < locals.size(); i++) {
          if (!locals[i]) {
            continue;
          }
          if (i < params.size() && !params[i]) {
            locals[i] = false;
            changed = true;
            continue;
          }
          for (auto* val : fs.localWrites[i]) {
            if (!isOneBitExpr(val)) {
              locals[i] = false;
              changed = true;
              break;
            }
          }
        }

        for (auto& rd : fs.reachingDefs) {
          if (rd.get->index < locals.size() && locals[rd.get->index]) {
            continue;
          }
          if (!oneBitLocalGets.contains(rd.get)) {
            continue;
          }
          bool ok = true;
          if (rd.paramReaches &&
              (rd.get->index >= params.size() || !params[rd.get->index])) {
            ok = false;
          }
          if (ok) {
            for (auto* defVal : rd.defs) {
              if (!isOneBitExpr(defVal)) {
                ok = false;
                break;
              }
            }
          }
          if (!ok) {
            oneBitLocalGets.erase(rd.get);
            changed = true;
          }
        }

        for (auto& [blk, vals] : fs.blockValues) {
          if (!oneBitBlocks.contains(blk)) {
            continue;
          }
          for (auto* val : vals) {
            if (!isOneBitExpr(val)) {
              oneBitBlocks.erase(blk);
              changed = true;
              break;
            }
          }
        }

        if (oneBitReturns.contains(f->name)) {
          bool ok = true;
          for (Name target : fs.tailCallTargets) {
            if (!oneBitReturns.contains(target)) {
              ok = false;
              break;
            }
          }
          if (ok) {
            for (CallRef* cr : fs.tailCallRefs) {
              if (!isOneBitFuncRefExpr(cr->target, provider.currFuncLocals)) {
                ok = false;
                break;
              }
            }
          }
          if (ok) {
            for (auto* val : fs.returnValues) {
              if (!isOneBitExpr(val)) {
                ok = false;
                break;
              }
            }
          }
          if (!ok) {
            oneBitReturns.erase(f->name);
            changed = true;
          }
        }

        for (auto& [gName, val] : fs.globalWrites) {
          if (oneBitGlobals.contains(gName) && !isOneBitExpr(val)) {
            oneBitGlobals.erase(gName);
            changed = true;
          }
        }

        for (auto& [field, val] : fs.structNewWrites) {
          if (isOneBitStructField(field.first, field.second) &&
              !isOneBitExpr(val)) {
            disqualifyStructFieldAndSupers(field.first, field.second);
          }
        }

        for (auto& [field, val] : fs.structSetWrites) {
          if (!disqualifiedStructSetFields.contains(field) &&
              !isOneBitExpr(val)) {
            disqualifyStructFieldSubtreeAndSupers(field.first, field.second);
          }
        }

        for (Call* call : fs.directCalls) {
          auto pIt = oneBitParams.find(call->target);
          auto lIt = oneBitLocals.find(call->target);
          if (pIt == oneBitParams.end() || lIt == oneBitLocals.end()) {
            continue;
          }
          auto& targetParams = pIt->second;
          auto& targetLocals = lIt->second;
          for (Index p = 0;
               p < call->operands.size() && p < targetParams.size();
               p++) {
            if (targetParams[p] && !isOneBitExpr(call->operands[p])) {
              targetParams[p] = false;
              if (p < targetLocals.size()) {
                targetLocals[p] = false;
              }
              changed = true;
            }
          }
        }

        if (canInferIndirectParams) {
          for (CallRef* cr : fs.callRefs) {
            HeapType sigHt = cr->target->type.getHeapType();
            auto sigIt = oneBitSigParams.find(sigHt);
            if (sigIt == oneBitSigParams.end()) {
              continue;
            }
            for (Index p = 0;
                 p < cr->operands.size() && p < sigIt->second.size();
                 p++) {
              if (sigIt->second[p] && !isOneBitExpr(cr->operands[p])) {
                disqualifySigParam(sigHt, p);
              }
            }
          }
        }
      }
    }

    std::vector<std::pair<Name, Name>> wrapperPairs;
    for (auto& f : wasm.functions) {
      if (f->imported() || !f->body || f->getResults() != Type::i32 ||
          f->getNumVars() != 0) {
        continue;
      }
      Expression* body = f->body;
      if (auto* ret = body->dynCast<Return>(); ret && ret->value) {
        body = ret->value;
      }
      auto* u = body->dynCast<Unary>();
      if (!u || u->op != EqZInt32) {
        continue;
      }
      auto* call = u->value->dynCast<Call>();
      if (!call || call->isReturn ||
          call->operands.size() != f->getNumParams() ||
          call->target == f->name) {
        continue;
      }
      auto* targetFunc = wasm.getFunctionOrNull(call->target);
      if (!targetFunc || targetFunc->getResults() != Type::i32 ||
          targetFunc->getNumParams() != f->getNumParams() ||
          !Type::isSubType(f->getParams(), targetFunc->getParams())) {
        continue;
      }
      bool argsMatch = true;
      for (Index i = 0; i < call->operands.size(); i++) {
        auto* lg = call->operands[i]->dynCast<LocalGet>();
        if (!lg || lg->index != i) {
          argsMatch = false;
          break;
        }
      }
      if (!argsMatch) {
        continue;
      }
      negatedBoolCallTwin[f->name] = call->target;
      if (oneBitReturns.contains(call->target)) {
        negatedOneBitCallTwin[f->name] = call->target;
      }
      wrapperPairs.emplace_back(f->name, call->target);
    }
    for (const auto& [wrapperName, targetName] : wrapperPairs) {
      if (directCallCount[wrapperName] >= 2) {
        if (!negatedOneBitCallTwin.contains(targetName)) {
          negatedOneBitCallTwin[targetName] = wrapperName;
        }
        if (!negatedBoolCallTwin.contains(targetName)) {
          negatedBoolCallTwin[targetName] = wrapperName;
        }
      }
    }

    for (auto& f : wasm.functions) {
      if (f->getResults() == Type::i32 && f->effects) {
        bool isOneBit = oneBitReturns.contains(f->name);
        if (f->effects->returnsOneBit != isOneBit) {
          if (f->effects.use_count() == 1) {
            const_cast<EffectAnalyzer*>(f->effects.get())->returnsOneBit =
              isOneBit;
          } else {
            auto copy = std::make_shared<EffectAnalyzer>(*f->effects);
            copy->returnsOneBit = isOneBit;
            f->effects = std::move(copy);
          }
        }
      }
    }
  }
};

} // namespace wasm

#endif // wasm_ir_one_bit_h
