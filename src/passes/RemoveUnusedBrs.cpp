/*
 * Copyright 2015 WebAssembly Community Group participants
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
// Removes branches for which we go to where they go anyhow.
//

#include "ir/branch-hints.h"
#include "ir/branch-utils.h"
#include "ir/cost.h"
#include "ir/drop.h"
#include "ir/effects.h"
#include "ir/find_all.h"
#include "ir/gc-type-utils.h"
#include "ir/iteration.h"
#include "ir/literal-utils.h"
#include "ir/local-graph.h"
#include "ir/localize.h"
#include "ir/properties.h"
#include "ir/utils.h"
#include "parsing.h"
#include "pass.h"
#include "support/small_set.h"
#include "wasm-builder.h"
#include "wasm-type.h"
#include "wasm.h"

namespace wasm {

// Grab a slice out of a block, replacing it with nops, and returning
// either another block with the contents (if more than 1) or a single
// expression.
// This does not finalize the input block; it leaves that for the caller.
static Expression*
stealSlice(Builder& builder, Block* input, Index from, Index to) {
  Expression* ret;
  if (to == from + 1) {
    // just one
    ret = input->list[from];
  } else {
    auto* block = builder.makeBlock();
    for (Index i = from; i < to; i++) {
      block->list.push_back(input->list[i]);
    }
    block->finalize();
    ret = block;
  }
  if (to == input->list.size()) {
    input->list.resize(from);
  } else {
    for (Index i = from; i < to; i++) {
      input->list[i] = builder.makeNop();
    }
  }
  return ret;
}

// Check if a single expression is too costly to move from a conditional to an
// unconditional place.
static bool tooCostlyToRunUnconditionally(const PassOptions& passOptions,
                                          Expression* curr);

// To turn an if into a br-if, we must be able to reorder the condition and
// possible value, and the possible value must not have side effects (as they
// would run unconditionally). Also, if there is a value, it must not be too
// expensive to run unconditionally.
static bool canTurnIfIntoBrIf(Expression* ifCondition,
                              Expression* brValue,
                              PassOptions& options,
                              Module& wasm) {
  // if the if isn't even reached, this is all dead code anyhow
  if (ifCondition->type == Type::unreachable) {
    return false;
  }
  if (!brValue) {
    return true;
  }
  if (tooCostlyToRunUnconditionally(options, brValue)) {
    return false;
  }
  EffectAnalyzer value(options, wasm, brValue);
  if (value.hasSideEffects()) {
    return false;
  }
  return !EffectAnalyzer(options, wasm, ifCondition).invalidates(value);
}

// This leads to similar choices as LLVM does in some cases, by balancing the
// extra work of code that is run unconditionally with the speedup from not
// branching to decide whether to run it or not.
// See:
//  * https://github.com/WebAssembly/binaryen/pull/4228
//  * https://github.com/WebAssembly/binaryen/issues/5983
const Index TooCostlyToRunUnconditionally = 8;

// Some costs are known to be too high to move from conditional to unconditional
// execution.
static_assert(CostAnalyzer::AtomicCost >= TooCostlyToRunUnconditionally,
              "We never run atomics unconditionally");
static_assert(CostAnalyzer::ThrowCost >= TooCostlyToRunUnconditionally,
              "We never run throws unconditionally");
static_assert(CostAnalyzer::CastCost > TooCostlyToRunUnconditionally / 2,
              "We only run casts unconditionally when optimizing for size");

static bool tooCostlyToRunUnconditionally(const PassOptions& passOptions,
                                          Index cost) {
  if (passOptions.shrinkLevel == 0) {
    // We are focused on speed. Any extra cost is risky, but allow a small
    // amount.
    return cost > TooCostlyToRunUnconditionally / 2;
  } else if (passOptions.shrinkLevel == 1) {
    // We are optimizing for size in a balanced manner. Allow some extra
    // overhead here.
    return cost >= TooCostlyToRunUnconditionally;
  } else {
    // We should have already decided what to do if shrink_level=2 and not
    // gotten here, and other values are invalid.
    WASM_UNREACHABLE("bad shrink level");
  }
}

static bool tooCostlyToRunUnconditionally(const PassOptions& passOptions,
                                          Expression* curr) {
  // If we care entirely about code size, just do it for that reason (early
  // exit to avoid work).
  if (passOptions.shrinkLevel >= 2) {
    return false;
  }
  auto cost = CostAnalyzer(curr).cost;
  return tooCostlyToRunUnconditionally(passOptions, cost);
}

// Check if it is not worth it to run code unconditionally. This
// assumes we are trying to run two expressions where previously
// only one of the two might have executed. We assume here that
// executing both is good for code size.
static bool tooCostlyToRunUnconditionally(const PassOptions& passOptions,
                                          Expression* one,
                                          Expression* two) {
  // If we care entirely about code size, just do it for that reason (early
  // exit to avoid work).
  if (passOptions.shrinkLevel >= 2) {
    return false;
  }

  // Consider the cost of executing all the code unconditionally, which adds
  // either the cost of running one or two, so the maximum is the worst case.
  auto max = std::max(CostAnalyzer(one).cost, CostAnalyzer(two).cost);
  return tooCostlyToRunUnconditionally(passOptions, max);
}

static bool isI32Const(Expression* expr, int32_t& out) {
  if (auto* c = expr->dynCast<Const>()) {
    if (c->type == Type::i32) {
      out = c->value.geti32();
      return true;
    }
  }
  return false;
}

static thread_local bool currentPassV4OverrideSet = false;
static thread_local bool currentPassV4Override = false;

static bool isV4Enabled(const PassOptions& passOptions) {
  if (passOptions.shrinkLevel < 2) {
    return false;
  }
  if (currentPassV4OverrideSet) {
    return currentPassV4Override;
  }
  auto it = passOptions.arguments.find("remove-unused-brs-v4");
  if (it != passOptions.arguments.end()) {
    return it->second != "0";
  }
  return true;
}

// Check whether `curr` is guaranteed to evaluate to 0 or 1 (extending
// Properties::emitsBoolean with 0/1 constants, bitwise ops, select, if, tee,
// and unnamed blocks).
static bool isOneBitI32(Expression* curr, Index depth = 0) {
  if (!curr || curr->type != Type::i32 || depth > 8) {
    return false;
  }
  if (Properties::emitsBoolean(curr)) {
    return true;
  }
  int32_t c = 0;
  if (isI32Const(curr, c)) {
    return c == 0 || c == 1;
  }
  if (auto* b = curr->dynCast<Binary>()) {
    if (b->op == AndInt32) {
      return isOneBitI32(b->left, depth + 1) ||
             isOneBitI32(b->right, depth + 1);
    }
    if (b->op == OrInt32 || b->op == XorInt32) {
      return isOneBitI32(b->left, depth + 1) &&
             isOneBitI32(b->right, depth + 1);
    }
    if (b->op == ShrUInt32) {
      int32_t shift = 0;
      if (isI32Const(b->right, shift) && (shift & 31) == 31) {
        return true;
      }
    }
    return false;
  }
  if (auto* sel = curr->dynCast<Select>()) {
    return isOneBitI32(sel->ifTrue, depth + 1) &&
           isOneBitI32(sel->ifFalse, depth + 1);
  }
  if (auto* iff = curr->dynCast<If>()) {
    return iff->ifFalse && isOneBitI32(iff->ifTrue, depth + 1) &&
           isOneBitI32(iff->ifFalse, depth + 1);
  }
  if (auto* set = curr->dynCast<LocalSet>()) {
    return set->isTee() && isOneBitI32(set->value, depth + 1);
  }
  if (auto* block = curr->dynCast<Block>()) {
    if (!block->name.is() && !block->list.empty()) {
      return isOneBitI32(block->list.back(), depth + 1);
    }
  }
  return false;
}

static bool tryInvertRelationalBinaryOp(BinaryOp op, BinaryOp& out) {
  switch (op) {
    case EqInt32:
      out = NeInt32;
      return true;
    case NeInt32:
      out = EqInt32;
      return true;
    case LtSInt32:
      out = GeSInt32;
      return true;
    case GeSInt32:
      out = LtSInt32;
      return true;
    case LtUInt32:
      out = GeUInt32;
      return true;
    case GeUInt32:
      out = LtUInt32;
      return true;
    case LeSInt32:
      out = GtSInt32;
      return true;
    case GtSInt32:
      out = LeSInt32;
      return true;
    case LeUInt32:
      out = GtUInt32;
      return true;
    case GtUInt32:
      out = LeUInt32;
      return true;
    case EqInt64:
      out = NeInt64;
      return true;
    case NeInt64:
      out = EqInt64;
      return true;
    case LtSInt64:
      out = GeSInt64;
      return true;
    case GeSInt64:
      out = LtSInt64;
      return true;
    case LtUInt64:
      out = GeUInt64;
      return true;
    case GeUInt64:
      out = LtUInt64;
      return true;
    case LeSInt64:
      out = GtSInt64;
      return true;
    case GtSInt64:
      out = LeSInt64;
      return true;
    case LeUInt64:
      out = GtUInt64;
      return true;
    case GtUInt64:
      out = LeUInt64;
      return true;
    case EqFloat32:
      out = NeFloat32;
      return true;
    case NeFloat32:
      out = EqFloat32;
      return true;
    case EqFloat64:
      out = NeFloat64;
      return true;
    case NeFloat64:
      out = EqFloat64;
      return true;
    default:
      return false;
  }
}

// Check whether `i32.eqz(curr)` can be pushed into `curr` with 0 net added
// `i32.eqz` instructions, producing a 1-bit boolean result (0 or 1).
static bool canInvertBoolExprFree(Expression* curr, Index depth = 0) {
  if (!curr || curr->type != Type::i32 || depth > 6) {
    return false;
  }
  int32_t c = 0;
  if (isI32Const(curr, c)) {
    return true;
  }
  if (auto* u = curr->dynCast<Unary>()) {
    if (u->op == EqZInt32) {
      return isOneBitI32(u->value);
    }
    return false;
  }
  if (auto* b = curr->dynCast<Binary>()) {
    BinaryOp inv;
    return tryInvertRelationalBinaryOp(b->op, inv);
  }
  if (auto* block = curr->dynCast<Block>()) {
    if (!block->name.is() && !block->list.empty()) {
      return canInvertBoolExprFree(block->list.back(), depth + 1);
    }
    return false;
  }
  if (auto* iff = curr->dynCast<If>()) {
    return iff->ifFalse && canInvertBoolExprFree(iff->ifTrue, depth + 1) &&
           canInvertBoolExprFree(iff->ifFalse, depth + 1);
  }
  if (auto* sel = curr->dynCast<Select>()) {
    return canInvertBoolExprFree(sel->ifTrue, depth + 1) &&
           canInvertBoolExprFree(sel->ifFalse, depth + 1);
  }
  return false;
}

static void invertBoolExprFree(Expression*& curr, Builder& builder) {
  int32_t c = 0;
  if (isI32Const(curr, c)) {
    curr->cast<Const>()->value = Literal(int32_t(c == 0 ? 1 : 0));
    return;
  }
  if (auto* u = curr->dynCast<Unary>()) {
    assert(u->op == EqZInt32 && isOneBitI32(u->value));
    curr = u->value;
    return;
  }
  if (auto* b = curr->dynCast<Binary>()) {
    BinaryOp inv;
    bool ok = tryInvertRelationalBinaryOp(b->op, inv);
    assert(ok);
    (void)ok;
    b->op = inv;
    return;
  }
  if (auto* block = curr->dynCast<Block>()) {
    assert(!block->name.is() && !block->list.empty());
    invertBoolExprFree(block->list.back(), builder);
    return;
  }
  if (auto* iff = curr->dynCast<If>()) {
    assert(iff->ifFalse);
    invertBoolExprFree(iff->ifTrue, builder);
    invertBoolExprFree(iff->ifFalse, builder);
    return;
  }
  if (auto* sel = curr->dynCast<Select>()) {
    invertBoolExprFree(sel->ifTrue, builder);
    invertBoolExprFree(sel->ifFalse, builder);
    return;
  }
  WASM_UNREACHABLE("unexpected expression in invertBoolExprFree");
}

// Check whether `i32.eqz(curr)` can be pushed into `curr` with <= 0 net added
// `i32.eqz` instructions (either 0 added `i32.eqz` via `canInvertBoolExprFree`,
// or pushing `i32.eqz` through a 2-arm `If (result i32)` where at least one arm
// is `canInvertBoolExprFree`).
static bool canPushEqzIntoIfOrExprFree(Expression* curr, Index depth = 0) {
  if (!curr || curr->type != Type::i32 || depth > 8) {
    return false;
  }
  if (canInvertBoolExprFree(curr, depth)) {
    return true;
  }
  if (auto* block = curr->dynCast<Block>()) {
    if (!block->name.is() && !block->list.empty()) {
      return canPushEqzIntoIfOrExprFree(block->list.back(), depth + 1);
    }
    return false;
  }
  if (auto* iff = curr->dynCast<If>()) {
    if (!iff->ifFalse || iff->condition->type == Type::unreachable ||
        iff->ifTrue->type != Type::i32 || iff->ifFalse->type != Type::i32) {
      return false;
    }
    return canInvertBoolExprFree(iff->ifTrue, depth + 1) ||
           canInvertBoolExprFree(iff->ifFalse, depth + 1);
  }
  return false;
}

static void pushEqzIntoIfOrExprFree(Expression*& curr,
                                    Builder& builder,
                                    Index depth = 0) {
  if (canInvertBoolExprFree(curr, depth)) {
    invertBoolExprFree(curr, builder);
    return;
  }
  if (auto* u = curr->dynCast<Unary>();
      u && u->op == EqZInt32 && isOneBitI32(u->value)) {
    curr = u->value;
    return;
  }
  if (auto* block = curr->dynCast<Block>()) {
    if (!block->name.is() && !block->list.empty() &&
        canPushEqzIntoIfOrExprFree(block->list.back(), depth + 1)) {
      pushEqzIntoIfOrExprFree(block->list.back(), builder, depth + 1);
      block->finalize(Type::i32);
      return;
    }
  }
  if (auto* iff = curr->dynCast<If>()) {
    if (iff->ifFalse && iff->condition->type != Type::unreachable &&
        iff->ifTrue->type == Type::i32 && iff->ifFalse->type == Type::i32 &&
        (canInvertBoolExprFree(iff->ifTrue, depth + 1) ||
         canInvertBoolExprFree(iff->ifFalse, depth + 1))) {
      pushEqzIntoIfOrExprFree(iff->ifTrue, builder, depth + 1);
      pushEqzIntoIfOrExprFree(iff->ifFalse, builder, depth + 1);
      iff->finalize(Type::i32);
      return;
    }
  }
  curr = builder.makeUnary(EqZInt32, curr);
}

// Negate `expr` for use as a boolean condition (where only zero vs non-zero
// matters): strips a leading `i32.eqz`, or under `shrink2` inverts `expr` in
// place if free, otherwise wraps in `i32.eqz`.
static Expression*
makeNegatedBool(Builder& builder, Expression* expr, bool shrink2) {
  if (auto* u = expr->dynCast<Unary>(); u && u->op == EqZInt32) {
    return u->value;
  }
  if (shrink2 && canInvertBoolExprFree(expr)) {
    invertBoolExprFree(expr, builder);
    return expr;
  }
  return builder.makeUnary(EqZInt32, expr);
}

static Expression* makeOneBitCond(Builder& builder, Expression* expr) {
  if (isOneBitI32(expr)) {
    return expr;
  }
  return builder.makeBinary(NeInt32, expr, builder.makeConst(int32_t(0)));
}

static Expression*
makeNegOneBitCond(Builder& builder, Expression* expr, bool shrink2) {
  if (shrink2 && canInvertBoolExprFree(expr)) {
    invertBoolExprFree(expr, builder);
    return expr;
  }
  if (auto* u = expr->dynCast<Unary>();
      u && u->op == EqZInt32 && isOneBitI32(u->value)) {
    return u->value;
  }
  return builder.makeUnary(EqZInt32, expr);
}

// Split an `if (result i32)` arm into its optional side-effecting prefix list
// `prefixOut` (all `Type::none` and free of external branches) and its final
// i32 value expression `valueOut`.
static bool splitBoolIfArm(Expression* arm,
                           const PassOptions& passOptions,
                           Module& wasm,
                           std::vector<Expression*>& prefixOut,
                           Expression*& valueOut) {
  prefixOut.clear();
  valueOut = nullptr;
  if (!arm || arm->type != Type::i32) {
    return false;
  }
  EffectAnalyzer effects(passOptions, wasm, arm);
  if (effects.branchesOut || effects.hasExternalBreakTargets()) {
    return false;
  }
  Expression* curr = arm;
  while (auto* block = curr->dynCast<Block>()) {
    if (block->name.is() && BranchUtils::BranchSeeker::has(block, block->name)) {
      return false;
    }
    if (block->list.empty() || block->list.back()->type != Type::i32) {
      return false;
    }
    for (Index i = 0; i + 1 < block->list.size(); i++) {
      if (block->list[i]->type != Type::none) {
        return false;
      }
      prefixOut.push_back(block->list[i]);
    }
    curr = block->list.back();
  }
  valueOut = curr;
  return true;
}

// Prepend `Type::none` statements `prefix` to `arm` (which may be nullptr).
static Expression* prependToArm(Builder& builder,
                                const std::vector<Expression*>& prefix,
                                Expression* arm) {
  if (prefix.empty()) {
    return arm;
  }
  if (!arm && prefix.size() == 1) {
    return prefix[0];
  }
  auto* block = builder.makeBlock();
  for (auto* item : prefix) {
    block->list.push_back(item);
  }
  if (arm) {
    if (auto* armBlock = arm->dynCast<Block>();
        armBlock && !armBlock->name.is()) {
      for (auto* item : armBlock->list) {
        block->list.push_back(item);
      }
    } else {
      block->list.push_back(arm);
    }
  }
  block->finalize(arm ? arm->type : Type::none);
  return block;
}

struct ShortCircuitStep {
  std::vector<Expression*> prefix;
  Expression* cond = nullptr;
  bool negated = false;
};

// Recursively decompose a boolean condition `expr` (interpreted as `expr != 0`
// when `!negated`, or `expr == 0` when `negated`) into a short-circuit
// conjunction of `ShortCircuitStep`s.
// Note that by De Morgan's law, calling `collectShortCircuitConjuncts` with
// `negated = true` decomposes a short-circuit disjunction `D_1 || D_2 || ...`
// into the conjunction `!D_1 && !D_2 && ...`.
static void collectShortCircuitConjuncts(Expression* expr,
                                         bool negated,
                                         const PassOptions& passOptions,
                                         Module& wasm,
                                         std::vector<ShortCircuitStep>& outSteps,
                                         std::vector<If*>& collapsedIfs,
                                         int& kEqzOut,
                                         std::vector<Expression*>& trailingPrefix) {
  int localEqz = 0;
  while (auto* u = expr->dynCast<Unary>()) {
    if (u->op == EqZInt32) {
      negated = !negated;
      expr = u->value;
      localEqz++;
    } else {
      break;
    }
  }
  if (auto* inner = expr->dynCast<If>();
      inner && inner->ifFalse && inner->type == Type::i32 &&
      inner->condition->type != Type::unreachable) {
    std::vector<Expression*> armPrefix;
    Expression* armVal = nullptr;
    int32_t falseConst = 0;
    int32_t armConst = 0;
    if (isI32Const(inner->ifFalse, falseConst) &&
        splitBoolIfArm(inner->ifTrue, passOptions, wasm, armPrefix, armVal) &&
        ((falseConst == 0 && !negated) || (falseConst != 0 && negated))) {
      bool armIsConst = isI32Const(armVal, armConst);
      bool armIsPassConst =
        armIsConst && !armPrefix.empty() &&
        ((falseConst == 0 && armConst != 0) ||
         (falseConst != 0 && armConst == 0));
      if (!armIsConst || armIsPassConst) {
        kEqzOut += localEqz;
        collapsedIfs.push_back(inner);
        collectShortCircuitConjuncts(inner->condition,
                                     false,
                                     passOptions,
                                     wasm,
                                     outSteps,
                                     collapsedIfs,
                                     kEqzOut,
                                     trailingPrefix);
        trailingPrefix.insert(
          trailingPrefix.end(), armPrefix.begin(), armPrefix.end());
        if (armIsPassConst) {
          kEqzOut += 2;
        } else {
          collectShortCircuitConjuncts(armVal,
                                       negated,
                                       passOptions,
                                       wasm,
                                       outSteps,
                                       collapsedIfs,
                                       kEqzOut,
                                       trailingPrefix);
        }
        return;
      }
    }
    int32_t trueConst = 0;
    if (isI32Const(inner->ifTrue, trueConst) &&
        splitBoolIfArm(inner->ifFalse, passOptions, wasm, armPrefix, armVal) &&
        ((trueConst == 0 && !negated) || (trueConst != 0 && negated))) {
      bool armIsConst = isI32Const(armVal, armConst);
      bool armIsPassConst =
        armIsConst && !armPrefix.empty() &&
        ((trueConst == 0 && armConst != 0) ||
         (trueConst != 0 && armConst == 0));
      if (!armIsConst || armIsPassConst) {
        kEqzOut += localEqz;
        collapsedIfs.push_back(inner);
        collectShortCircuitConjuncts(inner->condition,
                                     true,
                                     passOptions,
                                     wasm,
                                     outSteps,
                                     collapsedIfs,
                                     kEqzOut,
                                     trailingPrefix);
        trailingPrefix.insert(
          trailingPrefix.end(), armPrefix.begin(), armPrefix.end());
        if (armIsPassConst) {
          kEqzOut += 2;
        } else {
          collectShortCircuitConjuncts(armVal,
                                       negated,
                                       passOptions,
                                       wasm,
                                       outSteps,
                                       collapsedIfs,
                                       kEqzOut,
                                       trailingPrefix);
        }
        return;
      }
    }
  }
  kEqzOut += localEqz;
  ShortCircuitStep step{std::move(trailingPrefix), expr, negated};
  trailingPrefix.clear();
  outSteps.push_back(std::move(step));
}

static int estimateMinBinaryBytes(Expression* curr, Index depth = 0) {
  if (!curr || curr->is<Nop>() || depth > 6) {
    return 0;
  }
  if (curr->is<Unreachable>()) {
    return 1;
  }
  if (auto* c = curr->dynCast<Const>()) {
    if (c->type == Type::f64) {
      return 9;
    }
    if (c->type == Type::f32) {
      return 5;
    }
    return 2;
  }
  if (curr->is<RefNull>() || curr->is<LocalGet>() || curr->is<GlobalGet>()) {
    return 2;
  }
  if (auto* br = curr->dynCast<Break>()) {
    return 2 + (br->value ? estimateMinBinaryBytes(br->value, depth + 1) : 0) +
           (br->condition ? estimateMinBinaryBytes(br->condition, depth + 1)
                          : 0);
  }
  if (auto* ret = curr->dynCast<Return>()) {
    return 1 + (ret->value ? estimateMinBinaryBytes(ret->value, depth + 1) : 0);
  }
  if (auto* b = curr->dynCast<Block>()) {
    int sum = b->name.is() ? 3 : 0;
    for (auto* item : b->list) {
      sum += estimateMinBinaryBytes(item, depth + 1);
      if (sum >= 64) {
        return 64;
      }
    }
    return sum;
  }
  int sum = 1;
  for (auto* child : ChildIterator(curr)) {
    sum += estimateMinBinaryBytes(child, depth + 1);
    if (sum >= 64) {
      return 64;
    }
  }
  return sum;
}

// Try to fold a boolean-producing `if (result i32)` condition (optionally
// wrapped in `i32.eqz`) into `curr`.
static bool tryFoldBooleanIfCondition(If* curr,
                                      Function* func,
                                      Module& wasm,
                                      const PassOptions& passOptions,
                                      bool neverUnconditionalize = false) {
  bool negated = false;
  Expression* cond = curr->condition;
  while (auto* u = cond->dynCast<Unary>()) {
    if (u->op == EqZInt32) {
      negated = !negated;
      cond = u->value;
    } else {
      break;
    }
  }
  auto* inner = cond->dynCast<If>();
  if (!inner || !inner->ifFalse || inner->type != Type::i32 ||
      inner->condition->type == Type::unreachable) {
    return false;
  }

  Builder builder(wasm);
  std::vector<Expression*> prefix;
  Expression* val = nullptr;

  int32_t falseConst = 0;
  if (isI32Const(inner->ifFalse, falseConst) &&
      splitBoolIfArm(inner->ifTrue, passOptions, wasm, prefix, val)) {
    int32_t trueConst = 0;
    bool valIsConst = isI32Const(val, trueConst);
    // (a) `(if (if (result i32) c (then A 1) (else 0)) (then T) (else F))`
    //  or `(if (i32.eqz (if (result i32) c (then A 0) (else 1))) (then T) (else F))`
    //  => `(if c (then A T) (else F))`
    if (valIsConst && ((falseConst == 0 && !negated && trueConst != 0) ||
                       (falseConst != 0 && negated && trueConst == 0))) {
      curr->condition = inner->condition;
      curr->ifTrue = prependToArm(builder, prefix, curr->ifTrue);
      BranchHints::copyTo(inner, curr, func);
      curr->finalize();
      return true;
    }
    // (b) `(if (i32.eqz (if (result i32) c (then A 1) (else 0))) (then T) (else F))`
    //  or `(if (if (result i32) c (then A 0) (else 1)) (then T) (else F))`
    //  => `(if c (then A F) (else T))` (or `(if (i32.eqz c) (then T))` if !F && A.empty())
    if (valIsConst && ((falseConst == 0 && negated && trueConst != 0) ||
                       (falseConst != 0 && !negated && trueConst == 0))) {
      if (!curr->ifFalse && prefix.empty()) {
        curr->condition =
          makeNegatedBool(builder, inner->condition, isV4Enabled(passOptions));
        BranchHints::copyFlippedTo(inner, curr, func);
        curr->finalize();
        return true;
      }
      Expression* oldTrue = curr->ifTrue;
      Expression* oldFalse = curr->ifFalse;
      curr->condition = inner->condition;
      curr->ifTrue = prependToArm(builder, prefix, oldFalse);
      if (!curr->ifTrue) {
        curr->ifTrue = builder.makeNop();
      }
      curr->ifFalse = oldTrue;
      BranchHints::copyTo(inner, curr, func);
      curr->finalize();
      return true;
    }
    // (d) Short-circuit `&&`:
    // `(if (if (result i32) c (then A E) (else 0)) (then T) [else F])`
    // `(if (i32.eqz (if (result i32) c (then A E) (else 1))) (then T) [else F])`
    if (!valIsConst &&
        ((falseConst == 0 && !negated) || (falseConst != 0 && negated))) {
      Expression* innerCond =
        negated
          ? makeNegatedBool(builder, val, isV4Enabled(passOptions))
          : val;
      if (isV4Enabled(passOptions) && !neverUnconditionalize &&
          prefix.empty() &&
          !EffectAnalyzer(passOptions, wasm, val).hasSideEffects() &&
          !tooCostlyToRunUnconditionally(passOptions, val) &&
          (isOneBitI32(inner->condition) || isOneBitI32(innerCond))) {
        curr->condition =
          builder.makeBinary(AndInt32,
                             makeOneBitCond(builder, inner->condition),
                             makeOneBitCond(builder, innerCond));
        BranchHints::clear(curr, func);
        curr->finalize();
        return true;
      }
      if (!curr->ifFalse) {
        auto* newInnerIf = builder.makeIf(innerCond, curr->ifTrue);
        while (tryFoldBooleanIfCondition(
          newInnerIf, func, wasm, passOptions, neverUnconditionalize)) {
        }
        curr->condition = inner->condition;
        curr->ifTrue = prependToArm(builder, prefix, newInnerIf);
        BranchHints::copyTo(inner, curr, func);
        curr->finalize();
        return true;
      }
    }
    // (e) Short-circuit `||` (`c ? E : 1` or `!(c ? E : 0)`):
    if (!valIsConst && isV4Enabled(passOptions) && !neverUnconditionalize &&
        prefix.empty() &&
        ((falseConst != 0 && !negated) || (falseConst == 0 && negated)) &&
        !EffectAnalyzer(passOptions, wasm, val).hasSideEffects() &&
        !tooCostlyToRunUnconditionally(passOptions, val)) {
      Expression* innerCond =
        negated ? makeNegatedBool(builder, val, true) : val;
      curr->condition = builder.makeBinary(
        OrInt32, makeNegatedBool(builder, inner->condition, true), innerCond);
      BranchHints::clear(curr, func);
      curr->finalize();
      return true;
    }
  }

  int32_t trueConst = 0;
  if (isI32Const(inner->ifTrue, trueConst) &&
      splitBoolIfArm(inner->ifFalse, passOptions, wasm, prefix, val)) {
    int32_t falseValConst = 0;
    bool valIsConst = isI32Const(val, falseValConst);
    // `(if (if (result i32) c (then 0) (else A 1)) (then T) (else F))`
    if (valIsConst && ((trueConst == 0 && !negated && falseValConst != 0) ||
                       (trueConst != 0 && negated && falseValConst == 0))) {
      if (!curr->ifFalse) {
        curr->condition =
          makeNegatedBool(builder, inner->condition, isV4Enabled(passOptions));
        curr->ifTrue = prependToArm(builder, prefix, curr->ifTrue);
        BranchHints::copyFlippedTo(inner, curr, func);
        curr->finalize();
        return true;
      }
      Expression* oldTrue = curr->ifTrue;
      Expression* oldFalse = curr->ifFalse;
      curr->condition = inner->condition;
      curr->ifTrue = oldFalse;
      curr->ifFalse = prependToArm(builder, prefix, oldTrue);
      BranchHints::copyTo(inner, curr, func);
      curr->finalize();
      return true;
    }
    // `(if (if (result i32) c (then 1) (else A 0)) (then T) (else F))`
    if (valIsConst && ((trueConst != 0 && !negated && falseValConst == 0) ||
                       (trueConst == 0 && negated && falseValConst != 0))) {
      curr->condition = inner->condition;
      if (curr->ifFalse || !prefix.empty()) {
        curr->ifFalse = prependToArm(builder, prefix, curr->ifFalse);
      }
      BranchHints::copyTo(inner, curr, func);
      curr->finalize();
      return true;
    }
    // `(if (if (result i32) c (then 0) (else A E)) (then T) [else F])`
    if (!valIsConst &&
        ((trueConst == 0 && !negated) || (trueConst != 0 && negated))) {
      Expression* innerCond =
        negated
          ? makeNegatedBool(builder, val, isV4Enabled(passOptions))
          : val;
      if (isV4Enabled(passOptions) && !neverUnconditionalize &&
          prefix.empty() &&
          !EffectAnalyzer(passOptions, wasm, val).hasSideEffects() &&
          !tooCostlyToRunUnconditionally(passOptions, val)) {
        curr->condition = builder.makeBinary(
          AndInt32,
          makeNegOneBitCond(builder, inner->condition, true),
          makeOneBitCond(builder, innerCond));
        BranchHints::clear(curr, func);
        curr->finalize();
        return true;
      }
      if (!curr->ifFalse) {
        auto* newInnerIf = builder.makeIf(innerCond, curr->ifTrue);
        while (tryFoldBooleanIfCondition(
          newInnerIf, func, wasm, passOptions, neverUnconditionalize)) {
        }
        curr->condition =
          makeNegatedBool(builder, inner->condition, isV4Enabled(passOptions));
        curr->ifTrue = prependToArm(builder, prefix, newInnerIf);
        BranchHints::copyFlippedTo(inner, curr, func);
        curr->finalize();
        return true;
      }
    }
    // Short-circuit `||` (`c ? 1 : E` or `!(c ? 0 : E)`):
    if (!valIsConst && isV4Enabled(passOptions) && !neverUnconditionalize &&
        prefix.empty() &&
        ((trueConst != 0 && !negated) || (trueConst == 0 && negated)) &&
        !EffectAnalyzer(passOptions, wasm, val).hasSideEffects() &&
        !tooCostlyToRunUnconditionally(passOptions, val)) {
      Expression* innerCond =
        negated ? makeNegatedBool(builder, val, true) : val;
      curr->condition =
        builder.makeBinary(OrInt32, inner->condition, innerCond);
      BranchHints::clear(curr, func);
      curr->finalize();
      return true;
    }
  }

  std::vector<Expression*> prefixTrue;
  std::vector<Expression*> prefixFalse;
  Expression* valTrue = nullptr;
  Expression* valFalse = nullptr;
  int32_t cTrue = 0, cFalse = 0;
  if (splitBoolIfArm(inner->ifTrue, passOptions, wasm, prefixTrue, valTrue) &&
      splitBoolIfArm(inner->ifFalse, passOptions, wasm, prefixFalse, valFalse) &&
      isI32Const(valTrue, cTrue) && isI32Const(valFalse, cFalse) &&
      ((cTrue != 0 && cFalse == 0) || (cTrue == 0 && cFalse != 0))) {
    bool samePol =
      (cTrue != 0 && cFalse == 0 && !negated) ||
      (cTrue == 0 && cFalse != 0 && negated);
    Expression* oldTrue = curr->ifTrue;
    Expression* oldFalse = curr->ifFalse;
    curr->condition = inner->condition;
    if (samePol) {
      curr->ifTrue = prependToArm(builder, prefixTrue, oldTrue);
      if (oldFalse || !prefixFalse.empty()) {
        curr->ifFalse = prependToArm(builder, prefixFalse, oldFalse);
      }
    } else {
      curr->ifTrue = prependToArm(builder, prefixTrue, oldFalse);
      if (!curr->ifTrue) {
        curr->ifTrue = builder.makeNop();
      }
      curr->ifFalse = prependToArm(builder, prefixFalse, oldTrue);
    }
    BranchHints::copyTo(inner, curr, func);
    curr->finalize();
    return true;
  }

  return false;
}

// Try to fold a boolean-producing `if (result i32)` condition in `br_if $L`
// (or `(drop (br_if $L val cond))` when `outerDrop` is non-null).
// Returns the replacement Expression* (either `curr`/`outerDrop` if modified
// in place, or a new `If*`), or nullptr if not folded.
static Expression* tryFoldBooleanBreakCondition(Break* curr,
                                                Function* func,
                                                Module& wasm,
                                                const PassOptions& passOptions,
                                                bool neverUnconditionalize,
                                                Drop* outerDrop = nullptr) {
  if (!curr->condition || curr->type == Type::unreachable) {
    return nullptr;
  }
  if (curr->value) {
    if (!outerDrop) {
      return nullptr;
    }
    if (EffectAnalyzer(passOptions, wasm, curr->value).hasSideEffects() ||
        !EffectAnalyzer::canReorder(
          passOptions, wasm, curr->value, curr->condition)) {
      return nullptr;
    }
  }
  bool negated = false;
  Expression* cond = curr->condition;
  while (auto* u = cond->dynCast<Unary>()) {
    if (u->op == EqZInt32) {
      negated = !negated;
      cond = u->value;
    } else {
      break;
    }
  }
  auto* inner = cond->dynCast<If>();
  if (!inner || !inner->ifFalse || inner->type != Type::i32 ||
      inner->condition->type == Type::unreachable) {
    return nullptr;
  }

  Builder builder(wasm);
  auto finishInPlace = [&]() -> Expression* {
    curr->finalize();
    if (outerDrop) {
      outerDrop->finalize();
    }
    if (auto* rec = tryFoldBooleanBreakCondition(
          curr, func, wasm, passOptions, neverUnconditionalize, outerDrop)) {
      return rec;
    }
    return outerDrop ? static_cast<Expression*>(outerDrop)
                     : static_cast<Expression*>(curr);
  };
  auto finishConditionalArm = [&]() -> Expression* {
    curr->finalize();
    BranchHints::clear(curr, func);
    if (outerDrop) {
      outerDrop->finalize();
    }
    if (auto* rec = tryFoldBooleanBreakCondition(
          curr, func, wasm, passOptions, neverUnconditionalize, outerDrop)) {
      return rec;
    }
    return outerDrop ? static_cast<Expression*>(outerDrop)
                     : static_cast<Expression*>(curr);
  };
  auto finishUnconditionalArm = [&]() -> Expression* {
    curr->condition = nullptr;
    curr->finalize();
    BranchHints::clear(curr, func);
    return curr;
  };
  auto isTrivialValueToCopy = [](Expression* v) {
    return v->is<Const>() || v->is<RefNull>() || v->is<LocalGet>() ||
           v->is<GlobalGet>();
  };

  std::vector<Expression*> prefix;
  Expression* val = nullptr;
  int32_t falseConst = 0;
  if (isI32Const(inner->ifFalse, falseConst) &&
      splitBoolIfArm(inner->ifTrue, passOptions, wasm, prefix, val)) {
    int32_t trueConst = 0;
    bool valIsConst = isI32Const(val, trueConst);
    // (c) `br_if $L (i32.eqz (if (result i32) c (then A 1) (else 0)))`
    //  => `(if c (then A) (else (br $L)))` (or `br_if $L (i32.eqz c)` if A.empty())
    if (valIsConst && ((falseConst == 0 && negated && trueConst != 0) ||
                       (falseConst != 0 && !negated && trueConst == 0))) {
      if (prefix.empty()) {
        curr->condition =
          makeNegatedBool(builder, inner->condition, isV4Enabled(passOptions));
        BranchHints::copyFlippedTo(inner, curr, func);
        return finishInPlace();
      }
      auto* newIf = builder.makeIf(inner->condition,
                                   prependToArm(builder, prefix, nullptr),
                                   finishUnconditionalArm());
      BranchHints::copyTo(inner, newIf, func);
      return newIf;
    }
    // `br_if $L (if (result i32) c (then A 1) (else 0))`
    if (valIsConst && ((falseConst == 0 && !negated && trueConst != 0) ||
                       (falseConst != 0 && negated && trueConst == 0))) {
      if (prefix.empty()) {
        curr->condition = inner->condition;
        BranchHints::copyTo(inner, curr, func);
        return finishInPlace();
      }
      auto* newIf = builder.makeIf(
        inner->condition,
        prependToArm(builder, prefix, finishUnconditionalArm()));
      BranchHints::copyTo(inner, newIf, func);
      return newIf;
    }
    // `br_if $L (if (result i32) c (then A E) (else 0))`
    //  => `(if c (then A (br_if $L E)))`
    if (!valIsConst &&
        ((falseConst == 0 && !negated) || (falseConst != 0 && negated))) {
      Expression* rhs =
        negated
          ? makeNegatedBool(builder, val, isV4Enabled(passOptions))
          : val;
      if (isV4Enabled(passOptions) && prefix.empty() &&
          !neverUnconditionalize &&
          !EffectAnalyzer(passOptions, wasm, val).hasSideEffects() &&
          !tooCostlyToRunUnconditionally(passOptions, val) &&
          (isOneBitI32(inner->condition) || isOneBitI32(rhs))) {
        curr->condition =
          builder.makeBinary(AndInt32,
                             makeOneBitCond(builder, inner->condition),
                             makeOneBitCond(builder, rhs));
        BranchHints::clear(curr, func);
        return finishInPlace();
      }
      curr->condition = rhs;
      auto* newIf = builder.makeIf(
        inner->condition,
        prependToArm(builder, prefix, finishConditionalArm()));
      BranchHints::copyTo(inner, newIf, func);
      return newIf;
    }
    // `br_if $L (if (result i32) c (then A E) (else 1))`
    // or `br_if $L (i32.eqz (if (result i32) c (then A E) (else 0)))`
    if (!valIsConst &&
        ((falseConst != 0 && !negated) || (falseConst == 0 && negated))) {
      Expression* rhs =
        negated
          ? makeNegatedBool(builder, val, isV4Enabled(passOptions))
          : val;
      if (prefix.empty() && !neverUnconditionalize &&
          !EffectAnalyzer(passOptions, wasm, val).hasSideEffects() &&
          !tooCostlyToRunUnconditionally(passOptions, val)) {
        curr->condition = builder.makeBinary(
          OrInt32,
          makeNegatedBool(builder, inner->condition, isV4Enabled(passOptions)),
          rhs);
        BranchHints::clear(curr, func);
        return finishInPlace();
      }
      if (!curr->value && !outerDrop && isV4Enabled(passOptions) &&
          prefix.empty()) {
        Expression* c1 = makeNegatedBool(builder, inner->condition, true);
        auto* br1 = builder.makeBreak(curr->name, nullptr, c1);
        BranchHints::copyFlippedTo(inner, br1, func);
        Expression* rep1 = br1;
        if (auto* r1 = tryFoldBooleanBreakCondition(br1,
                                                    func,
                                                    wasm,
                                                    passOptions,
                                                    neverUnconditionalize,
                                                    nullptr)) {
          rep1 = r1;
        }
        curr->condition = rhs;
        Expression* rep2 = finishConditionalArm();
        return builder.makeBlock({rep1, rep2}, Type::none);
      }
      if (!curr->value ||
          (isTrivialValueToCopy(curr->value) &&
           (!negated || isV4Enabled(passOptions)))) {
        Expression* valCopy =
          curr->value ? ExpressionManipulator::copy(curr->value, wasm)
                      : nullptr;
        curr->condition = rhs;
        auto* uncondBr = builder.makeBreak(curr->name, valCopy);
        auto* newIf = builder.makeIf(
          inner->condition,
          prependToArm(builder, prefix, finishConditionalArm()),
          uncondBr);
        BranchHints::copyTo(inner, newIf, func);
        return newIf;
      }
    }
  }

  int32_t trueConst = 0;
  if (isI32Const(inner->ifTrue, trueConst) &&
      splitBoolIfArm(inner->ifFalse, passOptions, wasm, prefix, val)) {
    int32_t falseValConst = 0;
    bool valIsConst = isI32Const(val, falseValConst);
    if (valIsConst && ((trueConst == 0 && !negated && falseValConst != 0) ||
                       (trueConst != 0 && negated && falseValConst == 0))) {
      if (prefix.empty()) {
        curr->condition =
          makeNegatedBool(builder, inner->condition, isV4Enabled(passOptions));
        BranchHints::copyFlippedTo(inner, curr, func);
        return finishInPlace();
      }
      auto* newIf = builder.makeIf(
        makeNegatedBool(builder, inner->condition, isV4Enabled(passOptions)),
        prependToArm(builder, prefix, finishUnconditionalArm()));
      BranchHints::copyFlippedTo(inner, newIf, func);
      return newIf;
    }
    if (valIsConst && ((trueConst != 0 && !negated && falseValConst == 0) ||
                       (trueConst == 0 && negated && falseValConst != 0))) {
      if (prefix.empty()) {
        curr->condition = inner->condition;
        BranchHints::copyTo(inner, curr, func);
        return finishInPlace();
      }
      auto* newIf = builder.makeIf(inner->condition,
                                   finishUnconditionalArm(),
                                   prependToArm(builder, prefix, nullptr));
      BranchHints::copyTo(inner, newIf, func);
      return newIf;
    }
    if (!valIsConst &&
        ((trueConst == 0 && !negated) || (trueConst != 0 && negated))) {
      Expression* rhs =
        negated
          ? makeNegatedBool(builder, val, isV4Enabled(passOptions))
          : val;
      if (isV4Enabled(passOptions) && prefix.empty() &&
          !neverUnconditionalize &&
          !EffectAnalyzer(passOptions, wasm, val).hasSideEffects() &&
          !tooCostlyToRunUnconditionally(passOptions, val)) {
        curr->condition = builder.makeBinary(
          AndInt32,
          makeNegOneBitCond(builder, inner->condition, true),
          makeOneBitCond(builder, rhs));
        BranchHints::clear(curr, func);
        return finishInPlace();
      }
      curr->condition = rhs;
      auto* newIf = builder.makeIf(
        makeNegatedBool(builder, inner->condition, isV4Enabled(passOptions)),
        prependToArm(builder, prefix, finishConditionalArm()));
      BranchHints::copyFlippedTo(inner, newIf, func);
      return newIf;
    }
    // `br_if $L (if (result i32) c (then 1) (else A E))`
    // or `br_if $L (i32.eqz (if (result i32) c (then 0) (else A E)))`
    if (!valIsConst &&
        ((trueConst != 0 && !negated) || (trueConst == 0 && negated))) {
      Expression* rhs =
        negated
          ? makeNegatedBool(builder, val, isV4Enabled(passOptions))
          : val;
      if (prefix.empty() && !neverUnconditionalize &&
          !EffectAnalyzer(passOptions, wasm, val).hasSideEffects() &&
          !tooCostlyToRunUnconditionally(passOptions, val)) {
        curr->condition = builder.makeBinary(OrInt32, inner->condition, rhs);
        BranchHints::clear(curr, func);
        return finishInPlace();
      }
      if (!curr->value && !outerDrop && isV4Enabled(passOptions) &&
          prefix.empty()) {
        auto* br1 = builder.makeBreak(curr->name, nullptr, inner->condition);
        BranchHints::copyTo(inner, br1, func);
        Expression* rep1 = br1;
        if (auto* r1 = tryFoldBooleanBreakCondition(br1,
                                                    func,
                                                    wasm,
                                                    passOptions,
                                                    neverUnconditionalize,
                                                    nullptr)) {
          rep1 = r1;
        }
        curr->condition = rhs;
        Expression* rep2 = finishConditionalArm();
        return builder.makeBlock({rep1, rep2}, Type::none);
      }
      if (!curr->value || isTrivialValueToCopy(curr->value)) {
        Expression* valCopy =
          curr->value ? ExpressionManipulator::copy(curr->value, wasm)
                      : nullptr;
        curr->condition = rhs;
        auto* uncondBr = builder.makeBreak(curr->name, valCopy);
        auto* newIf = builder.makeIf(
          inner->condition,
          uncondBr,
          prependToArm(builder, prefix, finishConditionalArm()));
        BranchHints::copyTo(inner, newIf, func);
        return newIf;
      }
    }
  }

  std::vector<Expression*> prefixTrue;
  std::vector<Expression*> prefixFalse;
  Expression* valTrue = nullptr;
  Expression* valFalse = nullptr;
  int32_t cTrue = 0, cFalse = 0;
  if (splitBoolIfArm(inner->ifTrue, passOptions, wasm, prefixTrue, valTrue) &&
      splitBoolIfArm(inner->ifFalse, passOptions, wasm, prefixFalse, valFalse) &&
      isI32Const(valTrue, cTrue) && isI32Const(valFalse, cFalse) &&
      ((cTrue != 0 && cFalse == 0) || (cTrue == 0 && cFalse != 0))) {
    bool brOnTrue =
      (cTrue != 0 && cFalse == 0 && !negated) ||
      (cTrue == 0 && cFalse != 0 && negated);
    Expression* uncondArm = finishUnconditionalArm();
    auto* newIf =
      brOnTrue
        ? builder.makeIf(inner->condition,
                         prependToArm(builder, prefixTrue, uncondArm),
                         prependToArm(builder, prefixFalse, nullptr))
        : builder.makeIf(inner->condition,
                         prependToArm(builder, prefixTrue, nullptr),
                         prependToArm(builder, prefixFalse, uncondArm));
    BranchHints::copyTo(inner, newIf, func);
    return newIf;
  }

  return nullptr;
}

static bool stripVoidTrailingUnreachableExpr(Expression* curr,
                                             bool inVoidContext) {
  bool changed = false;
  if (auto* block = curr->dynCast<Block>()) {
    for (Index i = 0; i < block->list.size(); i++) {
      bool isLast = (i + 1 == block->list.size());
      bool childVoid = isLast ? (inVoidContext || block->type == Type::none)
                              : (block->list[i]->type == Type::none);
      if (stripVoidTrailingUnreachableExpr(block->list[i], childVoid)) {
        changed = true;
      }
    }
    if ((inVoidContext || block->type == Type::none) &&
        block->list.size() >= 2 && block->list.back()->is<Unreachable>()) {
      bool allPrevVoid = true;
      for (Index i = 0; i + 1 < block->list.size(); i++) {
        if (block->list[i]->type.isConcrete()) {
          allPrevVoid = false;
          break;
        }
      }
      if (allPrevVoid && block->name.is()) {
        BranchUtils::BranchSeeker seeker(block->name);
        Expression* temp = block;
        seeker.walk(temp);
        for (auto t : seeker.types) {
          if (t != Type::none) {
            allPrevVoid = false;
            break;
          }
        }
      }
      if (allPrevVoid) {
        block->list.pop_back();
        block->finalize();
        changed = true;
      }
    }
    if ((inVoidContext || block->type == Type::none) && block->name.is() &&
        !block->list.empty()) {
      if (auto* tryy = block->list.back()->dynCast<Try>();
          tryy && !tryy->isDelegate()) {
        bool allPrevVoid = true;
        for (Index i = 0; i + 1 < block->list.size(); i++) {
          if (block->list[i]->type.isConcrete()) {
            allPrevVoid = false;
            break;
          }
        }
        if (allPrevVoid) {
          BranchUtils::BranchSeeker seeker(block->name);
          Expression* temp = block;
          seeker.walk(temp);
          for (auto t : seeker.types) {
            if (t != Type::none) {
              allPrevVoid = false;
              break;
            }
          }
        }
        if (allPrevVoid) {
          std::function<bool(Expression*)> stripTrailingBrToBlock =
            [&](Expression* arm) -> bool {
            if (auto* br = arm->dynCast<Break>()) {
              if (!br->condition && !br->value && br->name == block->name) {
                ExpressionManipulator::nop(br);
                return true;
              }
            } else if (auto* innerTry = arm->dynCast<Try>();
                       innerTry && !innerTry->isDelegate()) {
              bool innerChanged = stripTrailingBrToBlock(innerTry->body);
              for (auto* cb : innerTry->catchBodies) {
                if (stripTrailingBrToBlock(cb)) {
                  innerChanged = true;
                }
              }
              if (innerChanged) {
                innerTry->finalize();
                return true;
              }
            } else if (auto* armBlock = arm->dynCast<Block>()) {
              if (!armBlock->list.empty()) {
                bool prevConcrete = false;
                for (Index j = 0; j + 1 < armBlock->list.size(); j++) {
                  if (armBlock->list[j]->type.isConcrete()) {
                    prevConcrete = false;
                    prevConcrete = true;
                    break;
                  }
                }
                if (!prevConcrete) {
                  if (auto* br = armBlock->list.back()->dynCast<Break>()) {
                    if (!br->condition && !br->value &&
                        br->name == block->name) {
                      if (armBlock->list.size() == 1) {
                        ExpressionManipulator::nop(br);
                      } else {
                        armBlock->list.pop_back();
                      }
                      armBlock->finalize();
                      return true;
                    }
                  } else if (armBlock->list.size() >= 2 &&
                             armBlock->list.back()->is<Unreachable>()) {
                    if (auto* innerTry =
                          armBlock->list[armBlock->list.size() - 2]
                            ->dynCast<Try>();
                        innerTry && !innerTry->isDelegate()) {
                      bool prev2Concrete = false;
                      for (Index j = 0; j + 2 < armBlock->list.size(); j++) {
                        if (armBlock->list[j]->type.isConcrete()) {
                          prev2Concrete = true;
                          break;
                        }
                      }
                      if (!prev2Concrete && stripTrailingBrToBlock(innerTry)) {
                        if (innerTry->type == Type::none) {
                          armBlock->list.pop_back();
                        }
                        armBlock->finalize();
                        return true;
                      }
                    }
                  } else if (auto* innerTry =
                               armBlock->list.back()->dynCast<Try>();
                             innerTry && !innerTry->isDelegate()) {
                    if (stripTrailingBrToBlock(innerTry)) {
                      armBlock->finalize();
                      return true;
                    }
                  }
                }
              }
            }
            return false;
          };
          bool tryChanged = stripTrailingBrToBlock(tryy->body);
          for (auto* catchBody : tryy->catchBodies) {
            if (stripTrailingBrToBlock(catchBody)) {
              tryChanged = true;
            }
          }
          if (tryChanged) {
            tryy->finalize();
            if (!BranchUtils::BranchSeeker::has(block, block->name)) {
              block->name = Name();
            }
            block->finalize();
            changed = true;
          }
        }
      }
    }
    if (changed && inVoidContext && block->type == Type::unreachable &&
        !block->list.empty() && block->list.back()->type == Type::none) {
      bool canBeVoid = true;
      for (auto* item : block->list) {
        if (item->type.isConcrete()) {
          canBeVoid = false;
          break;
        }
      }
      if (canBeVoid && block->name.is()) {
        BranchUtils::BranchSeeker seeker(block->name);
        Expression* temp = block;
        seeker.walk(temp);
        for (auto t : seeker.types) {
          if (t != Type::none) {
            canBeVoid = false;
            break;
          }
        }
      }
      if (canBeVoid) {
        block->finalize();
      }
    }
    return changed;
  }
  if (auto* iff = curr->dynCast<If>()) {
    if (stripVoidTrailingUnreachableExpr(iff->condition, false)) {
      changed = true;
    }
    bool armVoid = iff->condition->type != Type::unreachable &&
                   (iff->type == Type::none ||
                    (inVoidContext && iff->type == Type::unreachable));
    if (armVoid) {
      if (stripVoidTrailingUnreachableExpr(iff->ifTrue, true)) {
        changed = true;
      }
      if (iff->ifFalse && stripVoidTrailingUnreachableExpr(iff->ifFalse, true)) {
        changed = true;
      }
      if (changed) {
        iff->finalize();
      }
    } else {
      if ( stripVoidTrailingUnreachableExpr(
             iff->ifTrue, iff->ifTrue->type == Type::none)) {
        changed = true;
      }
      if (iff->ifFalse &&
          stripVoidTrailingUnreachableExpr(
            iff->ifFalse, iff->ifFalse->type == Type::none)) {
        changed = true;
      }
    }
    return changed;
  }
  if (auto* loop = curr->dynCast<Loop>()) {
    bool bodyVoid = loop->type == Type::none ||
                    (inVoidContext && loop->type == Type::unreachable);
    if (stripVoidTrailingUnreachableExpr(
          loop->body, bodyVoid || loop->body->type == Type::none)) {
      loop->finalize();
      changed = true;
    }
    return changed;
  }
  if (auto* tryy = curr->dynCast<Try>()) {
    bool armVoid = tryy->type == Type::none ||
                   (inVoidContext && tryy->type == Type::unreachable);
    if (stripVoidTrailingUnreachableExpr(
          tryy->body, armVoid || tryy->body->type == Type::none)) {
      changed = true;
    }
    for (auto* catchBody : tryy->catchBodies) {
      if (stripVoidTrailingUnreachableExpr(
            catchBody, armVoid || catchBody->type == Type::none)) {
        changed = true;
      }
    }
    if (changed && armVoid) {
      tryy->finalize();
    }
    return changed;
  }
  for (auto* child : ChildIterator(curr)) {
    if (stripVoidTrailingUnreachableExpr(child, child->type == Type::none)) {
      changed = true;
    }
  }
  return changed;
}

struct RemoveUnusedBrs : public WalkerPass<PostWalker<RemoveUnusedBrs>> {
  bool isFunctionParallel() override { return true; }

  std::unique_ptr<Pass> create() override {
    auto pass = std::make_unique<RemoveUnusedBrs>();
    pass->name = name;
    if (passArg) {
      pass->setPassArg(*passArg);
    }
    return pass;
  }

  bool anotherCycle;

  // Whether we are allowed to unconditionalize code, that is, make code run
  // that previously might not have. Unconditionalizing code is a problem for
  // fuzzing branch hints: a branch hint that never ran might be wrong, and if
  // we start to run it, the fuzzer would report a finding.
  bool neverUnconditionalize;

  using Flows = std::vector<Expression**>;

  // list of breaks that are currently flowing. if they reach their target
  // without interference, they can be removed (or their value forwarded TODO)
  Flows flows;

  // a stack for if-else contents, we merge their outputs
  std::vector<Flows> ifStack;

  // list of all loops, so we can optimize them
  std::vector<Loop*> loops;

  static void visitAny(RemoveUnusedBrs* self, Expression** currp) {
    auto* curr = *currp;
    auto& flows = self->flows;

    if (curr->is<Break>()) {
      flows.clear();
      auto* br = curr->cast<Break>();
      if (!br->condition) { // TODO: optimize?
        // a break, let's see where it flows to
        flows.push_back(currp);
      } else {
        self->stopValueFlow();
      }
    } else if (curr->is<Return>()) {
      flows.clear();
      flows.push_back(currp);
    } else if (curr->is<If>()) {
      auto* iff = curr->cast<If>();
      if (iff->condition->type == Type::unreachable) {
        // avoid trying to optimize this, we never reach it anyhow
        self->stopFlow();
        return;
      }
      if (iff->ifFalse) {
        assert(self->ifStack.size() > 0);
        auto ifTrueFlows = std::move(self->ifStack.back());
        self->ifStack.pop_back();
        // we can flow values out in most cases, except if one arm
        // has the none type - we will update the types later, but
        // there is no way to emit a proper type for one arm being
        // none and the other flowing a value; and there is no way
        // to flow a value from a none.
        if (iff->ifTrue->type == Type::none ||
            iff->ifFalse->type == Type::none) {
          self->removeValueFlow(ifTrueFlows);
          self->stopValueFlow();
        }
        for (auto* flow : ifTrueFlows) {
          flows.push_back(flow);
        }
      } else {
        // if without else stops the flow of values
        self->stopValueFlow();
      }
    } else if (auto* block = curr->dynCast<Block>()) {
      // any breaks flowing to here are unnecessary, as we get here anyhow
      auto name = block->name;
      auto& list = block->list;
      if (name.is()) {
        Index size = flows.size();
        Index skip = 0;
        for (Index i = 0; i < size; i++) {
          auto* flow = (*flows[i])->dynCast<Break>();
          if (flow && flow->name == name) {
            if (!flow->value) {
              // br => nop
              ExpressionManipulator::nop<Break>(flow);
            } else {
              // br with value => value
              *flows[i] = flow->value;
            }
            skip++;
            self->anotherCycle = true;
          } else if (skip > 0) {
            flows[i - skip] = flows[i];
          }
        }
        if (skip > 0) {
          flows.resize(size - skip);
        }
      }
      // Drop a nop at the end of a block, which prevents a value flowing. Note
      // that this is worth doing regardless of whether we have a name on this
      // block or not (which the if right above us checks) - such a nop is
      // always unneeded and can limit later optimizations.
      while (list.size() > 0 && list.back()->is<Nop>()) {
        list.resize(list.size() - 1);
        self->anotherCycle = true;
      }
      // A value flowing is only valid if it is a value that the block actually
      // flows out. If it is never reached, it does not flow out, and may be
      // invalid to represent as such.
      auto size = list.size();
      for (Index i = 0; i < size; i++) {
        if (i != size - 1 && list[i]->type == Type::unreachable) {
          // No value flows out of this block.
          self->stopValueFlow();
          break;
        }
      }
    } else if (curr->is<Nop>()) {
      // Ignore (could be result of a previous cycle).
      self->stopValueFlow();
    } else if (curr->is<Loop>() || curr->is<TryTable>()) {
      // Do nothing - it's ok for values to flow out.
      // TODO: Legacy Try as well?
    } else if (auto* sw = curr->dynCast<Switch>()) {
      self->stopFlow();
      self->optimizeSwitch(sw);
    } else {
      // Anything else stops the flow.
      self->stopFlow();
    }
  }

  void stopFlow() { flows.clear(); }

  void removeValueFlow(Flows& currFlows) {
    currFlows.erase(std::remove_if(currFlows.begin(),
                                   currFlows.end(),
                                   [&](Expression** currp) {
                                     auto* curr = *currp;
                                     if (auto* ret = curr->dynCast<Return>()) {
                                       return ret->value;
                                     }
                                     return curr->cast<Break>()->value;
                                   }),
                    currFlows.end());
  }

  void stopValueFlow() { removeValueFlow(flows); }

  static void clear(RemoveUnusedBrs* self, Expression** currp) {
    self->flows.clear();
  }

  static void saveIfTrue(RemoveUnusedBrs* self, Expression** currp) {
    self->ifStack.push_back(std::move(self->flows));
  }

  void visitLoop(Loop* curr) { loops.push_back(curr); }

  void optimizeSwitch(Switch* curr) {
    // if the final element is the default, we don't need it
    while (!curr->targets.empty() && curr->targets.back() == curr->default_) {
      curr->targets.pop_back();
    }
    // if the first element is the default, we can remove it by shifting
    // everything (which does add a subtraction of a constant, but often that is
    // worth it as the constant can be folded away and/or we remove multiple
    // elements here)
    Index removable = 0;
    while (removable < curr->targets.size() &&
           curr->targets[removable] == curr->default_) {
      removable++;
    }
    if (removable > 0) {
      for (Index i = removable; i < curr->targets.size(); i++) {
        curr->targets[i - removable] = curr->targets[i];
      }
      curr->targets.resize(curr->targets.size() - removable);
      Builder builder(*getModule());
      curr->condition = builder.makeBinary(
        SubInt32, curr->condition, builder.makeConst(int32_t(removable)));
    }
    // when there isn't a value, we can do some trivial optimizations without
    // worrying about the value being executed before the condition
    if (curr->value) {
      return;
    }
    if (curr->targets.size() == 0) {
      // a switch with just a default always goes there
      Builder builder(*getModule());
      replaceCurrent(builder.makeSequence(builder.makeDrop(curr->condition),
                                          builder.makeBreak(curr->default_)));
    } else if (curr->targets.size() == 1) {
      // a switch with two options is basically an if
      Builder builder(*getModule());
      replaceCurrent(builder.makeIf(curr->condition,
                                    builder.makeBreak(curr->default_),
                                    builder.makeBreak(curr->targets.front())));
    } else {
      if (isV4Enabled(getPassOptions())) {
        Name firstTarget = curr->targets.front();
        bool allSameTarget = true;
        for (Index i = 1; i < curr->targets.size(); i++) {
          if (curr->targets[i] != firstTarget) {
            allSameTarget = false;
            break;
          }
        }
        if (allSameTarget) {
          Builder builder(*getModule());
          replaceCurrent(builder.makeIf(
            builder.makeBinary(
              LtUInt32,
              curr->condition,
              builder.makeConst(int32_t(curr->targets.size()))),
            builder.makeBreak(firstTarget),
            builder.makeBreak(curr->default_)));
          anotherCycle = true;
          return;
        }
      }
      // there are also some other cases where we want to convert a switch into
      // ifs, especially if the switch is large and we are focusing on size. an
      // especially egregious case is a switch like this: [a b b [..] b b c]
      // with default b (which may be arrived at after we trim away excess
      // default values on both sides). in this case, we really have 3 values in
      // a simple form, so it is the next logical case after handling 1 and 2
      // values right above here. to optimize this, we must add a local + a
      // bunch of nodes (if*2, tee, eq, get, const, break*3), so the table must
      // be big enough for it to make sense

      // How many targets we need when shrinking. This is literally the size at
      // which the transformation begins to be smaller.
      const uint32_t MIN_SHRINK = 13;
      // How many targets we need when not shrinking, in which case, 2 ifs may
      // be slower, so we do this when the table is ridiculously large for one
      // with just 3 values in it.
      const uint32_t MIN_GENERAL = 128;

      auto shrink = getPassRunner()->options.shrinkLevel > 0;
      if ((curr->targets.size() >= MIN_SHRINK && shrink) ||
          (curr->targets.size() >= MIN_GENERAL && !shrink)) {
        for (Index i = 1; i < curr->targets.size() - 1; i++) {
          if (curr->targets[i] != curr->default_) {
            return;
          }
        }
        // great, we are in that case, optimize
        Builder builder(*getModule());
        auto temp = builder.addVar(getFunction(), Type::i32);
        Expression* z;
        replaceCurrent(z = builder.makeIf(
                         builder.makeLocalTee(temp, curr->condition, Type::i32),
                         builder.makeIf(builder.makeBinary(
                                          EqInt32,
                                          builder.makeLocalGet(temp, Type::i32),
                                          builder.makeConst(
                                            int32_t(curr->targets.size() - 1))),
                                        builder.makeBreak(curr->targets.back()),
                                        builder.makeBreak(curr->default_)),
                         builder.makeBreak(curr->targets.front())));
      }
    }
  }

  void visitIf(If* curr) {
    if (!curr->ifFalse) {
      bool v4 = isV4Enabled(getPassOptions());
      if (v4) {
        while (auto* b = curr->ifTrue->dynCast<Block>()) {
          if (!b->name.is() && b->list.size() == 1) {
            curr->ifTrue = b->list[0];
          } else {
            break;
          }
        }
      }
      // if without an else. try to reduce
      //    if (condition) br  =>  br_if (condition)
      Drop* innerDrop = v4 ? curr->ifTrue->dynCast<Drop>() : nullptr;
      Break* br = innerDrop ? innerDrop->value->dynCast<Break>()
                            : curr->ifTrue->dynCast<Break>();
      if (innerDrop && (!br || !br->value || br->type == Type::unreachable)) {
        innerDrop = nullptr;
        br = nullptr;
      }
      if (br) {
        if (canTurnIfIntoBrIf(
              curr->condition, br->value, getPassOptions(), *getModule())) {
          if (!br->condition) {
            br->condition = curr->condition;
            BranchHints::copyTo(curr, br, getFunction());
          } else {
            // In this case we can replace
            //   if (condition1) br_if (condition2)
            // =>
            //   br_if select (condition1) (condition2) (i32.const 0)
            // In other words, we replace an if (3 bytes) with a select and a
            // zero (also 3 bytes). The size is unchanged, but the select may
            // be further optimizable, and if select does not branch we also
            // avoid one branch.
            if (neverUnconditionalize) {
              // Creating a select, below, would unconditionally run the
              // select's condition.
              return;
            }
            // Multivalue selects are not supported.
            if (br->value && br->value->type.isTuple()) {
              return;
            }
            // If running the br's condition unconditionally is too expensive,
            // give up.
            auto* zero = LiteralUtils::makeZero(Type::i32, *getModule());
            if (tooCostlyToRunUnconditionally(
                  getPassOptions(), br->condition, zero)) {
              return;
            }
            // Of course we can't do this if the br's condition has side
            // effects, as we would then execute those unconditionally.
            if (EffectAnalyzer(getPassOptions(), *getModule(), br->condition)
                  .hasSideEffects()) {
              return;
            }
            Builder builder(*getModule());
            if (v4 && curr->condition->type != Type::unreachable &&
                br->condition->type != Type::unreachable &&
                (isOneBitI32(curr->condition) || isOneBitI32(br->condition))) {
              br->condition =
                builder.makeBinary(AndInt32,
                                   makeOneBitCond(builder, curr->condition),
                                   makeOneBitCond(builder, br->condition));
            } else {
              // Note that we use the br's condition as the select condition.
              // That keeps the order of the two conditions as it was
              // originally.
              br->condition =
                builder.makeSelect(br->condition, curr->condition, zero);
            }
            BranchHints::applyAndTo(curr, br, br, getFunction());
          }
          br->finalize();
          if (innerDrop) {
            innerDrop->finalize();
            replaceCurrent(innerDrop);
          } else {
            replaceCurrent(Builder(*getModule()).dropIfConcretelyTyped(br));
          }
          anotherCycle = true;
        }
      }

      // if (condition-A) { if (condition-B) .. }
      //   =>
      // if (condition-A ? condition-B : 0) { .. }
      //
      // This replaces an if, which is 3 bytes, with a select plus a zero, which
      // is also 3 bytes. The benefit is that the select may be faster, and also
      // further optimizations may be possible on the select.
      if (auto* child = curr->ifTrue->dynCast<If>()) {
        if (child->ifFalse) {
          return;
        }
        if (neverUnconditionalize) {
          // Creating a select, below, would unconditionally run the inner if's
          // condition (condition-B, in the comment above).
          return;
        }
        // If running the child's condition unconditionally is too expensive,
        // give up.
        if (tooCostlyToRunUnconditionally(getPassOptions(), child->condition)) {
          return;
        }
        // Of course we can't do this if the inner if's condition has side
        // effects, as we would then execute those unconditionally.
        if (EffectAnalyzer(getPassOptions(), *getModule(), child->condition)
              .hasSideEffects()) {
          return;
        }
        Builder builder(*getModule());
        if (v4 && curr->condition->type != Type::unreachable &&
            child->condition->type != Type::unreachable &&
            (isOneBitI32(curr->condition) || isOneBitI32(child->condition))) {
          curr->condition =
            builder.makeBinary(AndInt32,
                               makeOneBitCond(builder, curr->condition),
                               makeOneBitCond(builder, child->condition));
        } else {
          curr->condition = builder.makeSelect(
            child->condition, curr->condition, builder.makeConst(int32_t(0)));
        }
        BranchHints::applyAndTo(curr, child, curr, getFunction());
        curr->ifTrue = child->ifTrue;
      }
    }
    // TODO: if-else can be turned into a br_if as well, if one of the sides is
    //       a dead end we handle the case of a returned value to a local.set
    //       later down, see visitLocalSet.
  }

  // A stack of catching expressions that are parents of the current expression,
  // that is, Try and TryTable.
  std::vector<Expression*> catchers;

  static void popCatcher(RemoveUnusedBrs* self, Expression** currp) {
    assert(!self->catchers.empty() && self->catchers.back() == *currp);
    self->catchers.pop_back();
  }

  void visitThrow(Throw* curr) {
    // If a throw will definitely be caught, and it is not a catch with a
    // reference, then it is just a branch (i.e. the code is using exceptions as
    // control flow). Turn it into a branch here so that the rest of the pass
    // can optimize it with all other branches.
    //
    // To do so, look at the closest try and see if it will catch us, and
    // proceed outwards if not.
    auto thrownTag = curr->tag;
    for (int i = catchers.size() - 1; i >= 0; i--) {
      auto* tryy = catchers[i]->dynCast<TryTable>();
      if (!tryy) {
        // We do not handle mixtures of Try and TryTable.
        return;
      }
      for (Index j = 0; j < tryy->catchTags.size(); j++) {
        auto tag = tryy->catchTags[j];
        // The tag must match, or be a catch_all.
        if (tag == thrownTag || tag.isNull()) {
          // This must not be a catch with exnref.
          if (!tryy->catchRefs[j]) {
            // Success! Create a break to replace the throw.
            auto dest = tryy->catchDests[j];
            auto& wasm = *getModule();
            Builder builder(wasm);
            if (!tag.isNull()) {
              // We are catching a specific tag, so values might be sent.
              Expression* value = nullptr;
              if (curr->operands.size() == 1) {
                value = curr->operands[0];
              } else if (curr->operands.size() > 1) {
                value = builder.makeTupleMake(curr->operands);
              }
              auto* br = builder.makeBreak(dest, value);
              replaceCurrent(br);
              return;
            }

            // catch_all: no values are sent. Drop the throw's children (while
            // ignoring parent effects: the parent is a throw, but we have
            // proven we can remove that effect).
            auto* br = builder.makeBreak(dest);
            auto* rep = getDroppedChildrenAndAppend(
              curr, wasm, getPassOptions(), br, DropMode::IgnoreParentEffects);
            replaceCurrent(rep);
            // We modified the code here and may have added a drop, etc., so
            // stop the flow (rather than re-scan it somehow). We leave
            // optimizing anything that flows out for later iterations.
            stopFlow();
          }

          // Return even if we did not optimize: we found our tag was caught.
          return;
        }
      }
    }
  }

  // override scan to add a pre and a post check task to all nodes
  static void scan(RemoveUnusedBrs* self, Expression** currp) {
    self->pushTask(visitAny, currp);

    if (auto* iff = (*currp)->dynCast<If>()) {
      if (iff->condition->type == Type::unreachable) {
        // avoid trying to optimize this, we never reach it anyhow
        return;
      }
      self->pushTask(doVisitIf, currp);
      if (iff->ifFalse) {
        // we need to join up if-else control flow, and clear after the
        // condition
        self->pushTask(scan, &iff->ifFalse);
        // safe the ifTrue flow, we'll join it later
        self->pushTask(saveIfTrue, currp);
      }
      self->pushTask(scan, &iff->ifTrue);
      self->pushTask(clear, currp); // clear all flow after the condition
      self->pushTask(scan, &iff->condition);
      return;
    }
    if ((*currp)->is<TryTable>() || (*currp)->is<Try>()) {
      // Push the try we are reaching, and add a task to pop it, after all the
      // tasks that Super::scan will push for its children.
      self->catchers.push_back(*currp);
      self->pushTask(popCatcher, currp);
    }

    Super::scan(self, currp);
  }

  // optimizes a loop. returns true if we made changes
  bool optimizeLoop(Loop* loop) {
    // if a loop ends in
    // (loop $in
    //   (block $out
    //     if (..) br $in; else br $out;
    //   )
    // )
    // then our normal opts can remove the break out since it flows directly out
    // (and later passes make the if one-armed). however, the simple analysis
    // fails on patterns like
    //     if (..) br $out;
    //     br $in;
    // which is a common way to do a while (1) loop (end it with a jump to the
    // top), so we handle that here. Specifically we want to conditionalize
    // breaks to the loop top, i.e., put them behind a condition, so that other
    // code can flow directly out and thus brs out can be removed. (even if
    // the change is to let a break somewhere else flow out, that can still be
    // helpful, as it shortens the logical loop. it is also good to generate
    // an if-else instead of an if, as it might allow an eqz to be removed
    // by flipping arms)
    if (!loop->name.is()) {
      return false;
    }
    auto* block = loop->body->dynCast<Block>();
    if (!block) {
      return false;
    }
    // does the last element break to the top of the loop?
    auto& list = block->list;
    if (list.size() <= 1) {
      return false;
    }
    auto* last = list.back()->dynCast<Break>();
    if (!last || !ExpressionAnalyzer::isSimple(last) ||
        last->name != loop->name) {
      return false;
    }
    // last is a simple break to the top of the loop. if we can conditionalize
    // it, it won't block things from flowing out and not needing breaks to do
    // so.
    Index i = list.size() - 2;
    Builder builder(*getModule());
    while (1) {
      auto* curr = list[i];
      if (auto* iff = curr->dynCast<If>()) {
        // let's try to move the code going to the top of the loop into the
        // if-else
        if (!iff->ifFalse) {
          // we need the ifTrue to break, so it cannot reach the code we want to
          // move
          if (iff->ifTrue->type == Type::unreachable) {
            iff->ifFalse = stealSlice(builder, block, i + 1, list.size());
            iff->finalize();
            block->finalize();
            return true;
          }
        } else if (iff->condition->type != Type::unreachable) {
          // This is already an if-else. If one side is a dead end, we can
          // append to the other, if there is no returned value to concern us.
          // Note that we skip ifs with unreachable conditions, as they are dead
          // code that DCE can remove, and modifying them can lead to errors
          // (one of the arms may still be concrete, in which case appending to
          // it would be invalid).

          // can't be, since in the middle of a block
          assert(!iff->type.isConcrete());

          // ensures the first node is a block, if it isn't already, and merges
          // in the second, either as a single element or, if a block, by
          // appending to the first block. this keeps the order of operations in
          // place, that is, the appended element will be executed after the
          // first node's elements
          auto blockifyMerge = [&](Expression* any,
                                   Expression* append) -> Block* {
            Block* block = nullptr;
            if (any) {
              block = any->dynCast<Block>();
            }
            // if the first isn't a block, or it's a block with a name (so we
            // might branch to the end, and so can't append to it, we might skip
            // that code!) then make a new block
            if (!block || block->name.is()) {
              block = builder.makeBlock(any);
            } else {
              assert(!block->type.isConcrete());
            }
            auto* other = append->dynCast<Block>();
            if (!other) {
              block->list.push_back(append);
            } else {
              for (auto* item : other->list) {
                block->list.push_back(item);
              }
            }
            block->finalize();
            return block;
          };

          if (iff->ifTrue->type == Type::unreachable) {
            iff->ifFalse = blockifyMerge(
              iff->ifFalse, stealSlice(builder, block, i + 1, list.size()));
            iff->finalize();
            block->finalize();
            return true;
          } else if (iff->ifFalse->type == Type::unreachable) {
            iff->ifTrue = blockifyMerge(
              iff->ifTrue, stealSlice(builder, block, i + 1, list.size()));
            iff->finalize();
            block->finalize();
            return true;
          }
        }
        return false;
      } else if (auto* brIf = curr->dynCast<Break>()) {
        // br_if is similar to if.
        if (brIf->condition && !brIf->value && brIf->name != loop->name) {
          if (i == list.size() - 2) {
            // there is the br_if, and then the br to the top, so just flip them
            // and the condition
            brIf->condition = builder.makeUnary(EqZInt32, brIf->condition);
            last->name = brIf->name;
            brIf->name = loop->name;
            BranchHints::flip(brIf, getFunction());
            return true;
          } else {
            // there are elements in the middle,
            //   br_if $somewhere (condition)
            //   (..more..)
            //   br $in
            // we can convert the br_if to an if. this has a cost, though,
            // so only do it if it looks useful, which it definitely is if
            //  (a) $somewhere is straight out (so the br out vanishes), and
            //  (b) this br_if is the only branch to that block (so the block
            //      will vanish)
            if (brIf->name == block->name &&
                BranchUtils::BranchSeeker::count(block, block->name) == 1) {
              // note that we could drop the last element here, it is a br we
              // know for sure is removable, but telling stealSlice to steal all
              // to the end is more efficient, it can just truncate.
              list[i] =
                builder.makeIf(brIf->condition,
                               builder.makeBreak(brIf->name),
                               stealSlice(builder, block, i + 1, list.size()));
              BranchHints::copyTo(brIf, list[i], getFunction());
              block->finalize();
              return true;
            }
          }
        }
        return false;
      }
      // if there is control flow, we must stop looking
      if (EffectAnalyzer(getPassOptions(), *getModule(), curr)
            .transfersControlFlow()) {
        return false;
      }
      if (i == 0) {
        return false;
      }
      i--;
    }
  }

  bool sinkBlocks(Function* func) {
    struct Sinker : public PostWalker<Sinker> {
      bool worked = false;

      void visitBlock(Block* curr) {
        // If the block has a single child which is a loop, and the block is
        // named, then it is the exit for the loop. It's better to move it into
        // the loop, where it can be better optimized by other passes. Similar
        // logic for ifs: if the block is an exit for the if, we can move the
        // block in, consider for example:
        //    (block $label
        //     (if (..condition1..)
        //      (block
        //       (br_if $label (..condition2..))
        //       (..code..)
        //      )
        //     )
        //    )
        // After also merging the blocks, we have
        //    (if (..condition1..)
        //     (block $label
        //      (br_if $label (..condition2..))
        //      (..code..)
        //     )
        //    )
        // which can be further optimized later.
        if (curr->name.is() && curr->list.size() == 1) {
          if (auto* loop = curr->list[0]->dynCast<Loop>()) {
            curr->list[0] = loop->body;
            loop->body = curr;
            curr->finalize(curr->type);
            loop->finalize();
            replaceCurrent(loop);
            worked = true;
          } else if (auto* iff = curr->list[0]->dynCast<If>()) {
            if (iff->condition->type == Type::unreachable) {
              // The block result type may not be compatible with the arm result
              // types since the unreachable If can satisfy any type of block.
              // Just leave this for DCE.
              return;
            }
            // The label can't be used in the condition.
            if (BranchUtils::BranchSeeker::count(iff->condition, curr->name) ==
                0) {
              // We can move the block into either arm, if there are no uses in
              // the other.
              Expression** target = nullptr;
              if (!iff->ifFalse || BranchUtils::BranchSeeker::count(
                                     iff->ifFalse, curr->name) == 0) {
                target = &iff->ifTrue;
              } else if (BranchUtils::BranchSeeker::count(iff->ifTrue,
                                                          curr->name) == 0) {
                target = &iff->ifFalse;
              }
              if (target) {
                curr->list[0] = *target;
                *target = curr;
                // The block used to contain the if, and may have changed type
                // from unreachable to none, for example, if the if has an
                // unreachable condition but the arm is not unreachable.
                curr->finalize();
                iff->finalize();
                replaceCurrent(iff);
                worked = true;
                // Note that the type might change, e.g. if the if condition is
                // unreachable but the block that was on the outside had a
                // break.
              }
            }
          }
        }
      }
    } sinker;

    sinker.doWalkFunction(func);
    if (sinker.worked) {
      ReFinalize().walkFunctionInModule(func, getModule());
      return true;
    }
    return false;
  }

  // GC-specific optimizations. These are split out from the main code to keep
  // things as simple as possible.
  bool optimizeGC(Function* func) {
    if (!getModule()->features.hasGC()) {
      return false;
    }

    struct Optimizer : public PostWalker<Optimizer> {
      PassOptions& passOptions;
      bool worked = false;

      Optimizer(PassOptions& passOptions) : passOptions(passOptions) {}

      void visitBrOn(BrOn* curr) {
        // Ignore unreachable BrOns which we cannot improve anyhow. Note that
        // we must check the ref field manually, as we may be changing types as
        // we go here. (Another option would be to use a TypeUpdater here
        // instead of calling ReFinalize at the very end, but that would be more
        // complex and slower.)
        if (curr->type == Type::unreachable ||
            curr->ref->type == Type::unreachable) {
          return;
        }

        Builder builder(*getModule());

        Type refType =
          Properties::getFallthroughType(curr->ref, passOptions, *getModule());
        if (refType == Type::unreachable) {
          // Leave this to DCE.
          return;
        }
        assert(refType.isRef());

        // When we optimize based on all the fallthrough type information
        // available, we may need to insert a cast to maintain validity. For
        // example, in this case we know the cast will succeed, but it would be
        // invalid to send curr->ref directly:
        //
        //   (br_on_cast $l anyref i31ref
        //     (block (result anyref)
        //       (ref.i31 ...)))
        //
        // We could just always do the cast and leave removing the casts to
        // OptimizeInstructions, but it's simple enough to avoid unnecessary
        // casting here.
        auto maybeCast = [&](Expression* expr, Type type) -> Expression* {
          assert(expr->type.isRef() && type.isRef());
          if (Type::isSubType(expr->type, type)) {
            return expr;
          }
          if (type.isNonNullable() && expr->type.isNullable() &&
              Type::isSubType(expr->type.with(NonNullable), type)) {
            return builder.makeRefAs(RefAsNonNull, expr);
          }
          return builder.makeRefCast(expr, type);
        };

        // When we optimize out a cast, we still need the ref value to either
        // send on the optimized branch or return from the current expression.
        // When we have a descriptor cast, the ref value needs to be moved
        // across the descriptor value, which might have side effects. If so, we
        // need to use a scratch local.
        auto getRefValue = [&]() -> Expression* {
          if (curr->desc) {
            // Preserve the trap on a null descriptor.
            if (curr->desc->type.isNullable()) {
              curr->desc = builder.makeRefAs(RefAsNonNull, curr->desc);
            }
            Block* ref =
              ChildLocalizer(curr, getFunction(), *getModule(), passOptions)
                .getChildrenReplacement();
            ref->list.push_back(curr->ref);
            ref->type = curr->ref->type;
            return ref;
          }
          return curr->ref;
        };

        switch (curr->op) {
          case BrOnNull:
            if (refType.isNull()) {
              // The branch will definitely be taken.
              replaceCurrent(builder.makeSequence(
                builder.makeDrop(curr->ref), builder.makeBreak(curr->name)));
              worked = true;
              return;
            }
            if (refType.isNonNullable()) {
              // The branch will definitely not be taken.
              replaceCurrent(maybeCast(curr->ref, curr->type));
              worked = true;
              return;
            }
            return;
          case BrOnNonNull:
            if (refType.isNull()) {
              // Definitely not taken.
              replaceCurrent(builder.makeDrop(curr->ref));
              worked = true;
              return;
            }
            if (refType.isNonNullable()) {
              // Definitely taken.
              replaceCurrent(builder.makeBreak(
                curr->name, maybeCast(curr->ref, curr->getSentType())));
              worked = true;
              return;
            }
            return;
          case BrOnCast:
          case BrOnCastFail:
          case BrOnCastDescEq:
          case BrOnCastDescEqFail: {
            bool onFail =
              curr->op == BrOnCastFail || curr->op == BrOnCastDescEqFail;
            bool isDesc =
              curr->op == BrOnCastDescEq || curr->op == BrOnCastDescEqFail;

            // Improve the cast target type as much as possible given what we
            // know about the input. Unlike in BrOn::finalize(), we consider
            // type information from all the fallthrough values here. We can
            // continue to further optimizations after this, and those
            // optimizations might even benefit from this improvement.
            auto improvedType = curr->castType;
            if (!isDesc) {
              improvedType = Type::getGreatestLowerBound(improvedType, refType);
            } else {
              // For descriptor casts, the target heap type is controlled by the
              // descriptor operand, but we can still improve nullability.
              if (improvedType.isNullable() && refType.isNonNullable()) {
                improvedType = improvedType.with(NonNullable);
              }
            }
            if (!curr->castType.isExact()) {
              // When custom descriptors is not enabled, nontrivial exact casts
              // are not allowed.
              improvedType =
                improvedType.withInexactIfNoCustomDescs(getModule()->features);
            }
            if (onFail) {
              // BrOnCastFail sends the input type, with adjusted nullability.
              // The input heap type makes sense for the branch target, and we
              // will not change it anyhow, but we need to be careful with
              // nullability: if the cast type was nullable, then we were
              // sending a non-nullable value to the branch, and if we refined
              // the cast type to non- nullable, we would no longer be doing
              // that. In other words, we must not refine the nullability, as
              // that would *un*refine the send type.
              // TODO: Consider allowing this change if the branch target
              // expects a nullable type anyway.
              if (curr->castType.isNullable() && improvedType.isNonNullable()) {
                improvedType = improvedType.with(Nullable);
              }
            }
            if (improvedType != Type::unreachable &&
                improvedType != curr->castType) {
              curr->castType = improvedType;
              auto oldType = curr->type;
              curr->finalize();
              worked = true;

              // We refined the castType, which may *un*-refine the BrOn itself.
              // Imagine the castType was nullable before, then nulls would go
              // on the branch, and so the BrOn could only flow out a
              // non-nullable value, and that was its type. If we refine the
              // castType to be non-nullable then nulls no longer go through,
              // making the BrOn itself nullable. This should not normally
              // happen, but can occur because we look at the fallthrough of the
              // ref:
              //
              //   (br_on_cast
              //     (local.tee $unrefined
              //       (refined
              //
              // That is, we may see a more refined type for our GLB computation
              // than the wasm type system does, if a local.tee or such ends up
              // unrefining the type.
              //
              // To check for this and fix it, see if we need a cast in order to
              // be a subtype of the old type.
              auto* rep = maybeCast(curr, oldType);
              if (rep != curr) {
                replaceCurrent(rep);
                // Exit after doing so, leaving further work for other cycles.
                return;
              }
            }

            // Depending on what we know about the cast results, we may be able
            // to optimize.
            auto result =
              GCTypeUtils::evaluateCastCheck(refType, curr->castType);

            if (isDesc) {
              // Knowing that the types work out is insufficient to know that a
              // descriptor cast will succeed. We would need to know that the
              // descriptor values match as well.
              switch (result) {
                case GCTypeUtils::Success:
                case GCTypeUtils::SuccessOnlyIfNonNull:
                  // We cannot optimize.
                  return;
                case GCTypeUtils::Unknown:
                case GCTypeUtils::Failure:
                case GCTypeUtils::Unreachable:
                case GCTypeUtils::SuccessOnlyIfNull:
                  break;
              }
            }

            if (onFail) {
              result = GCTypeUtils::flipEvaluationResult(result);
            }

            switch (result) {
              case GCTypeUtils::Unknown:
                // Anything could happen, so we cannot optimize.
                return;
              case GCTypeUtils::Success: {
                replaceCurrent(builder.makeBreak(
                  curr->name, maybeCast(getRefValue(), curr->getSentType())));
                worked = true;
                return;
              }
              case GCTypeUtils::Failure: {
                replaceCurrent(maybeCast(getRefValue(), curr->type));
                worked = true;
                return;
              }
              case GCTypeUtils::SuccessOnlyIfNull: {
                // TODO: optimize this case using the following replacement,
                // which avoids using any scratch locals and only does a single
                // null check, but does require generating a fresh label:
                //
                //   (br_on_cast $l (ref null $X) (ref null $Y)
                //     (...)
                //   )
                //     =>
                //   (block $l' (result (ref $X))
                //     (br_on_non_null $l' ;; reuses `curr`
                //       (...)
                //     )
                //     (br $l
                //       (ref.null bot<X>)
                //     )
                //   )
                return;
              }
              case GCTypeUtils::SuccessOnlyIfNonNull: {
                // Perform this replacement:
                //
                //   (br_on_cast $l (ref null $X') (ref $X))
                //     (...)
                //   )
                //     =>
                //   (block (result (ref bot<X>))
                //     (br_on_non_null $l ;; reuses `curr`
                //       (...)
                //     (ref.null bot<X>)
                //   )
                //
                // A RefCast is added in some cases, but this is still generally
                // worth doing as the BrOnNonNull and the appended null may end
                // up optimized with surrounding code.
                auto* casted =
                  maybeCast(getRefValue(), curr->getSentType().with(Nullable));
                curr->ref = casted;
                curr->desc = nullptr;
                curr->op = BrOnNonNull;
                curr->castType = Type::none;
                curr->type = Type::none;

                assert(curr->ref->type.isRef());
                auto* refNull =
                  builder.makeRefNull(curr->ref->type.getHeapType());
                replaceCurrent(
                  builder.makeBlock({curr, refNull}, refNull->type));
                worked = true;
                return;
              }
              case GCTypeUtils::Unreachable: {
                // The cast is never executed, possibly because its input type
                // is uninhabitable. Replace it with unreachable.
                replaceCurrent(
                  getDroppedChildrenAndAppend(curr,
                                              *getModule(),
                                              passOptions,
                                              builder.makeUnreachable(),
                                              DropMode::IgnoreParentEffects));
                worked = true;
                return;
              }
            }
            break;
          }
        }
      }
    } optimizer(getPassOptions());

    optimizer.walkFunctionInModule(func, getModule());

    // If we removed any BrOn instructions, that might affect the reachability
    // of the things they used to break to, so update types.
    if (optimizer.worked) {
      ReFinalize().walkFunctionInModule(func, getModule());
      return true;
    }
    return false;
  }

  bool foldBooleanIfs(Function* func) {
    struct BoolIfFolder : public PostWalker<BoolIfFolder> {
      const PassOptions& passOptions;
      bool neverUnconditionalize;
      bool worked = false;

      BoolIfFolder(const PassOptions& passOptions, bool neverUnconditionalize)
        : passOptions(passOptions),
          neverUnconditionalize(neverUnconditionalize) {}

      void visitUnary(Unary* curr) {
        if (isV4Enabled(passOptions) && curr->op == EqZInt32 &&
            (curr->value->is<If>() || curr->value->is<Select>()) &&
            canPushEqzIntoIfOrExprFree(curr->value)) {
          Builder builder(*getModule());
          pushEqzIntoIfOrExprFree(curr->value, builder);
          replaceCurrent(curr->value);
          worked = true;
        }
      }

      void visitIf(If* curr) {
        while (tryFoldBooleanIfCondition(curr,
                                         getFunction(),
                                         *getModule(),
                                         passOptions,
                                         neverUnconditionalize)) {
          worked = true;
        }
      }

      void visitBreak(Break* curr) {
        if (auto* rep = tryFoldBooleanBreakCondition(curr,
                                                     getFunction(),
                                                     *getModule(),
                                                     passOptions,
                                                     neverUnconditionalize)) {
          if (rep != curr) {
            replaceCurrent(rep);
            if (auto* iff = rep->dynCast<If>()) {
              visitIf(iff);
            }
          }
          worked = true;
        }
      }

      void visitDrop(Drop* curr) {
        if (auto* br = curr->value->dynCast<Break>()) {
          if (auto* rep = tryFoldBooleanBreakCondition(br,
                                                       getFunction(),
                                                       *getModule(),
                                                       passOptions,
                                                       neverUnconditionalize,
                                                       curr)) {
            if (rep != curr) {
              replaceCurrent(rep);
              if (auto* iff = rep->dynCast<If>()) {
                visitIf(iff);
              }
            }
            worked = true;
          }
        }
      }
    } folder(getPassOptions(), neverUnconditionalize);

    folder.walkFunctionInModule(func, getModule());
    if (folder.worked) {
      ReFinalize().walkFunctionInModule(func, getModule());
      return true;
    }
    return false;
  }

  void optimizeTailBlockReturns(Function* func) {
    std::vector<Block*> tailBlocks;
    std::unordered_map<Name, Block*> nameToBlock;
    std::function<void(Expression*, bool)> collectTailBlocks =
      [&](Expression* curr, bool insideArm) {
        if (!curr) {
          return;
        }
        if (auto* block = curr->dynCast<Block>()) {
          if (block->list.empty()) {
            return;
          }
          for (Index i = 0; i + 1 < block->list.size(); i++) {
            if (block->list[i]->type == Type::unreachable) {
              return;
            }
          }
          if (block->name.is() && block->type == func->getResults() &&
              (!insideArm || block->list.back()->type != Type::unreachable)) {
            tailBlocks.push_back(block);
            nameToBlock[block->name] = block;
          }
          collectTailBlocks(block->list.back(), insideArm);
        } else if (auto* iff = curr->dynCast<If>()) {
          if (iff->condition->type == Type::unreachable ||
              iff->type != func->getResults()) {
            return;
          }
          collectTailBlocks(iff->ifTrue, true);
          if (iff->ifFalse) {
            collectTailBlocks(iff->ifFalse, true);
          }
        } else if (auto* loop = curr->dynCast<Loop>()) {
          if (loop->type == func->getResults()) {
            collectTailBlocks(loop->body, true);
          }
        }
      };
    collectTailBlocks(func->body, false);
    if (tailBlocks.empty()) {
      return;
    }

    struct TailBrRewriter : public PostWalker<TailBrRewriter> {
      const std::unordered_map<Name, Block*>& nameToBlock;
      Type funcResults;
      Module& wasm;
      Function* func;
      std::unordered_set<Name> modifiedBlocks;

      TailBrRewriter(const std::unordered_map<Name, Block*>& nameToBlock,
                     Type funcResults,
                     Module& wasm,
                     Function* func)
        : nameToBlock(nameToBlock), funcResults(funcResults), wasm(wasm),
          func(func) {}

      static void scan(TailBrRewriter* self, Expression** currp) {
        if (auto* iff = (*currp)->dynCast<If>()) {
          if (iff->condition->type == Type::unreachable) {
            return;
          }
        }
        PostWalker<TailBrRewriter>::scan(self, currp);
      }

      void visitBreak(Break* curr) {
        if (curr->condition || !nameToBlock.contains(curr->name)) {
          return;
        }
        if (funcResults == Type::none) {
          if (curr->value) {
            return;
          }
        } else {
          if (!curr->value ||
              !Type::isSubType(curr->value->type, funcResults)) {
            return;
          }
        }
        modifiedBlocks.insert(curr->name);
        BranchHints::clear(curr, func);
        replaceCurrent(Builder(wasm).makeReturn(curr->value));
      }
    } rewriter(nameToBlock, func->getResults(), *getModule(), func);

    rewriter.walkFunction(func);
    if (rewriter.modifiedBlocks.empty()) {
      return;
    }
    for (auto* block : tailBlocks) {
      if (rewriter.modifiedBlocks.contains(block->name) &&
          !BranchUtils::BranchSeeker::has(block, block->name)) {
        block->name = Name();
      }
    }
    ReFinalize().walkFunctionInModule(func, getModule());
  }

  void optimizeNoreturnReturnTails(Function* func) {
    if (!func->getResults().isConcrete() ||
        !getPassOptions().trapsNeverHappen) {
      return;
    }
    Builder builder(*getModule());
    Type funcResults = func->getResults();
    bool changed = false;

    auto negateCond = [&](Expression* cond) -> Expression* {
      if (auto* u = cond->dynCast<Unary>(); u && u->op == EqZInt32) {
        return u->value;
      }
      return builder.makeUnary(EqZInt32, cond);
    };

    auto stealSlice = [&](Block* b, Index start, Index end) -> Expression* {
      assert(start < end && end <= b->list.size());
      if (end - start == 1) {
        return b->list[start];
      }
      auto* res = builder.makeBlock();
      for (Index i = start; i < end; i++) {
        res->list.push_back(b->list[i]);
      }
      res->finalize();
      return res;
    };

    struct SingleBrRewriter : public PostWalker<SingleBrRewriter> {
      Name target;
      Expression* noreturnBody;
      Function* func;
      Module& wasm;
      const PassOptions& passOptions;
      bool replaced = false;

      SingleBrRewriter(Name target,
                       Expression* noreturnBody,
                       Function* func,
                       Module& wasm,
                       const PassOptions& passOptions)
        : target(target), noreturnBody(noreturnBody), func(func), wasm(wasm),
          passOptions(passOptions) {}

      void visitBreak(Break* curr) {
        if (replaced || curr->name != target || curr->value ||
            curr->type == Type::unreachable) {
          return;
        }
        Builder builder(wasm);
        if (curr->condition) {
          auto* iff = builder.makeIf(curr->condition, noreturnBody);
          BranchHints::copyTo(curr, iff, func);
          BranchHints::clear(curr, func);
          while (tryFoldBooleanIfCondition(iff, func, wasm, passOptions)) {
          }
          replaceCurrent(iff);
        } else {
          BranchHints::clear(curr, func);
          replaceCurrent(noreturnBody);
        }
        replaced = true;
      }
    };

    std::function<bool(Block*)> optimizeBlockTail = [&](Block* block) -> bool {
      bool localChanged = false;
      auto& list = block->list;
      if (!list.empty()) {
        if (auto* ret = list.back()->dynCast<Return>();
            ret && ret->value && Type::isSubType(ret->value->type, funcResults)) {
          bool allPrevVoid = true;
          for (Index i = 0; i + 1 < list.size(); i++) {
            if (list[i]->type != Type::none) {
              allPrevVoid = false;
              break;
            }
          }
          if (allPrevVoid) {
            list.back() = ret->value;
            block->finalize();
            localChanged = true;
          }
        }
      }

      while (list.size() >= 2 && list.back()->is<Unreachable>()) {
        bool didStep = false;
        if (auto* tryy = list[list.size() - 2]->dynCast<Try>();
            tryy && !tryy->isDelegate()) {
          bool allPrevVoid = true;
          for (Index i = 0; i + 2 < list.size(); i++) {
            if (list[i]->type != Type::none) {
              allPrevVoid = false;
              break;
            }
          }
          if (allPrevVoid) {
            bool tryChanged = false;
            if (auto* ret = tryy->body->dynCast<Return>();
                ret && ret->value &&
                Type::isSubType(ret->value->type, funcResults)) {
              tryy->body = ret->value;
              tryChanged = true;
            } else if (auto* b = tryy->body->dynCast<Block>();
                       b && !b->name.is()) {
              if (optimizeBlockTail(b)) {
                tryChanged = true;
              }
            }
            for (Index ci = 0; ci < tryy->catchBodies.size(); ci++) {
              Expression*& cb = tryy->catchBodies[ci];
              if (auto* ret = cb->dynCast<Return>();
                  ret && ret->value &&
                  Type::isSubType(ret->value->type, funcResults)) {
                cb = ret->value;
                tryChanged = true;
              } else if (auto* b = cb->dynCast<Block>(); b && !b->name.is()) {
                if (optimizeBlockTail(b)) {
                  tryChanged = true;
                }
              }
            }
            if (tryChanged &&
                (tryy->body->type != Type::unreachable ||
                 std::any_of(tryy->catchBodies.begin(),
                             tryy->catchBodies.end(),
                             [](Expression* e) {
                               return e->type != Type::unreachable;
                             }))) {
              tryy->finalize(funcResults);
              list.pop_back();
              block->finalize(funcResults);
              localChanged = true;
              break;
            }
          }
        }

        if (list.size() < 3 || list[list.size() - 2]->type.isConcrete()) {
          break;
        }
        for (Index k = list.size() - 3;; k--) {
          if (list[k]->type.isConcrete()) {
            break;
          }
          // Pattern 2(a): 1-arm If ending in (return VAL) before noreturn tail.
          if (auto* iff = list[k]->dynCast<If>();
              iff && !iff->ifFalse &&
              iff->condition->type != Type::unreachable) {
            std::vector<Expression*> thenPrefix;
            Return* ret = iff->ifTrue->dynCast<Return>();
            if (!ret) {
              if (auto* tb = iff->ifTrue->dynCast<Block>();
                  tb && !tb->name.is() && !tb->list.empty()) {
                if (auto* r = tb->list.back()->dynCast<Return>()) {
                  bool prefixOk = true;
                  for (Index j = 0; j + 1 < tb->list.size(); j++) {
                    if (tb->list[j]->type != Type::none) {
                      prefixOk = false;
                      break;
                    }
                    thenPrefix.push_back(tb->list[j]);
                  }
                  if (prefixOk) {
                    ret = r;
                  } else {
                    thenPrefix.clear();
                  }
                }
              }
            }
            if (ret && ret->value &&
                Type::isSubType(ret->value->type, funcResults)) {
              Expression* noreturnBody = stealSlice(block, k + 1, list.size() - 1);
              Expression* retVal = ret->value;
              iff->condition = negateCond(iff->condition);
              iff->ifTrue = noreturnBody;
              BranchHints::flip(iff, func);
              iff->finalize();
              while (tryFoldBooleanIfCondition(
                iff, func, *getModule(), getPassOptions())) {
              }
              list.resize(k + 1);
              for (auto* item : thenPrefix) {
                list.push_back(item);
              }
              list.push_back(retVal);
              block->finalize(funcResults);
              localChanged = true;
              didStep = true;
              break;
            }
          }

          // Pattern 2(b): Void-block tower ending in (return VAL) before
          // noreturn tail (supports chain depth d >= 2 of nested void blocks).
          if (auto* topTower = list[k]->dynCast<Block>();
              topTower && topTower->name.is() && topTower->type == Type::none &&
              topTower->list.size() >= 2) {
            std::vector<Block*> chain;
            Block* curB = topTower;
            Return* ret = nullptr;
            while (curB && curB->name.is() && curB->type == Type::none &&
                   curB->list.size() >= 2) {
              chain.push_back(curB);
              if (auto* r = curB->list.back()->dynCast<Return>();
                  r && r->value &&
                  Type::isSubType(r->value->type, funcResults)) {
                ret = r;
                break;
              }
              curB = curB->list[0]->dynCast<Block>();
            }
            if (ret) {
              Block* targetTower = chain.back();
              bool tailOk = true;
              for (size_t m = 0; m < chain.size() && tailOk; m++) {
                Block* b = chain[m];
                Index endJ =
                  (m + 1 == chain.size()) ? b->list.size() - 1 : b->list.size();
                for (Index j = 1; j < endJ; j++) {
                  if (b->list[j]->type != Type::none) {
                    tailOk = false;
                    break;
                  }
                  for (size_t p = 0; p <= m; p++) {
                    if (BranchUtils::BranchSeeker::has(b->list[j],
                                                       chain[p]->name)) {
                      tailOk = false;
                      break;
                    }
                  }
                  if (!tailOk) {
                    break;
                  }
                }
              }
              for (size_t p = 0; p < chain.size() && tailOk; p++) {
                if (BranchUtils::BranchSeeker::has(ret->value,
                                                   chain[p]->name)) {
                  tailOk = false;
                }
              }
              if (tailOk) {
                auto* innerTower = targetTower->list[0]->dynCast<Block>();
                if (innerTower && innerTower->name.is() &&
                    innerTower->type == Type::none &&
                    !innerTower->list.empty() &&
                    innerTower->list.back()->type == Type::unreachable) {
                  bool hasAllBranches = BranchUtils::BranchSeeker::has(
                    innerTower, innerTower->name);
                  for (Block* b : chain) {
                    if (!BranchUtils::BranchSeeker::has(innerTower, b->name)) {
                      hasAllBranches = false;
                      break;
                    }
                  }
                  if (hasAllBranches) {
                    chain.push_back(innerTower);
                    size_t d = chain.size();
                    std::vector<Expression*> towerTailPrefix;
                    for (Index j = 1; j + 1 < targetTower->list.size(); j++) {
                      towerTailPrefix.push_back(targetTower->list[j]);
                    }
                    Expression* retVal = ret->value;
                    std::vector<std::vector<Expression*>> savedTails(d - 2);
                    for (size_t m = 0; m + 2 < d; m++) {
                      for (Index j = 1; j < chain[m]->list.size(); j++) {
                        savedTails[m].push_back(chain[m]->list[j]);
                      }
                    }
                    std::vector<Expression*> noreturnItems;
                    for (Index j = k + 1; j + 1 < list.size(); j++) {
                      noreturnItems.push_back(list[j]);
                    }
                    Name returnArmName = chain[d - 1]->name;
                    for (size_t m = d - 1; m >= 1; m--) {
                      chain[m]->name = chain[m - 1]->name;
                    }
                    chain[0]->name = returnArmName;
                    chain[d - 1]->finalize();
                    for (size_t m = d - 2; m >= 1; m--) {
                      chain[m]->list.resize(1);
                      for (auto* item : savedTails[m - 1]) {
                        chain[m]->list.push_back(item);
                      }
                      chain[m]->finalize();
                    }
                    chain[0]->list.resize(1);
                    for (auto* item : noreturnItems) {
                      chain[0]->list.push_back(item);
                    }
                    chain[0]->finalize();
                    list.resize(k + 1);
                    for (auto* item : towerTailPrefix) {
                      list.push_back(item);
                    }
                    list.push_back(retVal);
                    block->finalize(funcResults);
                    localChanged = true;
                    didStep = true;
                    break;
                  }
                } else if (chain.size() == 1 &&
                           topTower->list[0]->type == Type::none &&
                           BranchUtils::BranchSeeker::count(
                             topTower, topTower->name) == 1) {
                  Expression* noreturnBody =
                    stealSlice(block, k + 1, list.size() - 1);
                  if (!EffectAnalyzer(
                         getPassOptions(), *getModule(), noreturnBody)
                         .hasExternalBreakTargets()) {
                    SingleBrRewriter rewriter(topTower->name,
                                              noreturnBody,
                                              func,
                                              *getModule(),
                                              getPassOptions());
                    Expression* temp = topTower;
                    rewriter.walk(temp);
                    if (rewriter.replaced) {
                      topTower->name = Name();
                      topTower->list.back() = ret->value;
                      topTower->finalize();
                      list.resize(k + 1);
                      block->finalize(funcResults);
                      localChanged = true;
                      didStep = true;
                      break;
                    }
                  }
                }
              }
            }
          }

          if (k == 0) {
            break;
          }
        }
        if (!didStep) {
          break;
        }
      }
      return localChanged;
    };

    std::function<void(Expression*)> walkReturnSpine = [&](Expression* curr) {
      if (auto* block = curr->dynCast<Block>()) {
        if (block->list.empty()) {
          return;
        }
        for (Index i = 0; i + 1 < block->list.size(); i++) {
          if (block->list[i]->type == Type::unreachable) {
            return;
          }
        }
        walkReturnSpine(block->list.back());
        if (block->type == funcResults || block->type == Type::unreachable) {
          if (optimizeBlockTail(block)) {
            changed = true;
          }
        }
      } else if (auto* iff = curr->dynCast<If>()) {
        if (iff->condition->type == Type::unreachable ||
            (iff->type != funcResults && iff->type != Type::unreachable)) {
          return;
        }
        walkReturnSpine(iff->ifTrue);
        if (iff->ifFalse) {
          walkReturnSpine(iff->ifFalse);
        }
      } else if (auto* loop = curr->dynCast<Loop>()) {
        if (loop->type == funcResults || loop->type == Type::unreachable) {
          walkReturnSpine(loop->body);
        }
      }
    };

    walkReturnSpine(func->body);
    if (changed) {
      ReFinalize().walkFunctionInModule(func, getModule());
    }
  }

  void lowerNonNullAndVoidReturnGuards(Function* func, bool allowFuncExit) {
    if (!func->body) {
      return;
    }
    bool hasGC = getModule()->features.hasGC();

    std::unordered_map<Name, Block*> nameToBlock;
    std::unordered_set<Name> usedNames;
    std::vector<Index> localGetCounts(func->getNumLocals(), 0);

    struct PreScanner : public PostWalker<PreScanner> {
      std::unordered_map<Name, Block*>& nameToBlock;
      std::unordered_set<Name>& usedNames;
      std::vector<Index>& localGetCounts;

      PreScanner(std::unordered_map<Name, Block*>& nameToBlock,
                 std::unordered_set<Name>& usedNames,
                 std::vector<Index>& localGetCounts)
        : nameToBlock(nameToBlock), usedNames(usedNames),
          localGetCounts(localGetCounts) {}

      void visitBlock(Block* curr) {
        if (curr->name.is()) {
          nameToBlock[curr->name] = curr;
          usedNames.insert(curr->name);
        }
      }
      void visitLoop(Loop* curr) {
        if (curr->name.is()) {
          usedNames.insert(curr->name);
        }
      }
      void visitTry(Try* curr) {
        if (curr->name.is()) {
          usedNames.insert(curr->name);
        }
      }
      void visitLocalGet(LocalGet* curr) { localGetCounts[curr->index]++; }
    } preScanner(nameToBlock, usedNames, localGetCounts);
    preScanner.walkFunction(func);

    LazyLocalGraph localGraph(func);
    Builder builder(*getModule());
    Type funcResults = func->getResults();

    Name funcExitName;
    auto ensureFuncExitLabel = [&]() -> Name {
      if (funcExitName.is()) {
        return funcExitName;
      }
      if (auto* bodyBlock = func->body->dynCast<Block>()) {
        if (!bodyBlock->name.is()) {
          Index id = 0;
          Name candidate;
          do {
            candidate = Name("func_exit$" + std::to_string(id++));
          } while (usedNames.contains(candidate));
          usedNames.insert(candidate);
          bodyBlock->name = candidate;
          nameToBlock[candidate] = bodyBlock;
        }
        if (bodyBlock->type != Type::unreachable && funcResults.isConcrete()) {
          bodyBlock->type = funcResults;
        }
        funcExitName = bodyBlock->name;
        return funcExitName;
      }
      Index id = 0;
      Name candidate;
      do {
        candidate = Name("func_exit$" + std::to_string(id++));
      } while (usedNames.contains(candidate));
      usedNames.insert(candidate);
      auto* wrapper = builder.makeBlock(candidate, {func->body}, funcResults);
      func->body = wrapper;
      nameToBlock[candidate] = wrapper;
      funcExitName = candidate;
      return funcExitName;
    };

    Index freshBlockId = 0;
    auto makeFreshBlockName = [&](const char* prefix) -> Name {
      Name candidate;
      do {
        candidate = Name(std::string(prefix) + std::to_string(freshBlockId++));
      } while (usedNames.contains(candidate));
      usedNames.insert(candidate);
      return candidate;
    };

    auto decrementLocalGets = [&](Expression* expr) {
      if (!expr) {
        return;
      }
      for (auto* get : FindAll<LocalGet>(expr).list) {
        assert(localGetCounts[get->index] > 0);
        localGetCounts[get->index]--;
      }
    };

    // Match (ref.is_null SRC) where either SRC is (local.get $x) /
    // (local.tee $x EXPR) and valueExpr is (local.get $x), or SRC is a
    // pure/non-trapping expression equal to valueExpr.
    auto matchRefIsNullGuard = [&](Expression* isNullExpr,
                                   Expression* valueExpr,
                                   Type targetType,
                                   Expression*& refOperandOut,
                                   LocalSet*& teeOut,
                                   LocalGet*& condGetOut,
                                   LocalGet*& valGetOut) -> bool {
      teeOut = nullptr;
      condGetOut = nullptr;
      valGetOut = nullptr;
      refOperandOut = nullptr;
      if (!hasGC || !targetType.isRef() || !valueExpr) {
        return false;
      }
      auto* isNull = isNullExpr ? isNullExpr->dynCast<RefIsNull>() : nullptr;
      if (!isNull) {
        return false;
      }
      Expression* src = isNull->value;
      if (!src->type.isRef() || !src->type.isNullable() ||
          src->type == Type::unreachable) {
        return false;
      }
      Type sentType(
        src->type.getHeapType(), NonNullable, src->type.getExactness());
      if (!Type::isSubType(sentType, targetType)) {
        return false;
      }
      if (auto* valGet = valueExpr->dynCast<LocalGet>()) {
        Index localIdx = valGet->index;
        if (auto* cGet = src->dynCast<LocalGet>()) {
          if (cGet->index != localIdx) {
            return false;
          }
          condGetOut = cGet;
        } else if (auto* cTee = src->dynCast<LocalSet>()) {
          if (!cTee->isTee() || cTee->index != localIdx ||
              cTee->value->type == Type::unreachable) {
            return false;
          }
          teeOut = cTee;
        } else {
          return false;
        }
        refOperandOut = src;
        valGetOut = valGet;
        return true;
      }
      if (ExpressionAnalyzer::equal(src, valueExpr) &&
          !EffectAnalyzer(getPassOptions(), *getModule(), src)
             .hasUnremovableSideEffects()) {
        refOperandOut = src;
        return true;
      }
      return false;
    };

    // Match (i32.eqz (ref.is_null SRC)) against valueExpr.
    auto matchNonNullGuard = [&](Expression* cond,
                                 Expression* valueExpr,
                                 Type targetType,
                                 Expression*& refOperandOut,
                                 LocalSet*& teeOut,
                                 LocalGet*& condGetOut,
                                 LocalGet*& valGetOut) -> bool {
      auto* eqz = cond ? cond->dynCast<Unary>() : nullptr;
      if (!eqz || eqz->op != EqZInt32) {
        return false;
      }
      return matchRefIsNullGuard(eqz->value,
                                 valueExpr,
                                 targetType,
                                 refOperandOut,
                                 teeOut,
                                 condGetOut,
                                 valGetOut);
    };

    std::unordered_set<Index> localsWithSynthesizedGets;

    auto stripDeadTeeIfPossible =
      [&](Expression* refOperand,
          LocalSet* tee,
          LocalGet* eliminatedValGet,
          Expression* eliminatedExpr = nullptr) -> Expression* {
      if (eliminatedValGet) {
        assert(localGetCounts[eliminatedValGet->index] > 0);
        localGetCounts[eliminatedValGet->index]--;
      } else if (eliminatedExpr) {
        decrementLocalGets(eliminatedExpr);
      }
      if (!tee) {
        return refOperand;
      }
      Index localIdx = tee->index;
      bool canStrip = (localGetCounts[localIdx] == 0);
      if (!canStrip && !func->getLocalType(localIdx).isNonNullable() &&
          !localsWithSynthesizedGets.contains(localIdx)) {
        const auto& influences = localGraph.getSetInfluences(tee);
        if (influences.size() == 1 &&
            *influences.begin() == eliminatedValGet) {
          canStrip = true;
        }
      }
      if (!canStrip) {
        return refOperand;
      }
      Expression* inner = tee->value;
      if (inner->type.isRef()) {
        return inner;
      }
      return refOperand;
    };

    using EnsureExitFn = decltype(ensureFuncExitLabel);
    using MakeBlockNameFn = decltype(makeFreshBlockName);
    using DecrementGetsFn = decltype(decrementLocalGets);
    using MatchIsNullFn = decltype(matchRefIsNullGuard);
    using MatchGuardFn = decltype(matchNonNullGuard);
    using StripTeeFn = decltype(stripDeadTeeIfPossible);

    struct GuardLowerer : public PostWalker<GuardLowerer> {
      Function* func;
      Module& wasm;
      const PassOptions& passOptions;
      Builder& builder;
      std::unordered_map<Name, Block*>& nameToBlock;
      std::vector<Index>& localGetCounts;
      std::unordered_set<Index>& localsWithSynthesizedGets;
      LazyLocalGraph& localGraph;
      EnsureExitFn& ensureFuncExitLabel;
      MakeBlockNameFn& makeFreshBlockName;
      DecrementGetsFn& decrementLocalGets;
      MatchIsNullFn& matchRefIsNullGuard;
      MatchGuardFn& matchNonNullGuard;
      StripTeeFn& stripDeadTeeIfPossible;
      bool allowFuncExit;
      bool neverUnconditionalize;
      Index tryDepth = 0;
      bool changed = false;

      GuardLowerer(Function* func,
                   Module& wasm,
                   const PassOptions& passOptions,
                   Builder& builder,
                   std::unordered_map<Name, Block*>& nameToBlock,
                   std::vector<Index>& localGetCounts,
                   std::unordered_set<Index>& localsWithSynthesizedGets,
                   LazyLocalGraph& localGraph,
                   EnsureExitFn& ensureFuncExitLabel,
                   MakeBlockNameFn& makeFreshBlockName,
                   DecrementGetsFn& decrementLocalGets,
                   MatchIsNullFn& matchRefIsNullGuard,
                   MatchGuardFn& matchNonNullGuard,
                   StripTeeFn& stripDeadTeeIfPossible,
                   bool allowFuncExit,
                   bool neverUnconditionalize)
        : func(func), wasm(wasm), passOptions(passOptions), builder(builder),
          nameToBlock(nameToBlock), localGetCounts(localGetCounts),
          localsWithSynthesizedGets(localsWithSynthesizedGets),
          localGraph(localGraph), ensureFuncExitLabel(ensureFuncExitLabel),
          makeFreshBlockName(makeFreshBlockName),
          decrementLocalGets(decrementLocalGets),
          matchRefIsNullGuard(matchRefIsNullGuard),
          matchNonNullGuard(matchNonNullGuard),
          stripDeadTeeIfPossible(stripDeadTeeIfPossible),
          allowFuncExit(allowFuncExit),
          neverUnconditionalize(neverUnconditionalize) {}

      static void scan(GuardLowerer* self, Expression** currp) {
        Expression* curr = *currp;
        if (curr->is<Try>() || curr->is<TryTable>()) {
          self->tryDepth++;
          PostWalker<GuardLowerer>::scan(self, currp);
          self->tryDepth--;
          return;
        }
        PostWalker<GuardLowerer>::scan(self, currp);
      }

      bool findTargetSlotOnSpine(Expression* src,
                                 Expression** rootSlot,
                                 bool allowNamedRootBlock,
                                 Expression**& matchedSlotOut,
                                 LocalSet*& teeOut,
                                 LocalGet*& matchedGetOut,
                                 Expression*& matchedExprOut,
                                 std::vector<Expression*>& spineAncestorsOut) {
        matchedSlotOut = nullptr;
        teeOut = nullptr;
        matchedGetOut = nullptr;
        matchedExprOut = nullptr;
        spineAncestorsOut.clear();
        if (!src || !src->type.isRef() || !src->type.isNullable() ||
            src->type == Type::unreachable) {
          return false;
        }
        bool hasTargetLocal = false;
        Index targetLocal = 0;
        Expression* pureX = nullptr;
        if (auto* cGet = src->dynCast<LocalGet>()) {
          hasTargetLocal = true;
          targetLocal = cGet->index;
        } else if (auto* cTee = src->dynCast<LocalSet>()) {
          if (cTee->isTee() && cTee->value->type != Type::unreachable &&
              cTee->value->type.isRef()) {
            hasTargetLocal = true;
            targetLocal = cTee->index;
            teeOut = cTee;
          }
        } else if (!EffectAnalyzer(passOptions, wasm, src)
                      .hasUnremovableSideEffects()) {
          pureX = src;
        }
        if (!hasTargetLocal && !pureX) {
          return false;
        }

        auto matchesTarget = [&](Expression* candidate,
                                 LocalGet*& getOut) -> bool {
          getOut = nullptr;
          if (hasTargetLocal) {
            if (auto* g = candidate->dynCast<LocalGet>();
                g && g->index == targetLocal) {
              getOut = g;
              return true;
            }
            return false;
          }
          return candidate->type == pureX->type &&
                 ExpressionAnalyzer::equal(candidate, pureX);
        };

        auto canStepPast = [&](Expression* prev) -> bool {
          EffectAnalyzer eff(passOptions, wasm, prev);
          if (eff.hasSideEffects()) {
            return false;
          }
          if (hasTargetLocal &&
              (eff.localsRead.contains(targetLocal) ||
               eff.localsWritten.contains(targetLocal))) {
            return false;
          }
          return EffectAnalyzer::canReorder(passOptions, wasm, src, prev);
        };

        std::function<bool(Expression**, bool, bool, bool, Index)> search =
          [&](Expression** slot,
              bool isRoot,
              bool pastConcreteOperand,
              bool allowDirectMatch,
              Index depth) -> bool {
          if (!slot || !*slot || depth > 24) {
            return false;
          }
          Expression* node = *slot;
          if (allowDirectMatch) {
            if (auto* ra = node->dynCast<RefAs>();
                ra && ra->op == RefAsNonNull) {
              LocalGet* g = nullptr;
              if (matchesTarget(ra->value, g)) {
                matchedSlotOut = slot;
                matchedGetOut = g;
                matchedExprOut = ra->value;
                return true;
              }
            }
            LocalGet* g = nullptr;
            if (matchesTarget(node, g)) {
              matchedSlotOut = slot;
              matchedGetOut = g;
              matchedExprOut = node;
              return true;
            }
          }

          auto tryChildren = [&](std::initializer_list<Expression**> children,
                                 bool childAllowDirectMatch = true) -> bool {
            spineAncestorsOut.push_back(node);
            bool curPastConcrete = pastConcreteOperand;
            for (Expression** childSlot : children) {
              if (!childSlot || !*childSlot) {
                continue;
              }
              if (search(childSlot,
                         false,
                         curPastConcrete,
                         childAllowDirectMatch,
                         depth + 1)) {
                return true;
              }
              if (!canStepPast(*childSlot)) {
                spineAncestorsOut.pop_back();
                return false;
              }
              if ((*childSlot)->type.isConcrete()) {
                curPastConcrete = true;
              }
            }
            spineAncestorsOut.pop_back();
            return false;
          };

          if (auto* b = node->dynCast<Block>()) {
            if (!pastConcreteOperand && !b->list.empty() &&
                (!b->name.is() ||
                 (isRoot && allowNamedRootBlock && b->type == Type::none))) {
              spineAncestorsOut.push_back(node);
              for (Index idx = 0; idx < b->list.size(); idx++) {
                if (b->list[idx]->is<Nop>()) {
                  continue;
                }
                if (search(&b->list[idx], false, false, true, depth + 1)) {
                  return true;
                }
                break;
              }
              spineAncestorsOut.pop_back();
            }
            return false;
          }
          if (auto* iff = node->dynCast<If>()) {
            if (!pastConcreteOperand) {
              return tryChildren({&iff->condition});
            }
            return false;
          }
          if (node->is<Loop>() || node->is<Try>() || node->is<TryTable>()) {
            return false;
          }
          if (auto* c = node->dynCast<Call>()) {
            spineAncestorsOut.push_back(node);
            bool curPastConcrete = pastConcreteOperand;
            for (Index idx = 0; idx < c->operands.size(); idx++) {
              if (search(
                    &c->operands[idx], false, curPastConcrete, true, depth + 1)) {
                return true;
              }
              if (!canStepPast(c->operands[idx])) {
                spineAncestorsOut.pop_back();
                return false;
              }
              curPastConcrete = true;
            }
            spineAncestorsOut.pop_back();
            return false;
          }
          if (auto* c = node->dynCast<CallIndirect>()) {
            spineAncestorsOut.push_back(node);
            bool curPastConcrete = pastConcreteOperand;
            for (Index idx = 0; idx < c->operands.size(); idx++) {
              if (search(
                    &c->operands[idx], false, curPastConcrete, true, depth + 1)) {
                return true;
              }
              if (!canStepPast(c->operands[idx])) {
                spineAncestorsOut.pop_back();
                return false;
              }
              curPastConcrete = true;
            }
            if (search(&c->target, false, curPastConcrete, true, depth + 1)) {
              return true;
            }
            spineAncestorsOut.pop_back();
            return false;
          }
          if (auto* c = node->dynCast<CallRef>()) {
            spineAncestorsOut.push_back(node);
            bool curPastConcrete = pastConcreteOperand;
            for (Index idx = 0; idx < c->operands.size(); idx++) {
              if (search(
                    &c->operands[idx], false, curPastConcrete, true, depth + 1)) {
                return true;
              }
              if (!canStepPast(c->operands[idx])) {
                spineAncestorsOut.pop_back();
                return false;
              }
              curPastConcrete = true;
            }
            if (search(&c->target, false, curPastConcrete, true, depth + 1)) {
              return true;
            }
            spineAncestorsOut.pop_back();
            return false;
          }
          if (auto* s = node->dynCast<LocalSet>()) {
            return tryChildren({&s->value});
          }
          if (auto* s = node->dynCast<GlobalSet>()) {
            return tryChildren({&s->value});
          }
          if (auto* d = node->dynCast<Drop>()) {
            return tryChildren({&d->value});
          }
          if (auto* r = node->dynCast<Return>()) {
            return r->value ? tryChildren({&r->value}) : false;
          }
          if (auto* br = node->dynCast<Break>()) {
            return tryChildren({br->value ? &br->value : nullptr,
                                br->condition ? &br->condition : nullptr});
          }
          if (auto* sw = node->dynCast<Switch>()) {
            return tryChildren(
              {sw->value ? &sw->value : nullptr, &sw->condition});
          }
          if (auto* u = node->dynCast<Unary>()) {
            return tryChildren({&u->value});
          }
          if (auto* bin = node->dynCast<Binary>()) {
            return tryChildren({&bin->left, &bin->right});
          }
          if (auto* sel = node->dynCast<Select>()) {
            return tryChildren({&sel->ifTrue, &sel->ifFalse, &sel->condition});
          }
          if (auto* sg = node->dynCast<StructGet>()) {
            return tryChildren({&sg->ref});
          }
          if (auto* ss = node->dynCast<StructSet>()) {
            return tryChildren({&ss->ref, &ss->value});
          }
          if (auto* sn = node->dynCast<StructNew>()) {
            if (sn->isWithDefault()) {
              return false;
            }
            spineAncestorsOut.push_back(node);
            bool curPastConcrete = pastConcreteOperand;
            for (Index idx = 0; idx < sn->operands.size(); idx++) {
              if (search(&sn->operands[idx],
                         false,
                         curPastConcrete,
                         true,
                         depth + 1)) {
                return true;
              }
              if (!canStepPast(sn->operands[idx])) {
                spineAncestorsOut.pop_back();
                return false;
              }
              curPastConcrete = true;
            }
            spineAncestorsOut.pop_back();
            return false;
          }
          if (auto* ag = node->dynCast<ArrayGet>()) {
            return tryChildren({&ag->ref, &ag->index});
          }
          if (auto* as = node->dynCast<ArraySet>()) {
            return tryChildren({&as->ref, &as->index, &as->value});
          }
          if (auto* al = node->dynCast<ArrayLen>()) {
            return tryChildren({&al->ref});
          }
          if (auto* af = node->dynCast<ArrayFill>()) {
            return tryChildren(
              {&af->ref, &af->index, &af->value, &af->size});
          }
          if (auto* ac = node->dynCast<ArrayCopy>()) {
            return tryChildren({&ac->destRef,
                                &ac->destIndex,
                                &ac->srcRef,
                                &ac->srcIndex,
                                &ac->length});
          }
          if (auto* an = node->dynCast<ArrayNew>()) {
            if (an->isWithDefault() || !an->init) {
              return false;
            }
            return tryChildren({&an->init, &an->size});
          }
          if (auto* anf = node->dynCast<ArrayNewFixed>()) {
            spineAncestorsOut.push_back(node);
            bool curPastConcrete = pastConcreteOperand;
            for (Index idx = 0; idx < anf->values.size(); idx++) {
              if (search(
                    &anf->values[idx], false, curPastConcrete, true, depth + 1)) {
                return true;
              }
              if (!canStepPast(anf->values[idx])) {
                spineAncestorsOut.pop_back();
                return false;
              }
              curPastConcrete = true;
            }
            spineAncestorsOut.pop_back();
            return false;
          }
          if (auto* rin = node->dynCast<RefIsNull>()) {
            return tryChildren({&rin->value}, /*childAllowDirectMatch=*/false);
          }
          if (auto* ra = node->dynCast<RefAs>()) {
            return tryChildren({&ra->value});
          }
          if (auto* rc = node->dynCast<RefCast>()) {
            return tryChildren({&rc->ref});
          }
          if (auto* rt = node->dynCast<RefTest>()) {
            return tryChildren({&rt->ref});
          }
          if (auto* re = node->dynCast<RefEq>()) {
            return tryChildren({&re->left, &re->right});
          }
          if (auto* bo = node->dynCast<BrOn>()) {
            bool allowDirect =
              (bo->op != BrOnNull && bo->op != BrOnNonNull);
            return tryChildren({&bo->ref}, allowDirect);
          }
          if (auto* l = node->dynCast<Load>()) {
            return tryChildren({&l->ptr});
          }
          if (auto* st = node->dynCast<Store>()) {
            return tryChildren({&st->ptr, &st->value});
          }
          return false;
        };

        return search(rootSlot,
                      /*isRoot=*/true,
                      /*pastConcreteOperand=*/false,
                      /*allowDirectMatch=*/true,
                      0);
      }

      static bool stmtUnconditionallyWritesLocal(Expression* stmt,
                                                 Index localIdx,
                                                 Index depth = 0) {
        if (!stmt || depth > 8) {
          return false;
        }
        if (auto* ls = stmt->dynCast<LocalSet>()) {
          return ls->index == localIdx ||
                 stmtUnconditionallyWritesLocal(ls->value, localIdx, depth + 1);
        }
        if (auto* b = stmt->dynCast<Block>()) {
          if (b->name.is()) {
            return false;
          }
          for (auto* item : b->list) {
            if (stmtUnconditionallyWritesLocal(item, localIdx, depth + 1)) {
              return true;
            }
          }
          return false;
        }
        if (auto* iff = stmt->dynCast<If>()) {
          return stmtUnconditionallyWritesLocal(
                   iff->condition, localIdx, depth + 1) ||
                 (iff->ifFalse &&
                  stmtUnconditionallyWritesLocal(
                    iff->ifTrue, localIdx, depth + 1) &&
                  stmtUnconditionallyWritesLocal(
                    iff->ifFalse, localIdx, depth + 1));
        }
        if (auto* ss = stmt->dynCast<StructSet>()) {
          return stmtUnconditionallyWritesLocal(ss->ref, localIdx, depth + 1) ||
                 stmtUnconditionallyWritesLocal(ss->value, localIdx, depth + 1);
        }
        if (auto* gs = stmt->dynCast<GlobalSet>()) {
          return stmtUnconditionallyWritesLocal(gs->value, localIdx, depth + 1);
        }
        if (auto* d = stmt->dynCast<Drop>()) {
          return stmtUnconditionallyWritesLocal(d->value, localIdx, depth + 1);
        }
        if (auto* c = stmt->dynCast<Call>()) {
          for (auto* op : c->operands) {
            if (stmtUnconditionallyWritesLocal(op, localIdx, depth + 1)) {
              return true;
            }
          }
          return false;
        }
        if (auto* sn = stmt->dynCast<StructNew>()) {
          if (!sn->isWithDefault()) {
            for (auto* op : sn->operands) {
              if (stmtUnconditionallyWritesLocal(op, localIdx, depth + 1)) {
                return true;
              }
            }
          }
          return false;
        }
        if (auto* as = stmt->dynCast<ArraySet>()) {
          return stmtUnconditionallyWritesLocal(as->ref, localIdx, depth + 1) ||
                 stmtUnconditionallyWritesLocal(
                   as->index, localIdx, depth + 1) ||
                 stmtUnconditionallyWritesLocal(as->value, localIdx, depth + 1);
        }
        return false;
      }

      void visitDrop(Drop* curr) {
        auto* br = curr->value->dynCast<Break>();
        if (!br || !br->condition || !br->value ||
            br->type == Type::unreachable) {
          return;
        }
        auto it = nameToBlock.find(br->name);
        if (it == nameToBlock.end()) {
          return;
        }
        Block* targetBlock = it->second;
        Expression* refOperand = nullptr;
        LocalSet* tee = nullptr;
        LocalGet* condGet = nullptr;
        LocalGet* valGet = nullptr;
        if (!matchNonNullGuard(br->condition,
                               br->value,
                               targetBlock->type,
                               refOperand,
                               tee,
                               condGet,
                               valGet)) {
          return;
        }
        Expression* ref =
          stripDeadTeeIfPossible(refOperand, tee, valGet, br->value);
        BranchHints::clear(br, func);
        replaceCurrent(builder.makeBrOn(BrOnNonNull, br->name, ref));
        changed = true;
      }

      void visitIf(If* curr) {
        if (curr->condition->type == Type::unreachable) {
          return;
        }
        if (curr->ifFalse) {
          // Part 3(a): 2-arm null-coalescing If ->
          // (block $b (result T) (br_on_non_null $b X') FALLBACK).
          if (!allowFuncExit || neverUnconditionalize || !curr->type.isRef()) {
            return;
          }
          Expression* refOperand = nullptr;
          LocalSet* tee = nullptr;
          LocalGet* condGet = nullptr;
          LocalGet* valGet = nullptr;
          Expression* fallback = nullptr;
          Expression* valExpr = nullptr;
          if (matchRefIsNullGuard(curr->condition,
                                  curr->ifFalse,
                                  curr->type,
                                  refOperand,
                                  tee,
                                  condGet,
                                  valGet)) {
            fallback = curr->ifTrue;
            valExpr = curr->ifFalse;
          } else if (matchNonNullGuard(curr->condition,
                                       curr->ifTrue,
                                       curr->type,
                                       refOperand,
                                       tee,
                                       condGet,
                                       valGet)) {
            fallback = curr->ifFalse;
            valExpr = curr->ifTrue;
          }
          if (!fallback) {
            return;
          }
          Expression* ref =
            stripDeadTeeIfPossible(refOperand, tee, valGet, valExpr);
          BranchHints::clear(curr, func);
          if (fallback->is<RefNull>() &&
              Type::isSubType(ref->type, curr->type)) {
            replaceCurrent(ref);
            changed = true;
            return;
          }
          Name bName = makeFreshBlockName("coalesce$");
          auto* brOn = builder.makeBrOn(BrOnNonNull, bName, ref);
          auto* blk = builder.makeBlock(bName, {brOn}, curr->type);
          if (auto* fb = fallback->dynCast<Block>(); fb && !fb->name.is()) {
            for (auto* item : fb->list) {
              blk->list.push_back(item);
            }
          } else {
            blk->list.push_back(fallback);
          }
          blk->finalize(curr->type);
          nameToBlock[bName] = blk;
          replaceCurrent(blk);
          changed = true;
          return;
        }

        // Unwrap single-element unnamed Block in ifTrue if present.
        Expression* arm = curr->ifTrue;
        while (auto* b = arm->dynCast<Block>()) {
          if (!b->name.is() && b->list.size() == 1) {
            arm = b->list[0];
          } else {
            break;
          }
        }

        // 2(a): (if (i32.eqz (ref.is_null X)) (then (br $L X)))
        if (auto* br = arm->dynCast<Break>();
            br && !br->condition && br->value) {
          auto it = nameToBlock.find(br->name);
          if (it != nameToBlock.end()) {
            Block* targetBlock = it->second;
            Expression* refOperand = nullptr;
            LocalSet* tee = nullptr;
            LocalGet* condGet = nullptr;
            LocalGet* valGet = nullptr;
            if (matchNonNullGuard(curr->condition,
                                  br->value,
                                  targetBlock->type,
                                  refOperand,
                                  tee,
                                  condGet,
                                  valGet)) {
              Expression* ref =
                stripDeadTeeIfPossible(refOperand, tee, valGet, br->value);
              BranchHints::clear(curr, func);
              BranchHints::clear(br, func);
              replaceCurrent(builder.makeBrOn(BrOnNonNull, br->name, ref));
              changed = true;
              return;
            }
          }
        }

        // 2(b) & 2(c): 1-arm If whose arm is (return [VAL]), outside try/try_table.
        if (allowFuncExit && tryDepth == 0) {
          if (auto* ret = arm->dynCast<Return>()) {
            Type funcResults = func->getResults();
            // 2(b): (if (i32.eqz (ref.is_null X)) (then (return X)))
            if (ret->value && funcResults.isRef()) {
              Expression* refOperand = nullptr;
              LocalSet* tee = nullptr;
              LocalGet* condGet = nullptr;
              LocalGet* valGet = nullptr;
              if (matchNonNullGuard(curr->condition,
                                    ret->value,
                                    funcResults,
                                    refOperand,
                                    tee,
                                    condGet,
                                    valGet)) {
                Expression* ref =
                  stripDeadTeeIfPossible(refOperand, tee, valGet, ret->value);
                Name exitLabel = ensureFuncExitLabel();
                BranchHints::clear(curr, func);
                replaceCurrent(builder.makeBrOn(BrOnNonNull, exitLabel, ref));
                changed = true;
                return;
              }
            }
            // 2(c): (if COND (then (return))) in a void function.
            if (!ret->value && funcResults == Type::none) {
              Name exitLabel = ensureFuncExitLabel();
              auto* brIf =
                builder.makeBreak(exitLabel, nullptr, curr->condition);
              BranchHints::copyTo(curr, brIf, func);
              BranchHints::clear(curr, func);
              replaceCurrent(brIf);
              changed = true;
              return;
            }
          }
        }

        // Part 2: br_on_null spine-consuming lowering for 1-arm void
        // (if (i32.eqz (ref.is_null X)) (then BODY)).
        if (wasm.features.hasGC() && allowFuncExit && !neverUnconditionalize &&
            curr->type == Type::none) {
          auto* eqz = curr->condition->dynCast<Unary>();
          auto* isNull = (eqz && eqz->op == EqZInt32)
                           ? eqz->value->dynCast<RefIsNull>()
                           : nullptr;
          if (isNull) {
            Expression* src = isNull->value;
            Expression** matchedSlot = nullptr;
            LocalSet* tee = nullptr;
            LocalGet* matchedGet = nullptr;
            Expression* matchedExpr = nullptr;
            std::vector<Expression*> spineAncestors;
            if (findTargetSlotOnSpine(src,
                                      &curr->ifTrue,
                                      /*allowNamedRootBlock=*/true,
                                      matchedSlot,
                                      tee,
                                      matchedGet,
                                      matchedExpr,
                                      spineAncestors)) {
              Expression* ref =
                stripDeadTeeIfPossible(src, tee, matchedGet, matchedExpr);
              Block* blk = curr->ifTrue->dynCast<Block>();
              Name bName;
              if (blk && !blk->name.is()) {
                bName = makeFreshBlockName("br_on_null$");
                blk->name = bName;
              } else if (blk && blk->name.is() &&
                         blk->type == Type::none) {
                bName = blk->name;
              } else {
                bName = makeFreshBlockName("br_on_null$");
                blk = nullptr;
              }
              *matchedSlot = builder.makeBrOn(BrOnNull, bName, ref);
              for (auto ancIt = spineAncestors.rbegin();
                   ancIt != spineAncestors.rend();
                   ++ancIt) {
                if (auto* b = (*ancIt)->dynCast<Block>()) {
                  b->finalize(b->type);
                } else {
                  (*ancIt)->finalize();
                }
              }
              if (!blk) {
                blk = builder.makeBlock(bName, {curr->ifTrue}, Type::none);
              } else {
                blk->finalize(Type::none);
              }
              nameToBlock[bName] = blk;
              BranchHints::clear(curr, func);
              replaceCurrent(blk);
              changed = true;
              return;
            }
          }
        }
      }

      void visitBlock(Block* curr) {
        Type targetType =
          (curr == func->body && func->getResults().isRef())
            ? func->getResults()
            : curr->type;

        // Part 2(b): Adjacent [(br_if $L (ref.is_null X)), NEXT_STMT] ->
        // fold (br_on_null $L X') directly into NEXT_STMT's spine.
        if (wasm.features.hasGC() && allowFuncExit && !neverUnconditionalize &&
            curr->list.size() >= 2) {
          for (Index idx = curr->list.size() - 1; idx > 0; idx--) {
            Index i = idx - 1;
            auto* br = curr->list[i]->dynCast<Break>();
            if (!br || br->value || !br->condition || br->type != Type::none) {
              continue;
            }
            auto it = nameToBlock.find(br->name);
            if (it == nameToBlock.end() || it->second->type != Type::none) {
              continue;
            }
            auto* isNull = br->condition->dynCast<RefIsNull>();
            if (!isNull) {
              continue;
            }
            Expression* src = isNull->value;
            Expression** matchedSlot = nullptr;
            LocalSet* tee = nullptr;
            LocalGet* matchedGet = nullptr;
            Expression* matchedExpr = nullptr;
            std::vector<Expression*> spineAncestors;
            if (!findTargetSlotOnSpine(src,
                                       &curr->list[i + 1],
                                       /*allowNamedRootBlock=*/false,
                                       matchedSlot,
                                       tee,
                                       matchedGet,
                                       matchedExpr,
                                       spineAncestors)) {
              continue;
            }
            Expression* ref =
              stripDeadTeeIfPossible(src, tee, matchedGet, matchedExpr);
            *matchedSlot = builder.makeBrOn(BrOnNull, br->name, ref);
            for (auto ancIt = spineAncestors.rbegin();
                 ancIt != spineAncestors.rend();
                 ++ancIt) {
              if (auto* b = (*ancIt)->dynCast<Block>()) {
                b->finalize(b->type);
              } else {
                (*ancIt)->finalize();
              }
            }
            BranchHints::clear(br, func);
            curr->list.erase(curr->list.begin() + i,
                             curr->list.begin() + i + 1);
            curr->finalize(curr->type);
            changed = true;
          }
        }

        // Part 3(c): Lazy-init 1-arm void (if (ref.is_null X) (then BODY))
        // followed by VAL / (return VAL) / (br $L VAL) -> br_on_non_null.
        if (wasm.features.hasGC() && allowFuncExit && !neverUnconditionalize &&
            curr->list.size() >= 2) {
          for (Index i = 0; i + 1 < curr->list.size(); i++) {
            auto* iff = curr->list[i]->dynCast<If>();
            if (!iff || iff->ifFalse || iff->type != Type::none ||
                iff->condition->type == Type::unreachable) {
              continue;
            }
            auto* isNull = iff->condition->dynCast<RefIsNull>();
            if (!isNull) {
              continue;
            }
            Expression* src = isNull->value;
            if (!src->type.isRef() || !src->type.isNullable() ||
                src->type == Type::unreachable) {
              continue;
            }
            Expression* nextExpr = curr->list[i + 1];
            Return* nextRet = nextExpr->dynCast<Return>();
            Break* nextBr = nextExpr->dynCast<Break>();
            Expression* valExpr = nullptr;
            Expression** valPtr = nullptr;
            Type lazyTargetType = Type::none;
            bool isRetMode = false;
            bool isBrMode = false;
            bool isTailValMode = false;
            bool isFuncExitTail = false;
            if (nextRet && nextRet->value && tryDepth == 0 &&
                func->getResults().isRef()) {
              isRetMode = true;
              valExpr = nextRet->value;
              valPtr = &nextRet->value;
              lazyTargetType = func->getResults();
              isFuncExitTail = true;
            } else if (nextBr && !nextBr->condition && nextBr->value) {
              auto it = nameToBlock.find(nextBr->name);
              if (it != nameToBlock.end() && it->second->type.isRef()) {
                isBrMode = true;
                valExpr = nextBr->value;
                valPtr = &nextBr->value;
                lazyTargetType = it->second->type;
              }
            } else if (i + 1 == curr->list.size() - 1 &&
                       targetType.isRef() && nextExpr->type.isRef()) {
              isTailValMode = true;
              valExpr = nextExpr;
              valPtr = &curr->list[i + 1];
              lazyTargetType = targetType;
              isFuncExitTail =
                (curr == func->body && func->getResults().isRef() &&
                 tryDepth == 0);
            }
            if (!valExpr || !lazyTargetType.isRef()) {
              continue;
            }
            Type sentType(
              src->type.getHeapType(), NonNullable, src->type.getExactness());
            if (!Type::isSubType(sentType, lazyTargetType)) {
              continue;
            }

            LocalGet* valGet = valExpr->dynCast<LocalGet>();
            LocalSet* condTee = nullptr;
            StructGet* sgSrc = src->dynCast<StructGet>();
            StructGet* sgVal = valExpr->dynCast<StructGet>();
            GlobalGet* ggSrc = src->dynCast<GlobalGet>();
            GlobalGet* ggVal = valExpr->dynCast<GlobalGet>();
            bool matchedPair = false;
            if (valGet) {
              if (auto* cGet = src->dynCast<LocalGet>();
                  cGet && cGet->index == valGet->index) {
                matchedPair = true;
              } else if (auto* cTee = src->dynCast<LocalSet>();
                         cTee && cTee->isTee() &&
                         cTee->index == valGet->index &&
                         cTee->value->type != Type::unreachable &&
                         cTee->value->type.isRef()) {
                condTee = cTee;
                matchedPair = true;
              }
            } else if (sgSrc && sgVal && sgSrc->index == sgVal->index &&
                       sgSrc->signed_ == sgVal->signed_ &&
                       sgSrc->order == sgVal->order &&
                       sgSrc->type == sgVal->type &&
                       !EffectAnalyzer(passOptions, wasm, sgVal)
                          .hasUnremovableSideEffects()) {
              if (!EffectAnalyzer(passOptions, wasm, sgSrc)
                     .hasUnremovableSideEffects() &&
                  ExpressionAnalyzer::equal(sgSrc->ref, sgVal->ref)) {
                matchedPair = true;
              } else if (auto* rTee = sgSrc->ref->dynCast<LocalSet>()) {
                if (auto* rGet = sgVal->ref->dynCast<LocalGet>();
                    rTee->isTee() && rTee->value->type != Type::unreachable &&
                    rGet && rTee->index == rGet->index) {
                  matchedPair = true;
                }
              }
            } else if (ggSrc && ggVal && ggSrc->name == ggVal->name) {
              matchedPair = true;
            }
            if (!matchedPair) {
              continue;
            }

            // Locate the last statement of iff->ifTrue.
            bool thenIsUnreachable = (iff->ifTrue->type == Type::unreachable);
            Block* thenBlock = iff->ifTrue->dynCast<Block>();
            Expression** lastStmtPtr =
              (thenBlock && !thenBlock->name.is() && !thenBlock->list.empty())
                ? &thenBlock->list.back()
                : &iff->ifTrue;
            Expression* lastStmt = *lastStmtPtr;

            Expression* refOperand = src;
            bool foldedLastSetIntoVal = false;
            if (thenIsUnreachable) {
              // iff->ifTrue never falls through to curr->list[i + 1], so
              // curr->list[i + 1] is only reached when src is non-null and can
              // be folded directly into br_on_non_null.
              refOperand =
                stripDeadTeeIfPossible(src, condTee, valGet, valExpr);
            } else if (valGet) {
              Index localIdx = valGet->index;
              bool bodyOverwritesLocalAtEnd =
                stmtUnconditionallyWritesLocal(iff->ifTrue, localIdx);
              if (auto* lastSet = lastStmt->dynCast<LocalSet>();
                  lastSet && !lastSet->isTee() &&
                  lastSet->index == localIdx &&
                  lastSet->value->type != Type::unreachable &&
                  Type::isSubType(lastSet->value->type, lazyTargetType)) {
                bodyOverwritesLocalAtEnd = true;
                foldedLastSetIntoVal = true;
              }
              if (condTee && bodyOverwritesLocalAtEnd) {
                bool canStripCondTee = false;
                if (foldedLastSetIntoVal && localGetCounts[localIdx] == 1) {
                  canStripCondTee = true;
                } else if (!func->getLocalType(localIdx).isNonNullable() &&
                           !localsWithSynthesizedGets.contains(localIdx)) {
                  const auto& inf = localGraph.getSetInfluences(condTee);
                  if (inf.size() == 1 && *inf.begin() == valGet) {
                    canStripCondTee = true;
                  }
                }
                if (canStripCondTee && condTee->value->type.isRef()) {
                  refOperand = condTee->value;
                }
              }
              if (foldedLastSetIntoVal) {
                auto* lastSet = lastStmt->cast<LocalSet>();
                assert(localGetCounts[localIdx] > 0);
                localGetCounts[localIdx]--;
                Expression* replacementVal = lastSet->value;
                bool canDropLastSet =
                  (localGetCounts[localIdx] == 0) || isFuncExitTail;
                if (!canDropLastSet &&
                    !func->getLocalType(localIdx).isNonNullable() &&
                    !localsWithSynthesizedGets.contains(localIdx)) {
                  const auto& inf = localGraph.getSetInfluences(lastSet);
                  if (inf.size() == 1 && *inf.begin() == valGet) {
                    canDropLastSet = true;
                  }
                }
                if (!canDropLastSet) {
                  Type localT = func->getLocalType(localIdx);
                  if (Type::isSubType(localT, lazyTargetType)) {
                    lastSet->makeTee(localT);
                    lastSet->finalize();
                    replacementVal = lastSet;
                  } else {
                    // Cannot tee because local type is wider than targetType;
                    // keep lastSet and valGet.
                    localGetCounts[localIdx]++;
                    foldedLastSetIntoVal = false;
                  }
                }
                if (foldedLastSetIntoVal) {
                  *valPtr = replacementVal;
                  if (thenBlock && !thenBlock->name.is() &&
                      !thenBlock->list.empty()) {
                    thenBlock->list.pop_back();
                  } else {
                    iff->ifTrue = builder.makeNop();
                  }
                }
              }
            } else if (sgVal) {
              if (auto* ss = lastStmt->dynCast<StructSet>();
                  ss && ss->index == sgVal->index &&
                  ss->ref->type != Type::unreachable &&
                  ss->value->type != Type::unreachable &&
                  ss->ref->type.getHeapType() ==
                    sgVal->ref->type.getHeapType() &&
                  ExpressionAnalyzer::equal(ss->ref, sgVal->ref) &&
                  Type::isSubType(ss->value->type, lazyTargetType)) {
                Index existingV = func->getNumLocals();
                if (auto* vTee = ss->value->dynCast<LocalSet>();
                    vTee && vTee->isTee()) {
                  Type lt = func->getLocalType(vTee->index);
                  if (Type::isSubType(lt, sgVal->type) &&
                      Type::isSubType(lt, lazyTargetType)) {
                    existingV = vTee->index;
                  }
                } else if (auto* vGet = ss->value->dynCast<LocalGet>()) {
                  Type lt = func->getLocalType(vGet->index);
                  if (Type::isSubType(lt, sgVal->type) &&
                      Type::isSubType(lt, lazyTargetType)) {
                    existingV = vGet->index;
                  }
                }
                if (existingV < func->getNumLocals()) {
                  decrementLocalGets(sgVal);
                  localGetCounts[existingV]++;
                  localsWithSynthesizedGets.insert(existingV);
                  *valPtr = builder.makeLocalGet(
                    existingV, func->getLocalType(existingV));
                } else if (isFuncExitTail) {
                  Index limit =
                    std::min<Index>(func->getNumLocals(), Index(128));
                  for (Index v = 0; v < limit; v++) {
                    Type lt = func->getLocalType(v);
                    if (Type::isSubType(ss->value->type, lt) &&
                        Type::isSubType(lt, sgVal->type) &&
                        Type::isSubType(lt, lazyTargetType)) {
                      ss->value = builder.makeLocalTee(v, ss->value, lt);
                      ss->finalize();
                      decrementLocalGets(sgVal);
                      localGetCounts[v]++;
                      localsWithSynthesizedGets.insert(v);
                      *valPtr = builder.makeLocalGet(v, lt);
                      break;
                    }
                  }
                }
              }
            } else if (ggVal) {
              if (auto* gs = lastStmt->dynCast<GlobalSet>();
                  gs && gs->name == ggVal->name &&
                  gs->value->type != Type::unreachable &&
                  Type::isSubType(gs->value->type, lazyTargetType)) {
                Index existingV = func->getNumLocals();
                if (auto* vTee = gs->value->dynCast<LocalSet>();
                    vTee && vTee->isTee()) {
                  if (Type::isSubType(func->getLocalType(vTee->index),
                                      lazyTargetType)) {
                    existingV = vTee->index;
                  }
                } else if (auto* vGet = gs->value->dynCast<LocalGet>()) {
                  if (Type::isSubType(func->getLocalType(vGet->index),
                                      lazyTargetType)) {
                    existingV = vGet->index;
                  }
                }
                if (existingV < func->getNumLocals()) {
                  localGetCounts[existingV]++;
                  localsWithSynthesizedGets.insert(existingV);
                  *valPtr = builder.makeLocalGet(
                    existingV, func->getLocalType(existingV));
                }
              }
            }

            Name brTarget;
            if (isRetMode) {
              brTarget = ensureFuncExitLabel();
              if (!thenIsUnreachable) {
                nextRet->finalize();
              }
            } else if (isBrMode) {
              brTarget = nextBr->name;
              if (!thenIsUnreachable) {
                nextBr->finalize();
              }
            } else {
              assert(isTailValMode);
              if (!curr->name.is()) {
                curr->name = makeFreshBlockName("lazy$");
                nameToBlock[curr->name] = curr;
              }
              brTarget = curr->name;
            }
            BranchHints::clear(iff, func);
            curr->list[i] = builder.makeBrOn(BrOnNonNull, brTarget, refOperand);
            if (thenIsUnreachable) {
              curr->list.erase(curr->list.begin() + i + 1,
                               curr->list.begin() + i + 2);
            }
            if (auto* tb = iff->ifTrue->dynCast<Block>();
                tb && !tb->name.is()) {
              Index insertPos = i + 1;
              for (auto* item : tb->list) {
                if (!item->is<Nop>()) {
                  curr->list.insertAt(insertPos++, item);
                }
              }
            } else if (!iff->ifTrue->is<Nop>()) {
              curr->list.insertAt(i + 1, iff->ifTrue);
            }
            curr->finalize(targetType);
            changed = true;
          }
        }

        if (!curr->name.is()) {
          return;
        }
        if (!targetType.isRef()) {
          return;
        }

        // Part 3(b): [(drop (br_if $curr FALLBACK (ref.is_null X))), VAL] ->
        // [(br_on_non_null $curr X'), FALLBACK].
        if (!neverUnconditionalize && curr->list.size() >= 2) {
          size_t sz = curr->list.size();
          if (auto* drop = curr->list[sz - 2]->dynCast<Drop>()) {
            auto* br = drop->value->dynCast<Break>();
            Expression* lastVal = curr->list[sz - 1];
            if (br && br->condition && br->value && br->name == curr->name &&
                br->type != Type::unreachable &&
                lastVal->type != Type::unreachable) {
              Expression* refOperand = nullptr;
              LocalSet* tee = nullptr;
              LocalGet* condGet = nullptr;
              LocalGet* valGet = nullptr;
              if (matchRefIsNullGuard(br->condition,
                                      lastVal,
                                      targetType,
                                      refOperand,
                                      tee,
                                      condGet,
                                      valGet) &&
                  !EffectAnalyzer(passOptions, wasm, br->value)
                     .hasUnremovableSideEffects() &&
                  EffectAnalyzer::canReorder(
                    passOptions, wasm, br->value, br->condition)) {
                Expression* ref =
                  stripDeadTeeIfPossible(refOperand, tee, valGet, lastVal);
                Expression* fallback = br->value;
                BranchHints::clear(br, func);
                curr->list[sz - 2] =
                  builder.makeBrOn(BrOnNonNull, curr->name, ref);
                curr->list[sz - 1] = fallback;
                curr->finalize(targetType);
                changed = true;
              }
            }
          }
        }

        if (!targetType.isNullable()) {
          return;
        }
        bool folded = false;
        while (curr->list.size() >= 2) {
          size_t sz = curr->list.size();
          auto* brOn = curr->list[sz - 2]->dynCast<BrOn>();
          if (!brOn || brOn->op != BrOnNonNull || brOn->name != curr->name ||
              brOn->type == Type::unreachable ||
              !Type::isSubType(brOn->ref->type, targetType) ||
              !curr->list[sz - 1]->is<RefNull>()) {
            break;
          }
          curr->list[sz - 2] = brOn->ref;
          curr->list.pop_back();
          folded = true;
          changed = true;
        }
        if (folded) {
          if (!BranchUtils::BranchSeeker::has(curr, curr->name)) {
            curr->name = Name();
            if (curr->list.size() == 1) {
              replaceCurrent(curr->list[0]);
              return;
            }
          }
          curr->finalize(targetType);
        }
      }
    } lowerer(func,
              *getModule(),
              getPassOptions(),
              builder,
              nameToBlock,
              localGetCounts,
              localsWithSynthesizedGets,
              localGraph,
              ensureFuncExitLabel,
              makeFreshBlockName,
              decrementLocalGets,
              matchRefIsNullGuard,
              matchNonNullGuard,
              stripDeadTeeIfPossible,
              allowFuncExit,
              neverUnconditionalize);

    // Walk func->body (and if ensureFuncExitLabel wraps func->body during the
    // walk, func->body is updated after the root completes or during the walk).
    Expression* root = func->body;
    lowerer.walk(root);
    if (func->body != root && funcExitName.is()) {
      // ensureFuncExitLabel wrapped the old func->body in a new Block* wrapper.
      auto* wrapper = func->body->cast<Block>();
      wrapper->list[0] = root;
      wrapper->finalize(funcResults);
    } else {
      func->body = root;
    }

    // Short-Circuit 2-Arm If (result T) Chain Rotation, Shared-Arm Collapsing,
    // and 1-Arm Void `||` Guard Lowering (Idea #104).
    if (allowFuncExit && func->body) {
      auto blockTypeBytes = [](Type t) -> int {
        if (t == Type::none || t == Type::unreachable) {
          return 1;
        }
        if (t.isRef() && (!t.isNullable() || !t.getHeapType().isBasic())) {
          return 2;
        }
        return 1;
      };
      auto canNegateFree = [](Expression* expr) -> bool {
        return canInvertBoolExprFree(expr) || canPushEqzIntoIfOrExprFree(expr);
      };
      auto posEqzCost = [&](const ShortCircuitStep& s) -> int {
        return (s.negated && !canNegateFree(s.cond)) ? 1 : 0;
      };
      auto negEqzCost = [&](const ShortCircuitStep& s) -> int {
        return (!s.negated && !canNegateFree(s.cond)) ? 1 : 0;
      };
      auto makeNegated = [&](Expression* expr) -> Expression* {
        if (canInvertBoolExprFree(expr)) {
          invertBoolExprFree(expr, builder);
          return expr;
        }
        if (canPushEqzIntoIfOrExprFree(expr)) {
          pushEqzIntoIfOrExprFree(expr, builder);
          return expr;
        }
        return makeNegatedBool(builder, expr, true);
      };
      auto materializePosCond = [&](const ShortCircuitStep& s) -> Expression* {
        return s.negated ? makeNegated(s.cond) : s.cond;
      };
      auto materializeNegCond = [&](const ShortCircuitStep& s) -> Expression* {
        return s.negated ? s.cond : makeNegated(s.cond);
      };
      auto materializePosOneBit =
        [&](const ShortCircuitStep& s) -> Expression* {
        return s.negated ? makeNegated(s.cond)
                         : makeOneBitCond(builder, s.cond);
      };
      auto materializeNegOneBit =
        [&](const ShortCircuitStep& s) -> Expression* {
        return s.negated ? makeOneBitCond(builder, s.cond)
                         : makeNegated(s.cond);
      };
      auto appendPrefixAndStmt = [&](Block* dst,
                                     const std::vector<Expression*>& prefix,
                                     Expression* stmt) {
        for (auto* p : prefix) {
          if (!p->is<Nop>()) {
            dst->list.push_back(p);
          }
        }
        if (stmt && !stmt->is<Nop>()) {
          if (auto* b = stmt->dynCast<Block>(); b && !b->name.is()) {
            for (auto* item : b->list) {
              if (!item->is<Nop>()) {
                dst->list.push_back(item);
              }
            }
          } else {
            dst->list.push_back(stmt);
          }
        }
      };

      struct ChainPlan {
        bool valid = false;
        bool flipInitial = false;
        int deltaBytes = 0;
        std::vector<ShortCircuitStep> steps;
        std::vector<Expression*> passPrefix;
        std::vector<If*> outerIfs;
        std::vector<If*> innerIfs;
        Expression* passArm = nullptr;
        Expression* shortArm = nullptr;
        bool foldLastConstIntoPass = false;
        Index numTrailingAndOrFolds = 0;
        bool foldAllToAndOr = false;
      };

      auto evaluateChainPlan = [&](If* rootIf,
                                   bool flipInitial,
                                   bool isReturnNode,
                                   bool isVoidTail,
                                   bool isFuncReturnTail,
                                   Type elidableTailType,
                                   Index tryDepth) -> ChainPlan {
        ChainPlan plan;
        plan.flipInitial = flipInitial;
        Expression* shortArm = flipInitial ? rootIf->ifTrue : rootIf->ifFalse;
        bool shortArmPure =
          !EffectAnalyzer(getPassOptions(), *getModule(), shortArm)
             .hasSideEffects();
        int kEqz = 0;
        If* curIf = rootIf;
        bool curFlip = flipInitial;
        Expression* curPass = nullptr;
        std::vector<Expression*> trailingPrefix;
        while (true) {
          plan.outerIfs.push_back(curIf);
          collectShortCircuitConjuncts(curIf->condition,
                                       curFlip,
                                       getPassOptions(),
                                       *getModule(),
                                       plan.steps,
                                       plan.innerIfs,
                                       kEqz,
                                       trailingPrefix);
          curPass = curFlip ? curIf->ifFalse : curIf->ifTrue;
          if (!shortArmPure || !trailingPrefix.empty()) {
            break;
          }
          auto* nextIf = curPass->dynCast<If>();
          if (!nextIf || !nextIf->ifFalse ||
              nextIf->condition->type == Type::unreachable) {
            break;
          }
          if (ExpressionAnalyzer::equal(nextIf->ifFalse, shortArm)) {
            curIf = nextIf;
            curFlip = false;
          } else if (ExpressionAnalyzer::equal(nextIf->ifTrue, shortArm)) {
            curIf = nextIf;
            curFlip = true;
          } else {
            break;
          }
        }
        plan.passPrefix = std::move(trailingPrefix);
        plan.passArm = curPass;
        plan.shortArm = shortArm;
        int mOuter = int(plan.outerIfs.size());
        int kInner = int(plan.innerIfs.size());
        if (mOuter + kInner < 2 || plan.steps.empty() ||
            (plan.steps.size() == 1 && plan.passPrefix.empty())) {
          return plan;
        }
        int bt = blockTypeBytes(rootIf->type);
        int shortBytes = (mOuter > 1) ? estimateMinBinaryBytes(shortArm) : 0;
        int origBytes = (isReturnNode ? 1 : 0) + mOuter * (3 + bt) +
                        (mOuter - 1) * shortBytes + 6 * kInner + kEqz;

        size_t remN = plan.steps.size();
        int extraPassBytes = 0;
        int32_t shortConst = 0, passConst = 0;
        bool isBoolShort =
          (rootIf->type == Type::i32 && isI32Const(shortArm, shortConst) &&
           (shortConst == 0 || shortConst == 1));
        bool passIsOneBitPureSmall = false;
        if (isBoolShort && plan.passPrefix.empty() && remN >= 2) {
          if (isI32Const(plan.passArm, passConst) &&
              passConst == (1 - shortConst)) {
            const ShortCircuitStep& sLast = plan.steps[remN - 1];
            int convCost = 0;
            if (shortConst == 0) {
              convCost = !sLast.negated ? (isOneBitI32(sLast.cond) ? 0 : 3)
                                        : posEqzCost(sLast);
            } else {
              convCost = sLast.negated ? (isOneBitI32(sLast.cond) ? 0 : 3)
                                       : negEqzCost(sLast);
            }
            plan.foldLastConstIntoPass = true;
            remN--;
            extraPassBytes += convCost - 2;
            passIsOneBitPureSmall =
              !neverUnconditionalize && sLast.prefix.empty() &&
              !EffectAnalyzer(getPassOptions(), *getModule(), sLast.cond)
                 .hasSideEffects() &&
              !tooCostlyToRunUnconditionally(getPassOptions(), sLast.cond);
          } else {
            passIsOneBitPureSmall =
              !neverUnconditionalize && isOneBitI32(plan.passArm) &&
              !EffectAnalyzer(getPassOptions(), *getModule(), plan.passArm)
                 .hasSideEffects() &&
              !tooCostlyToRunUnconditionally(getPassOptions(), plan.passArm);
          }
          bool lastFoldedSmallEnoughForS0 =
            (plan.foldLastConstIntoPass &&
             estimateMinBinaryBytes(plan.steps[remN].cond) <= 4) ||
            (!plan.foldLastConstIntoPass &&
             estimateMinBinaryBytes(plan.passArm) <= 4);
          while (passIsOneBitPureSmall && remN >= 2) {
            const ShortCircuitStep& sPrev = plan.steps[remN - 1];
            bool canOneBit = (shortConst == 0)
                               ? (sPrev.negated || isOneBitI32(sPrev.cond))
                               : (!sPrev.negated || isOneBitI32(sPrev.cond));
            if (!canOneBit) {
              break;
            }
            int eqzC =
              (shortConst == 0) ? posEqzCost(sPrev) : negEqzCost(sPrev);
            plan.numTrailingAndOrFolds++;
            remN--;
            extraPassBytes += 1 + eqzC;
            bool sPrevPure =
              sPrev.prefix.empty() &&
              !EffectAnalyzer(getPassOptions(), *getModule(), sPrev.cond)
                 .hasSideEffects();
            lastFoldedSmallEnoughForS0 =
              sPrevPure && estimateMinBinaryBytes(sPrev.cond) <= 4;
            passIsOneBitPureSmall =
              sPrevPure &&
              !tooCostlyToRunUnconditionally(getPassOptions(), sPrev.cond);
          }
          if (passIsOneBitPureSmall && lastFoldedSmallEnoughForS0 &&
              remN == 1 && shortArmPure) {
            const ShortCircuitStep& s0 = plan.steps[0];
            bool s0OneBit = (shortConst == 0)
                              ? (s0.negated || isOneBitI32(s0.cond))
                              : (!s0.negated || isOneBitI32(s0.cond));
            if (s0OneBit) {
              plan.foldAllToAndOr = true;
            }
          }
        }

        int newBytes = 0;
        if (plan.foldAllToAndOr) {
          assert(remN == 1);
          int eqz0 = (shortConst == 0) ? posEqzCost(plan.steps[0])
                                       : negEqzCost(plan.steps[0]);
          // Eliminates the outer If (3 + bt) AND the remaining shortArm (2 B),
          // replacing them with 1-byte i32.and/i32.or + eqz0.
          newBytes = (isReturnNode ? 1 : 0) + 1 + eqz0 + extraPassBytes - 2;
        } else if (remN == 1) {
          newBytes = (isReturnNode ? 1 : 0) + (3 + bt) +
                     posEqzCost(plan.steps[0]) + extraPassBytes;
        } else {
          int exitBytes = 0;
          if (isReturnNode) {
            int passRet =
              (!plan.foldLastConstIntoPass &&
               plan.numTrailingAndOrFolds == 0 &&
               plan.passArm->type == Type::unreachable)
                ? 0
                : 1;
            int shortRet = (plan.shortArm->type == Type::unreachable) ? 0 : 1;
            exitBytes = passRet + shortRet;
          } else if (!plan.foldLastConstIntoPass &&
                     plan.numTrailingAndOrFolds == 0 &&
                     plan.passArm->type == Type::unreachable) {
            exitBytes = 0;
          } else if (isFuncReturnTail && tryDepth == 0 &&
                     Type::isSubType(rootIf->type, funcResults)) {
            exitBytes = 1;
          } else if ((rootIf->type == Type::none && isVoidTail) ||
                     (elidableTailType != Type::unreachable &&
                      elidableTailType != Type::none &&
                      rootIf->type != Type::none &&
                      elidableTailType.asWrittenGivenFeatures(
                        getModule()->features) ==
                        rootIf->type.asWrittenGivenFeatures(
                          getModule()->features))) {
            exitBytes = 2;
          } else {
            exitBytes = 4 + bt;
          }
          newBytes = 3 + 2 * (int(remN) - 1) + exitBytes +
                     posEqzCost(plan.steps[0]) + extraPassBytes;
          for (size_t i = 1; i < remN; i++) {
            newBytes += negEqzCost(plan.steps[i]);
            if (plan.steps[i].cond->is<If>()) {
              std::vector<ShortCircuitStep> subSteps;
              std::vector<If*> subCollapsedIfs;
              std::vector<Expression*> subTrailing;
              int subEqz = 0;
              collectShortCircuitConjuncts(plan.steps[i].cond,
                                           !plan.steps[i].negated,
                                           getPassOptions(),
                                           *getModule(),
                                           subSteps,
                                           subCollapsedIfs,
                                           subEqz,
                                           subTrailing);
              if (!subCollapsedIfs.empty() && subTrailing.empty()) {
                newBytes -= 3 * int(subCollapsedIfs.size());
              }
            }
          }
        }
        plan.deltaBytes = newBytes - origBytes;
        plan.valid = (plan.deltaBytes < 0);
        return plan;
      };

      auto materializeChainPlan = [&](const ChainPlan& plan,
                                      If* rootIf,
                                      bool isReturnNode,
                                      bool isFuncReturnTail,
                                      Index tryDepth) -> Expression* {
        for (auto* iff : plan.outerIfs) {
          BranchHints::clear(iff, func);
        }
        for (auto* iff : plan.innerIfs) {
          BranchHints::clear(iff, func);
        }
        int32_t shortConst = 0;
        isI32Const(plan.shortArm, shortConst);
        size_t remN = plan.steps.size();
        Expression* newPassArm =
          prependToArm(builder, plan.passPrefix, plan.passArm);
        if (plan.foldLastConstIntoPass) {
          const ShortCircuitStep& sLast = plan.steps[remN - 1];
          remN--;
          Expression* oneBit = (shortConst == 0) ? materializePosOneBit(sLast)
                                                 : materializeNegOneBit(sLast);
          newPassArm = prependToArm(builder, sLast.prefix, oneBit);
        }
        for (Index f = 0; f < plan.numTrailingAndOrFolds; f++) {
          const ShortCircuitStep& sPrev = plan.steps[remN - 1];
          remN--;
          Expression* lhs = (shortConst == 0) ? materializePosOneBit(sPrev)
                                              : materializeNegOneBit(sPrev);
          BinaryOp op = (shortConst == 0) ? AndInt32 : OrInt32;
          Expression* comb = builder.makeBinary(op, lhs, newPassArm);
          newPassArm = prependToArm(builder, sPrev.prefix, comb);
        }
        if (plan.foldAllToAndOr) {
          assert(remN == 1);
          const ShortCircuitStep& s0 = plan.steps[0];
          Expression* lhs = (shortConst == 0) ? materializePosOneBit(s0)
                                              : materializeNegOneBit(s0);
          BinaryOp op = (shortConst == 0) ? AndInt32 : OrInt32;
          Expression* comb = builder.makeBinary(op, lhs, newPassArm);
          Expression* res = prependToArm(builder, s0.prefix, comb);
          return isReturnNode ? builder.makeReturn(res) : res;
        }
        if (remN == 1) {
          const ShortCircuitStep& s0 = plan.steps[0];
          Expression* cond0 = materializePosCond(s0);
          auto* singleIf =
            builder.makeIf(cond0, newPassArm, plan.shortArm, rootIf->type);
          Expression* res = prependToArm(builder, s0.prefix, singleIf);
          return isReturnNode ? builder.makeReturn(res) : res;
        }

        // Hybrid Form for remN >= 2:
        Name failName = makeFreshBlockName("sc_fail$");
        auto* failBlk = builder.makeBlock(failName, {}, Type::none);
        for (size_t i = 1; i < remN; i++) {
          const ShortCircuitStep& si = plan.steps[i];
          Expression* negC = materializeNegCond(si);
          auto* brFail = builder.makeBreak(failName, nullptr, negC);
          Expression* failStmt = brFail;
          if (auto* foldedBr = tryFoldBooleanBreakCondition(
                brFail,
                func,
                *getModule(),
                getPassOptions(),
                neverUnconditionalize,
                nullptr)) {
            failStmt = foldedBr;
            if (auto* foldedIf = failStmt->dynCast<If>()) {
              while (tryFoldBooleanIfCondition(foldedIf,
                                               func,
                                               *getModule(),
                                               getPassOptions(),
                                               neverUnconditionalize)) {
              }
            }
          }
          appendPrefixAndStmt(failBlk, si.prefix, failStmt);
        }

        bool useReturnExit =
          isReturnNode ||
          (isFuncReturnTail && tryDepth == 0 &&
           Type::isSubType(newPassArm->type, funcResults));
        bool passUnreach = (newPassArm->type == Type::unreachable);
        bool passVoid = (newPassArm->type == Type::none);
        Name outName;
        if (passUnreach) {
          appendPrefixAndStmt(failBlk, {}, newPassArm);
        } else if (useReturnExit) {
          if (passVoid) {
            appendPrefixAndStmt(failBlk, {}, newPassArm);
            appendPrefixAndStmt(failBlk, {}, builder.makeReturn());
          } else {
            appendPrefixAndStmt(failBlk, {}, builder.makeReturn(newPassArm));
          }
        } else {
          outName = makeFreshBlockName("sc_out$");
          if (passVoid) {
            appendPrefixAndStmt(failBlk, {}, newPassArm);
            appendPrefixAndStmt(failBlk, {}, builder.makeBreak(outName));
          } else {
            appendPrefixAndStmt(
              failBlk, {}, builder.makeBreak(outName, newPassArm));
          }
        }
        failBlk->finalize(Type::none);
        nameToBlock[failName] = failBlk;

        const ShortCircuitStep& s0 = plan.steps[0];
        auto* outerVoidIf =
          builder.makeIf(materializePosCond(s0), failBlk, nullptr, Type::none);

        Expression* tailExpr = plan.shortArm;
        if (isReturnNode && tailExpr->type != Type::unreachable) {
          tailExpr = builder.makeReturn(tailExpr);
        }

        Type wrapperType = isReturnNode ? Type::unreachable : rootIf->type;
        auto* outBlk = builder.makeBlock(outName, {}, wrapperType);
        appendPrefixAndStmt(outBlk, s0.prefix, outerVoidIf);
        appendPrefixAndStmt(outBlk, {}, tailExpr);
        outBlk->finalize(wrapperType);
        if (outName.is()) {
          nameToBlock[outName] = outBlk;
        }
        return outBlk;
      };

      std::function<void(Expression**, bool, bool, Type, Index)>
        walkShortCircuitChains = [&](Expression** currp,
                                     bool isVoidTail,
                                     bool isFuncReturnTail,
                                     Type elidableTailType,
                                     Index tryDepth) {
          Expression* curr = *currp;
          if (!curr) {
            return;
          }

          // Case R: (return (if (result T) ...)) outside try/try_table.
          if (auto* ret = curr->dynCast<Return>()) {
            if (tryDepth == 0 && ret->value && funcResults.isConcrete()) {
              if (auto* rootIf = ret->value->dynCast<If>();
                  rootIf && rootIf->ifFalse && rootIf->type.isConcrete() &&
                  rootIf->condition->type != Type::unreachable) {
                ChainPlan pFalse = evaluateChainPlan(rootIf,
                                                     /*flipInitial=*/false,
                                                     /*isReturnNode=*/true,
                                                     /*isVoidTail=*/false,
                                                     /*isFuncReturnTail=*/true,
                                                     funcResults,
                                                     tryDepth);
                ChainPlan pTrue = evaluateChainPlan(rootIf,
                                                    /*flipInitial=*/true,
                                                    /*isReturnNode=*/true,
                                                    /*isVoidTail=*/false,
                                                    /*isFuncReturnTail=*/true,
                                                    funcResults,
                                                    tryDepth);
                const ChainPlan* best = nullptr;
                if (pFalse.valid && pTrue.valid) {
                  best = (pTrue.deltaBytes < pFalse.deltaBytes) ? &pTrue
                                                                : &pFalse;
                } else if (pFalse.valid) {
                  best = &pFalse;
                } else if (pTrue.valid) {
                  best = &pTrue;
                }
                if (best) {
                  *currp = materializeChainPlan(*best,
                                                rootIf,
                                                /*isReturnNode=*/true,
                                                /*isFuncReturnTail=*/true,
                                                tryDepth);
                  lowerer.changed = true;
                  curr = *currp;
                }
              }
            }
          } else if (auto* iff = curr->dynCast<If>()) {
            if (iff->condition->type == Type::unreachable) {
              return;
            }
            if (!iff->ifFalse && iff->type == Type::none) {
              // Part 3: 1-arm void `||` guard lowering.
              std::vector<ShortCircuitStep> disjSteps;
              std::vector<If*> collapsedIfs;
              std::vector<Expression*> disjTrailing;
              int kEqz = 0;
              collectShortCircuitConjuncts(iff->condition,
                                           /*negated=*/true,
                                           getPassOptions(),
                                           *getModule(),
                                           disjSteps,
                                           collapsedIfs,
                                           kEqz,
                                           disjTrailing);
              if (disjSteps.size() >= 2 && disjTrailing.empty()) {
                ShortCircuitStep dLast = std::move(disjSteps.back());
                disjSteps.pop_back();
                std::vector<ShortCircuitStep> lastConjSteps;
                std::vector<Expression*> conjTrailing;
                collectShortCircuitConjuncts(dLast.cond,
                                             /*negated=*/!dLast.negated,
                                             getPassOptions(),
                                             *getModule(),
                                             lastConjSteps,
                                             collapsedIfs,
                                             kEqz,
                                             conjTrailing);
                if (!dLast.prefix.empty()) {
                  lastConjSteps[0].prefix.insert(
                    lastConjSteps[0].prefix.begin(),
                    dLast.prefix.begin(),
                    dLast.prefix.end());
                }
                int kIf = int(collapsedIfs.size());
                int m = int(lastConjSteps.size());
                int origBytes = 3 + 6 * kIf + kEqz;
                int newBytes = (isVoidTail ? 0 : 3) + 3 + 2 * m;
                for (const auto& g : lastConjSteps) {
                  newBytes += negEqzCost(g);
                }
                if (disjSteps.size() == 1) {
                  newBytes += posEqzCost(disjSteps[0]);
                } else {
                  newBytes += 2 * int(disjSteps.size());
                  for (const auto& d : disjSteps) {
                    newBytes += negEqzCost(d);
                  }
                }
                if (newBytes < origBytes) {
                  BranchHints::clear(iff, func);
                  for (auto* cIf : collapsedIfs) {
                    BranchHints::clear(cIf, func);
                  }
                  Name outName = makeFreshBlockName("sc_out$");
                  auto* outBlk = builder.makeBlock(outName, {}, Type::none);
                  if (disjSteps.size() == 1) {
                    const ShortCircuitStep& d1 = disjSteps[0];
                    auto* failChecks = builder.makeBlock();
                    for (const auto& g : lastConjSteps) {
                      auto* brOut = builder.makeBreak(
                        outName, nullptr, materializeNegCond(g));
                      appendPrefixAndStmt(failChecks, g.prefix, brOut);
                    }
                    failChecks->finalize();
                    Expression* guardBody = (failChecks->list.size() == 1)
                                              ? failChecks->list[0]
                                              : failChecks;
                    auto* guardIf = builder.makeIf(
                      materializePosCond(d1), guardBody, nullptr, Type::none);
                    appendPrefixAndStmt(outBlk, d1.prefix, guardIf);
                  } else {
                    Name doName = makeFreshBlockName("sc_do$");
                    auto* doBlk = builder.makeBlock(doName, {}, Type::none);
                    for (const auto& d : disjSteps) {
                      auto* brDo = builder.makeBreak(
                        doName, nullptr, materializeNegCond(d));
                      appendPrefixAndStmt(doBlk, d.prefix, brDo);
                    }
                    for (const auto& g : lastConjSteps) {
                      auto* brOut = builder.makeBreak(
                        outName, nullptr, materializeNegCond(g));
                      appendPrefixAndStmt(doBlk, g.prefix, brOut);
                    }
                    doBlk->finalize(Type::none);
                    nameToBlock[doName] = doBlk;
                    outBlk->list.push_back(doBlk);
                  }
                  appendPrefixAndStmt(outBlk, conjTrailing, iff->ifTrue);
                  outBlk->finalize(Type::none);
                  nameToBlock[outName] = outBlk;
                  *currp = outBlk;
                  lowerer.changed = true;
                  curr = *currp;
                }
              }
            } else if (iff->ifFalse && iff->type != Type::unreachable) {
              // Parts 1 & 2: 2-arm If chain rotation & collapsing.
              ChainPlan pFalse = evaluateChainPlan(iff,
                                                   /*flipInitial=*/false,
                                                   /*isReturnNode=*/false,
                                                   isVoidTail,
                                                   isFuncReturnTail,
                                                   elidableTailType,
                                                   tryDepth);
              ChainPlan pTrue = evaluateChainPlan(iff,
                                                  /*flipInitial=*/true,
                                                  /*isReturnNode=*/false,
                                                  isVoidTail,
                                                  isFuncReturnTail,
                                                  elidableTailType,
                                                  tryDepth);
              const ChainPlan* best = nullptr;
              if (pFalse.valid && pTrue.valid) {
                best = (pTrue.deltaBytes < pFalse.deltaBytes) ? &pTrue
                                                              : &pFalse;
              } else if (pFalse.valid) {
                best = &pFalse;
              } else if (pTrue.valid) {
                best = &pTrue;
              }
              if (best) {
                *currp = materializeChainPlan(*best,
                                              iff,
                                              /*isReturnNode=*/false,
                                              isFuncReturnTail,
                                              tryDepth);
                lowerer.changed = true;
                curr = *currp;
              }
            }
          }

          // Recurse into children of curr.
          if (curr->is<Try>() || curr->is<TryTable>()) {
            for (auto*& child : ChildIterator(curr)) {
              walkShortCircuitChains(
                &child, false, false, Type::unreachable, tryDepth + 1);
            }
            return;
          }
          if (auto* iff = curr->dynCast<If>()) {
            if (iff->condition->type == Type::unreachable) {
              return;
            }
            walkShortCircuitChains(
              &iff->condition, false, false, Type::unreachable, tryDepth);
            bool armVoidTail = (iff->type == Type::none);
            bool armFuncReturnTail =
              isFuncReturnTail &&
              (iff->type == funcResults || iff->type == Type::unreachable);
            Type armElidableType = iff->type;
            walkShortCircuitChains(&iff->ifTrue,
                                   armVoidTail &&
                                     iff->ifTrue->type == Type::none,
                                   armFuncReturnTail,
                                   armElidableType,
                                   tryDepth);
            if (iff->ifFalse) {
              walkShortCircuitChains(&iff->ifFalse,
                                     armVoidTail &&
                                       iff->ifFalse->type == Type::none,
                                     armFuncReturnTail,
                                     armElidableType,
                                     tryDepth);
            }
            return;
          }
          if (auto* blk = curr->dynCast<Block>()) {
            if (blk->list.empty()) {
              return;
            }
            bool hasEarlierUnreachable = false;
            for (Index i = 0; i + 1 < blk->list.size(); i++) {
              if (!hasEarlierUnreachable) {
                if (auto* iff = blk->list[i]->dynCast<If>();
                    iff && !iff->ifFalse && iff->type == Type::none &&
                    iff->condition->type != Type::unreachable) {
                  bool tailMidVoid = true;
                  for (Index j = i + 1; j < blk->list.size(); j++) {
                    if (j + 1 < blk->list.size() &&
                        blk->list[j]->type != Type::none) {
                      tailMidVoid = false;
                      break;
                    }
                    if (blk->name.is() &&
                        BranchUtils::BranchSeeker::has(blk->list[j],
                                                       blk->name)) {
                      tailMidVoid = false;
                      break;
                    }
                  }
                  if (tailMidVoid) {
                    bool isModeU =
                      (iff->ifTrue->type == Type::unreachable &&
                       blk->list.back()->type == Type::unreachable);
                    bool isModeR = false;
                    std::vector<Expression*> thenRetPrefix;
                    Expression* thenRetVal = nullptr;
                    if (!isModeU && isFuncReturnTail && tryDepth == 0 &&
                        funcResults.isConcrete() &&
                        blk->type != Type::unreachable &&
                        blk->list.back()->type != Type::unreachable &&
                        Type::isSubType(blk->list.back()->type, funcResults)) {
                      Return* ret1 = iff->ifTrue->dynCast<Return>();
                      if (!ret1) {
                        if (auto* tb = iff->ifTrue->dynCast<Block>();
                            tb && !tb->name.is() && !tb->list.empty()) {
                          if (auto* r = tb->list.back()->dynCast<Return>()) {
                            bool ok = true;
                            for (Index j = 0; j + 1 < tb->list.size(); j++) {
                              if (tb->list[j]->type != Type::none) {
                                ok = false;
                                break;
                              }
                              thenRetPrefix.push_back(tb->list[j]);
                            }
                            if (ok) {
                              ret1 = r;
                            } else {
                              thenRetPrefix.clear();
                            }
                          }
                        }
                      }
                      if (ret1 && ret1->value &&
                          ret1->value->type != Type::unreachable &&
                          Type::isSubType(ret1->value->type, blk->type) &&
                          Type::isSubType(ret1->value->type, funcResults)) {
                        isModeR = true;
                        thenRetVal = ret1->value;
                      }
                    }
                    if (isModeU || isModeR) {
                      std::vector<ShortCircuitStep> disjSteps;
                      std::vector<If*> collapsedIfs;
                      std::vector<Expression*> disjTrailing;
                      int kEqz = 0;
                      collectShortCircuitConjuncts(iff->condition,
                                                   /*negated=*/true,
                                                   getPassOptions(),
                                                   *getModule(),
                                                   disjSteps,
                                                   collapsedIfs,
                                                   kEqz,
                                                   disjTrailing);
                      if (disjSteps.size() >= 2 && disjTrailing.empty()) {
                        int kIf = int(collapsedIfs.size());
                        int origBytes = 3 + 6 * kIf + kEqz;
                        int newBytes = 3 + 2 * (int(disjSteps.size()) - 1) +
                                       posEqzCost(disjSteps[0]);
                        for (size_t m = 1; m < disjSteps.size(); m++) {
                          newBytes += negEqzCost(disjSteps[m]);
                        }
                        if (newBytes < origBytes) {
                          BranchHints::clear(iff, func);
                          for (auto* cIf : collapsedIfs) {
                            BranchHints::clear(cIf, func);
                          }
                          Name failName = makeFreshBlockName("sc_fail$");
                          auto* failBlk =
                            builder.makeBlock(failName, {}, Type::none);
                          for (size_t m = 1; m < disjSteps.size(); m++) {
                            const ShortCircuitStep& dm = disjSteps[m];
                            auto* brFail = builder.makeBreak(
                              failName, nullptr, materializeNegCond(dm));
                            Expression* failStmt = brFail;
                            if (auto* foldedBr =
                                  tryFoldBooleanBreakCondition(
                                    brFail,
                                    func,
                                    *getModule(),
                                    getPassOptions(),
                                    neverUnconditionalize,
                                    nullptr)) {
                              failStmt = foldedBr;
                              if (auto* foldedIf = failStmt->dynCast<If>()) {
                                while (tryFoldBooleanIfCondition(
                                  foldedIf,
                                  func,
                                  *getModule(),
                                  getPassOptions(),
                                  neverUnconditionalize)) {
                                }
                              }
                            }
                            appendPrefixAndStmt(failBlk, dm.prefix, failStmt);
                          }
                          for (Index j = i + 1; j < blk->list.size(); j++) {
                            Expression* restItem = blk->list[j];
                            if (isModeR && j + 1 == blk->list.size()) {
                              restItem = builder.makeReturn(restItem);
                            }
                            appendPrefixAndStmt(failBlk, {}, restItem);
                          }
                          failBlk->finalize(Type::none);
                          nameToBlock[failName] = failBlk;
                          const ShortCircuitStep& d0 = disjSteps[0];
                          auto* outerVoidIf = builder.makeIf(
                            materializePosCond(d0), failBlk, nullptr, Type::none);
                          blk->list.resize(i);
                          appendPrefixAndStmt(blk, d0.prefix, outerVoidIf);
                          if (isModeR) {
                            appendPrefixAndStmt(blk, thenRetPrefix, thenRetVal);
                          } else {
                            appendPrefixAndStmt(blk, {}, iff->ifTrue);
                          }
                          blk->finalize(blk->type);
                          lowerer.changed = true;
                          walkShortCircuitChains(currp,
                                                 isVoidTail,
                                                 isFuncReturnTail,
                                                 elidableTailType,
                                                 tryDepth);
                          return;
                        }
                      }
                    }
                  }
                }
              }
              if (blk->list[i]->type == Type::unreachable) {
                hasEarlierUnreachable = true;
              }
              walkShortCircuitChains(
                &blk->list[i], false, false, Type::unreachable, tryDepth);
            }
            if (hasEarlierUnreachable) {
              walkShortCircuitChains(&blk->list.back(),
                                     false,
                                     false,
                                     Type::unreachable,
                                     tryDepth);
              return;
            }
            bool isTargeted =
              blk->name.is() && BranchUtils::BranchSeeker::has(blk, blk->name);
            bool backIsVoidTail =
              blk->type == Type::none && blk->list.back()->type == Type::none &&
              (isVoidTail || isTargeted);
            bool backIsFuncReturnTail =
              isFuncReturnTail &&
              (blk->type == funcResults || blk->type == Type::unreachable);
            Type backElidableType = isTargeted ? blk->type : elidableTailType;
            walkShortCircuitChains(&blk->list.back(),
                                   backIsVoidTail,
                                   backIsFuncReturnTail,
                                   backElidableType,
                                   tryDepth);
            return;
          }
          for (auto*& child : ChildIterator(curr)) {
            walkShortCircuitChains(
              &child, false, false, Type::unreachable, tryDepth);
          }
        };

      walkShortCircuitChains(&func->body,
                             funcResults == Type::none,
                             /*isFuncReturnTail=*/true,
                             funcResults,
                             /*tryDepth=*/0);
    }

    // Part 1: Tail 1-arm void If -> elided (block $b (br_if $b !C) BODY)
    // at StackIR-elidable void tail positions (1(a) void function body,
    // 1(b.1/b.2) void If arm, 1(c) targeted void Block tail).
    if (allowFuncExit && func->body) {
      std::function<void(Expression**, bool)> walkVoidTails =
        [&](Expression** currp, bool isVoidTail) {
          Expression* curr = *currp;
          if (!curr) {
            return;
          }
          if (auto* iff = curr->dynCast<If>()) {
            if (iff->condition->type == Type::unreachable) {
              return;
            }
            bool isBareVoidReturn =
              iff->ifTrue->is<Return>() && !iff->ifTrue->cast<Return>()->value;
            if (isVoidTail && !iff->ifFalse && iff->type == Type::none &&
                !iff->ifTrue->is<Nop>() && !iff->ifTrue->is<Break>() &&
                !isBareVoidReturn) {
              Expression* negCond = nullptr;
              if (auto* u = iff->condition->dynCast<Unary>();
                  u && u->op == EqZInt32 && u->value->type == Type::i32) {
                negCond = u->value;
              } else if (canInvertBoolExprFree(iff->condition)) {
                negCond = iff->condition;
                invertBoolExprFree(negCond, builder);
              } else if (canPushEqzIntoIfOrExprFree(iff->condition)) {
                negCond = iff->condition;
                pushEqzIntoIfOrExprFree(negCond, builder);
              }
              if (negCond) {
                Name bName = makeFreshBlockName("tail_if$");
                auto* brIf = builder.makeBreak(bName, nullptr, negCond);
                BranchHints::copyFlippedTo(iff, brIf, func);
                BranchHints::clear(iff, func);
                auto* blk = builder.makeBlock(bName, {brIf}, Type::none);
                if (auto* tb = iff->ifTrue->dynCast<Block>();
                    tb && !tb->name.is()) {
                  for (auto* item : tb->list) {
                    blk->list.push_back(item);
                  }
                } else {
                  blk->list.push_back(iff->ifTrue);
                }
                blk->finalize(Type::none);
                nameToBlock[bName] = blk;
                *currp = blk;
                lowerer.changed = true;
                walkVoidTails(currp, true);
                return;
              }
            }
            bool armVoidTail = (iff->type == Type::none);
            walkVoidTails(&iff->ifTrue,
                          armVoidTail && iff->ifTrue->type == Type::none);
            if (iff->ifFalse) {
              walkVoidTails(&iff->ifFalse,
                            armVoidTail && iff->ifFalse->type == Type::none);
            }
            return;
          }
          if (auto* blk = curr->dynCast<Block>()) {
            if (blk->list.empty()) {
              return;
            }
            bool hasEarlierUnreachable = false;
            for (Index i = 0; i + 1 < blk->list.size(); i++) {
              if (blk->list[i]->type == Type::unreachable) {
                hasEarlierUnreachable = true;
              }
              walkVoidTails(&blk->list[i], false);
            }
            bool backIsVoidTail =
              !hasEarlierUnreachable && blk->type == Type::none &&
              blk->list.back()->type == Type::none &&
              (isVoidTail ||
               (blk->name.is() &&
                BranchUtils::BranchSeeker::has(blk, blk->name)));
            walkVoidTails(&blk->list.back(), backIsVoidTail);
            return;
          }
          for (auto*& child : ChildIterator(curr)) {
            walkVoidTails(&child, false);
          }
        };
      walkVoidTails(&func->body, funcResults == Type::none);
    }

    if (lowerer.changed) {
      ReFinalize().walkFunctionInModule(func, getModule());
      if (funcExitName.is()) {
        if (auto* bodyBlock = func->body->dynCast<Block>();
            bodyBlock && bodyBlock->name == funcExitName &&
            bodyBlock->type != Type::unreachable) {
          bodyBlock->type = funcResults;
        }
      }
    }
  }

  void doWalkFunction(Function* func) {
    auto instArg = getArgumentOrDefault("remove-unused-brs", "");
    if (instArg == "v4") {
      currentPassV4OverrideSet = true;
      currentPassV4Override = true;
    } else if (instArg == "v3") {
      currentPassV4OverrideSet = true;
      currentPassV4Override = false;
    } else {
      currentPassV4OverrideSet = false;
    }
    neverUnconditionalize =
      hasArgument("remove-unused-brs-never-unconditionalize");
    bool foldBoolIf = getPassRunner()->options.shrinkLevel >= 2 ||
                      !getPassRunner()->options.optimizeLevel;

    // multiple cycles may be needed
    do {
      anotherCycle = false;
      Super::doWalkFunction(func);
      assert(ifStack.empty());
      // flows may contain returns, which are flowing out and so can be
      // optimized
      for (Index i = 0; i < flows.size(); i++) {
        auto* flow = (*flows[i])->dynCast<Return>();
        if (!flow) {
          continue;
        }
        if (!flow->value) {
          // return => nop
          ExpressionManipulator::nop(flow);
        } else {
          // return with value => value
          *flows[i] = flow->value;
        }
        anotherCycle = true;
      }
      flows.clear();
      // optimize loops (we don't do it while tracking flows, as they can
      // interfere)
      for (auto* loop : loops) {
        anotherCycle |= optimizeLoop(loop);
      }
      loops.clear();
      if (anotherCycle) {
        ReFinalize().walkFunctionInModule(func, getModule());
      }
      if (sinkBlocks(func)) {
        anotherCycle = true;
      }
      if (optimizeGC(func)) {
        anotherCycle = true;
      }
      if (foldBoolIf && foldBooleanIfs(func)) {
        anotherCycle = true;
      }
    } while (anotherCycle);

    // Thread trivial jumps.
    struct JumpThreader
      : public PostWalker<JumpThreader,
                          UnifiedExpressionVisitor<JumpThreader>> {
      const PassOptions& passOptions;
      // Map of all labels (branch targets) to the branches going to them. (We
      // only care about blocks here, and not loops, but for simplicity we
      // store all branch targets since blocks are 99% of that set anyhow. Any
      // loops are ignored later.)
      std::unordered_map<Name, std::vector<Expression*>> labelToBranches;

      bool worked = false;
      bool trappedSwitchWorked = false;

      JumpThreader(const PassOptions& passOptions) : passOptions(passOptions) {}

      void visitExpression(Expression* curr) {
        // Find the relevant targets: targets that (as mentioned above) have no
        // value sent to them.
        SmallSet<Name, 2> relevantTargets;
        BranchUtils::operateOnScopeNameUsesAndSentTypes(
          curr, [&](Name name, Type sent) {
            if (sent == Type::none) {
              relevantTargets.insert(name);
            }
          });

        // Note ourselves on all relevant targets.
        for (auto target : relevantTargets) {
          labelToBranches[target].push_back(curr);
        }
      }

      void visitBlock(Block* curr) {
        auto& list = curr->list;
        if (list.size() == 1 && curr->name.is()) {
          // if this block has just one child, a sub-block, then jumps to the
          // former are jumps to us, really
          if (auto* child = list[0]->dynCast<Block>()) {
            if (child->name.is()) {
              redirectBranches(child, curr->name);
            }
          }
        } else if (list.size() >= 2) {
          // With two or more items, we look at the last two items:
          //
          //  (block ;; curr
          //    ...
          //    (block $child ;; child = list[list.size() - 2]
          //    )
          //    after ;; list.back()
          //  )
          //
          // Anything branching to the child will end up in that final
          // instruction |after|. That lets us optimize the cases where |after|
          // is a br or an unreachable.
          auto* child = list[list.size() - 2]->dynCast<Block>();
          auto* after = list.back();
          if (child && child->name) {
            if (auto* jump = after->dynCast<Break>()) {
              if (ExpressionAnalyzer::isSimple(jump)) {
                // Unconditional jumps to the child can skip ahead to where
                // the child jumps.
                redirectBranches(child, jump->name);
              }
            } else if (after->is<Unreachable>()) {
              // Unconditional jumps to a trap can just trap.
              for (auto* branch : labelToBranches[child->name]) {
                if (auto* br = branch->dynCast<Break>()) {
                  if (ExpressionAnalyzer::isSimple(br)) {
                    // It is safe to just modify the br in-place because it is
                    // only jumping here, so this the last place that will
                    // read it.
                    ExpressionManipulator::unreachable(br);
                  }
                } else if (auto* sw = branch->dynCast<Switch>();
                           sw && !sw->value && passOptions.trapsNeverHappen) {
                  if (sw->default_ == child->name) {
                    while (!sw->targets.empty() &&
                           sw->targets.back() == child->name) {
                      sw->targets.pop_back();
                      worked = true;
                      trappedSwitchWorked = true;
                    }
                    if (!sw->targets.empty()) {
                      Name newDefault = sw->targets.back();
                      sw->default_ = newDefault;
                      while (!sw->targets.empty() &&
                             sw->targets.back() == newDefault) {
                        sw->targets.pop_back();
                      }
                      for (auto& target : sw->targets) {
                        if (target == child->name) {
                          target = newDefault;
                        }
                      }
                      labelToBranches[newDefault].push_back(sw);
                      worked = true;
                      trappedSwitchWorked = true;
                    }
                  } else {
                    bool changedSw = false;
                    for (auto& target : sw->targets) {
                      if (target == child->name) {
                        target = sw->default_;
                        changedSw = true;
                      }
                    }
                    if (changedSw) {
                      while (!sw->targets.empty() &&
                             sw->targets.back() == sw->default_) {
                        sw->targets.pop_back();
                      }
                      worked = true;
                      trappedSwitchWorked = true;
                    }
                  }
                }
              }
              if (trappedSwitchWorked &&
                  !BranchUtils::BranchSeeker::has(child, child->name)) {
                child->name = Name();
                worked = true;
              }
            }
          }
        }
      }

      void redirectBranches(Block* from, Name to) {
        auto& branches = labelToBranches[from->name];
        for (auto* branch : branches) {
          if (BranchUtils::replacePossibleTarget(branch, from->name, to)) {
            worked = true;
          }
        }
        // if the jump is to another block then we can update the list, and
        // maybe push it even more later
        for (auto* branch : branches) {
          labelToBranches[to].push_back(branch);
        }
      }

      void finish(Function* func) {
        if (worked) {
          // by changing where brs go, we may change block types etc.
          ReFinalize().walkFunctionInModule(func, getModule());
        }
      }
    };
    while (true) {
      JumpThreader jumpThreader(getPassOptions());
      jumpThreader.setModule(getModule());
      jumpThreader.walkFunction(func);
      jumpThreader.finish(func);
      if (!jumpThreader.trappedSwitchWorked) {
        break;
      }
      // A trapping Switch target was removed; run a cycle of RemoveUnusedBrs
      // so 1-target Switches turn into Ifs and dead blocks are simplified.
      do {
        anotherCycle = false;
        Super::doWalkFunction(func);
        assert(ifStack.empty());
        for (Index i = 0; i < flows.size(); i++) {
          auto* flow = (*flows[i])->dynCast<Return>();
          if (!flow) {
            continue;
          }
          if (!flow->value) {
            ExpressionManipulator::nop(flow);
          } else {
            *flows[i] = flow->value;
          }
          anotherCycle = true;
        }
        flows.clear();
        for (auto* loop : loops) {
          anotherCycle |= optimizeLoop(loop);
        }
        loops.clear();
        if (anotherCycle) {
          ReFinalize().walkFunctionInModule(func, getModule());
        }
        if (sinkBlocks(func)) {
          anotherCycle = true;
        }
        if (optimizeGC(func)) {
          anotherCycle = true;
        }
        if (foldBoolIf && foldBooleanIfs(func)) {
          anotherCycle = true;
        }
      } while (anotherCycle);
    }

    bool shrink2 = getPassRunner()->options.shrinkLevel >= 2;
    if (shrink2 && !isV4Enabled(getPassOptions())) {
      optimizeTailBlockReturns(func);
      if (getPassOptions().trapsNeverHappen) {
        optimizeNoreturnReturnTails(func);
      }
    }

    // perform some final optimizations
    struct FinalOptimizer : public PostWalker<FinalOptimizer> {
      PassOptions& passOptions;

      bool shrink;
      bool shrink2;
      bool v4 = false;
      bool foldBoolIf;
      bool neverUnconditionalize;

      bool needUniqify = false;
      bool refinalize = false;
      std::unordered_set<Block*> unnestedIfBlocks;
      std::unordered_set<Name> tailBlockNames;
      std::unordered_set<Block*> tailBlocks;

      FinalOptimizer(PassOptions& passOptions) : passOptions(passOptions) {}

      static Expression* negateCond(Builder& builder, Expression* cond) {
        if (auto* u = cond->dynCast<Unary>(); u && u->op == EqZInt32) {
          return u->value;
        }
        return builder.makeUnary(EqZInt32, cond);
      }

      static Expression*
      stealSlice(Builder& builder, Block* b, Index start, Index end) {
        assert(start < end && end <= b->list.size());
        if (end - start == 1) {
          return b->list[start];
        }
        auto* res = builder.makeBlock();
        for (Index i = start; i < end; i++) {
          res->list.push_back(b->list[i]);
        }
        res->finalize();
        return res;
      }

      void tryOptimizeNoreturnBlockTail(Block* curr) {
        auto& list = curr->list;
        if (!curr->name.is() || !curr->type.isConcrete() || list.size() < 2 ||
            !list.back()->is<Unreachable>()) {
          return;
        }
        Builder builder(*getModule());

        // (1) Concrete Try followed by Unreachable where do/catch branch to
        // curr->name.
        if (auto* tryy = list[list.size() - 2]->dynCast<Try>();
            tryy && !tryy->isDelegate()) {
          bool allPrevVoid = true;
          for (Index i = 0; i + 2 < list.size(); i++) {
            if (list[i]->type != Type::none) {
              allPrevVoid = false;
              break;
            }
          }
          if (allPrevVoid) {
            auto stripOrInvertArm = [&](Expression*& arm) -> bool {
              if (auto* br = arm->dynCast<Break>();
                  br && !br->condition && br->name == curr->name &&
                  br->value && Type::isSubType(br->value->type, curr->type)) {
                BranchHints::clear(br, getFunction());
                arm = br->value;
                return true;
              }
              auto* b = arm->dynCast<Block>();
              if (!b || b->name.is() || b->list.empty()) {
                return false;
              }
              if (auto* br = b->list.back()->dynCast<Break>();
                  br && !br->condition && br->name == curr->name &&
                  br->value && Type::isSubType(br->value->type, curr->type)) {
                for (Index j = 0; j + 1 < b->list.size(); j++) {
                  if (b->list[j]->type != Type::none) {
                    return false;
                  }
                }
                BranchHints::clear(br, getFunction());
                b->list.back() = br->value;
                b->finalize();
                return true;
              }
              if (b->list.size() >= 3 && b->list.back()->is<Unreachable>() &&
                  !b->list[b->list.size() - 2]->type.isConcrete() &&
                  BranchUtils::BranchSeeker::count(b, curr->name) == 1) {
                for (Index k = b->list.size() - 3;; k--) {
                  if (b->list[k]->type.isConcrete()) {
                    break;
                  }
                  if (auto* drop = b->list[k]->dynCast<Drop>()) {
                    if (auto* br = drop->value->dynCast<Break>();
                        br && br->condition && br->name == curr->name &&
                        br->value && br->type != Type::unreachable &&
                        Type::isSubType(br->value->type, curr->type) &&
                        !EffectAnalyzer(passOptions, *getModule(), br->value)
                           .hasSideEffects() &&
                        EffectAnalyzer::canReorder(passOptions,
                                                   *getModule(),
                                                   br->condition,
                                                   br->value)) {
                      Expression* noreturnBody =
                        stealSlice(builder, b, k + 1, b->list.size() - 1);
                      auto* iff = builder.makeIf(
                        negateCond(builder, br->condition), noreturnBody);
                      BranchHints::copyFlippedTo(br, iff, getFunction());
                      BranchHints::clear(br, getFunction());
                      while (foldBoolIf &&
                             tryFoldBooleanIfCondition(
                               iff, getFunction(), *getModule(), passOptions)) {
                      }
                      Expression* val = br->value;
                      b->list[k] = iff;
                      b->list[k + 1] = val;
                      b->list.resize(k + 2);
                      b->finalize();
                      return true;
                    }
                  }
                  if (auto* iff = b->list[k]->dynCast<If>();
                      iff && !iff->ifFalse &&
                      iff->condition->type != Type::unreachable) {
                    std::vector<Expression*> thenPrefix;
                    Break* br = iff->ifTrue->dynCast<Break>();
                    if (!br) {
                      if (auto* tb = iff->ifTrue->dynCast<Block>();
                          tb && !tb->name.is() && !tb->list.empty()) {
                        if (auto* lastBr = tb->list.back()->dynCast<Break>()) {
                          bool prefixOk = true;
                          for (Index j = 0; j + 1 < tb->list.size(); j++) {
                            if (tb->list[j]->type != Type::none) {
                              prefixOk = false;
                              break;
                            }
                            thenPrefix.push_back(tb->list[j]);
                          }
                          if (prefixOk) {
                            br = lastBr;
                          } else {
                            thenPrefix.clear();
                          }
                        }
                      }
                    }
                    if (br && !br->condition && br->name == curr->name &&
                        br->value &&
                        Type::isSubType(br->value->type, curr->type)) {
                      Expression* noreturnBody =
                        stealSlice(builder, b, k + 1, b->list.size() - 1);
                      Expression* brVal = br->value;
                      BranchHints::clear(br, getFunction());
                      iff->condition = negateCond(builder, iff->condition);
                      iff->ifTrue = noreturnBody;
                      BranchHints::flip(iff, getFunction());
                      iff->finalize();
                      while (foldBoolIf &&
                             tryFoldBooleanIfCondition(
                               iff, getFunction(), *getModule(), passOptions)) {
                      }
                      b->list.resize(k + 1);
                      for (auto* item : thenPrefix) {
                        b->list.push_back(item);
                      }
                      b->list.push_back(brVal);
                      b->finalize();
                      return true;
                    }
                  }
                  if (k == 0) {
                    break;
                  }
                }
              }
              return false;
            };

            bool tryChanged = stripOrInvertArm(tryy->body);
            for (Index ci = 0; ci < tryy->catchBodies.size(); ci++) {
              if (stripOrInvertArm(tryy->catchBodies[ci])) {
                tryChanged = true;
              }
            }
            if (tryChanged &&
                (tryy->body->type != Type::unreachable ||
                 std::any_of(tryy->catchBodies.begin(),
                             tryy->catchBodies.end(),
                             [](Expression* e) {
                               return e->type != Type::unreachable;
                             }))) {
              tryy->finalize(curr->type);
              list.pop_back();
              if (!BranchUtils::BranchSeeker::has(curr, curr->name)) {
                curr->name = Name();
              }
              curr->finalize(curr->type);
              refinalize = true;
              return;
            }
          }
        }

        // (2) Void-block dispatch tower ending in (br $curr VAL) before
        // noreturn tail + Unreachable.
        if (list.size() < 3 || list[list.size() - 2]->type.isConcrete()) {
          return;
        }
        for (Index k = list.size() - 3;; k--) {
          if (list[k]->type.isConcrete()) {
            break;
          }
          if (auto* topTower = list[k]->dynCast<Block>();
              topTower && topTower->name.is() && topTower->type == Type::none &&
              topTower->list.size() >= 2) {
            std::vector<Block*> chain;
            Block* curB = topTower;
            Break* br = nullptr;
            while (curB && curB->name.is() && curB->type == Type::none &&
                   curB->list.size() >= 2) {
              chain.push_back(curB);
              if (auto* b = curB->list.back()->dynCast<Break>();
                  b && !b->condition && b->name == curr->name && b->value &&
                  Type::isSubType(b->value->type, curr->type)) {
                br = b;
                break;
              }
              curB = curB->list[0]->dynCast<Block>();
            }
            if (br) {
              Block* targetTower = chain.back();
              bool tailOk = true;
              for (size_t m = 0; m < chain.size() && tailOk; m++) {
                Block* b = chain[m];
                Index endJ =
                  (m + 1 == chain.size()) ? b->list.size() - 1 : b->list.size();
                for (Index j = 1; j < endJ; j++) {
                  if (b->list[j]->type != Type::none) {
                    tailOk = false;
                    break;
                  }
                  for (size_t p = 0; p <= m; p++) {
                    if (BranchUtils::BranchSeeker::has(b->list[j],
                                                       chain[p]->name)) {
                      tailOk = false;
                      break;
                    }
                  }
                  if (!tailOk) {
                    break;
                  }
                }
              }
              for (size_t p = 0; p < chain.size() && tailOk; p++) {
                if (BranchUtils::BranchSeeker::has(br->value, chain[p]->name)) {
                  tailOk = false;
                }
              }
              if (tailOk) {
                auto* innerTower = targetTower->list[0]->dynCast<Block>();
                if (innerTower && innerTower->name.is() &&
                    innerTower->type == Type::none &&
                    !innerTower->list.empty() &&
                    innerTower->list.back()->type == Type::unreachable) {
                  bool hasAllBranches = BranchUtils::BranchSeeker::has(
                    innerTower, innerTower->name);
                  for (Block* b : chain) {
                    if (!BranchUtils::BranchSeeker::has(innerTower, b->name)) {
                      hasAllBranches = false;
                      break;
                    }
                  }
                  if (hasAllBranches) {
                    chain.push_back(innerTower);
                    size_t d = chain.size();
                    std::vector<Expression*> towerTailPrefix;
                    for (Index j = 1; j + 1 < targetTower->list.size(); j++) {
                      towerTailPrefix.push_back(targetTower->list[j]);
                    }
                    Expression* brVal = br->value;
                    BranchHints::clear(br, getFunction());
                    std::vector<std::vector<Expression*>> savedTails(d - 2);
                    for (size_t m = 0; m + 2 < d; m++) {
                      for (Index j = 1; j < chain[m]->list.size(); j++) {
                        savedTails[m].push_back(chain[m]->list[j]);
                      }
                    }
                    std::vector<Expression*> noreturnItems;
                    for (Index j = k + 1; j + 1 < list.size(); j++) {
                      noreturnItems.push_back(list[j]);
                    }
                    Name returnArmName = chain[d - 1]->name;
                    for (size_t m = d - 1; m >= 1; m--) {
                      chain[m]->name = chain[m - 1]->name;
                    }
                    chain[0]->name = returnArmName;
                    chain[d - 1]->finalize();
                    for (size_t m = d - 2; m >= 1; m--) {
                      chain[m]->list.resize(1);
                      for (auto* item : savedTails[m - 1]) {
                        chain[m]->list.push_back(item);
                      }
                      chain[m]->finalize();
                    }
                    chain[0]->list.resize(1);
                    for (auto* item : noreturnItems) {
                      chain[0]->list.push_back(item);
                    }
                    chain[0]->finalize();
                    list.resize(k + 1);
                    for (auto* item : towerTailPrefix) {
                      list.push_back(item);
                    }
                    list.push_back(brVal);
                    if (!BranchUtils::BranchSeeker::has(curr, curr->name)) {
                      curr->name = Name();
                    }
                    curr->finalize(curr->type);
                    refinalize = true;
                    return;
                  }
                }
              }
            }
          }
          if (k == 0) {
            break;
          }
        }
      }

      void visitBlock(Block* curr) {
        // if a block has an if br else br, we can un-conditionalize the latter,
        // allowing the if to become a br_if.
        // * note that if not in a block already, then we need to create a block
        //   for this, so not useful otherwise
        // * note that this only happens at the end of a block, as code after
        //   the if is dead
        // * note that we do this at the end, because un-conditionalizing can
        //   interfere with optimizeLoop()ing.
        auto& list = curr->list;
        if (v4) {
          if (curr->name.is() &&
              !BranchUtils::BranchSeeker::has(curr, curr->name)) {
            curr->name = Name();
          }
          Index writeIdx = 0;
          for (Index readIdx = 0; readIdx < list.size(); readIdx++) {
            if (!list[readIdx]->is<Nop>()) {
              list[writeIdx++] = list[readIdx];
            }
          }
          if (writeIdx != list.size()) {
            list.resize(writeIdx);
            curr->finalize(curr->type);
          }
          if (!curr->name.is() && list.empty() && curr->type == Type::none) {
            ExpressionManipulator::nop(curr);
            return;
          }
        }
        for (Index i = 0; i < list.size(); i++) {
          while (auto* sub = list[i]->dynCast<Block>()) {
            bool canSplice = unnestedIfBlocks.contains(sub);
            if (!canSplice && v4 && !sub->name.is() &&
                !sub->list.empty()) {
              bool prefixVoid = true;
              for (Index j = 0; j + 1 < sub->list.size(); j++) {
                if (sub->list[j]->type != Type::none) {
                  prefixVoid = false;
                  break;
                }
              }
              if (prefixVoid &&
                  (i + 1 == list.size() ||
                   sub->list.back()->type == Type::none)) {
                canSplice = true;
              }
            }
            if (!canSplice) {
              break;
            }
            unnestedIfBlocks.erase(sub);
            list[i] = sub->list[0];
            for (Index j = 1; j < sub->list.size(); j++) {
              list.insertAt(i + j, sub->list[j]);
            }
            curr->finalize(curr->type);
            if (!v4) {
              break;
            }
          }
          auto* iff = list[i]->dynCast<If>();
          if (v4 && iff && iff->ifFalse && iff->ifFalse->is<Nop>()) {
            iff->ifFalse = nullptr;
            iff->finalize();
          }
          if (!iff || !iff->ifFalse) {
            // if it lacked an if-false, it would already be a br_if, as that's
            // the easy case
            continue;
          }
          auto* ifTrueBreak = iff->ifTrue->dynCast<Break>();
          if (ifTrueBreak && !ifTrueBreak->condition &&
              canTurnIfIntoBrIf(iff->condition,
                                ifTrueBreak->value,
                                passOptions,
                                *getModule())) {
            // we are an if-else where the ifTrue is a break without a
            // condition, so we can do this
            ifTrueBreak->condition = iff->condition;
            BranchHints::copyTo(iff, ifTrueBreak, getFunction());
            ifTrueBreak->finalize();
            list[i] = Builder(*getModule()).dropIfConcretelyTyped(ifTrueBreak);
            ExpressionManipulator::spliceIntoBlock(curr, i + 1, iff->ifFalse);
            continue;
          }
          // otherwise, perhaps we can flip the if
          auto* ifFalseBreak = iff->ifFalse->dynCast<Break>();
          if (ifFalseBreak && !ifFalseBreak->condition &&
              canTurnIfIntoBrIf(iff->condition,
                                ifFalseBreak->value,
                                passOptions,
                                *getModule())) {
            Builder builder(*getModule());
            ifFalseBreak->condition =
              makeNegatedBool(builder, iff->condition, v4);
            BranchHints::copyFlippedTo(iff, ifFalseBreak, getFunction());
            ifFalseBreak->finalize();
            list[i] = builder.dropIfConcretelyTyped(ifFalseBreak);
            ExpressionManipulator::spliceIntoBlock(curr, i + 1, iff->ifTrue);
            continue;
          }
        }
        if (list.size() >= 2) {
          // combine/optimize adjacent br_ifs + a br (maybe _if) right after it
          bool mergedWithNop = false;
          for (Index i = 0; i < list.size() - 1; i++) {
            if (auto* br1 = list[i]->dynCast<Break>()) {
              // avoid unreachable brs, as they are dead code anyhow, and after
              // merging them the outer scope could need type changes
              if (!br1->condition || br1->type == Type::unreachable) {
                continue;
              }
              assert(!br1->value);
              auto* br2 = list[i + 1]->dynCast<Break>();
              if (!br2 || br1->name != br2->name) {
                continue;
              }
              assert(!br2->value); // same target as previous, which has no value
              // a br_if and then a br[_if] with the same target right after it
              if (br2->condition) {
                if (shrink && !neverUnconditionalize &&
                    br2->type != Type::unreachable) {
                  // Join adjacent br_ifs to the same target, making one br_if
                  // with a "selectified" condition that executes both.
                  if (!EffectAnalyzer(passOptions, *getModule(), br2->condition)
                         .hasSideEffects() &&
                      !tooCostlyToRunUnconditionally(passOptions,
                                                     br2->condition)) {
                    // it's ok to execute them both, do it
                    Builder builder(*getModule());
                    br1->condition = builder.makeBinary(
                      OrInt32, br1->condition, br2->condition);
                    BranchHints::applyOrTo(br1, br2, br1, getFunction());
                    ExpressionManipulator::nop(br2);
                    BranchHints::clear(br2, getFunction());
                  }
                }
              } else {
                // merge, we go there anyhow
                Builder builder(*getModule());
                list[i] = builder.makeDrop(br1->condition);
              }
              continue;
            }
            if (shrink && !neverUnconditionalize) {
              auto* drop1 = list[i]->dynCast<Drop>();
              auto* drop2 = list[i + 1]->dynCast<Drop>();
              if (!drop1 || !drop2) {
                continue;
              }
              auto* br1 = drop1->value->dynCast<Break>();
              auto* br2 = drop2->value->dynCast<Break>();
              if (!br1 || !br2 || !br1->condition || !br2->condition ||
                  !br1->value || !br2->value ||
                  br1->type == Type::unreachable ||
                  br2->type == Type::unreachable || br1->name != br2->name ||
                  !ExpressionAnalyzer::equal(br1->value, br2->value)) {
                continue;
              }
              if (!EffectAnalyzer(passOptions, *getModule(), br1->value)
                     .hasSideEffects() &&
                  !EffectAnalyzer(passOptions, *getModule(), br2->condition)
                     .hasSideEffects() &&
                  !tooCostlyToRunUnconditionally(passOptions, br2->condition) &&
                  EffectAnalyzer::canReorder(
                    passOptions, *getModule(), br1->condition, br2->value)) {
                Builder builder(*getModule());
                br1->condition =
                  builder.makeBinary(OrInt32, br1->condition, br2->condition);
                BranchHints::applyOrTo(br1, br2, br1, getFunction());
                ExpressionManipulator::nop(drop2);
                BranchHints::clear(br2, getFunction());
                mergedWithNop = true;
                // Look ahead to merge runs of >= 3 adjacent same-value
                // Drop(br_if)s (e.g. 13 in $2855).
                while (i + 2 < list.size()) {
                  auto* dropNext = list[i + 2]->dynCast<Drop>();
                  if (!dropNext) {
                    break;
                  }
                  auto* brNext = dropNext->value->dynCast<Break>();
                  if (!brNext || !brNext->condition || !brNext->value ||
                      brNext->type == Type::unreachable ||
                      brNext->name != br1->name ||
                      !ExpressionAnalyzer::equal(br1->value, brNext->value) ||
                      EffectAnalyzer(passOptions, *getModule(), brNext->condition)
                        .hasSideEffects() ||
                      tooCostlyToRunUnconditionally(passOptions,
                                                    brNext->condition) ||
                      !EffectAnalyzer::canReorder(passOptions,
                                                  *getModule(),
                                                  br1->condition,
                                                  brNext->value)) {
                    break;
                  }
                  br1->condition = builder.makeBinary(
                    OrInt32, br1->condition, brNext->condition);
                  BranchHints::applyOrTo(br1, brNext, br1, getFunction());
                  ExpressionManipulator::nop(dropNext);
                  BranchHints::clear(brNext, getFunction());
                  i++;
                }
              }
            }
          }
          if (v4 && !neverUnconditionalize && list.size() >= 2) {
            // Fold adjacent `[(drop (br_if $L V1 COND)), (br $L V2)]`,
            // `[(drop (br_if $tail V1 COND)), (return V2)]`,
            // `[(if COND (then <prefix> (br $L V1))), (br $L V2)]`, and
            // `[(if COND (then <prefix> (return V1))), (return V2)]` into
            // `(br $L (if (result T) COND ...))` / `(return (if ...))` +
            // `selectify`.
            auto isOneByteType = [](Type t) {
              return t.isConcrete() && !t.isTuple() &&
                     (!t.isRef() ||
                      (t.isNullable() && t.getHeapType().isBasic()));
            };
            auto matchNonNullSrc = [&](Expression* cond,
                                       Expression* val) -> Expression* {
              if (!val) {
                return nullptr;
              }
              auto* eqz = cond ? cond->dynCast<Unary>() : nullptr;
              if (!eqz || eqz->op != EqZInt32) {
                return nullptr;
              }
              auto* isNull = eqz->value->dynCast<RefIsNull>();
              if (!isNull) {
                return nullptr;
              }
              Expression* src = isNull->value;
              if (!src->type.isRef() || !src->type.isNullable() ||
                  src->type == Type::unreachable) {
                return nullptr;
              }
              if (auto* valGet = val->dynCast<LocalGet>()) {
                if (auto* g = src->dynCast<LocalGet>()) {
                  if (g->index == valGet->index) {
                    return g;
                  }
                } else if (auto* tee = src->dynCast<LocalSet>()) {
                  if (tee->isTee() && tee->index == valGet->index &&
                      tee->value->type != Type::unreachable) {
                    return tee;
                  }
                }
                return nullptr;
              }
              if (ExpressionAnalyzer::equal(src, val) &&
                  !EffectAnalyzer(passOptions, *getModule(), src)
                     .hasUnremovableSideEffects()) {
                return src;
              }
              return nullptr;
            };
            for (Index idx = list.size() - 1; idx > 0; idx--) {
              auto* drop1 = list[idx - 1]->dynCast<Drop>();
              auto* iff1 = list[idx - 1]->dynCast<If>();
              Expression* cond1 = nullptr;
              Expression* val1 = nullptr;
              Expression* hintSrc = nullptr;
              std::vector<Expression*> thenPrefix;
              bool isReturn1 = false;
              Name name1;
              if (drop1) {
                auto* br1 = drop1->value->dynCast<Break>();
                if (!br1 || !br1->condition || !br1->value ||
                    br1->type == Type::unreachable ||
                    !canTurnIfIntoBrIf(
                      br1->condition, br1->value, passOptions, *getModule())) {
                  continue;
                }
                cond1 = br1->condition;
                val1 = br1->value;
                hintSrc = br1;
                name1 = br1->name;
              } else if (iff1 && !iff1->ifFalse &&
                         iff1->condition->type != Type::unreachable) {
                Expression* lastStmt = iff1->ifTrue;
                if (auto* tb = iff1->ifTrue->dynCast<Block>();
                    tb && !tb->name.is() && !tb->list.empty()) {
                  bool prefixOk = true;
                  for (Index j = 0; j + 1 < tb->list.size(); j++) {
                    if (tb->list[j]->type != Type::none) {
                      prefixOk = false;
                      break;
                    }
                    thenPrefix.push_back(tb->list[j]);
                  }
                  if (!prefixOk) {
                    continue;
                  }
                  lastStmt = tb->list.back();
                }
                if (auto* b1 = lastStmt->dynCast<Break>();
                    b1 && !b1->condition && b1->value &&
                    b1->value->type != Type::unreachable) {
                  cond1 = iff1->condition;
                  val1 = b1->value;
                  hintSrc = iff1;
                  name1 = b1->name;
                } else if (auto* r1 = lastStmt->dynCast<Return>();
                           r1 && r1->value &&
                           r1->value->type != Type::unreachable) {
                  cond1 = iff1->condition;
                  val1 = r1->value;
                  hintSrc = iff1;
                  isReturn1 = true;
                } else {
                  continue;
                }
              } else {
                continue;
              }
              Break* br2 = list[idx]->dynCast<Break>();
              Return* ret2 = list[idx]->dynCast<Return>();
              Expression** val2Ptr = nullptr;
              if (!isReturn1) {
                if (br2 && !br2->condition && br2->value &&
                    br2->name == name1) {
                  val2Ptr = &br2->value;
                } else if (ret2 && ret2->value &&
                           tailBlockNames.contains(name1)) {
                  val2Ptr = &ret2->value;
                }
              } else {
                if (ret2 && ret2->value) {
                  val2Ptr = &ret2->value;
                } else if (br2 && !br2->condition && br2->value &&
                           tailBlockNames.contains(br2->name)) {
                  val2Ptr = &br2->value;
                }
              }
              if (!val2Ptr || (*val2Ptr)->type == Type::unreachable) {
                continue;
              }
              Expression* val2 = *val2Ptr;
              Type ilub = Type::getLeastUpperBound(val1->type, val2->type);
              if (ilub == Type::none) {
                continue;
              }
              if (thenPrefix.empty()) {
                if (Expression* nonNullSrc = matchNonNullSrc(cond1, val1)) {
                  if (val2->is<RefNull>() &&
                      Type::isSubType(nonNullSrc->type, ilub)) {
                    *val2Ptr = nonNullSrc;
                    if (br2) {
                      br2->finalize();
                    } else {
                      ret2->finalize();
                    }
                    BranchHints::clear(hintSrc, getFunction());
                    list[idx - 1] = list[idx];
                    list[idx] = Builder(*getModule()).makeNop();
                    mergedWithNop = true;
                    refinalize = true;
                    continue;
                  }
                  if (iff1) {
                    // Leave non-null return/break guard for br_on_non_null.
                    continue;
                  }
                }
                if (ExpressionAnalyzer::equal(val1, val2)) {
                  EffectAnalyzer v1Eff(passOptions, *getModule(), val1);
                  EffectAnalyzer c1Eff(passOptions, *getModule(), cond1);
                  if (!v1Eff.hasUnremovableSideEffects() &&
                      !c1Eff.hasUnremovableSideEffects()) {
                    BranchHints::clear(hintSrc, getFunction());
                    list[idx - 1] = list[idx];
                    list[idx] = Builder(*getModule()).makeNop();
                    mergedWithNop = true;
                    refinalize = true;
                    continue;
                  }
                }
              }
              Builder builder(*getModule());
              bool oneByte = isOneByteType(ilub);
              bool eqzCond = false;
              if (auto* u = cond1->dynCast<Unary>(); u && u->op == EqZInt32) {
                eqzCond = true;
              }
              Expression* thenArm = prependToArm(builder, thenPrefix, val1);
              auto* iff = builder.makeIf(cond1, thenArm, val2, ilub);
              BranchHints::copyTo(hintSrc, iff, getFunction());
              bool foldedBool = false;
              while (foldBoolIf &&
                     tryFoldBooleanIfCondition(
                       iff, getFunction(), *getModule(), passOptions)) {
                foldedBool = true;
              }
              if (auto* u = iff->condition->dynCast<Unary>();
                  u && u->op == EqZInt32) {
                iff->condition = u->value;
                std::swap(iff->ifTrue, iff->ifFalse);
                BranchHints::flip(iff, getFunction());
                eqzCond = true;
              }
              Expression* valExpr = iff;
              bool didSelectify = false;
              if (auto* sel = selectify(iff)) {
                valExpr = sel;
                didSelectify = true;
              }
              if (iff1 && isReturn1 && !didSelectify && !foldedBool &&
                  !oneByte && !eqzCond) {
                BranchHints::clear(iff, getFunction());
                continue;
              }
              BranchHints::clear(hintSrc, getFunction());
              *val2Ptr = valExpr;
              if (br2) {
                br2->finalize();
              } else {
                ret2->finalize();
              }
              list[idx - 1] = list[idx];
              list[idx] = builder.makeNop();
              mergedWithNop = true;
              refinalize = true;
            }
          }
          if (shrink && (mergedWithNop || v4)) {
            Index writeIdx = 0;
            for (Index readIdx = 0; readIdx < list.size(); readIdx++) {
              if (!list[readIdx]->is<Nop>()) {
                list[writeIdx++] = list[readIdx];
              }
            }
            list.resize(writeIdx);
          }
          if (v4 && !list.empty() &&
              (curr == getFunction()->body || tailBlocks.contains(curr)) &&
              getFunction()->getResults().isConcrete()) {
            if (auto* ret = list.back()->dynCast<Return>();
                ret && ret->value && ret->value->type != Type::unreachable &&
                Type::isSubType(ret->value->type,
                                getFunction()->getResults())) {
              bool allPrevVoid = true;
              for (Index j = 0; j + 1 < list.size(); j++) {
                if (list[j]->type != Type::none) {
                  allPrevVoid = false;
                  break;
                }
              }
              if (allPrevVoid) {
                list.back() = ret->value;
                curr->finalize(getFunction()->getResults());
                refinalize = true;
              }
            }
          }
          if (v4 && !curr->name.is() && list.size() == 1) {
            replaceCurrent(list[0]);
            return;
          }
          if (list.size() < 2) {
            return;
          }
          // Combine adjacent br_ifs that test the same value into a br_table,
          // when that makes sense.
          tablify(curr);

          if (!shrink) {
            // Pattern-patch ifs, recreating them when it makes sense.
            restructureIf(curr);

            size_t size = curr->list.size();
            auto* secondLast = curr->list[size - 2];
            auto* last = curr->list[size - 1];
            if (auto* drop = secondLast->dynCast<Drop>()) {
              if (auto* br = drop->value->dynCast<Break>();
                  br && br->value && br->condition && br->name == curr->name &&
                  ExpressionAnalyzer::equal(br->value, last)) {
                auto valueEffects =
                  EffectAnalyzer(passOptions, *getModule(), br->value);
                if (!valueEffects.hasUnremovableSideEffects()) {
                  auto conditionEffects =
                    EffectAnalyzer(passOptions, *getModule(), br->condition);
                  if (!conditionEffects.invalidates(valueEffects)) {
                    drop->value = br->condition;
                  }
                }
              }
            }
          } else {
            if (shrink2 && passOptions.trapsNeverHappen) {
              tryOptimizeNoreturnBlockTail(curr);
              if (curr->list.size() < 2) {
                return;
              }
            }
            // Under shrink, loop backward over block-tail
            // `(drop (br_if $curr (val) (cond)))` + `last` chains before
            // `restructureIf` so 2+-branch blocks (and 2-element boolean/select
            // blocks) fold completely.
            Builder builder(*getModule());
            bool tailChanged = false;
            while (curr->list.size() >= 2 && curr->name.is()) {
              size_t size = curr->list.size();
              auto* secondLast = curr->list[size - 2];
              auto* last = curr->list[size - 1];
              if (v4 && curr->type.isRef() && curr->type.isNullable() &&
                  last->is<RefNull>()) {
                if (auto* brOn = secondLast->dynCast<BrOn>();
                    brOn && brOn->op == BrOnNonNull &&
                    brOn->name == curr->name &&
                    brOn->type != Type::unreachable &&
                    Type::isSubType(brOn->ref->type, curr->type)) {
                  curr->list[size - 2] = brOn->ref;
                  curr->list.pop_back();
                  tailChanged = true;
                  continue;
                }
              }
              if (v4 && curr->type == Type::i32) {
                int32_t lastConst = 0;
                if (isI32Const(last, lastConst) &&
                    (lastConst == 0 || lastConst == 1)) {
                  if (auto* iff = secondLast->dynCast<If>();
                      iff && !iff->ifFalse &&
                      iff->condition->type != Type::unreachable &&
                      BranchUtils::BranchSeeker::count(iff, curr->name) == 1) {
                    Block* tb = iff->ifTrue->dynCast<Block>();
                    Expression** armEndPtr = nullptr;
                    if (tb && !tb->name.is() && !tb->list.empty()) {
                      bool ok = true;
                      for (Index j = 0; j + 1 < tb->list.size(); j++) {
                        if (tb->list[j]->type != Type::none) {
                          ok = false;
                          break;
                        }
                      }
                      if (ok) {
                        armEndPtr = &tb->list.back();
                      }
                    } else if (!tb) {
                      armEndPtr = &iff->ifTrue;
                    }
                    if (armEndPtr) {
                      if (auto* d = (*armEndPtr)->dynCast<Drop>()) {
                        if (auto* b = d->value->dynCast<Break>();
                            b && b->condition && b->value &&
                            b->name == curr->name &&
                            b->type != Type::unreachable) {
                          int32_t brConst = 0;
                          if (isI32Const(b->value, brConst) &&
                              ((brConst == 1 && lastConst == 0) ||
                               (brConst == 0 && lastConst == 1))) {
                            Expression* condVal =
                              (brConst == 1)
                                ? makeOneBitCond(builder, b->condition)
                                : makeNegOneBitCond(builder, b->condition, true);
                            BranchHints::clear(b, getFunction());
                            *armEndPtr = builder.makeBreak(curr->name, condVal);
                            if (tb) {
                              tb->finalize();
                            }
                            iff->finalize();
                            tailChanged = true;
                            refinalize = true;
                            break;
                          }
                        }
                      }
                    }
                  }
                }
              }
              auto* drop = secondLast->dynCast<Drop>();
              if (!drop) {
                break;
              }
              auto* br = drop->value->dynCast<Break>();
              if (!br || !br->value || !br->condition ||
                  br->name != curr->name || br->type == Type::unreachable ||
                  last->type == Type::unreachable) {
                break;
              }
              // (a) Equal values: `br->value == last`.
              if (ExpressionAnalyzer::equal(br->value, last)) {
                EffectAnalyzer valEff(passOptions, *getModule(), br->value);
                if (!valEff.hasUnremovableSideEffects()) {
                  EffectAnalyzer condEff(
                    passOptions, *getModule(), br->condition);
                  if (!condEff.invalidates(valEff)) {
                    BranchHints::clear(br, getFunction());
                    if (!condEff.hasUnremovableSideEffects()) {
                      curr->list[size - 2] = last;
                      curr->list.pop_back();
                      tailChanged = true;
                      continue;
                    }
                    drop->value = br->condition;
                    tailChanged = true;
                  }
                }
                break;
              }
              // (b) Opposite boolean constants: `{br->value, last} == {0, 1}`.
              int32_t brConst = 0, lastConst = 0;
              if (isI32Const(br->value, brConst) &&
                  isI32Const(last, lastConst) &&
                  ((brConst == 0 && lastConst == 1) ||
                   (brConst == 1 && lastConst == 0))) {
                Expression* rep = nullptr;
                if (brConst == 0 && lastConst == 1) {
                  if (auto* u = br->condition->dynCast<Unary>();
                      u && u->op == EqZInt32 &&
                      (v4 ? isOneBitI32(u->value)
                          : Properties::emitsBoolean(u->value))) {
                    rep = u->value;
                  } else {
                    rep = makeNegatedBool(builder, br->condition, v4);
                  }
                } else {
                  if (v4 ? isOneBitI32(br->condition)
                         : Properties::emitsBoolean(br->condition)) {
                    rep = br->condition;
                  } else {
                    rep = builder.makeBinary(
                      NeInt32, br->condition, builder.makeConst(int32_t(0)));
                  }
                }
                BranchHints::clear(br, getFunction());
                curr->list[size - 2] = rep;
                curr->list.pop_back();
                tailChanged = true;
                continue;
              }
              // (c) General `select` fold:
              // - For non-reference types, untyped `select` (1 B) is always
              //   smaller than `br_if + drop` (3 B), even for intermediate
              //   steps in a multi-branch block.
              // - For reference types, `SelectWithType` (`0x1c 0x01 <type>`) is
              //   smaller than `if <type> .. else .. end` by 1 B only when the
              //   enclosing block is eliminated (`size == 2` and single branch)
              //   and `sel->type` does not have a larger encoding than
              //   `curr->type`.
              if (!neverUnconditionalize &&
                  (!curr->type.isRef() ||
                   (size == 2 &&
                    BranchUtils::BranchSeeker::count(curr, curr->name) == 1)) &&
                  Properties::canEmitSelectWithArms(br->value, last) &&
                  !tooCostlyToRunUnconditionally(
                    passOptions, br->value, last)) {
                EffectAnalyzer valEff(passOptions, *getModule(), br->value);
                EffectAnalyzer lastEff(passOptions, *getModule(), last);
                if (!valEff.hasSideEffects() && !lastEff.hasSideEffects()) {
                  EffectAnalyzer condEff(
                    passOptions, *getModule(), br->condition);
                  if (!condEff.invalidates(valEff) &&
                      EffectAnalyzer::canReorder(
                        passOptions, *getModule(), br->condition, last)) {
                    auto* sel =
                      builder.makeSelect(br->condition, br->value, last);
                    if (!sel->type.isRef() || sel->type == curr->type ||
                        sel->type.getHeapType().isBasic()) {
                      if (sel->type != curr->type) {
                        refinalize = true;
                      }
                      BranchHints::clear(br, getFunction());
                      curr->list[size - 2] = sel;
                      curr->list.pop_back();
                      tailChanged = true;
                      continue;
                    }
                  }
                }
              }
              break;
            }
            if ((tailChanged || v4) && curr->name.is() &&
                !BranchUtils::BranchSeeker::has(curr, curr->name)) {
              curr->name = Name();
              curr->finalize(curr->type);
              refinalize = true;
            }
            auto peelTailReturnIfs = [&]() {
              Type funcResults = getFunction()->getResults();
              if (!v4 || getCurrent() != curr || !funcResults.isConcrete() ||
                  (curr != getFunction()->body && !tailBlocks.contains(curr))) {
                return;
              }
              auto isOneByteType = [](Type t) {
                return t.isConcrete() && !t.isTuple() &&
                       (!t.isRef() ||
                        (t.isNullable() && t.getHeapType().isBasic()));
              };
              auto isEqz = [](Expression* c) {
                auto* u = c->dynCast<Unary>();
                return u && u->op == EqZInt32;
              };
              auto isBrOnNonNullGuard = [&](Expression* cond,
                                            Expression* retVal) {
                if (!funcResults.isRef() || !retVal ||
                    !Type::isSubType(retVal->type, funcResults)) {
                  return false;
                }
                auto* eqz = cond->dynCast<Unary>();
                if (!eqz || eqz->op != EqZInt32) {
                  return false;
                }
                auto* isNull = eqz->value->dynCast<RefIsNull>();
                if (!isNull) {
                  return false;
                }
                Expression* src = isNull->value;
                if (!src->type.isRef() || !src->type.isNullable() ||
                    src->type == Type::unreachable) {
                  return false;
                }
                if (auto* valGet = retVal->dynCast<LocalGet>()) {
                  if (auto* g = src->dynCast<LocalGet>()) {
                    return g->index == valGet->index;
                  }
                  if (auto* tee = src->dynCast<LocalSet>()) {
                    return tee->isTee() && tee->index == valGet->index &&
                           tee->value->type != Type::unreachable &&
                           Type::isSubType(tee->value->type, funcResults);
                  }
                  return false;
                }
                Type sentType(src->type.getHeapType(),
                              NonNullable,
                              src->type.getExactness());
                return Type::isSubType(sentType, funcResults) &&
                       ExpressionAnalyzer::equal(src, retVal) &&
                       !EffectAnalyzer(passOptions, *getModule(), src)
                          .hasUnremovableSideEffects();
              };
              std::vector<Expression*> committed(curr->list.begin(),
                                                 curr->list.end());
              bool hasSpeculative = false;
              bool anyCommitted = false;
              while (curr->list.size() >= 2) {
                size_t sz = curr->list.size();
                auto* iff = curr->list[sz - 2]->dynCast<If>();
                Expression* v2 = curr->list[sz - 1];
                if (!iff || iff->ifFalse ||
                    iff->condition->type == Type::unreachable ||
                    v2->type == Type::unreachable ||
                    !Type::isSubType(v2->type, funcResults)) {
                  break;
                }
                if (curr->name.is() &&
                    BranchUtils::BranchSeeker::has(iff, curr->name)) {
                  break;
                }
                std::vector<Expression*> thenPrefix;
                Return* ret1 = nullptr;
                if (auto* r = iff->ifTrue->dynCast<Return>()) {
                  ret1 = r;
                } else if (auto* tb = iff->ifTrue->dynCast<Block>();
                           tb && !tb->name.is() && !tb->list.empty()) {
                  if (auto* r = tb->list.back()->dynCast<Return>()) {
                    bool ok = true;
                    for (Index j = 0; j + 1 < tb->list.size(); j++) {
                      if (tb->list[j]->type != Type::none) {
                        ok = false;
                        break;
                      }
                      thenPrefix.push_back(tb->list[j]);
                    }
                    if (ok) {
                      ret1 = r;
                    }
                  }
                }
                if (!ret1 || !ret1->value ||
                    ret1->value->type == Type::unreachable ||
                    !Type::isSubType(ret1->value->type, funcResults)) {
                  break;
                }
                Expression* v1 = ret1->value;
                if (thenPrefix.empty() &&
                    isBrOnNonNullGuard(iff->condition, v1)) {
                  break;
                }
                if (thenPrefix.empty() && ExpressionAnalyzer::equal(v1, v2)) {
                  EffectAnalyzer v1Eff(passOptions, *getModule(), v1);
                  EffectAnalyzer condEff(
                    passOptions, *getModule(), iff->condition);
                  if (!v1Eff.hasUnremovableSideEffects() &&
                      !condEff.hasUnremovableSideEffects()) {
                    BranchHints::clear(iff, getFunction());
                    curr->list[sz - 2] = v2;
                    curr->list.pop_back();
                    committed.assign(curr->list.begin(), curr->list.end());
                    hasSpeculative = false;
                    anyCommitted = true;
                    continue;
                  }
                }
                Type ilub = Type::getLeastUpperBound(v1->type, v2->type);
                if (ilub == Type::none) {
                  break;
                }
                bool oneByte = isOneByteType(ilub);
                bool eqzCond = isEqz(iff->condition);
                Expression* thenArm = prependToArm(builder, thenPrefix, v1);
                auto* newIf = builder.makeIf(iff->condition, thenArm, v2, ilub);
                BranchHints::copyTo(iff, newIf, getFunction());
                bool foldedBool = false;
                while (foldBoolIf &&
                       tryFoldBooleanIfCondition(
                         newIf, getFunction(), *getModule(), passOptions)) {
                  foldedBool = true;
                }
                if (auto* u = newIf->condition->dynCast<Unary>();
                    u && u->op == EqZInt32) {
                  newIf->condition = u->value;
                  std::swap(newIf->ifTrue, newIf->ifFalse);
                  BranchHints::flip(newIf, getFunction());
                  eqzCond = true;
                }
                Expression* rep = newIf;
                bool didSelectify = false;
                if (auto* sel = selectify(newIf)) {
                  if (oneByte || eqzCond || foldedBool || !sel->type.isRef() ||
                      (sel->type.isNullable() &&
                       sel->type.getHeapType().isBasic())) {
                    rep = sel;
                    didSelectify = true;
                  }
                }
                bool strictlyProfitable =
                  didSelectify || foldedBool || (oneByte && eqzCond);
                if (!strictlyProfitable && !oneByte) {
                  BranchHints::clear(newIf, getFunction());
                  break;
                }
                curr->list[sz - 2] = rep;
                curr->list.pop_back();
                if (strictlyProfitable) {
                  BranchHints::clear(iff, getFunction());
                  committed.assign(curr->list.begin(), curr->list.end());
                  hasSpeculative = false;
                  anyCommitted = true;
                } else {
                  hasSpeculative = true;
                }
              }
              if (hasSpeculative) {
                curr->list.resize(committed.size());
                for (Index i = 0; i < committed.size(); i++) {
                  curr->list[i] = committed[i];
                }
              }
              if (anyCommitted) {
                if (!curr->name.is() && curr->list.size() == 1) {
                  replaceCurrent(curr->list[0]);
                  refinalize = true;
                  return;
                }
                curr->finalize(curr->type);
                refinalize = true;
              }
            };
            peelTailReturnIfs();
            if (getCurrent() != curr) {
              return;
            }
            if (curr->list.size() >= 2) {
              restructureIf(curr);
            }
            if (v4 && getCurrent() == curr) {
              bool unnestedAny = false;
              for (Index i = 0; i < curr->list.size(); i++) {
                if (auto* iff = curr->list[i]->dynCast<If>();
                    iff && iff->ifFalse) {
                  if (auto* seq = tryUnnestOneUnreachableArmIf(
                        iff, /*allowBrIfChain=*/true)) {
                    unnestedIfBlocks.erase(seq);
                    curr->list[i] = seq->list[0];
                    for (Index j = 1; j < seq->list.size(); j++) {
                      curr->list.insertAt(i + j, seq->list[j]);
                    }
                    unnestedAny = true;
                  }
                }
              }
              if (unnestedAny) {
                curr->finalize(curr->type);
                refinalize = true;
              }
            }
            peelTailReturnIfs();
            if (getCurrent() != curr) {
              return;
            }
            if (shrink2 && getCurrent() == curr && !curr->list.empty() &&
                curr->type.isConcrete()) {
              if (auto* iff = curr->list.back()->dynCast<If>();
                  iff && iff->ifFalse && iff->type.isConcrete() &&
                  iff->condition->type != Type::unreachable &&
                  ((iff->ifTrue->type == Type::unreachable &&
                    iff->ifFalse->type.isConcrete()) ||
                   (iff->ifFalse->type == Type::unreachable &&
                    iff->ifTrue->type.isConcrete()))) {
                bool unreachWasTrue = iff->ifTrue->type == Type::unreachable;
                Expression* unreachArm =
                  unreachWasTrue ? iff->ifTrue : iff->ifFalse;
                Expression* concreteArm =
                  unreachWasTrue ? iff->ifFalse : iff->ifTrue;
                bool canStripUnreach = false;
                if (passOptions.trapsNeverHappen) {
                  if (auto* b = unreachArm->dynCast<Block>();
                      b && !b->name.is() && b->list.size() >= 2 &&
                      b->list.back()->is<Unreachable>() &&
                      !b->list[b->list.size() - 2]->type.isConcrete()) {
                    canStripUnreach = true;
                  }
                }
                bool condIsEqz = false;
                if (auto* u = iff->condition->dynCast<Unary>();
                    u && u->op == EqZInt32) {
                  condIsEqz = true;
                } else if (v4 && canInvertBoolExprFree(iff->condition)) {
                  condIsEqz = true;
                }
                bool indexedType =
                  iff->type.isRef() && !iff->type.getHeapType().isBasic();
                if (canStripUnreach || unreachWasTrue || condIsEqz ||
                    indexedType) {
                  if (canStripUnreach) {
                    auto* b = unreachArm->cast<Block>();
                    b->list.pop_back();
                    if (b->list.size() == 1) {
                      unreachArm = b->list[0];
                    } else {
                      b->finalize();
                    }
                  }
                  if (!unreachWasTrue) {
                    iff->condition =
                      v4 ? makeNegatedBool(builder, iff->condition, true)
                         : negateCond(builder, iff->condition);
                    BranchHints::flip(iff, getFunction());
                  }
                  iff->ifTrue = unreachArm;
                  iff->ifFalse = nullptr;
                  iff->finalize();
                  while (foldBoolIf &&
                         tryFoldBooleanIfCondition(
                           iff, getFunction(), *getModule(), passOptions)) {
                  }
                  curr->list.back() = iff;
                  if (auto* cb = concreteArm->dynCast<Block>();
                      cb && !cb->name.is()) {
                    for (auto* item : cb->list) {
                      curr->list.push_back(item);
                    }
                  } else {
                    curr->list.push_back(concreteArm);
                  }
                  curr->finalize(curr->type);
                  refinalize = true;
                }
              }
            }
          }
        }
      }

      void visitSwitch(Switch* curr) {
        if (BranchUtils::getUniqueTargets(curr).size() == 1) {
          // This switch has just one target no matter what; replace with a br
          // if we can (to do so, we must put the condition before a possible
          // value).
          if (!curr->value ||
              EffectAnalyzer::canReorder(
                passOptions, *getModule(), curr->condition, curr->value)) {
            Builder builder(*getModule());
            replaceCurrent(builder.makeSequence(
              builder.makeDrop(curr->condition), // might have side effects
              builder.makeBreak(curr->default_, curr->value)));
          }
        }
      }

      // Restructuring of ifs: if we have
      //   (block $x
      //     (drop (br_if $x (cond)))
      //     .., no other references to $x
      //   )
      // then we can turn that into (if (!cond) ..).
      // Code size wise, we turn the block into an if (no change), and
      // lose the br_if (-2). .. turns into the body of the if in the binary
      // format. We need to flip the condition, which at worst adds 1.
      // If the block has a return value, we can do something similar, removing
      // the drop from the br_if and putting the if on the outside,
      //   (block $x
      //     (drop (br_if $x (value) (cond)))
      //     .., no other references to $x
      //     ..final element..
      //   )
      // =>
      //   (if
      //     (cond)
      //     (value) ;; must not have side effects!
      //     (block
      //       .., no other references to $x
      //       ..final element..
      //     )
      //   )
      // This is beneficial as the block will likely go away in the binary
      // format (the if arm is an implicit block), and the drop is removed.
      void restructureIf(Block* curr) {
        auto& list = curr->list;
        // We should be called only on potentially-interesting lists.
        assert(list.size() >= 2);
        if (!curr->name.is()) {
          return;
        }
        Builder builder(*getModule());
        if (!shrink2) {
          Break* br = nullptr;
          Drop* drop = list[0]->dynCast<Drop>();
          if (drop) {
            br = drop->value->dynCast<Break>();
          } else {
            br = list[0]->dynCast<Break>();
          }
          if (br && br->condition && br->name == curr->name &&
              br->type != Type::unreachable) {
            if (BranchUtils::BranchSeeker::count(curr, curr->name) == 1) {
              if (!drop) {
                assert(!br->value);
                auto* iff = builder.makeIf(
                  builder.makeUnary(EqZInt32, br->condition), curr);
                replaceCurrent(iff);
                BranchHints::copyFlippedTo(br, iff, getFunction());
                ExpressionManipulator::nop(br);
                BranchHints::clear(br, getFunction());
                curr->finalize(curr->type);
                while (foldBoolIf &&
                       tryFoldBooleanIfCondition(
                         iff, getFunction(), *getModule(), passOptions)) {
                }
              } else {
                if (!EffectAnalyzer(passOptions, *getModule(), br->value)
                       .hasSideEffects()) {
                  if (EffectAnalyzer::canReorder(
                        passOptions, *getModule(), br->condition, br->value)) {
                    ExpressionManipulator::nop(list[0]);
                    auto* iff = builder.makeIf(br->condition, br->value, curr);
                    BranchHints::copyTo(br, iff, getFunction());
                    replaceCurrent(iff);
                    while (foldBoolIf &&
                           tryFoldBooleanIfCondition(
                             iff, getFunction(), *getModule(), passOptions)) {
                    }
                  }
                } else {
                  if (neverUnconditionalize) {
                    return;
                  }
                  Expression* old = list[0];
                  Nop nop;
                  list[0] = &nop;
                  auto canReorder = EffectAnalyzer::canReorder(
                    passOptions, *getModule(), br->condition, curr);
                  auto hasSideEffects =
                    EffectAnalyzer(passOptions, *getModule(), curr)
                      .hasSideEffects();
                  list[0] = old;
                  if (canReorder && !hasSideEffects &&
                      Properties::canEmitSelectWithArms(br->value, curr)) {
                    ExpressionManipulator::nop(list[0]);
                    replaceCurrent(
                      builder.makeSelect(br->condition, br->value, curr));
                  }
                }
              }
            }
          }
          return;
        }

        if (!v4) {
          // shrink2 (v3): scan list[k] (0 <= k < list.size() - 1) for the
          // single branch targeting curr->name: either a conditional Break /
          // Drop(Break) or a 1-arm If ending in an unconditional Break to
          // curr->name.
          for (Index k = 0; k + 1 < list.size(); k++) {
            if (list[k]->type == Type::unreachable) {
              return;
            }
            Break* br = nullptr;
            Drop* drop = list[k]->dynCast<Drop>();
            if (drop) {
              br = drop->value->dynCast<Break>();
            } else {
              br = list[k]->dynCast<Break>();
            }
            if (br && br->condition && br->name == curr->name &&
                br->type != Type::unreachable) {
              if (BranchUtils::BranchSeeker::count(curr, curr->name) != 1) {
                return;
              }
              if (!drop) {
                assert(!br->value);
                if (curr->type != Type::none) {
                  return;
                }
                if (k == 0) {
                  if (passOptions.trapsNeverHappen && list.size() >= 3 &&
                      list.back()->is<Unreachable>() &&
                      !list[list.size() - 2]->type.isConcrete()) {
                    list.pop_back();
                  }
                  auto* iff = builder.makeIf(
                    builder.makeUnary(EqZInt32, br->condition), curr);
                  replaceCurrent(iff);
                  BranchHints::copyFlippedTo(br, iff, getFunction());
                  ExpressionManipulator::nop(br);
                  BranchHints::clear(br, getFunction());
                  curr->finalize(curr->type);
                  while (foldBoolIf &&
                         tryFoldBooleanIfCondition(
                           iff, getFunction(), *getModule(), passOptions)) {
                  }
                } else {
                  Index endIdx = list.size();
                  if (passOptions.trapsNeverHappen && endIdx > k + 2 &&
                      list.back()->is<Unreachable>() &&
                      !list[endIdx - 2]->type.isConcrete()) {
                    endIdx--;
                  }
                  Expression* body = stealSlice(builder, curr, k + 1, endIdx);
                  auto* iff =
                    builder.makeIf(negateCond(builder, br->condition), body);
                  BranchHints::copyFlippedTo(br, iff, getFunction());
                  BranchHints::clear(br, getFunction());
                  while (foldBoolIf &&
                         tryFoldBooleanIfCondition(
                           iff, getFunction(), *getModule(), passOptions)) {
                  }
                  list[k] = iff;
                  list.resize(k + 1);
                  curr->name = Name();
                  curr->finalize(curr->type);
                }
                return;
              }

              // drop(br_if $curr VAL COND)
              if (!EffectAnalyzer(passOptions, *getModule(), br->value)
                     .hasSideEffects() &&
                  EffectAnalyzer::canReorder(
                    passOptions, *getModule(), br->condition, br->value)) {
                if (passOptions.trapsNeverHappen && curr->type.isConcrete() &&
                    k + 2 < list.size() && list.back()->is<Unreachable>()) {
                  bool tailVoid = true;
                  for (Index j = k + 1; j + 1 < list.size(); j++) {
                    if (list[j]->type.isConcrete()) {
                      tailVoid = false;
                      break;
                    }
                  }
                  if (tailVoid) {
                    Expression* noreturnBody =
                      stealSlice(builder, curr, k + 1, list.size() - 1);
                    auto* iff = builder.makeIf(
                      negateCond(builder, br->condition), noreturnBody);
                    BranchHints::copyFlippedTo(br, iff, getFunction());
                    BranchHints::clear(br, getFunction());
                    while (foldBoolIf &&
                           tryFoldBooleanIfCondition(
                             iff, getFunction(), *getModule(), passOptions)) {
                    }
                    Expression* val = br->value;
                    list[k] = iff;
                    list[k + 1] = val;
                    list.resize(k + 2);
                    curr->name = Name();
                    curr->finalize(curr->type);
                    refinalize = true;
                    return;
                  }
                }
                if (k == 0) {
                  ExpressionManipulator::nop(list[0]);
                  auto* iff = builder.makeIf(br->condition, br->value, curr);
                  BranchHints::copyTo(br, iff, getFunction());
                  replaceCurrent(iff);
                  while (foldBoolIf &&
                         tryFoldBooleanIfCondition(
                           iff, getFunction(), *getModule(), passOptions)) {
                  }
                  if (iff->type != curr->type) {
                    refinalize = true;
                  }
                  if (auto* seq = tryUnnestOneUnreachableArmIf(iff)) {
                    replaceCurrent(seq);
                  }
                } else {
                  Expression* falseArm =
                    stealSlice(builder, curr, k + 1, list.size());
                  auto* iff = builder.makeIf(
                    br->condition, br->value, falseArm, curr->type);
                  BranchHints::copyTo(br, iff, getFunction());
                  BranchHints::clear(br, getFunction());
                  while (foldBoolIf &&
                         tryFoldBooleanIfCondition(
                           iff, getFunction(), *getModule(), passOptions)) {
                  }
                  list[k] = iff;
                  list.resize(k + 1);
                  curr->name = Name();
                  curr->finalize(curr->type);
                  if (iff->type != curr->type) {
                    refinalize = true;
                  }
                }
                return;
              }
              if (passOptions.trapsNeverHappen && curr->type.isConcrete() &&
                  k + 2 < list.size() && list.back()->is<Unreachable>()) {
                if (auto* tee = br->value->dynCast<LocalSet>();
                    tee && tee->isTee() &&
                    Type::isSubType(tee->type, curr->type) &&
                    !EffectAnalyzer(passOptions, *getModule(), br->condition)
                       .localsWritten.contains(tee->index)) {
                  bool tailVoid = true;
                  for (Index j = k + 1; j + 1 < list.size(); j++) {
                    if (list[j]->type.isConcrete()) {
                      tailVoid = false;
                      break;
                    }
                  }
                  if (tailVoid) {
                    Expression* noreturnBody =
                      stealSlice(builder, curr, k + 1, list.size() - 1);
                    auto* iff = builder.makeIf(
                      negateCond(builder, br->condition), noreturnBody);
                    BranchHints::copyFlippedTo(br, iff, getFunction());
                    BranchHints::clear(br, getFunction());
                    while (foldBoolIf &&
                           tryFoldBooleanIfCondition(
                             iff, getFunction(), *getModule(), passOptions)) {
                    }
                    Type localType = getFunction()->getLocalType(tee->index);
                    tee->makeSet();
                    list[k] = tee;
                    list[k + 1] = iff;
                    if (k + 2 < list.size()) {
                      list[k + 2] = builder.makeLocalGet(tee->index, localType);
                      list.resize(k + 3);
                    } else {
                      list.push_back(
                        builder.makeLocalGet(tee->index, localType));
                    }
                    curr->name = Name();
                    curr->finalize(curr->type);
                    refinalize = true;
                    return;
                  }
                }
              }
              if (k == 0 && !neverUnconditionalize) {
                Expression* old = list[0];
                Nop nop;
                list[0] = &nop;
                auto canReorder = EffectAnalyzer::canReorder(
                  passOptions, *getModule(), br->condition, curr);
                auto hasSideEffects =
                  EffectAnalyzer(passOptions, *getModule(), curr)
                    .hasSideEffects();
                list[0] = old;
                if (canReorder && !hasSideEffects &&
                    Properties::canEmitSelectWithArms(br->value, curr)) {
                  ExpressionManipulator::nop(list[0]);
                  replaceCurrent(
                    builder.makeSelect(br->condition, br->value, curr));
                }
              }
              return;
            }

            // Part 1(a): 1-arm If ending in unconditional (br $curr [VAL]).
            if (auto* iff = list[k]->dynCast<If>();
                iff && !iff->ifFalse &&
                iff->condition->type != Type::unreachable) {
              std::vector<Expression*> thenPrefix;
              Break* thenBr = iff->ifTrue->dynCast<Break>();
              if (!thenBr) {
                if (auto* tb = iff->ifTrue->dynCast<Block>();
                    tb && !tb->name.is() && !tb->list.empty()) {
                  if (auto* b = tb->list.back()->dynCast<Break>()) {
                    bool prefixOk = true;
                    for (Index j = 0; j + 1 < tb->list.size(); j++) {
                      if (tb->list[j]->type != Type::none) {
                        prefixOk = false;
                        break;
                      }
                      thenPrefix.push_back(tb->list[j]);
                    }
                    if (prefixOk) {
                      thenBr = b;
                    } else {
                      thenPrefix.clear();
                    }
                  }
                }
              }
              if (thenBr && !thenBr->condition && thenBr->name == curr->name) {
                if (BranchUtils::BranchSeeker::count(curr, curr->name) != 1) {
                  return;
                }
                if (curr->type == Type::none) {
                  if (thenBr->value) {
                    return;
                  }
                } else if (curr->type.isConcrete()) {
                  if (!thenBr->value ||
                      !Type::isSubType(thenBr->value->type, curr->type)) {
                    return;
                  }
                } else {
                  return;
                }

                // Noreturn-tail inversion when curr is concrete and ends in
                // ...noreturn_tail... (unreachable).
                if (passOptions.trapsNeverHappen && curr->type.isConcrete() &&
                    k + 2 < list.size() && list.back()->is<Unreachable>()) {
                  bool tailVoid = true;
                  for (Index j = k + 1; j + 1 < list.size(); j++) {
                    if (list[j]->type.isConcrete()) {
                      tailVoid = false;
                      break;
                    }
                  }
                  if (tailVoid) {
                    Expression* noreturnBody =
                      stealSlice(builder, curr, k + 1, list.size() - 1);
                    Expression* brVal = thenBr->value;
                    BranchHints::clear(thenBr, getFunction());
                    iff->condition = negateCond(builder, iff->condition);
                    iff->ifTrue = noreturnBody;
                    BranchHints::flip(iff, getFunction());
                    iff->finalize();
                    while (foldBoolIf &&
                           tryFoldBooleanIfCondition(
                             iff, getFunction(), *getModule(), passOptions)) {
                    }
                    list.resize(k + 1);
                    for (auto* item : thenPrefix) {
                      list.push_back(item);
                    }
                    list.push_back(brVal);
                    curr->name = Name();
                    curr->finalize(curr->type);
                    refinalize = true;
                    return;
                  }
                }

                // Normal 1-arm If -> 2-arm If restructuring.
                if (curr->type == Type::none) {
                  Index endIdx = list.size();
                  if (passOptions.trapsNeverHappen && endIdx > k + 2 &&
                      list.back()->is<Unreachable>() &&
                      !list[endIdx - 2]->type.isConcrete()) {
                    endIdx--;
                  }
                  Expression* falseArm =
                    stealSlice(builder, curr, k + 1, endIdx);
                  BranchHints::clear(thenBr, getFunction());
                  if (thenPrefix.empty()) {
                    iff->condition = negateCond(builder, iff->condition);
                    iff->ifTrue = falseArm;
                    BranchHints::flip(iff, getFunction());
                  } else {
                    auto* tb = iff->ifTrue->cast<Block>();
                    tb->list.pop_back();
                    if (tb->list.size() == 1) {
                      iff->ifTrue = tb->list[0];
                    } else {
                      tb->finalize();
                    }
                    iff->ifFalse = falseArm;
                  }
                  iff->finalize(Type::none);
                } else {
                  Expression* falseArm =
                    stealSlice(builder, curr, k + 1, list.size());
                  Expression* brVal = thenBr->value;
                  BranchHints::clear(thenBr, getFunction());
                  if (thenPrefix.empty()) {
                    iff->ifTrue = brVal;
                  } else {
                    auto* tb = iff->ifTrue->cast<Block>();
                    tb->list.back() = brVal;
                    tb->finalize();
                  }
                  iff->ifFalse = falseArm;
                  iff->finalize(curr->type);
                }
                while (foldBoolIf &&
                       tryFoldBooleanIfCondition(
                         iff, getFunction(), *getModule(), passOptions)) {
                }
                Expression* rep = iff;
                if (auto* sel = selectify(iff)) {
                  if (!sel->type.isRef() || sel->type == curr->type ||
                      sel->type.getHeapType().isBasic()) {
                    rep = sel;
                  }
                }
                if (k == 0) {
                  curr->name = Name();
                  if (auto* repIf = rep->dynCast<If>()) {
                    if (auto* seq = tryUnnestOneUnreachableArmIf(repIf)) {
                      rep = seq;
                    }
                  }
                  replaceCurrent(rep);
                  if (rep->type != curr->type) {
                    refinalize = true;
                  }
                } else {
                  list[k] = rep;
                  list.resize(k + 1);
                  curr->name = Name();
                  curr->finalize(curr->type);
                  if (rep->type != curr->type) {
                    refinalize = true;
                  }
                }
                return;
              }
            }
            if (BranchUtils::BranchSeeker::has(list[k], curr->name)) {
              return;
            }
          }
          return;
        }

        // shrink2 (v4): iteratively peel branches targeting curr->name from the
        // tail of curr->list backward (supporting multi-branch blocks,
        // 1-arm Ifs, if/else-if chains, 2-level nested 1-arm Ifs, and
        // drop(br_if) / void br_if).
        auto peelTailDropBrIfs = [&]() {
          while (curr->list.size() >= 2 && curr->name.is()) {
            size_t sz = curr->list.size();
            auto* secondLast = curr->list[sz - 2];
            auto* last = curr->list[sz - 1];
            if (curr->type == Type::i32) {
              int32_t lastConst = 0;
              if (isI32Const(last, lastConst) &&
                  (lastConst == 0 || lastConst == 1)) {
                if (auto* iff = secondLast->dynCast<If>();
                    iff && !iff->ifFalse &&
                    iff->condition->type != Type::unreachable &&
                    BranchUtils::BranchSeeker::count(iff, curr->name) == 1) {
                  Block* tb = iff->ifTrue->dynCast<Block>();
                  Expression** armEndPtr = nullptr;
                  if (tb && !tb->name.is() && !tb->list.empty()) {
                    bool ok = true;
                    for (Index j = 0; j + 1 < tb->list.size(); j++) {
                      if (tb->list[j]->type != Type::none) {
                        ok = false;
                        break;
                      }
                    }
                    if (ok) {
                      armEndPtr = &tb->list.back();
                    }
                  } else if (!tb) {
                    armEndPtr = &iff->ifTrue;
                  }
                  if (armEndPtr) {
                    if (auto* d = (*armEndPtr)->dynCast<Drop>()) {
                      if (auto* b = d->value->dynCast<Break>();
                          b && b->condition && b->value &&
                          b->name == curr->name &&
                          b->type != Type::unreachable) {
                        int32_t brConst = 0;
                        if (isI32Const(b->value, brConst) &&
                            ((brConst == 1 && lastConst == 0) ||
                             (brConst == 0 && lastConst == 1))) {
                          Expression* condVal =
                            (brConst == 1)
                              ? makeOneBitCond(builder, b->condition)
                              : makeNegOneBitCond(builder, b->condition, true);
                          BranchHints::clear(b, getFunction());
                          *armEndPtr = builder.makeBreak(curr->name, condVal);
                          if (tb) {
                            tb->finalize();
                          }
                          iff->finalize();
                          refinalize = true;
                          break;
                        }
                      }
                    }
                  }
                }
              }
            }
            auto* drop = secondLast->dynCast<Drop>();
            if (!drop) {
              break;
            }
            auto* br = drop->value->dynCast<Break>();
            if (!br || !br->value || !br->condition ||
                br->name != curr->name || br->type == Type::unreachable ||
                last->type == Type::unreachable) {
              break;
            }
            if (ExpressionAnalyzer::equal(br->value, last)) {
              EffectAnalyzer valEff(passOptions, *getModule(), br->value);
              if (!valEff.hasUnremovableSideEffects()) {
                EffectAnalyzer condEff(
                  passOptions, *getModule(), br->condition);
                if (!condEff.invalidates(valEff)) {
                  BranchHints::clear(br, getFunction());
                  if (!condEff.hasUnremovableSideEffects()) {
                    curr->list[sz - 2] = last;
                    curr->list.pop_back();
                    continue;
                  }
                  drop->value = br->condition;
                }
              }
              break;
            }
            int32_t brConst = 0, lastConst = 0;
            if (isI32Const(br->value, brConst) &&
                isI32Const(last, lastConst) &&
                ((brConst == 0 && lastConst == 1) ||
                 (brConst == 1 && lastConst == 0))) {
              Expression* rep = nullptr;
              if (brConst == 0 && lastConst == 1) {
                if (auto* u = br->condition->dynCast<Unary>();
                    u && u->op == EqZInt32 && isOneBitI32(u->value)) {
                  rep = u->value;
                } else {
                  rep = makeNegatedBool(builder, br->condition, shrink2);
                }
              } else {
                if (isOneBitI32(br->condition)) {
                  rep = br->condition;
                } else {
                  rep = builder.makeBinary(
                    NeInt32, br->condition, builder.makeConst(int32_t(0)));
                }
              }
              BranchHints::clear(br, getFunction());
              curr->list[sz - 2] = rep;
              curr->list.pop_back();
              continue;
            }
            if (!neverUnconditionalize &&
                (!curr->type.isRef() ||
                 (sz == 2 &&
                  BranchUtils::BranchSeeker::count(curr, curr->name) == 1)) &&
                Properties::canEmitSelectWithArms(br->value, last) &&
                !tooCostlyToRunUnconditionally(
                  passOptions, br->value, last)) {
              EffectAnalyzer valEff(passOptions, *getModule(), br->value);
              EffectAnalyzer lastEff(passOptions, *getModule(), last);
              if (!valEff.hasSideEffects() && !lastEff.hasSideEffects()) {
                EffectAnalyzer condEff(
                  passOptions, *getModule(), br->condition);
                if (!condEff.invalidates(valEff) &&
                    EffectAnalyzer::canReorder(
                      passOptions, *getModule(), br->condition, last)) {
                  auto* sel =
                    builder.makeSelect(br->condition, br->value, last);
                  if (!sel->type.isRef() || sel->type == curr->type ||
                      sel->type.getHeapType().isBasic()) {
                    if (sel->type != curr->type) {
                      refinalize = true;
                    }
                    BranchHints::clear(br, getFunction());
                    curr->list[sz - 2] = sel;
                    curr->list.pop_back();
                    continue;
                  }
                }
              }
            }
            break;
          }
        };

        auto extractThenBr = [&](Expression* arm,
                                 std::vector<Expression*>& prefixOut) -> Break* {
          prefixOut.clear();
          if (auto* br = arm->dynCast<Break>()) {
            return br;
          }
          if (auto* tb = arm->dynCast<Block>();
              tb && !tb->name.is() && !tb->list.empty()) {
            if (auto* b = tb->list.back()->dynCast<Break>()) {
              for (Index j = 0; j + 1 < tb->list.size(); j++) {
                if (tb->list[j]->type != Type::none) {
                  prefixOut.clear();
                  return nullptr;
                }
                prefixOut.push_back(tb->list[j]);
              }
              return b;
            }
          }
          return nullptr;
        };

        auto matchBrIfChain =
          [&](If* rootIf,
              Index expectedRefs,
              std::vector<If*>& chainIfs,
              std::vector<std::vector<Expression*>>& chainPrefixes,
              std::vector<Break*>& chainBrs) -> bool {
          chainIfs.clear();
          chainPrefixes.clear();
          chainBrs.clear();
          If* cur = rootIf;
          while (cur) {
            if (cur->condition->type == Type::unreachable) {
              return false;
            }
            std::vector<Expression*> prefix;
            Break* thenBr = extractThenBr(cur->ifTrue, prefix);
            if (!thenBr || thenBr->condition || thenBr->name != curr->name) {
              return false;
            }
            if (curr->type == Type::none) {
              if (thenBr->value) {
                return false;
              }
            } else if (curr->type.isConcrete()) {
              if (!thenBr->value ||
                  !Type::isSubType(thenBr->value->type, curr->type)) {
                return false;
              }
            } else {
              return false;
            }
            chainIfs.push_back(cur);
            chainPrefixes.push_back(std::move(prefix));
            chainBrs.push_back(thenBr);
            if (!cur->ifFalse) {
              break;
            }
            cur = cur->ifFalse->dynCast<If>();
            if (!cur) {
              return false;
            }
          }
          return chainIfs.size() == expectedRefs;
        };

        auto isFreeNegCond = [](Expression* c) {
          if (auto* u = c->dynCast<Unary>(); u && u->op == EqZInt32) {
            return true;
          }
          return canInvertBoolExprFree(c);
        };

        int typeBytes =
          (curr->type == Type::none)
            ? 0
            : ((curr->type.isRef() && !curr->type.getHeapType().isBasic()) ? 2
                                                                           : 1);

        auto canProfitablyPeelAll = [&](Index upToK) -> bool {
          int totalDelta = 0;
          Index firstBrIdx = upToK + 1;
          for (Index i = 0; i <= upToK; i++) {
            Index refs = BranchUtils::BranchSeeker::count(list[i], curr->name);
            if (refs == 0) {
              continue;
            }
            if (firstBrIdx > upToK) {
              firstBrIdx = i;
            }
            Break* br = nullptr;
            Drop* drop = list[i]->dynCast<Drop>();
            if (drop) {
              br = drop->value->dynCast<Break>();
            } else {
              br = list[i]->dynCast<Break>();
            }
            if (br && br->condition && br->name == curr->name &&
                br->type != Type::unreachable) {
              if (refs != 1) {
                return false;
              }
              if (!drop) {
                if (curr->type != Type::none) {
                  return false;
                }
                totalDelta += isFreeNegCond(br->condition) ? -1 : 1;
                continue;
              }
              if (!curr->type.isConcrete() ||
                  EffectAnalyzer(passOptions, *getModule(), br->value)
                    .hasSideEffects() ||
                  !EffectAnalyzer::canReorder(
                    passOptions, *getModule(), br->condition, br->value)) {
                return false;
              }
              bool condEqz = false;
              if (auto* u = br->condition->dynCast<Unary>();
                  u && u->op == EqZInt32) {
                condEqz = true;
              }
              totalDelta += typeBytes - (condEqz ? 1 : 0);
              continue;
            }
            if (auto* iff = list[i]->dynCast<If>()) {
              std::vector<If*> cIfs;
              std::vector<std::vector<Expression*>> cPrefixes;
              std::vector<Break*> cBrs;
              if (matchBrIfChain(iff, refs, cIfs, cPrefixes, cBrs)) {
                int d = int(cIfs.size());
                if (curr->type == Type::none) {
                  if (d == 1 && cPrefixes[0].empty()) {
                    totalDelta += isFreeNegCond(iff->condition) ? -2 : -1;
                  } else {
                    totalDelta += -(2 * d - 1);
                  }
                } else {
                  bool condEqz = false;
                  if (d == 1 && cPrefixes[0].empty()) {
                    if (auto* u = iff->condition->dynCast<Unary>();
                        u && u->op == EqZInt32) {
                      condEqz = true;
                    }
                  }
                  totalDelta += typeBytes + 1 - 2 * d - (condEqz ? 1 : 0);
                }
                continue;
              }
            }
            return false;
          }
          if (firstBrIdx > upToK) {
            return false;
          }
          if (curr->type == Type::none) {
            int blockSavings = (curr != getFunction()->body) ? 3 : 0;
            return totalDelta - blockSavings < 0;
          }
          int blockSavings = 2 + typeBytes;
          return totalDelta - blockSavings <= 0;
        };

        auto finishPeelAtK = [&](Index k,
                                 Expression* rep,
                                 bool isLastRefInCurr) -> bool {
          if (isLastRefInCurr) {
            if (k == 0) {
              curr->name = Name();
              if (auto* repIf = rep->dynCast<If>()) {
                if (auto* seq = tryUnnestOneUnreachableArmIf(repIf)) {
                  rep = seq;
                }
              }
              replaceCurrent(rep);
              if (rep->type != curr->type) {
                refinalize = true;
              }
            } else {
              list[k] = rep;
              list.resize(k + 1);
              curr->name = Name();
              curr->finalize(curr->type);
              if (rep->type != curr->type) {
                refinalize = true;
              }
            }
            return false;
          }
          list[k] = rep;
          list.resize(k + 1);
          curr->finalize(curr->type);
          if (rep->type != curr->type) {
            refinalize = true;
          }
          peelTailDropBrIfs();
          return true;
        };

        while (curr->name.is() && list.size() >= 2) {
          for (Index i = 0; i + 1 < list.size(); i++) {
            if (list[i]->type == Type::unreachable) {
              return;
            }
          }
          if (BranchUtils::BranchSeeker::count(list.back(), curr->name) != 0) {
            return;
          }
          Index k = list.size();
          Index refsInK = 0;
          for (Index i = list.size() - 1; i > 0; i--) {
            Index c = BranchUtils::BranchSeeker::count(list[i - 1], curr->name);
            if (c > 0) {
              k = i - 1;
              refsInK = c;
              break;
            }
          }
          if (k == list.size()) {
            curr->name = Name();
            curr->finalize(curr->type);
            refinalize = true;
            return;
          }
          Index earlierRefs = 0;
          for (Index i = 0; i < k; i++) {
            earlierRefs +=
              BranchUtils::BranchSeeker::count(list[i], curr->name);
          }
          bool isLastRefInCurr = (earlierRefs == 0);

          // Check whether list[k + 1 .. list.size() - 2] are all void.
          bool midVoid = true;
          for (Index j = k + 1; j + 1 < list.size(); j++) {
            if (list[j]->type != Type::none) {
              midVoid = false;
              break;
            }
          }

          Break* br = nullptr;
          Drop* drop = list[k]->dynCast<Drop>();
          if (drop) {
            br = drop->value->dynCast<Break>();
          } else {
            br = list[k]->dynCast<Break>();
          }
          if (br && br->condition && br->name == curr->name &&
              br->type != Type::unreachable) {
            if (refsInK != 1) {
              return;
            }
            if (!drop) {
              assert(!br->value);
              if (curr->type != Type::none ||
                  curr->name.startsWith("sc_fail$") ||
                  curr->name.startsWith("sc_do$") ||
                  (curr == getFunction()->body &&
                   getFunction()->getResults() == Type::none)) {
                return;
              }
              if (!isLastRefInCurr && !canProfitablyPeelAll(k)) {
                return;
              }
              Index endIdx = list.size();
              if (passOptions.trapsNeverHappen && endIdx > k + 2 &&
                  list.back()->is<Unreachable>() &&
                  !list[endIdx - 2]->type.isConcrete()) {
                endIdx--;
              }
              Expression* body = stealSlice(builder, curr, k + 1, endIdx);
              auto* iff = builder.makeIf(
                makeNegatedBool(builder, br->condition, shrink2), body);
              BranchHints::copyFlippedTo(br, iff, getFunction());
              BranchHints::clear(br, getFunction());
              while (foldBoolIf &&
                     tryFoldBooleanIfCondition(
                       iff, getFunction(), *getModule(), passOptions)) {
              }
              if (!finishPeelAtK(k, iff, isLastRefInCurr)) {
                return;
              }
              continue;
            }

            // drop(br_if $curr VAL COND)
            if (!EffectAnalyzer(passOptions, *getModule(), br->value)
                   .hasSideEffects() &&
                EffectAnalyzer::canReorder(
                  passOptions, *getModule(), br->condition, br->value)) {
              // (1) Noreturn-tail inversion: valid for any totalRefs >= 1.
              if (passOptions.trapsNeverHappen && curr->type.isConcrete() &&
                  k + 2 < list.size() && list.back()->is<Unreachable>() &&
                  midVoid) {
                Expression* noreturnBody =
                  stealSlice(builder, curr, k + 1, list.size() - 1);
                auto* iff = builder.makeIf(
                  makeNegatedBool(builder, br->condition, shrink2),
                  noreturnBody);
                BranchHints::copyFlippedTo(br, iff, getFunction());
                BranchHints::clear(br, getFunction());
                while (foldBoolIf &&
                       tryFoldBooleanIfCondition(
                         iff, getFunction(), *getModule(), passOptions)) {
                }
                Expression* val = br->value;
                list[k] = iff;
                list[k + 1] = val;
                list.resize(k + 2);
                if (isLastRefInCurr) {
                  curr->name = Name();
                }
                curr->finalize(curr->type);
                refinalize = true;
                if (isLastRefInCurr) {
                  return;
                }
                peelTailDropBrIfs();
                continue;
              }
              // (2) Shared tail value `br->value == list.back()` with void
              // intermediate statements `list[k + 1 .. list.size() - 2]`.
              if (curr->type.isConcrete() && k + 2 < list.size() && midVoid &&
                  ExpressionAnalyzer::equal(br->value, list.back())) {
                Expression* midBody =
                  stealSlice(builder, curr, k + 1, list.size() - 1);
                Expression* lastVal = list.back();
                auto* iff = builder.makeIf(
                  makeNegatedBool(builder, br->condition, shrink2), midBody);
                BranchHints::copyFlippedTo(br, iff, getFunction());
                BranchHints::clear(br, getFunction());
                while (foldBoolIf &&
                       tryFoldBooleanIfCondition(
                         iff, getFunction(), *getModule(), passOptions)) {
                }
                list[k] = iff;
                list[k + 1] = lastVal;
                list.resize(k + 2);
                if (isLastRefInCurr) {
                  curr->name = Name();
                }
                curr->finalize(curr->type);
                refinalize = true;
                if (isLastRefInCurr) {
                  return;
                }
                peelTailDropBrIfs();
                continue;
              }
              // (3) General 2-arm If peel.
              if (!isLastRefInCurr) {
                bool singleStepFree = false;
                if (typeBytes == 1) {
                  if (auto* u = br->condition->dynCast<Unary>();
                      u && u->op == EqZInt32) {
                    singleStepFree = true;
                  }
                }
                if (!singleStepFree && !canProfitablyPeelAll(k)) {
                  return;
                }
              }
              Expression* falseArm =
                stealSlice(builder, curr, k + 1, list.size());
              auto* iff =
                builder.makeIf(br->condition, br->value, falseArm, curr->type);
              BranchHints::copyTo(br, iff, getFunction());
              BranchHints::clear(br, getFunction());
              while (foldBoolIf &&
                     tryFoldBooleanIfCondition(
                       iff, getFunction(), *getModule(), passOptions)) {
              }
              if (auto* u = iff->condition->dynCast<Unary>();
                  u && u->op == EqZInt32) {
                iff->condition = u->value;
                std::swap(iff->ifTrue, iff->ifFalse);
                BranchHints::flip(iff, getFunction());
              }
              Expression* rep = iff;
              if (auto* sel = selectify(iff)) {
                if (!sel->type.isRef() || sel->type == curr->type ||
                    sel->type.getHeapType().isBasic()) {
                  rep = sel;
                }
              }
              if (!finishPeelAtK(k, rep, isLastRefInCurr)) {
                return;
              }
              continue;
            }
            if (isLastRefInCurr && passOptions.trapsNeverHappen &&
                curr->type.isConcrete() && k + 2 < list.size() &&
                list.back()->is<Unreachable>() && midVoid) {
              if (auto* tee = br->value->dynCast<LocalSet>();
                  tee && tee->isTee() &&
                  Type::isSubType(tee->type, curr->type) &&
                  !EffectAnalyzer(passOptions, *getModule(), br->condition)
                     .localsWritten.contains(tee->index)) {
                Expression* noreturnBody =
                  stealSlice(builder, curr, k + 1, list.size() - 1);
                auto* iff = builder.makeIf(
                  makeNegatedBool(builder, br->condition, shrink2),
                  noreturnBody);
                BranchHints::copyFlippedTo(br, iff, getFunction());
                BranchHints::clear(br, getFunction());
                while (foldBoolIf &&
                       tryFoldBooleanIfCondition(
                         iff, getFunction(), *getModule(), passOptions)) {
                }
                Type localType = getFunction()->getLocalType(tee->index);
                tee->makeSet();
                list[k] = tee;
                list[k + 1] = iff;
                if (k + 2 < list.size()) {
                  list[k + 2] = builder.makeLocalGet(tee->index, localType);
                  list.resize(k + 3);
                } else {
                  list.push_back(builder.makeLocalGet(tee->index, localType));
                }
                curr->name = Name();
                curr->finalize(curr->type);
                refinalize = true;
                return;
              }
            }
            if (isLastRefInCurr && k == 0 && !neverUnconditionalize) {
              Expression* old = list[0];
              Nop nop;
              list[0] = &nop;
              auto canReorder = EffectAnalyzer::canReorder(
                passOptions, *getModule(), br->condition, curr);
              auto hasSideEffects =
                EffectAnalyzer(passOptions, *getModule(), curr).hasSideEffects();
              list[0] = old;
              if (canReorder && !hasSideEffects &&
                  Properties::canEmitSelectWithArms(br->value, curr)) {
                ExpressionManipulator::nop(list[0]);
                replaceCurrent(
                  builder.makeSelect(br->condition, br->value, curr));
              }
            }
            return;
          }

          // Part 1(a): 1-arm If, if/else-if chain, or 2-level nested 1-arm If
          // ending in unconditional (br $curr [VAL]).
          if (auto* iff = list[k]->dynCast<If>();
              iff && iff->condition->type != Type::unreachable) {
            std::vector<If*> chainIfs;
            std::vector<std::vector<Expression*>> chainPrefixes;
            std::vector<Break*> chainBrs;
            if (matchBrIfChain(
                  iff, refsInK, chainIfs, chainPrefixes, chainBrs)) {
              Index d = chainIfs.size();
              // (1) Noreturn-tail inversion for a 1-arm If.
              if (d == 1 && passOptions.trapsNeverHappen &&
                  curr->type.isConcrete() && k + 2 < list.size() &&
                  list.back()->is<Unreachable>() && midVoid) {
                Expression* noreturnBody =
                  stealSlice(builder, curr, k + 1, list.size() - 1);
                Expression* brVal = chainBrs[0]->value;
                BranchHints::clear(chainBrs[0], getFunction());
                iff->condition =
                  makeNegatedBool(builder, iff->condition, shrink2);
                iff->ifTrue = noreturnBody;
                BranchHints::flip(iff, getFunction());
                iff->finalize();
                while (foldBoolIf &&
                       tryFoldBooleanIfCondition(
                         iff, getFunction(), *getModule(), passOptions)) {
                }
                list.resize(k + 1);
                for (auto* item : chainPrefixes[0]) {
                  list.push_back(item);
                }
                list.push_back(brVal);
                if (isLastRefInCurr) {
                  curr->name = Name();
                }
                curr->finalize(curr->type);
                refinalize = true;
                if (isLastRefInCurr) {
                  return;
                }
                peelTailDropBrIfs();
                continue;
              }
              // (2) Shared tail value `thenBr->value == list.back()` for d == 1.
              if (d == 1 && curr->type.isConcrete() && midVoid &&
                  ExpressionAnalyzer::equal(chainBrs[0]->value, list.back()) &&
                  !EffectAnalyzer(passOptions, *getModule(), chainBrs[0]->value)
                     .hasSideEffects()) {
                Expression* lastVal = list.back();
                BranchHints::clear(chainBrs[0], getFunction());
                if (k + 2 == list.size()) {
                  if (chainPrefixes[0].empty()) {
                    list[k] = builder.makeDrop(iff->condition);
                  } else {
                    auto* tb = iff->ifTrue->cast<Block>();
                    tb->list.pop_back();
                    if (tb->list.size() == 1) {
                      iff->ifTrue = tb->list[0];
                    } else {
                      tb->finalize();
                    }
                    iff->finalize(Type::none);
                  }
                } else {
                  Expression* midBody =
                    stealSlice(builder, curr, k + 1, list.size() - 1);
                  if (chainPrefixes[0].empty()) {
                    iff->condition =
                      makeNegatedBool(builder, iff->condition, shrink2);
                    iff->ifTrue = midBody;
                    BranchHints::flip(iff, getFunction());
                  } else {
                    auto* tb = iff->ifTrue->cast<Block>();
                    tb->list.pop_back();
                    if (tb->list.size() == 1) {
                      iff->ifTrue = tb->list[0];
                    } else {
                      tb->finalize();
                    }
                    iff->ifFalse = midBody;
                  }
                  iff->finalize(Type::none);
                  while (foldBoolIf &&
                         tryFoldBooleanIfCondition(
                           iff, getFunction(), *getModule(), passOptions)) {
                  }
                  list[k + 1] = lastVal;
                  list.resize(k + 2);
                }
                if (isLastRefInCurr) {
                  curr->name = Name();
                }
                curr->finalize(curr->type);
                refinalize = true;
                if (isLastRefInCurr) {
                  return;
                }
                peelTailDropBrIfs();
                continue;
              }
              // (3) General 1-arm If / if-else-if chain peel.
              if (!isLastRefInCurr) {
                bool singleStepFree =
                  (typeBytes <= 1) || (d >= 2);
                if (!singleStepFree && d == 1 && chainPrefixes[0].empty()) {
                  if (auto* u = iff->condition->dynCast<Unary>();
                      u && u->op == EqZInt32) {
                    singleStepFree = true;
                  }
                }
                if (!singleStepFree && !canProfitablyPeelAll(k)) {
                  return;
                }
              }
              if (curr->type == Type::none) {
                Index endIdx = list.size();
                if (passOptions.trapsNeverHappen && endIdx > k + 2 &&
                    list.back()->is<Unreachable>() &&
                    !list[endIdx - 2]->type.isConcrete()) {
                  endIdx--;
                }
                Expression* falseArm = stealSlice(builder, curr, k + 1, endIdx);
                for (Index idx = 0; idx < d; idx++) {
                  BranchHints::clear(chainBrs[idx], getFunction());
                  If* curIf = chainIfs[idx];
                  if (idx + 1 == d && chainPrefixes[idx].empty()) {
                    curIf->condition =
                      makeNegatedBool(builder, curIf->condition, shrink2);
                    curIf->ifTrue = falseArm;
                    curIf->ifFalse = nullptr;
                    BranchHints::flip(curIf, getFunction());
                  } else {
                    if (chainPrefixes[idx].empty()) {
                      curIf->ifTrue = builder.makeNop();
                    } else {
                      auto* tb = curIf->ifTrue->cast<Block>();
                      tb->list.pop_back();
                      if (tb->list.size() == 1) {
                        curIf->ifTrue = tb->list[0];
                      } else {
                        tb->finalize();
                      }
                    }
                    if (idx + 1 == d) {
                      curIf->ifFalse = falseArm;
                    }
                  }
                }
                for (Index idx = d; idx > 0; idx--) {
                  If* curIf = chainIfs[idx - 1];
                  curIf->finalize(Type::none);
                  while (foldBoolIf &&
                         tryFoldBooleanIfCondition(
                           curIf, getFunction(), *getModule(), passOptions)) {
                  }
                }
              } else {
                Expression* falseArm =
                  stealSlice(builder, curr, k + 1, list.size());
                for (Index idx = 0; idx < d; idx++) {
                  Expression* brVal = chainBrs[idx]->value;
                  BranchHints::clear(chainBrs[idx], getFunction());
                  If* curIf = chainIfs[idx];
                  if (chainPrefixes[idx].empty()) {
                    curIf->ifTrue = brVal;
                  } else {
                    auto* tb = curIf->ifTrue->cast<Block>();
                    tb->list.back() = brVal;
                    tb->finalize();
                  }
                }
                chainIfs.back()->ifFalse = falseArm;
                for (Index idx = d; idx > 0; idx--) {
                  If* curIf = chainIfs[idx - 1];
                  curIf->finalize(curr->type);
                  while (foldBoolIf &&
                         tryFoldBooleanIfCondition(
                           curIf, getFunction(), *getModule(), passOptions)) {
                  }
                  if (auto* u = curIf->condition->dynCast<Unary>();
                      u && u->op == EqZInt32) {
                    curIf->condition = u->value;
                    std::swap(curIf->ifTrue, curIf->ifFalse);
                    BranchHints::flip(curIf, getFunction());
                  }
                  if (idx > 1) {
                    if (auto* sel = selectify(curIf)) {
                      if (!sel->type.isRef() || sel->type == curr->type ||
                          sel->type.getHeapType().isBasic()) {
                        chainIfs[idx - 2]->ifFalse = sel;
                      }
                    }
                  }
                }
              }
              Expression* rep = iff;
              if (auto* sel = selectify(iff)) {
                if (!sel->type.isRef() || sel->type == curr->type ||
                    sel->type.getHeapType().isBasic()) {
                  rep = sel;
                }
              }
              if (!finishPeelAtK(k, rep, isLastRefInCurr)) {
                return;
              }
              continue;
            }

            // (4) 2-level nested 1-arm If:
            // `(if C1 (then <prefix1> (if C2 (then <prefix2> (br $curr VAL)))))`
            if (!iff->ifFalse && refsInK == 1 && curr->type.isConcrete()) {
              std::vector<Expression*> prefix1;
              If* innerIf = iff->ifTrue->dynCast<If>();
              if (!innerIf) {
                if (auto* tb = iff->ifTrue->dynCast<Block>();
                    tb && !tb->name.is() && !tb->list.empty()) {
                  if (auto* lastIf = tb->list.back()->dynCast<If>()) {
                    bool ok = true;
                    for (Index j = 0; j + 1 < tb->list.size(); j++) {
                      if (tb->list[j]->type != Type::none) {
                        ok = false;
                        break;
                      }
                      prefix1.push_back(tb->list[j]);
                    }
                    if (ok) {
                      innerIf = lastIf;
                    } else {
                      prefix1.clear();
                    }
                  }
                }
              }
              if (innerIf && !innerIf->ifFalse &&
                  innerIf->condition->type != Type::unreachable) {
                std::vector<Expression*> prefix2;
                Break* thenBr = extractThenBr(innerIf->ifTrue, prefix2);
                if (thenBr && !thenBr->condition &&
                    thenBr->name == curr->name && thenBr->value &&
                    Type::isSubType(thenBr->value->type, curr->type)) {
                  // Shared tail value `thenBr->value == list.back()` with void
                  // intermediate statements `list[k + 1 .. list.size() - 2]`:
                  // rewrite to a void If `(if (if (result i32) C1 (then prefix1
                  // !C2) (else 1)) (then midBody))` + `list.back()`.
                  if (prefix2.empty() && k + 2 < list.size() && midVoid &&
                      ExpressionAnalyzer::equal(thenBr->value, list.back()) &&
                      !EffectAnalyzer(passOptions, *getModule(), thenBr->value)
                         .hasSideEffects()) {
                    Expression* midBody =
                      stealSlice(builder, curr, k + 1, list.size() - 1);
                    Expression* lastVal = list.back();
                    BranchHints::clear(thenBr, getFunction());
                    Expression* negC2 =
                      makeNegatedBool(builder, innerIf->condition, shrink2);
                    auto* condIf = builder.makeIf(
                      iff->condition,
                      prependToArm(builder, prefix1, negC2),
                      builder.makeConst(int32_t(1)),
                      Type::i32);
                    BranchHints::copyTo(iff, condIf, getFunction());
                    while (foldBoolIf &&
                           tryFoldBooleanIfCondition(
                             condIf, getFunction(), *getModule(), passOptions)) {
                    }
                    Expression* condExpr = condIf;
                    if (auto* sel = selectify(condIf)) {
                      condExpr = sel;
                    }
                    auto* newOuterIf = builder.makeIf(condExpr, midBody);
                    while (foldBoolIf &&
                           tryFoldBooleanIfCondition(newOuterIf,
                                                     getFunction(),
                                                     *getModule(),
                                                     passOptions)) {
                    }
                    list[k] = newOuterIf;
                    list[k + 1] = lastVal;
                    list.resize(k + 2);
                    if (isLastRefInCurr) {
                      curr->name = Name();
                    }
                    curr->finalize(curr->type);
                    refinalize = true;
                    if (isLastRefInCurr) {
                      return;
                    }
                    peelTailDropBrIfs();
                    continue;
                  }
                  if (isLastRefInCurr) {
                    auto* condIf = builder.makeIf(
                      iff->condition,
                      prependToArm(builder, prefix1, innerIf->condition),
                      builder.makeConst(int32_t(0)),
                      Type::i32);
                    BranchHints::copyTo(iff, condIf, getFunction());
                    while (foldBoolIf &&
                           tryFoldBooleanIfCondition(
                             condIf, getFunction(), *getModule(), passOptions)) {
                    }
                    Expression* condExpr = condIf;
                    if (auto* sel = selectify(condIf)) {
                      condExpr = sel;
                    }
                    if (k == 0 || condExpr != condIf) {
                      Expression* falseArm =
                        stealSlice(builder, curr, k + 1, list.size());
                      Expression* trueArm =
                        prependToArm(builder, prefix2, thenBr->value);
                      BranchHints::clear(thenBr, getFunction());
                      auto* newOuterIf = builder.makeIf(
                        condExpr, trueArm, falseArm, curr->type);
                      while (foldBoolIf &&
                             tryFoldBooleanIfCondition(newOuterIf,
                                                       getFunction(),
                                                       *getModule(),
                                                       passOptions)) {
                      }
                      Expression* rep = newOuterIf;
                      if (auto* sel = selectify(newOuterIf)) {
                        if (!sel->type.isRef() || sel->type == curr->type ||
                            sel->type.getHeapType().isBasic()) {
                          rep = sel;
                        }
                      }
                      finishPeelAtK(k, rep, true);
                      return;
                    }
                  }
                }
              }
            }
          }
          return;
        }
      }

      bool isBrIfChain(If* rootIf) {
        if (!rootIf || !rootIf->ifFalse || rootIf->type != Type::none) {
          return false;
        }
        Name targetName;
        If* cur = rootIf;
        Index count = 0;
        while (cur) {
          if (cur->condition->type == Type::unreachable) {
            return false;
          }
          Break* thenBr = cur->ifTrue->dynCast<Break>();
          if (!thenBr) {
            if (auto* tb = cur->ifTrue->dynCast<Block>();
                tb && !tb->name.is() && !tb->list.empty()) {
              if (auto* b = tb->list.back()->dynCast<Break>()) {
                bool ok = true;
                for (Index j = 0; j + 1 < tb->list.size(); j++) {
                  if (tb->list[j]->type != Type::none) {
                    ok = false;
                    break;
                  }
                }
                if (ok) {
                  thenBr = b;
                }
              }
            }
          }
          if (!thenBr || thenBr->condition) {
            return false;
          }
          if (count == 0) {
            targetName = thenBr->name;
          } else if (thenBr->name != targetName) {
            return false;
          }
          count++;
          if (!cur->ifFalse) {
            return count >= 2;
          }
          cur = cur->ifFalse->dynCast<If>();
          if (!cur) {
            return false;
          }
        }
        return false;
      }

      Block* tryUnnestOneUnreachableArmIf(If* iff,
                                          bool allowBrIfChain = false) {
        if (!shrink2 || !iff || !iff->ifFalse ||
            !(iff->type.isConcrete() || (v4 && iff->type == Type::none)) ||
            iff->condition->type == Type::unreachable) {
          return nullptr;
        }
        auto isArmUnreachable = [&](Expression* arm) {
          if (arm->type == Type::unreachable) {
            return true;
          }
          if (v4) {
            if (auto* b = arm->dynCast<Block>();
                b && !b->name.is() && !b->list.empty() &&
                b->list.back()->type == Type::unreachable) {
              return true;
            }
          }
          return false;
        };
        bool trueUnreach = isArmUnreachable(iff->ifTrue);
        bool falseUnreach = isArmUnreachable(iff->ifFalse);
        if (trueUnreach == falseUnreach) {
          return nullptr;
        }
        if (v4 && !allowBrIfChain && iff->type == Type::none &&
            isBrIfChain(iff)) {
          return nullptr;
        }
        bool unreachWasTrue = trueUnreach;
        Expression* unreachArm = unreachWasTrue ? iff->ifTrue : iff->ifFalse;
        Expression* concreteArm = unreachWasTrue ? iff->ifFalse : iff->ifTrue;
        bool canStripUnreach = false;
        if (passOptions.trapsNeverHappen) {
          if (auto* b = unreachArm->dynCast<Block>();
              b && !b->name.is() && b->list.size() >= 2 &&
              b->list.back()->is<Unreachable>() &&
              !b->list[b->list.size() - 2]->type.isConcrete()) {
            canStripUnreach = true;
          }
        }
        bool condIsEqz = false;
        if (auto* u = iff->condition->dynCast<Unary>();
            u && u->op == EqZInt32) {
          condIsEqz = true;
        } else if (v4 && (canInvertBoolExprFree(iff->condition) ||
                          canPushEqzIntoIfOrExprFree(iff->condition))) {
          condIsEqz = true;
        }
        bool indexedType =
          iff->type.isRef() && !iff->type.getHeapType().isBasic();
        if (!canStripUnreach && !unreachWasTrue && !condIsEqz && !indexedType) {
          return nullptr;
        }
        Builder builder(*getModule());
        if (canStripUnreach) {
          auto* b = unreachArm->cast<Block>();
          b->list.pop_back();
          if (b->list.size() == 1) {
            unreachArm = b->list[0];
          } else {
            b->finalize();
          }
        } else if (auto* b = unreachArm->dynCast<Block>();
                   b && !b->name.is()) {
          b->finalize();
        }
        if (!unreachWasTrue) {
          if (v4 && canPushEqzIntoIfOrExprFree(iff->condition) &&
              !canInvertBoolExprFree(iff->condition)) {
            pushEqzIntoIfOrExprFree(iff->condition, builder);
          } else {
            iff->condition =
              v4 ? makeNegatedBool(builder, iff->condition, true)
                 : negateCond(builder, iff->condition);
          }
          BranchHints::flip(iff, getFunction());
        }
        iff->ifTrue = unreachArm;
        iff->ifFalse = nullptr;
        iff->finalize();
        while (foldBoolIf &&
               tryFoldBooleanIfCondition(
                 iff, getFunction(), *getModule(), passOptions)) {
        }
        Expression* firstStmt = iff;
        if (v4 && iff->condition->type != Type::unreachable) {
          while (auto* b = iff->ifTrue->dynCast<Block>()) {
            if (!b->name.is() && b->list.size() == 1) {
              iff->ifTrue = b->list[0];
            } else {
              break;
            }
          }
          Drop* innerDrop = iff->ifTrue->dynCast<Drop>();
          Break* br = innerDrop ? innerDrop->value->dynCast<Break>()
                                : iff->ifTrue->dynCast<Break>();
          if (innerDrop &&
              (!br || !br->value || br->type == Type::unreachable)) {
            innerDrop = nullptr;
            br = nullptr;
          }
          if (br && !br->condition &&
              canTurnIfIntoBrIf(
                iff->condition, br->value, passOptions, *getModule())) {
            br->condition = iff->condition;
            BranchHints::copyTo(iff, br, getFunction());
            br->finalize();
            Expression* rep = innerDrop
                                ? static_cast<Expression*>(innerDrop)
                                : builder.dropIfConcretelyTyped(br);
            if (innerDrop) {
              innerDrop->finalize();
            }
            if (foldBoolIf) {
              if (auto* folded =
                    tryFoldBooleanBreakCondition(br,
                                                 getFunction(),
                                                 *getModule(),
                                                 passOptions,
                                                 neverUnconditionalize,
                                                 rep->dynCast<Drop>())) {
                rep = folded;
                if (auto* foldedIf = rep->dynCast<If>()) {
                  while (tryFoldBooleanIfCondition(
                    foldedIf, getFunction(), *getModule(), passOptions)) {
                  }
                }
              }
            }
            firstStmt = rep;
          }
        }
        if (v4) {
          if (auto* cIf = concreteArm->dynCast<If>()) {
            if (auto* subSeq =
                  tryUnnestOneUnreachableArmIf(cIf, allowBrIfChain)) {
              unnestedIfBlocks.erase(subSeq);
              concreteArm = subSeq;
            }
          }
        }
        auto* seq = builder.makeBlock(firstStmt);
        if (auto* cb = concreteArm->dynCast<Block>(); cb && !cb->name.is()) {
          unnestedIfBlocks.erase(cb);
          for (auto* item : cb->list) {
            seq->list.push_back(item);
          }
        } else {
          seq->list.push_back(concreteArm);
        }
        seq->finalize();
        unnestedIfBlocks.insert(seq);
        refinalize = true;
        return seq;
      }

      void visitUnary(Unary* curr) {
        if (!v4 || curr->op != EqZInt32) {
          return;
        }
        Builder builder(*getModule());
        if ((curr->value->is<If>() || curr->value->is<Select>()) &&
            canPushEqzIntoIfOrExprFree(curr->value)) {
          pushEqzIntoIfOrExprFree(curr->value, builder);
          Expression* rep = curr->value;
          if (auto* iff = rep->dynCast<If>()) {
            if (auto* sel = selectify(iff)) {
              rep = sel;
            }
          }
          replaceCurrent(rep);
          return;
        }
        if (auto* iff = curr->value->dynCast<If>();
            iff && iff->ifFalse && iff->type == Type::i32 &&
            iff->condition->type != Type::unreachable &&
            !neverUnconditionalize) {
          bool trueFree = canInvertBoolExprFree(iff->ifTrue);
          bool falseFree = canInvertBoolExprFree(iff->ifFalse);
          if ((trueFree || falseFree) &&
              Properties::canEmitSelectWithArms(iff->ifTrue, iff->ifFalse) &&
              !tooCostlyToRunUnconditionally(
                passOptions, iff->ifTrue, iff->ifFalse)) {
            EffectAnalyzer ifTrueEff(passOptions, *getModule(), iff->ifTrue);
            EffectAnalyzer ifFalseEff(passOptions, *getModule(), iff->ifFalse);
            if (!ifTrueEff.hasSideEffects() && !ifFalseEff.hasSideEffects()) {
              EffectAnalyzer condEff(passOptions, *getModule(), iff->condition);
              int32_t dummy = 0;
              bool hasConstArm = (trueFree && isI32Const(iff->ifTrue, dummy)) ||
                                 (falseFree && isI32Const(iff->ifFalse, dummy));
              if (hasConstArm || (!condEff.invalidates(ifTrueEff) &&
                                  !condEff.invalidates(ifFalseEff))) {
                if (trueFree) {
                  invertBoolExprFree(iff->ifTrue, builder);
                } else {
                  iff->ifTrue = builder.makeUnary(EqZInt32, iff->ifTrue);
                }
                if (falseFree) {
                  invertBoolExprFree(iff->ifFalse, builder);
                } else {
                  iff->ifFalse = builder.makeUnary(EqZInt32, iff->ifFalse);
                }
                iff->finalize(Type::i32);
                Expression* rep = iff;
                if (auto* sel = selectify(iff)) {
                  rep = sel;
                }
                replaceCurrent(rep);
                return;
              }
            }
          }
        }
      }

      void visitIf(If* curr) {
        if (shrink2) {
          if (auto* trueIf = curr->ifTrue->dynCast<If>()) {
            if (auto* seq = tryUnnestOneUnreachableArmIf(trueIf)) {
              unnestedIfBlocks.erase(seq);
              curr->ifTrue = seq;
              curr->finalize(curr->type);
            }
          }
          if (curr->ifFalse) {
            if (auto* falseIf = curr->ifFalse->dynCast<If>()) {
              if (auto* seq = tryUnnestOneUnreachableArmIf(falseIf)) {
                unnestedIfBlocks.erase(seq);
                curr->ifFalse = seq;
                curr->finalize(curr->type);
              }
            }
          }
        }
        while (foldBoolIf &&
               tryFoldBooleanIfCondition(
                 curr, getFunction(), *getModule(), passOptions)) {
        }
        if (v4 && !curr->ifFalse &&
            curr->condition->type != Type::unreachable) {
          Builder builder(*getModule());
          auto unwrapSingleElemBlock = [](Expression*& arm) {
            while (auto* b = arm->dynCast<Block>()) {
              if (!b->name.is() && b->list.size() == 1) {
                arm = b->list[0];
              } else {
                break;
              }
            }
          };
          unwrapSingleElemBlock(curr->ifTrue);
          while (auto* child = curr->ifTrue->dynCast<If>()) {
            if (child->ifFalse ||
                child->condition->type == Type::unreachable ||
                neverUnconditionalize ||
                tooCostlyToRunUnconditionally(passOptions, child->condition) ||
                EffectAnalyzer(passOptions, *getModule(), child->condition)
                  .hasSideEffects() ||
                (!isOneBitI32(curr->condition) &&
                 !isOneBitI32(child->condition))) {
              break;
            }
            curr->condition =
              builder.makeBinary(AndInt32,
                                 makeOneBitCond(builder, curr->condition),
                                 makeOneBitCond(builder, child->condition));
            BranchHints::applyAndTo(curr, child, curr, getFunction());
            curr->ifTrue = child->ifTrue;
            unwrapSingleElemBlock(curr->ifTrue);
            curr->finalize();
            while (foldBoolIf &&
                   tryFoldBooleanIfCondition(
                     curr, getFunction(), *getModule(), passOptions)) {
            }
          }
          Drop* innerDrop = curr->ifTrue->dynCast<Drop>();
          Break* br = innerDrop ? innerDrop->value->dynCast<Break>()
                                : curr->ifTrue->dynCast<Break>();
          if (innerDrop &&
              (!br || !br->value || br->type == Type::unreachable)) {
            innerDrop = nullptr;
            br = nullptr;
          }
          if (br &&
              canTurnIfIntoBrIf(
                curr->condition, br->value, passOptions, *getModule())) {
            if (!br->condition) {
              br->condition = curr->condition;
              BranchHints::copyTo(curr, br, getFunction());
              br->finalize();
              Expression* rep = innerDrop
                                  ? static_cast<Expression*>(innerDrop)
                                  : builder.dropIfConcretelyTyped(br);
              if (innerDrop) {
                innerDrop->finalize();
              }
              if (foldBoolIf) {
                if (auto* folded =
                      tryFoldBooleanBreakCondition(br,
                                                   getFunction(),
                                                   *getModule(),
                                                   passOptions,
                                                   neverUnconditionalize,
                                                   rep->dynCast<Drop>())) {
                  rep = folded;
                }
              }
              replaceCurrent(rep);
              return;
            } else if (!neverUnconditionalize &&
                       (!br->value || !br->value->type.isTuple()) &&
                       br->condition->type != Type::unreachable &&
                       !tooCostlyToRunUnconditionally(passOptions,
                                                      br->condition) &&
                       !EffectAnalyzer(passOptions, *getModule(), br->condition)
                          .hasSideEffects() &&
                       (isOneBitI32(curr->condition) ||
                        isOneBitI32(br->condition))) {
              br->condition =
                builder.makeBinary(AndInt32,
                                   makeOneBitCond(builder, curr->condition),
                                   makeOneBitCond(builder, br->condition));
              BranchHints::applyAndTo(curr, br, br, getFunction());
              br->finalize();
              Expression* rep = innerDrop
                                  ? static_cast<Expression*>(innerDrop)
                                  : builder.dropIfConcretelyTyped(br);
              if (innerDrop) {
                innerDrop->finalize();
              }
              replaceCurrent(rep);
              return;
            }
          }
        }
        // we may have simplified ifs enough to turn them into selects
        if (auto* select = selectify(curr)) {
          replaceCurrent(select);
          return;
        }
        if (shrink && curr->ifFalse &&
            curr->condition->type != Type::unreachable &&
            curr->ifTrue->type == Type::unreachable &&
            curr->ifFalse->type == Type::unreachable) {
          Builder builder(*getModule());
          auto makeUnnestedSeq = [&](Expression* first,
                                     Expression* second) -> Block* {
            auto* seq = builder.makeBlock(first);
            if (auto* secondBlock = second->dynCast<Block>();
                secondBlock && !secondBlock->name.is()) {
              for (auto* item : secondBlock->list) {
                seq->list.push_back(item);
              }
            } else {
              seq->list.push_back(second);
            }
            seq->finalize();
            unnestedIfBlocks.insert(seq);
            return seq;
          };
          auto* ifTrueBreak = curr->ifTrue->dynCast<Break>();
          if (ifTrueBreak && !ifTrueBreak->condition &&
              canTurnIfIntoBrIf(curr->condition,
                                ifTrueBreak->value,
                                passOptions,
                                *getModule())) {
            ifTrueBreak->condition = curr->condition;
            BranchHints::copyTo(curr, ifTrueBreak, getFunction());
            ifTrueBreak->finalize();
            Expression* first = builder.dropIfConcretelyTyped(ifTrueBreak);
            if (foldBoolIf) {
              if (auto* rep =
                    tryFoldBooleanBreakCondition(ifTrueBreak,
                                                 getFunction(),
                                                 *getModule(),
                                                 passOptions,
                                                 neverUnconditionalize,
                                                 first->dynCast<Drop>())) {
                first = rep;
                if (auto* iff = first->dynCast<If>()) {
                  while (tryFoldBooleanIfCondition(
                    iff, getFunction(), *getModule(), passOptions)) {
                  }
                }
              }
            }
            replaceCurrent(makeUnnestedSeq(first, curr->ifFalse));
            return;
          }
          auto* ifFalseBreak = curr->ifFalse->dynCast<Break>();
          if (ifFalseBreak && !ifFalseBreak->condition &&
              canTurnIfIntoBrIf(curr->condition,
                                ifFalseBreak->value,
                                passOptions,
                                *getModule())) {
            ifFalseBreak->condition =
              makeNegatedBool(builder, curr->condition, v4);
            BranchHints::copyFlippedTo(curr, ifFalseBreak, getFunction());
            ifFalseBreak->finalize();
            Expression* first = builder.dropIfConcretelyTyped(ifFalseBreak);
            if (foldBoolIf) {
              if (auto* rep =
                    tryFoldBooleanBreakCondition(ifFalseBreak,
                                                 getFunction(),
                                                 *getModule(),
                                                 passOptions,
                                                 neverUnconditionalize,
                                                 first->dynCast<Drop>())) {
                first = rep;
                if (auto* iff = first->dynCast<If>()) {
                  while (tryFoldBooleanIfCondition(
                    iff, getFunction(), *getModule(), passOptions)) {
                  }
                }
              }
            }
            replaceCurrent(makeUnnestedSeq(first, curr->ifTrue));
            return;
          }
          if (auto* u = curr->condition->dynCast<Unary>();
              u && u->op == EqZInt32) {
            curr->condition = u->value;
            std::swap(curr->ifTrue, curr->ifFalse);
            BranchHints::flip(curr, getFunction());
          }
          Expression* after = curr->ifFalse;
          curr->ifFalse = nullptr;
          curr->finalize();
          while (foldBoolIf &&
                 tryFoldBooleanIfCondition(
                   curr, getFunction(), *getModule(), passOptions)) {
          }
          replaceCurrent(makeUnnestedSeq(curr, after));
          return;
        }
        if (v4 && curr->ifFalse) {
          if (auto* seq = tryUnnestOneUnreachableArmIf(curr)) {
            replaceCurrent(seq);
            return;
          }
        }
      }

      // Convert an if into a select (or under shrink2, a boolean expression or
      // left-to-right i32.and/i32.or), if possible and beneficial to do so.
      Expression* selectify(If* iff) {
        if (neverUnconditionalize) {
          return nullptr;
        }
        // Only an if-else can be turned into a select.
        if (!iff->ifFalse) {
          return nullptr;
        }
        if (!Properties::canEmitSelectWithArms(iff->ifTrue, iff->ifFalse)) {
          return nullptr;
        }
        if (iff->condition->type == Type::unreachable) {
          // An if with an unreachable condition may nonetheless have a type
          // that is not unreachable,
          //
          // (if (result i32) (unreachable) ..)
          //
          // Turning such an if into a select would change the type of the
          // expression, which would require updating types further up. Avoid
          // that, leaving dead code elimination to that dedicated pass.
          return nullptr;
        }
        // This is always helpful for code size, but can be a tradeoff with
        // performance as we run both code paths. So when shrinking we always
        // try to do this, but otherwise must consider more carefully.
        if (tooCostlyToRunUnconditionally(
              passOptions, iff->ifTrue, iff->ifFalse)) {
          return nullptr;
        }
        // Check if side effects allow this: we need to execute the two arms
        // unconditionally, and also to make the condition run last.
        EffectAnalyzer ifTrue(passOptions, *getModule(), iff->ifTrue);
        if (ifTrue.hasSideEffects()) {
          return nullptr;
        }
        EffectAnalyzer ifFalse(passOptions, *getModule(), iff->ifFalse);
        if (ifFalse.hasSideEffects()) {
          return nullptr;
        }
        Builder builder(*getModule());
        int32_t trueConst = 0, falseConst = 0;
        bool trueIsConst = isI32Const(iff->ifTrue, trueConst);
        bool falseIsConst = isI32Const(iff->ifFalse, falseConst);
        if (v4 && iff->type == Type::i32) {
          if (trueIsConst && falseIsConst) {
            if (trueConst == 1 && falseConst == 0) {
              if (isOneBitI32(iff->condition)) {
                return iff->condition;
              }
              return builder.makeBinary(
                NeInt32, iff->condition, builder.makeConst(int32_t(0)));
            }
            if (trueConst == 0 && falseConst == 1) {
              if (auto* u = iff->condition->dynCast<Unary>();
                  u && u->op == EqZInt32 && isOneBitI32(u->value)) {
                return u->value;
              }
              return makeNegatedBool(builder, iff->condition, v4);
            }
          }
        }
        EffectAnalyzer condition(passOptions, *getModule(), iff->condition);
        bool condInvalidates =
          condition.invalidates(ifTrue) || condition.invalidates(ifFalse);
        if (v4 && iff->type == Type::i32) {
          if (falseIsConst && falseConst == 0 && isOneBitI32(iff->ifTrue) &&
              (condInvalidates || isOneBitI32(iff->condition))) {
            return builder.makeBinary(
              AndInt32, makeOneBitCond(builder, iff->condition), iff->ifTrue);
          }
          if (trueIsConst && trueConst == 0 && isOneBitI32(iff->ifFalse)) {
            return builder.makeBinary(
              AndInt32,
              makeNegOneBitCond(builder, iff->condition, true),
              iff->ifFalse);
          }
          if (trueIsConst && trueConst == 1 && isOneBitI32(iff->ifFalse) &&
              (condInvalidates || isOneBitI32(iff->condition))) {
            return builder.makeBinary(
              OrInt32, makeOneBitCond(builder, iff->condition), iff->ifFalse);
          }
          if (falseIsConst && falseConst == 1 && isOneBitI32(iff->ifTrue)) {
            return builder.makeBinary(
              OrInt32,
              makeNegOneBitCond(builder, iff->condition, true),
              iff->ifTrue);
          }
        }
        if (condInvalidates) {
          return nullptr;
        }
        auto* select =
          builder.makeSelect(iff->condition, iff->ifTrue, iff->ifFalse);
        if (select->type != iff->type) {
          // If the select is more refined than the if it replaces, we must
          // propagate that outwards.
          refinalize = true;
        }
        return select;
      }

      void visitLocalSet(LocalSet* curr) {
        // Sets of an if can be optimized in various ways that remove part of
        // the if branching, or all of it.
        // The optimizations we can do here can recurse and call each
        // other, so pass around a pointer to the output.
        optimizeSetIf(getCurrentPointer());
      }

      // Flip an if's condition with an eqz, and flip its arms.
      void flip(If* iff) {
        std::swap(iff->ifTrue, iff->ifFalse);
        iff->condition =
          Builder(*getModule()).makeUnary(EqZInt32, iff->condition);
        BranchHints::flip(iff, getFunction());
      }

      void optimizeSetIf(Expression** currp) {
        if (optimizeSetIfWithBrArm(currp)) {
          return;
        }
        if (optimizeSetIfWithCopyArm(currp)) {
          return;
        }
      }

      // If one arm is a br, we prefer a br_if and the set later:
      //  (local.set $x
      //    (if (result i32)
      //      (..condition..)
      //      (br $somewhere)
      //      (..result)
      //    )
      //  )
      // =>
      //  (br_if $somewhere
      //    (..condition..)
      //  )
      //  (local.set $x
      //    (..result)
      //  )
      // TODO: handle a condition in the br? need to watch for side effects
      bool optimizeSetIfWithBrArm(Expression** currp) {
        auto* set = (*currp)->cast<LocalSet>();
        auto* iff = set->value->dynCast<If>();
        if (!iff || !iff->type.isConcrete() ||
            !iff->condition->type.isConcrete()) {
          return false;
        }
        auto tryToOptimize =
          [&](Expression* one, Expression* two, bool flipCondition) {
            if (one->type == Type::unreachable &&
                two->type != Type::unreachable) {
              if (auto* br = one->dynCast<Break>()) {
                if (ExpressionAnalyzer::isSimple(br)) {
                  // Wonderful, do it!
                  Builder builder(*getModule());
                  if (flipCondition) {
                    flip(iff);
                  }
                  br->condition = iff->condition;
                  BranchHints::copyTo(iff, br, getFunction());
                  br->finalize();
                  set->value = two;
                  auto* block = builder.makeSequence(br, set);
                  *currp = block;
                  // Recurse on the set, which now has a new value.
                  optimizeSetIf(&block->list[1]);
                  return true;
                }
              }
            }
            return false;
          };
        return tryToOptimize(iff->ifTrue, iff->ifFalse, false) ||
               tryToOptimize(iff->ifFalse, iff->ifTrue, true);
      }

      // If one arm is a get of the same outer set, it is a copy which
      // we can remove. If this is not a tee, then we remove the get
      // as well as the if-else opcode in the binary format, which is
      // great:
      //  (local.set $x
      //    (if (result i32)
      //      (..condition..)
      //      (..result)
      //      (local.get $x)
      //    )
      //  )
      // =>
      //  (if
      //    (..condition..)
      //    (local.set $x
      //      (..result)
      //    )
      //  )
      // If this is a tee, then we can do the same operation but
      // inside a block, and keep the get:
      //  (local.tee $x
      //    (if (result i32)
      //      (..condition..)
      //      (..result)
      //      (local.get $x)
      //    )
      //  )
      // =>
      //  (block (result i32)
      //    (if
      //      (..condition..)
      //      (local.set $x
      //        (..result)
      //      )
      //    )
      //    (local.get $x)
      //  )
      // We save the if-else opcode, and add the block's opcodes.
      // This may be detrimental, however, often the block can be
      // merged or eliminated given the outside scope, and we
      // removed one of the if branches.
      bool optimizeSetIfWithCopyArm(Expression** currp) {
        auto* set = (*currp)->cast<LocalSet>();
        auto* iff = set->value->dynCast<If>();
        if (!iff || !iff->type.isConcrete() ||
            !iff->condition->type.isConcrete()) {
          return false;
        }
        Builder builder(*getModule());
        LocalGet* get = iff->ifTrue->dynCast<LocalGet>();
        if (get && get->index == set->index) {
          flip(iff);
        } else {
          get = iff->ifFalse->dynCast<LocalGet>();
          if (get && get->index != set->index) {
            get = nullptr;
          }
        }
        if (!get) {
          return false;
        }
        // We can do it!
        bool tee = set->isTee();
        assert(set->index == get->index);
        assert(iff->ifFalse == get);
        set->value = iff->ifTrue;
        set->finalize();
        iff->ifTrue = set;
        iff->ifFalse = nullptr;
        iff->finalize();
        Expression* replacement = iff;
        if (tee) {
          set->makeSet();
          // We need a block too.
          replacement = builder.makeSequence(iff,
                                             get // reuse the get
          );
        }
        *currp = replacement;
        // Recurse on the set, which now has a new value.
        optimizeSetIf(&iff->ifTrue);
        return true;
      }

      // (br_if)+ => br_table
      // we look for the specific pattern of
      //  (br_if ..target1..
      //    (i32.eq
      //      (..input..)
      //      (i32.const ..value1..)
      //    )
      //  )
      //  (br_if ..target2..
      //    (i32.eq
      //      (..input..)
      //      (i32.const ..value2..)
      //    )
      //  )
      // TODO: consider also looking at <= etc. and not just eq
      void tablify(Block* block) {
        auto& list = block->list;
        if (list.size() <= 1) {
          return;
        }

        // Heuristics. These are slightly inspired by the constants from the
        // asm.js backend.

        // How many br_ifs we need to see to consider doing this
        const uint32_t MIN_NUM = 3;
        // How much of a range of values is definitely too big
        const uint32_t MAX_RANGE = 1024;
        // Multiplied by the number of br_ifs, then compared to the range. When
        // this is high, we allow larger ranges.
        const uint32_t NUM_TO_RANGE_FACTOR = 3;

        // check if the input is a proper br_if on an i32.eq of a condition
        // value to a const, and the const is in the proper range,
        // [0-int32_max), to avoid overflow concerns. returns the br_if if so,
        // or nullptr otherwise
        auto getProperBrIf = [](Expression* curr) -> Break* {
          auto* br = curr->dynCast<Break>();
          if (!br) {
            return nullptr;
          }
          if (!br->condition || br->value) {
            return nullptr;
          }
          if (br->type != Type::none) {
            // no value, so can be unreachable or none. ignore unreachable ones,
            // dce will clean it up
            return nullptr;
          }
          auto* condition = br->condition;
          // Also support eqz, which is the same as == 0.
          if (auto* unary = condition->dynCast<Unary>()) {
            if (unary->op == EqZInt32) {
              return br;
            }
            return nullptr;
          }
          auto* binary = condition->dynCast<Binary>();
          if (!binary) {
            return nullptr;
          }
          if (binary->op != EqInt32) {
            return nullptr;
          }
          auto* c = binary->right->dynCast<Const>();
          if (!c) {
            return nullptr;
          }
          uint32_t value = c->value.geti32();
          if (value >= uint32_t(std::numeric_limits<int32_t>::max())) {
            return nullptr;
          }
          return br;
        };

        // check if the input is a proper br_if
        // and returns the condition if so, or nullptr otherwise
        auto getProperBrIfConditionValue =
          [&getProperBrIf](Expression* curr) -> Expression* {
          auto* br = getProperBrIf(curr);
          if (!br) {
            return nullptr;
          }
          auto* condition = br->condition;
          if (auto* binary = condition->dynCast<Binary>()) {
            return binary->left;
          } else if (auto* unary = condition->dynCast<Unary>()) {
            assert(unary->op == EqZInt32);
            return unary->value;
          } else {
            WASM_UNREACHABLE("invalid br_if condition");
          }
        };

        // returns the constant value, as a uint32_t
        auto getProperBrIfConstant =
          [&getProperBrIf](Expression* curr) -> uint32_t {
          auto* condition = getProperBrIf(curr)->condition;
          if (auto* binary = condition->dynCast<Binary>()) {
            return binary->right->cast<Const>()->value.geti32();
          } else if ([[maybe_unused]] auto* unary =
                       condition->dynCast<Unary>()) {
            assert(unary->op == EqZInt32);
            return 0;
          } else {
            WASM_UNREACHABLE("invalid br_if condition");
          }
        };
        Index start = 0;
        while (start < list.size() - 1) {
          auto* conditionValue = getProperBrIfConditionValue(list[start]);
          if (!conditionValue) {
            start++;
            continue;
          }
          // If the first condition value is a tee, that is ok, so long as the
          // others afterwards are gets of the value that is tee'd.
          LocalGet get;
          if (auto* tee = conditionValue->dynCast<LocalSet>()) {
            get.index = tee->index;
            get.type = getFunction()->getLocalType(get.index);
            conditionValue = &get;
          }
          // if the condition has side effects, we can't replace many
          // appearances of it with a single one
          if (EffectAnalyzer(passOptions, *getModule(), conditionValue)
                .hasSideEffects()) {
            start++;
            continue;
          }
          // look for a "run" of br_ifs with all the same conditionValue, and
          // having unique constants (an overlapping constant could be handled,
          // just the first branch is taken, but we can't remove the other br_if
          // (it may be the only branch keeping a block reachable), which may
          // make this bad for code size.
          Index end = start + 1;
          std::unordered_set<uint32_t> usedConstants;
          usedConstants.insert(getProperBrIfConstant(list[start]));
          while (end < list.size() &&
                 ExpressionAnalyzer::equal(
                   getProperBrIfConditionValue(list[end]), conditionValue)) {
            if (!usedConstants.insert(getProperBrIfConstant(list[end]))
                   .second) {
              // this constant already appeared
              break;
            }
            end++;
          }
          auto num = end - start;
          if (num >= 2 && num >= MIN_NUM) {
            // we found a suitable range, [start, end), containing more than 1
            // element. let's see if it's worth it
            auto min = getProperBrIfConstant(list[start]);
            auto max = min;
            for (Index i = start + 1; i < end; i++) {
              auto* curr = list[i];
              min = std::min(min, getProperBrIfConstant(curr));
              max = std::max(max, getProperBrIfConstant(curr));
            }
            uint32_t range = max - min;
            // decision time
            if (range <= MAX_RANGE && range <= num * NUM_TO_RANGE_FACTOR) {
              // great! let's do this
              std::unordered_set<Name> usedNames;
              for (Index i = start; i < end; i++) {
                usedNames.insert(getProperBrIf(list[i])->name);
              }
              // we need a name for the default too
              Name defaultName;
              Index i = 0;
              while (1) {
                defaultName = "tablify|" + std::to_string(i++);
                if (!usedNames.contains(defaultName)) {
                  break;
                }
              }
              std::vector<Name> table;
              for (Index i = start; i < end; i++) {
                auto name = getProperBrIf(list[i])->name;
                auto index = getProperBrIfConstant(list[i]);
                index -= min;
                while (table.size() <= index) {
                  table.push_back(defaultName);
                }
                // we should have made sure there are no overlaps
                assert(table[index] == defaultName);
                table[index] = name;
              }
              Builder builder(*getModule());
              // the table and condition are offset by the min
              auto* newCondition = getProperBrIfConditionValue(list[start]);

              if (min != 0) {
                newCondition = builder.makeBinary(
                  SubInt32, newCondition, builder.makeConst(int32_t(min)));
              }
              list[end - 1] = builder.makeBlock(
                defaultName,
                builder.makeSwitch(table, defaultName, newCondition));
              for (Index i = start; i < end - 1; i++) {
                ExpressionManipulator::nop(list[i]);
              }
              // the defaultName may exist elsewhere in this function,
              // uniquify it later
              needUniqify = true;
            }
          }
          start = end;
        }
      }

      void visitBreak(Break* curr) {
        if (!curr->condition) {
          return;
        }
        if (foldBoolIf) {
          if (auto* rep = tryFoldBooleanBreakCondition(curr,
                                                       getFunction(),
                                                       *getModule(),
                                                       passOptions,
                                                       neverUnconditionalize)) {
            if (rep != curr) {
              replaceCurrent(rep);
              if (auto* iff = rep->dynCast<If>()) {
                visitIf(iff);
              } else if (auto* blk = rep->dynCast<Block>()) {
                unnestedIfBlocks.insert(blk);
                for (auto*& item : blk->list) {
                  if (auto* subIf = item->dynCast<If>()) {
                    while (tryFoldBooleanIfCondition(subIf,
                                                     getFunction(),
                                                     *getModule(),
                                                     passOptions,
                                                     neverUnconditionalize)) {
                    }
                  }
                }
              }
              return;
            }
          }
        }
        auto* value = Properties::getFallthrough(
          curr->condition, passOptions, *getModule());
        // Optimize if condition's fallthrough is a constant.
        if (auto* c = value->dynCast<Const>()) {
          ChildLocalizer localizer(
            curr, getFunction(), *getModule(), passOptions);
          auto* block = localizer.getChildrenReplacement();
          if (c->value.geti32()) {
            // the branch is always taken, make it unconditional
            curr->condition = nullptr;
            curr->type = Type::unreachable;
            block->list.push_back(curr);
            block->finalize();
            BranchHints::clear(curr, getFunction());
          } else {
            // the branch is never taken, allow control flow to fall through
            if (curr->value) {
              block->list.push_back(curr->value);
              block->finalize();
            }
          }
          replaceCurrent(block);
          // The type changes if we make the break unconditional (it becomes
          // unreachable), but a change may also happen if we see the break is
          // not taken, as then the target block may have no more breaks, and
          // become unreachable. Either way, refinalize.
          refinalize = true;
        }
      }

      void visitDrop(Drop* curr) {
        if (!foldBoolIf) {
          return;
        }
        if (auto* br = curr->value->dynCast<Break>()) {
          if (auto* rep = tryFoldBooleanBreakCondition(br,
                                                       getFunction(),
                                                       *getModule(),
                                                       passOptions,
                                                       neverUnconditionalize,
                                                       curr)) {
            if (rep != curr) {
              replaceCurrent(rep);
              if (auto* iff = rep->dynCast<If>()) {
                visitIf(iff);
              }
            }
          }
        }
      }

      void visitFunction(Function* curr) {
        if (shrink2 && curr->body) {
          if (auto* bodyIf = curr->body->dynCast<If>()) {
            if (auto* seq = tryUnnestOneUnreachableArmIf(bodyIf)) {
              unnestedIfBlocks.erase(seq);
              curr->body = seq;
            }
          }
        }
      }
    };
    bool v4 = isV4Enabled(getPassOptions());
    if (v4) {
      lowerNonNullAndVoidReturnGuards(func, /*allowFuncExit=*/false);
    }
    FinalOptimizer finalOptimizer(getPassOptions());
    finalOptimizer.setModule(getModule());
    finalOptimizer.shrink = getPassRunner()->options.shrinkLevel > 0;
    finalOptimizer.shrink2 = shrink2;
    finalOptimizer.v4 = v4;
    finalOptimizer.foldBoolIf = foldBoolIf;
    finalOptimizer.neverUnconditionalize = neverUnconditionalize;
    if (finalOptimizer.v4 && func->body) {
      std::function<void(Expression*, bool)> collectTailNames =
        [&](Expression* curr, bool insideArm) {
          if (!curr) {
            return;
          }
          if (auto* block = curr->dynCast<Block>()) {
            if (block->list.empty()) {
              return;
            }
            for (Index i = 0; i + 1 < block->list.size(); i++) {
              if (block->list[i]->type == Type::unreachable) {
                return;
              }
            }
            if (block->type == func->getResults() &&
                (!insideArm || block->list.back()->type != Type::unreachable)) {
              finalOptimizer.tailBlocks.insert(block);
              if (block->name.is()) {
                finalOptimizer.tailBlockNames.insert(block->name);
              }
            }
            collectTailNames(block->list.back(), insideArm);
          } else if (auto* iff = curr->dynCast<If>()) {
            if (iff->condition->type == Type::unreachable ||
                iff->type != func->getResults()) {
              return;
            }
            collectTailNames(iff->ifTrue, true);
            if (iff->ifFalse) {
              collectTailNames(iff->ifFalse, true);
            }
          } else if (auto* loop = curr->dynCast<Loop>()) {
            if (loop->type == func->getResults()) {
              collectTailNames(loop->body, true);
            }
          }
        };
      collectTailNames(func->body, false);
    }

    finalOptimizer.walkFunction(func);
    if (finalOptimizer.needUniqify) {
      wasm::UniqueNameMapper::uniquify(func->body);
    }
    if (finalOptimizer.refinalize) {
      ReFinalize().walkFunctionInModule(func, getModule());
    }
    if (shrink2) {
      optimizeTailBlockReturns(func);
      if (getPassOptions().trapsNeverHappen) {
        optimizeNoreturnReturnTails(func);
      }
    }
    if (foldBoolIf && getPassOptions().trapsNeverHappen) {
      if (stripVoidTrailingUnreachableExpr(
            func->body, func->getResults() == Type::none)) {
        ReFinalize().walkFunctionInModule(func, getModule());
      }
    }
    if (v4) {
      lowerNonNullAndVoidReturnGuards(func, /*allowFuncExit=*/true);
    }
  }
};

Pass* createRemoveUnusedBrsPass() { return new RemoveUnusedBrs(); }

} // namespace wasm
