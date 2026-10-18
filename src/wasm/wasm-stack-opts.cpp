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
// Operations on Stack IR.
//

#include "ir/branch-utils.h"
#include "ir/iteration.h"
#include "ir/local-graph.h"
#include "ir/properties.h"
#include "pass.h"
#include "wasm-stack.h"
#include "wasm.h"

#ifndef STACK_OPT_DEBUG
#define STACK_OPT_DEBUG 0
#endif

namespace wasm {

StackIROptimizer::StackIROptimizer(Module& module,
                                   Function* func,
                                   StackIR& insts,
                                   const PassOptions& passOptions,
                                   FeatureSet features)
  : module(module), func(func), insts(insts), passOptions(passOptions),
    features(features) {}

void StackIROptimizer::run() {
  dce();
  vacuum();
  removeUnneededBlocks();
  dce();
  // FIXME: local2Stack is currently rather slow (due to localGraph),
  //        so for now run it only when really optimizing
  if (passOptions.optimizeLevel >= 3 || passOptions.shrinkLevel >= 1) {
    local2Stack();
    vacuum();
    removeUnneededBlocks();
    dce();
  }
  vacuum();
}

void StackIROptimizer::dce() {
  // Remove code after an unreachable instruction: anything after it, up to the
  // next control flow barrier, can simply be removed.
  bool inUnreachableCode = false;
  for (Index i = 0; i < insts.size(); i++) {
    auto* inst = insts[i];
    if (!inst) {
      continue;
    }
    if (inUnreachableCode) {
      // Does the unreachable code end here?
      if (isControlFlowBarrier(inst)) {
        inUnreachableCode = false;
      } else {
        // We can remove this.
        removeAt(i);
      }
    } else if (inst->type == Type::unreachable) {
      inUnreachableCode = true;
    }
  }

  // Remove code before an Unreachable. Consider this:
  //
  //  (drop
  //   ..
  //  )
  //  (unreachable)
  //
  // The drop is not needed, as the unreachable puts the stack in the
  // polymorphic state anyhow. Note that we don't need to optimize anything
  // other than a drop here, as in general the Binaryen IR DCE pass will handle
  // everything else. A drop followed by an unreachable is the only thing that
  // pass cannot handle, as the structured form of Binaryen IR does not allow
  // removing such a drop, and so we can only do it here in StackIR.
  //
  // TODO: We can look even further back, say if there is another drop of
  //       something before, then we can remove that drop as well. To do that
  //       we'd need to inspect the stack going backwards.
  for (Index i = 1; i < insts.size(); i++) {
    auto* inst = insts[i];
    if (!inst || inst->op != StackInst::Basic ||
        !inst->origin->is<Unreachable>()) {
      continue;
    }

    // Look back past nulls.
    Index j = i - 1;
    while (j > 0 && !insts[j]) {
      j--;
    }

    auto*& prev = insts[j];
    if (prev && prev->op == StackInst::Basic && prev->origin->is<Drop>()) {
      prev = nullptr;
    }
  }
}

// Remove obviously-unneeded code.
void StackIROptimizer::vacuum() {
  // In the wasm binary format a nop is never needed. (In Binaryen IR, in
  // comparison, it is necessary e.g. in a function body or an if arm.)
  //
  // It is especially important to remove nops because we add nops when we
  // read wasm into Binaryen IR. That is, this avoids a potential increase in
  // code size.
  for (Index i = 0; i < insts.size(); i++) {
    auto*& inst = insts[i];
    if (inst && inst->origin->is<Nop>()) {
      inst = nullptr;
    }
  }
}

// Whether the local.get at getIndex is consumed by an operation whose operands
// can be swapped.
bool StackIROptimizer::isConsumedBySymmetricOp(Index getIndex) {
  Index k = getIndex + 1;
  while (k < insts.size() && !insts[k]) {
    ++k;
  }
  if (k >= insts.size()) {
    return false;
  }
  auto* inst = insts[k];
  if (inst->op != StackInst::Basic) {
    return false;
  }
  if (auto* binary = inst->origin->dynCast<Binary>()) {
    return Properties::isSymmetric(binary);
  }
  return inst->origin->is<RefEq>();
}

bool StackIROptimizer::isPureSlice(Index start,
                                   Index end,
                                   SmallLocalSet& readLocals,
                                   bool allowTrapsNeverHappen) {
  if (start > end) {
    return false;
  }
  int needed = 1;
  bool seenAny = false;
  for (Index i = end;; i--) {
    if (auto* inst = insts[i]) {
      if (needed <= 0) {
        return false;
      }
      if (inst->op != StackInst::Basic || !inst->type.isConcrete() ||
          !inst->type.isSingle()) {
        return false;
      }
      auto* orig = inst->origin;
      if (orig->is<LocalSet>()) {
        return false;
      }
      if (auto* get = orig->dynCast<LocalGet>()) {
        readLocals.insert(get->index);
      } else if (orig->is<Const>() || orig->is<RefNull>() ||
                 orig->is<RefFunc>() || orig->is<Select>() ||
                 orig->is<RefIsNull>() || orig->is<RefEq>()) {
        // Pure.
      } else if (auto* gg = orig->dynCast<GlobalGet>()) {
        auto* g = module.getGlobalOrNull(gg->name);
        if (!g || g->mutable_ != Immutable) {
          return false;
        }
      } else if (auto* bin = orig->dynCast<Binary>()) {
        switch (bin->op) {
          case DivSInt32:
          case DivUInt32:
          case RemSInt32:
          case RemUInt32:
          case DivSInt64:
          case DivUInt64:
          case RemSInt64:
          case RemUInt64:
            return false;
          default:
            break;
        }
      } else if (auto* un = orig->dynCast<Unary>()) {
        switch (un->op) {
          case TruncSFloat32ToInt32:
          case TruncUFloat32ToInt32:
          case TruncSFloat64ToInt32:
          case TruncUFloat64ToInt32:
          case TruncSFloat32ToInt64:
          case TruncUFloat32ToInt64:
          case TruncSFloat64ToInt64:
          case TruncUFloat64ToInt64:
            return false;
          default:
            break;
        }
      } else if (orig->is<RefTest>()) {
        // ref.test is always pure and non-trapping.
      } else if (auto* rc = orig->dynCast<RefCast>()) {
        if (!allowTrapsNeverHappen || rc->desc || !rc->ref->type.isRef() ||
            (rc->type.isNonNullable() && rc->ref->type.isNull())) {
          return false;
        }
      } else if (auto* ra = orig->dynCast<RefAs>()) {
        if (!allowTrapsNeverHappen || ra->op != RefAsNonNull ||
            !ra->value->type.isRef() || ra->value->type.isNull()) {
          return false;
        }
      } else if (auto* sg = orig->dynCast<StructGet>()) {
        if (!sg->ref->type.isStruct() || sg->ref->type.isNull() ||
            sg->order != MemoryOrder::Unordered ||
            (!allowTrapsNeverHappen && sg->ref->type.isNullable()) ||
            sg->ref->type.getHeapType().getStruct().fields[sg->index].mutable_ !=
              Immutable) {
          return false;
        }
      } else {
        return false;
      }
      needed = needed - 1 + int(getNumConsumedValues(inst));
      seenAny = true;
    }
    if (i == start) {
      break;
    }
  }
  return seenAny && needed == 0;
}

// If ordered properly, we can avoid a local.set/local.get pair,
// and use the value directly from the stack, for example
//    [..produce a value on the stack..]
//    local.set $x
//    [..much code..]
//    local.get $x
//    call $foo ;; use the value, foo(value)
// As long as the code in between does not modify $x, and has
// no control flow branching out, we can remove both the set
// and the get.
void StackIROptimizer::local2Stack() {
  // We use the localGraph to tell us if a get-set pair is indeed a set that is
  // read by that get, and only that get. Note that we run this on Binaryen IR,
  // so we are assuming that no previous opt has changed the interaction of
  // local operations.
  //
  // We use a lazy graph here as we only query in the rare case when we find a
  // set/get pair that looks optimizable.
  LazyLocalGraph localGraph(func);
  // The binary writing of StringWTF16Get and StringSliceWTF is optimized to use
  // fewer scratch locals when their operands are already LocalGets. To avoid
  // interfering with that optimization, we have to avoid removing such
  // LocalGets.
  auto deferredGets = findStringViewDeferredGets();

  bool hasTry = false;
  for (auto* inst : insts) {
    if (inst && (inst->op == StackInst::TryBegin ||
                 inst->op == StackInst::TryTableBegin)) {
      hasTry = true;
      break;
    }
  }

  // We maintain a stack of relevant values. This contains:
  //  * a null setIndex for each actual value that the value stack would have
  //  * an index of each LocalSet that *could* be on the value
  //    stack at that location, along with its value slice bounds.
  const Index null = -1;
  struct StackEntry {
    Index setIndex;
    Index valStart;
    Index valEnd;
  };
  struct SavedStack {
    std::vector<StackEntry> values;
    Index controlFlowStart;
  };
  std::vector<StackEntry> values;
  // We also maintain a stack of values vectors for control flow,
  // saving the stack as we enter and restoring it when we exit.
  std::vector<SavedStack> savedValues;
#if STACK_OPT_DEBUG
  std::cout << "func: " << func->name << '\n' << insts << '\n';
#endif
  for (Index instIndex = 0; instIndex < insts.size(); instIndex++) {
    auto* inst = insts[instIndex];
    if (!inst) {
      continue;
    }
    // First, consume values from the stack as required.
    auto consumed = getNumConsumedValues(inst);
#if STACK_OPT_DEBUG
    std::cout << "  " << instIndex << " : " << *inst << ", " << values.size()
              << " on stack, will consume " << consumed << "\n    ";
    for (auto s : values)
      std::cout << s.setIndex << ' ';
    std::cout << '\n';
#endif
    // TODO: currently we run dce before this, but if we didn't, we'd need
    //       to handle unreachable code here - it's ok to pop multiple values
    //       there even if the stack is at size 0.
    Index firstOperandStart = instIndex;
    while (consumed > 0) {
      assert(values.size() > 0);
      // Whenever we hit a possible stack value, kill it - it would
      // be consumed here, so we can never optimize to it.
      while (values.back().setIndex != null) {
        values.pop_back();
        assert(values.size() > 0);
      }
      // Finally, consume the actual value that is consumed here.
      firstOperandStart = values.back().valStart;
      values.pop_back();
      consumed--;
    }
    // After consuming, we can see what to do with this. First, handle
    // control flow.
    if (isControlFlowBegin(inst)) {
      // Save the stack for when we end this control flow.
      savedValues.push_back({std::move(values), firstOperandStart});
      values.clear();
    } else if (isControlFlowEnd(inst)) {
      assert(!savedValues.empty());
      firstOperandStart = savedValues.back().controlFlowStart;
      values = std::move(savedValues.back().values);
      savedValues.pop_back();
    } else if (isControlFlow(inst)) {
      // Otherwise, in the middle of control flow, just clear it
      values.clear();
    }
    // This is something we should handle, look into it.
    if (inst->type.isConcrete()) {
      bool optimized = false;
      // Do not optimize multivalue locals, since those will be better
      // optimized when they are visited in the binary writer and this
      // optimization would interfere with that one.
      if (auto* get = inst->origin->dynCast<LocalGet>();
          get && inst->type.isSingle() && !deferredGets.contains(get)) {
        // Use another local to clarify what instIndex means in this scope.
        auto getIndex = instIndex;

        // This is a potential optimization opportunity! See if we
        // can reach the set.
        if (values.size() > 0) {
          Index j = values.size() - 1;
          Index numInterveningVals = 0;
          while (1) {
            auto setIndex = values[j].setIndex;
            if (setIndex == null) {
              numInterveningVals++;
              if (numInterveningVals > 6 || j == 0) {
                break;
              }
              j--;
              continue;
            }
            auto* set = insts[setIndex]->origin->cast<LocalSet>();
            if (set->index == get->index) {
              // This might be a proper set-get pair, where the set is
              // used by this get and nothing else, check that.
              auto& sets = localGraph.getSets(get);
              if (sets.size() == 1 && *sets.begin() == set) {
                auto& setInfluences = localGraph.getSetInfluences(set);
                if (numInterveningVals == 0 ||
                    (numInterveningVals == 1 &&
                     isConsumedBySymmetricOp(getIndex))) {
                  // If this has the proper value of 1, also do the potentially-
                  // expensive check of whether we can remove this pair at all.
                  if (setInfluences.size() == 1 &&
                      canRemoveSetGetPair(setIndex, getIndex)) {
                    assert(*setInfluences.begin() == get);
                    // Do it! The set and the get can go away, the proper
                    // value is on the stack.
#if STACK_OPT_DEBUG
                    std::cout << "  stackify the get\n";
#endif
                    insts[setIndex] = nullptr;
                    insts[getIndex] = nullptr;
                  } else {
                    // The local value is still needed for other gets (or for
                    // validation dominance), so we cannot remove the set, but
                    // we can turn the set into a tee and remove this get.
                    insts[setIndex]->type = func->getLocalType(set->index);
                    insts[getIndex] = nullptr;
                  }
                  // Continuing on from here, replace this on the stack
                  // with a null, representing a regular value. We
                  // keep possible values above us active - they may
                  // be optimized later, as they would be pushed after
                  // us, and used before us, so there is no conflict.
                  values[j] = {null, values[j].valStart, setIndex};
                  optimized = true;
                  break;
                }

                // We have 1..6 intervening values on the stack.
                // Direction 1: Forward pure-set sinking.
                Index setValStart = values[j].valStart;
                bool canSinkForward = setValStart < setIndex;
                for (size_t m = j + 1; canSinkForward && m < values.size();
                     ++m) {
                  if (values[m].valStart <= setIndex) {
                    canSinkForward = false;
                  }
                }
                bool hasThrowingCall = false;
                for (Index p = setIndex + 1; canSinkForward && p < getIndex;
                     ++p) {
                  auto* pInst = insts[p];
                  if (!pInst) {
                    continue;
                  }
                  if (pInst->op != StackInst::Basic) {
                    canSinkForward = false;
                    break;
                  }
                  auto* orig = pInst->origin;
                  if (Properties::isBranch(orig) || orig->is<Return>() ||
                      orig->is<Throw>() || orig->is<Rethrow>() ||
                      orig->is<Unreachable>() || orig->is<Pop>()) {
                    canSinkForward = false;
                    break;
                  }
                  if (orig->is<Call>() || orig->is<CallRef>() ||
                      orig->is<CallIndirect>()) {
                    if (hasTry && setInfluences.size() > 1) {
                      canSinkForward = false;
                      break;
                    }
                    hasThrowingCall = true;
                  }
                  if (auto* lg = orig->dynCast<LocalGet>()) {
                    if (lg->index == set->index) {
                      canSinkForward = false;
                      break;
                    }
                  } else if (auto* ls = orig->dynCast<LocalSet>()) {
                    if (ls->index == set->index) {
                      canSinkForward = false;
                      break;
                    }
                  }
                }
                SmallLocalSet setReadLocals;
                if (canSinkForward &&
                    !isPureSlice(setValStart,
                                 setIndex - 1,
                                 setReadLocals,
                                 passOptions.trapsNeverHappen &&
                                   !hasThrowingCall)) {
                  canSinkForward = false;
                }
                if (canSinkForward && !setReadLocals.empty()) {
                  for (Index p = setIndex + 1; p < getIndex; ++p) {
                    auto* pInst = insts[p];
                    if (!pInst) {
                      continue;
                    }
                    if (auto* ls = pInst->origin->dynCast<LocalSet>()) {
                      if (setReadLocals.contains(ls->index)) {
                        canSinkForward = false;
                        break;
                      }
                    }
                  }
                }
                if (canSinkForward) {
                  if (setInfluences.size() == 1 &&
                      canRemoveSetGetPair(setIndex, getIndex)) {
                    insts[setIndex] = nullptr;
                    insts[getIndex] = nullptr;
                  } else {
                    insts[setIndex]->type = func->getLocalType(set->index);
                    insts[getIndex] = nullptr;
                  }
                  Index sliceLen = setIndex - setValStart + 1;
                  std::rotate(insts.begin() + setValStart,
                              insts.begin() + setIndex + 1,
                              insts.begin() + getIndex);
                  values.erase(values.begin() + j);
                  for (size_t m = j; m < values.size(); ++m) {
                    if (values[m].setIndex != null) {
                      values[m].setIndex -= sliceLen;
                    }
                    values[m].valStart -= sliceLen;
                    values[m].valEnd -= sliceLen;
                  }
                  values.push_back({null, getIndex - sliceLen, getIndex - 1});
                  optimized = true;
                  break;
                }

                // Direction 2: Backward pure-subtree rotation.
                size_t firstValIdx = values.size() - numInterveningVals;
                bool canRotateBackward =
                  setValStart < setIndex && firstValIdx > j;
                for (size_t m = j + 1; canRotateBackward && m < firstValIdx;
                     ++m) {
                  if (values[m].setIndex == null ||
                      values[m].valStart <= setIndex) {
                    canRotateBackward = false;
                  }
                }
                for (size_t m = firstValIdx;
                     canRotateBackward && m < values.size();
                     ++m) {
                  if (values[m].setIndex != null ||
                      values[m].valStart > values[m].valEnd) {
                    canRotateBackward = false;
                    break;
                  }
                  Index nextStart =
                    (m + 1 < values.size()) ? values[m + 1].valStart : getIndex;
                  if (values[m].valEnd >= nextStart) {
                    canRotateBackward = false;
                    break;
                  }
                  for (Index p = values[m].valEnd + 1; p < nextStart; ++p) {
                    if (insts[p]) {
                      canRotateBackward = false;
                      break;
                    }
                  }
                }
                Index ivFirstStart =
                  canRotateBackward ? values[firstValIdx].valStart : 0;
                if (canRotateBackward && ivFirstStart <= setIndex) {
                  canRotateBackward = false;
                }
                for (size_t m = j + 1; canRotateBackward && m < firstValIdx;
                     ++m) {
                  if (values[m].setIndex >= ivFirstStart) {
                    canRotateBackward = false;
                  }
                }
                int depth = 0;
                bool hasBranchOrThrow = false;
                for (Index p = setValStart;
                     canRotateBackward && p < ivFirstStart;
                     ++p) {
                  auto* pInst = insts[p];
                  if (!pInst) {
                    continue;
                  }
                  if (pInst->op != StackInst::Basic) {
                    if (pInst->op == StackInst::LoopBegin ||
                        pInst->op == StackInst::LoopEnd ||
                        pInst->op == StackInst::TryBegin ||
                        pInst->op == StackInst::TryEnd ||
                        pInst->op == StackInst::Catch ||
                        pInst->op == StackInst::CatchAll ||
                        pInst->op == StackInst::Delegate ||
                        pInst->op == StackInst::TryTableBegin ||
                        pInst->op == StackInst::TryTableEnd) {
                      canRotateBackward = false;
                      break;
                    }
                    if (isControlFlowBegin(pInst)) {
                      depth++;
                      hasBranchOrThrow = true;
                    } else if (isControlFlowEnd(pInst)) {
                      if (depth == 0) {
                        canRotateBackward = false;
                        break;
                      }
                      depth--;
                      hasBranchOrThrow = true;
                    } else if (pInst->op == StackInst::IfElse) {
                      if (depth == 0) {
                        canRotateBackward = false;
                        break;
                      }
                      hasBranchOrThrow = true;
                    }
                    continue;
                  }
                  auto* orig = pInst->origin;
                  if (orig->is<Pop>() || orig->is<Return>() ||
                      orig->is<Throw>() || orig->is<Rethrow>() ||
                      orig->is<Unreachable>()) {
                    canRotateBackward = false;
                    break;
                  }
                  if (Properties::isBranch(orig)) {
                    if (depth == 0) {
                      canRotateBackward = false;
                      break;
                    }
                    hasBranchOrThrow = true;
                  }
                  if (orig->is<Call>() || orig->is<CallRef>() ||
                      orig->is<CallIndirect>()) {
                    if (!passOptions.trapsNeverHappen || depth > 0) {
                      hasBranchOrThrow = true;
                    }
                  }
                  if (p > setIndex) {
                    if (auto* ls = orig->dynCast<LocalSet>()) {
                      if (ls->index == set->index) {
                        canRotateBackward = false;
                        break;
                      }
                    } else if (auto* lg = orig->dynCast<LocalGet>()) {
                      if (lg->index == set->index) {
                        canRotateBackward = false;
                        break;
                      }
                    }
                  }
                }
                if (depth != 0) {
                  canRotateBackward = false;
                }
                SmallLocalSet ivReadLocals;
                for (size_t m = firstValIdx;
                     canRotateBackward && m < values.size();
                     ++m) {
                  if (!isPureSlice(values[m].valStart,
                                   values[m].valEnd,
                                   ivReadLocals,
                                   passOptions.trapsNeverHappen &&
                                     !hasBranchOrThrow)) {
                    canRotateBackward = false;
                    break;
                  }
                }
                if (ivReadLocals.contains(set->index)) {
                  canRotateBackward = false;
                }
                if (canRotateBackward && !ivReadLocals.empty()) {
                  for (Index p = setValStart; p < getIndex; ++p) {
                    auto* pInst = insts[p];
                    if (!pInst) {
                      continue;
                    }
                    if (auto* ls = pInst->origin->dynCast<LocalSet>()) {
                      if (ivReadLocals.contains(ls->index)) {
                        canRotateBackward = false;
                        break;
                      }
                    }
                  }
                }
                if (canRotateBackward) {
                  if (setInfluences.size() == 1 &&
                      canRemoveSetGetPair(setIndex, getIndex)) {
                    insts[setIndex] = nullptr;
                    insts[getIndex] = nullptr;
                  } else {
                    insts[setIndex]->type = func->getLocalType(set->index);
                    insts[getIndex] = nullptr;
                  }
                  Index ivLen = getIndex - ivFirstStart;
                  Index prefixLen = ivFirstStart - setValStart;
                  std::rotate(insts.begin() + setValStart,
                              insts.begin() + ivFirstStart,
                              insts.begin() + getIndex);
                  values[j] = {null, setValStart + ivLen, setIndex + ivLen};
                  for (size_t m = j + 1; m < firstValIdx; ++m) {
                    values[m].setIndex += ivLen;
                    values[m].valStart += ivLen;
                    values[m].valEnd += ivLen;
                  }
                  for (size_t m = firstValIdx; m < values.size(); ++m) {
                    values[m].valStart -= prefixLen;
                    values[m].valEnd -= prefixLen;
                  }
                  std::rotate(values.begin() + j,
                              values.begin() + firstValIdx,
                              values.end());
                  optimized = true;
                  break;
                }
              }
            }
            // We failed here. Can we look some more?
            if (j == 0) {
              break;
            }
            j--;
          }
        }
      }
      if (!optimized) {
        // This is an actual regular value on the value stack.
        values.push_back({null, firstOperandStart, instIndex});
      }
    } else if (inst->origin->is<LocalSet>() && inst->type == Type::none) {
      // This set is potentially optimizable later, add to stack.
      values.push_back({instIndex, firstOperandStart, instIndex - 1});
    }
  }

  // Optimize the simple case of a multivalue tee and extract. If an expression
  // returns a tuple, and that tuple is immediately consumed, we end up with
  // something like this:
  //
  //  local.tee $1
  //  tuple.extract 4 0
  //  local.get $1
  //  tuple.extract 4 1
  //  local.get $1
  //  tuple.extract 4 2
  //  local.get $1
  //  tuple.extract 4 3
  //
  // The tuple is teed, then we extract the components one by one. If no other
  // uses of the tee exist, we can just remove all of this.
  for (Index instIndex = 0; instIndex < insts.size(); instIndex++) {
    auto* inst = insts[instIndex];
    if (!inst) {
      continue;
    }
    auto* tee = inst->origin->dynCast<LocalSet>();
    if (!tee || !tee->type.isTuple()) {
      continue;
    }

    // The tee must be read by exactly the proper number of gets, and no more,
    // which is one less than the tuple size (the tee provides one get).
    auto size = tee->type.size();
    auto& setInfluences = localGraph.getSetInfluences(tee);
    if (setInfluences.size() != size - 1) {
      continue;
    }

    // This is a tee of a tuple. Look for the expected extracts/gets. Each
    // tuple index has 2 items.
    bool ok = true;
    for (Index i = 0; i < size; i++) {
      // Each tuple index has a pair of items.
      auto tupleIndexStart = instIndex + i * 2;
      if (tupleIndexStart + 1 >= insts.size()) {
        ok = false;
        break;
      }
      auto* first = insts[tupleIndexStart];
      auto* second = insts[tupleIndexStart + 1];
      if (!first || !second) {
        ok = false;
        break;
      }
      // The first tuple index has the tee (already validated). Others have a
      // get.
      if (i != 0) {
        auto* get = first->origin->dynCast<LocalGet>();
        if (!get || get->index != tee->index) {
          ok = false;
          break;
        }
      }
      // The second item of the pair is an extract.
      auto* extract = second->origin->dynCast<TupleExtract>();
      if (!extract || extract->index != i) {
        ok = false;
        break;
      }
    }
    if (ok) {
      // Optimize.
      for (Index i = 0; i < size * 2; i++) {
        insts[instIndex + i] = nullptr;
      }
    }
  }
}

// There may be unnecessary blocks we can remove: blocks without arriving
// branches are always ok to remove, as are blocks whose end coincides with an
// enclosing function body, if arm, if expression, or outer block of the same
// written result type.
void StackIROptimizer::removeUnneededBlocks() {
  auto resolveLabel = [&](Name name) {
    while (true) {
      auto it = insts.labelAliases.find(name);
      if (it == insts.labelAliases.end()) {
        break;
      }
      name = it->second;
    }
    return name;
  };

  // First, find all branch targets.
  std::unordered_set<Name> targets;
  for (auto*& inst : insts) {
    if (inst) {
      BranchUtils::operateOnScopeNameUses(inst->origin, [&](Name& name) {
        targets.insert(name);
        targets.insert(resolveLabel(name));
      });
    }
  }

  // Remove untargeted blocks.
  for (auto*& inst : insts) {
    if (!inst) {
      continue;
    }
    if (auto* block = inst->origin->dynCast<Block>()) {
      if (!block->name.is() || !targets.contains(block->name)) {
        // TODO optimize, maybe run remove-unused-names
        inst = nullptr;
      }
    }
  }

  auto nextNonNull = [&](Index idx) -> Index {
    for (Index j = idx + 1; j < insts.size(); j++) {
      if (insts[j]) {
        return j;
      }
    }
    return insts.size();
  };

  // Index the remaining control-flow begins and ends, and track labels targeted
  // by non-dropped value-carrying br_if instructions (whose fallthrough type in
  // Wasm binary validation depends on the target label's exact written type).
  std::unordered_map<Block*, Index> blockBeginIdx;
  std::unordered_map<Block*, Index> blockEndIdx;
  std::unordered_map<Name, Block*> blockByName;
  std::unordered_map<If*, Index> ifBeginIdx;
  std::unordered_map<If*, Index> ifElseIdx;
  std::unordered_map<If*, Index> ifEndIdx;
  std::unordered_map<Loop*, Index> loopBeginIdx;
  std::unordered_map<Loop*, Index> loopEndIdx;
  std::unordered_set<Name> labelsWithUndroppedBrIf;
  std::vector<Block*> blocksByEndOrder;
  for (Index i = 0; i < insts.size(); i++) {
    auto* inst = insts[i];
    if (!inst) {
      continue;
    }
    switch (inst->op) {
      case StackInst::BlockBegin: {
        auto* b = inst->origin->cast<Block>();
        blockBeginIdx[b] = i;
        if (b->name.is()) {
          blockByName[b->name] = b;
        }
        break;
      }
      case StackInst::BlockEnd: {
        auto* b = inst->origin->cast<Block>();
        blockEndIdx[b] = i;
        blocksByEndOrder.push_back(b);
        break;
      }
      case StackInst::IfBegin:
        ifBeginIdx[inst->origin->cast<If>()] = i;
        break;
      case StackInst::IfElse:
        ifElseIdx[inst->origin->cast<If>()] = i;
        break;
      case StackInst::IfEnd:
        ifEndIdx[inst->origin->cast<If>()] = i;
        break;
      case StackInst::LoopBegin:
        loopBeginIdx[inst->origin->cast<Loop>()] = i;
        break;
      case StackInst::LoopEnd:
        loopEndIdx[inst->origin->cast<Loop>()] = i;
        break;
      case StackInst::Basic: {
        if (auto* br = inst->origin->dynCast<Break>()) {
          if (br->condition && br->value) {
            Index nextIdx = nextNonNull(i);
            if (nextIdx >= insts.size() ||
                insts[nextIdx]->op != StackInst::Basic ||
                !insts[nextIdx]->origin->is<Drop>()) {
              labelsWithUndroppedBrIf.insert(br->name);
              labelsWithUndroppedBrIf.insert(resolveLabel(br->name));
            }
          }
        }
        break;
      }
      default:
        break;
    }
  }

  auto compatibleWrittenType = [&](Type inner, Type outer, Block* innerBlock) {
    if (inner == Type::unreachable || outer == Type::unreachable) {
      return false;
    }
    Type innerW = inner.asWrittenGivenFeatures(features);
    Type outerW = outer.asWrittenGivenFeatures(features);
    if (innerW == outerW) {
      return true;
    }
    if (!Type::isSubType(innerW, outerW)) {
      return false;
    }
    return !labelsWithUndroppedBrIf.contains(innerBlock->name) &&
           !labelsWithUndroppedBrIf.contains(resolveLabel(innerBlock->name));
  };

  auto elideBlock = [&](Block* b) {
    auto itBegin = blockBeginIdx.find(b);
    auto itEnd = blockEndIdx.find(b);
    if (itBegin == blockBeginIdx.end() || itEnd == blockEndIdx.end()) {
      return false;
    }
    insts[itBegin->second] = nullptr;
    insts[itEnd->second] = nullptr;
    return true;
  };

  auto aliasLabel = [&](Name from, Name to) {
    Name resolvedTo = resolveLabel(to);
    if (from != resolvedTo) {
      insts.labelAliases[from] = resolvedTo;
      if (labelsWithUndroppedBrIf.contains(from)) {
        labelsWithUndroppedBrIf.insert(resolvedTo);
      }
    }
  };

  auto bindLabel = [&](Name& slot, Name newName) {
    if (!slot.is()) {
      slot = resolveLabel(newName);
    } else {
      aliasLabel(newName, slot);
    }
  };

  // 1(c) Nested tail-Block elision: when BlockEnd(b) is immediately followed
  // by BlockEnd(outerB) and b's written result type is compatible with outerB,
  // elide b and alias its label to outerB->name. Iterate in forward BlockEnd
  // order so inner blocks are elided into outer blocks before the outer blocks
  // are considered for function-body or if-scope elision.
  for (auto* b : blocksByEndOrder) {
    auto itBegin = blockBeginIdx.find(b);
    auto itEnd = blockEndIdx.find(b);
    if (itBegin == blockBeginIdx.end() || itEnd == blockEndIdx.end() ||
        !insts[itEnd->second]) {
      continue;
    }
    Index nextIdx = nextNonNull(itEnd->second);
    if (nextIdx >= insts.size()) {
      continue;
    }
    auto* nextInst = insts[nextIdx];
    if (nextInst->op == StackInst::BlockEnd) {
      auto* outerB = nextInst->origin->cast<Block>();
      auto itOuterBegin = blockBeginIdx.find(outerB);
      if (itOuterBegin != blockBeginIdx.end() &&
          itOuterBegin->second < itBegin->second &&
          compatibleWrittenType(b->type, outerB->type, b)) {
        if (elideBlock(b)) {
          aliasLabel(b->name, outerB->name);
        }
      }
    }
  }

  // 1(b.1) & 1(b.2) If-arm tail-Block elision: when BlockEnd(b) inside an If
  // arm is immediately followed by IfElse(iff) or IfEnd(iff) and b has a
  // written result type compatible with iff, elide b and bind iff's arm label.
  for (auto* b : blocksByEndOrder) {
    auto itBegin = blockBeginIdx.find(b);
    auto itEnd = blockEndIdx.find(b);
    if (itBegin == blockBeginIdx.end() || itEnd == blockEndIdx.end() ||
        !insts[itEnd->second]) {
      continue;
    }
    Index nextIdx = nextNonNull(itEnd->second);
    if (nextIdx >= insts.size()) {
      continue;
    }
    auto* nextInst = insts[nextIdx];
    if (nextInst->op == StackInst::IfElse) {
      auto* iff = nextInst->origin->cast<If>();
      auto itIfBegin = ifBeginIdx.find(iff);
      if (itIfBegin != ifBeginIdx.end() &&
          itIfBegin->second < itBegin->second &&
          compatibleWrittenType(b->type, iff->type, b)) {
        if (elideBlock(b)) {
          bindLabel(insts.ifTrueLabels[iff], b->name);
        }
      }
    } else if (nextInst->op == StackInst::IfEnd) {
      auto* iff = nextInst->origin->cast<If>();
      auto itIfBegin = ifBeginIdx.find(iff);
      if (itIfBegin != ifBeginIdx.end() &&
          itIfBegin->second < itBegin->second &&
          compatibleWrittenType(b->type, iff->type, b)) {
        auto itIfElse = ifElseIdx.find(iff);
        if (itIfElse == ifElseIdx.end()) {
          // 1-arm If: b is inside ifTrue.
          if (elideBlock(b)) {
            bindLabel(insts.ifTrueLabels[iff], b->name);
          }
        } else if (itIfElse->second < itBegin->second) {
          // 2-arm If: b is inside ifFalse.
          if (elideBlock(b)) {
            bindLabel(insts.ifFalseLabels[iff], b->name);
          }
        }
      }
    }
  }

  // 1(b.3) If-wrapper Block elision: when BlockEnd(b) is immediately after
  // IfEnd(iff), BlockBegin(b) precedes IfBegin(iff), no instruction in
  // (BlockBegin(b), IfBegin(iff)] branches to b->name, and b has a written
  // result type compatible with iff, elide b and bind iff's arm label(s).
  for (auto* b : blocksByEndOrder) {
    auto itBegin = blockBeginIdx.find(b);
    auto itEnd = blockEndIdx.find(b);
    if (itBegin == blockBeginIdx.end() || itEnd == blockEndIdx.end() ||
        !insts[itEnd->second]) {
      continue;
    }
    Index prevIdx = itEnd->second;
    while (prevIdx > 0 && !insts[prevIdx - 1]) {
      prevIdx--;
    }
    if (prevIdx == 0) {
      continue;
    }
    auto* prevInst = insts[prevIdx - 1];
    if (prevInst->op != StackInst::IfEnd) {
      continue;
    }
    auto* iff = prevInst->origin->cast<If>();
    auto itIfBegin = ifBeginIdx.find(iff);
    if (itIfBegin == ifBeginIdx.end() ||
        !(itBegin->second < itIfBegin->second) ||
        !compatibleWrittenType(b->type, iff->type, b)) {
      continue;
    }
    bool branchBeforeIf = false;
    for (Index j = itBegin->second + 1; j <= itIfBegin->second; j++) {
      if (!insts[j]) {
        continue;
      }
      BranchUtils::operateOnScopeNameUses(insts[j]->origin, [&](Name& name) {
        if (resolveLabel(name) == b->name) {
          branchBeforeIf = true;
        }
      });
      if (branchBeforeIf) {
        break;
      }
    }
    if (branchBeforeIf) {
      continue;
    }
    if (elideBlock(b)) {
      auto bindOuterToArm = [&](Name& armLabel) {
        if (!armLabel.is()) {
          armLabel = b->name;
        } else {
          Name resolvedArm = resolveLabel(armLabel);
          if (resolvedArm != b->name) {
            aliasLabel(resolvedArm, b->name);
          }
          armLabel = b->name;
        }
      };
      bindOuterToArm(insts.ifTrueLabels[iff]);
      if (ifElseIdx.contains(iff)) {
        bindOuterToArm(insts.ifFalseLabels[iff]);
      }
    }
  }

  // 1(a) Function-body tail Block elision: when BlockEnd(b) is the last
  // non-null instruction in stackIR and b has a written result type compatible
  // with func->getResults(), elide b and bind functionBodyLabel.
  for (Index i = insts.size(); i > 0; i--) {
    auto* inst = insts[i - 1];
    if (!inst) {
      continue;
    }
    if (inst->op == StackInst::BlockEnd) {
      auto* b = inst->origin->cast<Block>();
      if (compatibleWrittenType(b->type, func->getResults(), b)) {
        if (elideBlock(b)) {
          bindLabel(insts.functionBodyLabel, b->name);
          continue;
        }
      }
    }
    break;
  }

  // 1(d) Unconditional-exit Block elision: when BlockEnd(b) is immediately
  // followed by an unconditional Return or Break consuming b's result (or void
  // after a void block), elide b and redirect its label to the exit target.
  for (auto* b : blocksByEndOrder) {
    auto itBegin = blockBeginIdx.find(b);
    auto itEnd = blockEndIdx.find(b);
    if (itBegin == blockBeginIdx.end() || itEnd == blockEndIdx.end() ||
        !insts[itEnd->second]) {
      continue;
    }
    Index nextIdx = nextNonNull(itEnd->second);
    if (nextIdx >= insts.size()) {
      continue;
    }
    auto* nextInst = insts[nextIdx];
    if (nextInst->op != StackInst::Basic) {
      continue;
    }
    auto elideBeforeExit = [&]() {
      if (!elideBlock(b)) {
        return false;
      }
      Index prevIdx = itEnd->second;
      while (prevIdx > 0 && !insts[prevIdx - 1]) {
        prevIdx--;
      }
      if (prevIdx > 0 && insts[prevIdx - 1]->type == Type::unreachable) {
        insts[nextIdx] = nullptr;
      }
      return true;
    };
    if (auto* ret = nextInst->origin->dynCast<Return>()) {
      if ((b->type != Type::none) == (ret->value != nullptr) &&
          compatibleWrittenType(b->type, func->getResults(), b)) {
        if (elideBeforeExit()) {
          bindLabel(insts.functionBodyLabel, b->name);
        }
      }
    } else if (auto* br = nextInst->origin->dynCast<Break>()) {
      if (br->condition || (b->type != Type::none) != (br->value != nullptr)) {
        continue;
      }
      Name targetName = resolveLabel(br->name);
      if (!targetName.is() || targetName == b->name) {
        continue;
      }
      if (insts.functionBodyLabel.is() &&
          targetName == resolveLabel(insts.functionBodyLabel)) {
        if (compatibleWrittenType(b->type, func->getResults(), b)) {
          if (elideBeforeExit()) {
            bindLabel(insts.functionBodyLabel, b->name);
          }
        }
        continue;
      }
      if (auto itTargetB = blockByName.find(targetName);
          itTargetB != blockByName.end()) {
        auto* targetB = itTargetB->second;
        auto itTargetBegin = blockBeginIdx.find(targetB);
        auto itTargetEnd = blockEndIdx.find(targetB);
        if (itTargetBegin != blockBeginIdx.end() &&
            itTargetEnd != blockEndIdx.end() && insts[itTargetBegin->second] &&
            itTargetBegin->second < itBegin->second &&
            itEnd->second < itTargetEnd->second &&
            compatibleWrittenType(b->type, targetB->type, b)) {
          if (elideBeforeExit()) {
            aliasLabel(b->name, targetB->name);
          }
        }
        continue;
      }
      bool matchedIf = false;
      for (auto& [iff, trueLabel] : insts.ifTrueLabels) {
        if (trueLabel.is() && resolveLabel(trueLabel) == targetName) {
          auto itIfBegin = ifBeginIdx.find(iff);
          auto itIfEnd = ifEndIdx.find(iff);
          if (itIfBegin != ifBeginIdx.end() && itIfEnd != ifEndIdx.end() &&
              itIfBegin->second < itBegin->second &&
              itEnd->second < itIfEnd->second &&
              compatibleWrittenType(b->type, iff->type, b)) {
            if (elideBeforeExit()) {
              aliasLabel(b->name, targetName);
            }
          }
          matchedIf = true;
          break;
        }
      }
      if (matchedIf) {
        continue;
      }
      for (auto& [iff, falseLabel] : insts.ifFalseLabels) {
        if (falseLabel.is() && resolveLabel(falseLabel) == targetName) {
          auto itIfBegin = ifBeginIdx.find(iff);
          auto itIfEnd = ifEndIdx.find(iff);
          if (itIfBegin != ifBeginIdx.end() && itIfEnd != ifEndIdx.end() &&
              itIfBegin->second < itBegin->second &&
              itEnd->second < itIfEnd->second &&
              compatibleWrittenType(b->type, iff->type, b)) {
            if (elideBeforeExit()) {
              aliasLabel(b->name, targetName);
            }
          }
          break;
        }
      }
    }
  }
}

// A control flow "barrier" - a point where stack machine
// unreachability ends.
bool StackIROptimizer::isControlFlowBarrier(StackInst* inst) {
  switch (inst->op) {
    case StackInst::BlockEnd:
    case StackInst::IfElse:
    case StackInst::IfEnd:
    case StackInst::LoopEnd:
    case StackInst::Catch:
    case StackInst::CatchAll:
    case StackInst::Delegate:
    case StackInst::TryEnd:
    case StackInst::TryTableEnd: {
      return true;
    }
    default: {
      return false;
    }
  }
}

// A control flow beginning.
bool StackIROptimizer::isControlFlowBegin(StackInst* inst) {
  switch (inst->op) {
    case StackInst::BlockBegin:
    case StackInst::IfBegin:
    case StackInst::LoopBegin:
    case StackInst::TryBegin:
    case StackInst::TryTableBegin: {
      return true;
    }
    default: {
      return false;
    }
  }
}

// A control flow ending.
bool StackIROptimizer::isControlFlowEnd(StackInst* inst) {
  switch (inst->op) {
    case StackInst::BlockEnd:
    case StackInst::IfEnd:
    case StackInst::LoopEnd:
    case StackInst::TryEnd:
    case StackInst::Delegate:
    case StackInst::TryTableEnd: {
      return true;
    }
    default: {
      return false;
    }
  }
}

bool StackIROptimizer::isControlFlow(StackInst* inst) {
  return inst->op != StackInst::Basic;
}

// Remove the instruction at index i. If the instruction
// is control flow, and so has been expanded to multiple
// instructions, remove them as well.
void StackIROptimizer::removeAt(Index i) {
  auto* inst = insts[i];
  insts[i] = nullptr;
  if (inst->op == StackInst::Basic) {
    return; // that was it
  }
  auto* origin = inst->origin;
  while (1) {
    i++;
    assert(i < insts.size());
    inst = insts[i];
    insts[i] = nullptr;
    if (inst && inst->origin == origin && isControlFlowEnd(inst)) {
      return; // that's it, we removed it all
    }
  }
}

Index StackIROptimizer::getNumConsumedValues(StackInst* inst) {
  if (isControlFlow(inst)) {
    // If consumes 1; that's it.
    if (inst->op == StackInst::IfBegin) {
      return 1;
    }
    return 0;
  }
  // Otherwise, for basic instructions, just count the expression children.
  return ChildIterator(inst->origin).children.size();
}

// Given a pair of a local.set and local.get, see if we can remove them
// without breaking validation. Specifically, we must keep sets of non-
// nullable locals that dominate a get until the end of the block, such as
// here:
//
//  local.set 0    ;; Can we remove
//  local.get 0    ;; this pair?
//  if
//    local.set 0
//  else
//    local.set 0
//  end
//  local.get 0    ;; This get poses a problem.
//
// Logically the 2nd&3rd sets ensure a value is applied to the local before we
// read it, but the validation rules only track each set until the end of its
// scope, so the 1st set (before the if, in the pair) is necessary.
//
// The logic below is related to LocalStructuralDominance, but sharing code
// with it is difficult as this uses StackIR and not BinaryenIR, and it checks
// a particular set/get pair.
//
// We are given the indexes of the set and get instructions in |insts|.
bool StackIROptimizer::canRemoveSetGetPair(Index setIndex, Index getIndex) {
  // The set must be before the get.
  assert(setIndex < getIndex);

  auto* set = insts[setIndex]->origin->cast<LocalSet>();
  auto localType = func->getLocalType(set->index);
  // Note we do not need to handle tuples here, as the parent ignores them
  // anyhow (hence we can check non-nullability instead of non-
  // defaultability).
  assert(localType.isSingle());
  if (func->isParam(set->index) || !localType.isNonNullable()) {
    // This local cannot pose a problem for validation (params are always
    // initialized, and it is ok if nullable locals are uninitialized).
    return true;
  }

  // Track the depth (in block/if/loop/etc. scopes) relative to our starting
  // point. Anything less deep than that is not interesting, as we can only
  // help things at our depth or deeper to validate.
  Index currDepth = 0;

  // Look for a different get than the one in getIndex (since that one is
  // being removed) which would stop validating without the set. While doing
  // so, note other sets that ensure validation even if our set is removed. We
  // track those in this stack of booleans, one for each scope, which is true
  // if another sets covers us and ours is not needed.
  //
  // We begin in the current scope and with no other set covering us.
  std::vector<bool> coverStack = {false};

  // Track the total number of covers as well, for quick checking below.
  Index covers = 0;

  // TODO: We could look before us as well, but then we might end up scanning
  //       much of the function every time.
  for (Index i = setIndex + 1; i < insts.size(); i++) {
    auto* inst = insts[i];
    if (!inst) {
      continue;
    }
    if (isControlFlowBegin(inst)) {
      // A new scope begins.
      currDepth++;
      coverStack.push_back(false);
    } else if (isControlFlowEnd(inst)) {
      if (currDepth == 0) {
        // Less deep than the start, so we found no problem.
        return true;
      }
      currDepth--;

      if (coverStack.back()) {
        // A cover existed in the scope which ended.
        covers--;
      }
      coverStack.pop_back();
    } else if (isControlFlowBarrier(inst)) {
      // A barrier, like the else in an if-else, not only ends a scope but
      // opens a new one.
      if (currDepth == 0) {
        // Another scope with the same depth begins, but ours ended, so stop.
        return true;
      }

      if (coverStack.back()) {
        // A cover existed in the scope which ended.
        covers--;
      }
      coverStack.back() = false;
    } else if (auto* otherSet = inst->origin->dynCast<LocalSet>()) {
      // We are covered in this scope henceforth.
      if (otherSet->index == set->index) {
        if (!coverStack.back()) {
          covers++;
          if (currDepth == 0) {
            // We have a cover at depth 0, so everything from here on out
            // will be covered.
            return true;
          }
          coverStack.back() = true;
        }
      }
    } else if (auto* otherGet = inst->origin->dynCast<LocalGet>()) {
      if (otherGet->index == set->index && i != getIndex && !covers) {
        // We found a get that might be a problem: it uses the same index, but
        // is not the get we were told about, and no other set covers us.
        return false;
      }
    }
  }

  // No problem.
  return true;
}

std::unordered_set<LocalGet*> StackIROptimizer::findStringViewDeferredGets() {
  std::unordered_set<LocalGet*> deferred;
  auto note = [&](Expression* e) {
    if (auto* get = e->dynCast<LocalGet>()) {
      deferred.insert(get);
    }
  };
  for (auto* inst : insts) {
    if (!inst) {
      continue;
    }
    if (auto* curr = inst->origin->dynCast<StringWTF16Get>()) {
      note(curr->pos);
    } else if (auto* curr = inst->origin->dynCast<StringSliceWTF>()) {
      note(curr->start);
      note(curr->end);
    }
  }
  return deferred;
}

} // namespace wasm
