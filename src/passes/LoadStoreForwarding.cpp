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

//
// Load-store forwarding for GC references. When a struct field is stored
// with a value we can repeat later - a local.get or a constant - and then
// loaded from the same object before anything can change the field, we can
// use the stored value directly, avoiding the load:
//
//  (struct.set $T 0 (local.get $ref) (local.get $value))
//  ..
//  (struct.get $T 0 (local.get $ref)) ;; can be (local.get $value)
//
// We also learn the values of constant-initialized fields from allocations:
//
//  (local.set $ref (struct.new $T (i32.const 42)))
//  ..
//  (struct.get $T 0 (local.get $ref)) ;; can be (i32.const 42)
//
// This is a forward "must be available" dataflow analysis on the CFG: we
// track the values known to be in (reference local, field index) locations
// through each basic block, and at control flow merges keep only the values
// that all predecessors agree on. In particular, values flow into and around
// loops when no part of the loop interferes with them. A tracked value is
// forgotten when
//
//  * the reference local, or the local the value was read from, is set,
//  * a possibly-aliasing reference (one of a related type) is used to write
//    to the same field index, or
//  * anything executes whose effects may write to a struct field, such as a
//    call.
//
// Note that a forwarded load can never trap: every path to the load passed
// through the store we forward from, which already proved that the
// (since-unchanged) reference is non-null.
//
// TODO: Also forward from packed fields, by masking the stored value.
// TODO: Also forward array.set values to array.gets of constant indexes.
//

#include <optional>

#include "cfg/cfg-traversal.h"
#include "ir/effects.h"
#include "ir/manipulation.h"
#include "ir/properties.h"
#include "ir/utils.h"
#include "pass.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

// A value known to be present in a struct field.
struct KnownValue {
  // The heap type of the reference the value was stored through. Used to
  // reason about aliasing: a write through a reference of an unrelated type
  // cannot refer to the same object.
  HeapType type;
  // The stored value, a local.get or a constant.
  Expression* value;

  bool operator==(const KnownValue& other) const {
    return type == other.type && ExpressionAnalyzer::equal(value, other.value);
  }
};

// Maps (reference local index, field index) to the value last stored there.
using KnownValues = std::map<std::pair<Index, Index>, KnownValue>;

// The information we track per basic block.
struct Info {
  // The known values at the start of the block. As long as this is nullopt
  // the block has not been reached by the flow (and if it remains so, the
  // block is unreachable).
  std::optional<KnownValues> start;
  // The expressions in the block that can affect or use the known values:
  // struct.sets, struct.gets, local.sets, and clobbers (things that may write
  // to struct fields, like calls).
  std::vector<Expression**> actions;
};

struct LoadStoreForwarding
  : public WalkerPass<
      CFGWalker<LoadStoreForwarding,
                UnifiedExpressionVisitor<LoadStoreForwarding>,
                Info>> {
  using Super =
    WalkerPass<CFGWalker<LoadStoreForwarding,
                         UnifiedExpressionVisitor<LoadStoreForwarding>,
                         Info>>;

  bool isFunctionParallel() override { return true; }

  // We only replace struct.gets with existing values; locals are unchanged.
  bool requiresNonNullableLocalFixups() override { return false; }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<LoadStoreForwarding>();
  }

  // Branches outside of the function can be ignored, as a path that leaves
  // the function can never reach one of the loads we optimize.
  bool ignoreBranchesOutsideOfFunc = true;

  // Whether we replaced a get with a value of a more refined type, which
  // requires refinalization at the end.
  bool refinalize = false;

  // While building the CFG, note the expressions relevant to the analysis.
  void visitExpression(Expression* curr) {
    if (!currBasicBlock) {
      return;
    }
    if (curr->is<StructSet>() || curr->is<StructGet>() || curr->is<LocalSet>()) {
      currBasicBlock->contents.actions.push_back(getCurrentPointer());
      return;
    }
    // Note anything else that may write to a struct field, as a clobber.
    // (Shallow effects suffice: children are recorded separately.)
    ShallowEffectAnalyzer effects(getPassOptions(), *getModule(), curr);
    // We need not check for writes to shared structs, as we only track
    // unshared ones (and an unshared reference can never alias a shared one).
    if (effects.calls || effects.writesStruct) {
      currBasicBlock->contents.actions.push_back(getCurrentPointer());
    }
  }

  void doWalkFunction(Function* func) {
    if (!getModule()->features.hasGC()) {
      return;
    }
    refinalize = false;

    // Build the CFG, noting the relevant actions in each basic block.
    Super::doWalkFunction(func);

    // Flow the known values to a fixed point. As states only ever shrink
    // under the intersections at merges, this must converge.
    entry->contents.start.emplace();
    std::vector<BasicBlock*> work = {entry};
    while (!work.empty()) {
      auto* block = work.back();
      work.pop_back();
      // Compute the state at the end of the block, and propagate it to our
      // successors, adding them to the work list if that changed anything.
      auto state = *block->contents.start;
      for (auto** currp : block->contents.actions) {
        transfer(state, currp, /*apply=*/false);
      }
      for (auto* succ : block->out) {
        auto& succStart = succ->contents.start;
        if (!succStart) {
          // The first time the successor is reached, it gets our state.
          succStart = state;
          work.push_back(succ);
        } else if (intersectInto(*succStart, state)) {
          work.push_back(succ);
        }
      }
    }

    // Now that we know the state at the start of each block, do a final scan
    // of each reachable block in which we forward values to loads.
    for (auto& block : basicBlocks) {
      if (!block->contents.start) {
        // This block is unreachable.
        continue;
      }
      auto state = *block->contents.start;
      for (auto** currp : block->contents.actions) {
        transfer(state, currp, /*apply=*/true);
      }
    }

    if (refinalize) {
      ReFinalize().walkFunctionInModule(func, getModule());
    }
  }

  // Intersect |src| into |target|, keeping only the values they agree on.
  // Returns whether |target| changed.
  bool intersectInto(KnownValues& target, const KnownValues& src) {
    bool changed = false;
    for (auto it = target.begin(); it != target.end();) {
      auto srcIt = src.find(it->first);
      if (srcIt != src.end() && srcIt->second == it->second) {
        ++it;
      } else {
        it = target.erase(it);
        changed = true;
      }
    }
    return changed;
  }

  // Update |state| across one action. In apply mode we also forward known
  // values to loads, replacing them through the given pointer.
  void transfer(KnownValues& state, Expression** currp, bool apply) {
    auto* curr = *currp;
    if (auto* set = curr->dynCast<StructSet>()) {
      transferStructSet(state, set);
    } else if (auto* get = curr->dynCast<StructGet>()) {
      transferStructGet(state, get, apply ? currp : nullptr);
    } else if (auto* set = curr->dynCast<LocalSet>()) {
      transferLocalSet(state, set);
    } else {
      // We only recorded one other kind of action, a clobber.
      state.clear();
    }
  }

  void transferStructSet(KnownValues& state, StructSet* curr) {
    auto refType = curr->ref->type;
    if (!refType.isStruct()) {
      // This is unreachable code, or it traps on a null reference; either
      // way, nothing is written.
      return;
    }
    auto heapType = refType.getHeapType();

    // This write may alias anything we track with the same field index and a
    // related type, making those values stale. (Types in unrelated parts of
    // the hierarchy have no common subtype, so they can never refer to the
    // same object.) If this set is to the very same local and field we may
    // re-record it below.
    std::erase_if(state, [&](const KnownValues::value_type& entry) {
      auto& [key, known] = entry;
      return key.second == curr->index &&
             (HeapType::isSubType(known.type, heapType) ||
              HeapType::isSubType(heapType, known.type));
    });

    if (heapType.isShared()) {
      // Other threads might write to this object, so do not track it.
      return;
    }
    if (curr->order != MemoryOrder::Unordered) {
      // Leave atomics alone.
      return;
    }
    auto* ref = curr->ref->dynCast<LocalGet>();
    if (!ref) {
      return;
    }
    if (heapType.getStruct().fields[curr->index].isPacked()) {
      // The stored value is truncated, so the loaded value may differ.
      return;
    }
    if (!isForwardable(curr->value)) {
      return;
    }
    state[{ref->index, curr->index}] = {heapType, curr->value};
  }

  void transferStructGet(KnownValues& state, StructGet* curr,
                         Expression** currp) {
    if (!currp) {
      // We are only computing states, and a load changes nothing.
      return;
    }
    if (curr->order != MemoryOrder::Unordered) {
      return;
    }
    auto* ref = curr->ref->dynCast<LocalGet>();
    if (!ref) {
      return;
    }
    auto iter = state.find({ref->index, curr->index});
    if (iter == state.end()) {
      return;
    }
    auto* value = ExpressionManipulator::copy(iter->second.value, *getModule());
    if (value->type != curr->type) {
      // The known value's type may be more refined than the field's.
      refinalize = true;
    }
    *currp = value;
  }

  void transferLocalSet(KnownValues& state, LocalSet* curr) {
    // The local has a new value: forget anything stored through it, and any
    // known value that reads it.
    std::erase_if(state, [&](const KnownValues::value_type& entry) {
      auto& [key, known] = entry;
      if (key.first == curr->index) {
        return true;
      }
      auto* get = known.value->dynCast<LocalGet>();
      return get && get->index == curr->index;
    });

    // If the local now refers to a fresh allocation then we know the contents
    // of constant-initialized fields.
    auto* new_ = curr->value->dynCast<StructNew>();
    if (!new_ || new_->type == Type::unreachable) {
      return;
    }
    auto heapType = new_->type.getHeapType();
    if (heapType.isShared()) {
      return;
    }
    auto& fields = heapType.getStruct().fields;
    Builder builder(*getModule());
    for (Index i = 0; i < fields.size(); i++) {
      if (new_->isWithDefault()) {
        // (Packed fields are fine here, as truncating zero changes nothing.)
        state[{curr->index, i}] = {
          heapType,
          builder.makeConstantExpression(Literal::makeZero(fields[i].type))};
      } else if (!fields[i].isPacked() &&
                 Properties::isSingleConstantExpression(new_->operands[i])) {
        // Note that we do not track local.get operands here, unlike for
        // struct.set: a later operand or the descriptor could change the
        // local after it is read but before the allocation is assigned.
        state[{curr->index, i}] = {heapType, new_->operands[i]};
      }
    }
  }

  bool isForwardable(Expression* curr) {
    // local.gets are handled by invalidating on sets of the local; constants
    // never change.
    return curr->is<LocalGet>() || Properties::isSingleConstantExpression(curr);
  }
};

} // anonymous namespace

Pass* createLoadStoreForwardingPass() { return new LoadStoreForwarding(); }

} // namespace wasm
