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
// We track values for (reference local, field index) pairs in linear
// execution traces, and so we only handle loads dominated by the store, with
// nothing dangerous in between. A tracked value is forgotten when
//
//  * the reference local or the local the value was read from is set,
//  * a possibly-aliasing reference (one of a related type) is used to write
//    to the same field index, or
//  * anything with effects that may write to a struct field executes,
//    such as a call.
//
// Note that a forwarded load can never trap: the store we forward from
// already proved that the (unchanged) reference is non-null.
//
// TODO: Also forward from packed fields, by masking the stored value.
// TODO: Also forward array.set values to array.gets of constant indexes.
//

#include "ir/effects.h"
#include "ir/linear-execution.h"
#include "ir/manipulation.h"
#include "ir/properties.h"
#include "ir/utils.h"
#include "pass.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

struct LoadStoreForwarding
  : public WalkerPass<
      LinearExecutionWalker<LoadStoreForwarding,
                            UnifiedExpressionVisitor<LoadStoreForwarding>>> {
  using Super = WalkerPass<
    LinearExecutionWalker<LoadStoreForwarding,
                          UnifiedExpressionVisitor<LoadStoreForwarding>>>;

  bool isFunctionParallel() override { return true; }

  // We only replace struct.gets with existing values; locals are unchanged.
  bool requiresNonNullableLocalFixups() override { return false; }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<LoadStoreForwarding>();
  }

  LoadStoreForwarding() {
    // We only optimize loads dominated by the store, and never move code, so
    // it is safe to look at adjacent basic blocks together: if the later part
    // of the trace is not reached, a change there does not matter.
    connectAdjacentBlocks = true;
  }

  // A value known to be present in a struct field.
  struct KnownValue {
    // The heap type of the reference the value was stored through. Used to
    // reason about aliasing: a write through a reference of an unrelated type
    // cannot refer to the same object.
    HeapType type;
    // The stored value, a local.get or a constant.
    Expression* value;
  };

  // Maps (reference local index, field index) to the value last stored there.
  using KnownValues = std::map<std::pair<Index, Index>, KnownValue>;
  KnownValues knownValues;

  // Whether we replaced a get with a value of a more refined type, which
  // requires refinalization at the end.
  bool refinalize = false;

  void doWalkFunction(Function* func) {
    if (!getModule()->features.hasGC()) {
      return;
    }
    knownValues.clear();
    refinalize = false;
    Super::doWalkFunction(func);
    if (refinalize) {
      ReFinalize().walkFunctionInModule(func, getModule());
    }
  }

  void noteNonLinear(Expression* curr) { knownValues.clear(); }

  void visitExpression(Expression* curr) {
    if (auto* set = curr->dynCast<StructSet>()) {
      handleStructSet(set);
      return;
    }
    if (auto* get = curr->dynCast<StructGet>()) {
      handleStructGet(get);
      return;
    }
    if (auto* set = curr->dynCast<LocalSet>()) {
      handleLocalSet(set);
      return;
    }

    // Anything else: check (shallowly; children were already visited) for
    // effects that could change a struct field.
    if (knownValues.empty()) {
      return;
    }
    ShallowEffectAnalyzer effects(getPassOptions(), *getModule(), curr);
    // Note that we need not check for writes to shared structs, as we only
    // track unshared ones (and an unshared reference can never alias a shared
    // one).
    if (effects.calls || effects.writesStruct) {
      knownValues.clear();
    }
  }

  void handleStructSet(StructSet* curr) {
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
    std::erase_if(knownValues, [&](const KnownValues::value_type& entry) {
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
    knownValues[{ref->index, curr->index}] = {heapType, curr->value};
  }

  void handleStructGet(StructGet* curr) {
    if (curr->order != MemoryOrder::Unordered) {
      return;
    }
    auto* ref = curr->ref->dynCast<LocalGet>();
    if (!ref) {
      return;
    }
    auto iter = knownValues.find({ref->index, curr->index});
    if (iter == knownValues.end()) {
      return;
    }
    auto* value = ExpressionManipulator::copy(iter->second.value, *getModule());
    if (value->type != curr->type) {
      // The known value's type may be more refined than the field's.
      refinalize = true;
    }
    replaceCurrent(value);
  }

  void handleLocalSet(LocalSet* curr) {
    // The local has a new value: forget anything stored through it, and any
    // known value that reads it.
    std::erase_if(knownValues, [&](const KnownValues::value_type& entry) {
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
        knownValues[{curr->index, i}] = {
          heapType,
          builder.makeConstantExpression(Literal::makeZero(fields[i].type))};
      } else if (!fields[i].isPacked() &&
                 Properties::isSingleConstantExpression(new_->operands[i])) {
        // Note that we do not track local.get operands here, unlike for
        // struct.set: a later operand or the descriptor could change the
        // local after it is read but before the allocation is assigned.
        knownValues[{curr->index, i}] = {heapType, new_->operands[i]};
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
