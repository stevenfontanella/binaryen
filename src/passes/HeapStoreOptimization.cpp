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
#include "ir/local-graph.h"
#include "ir/manipulation.h"
#include "ir/properties.h"
#include "ir/utils.h"
#include "pass.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

// In each basic block we will store the relevant heap store operations and
// other actions that matter to our analysis.
struct Info {
  std::vector<Expression**> actions;
};

struct HeapStoreOptimization
  : public WalkerPass<
      CFGWalker<HeapStoreOptimization, Visitor<HeapStoreOptimization>, Info>> {
  bool isFunctionParallel() override { return true; }

  // Locals are not modified here.
  bool requiresNonNullableLocalFixups() override { return false; }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<HeapStoreOptimization>();
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
        auto* curr = *currp;
        if (auto* set = curr->dynCast<StructSet>()) {
          optimizeStructSet(set, currp);
        } else if (auto* block = curr->dynCast<Block>()) {
          optimizeBlock(block);
        } else {
          WASM_UNREACHABLE("bad action");
        }
      }
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
    // If our reference is a tee of a struct.new, we may be able to fold the
    // stored value into the new itself:
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
      if (!new_ || new_->type == Type::unreachable) {
        continue;
      }

      HeapType newHT = new_->type.getHeapType();
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
        if (!structSet || !localGet || localGet->index != localSet->index) {
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
        if (canPassIntervening &&
            optimizeSubsequentStructSet(new_, structSet, localSet)) {
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

    if (list[j]->is<LocalSet>() &&
        list[j]->dynCast<LocalSet>()->value->is<StructNew>()) {
      // Don't swap two struct.new instructions to avoid going back and forth.
      return false;
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
    if (new_->isWithDefault() && getPassOptions().shrinkLevel > 0) {
      if (Properties::isConstantExpression(set->value) &&
          Properties::getLiteral(set->value) ==
            Literal::makeZero(fields[index].type)) {
        isRedundantDefaultStore = true;
      } else if (fields.size() > 3) {
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
        auto& field = fields[f2];
        Expression* rep = nullptr;
        if (new_->isWithDefault()) {
          rep = builder.makeConstantExpression(Literal::makeZero(field.type));
        } else if (auto* tee2 = operands[f2]->dynCast<LocalSet>()) {
          rep = builder.makeLocalGet(tee2->index, tee2->type);
        } else {
          rep = ExpressionManipulator::copy(operands[f2], *getModule());
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
        *getp = rep;
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
