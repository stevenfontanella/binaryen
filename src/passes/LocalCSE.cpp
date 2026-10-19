/*
 * Copyright 2017 WebAssembly Community Group participants
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
// Local CSE
//
// This finds common sub-expressions and saves them to a local to avoid
// recomputing them. It runs in each basic block separately, and uses a simple
// algorithm, where we track "requests" to reuse a value. That is, if we see
// an add operation appear twice, and the inputs must be identical in both
// cases, then the second one requests to reuse the computed value from the
// first. The first one to appear is the "original" expression that will remain
// in the code; we will save its value to a local, and get it from that local
// later:
//
//  (i32.add (A) (B))
//  (i32.add (A) (B))
//
//    =>
//
//  (local.tee $temp (i32.add (A) (B)))
//  (local.get $temp)
//
// The algorithm used here is as follows:
//
//  * Scan: Hash each expression and see if it repeats later.
//          If it does:
//            * Note that the original appearance is requested to be reused
//              an additional time.
//            * Link the first expression as the original appearance of the
//              later one.
//            * Scan the children of the repeat and undo their requests to be
//              replaced (as if we will reuse the parent, we don't need to do
//              anything for the children, see below).
//
//  * Check: Check if effects prevent some of the requests from being done. For
//           example, if the value contains a load from memory, we cannot
//           optimize around a store, as the value before the store might be
//           different (see below).
//
//  * Apply: Go through the basic block again, this time adding a tee on an
//           expression whose value we want to use later, and replacing the
//           uses with gets.
//
// For example, here is what the algorithm would do on
//
//  (i32.eqz (A))
//  ..
//  (i32.eqz (A))
//
// Assuming A does not have side effects that interfere, this will happen:
//
//  1. Scan A and add it to the hash map of things we have seen.
//  2. Scan the eqz, and do the same for it.
//  3. Scan the second A. Increment the first A's requests counter, and mark the
//     second A as intended to be replaced by the original A.
//  4. Scan the second eqz, and do similar things for it: increment the requests
//     for the original eqz, and point to the original from the repeat.
//     * Then also scan its children, in this case A, and decrement the first
//       A's reuse counter, and unmark the second A's note that it is intended
//       to be replaced.
//  5. After that, the second eqz requests to be replaced by the first, and
//     there is no request on A.
//  6. Run through the block again, adding a tee and replacing the second eqz,
//     resulting in:
//
//   (local.tee $temp
//     (i32.eqz
//       (A)
//     )
//   )
//   (local.get $temp)
//
// Note how the scanning of children avoids us adding a local for A: when we
// reuse the parent, we don't need to also try to reuse the child.
//
// Effects must be considered carefully by the Checker phase. E.g.:
//
//  x = load(a);
//  store(..)
//  y = load(a);
//
// Even though the load looks identical, the store means we may load a
// different value, so we will invalidate the request to optimize here.
//
// This pass only finds entire expression trees, and not parts of them, so we
// will not optimize this:
//
//  (A (B (C (D1))))
//  (A (B (C (D2))))
//
// The innermost child is different, so the trees are not identical. However,
// running flatten before running this pass would allow those to be optimized as
// well (we would also need to simplify locals somewhat to allow the locals to
// be identified as identical, see pass.cpp).
//
// TODO: Global, inter-block gvn etc. However, note that atm the cost of our
//       adding new locals here is low because their lifetimes are all within a
//       single basic block. A global optimization here could add long-lived
//       locals with register allocation costs in the entire function.
//

#include <algorithm>
#include <memory>

#include <ir/cost.h>
#include <ir/effects.h>
#include <ir/find_all.h>
#include <ir/intrinsics.h>
#include <ir/iteration.h>
#include <ir/linear-execution.h>
#include <ir/manipulation.h>
#include <ir/module-utils.h>
#include <ir/properties.h>
#include <ir/type-updating.h>
#include <ir/utils.h>
#include <pass.h>
#include <wasm-builder.h>
#include <wasm-traversal.h>
#include <wasm.h>

namespace wasm {

namespace {

// An expression with a cached hash value.
struct HashedExpression {
  Expression* expr;
  size_t digest;

  HashedExpression(Expression* expr, size_t digest)
    : expr(expr), digest(digest) {}

  HashedExpression(const HashedExpression& other)
    : expr(other.expr), digest(other.digest) {}
};

struct HEHasher {
  size_t operator()(const HashedExpression hashed) const {
    return hashed.digest;
  }
};

// A full equality check for HashedExpressions. The hash is used as a speedup,
// but if it matches we still verify the contents are identical.
struct HEComparer {
  bool operator()(const HashedExpression a, const HashedExpression b) const {
    if (a.digest != b.digest) {
      return false;
    }
    // Note that we do not consider metadata here. That means we may replace two
    // identical expressions with different metadata, say, different branch
    // hints, but that is ok: we are only removing things from executing (by
    // reusing the first computed value), so this will not cause new invalid
    // branch hints to execute.
    return ExpressionAnalyzer::equal(a.expr, b.expr);
  }
};

// Maps hashed expressions to the list of expressions that match. That is, all
// expressions that are equivalent (same hash, and also compare equal) will
// be in a vector for the corresponding entry in this map.
using HashedExprs = std::unordered_map<HashedExpression,
                                       SmallVector<Expression*, 1>,
                                       HEHasher,
                                       HEComparer>;

// Request information about an expression: whether it requests to be replaced,
// or it is an original expression whose value can be used to replace others
// later.
struct RequestInfo {
  // The number of other expressions that would like to reuse this value.
  Index requests = 0;

  // If this is a repeat value, this points to the original. (And we have
  // incremented the original's |requests|.)
  Expression* original = nullptr;

  void validate() const {
    // An expression cannot both be requested and make requests. If we see A
    // appear three times, both repeats must request from the very first.
    assert(!(requests && original));

    // When we encounter a requestInfo (after its initial creation) then it
    // must either request or be requested - otherwise, it should not exist,
    // as we remove unneeded things from our tracking.
    assert(requests || original);
  }
};

// A map of expressions to their request info.
struct RequestInfoMap : public std::unordered_map<Expression*, RequestInfo> {
  void dump(std::ostream& o) {
    for (auto& [curr, info] : *this) {
      o << *curr << " has " << info.requests << " reqs, orig: " << info.original
        << '\n';
    }
  }
};

// Check if a call is to an idempotent function. Note that we do not check for
// an annotation on the call itself - optimizing that would require us to verify
// idempotency on the later one specifically, but that is complicated in this
// pass (we'd end up tracking the first one unnecessarily, hoping the second is
// idempotent). Instead, we look on the called function, which is identical for
// all callers.
// TODO if users want the call annotation, handle that too.
bool isIdempotent(Expression* curr, Module& wasm) {
  if (auto* call = curr->dynCast<Call>()) {
    if (call->isReturn) {
      return false;
    }
    auto* target = wasm.getFunctionOrNull(call->target);
    if (!target) {
      return false;
    }
    if (Intrinsics::getAnnotations(target).idempotent) {
      return true;
    }
    if (target->effects && target->effects->isIdempotent) {
      return true;
    }
  }
  return false;
}

bool isPureReadOnlyCall(Expression* curr,
                        Module& wasm,
                        const PassOptions& options) {
  if (auto* call = curr->dynCast<Call>()) {
    if (call->isReturn) {
      return false;
    }
    auto* target = wasm.getFunctionOrNull(call->target);
    if (!target || target->imported() || !target->effects) {
      return false;
    }
    const auto& eff = *target->effects;
    if (!options.trapsNeverHappen && eff.trap) {
      return false;
    }
    return !eff.hasNonTrapSideEffects() && !eff.generative && !eff.throws();
  }
  return false;
}

struct Scanner
  : public LinearExecutionWalker<Scanner, UnifiedExpressionVisitor<Scanner>> {
  PassOptions& options;

  // Request info for all expressions ever seen.
  RequestInfoMap& requestInfos;

  Scanner(PassOptions& options, RequestInfoMap& requestInfos)
    : options(options), requestInfos(requestInfos) {}

  // Currently active hashed expressions in the current basic block. If we see
  // an active expression before us that is identical to us, then it becomes our
  // original expression that we request from.
  HashedExprs activeExprs;

  // Stack of information of all active expressions. We store hash values and
  // possibility (as computed by isPossible), which we compute incrementally so
  // as to avoid N^2 work (which could happen if we recomputed children).
  using HashPossibility = std::pair<size_t, bool>;
  SmallVector<HashPossibility, 10> activeIncrementalInfo;

  static void doNoteNonLinear(Scanner* self, Expression** currp) {
    // We are starting a new basic block. Forget all the currently-hashed
    // expressions, as we no longer want to make connections to anything from
    // another block.
    self->activeExprs.clear();
    self->activeIncrementalInfo.clear();
    // Note that we do not clear requestInfos - that is information we will use
    // later in the Applier class. That is, we've cleared all the active
    // information, leaving the things we need later.
  }

  // It is ok to look at adjacent blocks together, as if a later part of a block
  // is not reached that is fine - changes we make there would not be reached in
  // that case.
  bool connectAdjacentBlocks = true;

  void visitExpression(Expression* curr) {
    // Compute the hash, using the pre-computed hashes of the children, which
    // are saved. This allows us to hash everything in linear time.
    //
    // Note that we must compute the hash first, as we need it even for things
    // that are not isRelevant() (if they are the children of a relevant thing).
    auto numChildren = Properties::getNumChildren(curr);
    auto hash = ExpressionAnalyzer::shallowHash(curr);
    auto possible = isPossible(curr);
    for (Index i = 0; i < numChildren; i++) {
      if (activeIncrementalInfo.empty()) {
        // The child was in another block, so this expression cannot be
        // optimized.
        return;
      }
      auto [currHash, currPossible] = activeIncrementalInfo.back();
      activeIncrementalInfo.pop_back();
      hash_combine(hash, currHash);
      if (!currPossible) {
        possible = false;
      }
    }
    activeIncrementalInfo.emplace_back(hash, possible);

    // Check if this is something possible and also relevant for optimization.
    if (!possible || !isRelevant(curr)) {
      return;
    }

    auto& vec = activeExprs[HashedExpression(curr, hash)];
    vec.push_back(curr);
    if (vec.size() > 1) {
      // This is a repeat expression. Add a request for it.
      auto& info = requestInfos[curr];
      auto* original = vec[0];
      info.original = original;

      // Mark the request on the original. Note that this may create the
      // requestInfo for it, if it is the first request (this avoids us creating
      // requests eagerly).
      requestInfos[original].requests++;

      // Remove any requests from the expression's children, as we will replace
      // the entire thing (see explanation earlier). Note that we just need to
      // go over our direct children, as grandchildren etc. have already been
      // processed. That is, if we have
      //
      //  (A (B (C))
      //  (A (B (C))
      //
      // Then when we see the second B we will mark the second C as no longer
      // requesting replacement. And then when we see the second A, all it needs
      // to update is the second B.
      for (auto* child : ChildIterator(curr)) {
        if (!requestInfos.contains(child)) {
          // The child never had a request. While it repeated (since the parent
          // repeats), it was not relevant for the optimization so we never
          // created a requestInfo for it.
          continue;
        }

        // Remove the child.
        auto& childInfo = requestInfos[child];
        auto* childOriginal = childInfo.original;
        requestInfos.erase(child);

        // Update the child's original, potentially erasing it too if no
        // requests remain.
        assert(childOriginal);
        auto& childOriginalRequests = requestInfos[childOriginal].requests;
        assert(childOriginalRequests > 0);
        childOriginalRequests--;
        if (childOriginalRequests == 0) {
          requestInfos.erase(childOriginal);
        }
      }
    }
  }

  // Only some values are relevant to be optimized.
  bool isRelevant(Expression* curr) {
    // * Ignore anything that is not a concrete type, as we are looking for
    //   computed values to reuse, and so none and unreachable are irrelevant.
    // * Ignore local.get and set, as those are the things we optimize to.
    // * Ignore constants so that we don't undo the effects of constant
    //   propagation.
    // * Ignore things we cannot put in a local, as then we can't do this
    //   optimization at all.
    //
    // More things matter here, like having side effects or not, but computing
    // them is not cheap, so leave them for later, after we know if there
    // actually are any requests for reuse of this value (which is rare).
    if (!curr->type.isConcrete() || curr->is<LocalGet>() ||
        curr->is<LocalSet>() || Properties::isConstantExpression(curr)) {
      return false;
    }

    // If the size is at least 3, then if we have two of them we have 6,
    // and so adding one set+one get and removing one of the items itself
    // is not detrimental, and may be beneficial.
    // TODO: investigate size 2
    auto size = Measurer::measure(curr);
    if (options.shrinkLevel > 0 && size >= 3) {
      return true;
    }

    // If we focus on speed, any reduction in cost is beneficial, as the
    // cost of a get is essentially free. However, we need to balance that with
    // the fact that the VM will also do CSE/GVN itself, so minor improvements
    // are not worthwhile, so skip things of size 1 (like a global.get).
    if (options.shrinkLevel == 0 && CostAnalyzer(curr).cost > 0 && size >= 2) {
      return true;
    }

    return false;
  }

  // Some things are not possible, and also prevent their parents from being
  // possible as well. This is different from isRelevant in that relevance is
  // considered for the entire expression, including children - e.g., is the
  // total size big enough - while isPossible checks conditions that prevent
  // using an expression at all.
  bool isPossible(Expression* curr) {
    // A call to an idempotent function is optimizable: it may have effects the
    // first time, but those do not invalidate the second appearance, and also
    // it will return the same value.
    if (isIdempotent(curr, *getModule())) {
      return true;
    }

    // We will fully compute effects later, but consider shallow effects at this
    // early time to ignore things that cannot be optimized later, because we
    // use a greedy algorithm. Specifically, imagine we see this:
    //
    //  (call
    //    (i32.add
    //      ..
    //    )
    //  )
    //
    // If we considered the call relevant then we'd start to look for that
    // larger pattern that contains the add, but then when we find that it
    // cannot be optimized later it is too late for the add. (Instead of
    // checking effects here we could perhaps add backtracking, but that sounds
    // more complex.)
    //
    // We use |hasNonTrapSideEffects| because if a trap occurs the optimization
    // remains valid: both this and the copy of it would trap, which means the
    // first traps and the second isn't reached anyhow.
    //
    // (We don't stash these effects because we may compute many of them here,
    // and only need the few for those patterns that repeat.)
    if (ShallowEffectAnalyzer(options, *getModule(), curr)
          .hasNonTrapSideEffects()) {
      return false;
    }

    // We also cannot optimize away something that is intrinsically
    // nondeterministic: even if it has no side effects, if it may return a
    // different result each time, and then we cannot optimize away repeats.
    if (Properties::isShallowlyGenerative(curr, getFunction(), *getModule())) {
      return false;
    }

    return true;
  }
};

// Check for invalidations due to effects. We do this after scanning as effect
// computation is not cheap, and there are usually not many identical fragments
// of code.
//
// This updates the RequestInfos of things it sees are invalidated, which will
// make Applier ignore them.
struct Checker
  : public LinearExecutionWalker<Checker, UnifiedExpressionVisitor<Checker>> {
  PassOptions& options;
  RequestInfoMap& requestInfos;

  Checker(PassOptions& options, RequestInfoMap& requestInfos)
    : options(options), requestInfos(requestInfos) {}

  struct ActiveOriginalInfo {
    // How many of the requests remain to be seen during our walk. When this
    // reaches 0, we know that the original is no longer requested from later in
    // the block.
    Index requestsLeft;

    // The effects in the expression.
    EffectAnalyzer effects;
  };

  // The currently relevant original expressions, that is, the ones that may be
  // optimized in the current basic block.
  std::unordered_map<Expression*, ActiveOriginalInfo> activeOriginals;

  void visitExpression(Expression* curr) {
    // This is the first time we encounter this expression.
    assert(!activeOriginals.contains(curr));

    // Given the current expression, see what it invalidates of the currently-
    // hashed expressions, if there are any.
    if (!activeOriginals.empty()) {
      // We only need to visit this node itself, as we have already visited its
      // children by the time we get here.
      ShallowEffectAnalyzer effects(options, *getModule(), curr);
      // We can ignore traps here:
      //
      //  (ORIGINAL)
      //  (curr)
      //  (COPY)
      //
      // We are some code in between an original and a copy of it, and we are
      // trying to turn COPY into a local.get of a value that we stash at the
      // original. If |curr| traps then we simply don't reach the copy anyhow.
      //
      // Note, however, that this is really for future use, as atm this does not
      // help: the only effect a trap can interact with is a set of global state
      // (the trap could prevent that writing, which would be noticeable) - but
      // LocalCSE does not deduplicate expressions with such effects, as they
      // must happen twice. That is, removing trapping from (curr) in the
      // example above has no effect as (ORIGINAL) never has global write
      // effects.
      effects.trap = false;

      auto idempotent = isIdempotent(curr, *getModule());

      std::vector<Expression*> invalidated;
      for (auto& kv : activeOriginals) {
        auto* original = kv.first;
        if (idempotent && ExpressionAnalyzer::shallowEqual(curr, original)) {
          // |curr| is idempotent, so it does not invalidate later appearances
          // of itself.
          continue;
        }
        auto& originalInfo = kv.second;
        // Check whether curr must remain before COPY. We use ORIGINAL's effects
        // in the check because we know they are the same as COPY's effects.
        if (effects.orderedBefore(originalInfo.effects)) {
          invalidated.push_back(original);
        }
      }

      for (auto* original : invalidated) {
        // Remove all requests after this expression, as we cannot optimize to
        // them.
        requestInfos[original].requests -=
          activeOriginals.at(original).requestsLeft;

        // If no requests remain at all (that is, there were no requests we
        // could provide before we ran into this invalidation) then we do not
        // need this original at all.
        if (requestInfos[original].requests == 0) {
          requestInfos.erase(original);
        }

        activeOriginals.erase(original);
      }
    }

    auto iter = requestInfos.find(curr);
    if (iter == requestInfos.end()) {
      return;
    }
    auto& info = iter->second;
    info.validate();

    if (info.requests > 0) {
      // This is an original. Compute its side effects, as we cannot optimize
      // away repeated appearances if it has any.
      EffectAnalyzer effects(options, *getModule(), curr);

      // We can ignore traps here, as we replace a repeating expression with a
      // single appearance of it, a store to a local, and gets in the other
      // locations, and so if the expression traps then the first appearance -
      // that we keep around - would trap, and the others are never reached
      // anyhow. (The other checks we perform here, including invalidation and
      // determinism, will ensure that either all of the appearances trap, or
      // none of them.)
      effects.trap = false;

      // Note that we've already checked above that this has no side effects or
      // generativity: if we got here, then it is good to go from the
      // perspective of this expression itself (but may be invalidated by other
      // code in between, see above).
      activeOriginals.emplace(
        curr, ActiveOriginalInfo{info.requests, std::move(effects)});
    } else if (info.original) {
      // The original may have already been invalidated. If so, remove our info
      // as well.
      auto originalIter = activeOriginals.find(info.original);
      if (originalIter == activeOriginals.end()) {
        requestInfos.erase(iter);
        return;
      }

      // After visiting this expression, we have one less request for its
      // original, and perhaps none are left.
      auto& originalInfo = originalIter->second;
      if (originalInfo.requestsLeft == 1) {
        activeOriginals.erase(info.original);
      } else {
        originalInfo.requestsLeft--;
      }
    }
  }

  static void doNoteNonLinear(Checker* self, Expression** currp) {
    // Between basic blocks there can be no active originals.
    assert(self->activeOriginals.empty());
  }

  // See the same code above.
  bool connectAdjacentBlocks = true;

  void visitFunction(Function* curr) {
    // At the end of the function there can be no active originals.
    assert(activeOriginals.empty());
  }
};

// Applies the optimization now that we know which requests are valid.
struct Applier
  : public LinearExecutionWalker<Applier, UnifiedExpressionVisitor<Applier>> {
  RequestInfoMap requestInfos;

  Applier(RequestInfoMap& requestInfos) : requestInfos(requestInfos) {}

  // Maps the original expressions that we save to locals to the local indexes
  // for them.
  std::unordered_map<Expression*, Index> originalLocalMap;

  void visitExpression(Expression* curr) {
    auto iter = requestInfos.find(curr);
    if (iter == requestInfos.end()) {
      return;
    }

    const auto& info = iter->second;
    info.validate();

    if (info.requests) {
      // We have requests for this value. Add a local and tee the value to
      // there.
      Index local = originalLocalMap[curr] =
        Builder::addVar(getFunction(), curr->type);
      replaceCurrent(
        Builder(*getModule()).makeLocalTee(local, curr, curr->type));
    } else if (info.original) {
      auto& originalInfo = requestInfos.at(info.original);
      if (originalInfo.requests) {
        // This is a valid request of an original value. Get the value from the
        // local.
        assert(originalLocalMap.contains(info.original));
        replaceCurrent(
          Builder(*getModule())
            .makeLocalGet(originalLocalMap[info.original], curr->type));
        originalInfo.requests--;
      }
    }
  }

  static void doNoteNonLinear(Applier* self, Expression** currp) {
    // Clear the state between blocks.
    self->originalLocalMap.clear();
  }

  // See the same code above.
  bool connectAdjacentBlocks = true;
};

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

static bool exprDoesNotReturn(Expression* e, Module& wasm) {
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
      if (exprDoesNotReturn(elem, wasm)) {
        return true;
      }
    }
    return false;
  }
  if (auto* iff = e->dynCast<If>()) {
    if (exprDoesNotReturn(iff->condition, wasm)) {
      return true;
    }
    return iff->ifFalse && exprDoesNotReturn(iff->ifTrue, wasm) &&
           exprDoesNotReturn(iff->ifFalse, wasm);
  }
  if (auto* d = e->dynCast<Drop>()) {
    return exprDoesNotReturn(d->value, wasm);
  }
  if (auto* ls = e->dynCast<LocalSet>()) {
    return exprDoesNotReturn(ls->value, wasm);
  }
  if (auto* call = e->dynCast<Call>()) {
    if (auto* target = wasm.getFunctionOrNull(call->target)) {
      if (target->effects && target->effects->mustNotReturn) {
        return true;
      }
    }
    for (auto* op : call->operands) {
      if (exprDoesNotReturn(op, wasm)) {
        return true;
      }
    }
    return false;
  }
  return false;
}

static bool forwardStructNewFields(Function* func,
                                   Module& wasm,
                                   PassOptions& options) {
  if (!wasm.features.hasGC() || !func->body) {
    return false;
  }

  Index numLocals = func->getNumLocals();
  Index numParams = func->getNumParams();
  std::vector<Index> setCounts(numLocals, 0);
  SmallVector<std::pair<HeapType, Index>, 8> writtenStructFields;
  bool hasUnknownStructWrite = false;
  bool hasCandidateStructNew = false;

  auto unwrapRefAs = [](Expression* e) -> Expression* {
    while (auto* ra = e->dynCast<RefAs>()) {
      if (ra->op == RefAsNonNull) {
        e = ra->value;
      } else {
        break;
      }
    }
    return e;
  };

  struct PreScanner
    : public PostWalker<PreScanner, UnifiedExpressionVisitor<PreScanner>> {
    Module& wasm;
    PassOptions& options;
    Index numParams;
    std::vector<Index>& setCounts;
    SmallVector<std::pair<HeapType, Index>, 8>& writtenStructFields;
    bool& hasUnknownStructWrite;
    bool& hasCandidateStructNew;

    PreScanner(Module& wasm,
               PassOptions& options,
               Index numParams,
               std::vector<Index>& setCounts,
               SmallVector<std::pair<HeapType, Index>, 8>& writtenStructFields,
               bool& hasUnknownStructWrite,
               bool& hasCandidateStructNew)
      : wasm(wasm), options(options), numParams(numParams),
        setCounts(setCounts), writtenStructFields(writtenStructFields),
        hasUnknownStructWrite(hasUnknownStructWrite),
        hasCandidateStructNew(hasCandidateStructNew) {}

    void visitExpression(Expression* curr) {
      if (auto* ls = curr->dynCast<LocalSet>()) {
        setCounts[ls->index]++;
        if (ls->index >= numParams) {
          Expression* v = ls->value;
          while (auto* ra = v->dynCast<RefAs>()) {
            if (ra->op == RefAsNonNull) {
              v = ra->value;
            } else {
              break;
            }
          }
          if (auto* lt = v->dynCast<LocalSet>()) {
            if (lt->isTee()) {
              v = lt->value;
            }
          }
          if (auto* sn = v->dynCast<StructNew>()) {
            if (!sn->isWithDefault() && sn->type.isRef() &&
                sn->type != Type::unreachable) {
              hasCandidateStructNew = true;
            }
          }
        }
        return;
      }
      if (!hasUnknownStructWrite) {
        ShallowEffectAnalyzer eff(options, wasm, curr);
        if (eff.writesSharedStruct || eff.calls ||
            eff.hasUnknownMutableStructWrite) {
          hasUnknownStructWrite = true;
        } else if (eff.writesStruct) {
          for (const auto& key : eff.mutableStructFieldsWritten) {
            if (std::find(writtenStructFields.begin(),
                          writtenStructFields.end(),
                          key) == writtenStructFields.end()) {
              writtenStructFields.push_back(key);
            }
          }
        }
      }
    }
  } preScanner(wasm,
               options,
               numParams,
               setCounts,
               writtenStructFields,
               hasUnknownStructWrite,
               hasCandidateStructNew);
  preScanner.walk(func->body);

  if (!hasCandidateStructNew) {
    return false;
  }

  auto isFieldUnmodified =
    [&](HeapType ht, Index fieldIdx, const Field& field) -> bool {
    if (field.mutable_ == Immutable) {
      return true;
    }
    if (hasUnknownStructWrite) {
      return false;
    }
    for (const auto& [wht, wIdx] : writtenStructFields) {
      if (wIdx == fieldIdx &&
          (HeapType::isSubType(wht, ht) || HeapType::isSubType(ht, wht))) {
        return false;
      }
    }
    return true;
  };

  std::unordered_map<Index, StructNew*> activeStructNews;
  std::unordered_set<Index> activeSingleSets;
  bool changed = false;
  Builder builder(wasm);

  std::function<void(Expression**)> walkDom = [&](Expression** currp) {
    if (!currp || !*currp) {
      return;
    }
    Expression* curr = *currp;
    if (auto* block = curr->dynCast<Block>()) {
      Index firstBranchIdx = block->list.size();
      if (block->name.is()) {
        for (Index i = 0; i < block->list.size(); i++) {
          if (SubtreeTargetsName(block->list[i], block->name)) {
            firstBranchIdx = i;
            break;
          }
        }
      }
      std::unordered_map<Index, StructNew*> savedSN;
      std::unordered_set<Index> savedSS;
      for (Index i = 0; i < block->list.size(); i++) {
        if (i == firstBranchIdx) {
          savedSN = activeStructNews;
          savedSS = activeSingleSets;
        }
        walkDom(&block->list[i]);
      }
      if (firstBranchIdx < block->list.size()) {
        activeStructNews = std::move(savedSN);
        activeSingleSets = std::move(savedSS);
      }
      return;
    }
    if (auto* iff = curr->dynCast<If>()) {
      walkDom(&iff->condition);
      auto savedSN = activeStructNews;
      auto savedSS = activeSingleSets;
      walkDom(&iff->ifTrue);
      auto afterTrueSN = std::move(activeStructNews);
      auto afterTrueSS = std::move(activeSingleSets);
      auto afterFalseSN = savedSN;
      auto afterFalseSS = savedSS;
      if (iff->ifFalse) {
        activeStructNews = savedSN;
        activeSingleSets = savedSS;
        walkDom(&iff->ifFalse);
        afterFalseSN = std::move(activeStructNews);
        afterFalseSS = std::move(activeSingleSets);
      }
      bool thenNR = exprDoesNotReturn(iff->ifTrue, wasm);
      bool elseNR = iff->ifFalse && exprDoesNotReturn(iff->ifFalse, wasm);
      if (thenNR && !elseNR) {
        activeStructNews = std::move(afterFalseSN);
        activeSingleSets = std::move(afterFalseSS);
      } else if (elseNR && !thenNR) {
        activeStructNews = std::move(afterTrueSN);
        activeSingleSets = std::move(afterTrueSS);
      } else {
        activeStructNews = std::move(savedSN);
        activeSingleSets = std::move(savedSS);
      }
      return;
    }
    if (auto* loop = curr->dynCast<Loop>()) {
      auto savedSN = activeStructNews;
      auto savedSS = activeSingleSets;
      walkDom(&loop->body);
      activeStructNews = std::move(savedSN);
      activeSingleSets = std::move(savedSS);
      return;
    }
    if (auto* tryy = curr->dynCast<Try>()) {
      auto savedSN = activeStructNews;
      auto savedSS = activeSingleSets;
      walkDom(&tryy->body);
      for (auto*& cb : tryy->catchBodies) {
        activeStructNews = savedSN;
        activeSingleSets = savedSS;
        walkDom(&cb);
      }
      activeStructNews = std::move(savedSN);
      activeSingleSets = std::move(savedSS);
      return;
    }
    if (auto* tt = curr->dynCast<TryTable>()) {
      auto savedSN = activeStructNews;
      auto savedSS = activeSingleSets;
      walkDom(&tt->body);
      activeStructNews = std::move(savedSN);
      activeSingleSets = std::move(savedSS);
      return;
    }

    for (auto*& child : ChildIterator(curr)) {
      walkDom(&child);
    }

    if (auto* br = curr->dynCast<Break>()) {
      if (!br->condition) {
        activeStructNews.clear();
        activeSingleSets.clear();
      }
      return;
    }
    if (curr->is<Switch>() || curr->is<Return>() || curr->is<Throw>() ||
        curr->is<Rethrow>() || curr->is<Unreachable>()) {
      activeStructNews.clear();
      activeSingleSets.clear();
      return;
    }

    if (auto* ls = curr->dynCast<LocalSet>()) {
      if (setCounts[ls->index] == 1) {
        activeSingleSets.insert(ls->index);
        if (ls->index >= numParams) {
          Expression* v = unwrapRefAs(ls->value);
          if (auto* lt = v->dynCast<LocalSet>()) {
            if (lt->isTee() && setCounts[lt->index] == 1) {
              v = unwrapRefAs(lt->value);
            }
          }
          if (auto* sn = v->dynCast<StructNew>()) {
            if (!sn->isWithDefault() && sn->type.isRef() &&
                sn->type != Type::unreachable) {
              activeStructNews[ls->index] = sn;
            }
          }
        }
      }
      return;
    }

    if (auto* sg = curr->dynCast<StructGet>()) {
      if (sg->order != MemoryOrder::Unordered) {
        return;
      }
      auto* lg = unwrapRefAs(sg->ref)->dynCast<LocalGet>();
      if (!lg) {
        return;
      }
      auto it = activeStructNews.find(lg->index);
      if (it == activeStructNews.end()) {
        return;
      }
      StructNew* sn = it->second;
      HeapType ht = sn->type.getHeapType();
      if (!sg->ref->type.isRef() || sg->ref->type.getHeapType() != ht ||
          sg->index >= sn->operands.size()) {
        return;
      }
      const Field& field = ht.getStruct().fields[sg->index];
      if (!isFieldUnmodified(ht, sg->index, field)) {
        return;
      }
      Expression* op = sn->operands[sg->index];
      if (auto* c = op->dynCast<Const>()) {
        if (field.isPacked()) {
          int32_t raw = c->value.geti32();
          if (field.packedType == Field::i8) {
            raw = sg->signed_ ? int32_t(int8_t(uint8_t(raw)))
                              : int32_t(uint8_t(raw));
          } else {
            raw = sg->signed_ ? int32_t(int16_t(uint16_t(raw)))
                              : int32_t(uint16_t(raw));
          }
          *currp = builder.makeConst(Literal(raw));
        } else {
          *currp = ExpressionManipulator::copy(c, wasm);
        }
        changed = true;
        return;
      }
      if (field.isPacked()) {
        return;
      }
      if ((op->is<RefNull>() || op->is<RefFunc>()) &&
          Type::isSubType(op->type, sg->type)) {
        *currp = ExpressionManipulator::copy(op, wasm);
        changed = true;
        return;
      }
      if (auto* gg = op->dynCast<GlobalGet>()) {
        if (!wasm.getGlobal(gg->name)->mutable_ &&
            Type::isSubType(gg->type, sg->type)) {
          *currp = ExpressionManipulator::copy(gg, wasm);
          changed = true;
          return;
        }
      }
      auto isStableLocal = [&](Index yIdx) -> bool {
        return setCounts[yIdx] == 0 ||
               (setCounts[yIdx] == 1 && activeSingleSets.contains(yIdx));
      };
      if (auto* flg = op->dynCast<LocalGet>()) {
        Type yType = func->getLocalType(flg->index);
        if (isStableLocal(flg->index) && Type::isSubType(yType, sg->type)) {
          *currp = builder.makeLocalGet(flg->index, yType);
          changed = true;
          return;
        }
      }
      if (auto* flt = op->dynCast<LocalSet>()) {
        if (flt->isTee()) {
          Type yType = func->getLocalType(flt->index);
          if (isStableLocal(flt->index) && Type::isSubType(yType, sg->type)) {
            *currp = builder.makeLocalGet(flt->index, yType);
            changed = true;
            return;
          }
        }
      }
      if (auto* ra = op->dynCast<RefAs>()) {
        if (ra->op == RefAsNonNull) {
          Index yIdx = Index(-1);
          if (auto* flg = ra->value->dynCast<LocalGet>()) {
            yIdx = flg->index;
          } else if (auto* flt = ra->value->dynCast<LocalSet>()) {
            if (flt->isTee()) {
              yIdx = flt->index;
            }
          }
          if (yIdx != Index(-1) && isStableLocal(yIdx)) {
            Type yType = func->getLocalType(yIdx);
            if (yType.isRef()) {
              Type nonNullY = Type(yType.getHeapType(), NonNullable);
              if (Type::isSubType(nonNullY, sg->type)) {
                *currp = builder.makeRefAs(RefAsNonNull,
                                           builder.makeLocalGet(yIdx, yType));
                changed = true;
                return;
              }
            }
          }
        }
      }
    }
  };

  walkDom(&func->body);
  return changed;
}

static bool unaliasSingleFieldProjectionLocals(Function* func,
                                               Module& wasm,
                                               PassOptions& options) {
  if (!wasm.features.hasGC() || !func->body) {
    return false;
  }

  Index numParams = func->getNumParams();
  bool anyChanged = false;
  Builder builder(wasm);

  for (int iter = 0; iter < 4; iter++) {
    Index numLocals = func->getNumLocals();
    std::vector<std::vector<Expression**>> sets(numLocals);
    std::vector<std::vector<Expression**>> gets(numLocals);
    std::vector<Index> dominatedGets(numLocals, 0);
    std::unordered_map<Expression*, Expression**> parentStructGets;
    SmallVector<std::pair<HeapType, Index>, 8> writtenStructFields;
    bool hasUnknownStructWrite = false;
    std::unordered_set<Index> activeSingleSets;

    std::function<void(Expression**)> scanDom = [&](Expression** currp) {
      if (!currp || !*currp) {
        return;
      }
      Expression* curr = *currp;
      if (auto* block = curr->dynCast<Block>()) {
        Index firstBranchIdx = block->list.size();
        if (block->name.is()) {
          for (Index i = 0; i < block->list.size(); i++) {
            if (SubtreeTargetsName(block->list[i], block->name)) {
              firstBranchIdx = i;
              break;
            }
          }
        }
        std::unordered_set<Index> savedSS;
        for (Index i = 0; i < block->list.size(); i++) {
          if (i == firstBranchIdx) {
            savedSS = activeSingleSets;
          }
          scanDom(&block->list[i]);
        }
        if (firstBranchIdx < block->list.size()) {
          activeSingleSets = std::move(savedSS);
        }
        return;
      }
      if (auto* iff = curr->dynCast<If>()) {
        scanDom(&iff->condition);
        auto savedSS = activeSingleSets;
        scanDom(&iff->ifTrue);
        auto afterTrueSS = std::move(activeSingleSets);
        auto afterFalseSS = savedSS;
        if (iff->ifFalse) {
          activeSingleSets = savedSS;
          scanDom(&iff->ifFalse);
          afterFalseSS = std::move(activeSingleSets);
        }
        bool thenNR = exprDoesNotReturn(iff->ifTrue, wasm);
        bool elseNR = iff->ifFalse && exprDoesNotReturn(iff->ifFalse, wasm);
        if (thenNR && !elseNR) {
          activeSingleSets = std::move(afterFalseSS);
        } else if (elseNR && !thenNR) {
          activeSingleSets = std::move(afterTrueSS);
        } else {
          activeSingleSets = std::move(savedSS);
        }
        return;
      }
      if (auto* loop = curr->dynCast<Loop>()) {
        auto savedSS = activeSingleSets;
        scanDom(&loop->body);
        activeSingleSets = std::move(savedSS);
        return;
      }
      if (auto* tryy = curr->dynCast<Try>()) {
        auto savedSS = activeSingleSets;
        scanDom(&tryy->body);
        for (auto*& cb : tryy->catchBodies) {
          activeSingleSets = savedSS;
          scanDom(&cb);
        }
        activeSingleSets = std::move(savedSS);
        return;
      }
      if (auto* tt = curr->dynCast<TryTable>()) {
        auto savedSS = activeSingleSets;
        scanDom(&tt->body);
        activeSingleSets = std::move(savedSS);
        return;
      }

      if (auto* sg = curr->dynCast<StructGet>()) {
        if (sg->order == MemoryOrder::Unordered &&
            (sg->ref->is<LocalGet>() ||
             (sg->ref->is<LocalSet>() && sg->ref->cast<LocalSet>()->isTee()))) {
          parentStructGets[sg->ref] = currp;
        }
      }

      for (auto*& child : ChildIterator(curr)) {
        scanDom(&child);
      }

      if (auto* br = curr->dynCast<Break>()) {
        if (!br->condition) {
          activeSingleSets.clear();
        }
        return;
      }
      if (curr->is<Switch>() || curr->is<Return>() || curr->is<Throw>() ||
          curr->is<Rethrow>() || curr->is<Unreachable>()) {
        activeSingleSets.clear();
        return;
      }

      if (auto* lg = curr->dynCast<LocalGet>()) {
        gets[lg->index].push_back(currp);
        if (activeSingleSets.contains(lg->index)) {
          dominatedGets[lg->index]++;
        }
        return;
      }
      if (auto* ls = curr->dynCast<LocalSet>()) {
        sets[ls->index].push_back(currp);
        if (sets[ls->index].size() == 1) {
          activeSingleSets.insert(ls->index);
        } else {
          activeSingleSets.erase(ls->index);
        }
        return;
      }
      if (!hasUnknownStructWrite) {
        ShallowEffectAnalyzer eff(options, wasm, curr);
        if (eff.writesSharedStruct || eff.calls ||
            eff.hasUnknownMutableStructWrite) {
          hasUnknownStructWrite = true;
        } else if (eff.writesStruct) {
          for (const auto& key : eff.mutableStructFieldsWritten) {
            if (std::find(writtenStructFields.begin(),
                          writtenStructFields.end(),
                          key) == writtenStructFields.end()) {
              writtenStructFields.push_back(key);
            }
          }
        }
      }
    };

    scanDom(&func->body);

    auto isFieldUnmodified =
      [&](HeapType ht, Index fieldIdx, const Field& field) -> bool {
      if (field.mutable_ == Immutable) {
        return true;
      }
      if (hasUnknownStructWrite) {
        return false;
      }
      for (const auto& [wht, wIdx] : writtenStructFields) {
        if (wIdx == fieldIdx &&
            (HeapType::isSubType(wht, ht) || HeapType::isSubType(ht, wht))) {
          return false;
        }
      }
      return true;
    };

    bool iterChanged = false;
    std::unordered_set<Expression*> touchedNodes;

    for (Index x = numParams; x < numLocals; x++) {
      if (sets[x].size() != 1 || gets[x].empty() ||
          dominatedGets[x] != gets[x].size()) {
        continue;
      }
      Expression** setPtr = sets[x][0];
      auto* ls = (*setPtr)->cast<LocalSet>();
      Index totalProjections = gets[x].size() + (ls->isTee() ? 1 : 0);
      if (totalProjections < 2) {
        continue;
      }
      if (!options.trapsNeverHappen && !ls->isTee() &&
          ls->value->type.isNullable()) {
        continue;
      }
      if (touchedNodes.contains(ls) || touchedNodes.contains(ls->value)) {
        continue;
      }
      StructGet* protoSG = nullptr;
      bool allMatch = true;
      auto checkUse = [&](Expression* useExpr) -> bool {
        auto it = parentStructGets.find(useExpr);
        if (it == parentStructGets.end()) {
          return false;
        }
        auto* sg = (*it->second)->cast<StructGet>();
        if (touchedNodes.contains(sg) || !sg->ref->type.isRef()) {
          return false;
        }
        if (!protoSG) {
          protoSG = sg;
          return true;
        }
        return sg->ref->type.getHeapType() ==
                 protoSG->ref->type.getHeapType() &&
               sg->index == protoSG->index && sg->signed_ == protoSG->signed_ &&
               sg->type == protoSG->type;
      };
      if (ls->isTee() && !checkUse(ls)) {
        continue;
      }
      for (Expression** gp : gets[x]) {
        if (!checkUse(*gp)) {
          allMatch = false;
          break;
        }
      }
      if (!allMatch || !protoSG) {
        continue;
      }
      HeapType ht = protoSG->ref->type.getHeapType();
      if (!ht.isStruct() || protoSG->index >= ht.getStruct().fields.size()) {
        continue;
      }
      const Field& field = ht.getStruct().fields[protoSG->index];
      if (!isFieldUnmodified(ht, protoSG->index, field)) {
        continue;
      }
      Type newType = protoSG->type;
      if (!newType.isConcrete()) {
        continue;
      }

      // Rewrite local x to store the projected field directly.
      func->vars[x - numParams] = newType;
      touchedNodes.insert(ls);
      touchedNodes.insert(ls->value);
      ls->value = builder.makeStructGet(
        protoSG->index, ls->value, MemoryOrder::Unordered, newType, protoSG->signed_);
      if (ls->isTee()) {
        Expression** teeParentPtr = parentStructGets.at(ls);
        touchedNodes.insert(*teeParentPtr);
        ls->type = newType;
        ls->finalize();
        *teeParentPtr = ls;
      }
      for (Expression** gp : gets[x]) {
        Expression* lg = *gp;
        Expression** sgPtr = parentStructGets.at(lg);
        touchedNodes.insert(lg);
        touchedNodes.insert(*sgPtr);
        *sgPtr = builder.makeLocalGet(x, newType);
      }
      iterChanged = true;
      anyChanged = true;
    }

    if (!iterChanged) {
      break;
    }
  }

  return anyChanged;
}

struct ScopedDominatorCSE {
  Function* func;
  Module& wasm;
  PassOptions& options;

  ScopedDominatorCSE(Function* func, Module& wasm, PassOptions& options)
    : func(func), wasm(wasm), options(options) {}

  struct OrigRecord {
    Expression* orig;
    size_t hash;
    Index estBytes;
    Index existingLocal = Index(-1);
    bool existingLocalNeedsNonNull = false;
    Literal knownConst;
    int8_t knownNullness = 0;
    bool knownNonZero = false;
    bool isFactOnly = false;
    BinaryOp altBinaryOp = InvalidBinary;
    Index reuses = 0;
    SmallVector<Index, 4> localsRead;
    SmallVector<Name, 2> mutableGlobalsRead;
    SmallVector<std::pair<HeapType, Index>, 2> mutableStructFieldsRead;
    bool readsMutableStruct = false;
    bool readsUnknownMutableStruct = false;
    bool readsMutableArray = false;
    bool readsMemory = false;
    bool readsTable = false;
    bool exitedArmScope = false;
    bool crossedScope = false;
  };

  struct NodeInfo {
    size_t hash = 0;
    Index estBytes = 0;
    bool possible = false;
    bool hasTee = false;
  };

  std::vector<OrigRecord> origRecords;
  std::unordered_map<Expression*, Index> exprToOrigIdx;
  std::unordered_map<Expression*, Index> repeatToOrigIdx;
  std::unordered_map<Expression*, SmallVector<std::pair<Expression*, Index>, 2>>
    unmatchedChildren;

  // Maps normalized hash -> active original indices in the current scope.
  using ActiveMap = std::unordered_map<size_t, SmallVector<Index, 1>>;
  ActiveMap activeMap;
  Expression* lastCondExpr = nullptr;
  ActiveMap lastCondTrueMap;
  ActiveMap lastCondFalseMap;
  bool hasLastBrExitMap = false;
  ActiveMap lastBrExitMap;
  ActiveMap lastBrTakenMap;
  std::unordered_map<Expression*, NodeInfo> exprInfoMap;
  std::unordered_set<Type> activeVarTypes;

  static bool isConstZeroI32(Expression* e) {
    while (e) {
      if (auto* c = e->dynCast<Const>()) {
        return c->type == Type::i32 && c->value.geti32() == 0;
      }
      if (auto* b = e->dynCast<Block>()) {
        if (!b->list.empty() && !SubtreeTargetsName(b, b->name)) {
          e = b->list.back();
          continue;
        }
      }
      break;
    }
    return false;
  }

  static bool isConstNonZeroI32(Expression* e) {
    while (e) {
      if (auto* c = e->dynCast<Const>()) {
        return c->type == Type::i32 && c->value.geti32() != 0;
      }
      if (auto* b = e->dynCast<Block>()) {
        if (!b->list.empty() && !SubtreeTargetsName(b, b->name)) {
          e = b->list.back();
          continue;
        }
      }
      break;
    }
    return false;
  }

  bool doesNotReturn(Expression* e) const {
    return exprDoesNotReturn(e, wasm);
  }

  bool isBooleanValueExpr(Expression* e) const {
    while (e) {
      if (e->is<RefIsNull>() || e->is<RefEq>() || e->is<RefTest>() ||
          e->is<StringTest>()) {
        return true;
      }
      if (auto* seq = e->dynCast<StringEq>()) {
        return seq->op == StringEqEqual;
      }
      if (auto* u = e->dynCast<Unary>()) {
        return u->op == EqZInt32 || u->op == EqZInt64;
      }
      if (auto* bin = e->dynCast<Binary>()) {
        return bin->isRelational();
      }
      if (auto* c = e->dynCast<Const>()) {
        return c->type == Type::i32 &&
               (c->value.geti32() == 0 || c->value.geti32() == 1);
      }
      if (auto* iff = e->dynCast<If>()) {
        if (!iff->ifFalse) {
          return false;
        }
        bool tOk = doesNotReturn(iff->ifTrue) || isBooleanValueExpr(iff->ifTrue);
        bool fOk =
          doesNotReturn(iff->ifFalse) || isBooleanValueExpr(iff->ifFalse);
        return tOk && fOk;
      }
      if (auto* sel = e->dynCast<Select>()) {
        return isBooleanValueExpr(sel->ifTrue) &&
               isBooleanValueExpr(sel->ifFalse);
      }
      if (auto* b = e->dynCast<Block>()) {
        if (!b->list.empty() && !SubtreeTargetsName(b, b->name)) {
          e = b->list.back();
          continue;
        }
        return false;
      }
      if (auto* ls = e->dynCast<LocalSet>()) {
        if (ls->isTee()) {
          e = ls->value;
          continue;
        }
        return false;
      }
      break;
    }
    return false;
  }

  static BinaryOp getComplementaryIntRelOp(BinaryOp op) {
    switch (op) {
      case EqInt32:
        return NeInt32;
      case NeInt32:
        return EqInt32;
      case LtSInt32:
        return GeSInt32;
      case GeSInt32:
        return LtSInt32;
      case LtUInt32:
        return GeUInt32;
      case GeUInt32:
        return LtUInt32;
      case GtSInt32:
        return LeSInt32;
      case LeSInt32:
        return GtSInt32;
      case GtUInt32:
        return LeUInt32;
      case LeUInt32:
        return GtUInt32;
      case EqInt64:
        return NeInt64;
      case NeInt64:
        return EqInt64;
      case LtSInt64:
        return GeSInt64;
      case GeSInt64:
        return LtSInt64;
      case LtUInt64:
        return GeUInt64;
      case GeUInt64:
        return LtUInt64;
      case GtSInt64:
        return LeSInt64;
      case LeSInt64:
        return GtSInt64;
      case GtUInt64:
        return LeUInt64;
      case LeUInt64:
        return GtUInt64;
      default:
        return InvalidBinary;
    }
  }

  static bool evalIntRelOp(BinaryOp op, Literal l, Literal r, Literal& out) {
    if (l.type != r.type) {
      return false;
    }
    if (l.type == Type::i32) {
      int32_t a = l.geti32();
      int32_t b = r.geti32();
      uint32_t ua = uint32_t(a);
      uint32_t ub = uint32_t(b);
      int32_t res = 0;
      switch (op) {
        case EqInt32:
          res = (a == b);
          break;
        case NeInt32:
          res = (a != b);
          break;
        case LtSInt32:
          res = (a < b);
          break;
        case LeSInt32:
          res = (a <= b);
          break;
        case GtSInt32:
          res = (a > b);
          break;
        case GeSInt32:
          res = (a >= b);
          break;
        case LtUInt32:
          res = (ua < ub);
          break;
        case LeUInt32:
          res = (ua <= ub);
          break;
        case GtUInt32:
          res = (ua > ub);
          break;
        case GeUInt32:
          res = (ua >= ub);
          break;
        default:
          return false;
      }
      out = Literal(res);
      return true;
    }
    if (l.type == Type::i64) {
      int64_t a = l.geti64();
      int64_t b = r.geti64();
      uint64_t ua = uint64_t(a);
      uint64_t ub = uint64_t(b);
      int32_t res = 0;
      switch (op) {
        case EqInt64:
          res = (a == b);
          break;
        case NeInt64:
          res = (a != b);
          break;
        case LtSInt64:
          res = (a < b);
          break;
        case LeSInt64:
          res = (a <= b);
          break;
        case GtSInt64:
          res = (a > b);
          break;
        case GeSInt64:
          res = (a >= b);
          break;
        case LtUInt64:
          res = (ua < ub);
          break;
        case LeUInt64:
          res = (ua <= ub);
          break;
        case GtUInt64:
          res = (ua > ub);
          break;
        case GeUInt64:
          res = (ua >= ub);
          break;
        default:
          return false;
      }
      out = Literal(res);
      return true;
    }
    return false;
  }

  static Index getConstEstBytes(Literal val) {
    if (val.type == Type::i32) {
      int32_t v = val.geti32();
      return (-64 <= v && v <= 63) ? 2 : ((-8192 <= v && v <= 8191) ? 3 : 5);
    }
    if (val.type == Type::i64) {
      int64_t v = val.geti64();
      return (-64 <= v && v <= 63) ? 2 : ((-8192 <= v && v <= 8191) ? 3 : 6);
    }
    return val.type == Type::f64 ? 9 : 5;
  }

  bool collectPureCallStructFieldsRead(
    Function* target,
    SmallVector<std::pair<HeapType, Index>, 2>& fieldsOut) const {
    if (!target || target->imported() || !target->effects ||
        target->effects->readsSharedMutableStruct ||
        target->effects->hasUnknownMutableStructRead ||
        target->effects->calls) {
      return false;
    }
    for (const auto& key : target->effects->mutableStructFieldsRead) {
      if (std::find(fieldsOut.begin(), fieldsOut.end(), key) ==
          fieldsOut.end()) {
        fieldsOut.push_back(key);
      }
    }
    return true;
  }

  static Type getBinaryVarType(Type t) {
    if (t.isRef()) {
      return Type(t.getHeapType(), Nullable, Inexact);
    }
    return t;
  }

  static bool getNormLocalIndex(Expression* e, Index& idxOut) {
    if (auto* lg = e->dynCast<LocalGet>()) {
      idxOut = lg->index;
      return true;
    }
    if (auto* ls = e->dynCast<LocalSet>()) {
      if (ls->isTee()) {
        idxOut = ls->index;
        return true;
      }
    }
    return false;
  }

  static bool normEqual(Expression* a, Expression* b) {
    if (a->type != b->type) {
      return false;
    }
    return ExpressionAnalyzer::flexibleEqual(
      a, b, [](Expression* left, Expression* right) {
        Index lIdx, rIdx;
        if (getNormLocalIndex(left, lIdx) && getNormLocalIndex(right, rIdx)) {
          return lIdx == rIdx;
        }
        return false;
      });
  }

  Index getShallowEstBytes(Expression* curr) const {
    if (curr->is<LocalGet>() || curr->is<LocalSet>()) {
      return 2;
    }
    if (curr->is<GlobalGet>() || curr->is<GlobalSet>()) {
      return 3;
    }
    if (auto* c = curr->dynCast<Const>()) {
      if (c->type == Type::i32) {
        int32_t v = c->value.geti32();
        return (-64 <= v && v <= 63) ? 2 : ((-8192 <= v && v <= 8191) ? 3 : 5);
      }
      if (c->type == Type::i64) {
        int64_t v = c->value.geti64();
        return (-64 <= v && v <= 63) ? 2 : ((-8192 <= v && v <= 8191) ? 3 : 6);
      }
      return c->type == Type::f64 ? 9 : 5;
    }
    if (curr->is<RefNull>()) {
      return 2;
    }
    if (curr->is<RefFunc>()) {
      return 4;
    }
    if (auto* sg = curr->dynCast<StructGet>()) {
      return 5 + (sg->index >= 128 ? 1 : 0);
    }
    if (curr->is<RefCast>() || curr->is<RefTest>()) {
      return 4;
    }
    if (curr->is<ArrayGet>() || curr->is<Load>()) {
      return 3;
    }
    if (auto* call = curr->dynCast<Call>()) {
      if (auto* f = wasm.getFunctionOrNull(call->target)) {
        if (f->imported()) {
          return 2;
        }
      }
      return 3;
    }
    if (curr->is<ArrayLen>() || curr->is<RefI31>() || curr->is<I31Get>()) {
      return 2;
    }
    if (curr->is<RefAs>() || curr->is<RefIsNull>() || curr->is<RefEq>() ||
        curr->is<Unary>() || curr->is<Binary>() || curr->is<Select>()) {
      return 1;
    }
    return 2;
  }

  void populateNormReadEffects(Expression* root, OrigRecord& rec) {
    std::function<void(Expression*)> walkNorm = [&](Expression* e) {
      Index lIdx;
      if (e != root && getNormLocalIndex(e, lIdx)) {
        if (std::find(rec.localsRead.begin(), rec.localsRead.end(), lIdx) ==
            rec.localsRead.end()) {
          rec.localsRead.push_back(lIdx);
        }
        return;
      }
      if (!isIdempotent(e, wasm)) {
        ShallowEffectAnalyzer eff(options, wasm, e);
        for (Index l : eff.localsRead) {
          if (std::find(rec.localsRead.begin(), rec.localsRead.end(), l) ==
              rec.localsRead.end()) {
            rec.localsRead.push_back(l);
          }
        }
        for (Name g : eff.mutableGlobalsRead) {
          if (std::find(rec.mutableGlobalsRead.begin(),
                        rec.mutableGlobalsRead.end(),
                        g) == rec.mutableGlobalsRead.end()) {
            rec.mutableGlobalsRead.push_back(g);
          }
        }
        if (eff.readsMutableStruct || eff.readsSharedMutableStruct) {
          rec.readsMutableStruct = true;
          if (auto* sg = e->dynCast<StructGet>()) {
            if (!eff.readsSharedMutableStruct && sg->ref->type.isRef()) {
              std::pair<HeapType, Index> key{sg->ref->type.getHeapType(),
                                             sg->index};
              if (std::find(rec.mutableStructFieldsRead.begin(),
                            rec.mutableStructFieldsRead.end(),
                            key) == rec.mutableStructFieldsRead.end()) {
                rec.mutableStructFieldsRead.push_back(key);
              }
            } else {
              rec.readsUnknownMutableStruct = true;
            }
          } else if (auto* call = e->dynCast<Call>()) {
            if (eff.readsSharedMutableStruct ||
                !isPureReadOnlyCall(call, wasm, options) ||
                !collectPureCallStructFieldsRead(
                  wasm.getFunctionOrNull(call->target),
                  rec.mutableStructFieldsRead)) {
              rec.readsUnknownMutableStruct = true;
            }
          } else {
            rec.readsUnknownMutableStruct = true;
          }
        }
        if (eff.readsMutableArray || eff.readsSharedMutableArray) {
          rec.readsMutableArray = true;
        }
        if (eff.readsMemory || eff.readsSharedMemory) {
          rec.readsMemory = true;
        }
        if (eff.readsTable) {
          rec.readsTable = true;
        }
      }
      for (auto* ch : ChildIterator(e)) {
        walkNorm(ch);
      }
    };
    walkNorm(root);
  }

  bool isKilledByEffects(const OrigRecord& rec,
                         const EffectAnalyzer& eff,
                         Expression* killerExpr = nullptr) const {
    if (killerExpr && isIdempotent(killerExpr, wasm) &&
        ExpressionAnalyzer::shallowEqual(killerExpr, rec.orig)) {
      return false;
    }
    for (Index l : rec.localsRead) {
      if (eff.localsWritten.contains(l)) {
        return true;
      }
    }
    if (rec.readsMutableStruct &&
        (eff.writesStruct || eff.writesSharedStruct || eff.calls)) {
      if (rec.readsUnknownMutableStruct || eff.writesSharedStruct ||
          eff.calls || eff.hasUnknownMutableStructWrite) {
        return true;
      }
      bool aliasFound = false;
      for (const auto& [wHT, wIdx] : eff.mutableStructFieldsWritten) {
        for (const auto& [rHT, rIdx] : rec.mutableStructFieldsRead) {
          if (wIdx == rIdx &&
              (HeapType::isSubType(wHT, rHT) ||
               HeapType::isSubType(rHT, wHT))) {
            aliasFound = true;
            break;
          }
        }
        if (aliasFound) {
          break;
        }
      }
      if (aliasFound) {
        return true;
      }
    }
    if (rec.readsMutableArray &&
        (eff.writesArray || eff.writesSharedArray || eff.calls)) {
      return true;
    }
    if (rec.readsMemory &&
        (eff.writesMemory || eff.writesSharedMemory || eff.calls)) {
      return true;
    }
    if (rec.readsTable && (eff.writesTable || eff.calls)) {
      return true;
    }
    if (!rec.mutableGlobalsRead.empty()) {
      if (eff.calls) {
        return true;
      }
      for (Name g : rec.mutableGlobalsRead) {
        if (eff.globalsWritten.contains(g)) {
          return true;
        }
      }
    }
    return false;
  }

  void invalidateMapByLocalWrite(ActiveMap& map, Index localIdx) {
    for (auto it = map.begin(); it != map.end();) {
      auto& vec = it->second;
      vec.erase(
        std::remove_if(vec.begin(),
                       vec.end(),
                       [&](Index origIdx) {
                         const auto& rec = origRecords[origIdx];
                         return std::find(rec.localsRead.begin(),
                                          rec.localsRead.end(),
                                          localIdx) != rec.localsRead.end();
                       }),
        vec.end());
      if (vec.empty()) {
        it = map.erase(it);
      } else {
        ++it;
      }
    }
  }

  void invalidateByLocalWrite(Index localIdx) {
    invalidateMapByLocalWrite(activeMap, localIdx);
  }

  void invalidateByEffects(const EffectAnalyzer& eff,
                           Expression* killerExpr = nullptr) {
    if (!eff.hasNonTrapSideEffects()) {
      return;
    }
    for (auto it = activeMap.begin(); it != activeMap.end();) {
      auto& vec = it->second;
      vec.erase(std::remove_if(vec.begin(),
                               vec.end(),
                               [&](Index origIdx) {
                                 return isKilledByEffects(
                                   origRecords[origIdx], eff, killerExpr);
                               }),
                vec.end());
      if (vec.empty()) {
        it = activeMap.erase(it);
      } else {
        ++it;
      }
    }
  }

  bool isMultiChildTeeSafe(Expression* curr) {
    SmallVector<Expression*, 4> children;
    for (auto* ch : ChildIterator(curr)) {
      children.push_back(ch);
    }
    if (children.size() <= 1) {
      return true;
    }
    SmallVector<OrigRecord, 4> childNormRecs(children.size());
    for (Index i = 0; i < children.size(); i++) {
      Index lIdx;
      if (getNormLocalIndex(children[i], lIdx)) {
        childNormRecs[i].localsRead.push_back(lIdx);
      } else {
        populateNormReadEffects(children[i], childNormRecs[i]);
      }
    }
    for (Index j = 1; j < children.size(); j++) {
      EffectAnalyzer jEff(options, wasm, children[j]);
      jEff.trap = false;
      if (!jEff.hasNonTrapSideEffects()) {
        continue;
      }
      for (Index i = 0; i < j; i++) {
        if (isKilledByEffects(childNormRecs[i], jEff, children[j])) {
          return false;
        }
      }
    }
    return true;
  }

  static bool factMatches(const OrigRecord& rec, Expression* curr) {
    if (rec.altBinaryOp != InvalidBinary) {
      auto* bin = curr->dynCast<Binary>();
      auto* origBin = rec.orig->dynCast<Binary>();
      if (!bin || !origBin || bin->op != rec.altBinaryOp ||
          bin->type != origBin->type) {
        return false;
      }
      return normEqual(origBin->left, bin->left) &&
             normEqual(origBin->right, bin->right);
    }
    return normEqual(rec.orig, curr);
  }

  void addFactRecord(ActiveMap& map,
                     Expression* expr,
                     size_t hash,
                     Index estBytes,
                     Literal knownConst,
                     int8_t knownNullness = 0,
                     bool knownNonZero = false,
                     BinaryOp altBinaryOp = InvalidBinary) {
    if (!expr || !expr->type.isConcrete()) {
      return;
    }
    Index newIdx = origRecords.size();
    OrigRecord rec;
    rec.orig = expr;
    rec.hash = hash;
    rec.estBytes = estBytes;
    rec.knownConst = knownConst;
    rec.knownNullness = knownNullness;
    rec.knownNonZero = knownNonZero;
    rec.isFactOnly = true;
    rec.altBinaryOp = altBinaryOp;
    Index lIdx;
    if (getNormLocalIndex(expr, lIdx)) {
      rec.localsRead.push_back(lIdx);
    } else {
      populateNormReadEffects(expr, rec);
    }
    origRecords.push_back(std::move(rec));
    map[hash].push_back(newIdx);
  }

  void addFactForExpr(ActiveMap& map,
                      Expression* expr,
                      Literal knownConst,
                      int8_t knownNullness = 0,
                      bool knownNonZero = false) {
    while (expr) {
      if (auto* b = expr->dynCast<Block>()) {
        if (!b->list.empty() && !SubtreeTargetsName(b, b->name)) {
          expr = b->list.back();
          continue;
        }
      }
      break;
    }
    if (!expr) {
      return;
    }
    auto it = exprInfoMap.find(expr);
    if (it != exprInfoMap.end() && it->second.possible) {
      addFactRecord(map,
                    expr,
                    it->second.hash,
                    it->second.estBytes,
                    knownConst,
                    knownNullness,
                    knownNonZero);
    }
    if (auto* ls = expr->dynCast<LocalSet>()) {
      if (ls->isTee() &&
          !EffectAnalyzer(options, wasm, ls->value)
             .localsWritten.contains(ls->index)) {
        addFactForExpr(
          map, ls->value, knownConst, knownNullness, knownNonZero);
      }
    } else if (knownNullness != 0) {
      if (auto* rc = expr->dynCast<RefCast>()) {
        addFactForExpr(map, rc->ref, Literal(), knownNullness, false);
      } else if (auto* ra = expr->dynCast<RefAs>()) {
        if (ra->op == RefAsNonNull) {
          addFactForExpr(map, ra->value, Literal(), knownNullness, false);
        }
      }
    }
  }

  void recordConditionFacts(Expression* cond,
                            ActiveMap& trueMap,
                            ActiveMap& falseMap) {
    while (cond) {
      if (auto* b = cond->dynCast<Block>()) {
        if (!b->list.empty() && !SubtreeTargetsName(b, b->name)) {
          cond = b->list.back();
          continue;
        }
      }
      break;
    }
    if (!cond) {
      return;
    }
    LocalSet* teeExpr = nullptr;
    if (auto* ls = cond->dynCast<LocalSet>()) {
      if (ls->isTee()) {
        teeExpr = ls;
        cond = ls->value;
      }
    }
    if (auto* u = cond->dynCast<Unary>()) {
      if (u->op == EqZInt32) {
        if (teeExpr) {
          size_t teeHash = wasm::hash(teeExpr->index);
          addFactRecord(
            trueMap, teeExpr, teeHash, 2, Literal(int32_t(1)), 0, true);
          if (!EffectAnalyzer(options, wasm, cond)
                 .localsWritten.contains(teeExpr->index)) {
            addFactRecord(falseMap, teeExpr, teeHash, 2, Literal(int32_t(0)));
          }
        }
        recordConditionFacts(u->value, falseMap, trueMap);
        return;
      }
    }

    auto infoIt = exprInfoMap.find(cond);
    NodeInfo condInfo =
      (infoIt != exprInfoMap.end()) ? infoIt->second : NodeInfo{};

    if (condInfo.possible && cond->type == Type::i32) {
      bool isBool = isBooleanValueExpr(cond);
      if (isBool) {
        addFactRecord(trueMap,
                      cond,
                      condInfo.hash,
                      condInfo.estBytes,
                      Literal(int32_t(1)),
                      0,
                      true);
      } else {
        addFactRecord(
          trueMap, cond, condInfo.hash, condInfo.estBytes, Literal(), 0, true);
      }
      addFactRecord(falseMap,
                    cond,
                    condInfo.hash,
                    condInfo.estBytes,
                    Literal(int32_t(0)));
      if (teeExpr) {
        size_t teeHash = wasm::hash(teeExpr->index);
        if (isBool) {
          addFactRecord(
            trueMap, teeExpr, teeHash, 2, Literal(int32_t(1)), 0, true);
        } else {
          addFactRecord(trueMap, teeExpr, teeHash, 2, Literal(), 0, true);
        }
        if (!EffectAnalyzer(options, wasm, cond)
               .localsWritten.contains(teeExpr->index)) {
          addFactRecord(falseMap, teeExpr, teeHash, 2, Literal(int32_t(0)));
        }
      }
    }

    if (auto* rn = cond->dynCast<RefIsNull>()) {
      addFactForExpr(trueMap, rn->value, Literal(), 1);
      addFactForExpr(falseMap, rn->value, Literal(), -1);
    } else if (auto* rt = cond->dynCast<RefTest>()) {
      if (rt->castType.isNonNullable()) {
        addFactForExpr(trueMap, rt->ref, Literal(), -1);
      } else {
        addFactForExpr(falseMap, rt->ref, Literal(), -1);
      }
    } else if (auto* req = cond->dynCast<RefEq>()) {
      Expression* other = nullptr;
      if (req->left->is<RefNull>()) {
        other = req->right;
      } else if (req->right->is<RefNull>()) {
        other = req->left;
      }
      if (other) {
        addFactForExpr(trueMap, other, Literal(), 1);
        addFactForExpr(falseMap, other, Literal(), -1);
      }
    } else if (auto* bin = cond->dynCast<Binary>()) {
      if (bin->op == AndInt32) {
        ActiveMap ignoredFalseMap;
        EffectAnalyzer rightEff(options, wasm, bin->right);
        rightEff.trap = false;
        if (!rightEff.hasNonTrapSideEffects()) {
          recordConditionFacts(bin->left, trueMap, ignoredFalseMap);
        }
        recordConditionFacts(bin->right, trueMap, ignoredFalseMap);
      } else if (bin->op == OrInt32) {
        ActiveMap ignoredTrueMap;
        EffectAnalyzer rightEff(options, wasm, bin->right);
        rightEff.trap = false;
        if (!rightEff.hasNonTrapSideEffects()) {
          recordConditionFacts(bin->left, ignoredTrueMap, falseMap);
        }
        recordConditionFacts(bin->right, ignoredTrueMap, falseMap);
      } else if (condInfo.possible && bin->isRelational()) {
        BinaryOp compOp = getComplementaryIntRelOp(bin->op);
        if (compOp != InvalidBinary) {
          auto lIt = exprInfoMap.find(bin->left);
          auto rIt = exprInfoMap.find(bin->right);
          if (lIt != exprInfoMap.end() && rIt != exprInfoMap.end() &&
              lIt->second.possible && rIt->second.possible) {
            BinaryOp origOp = bin->op;
            bin->op = compOp;
            size_t compHash = ExpressionAnalyzer::shallowHash(bin);
            bin->op = origOp;
            hash_combine(compHash, lIt->second.hash);
            hash_combine(compHash, rIt->second.hash);
            addFactRecord(trueMap,
                          bin,
                          compHash,
                          condInfo.estBytes,
                          Literal(int32_t(0)),
                          0,
                          false,
                          compOp);
            addFactRecord(falseMap,
                          bin,
                          compHash,
                          condInfo.estBytes,
                          Literal(int32_t(1)),
                          0,
                          true,
                          compOp);
          }
        }
        if (bin->op == EqInt32 || bin->op == NeInt32 || bin->op == EqInt64 ||
            bin->op == NeInt64) {
          Const* c = bin->right->dynCast<Const>();
          Expression* other = bin->left;
          if (!c) {
            c = bin->left->dynCast<Const>();
            other = bin->right;
          }
          if (c && !Properties::isConstantExpression(other)) {
            bool isEq = (bin->op == EqInt32 || bin->op == EqInt64);
            ActiveMap& eqMap = isEq ? trueMap : falseMap;
            ActiveMap& neMap = isEq ? falseMap : trueMap;
            bool isZeroConst =
              (c->type == Type::i32 && c->value.geti32() == 0) ||
              (c->type == Type::i64 && c->value.geti64() == 0);
            addFactForExpr(eqMap, other, c->value, 0, !isZeroConst);
            if (isZeroConst) {
              addFactForExpr(neMap, other, Literal(), 0, true);
            }
          }
        }
      }
    } else if (auto* u = cond->dynCast<Unary>()) {
      if (u->op == EqZInt64) {
        addFactForExpr(trueMap, u->value, Literal(int64_t(0)));
        addFactForExpr(falseMap, u->value, Literal(), 0, true);
      }
    }
  }

  void recordNonNullFact(Expression* ref) {
    addFactForExpr(activeMap, ref, Literal(), -1);
  }

  NodeInfo walkExpr(Expression* curr) {
    lastCondExpr = nullptr;
    if (!curr) {
      return {};
    }
    if (auto* block = curr->dynCast<Block>()) {
      Index firstBranchIdx = block->list.size();
      if (block->name.is()) {
        for (Index i = 0; i < block->list.size(); i++) {
          if (SubtreeTargetsName(block->list[i], block->name)) {
            firstBranchIdx = i;
            break;
          }
        }
      }
      ActiveMap savedExitMap;
      ActiveMap savedTakenExitMap;
      bool capturedAfterDirectBr = false;
      for (Index i = 0; i < block->list.size(); i++) {
        if (i == firstBranchIdx) {
          bool isDirectCondBr = false;
          Expression* brStmt = block->list[i];
          if (auto* d = brStmt->dynCast<Drop>()) {
            brStmt = d->value;
          }
          if (auto* br = brStmt->dynCast<Break>()) {
            isDirectCondBr = br->name == block->name && br->condition &&
                             !SubtreeTargetsName(br->value, block->name) &&
                             !SubtreeTargetsName(br->condition, block->name);
          } else if (auto* bron = brStmt->dynCast<BrOn>()) {
            isDirectCondBr = bron->name == block->name &&
                             !SubtreeTargetsName(bron->ref, block->name) &&
                             !SubtreeTargetsName(bron->desc, block->name);
          }
          if (!isDirectCondBr) {
            savedExitMap = activeMap;
          } else {
            capturedAfterDirectBr = true;
          }
        }
        hasLastBrExitMap = false;
        walkExpr(block->list[i]);
        if (i == firstBranchIdx && capturedAfterDirectBr) {
          if (hasLastBrExitMap) {
            savedExitMap = std::move(lastBrExitMap);
            savedTakenExitMap = std::move(lastBrTakenMap);
          } else {
            savedExitMap = activeMap;
            savedTakenExitMap = activeMap;
          }
          hasLastBrExitMap = false;
        }
      }
      if (block->type == Type::unreachable) {
        activeMap.clear();
        lastCondExpr = nullptr;
      } else if (firstBranchIdx < block->list.size()) {
        bool onlyExitsViaFirstDirectBr =
          capturedAfterDirectBr && firstBranchIdx + 1 < block->list.size();
        bool tailDoesNotReturn = false;
        if (onlyExitsViaFirstDirectBr) {
          for (Index i = firstBranchIdx + 1; i < block->list.size(); i++) {
            if (SubtreeTargetsName(block->list[i], block->name)) {
              onlyExitsViaFirstDirectBr = false;
              break;
            }
            if (doesNotReturn(block->list[i])) {
              tailDoesNotReturn = true;
            }
          }
        }
        if (onlyExitsViaFirstDirectBr && tailDoesNotReturn) {
          activeMap = std::move(savedTakenExitMap);
        } else {
          activeMap = std::move(savedExitMap);
          for (Index i = firstBranchIdx; i < block->list.size(); i++) {
            if (i == firstBranchIdx && capturedAfterDirectBr) {
              continue;
            }
            EffectAnalyzer eff(options, wasm, block->list[i]);
            eff.trap = false;
            invalidateByEffects(eff, block->list[i]);
          }
        }
        lastCondExpr = nullptr;
      } else if (!block->list.empty() && lastCondExpr == block->list.back()) {
        lastCondExpr = block;
      } else {
        lastCondExpr = nullptr;
      }
      return {};
    }
    if (auto* iff = curr->dynCast<If>()) {
      walkExpr(iff->condition);
      ActiveMap afterCond = activeMap;
      ActiveMap condTrueMap = (lastCondExpr == iff->condition)
                                ? std::move(lastCondTrueMap)
                                : afterCond;
      ActiveMap condFalseMap = (lastCondExpr == iff->condition)
                                 ? std::move(lastCondFalseMap)
                                 : afterCond;
      lastCondExpr = nullptr;
      recordConditionFacts(iff->condition, condTrueMap, condFalseMap);

      Index origStart = origRecords.size();
      activeMap = std::move(condTrueMap);
      walkExpr(iff->ifTrue);
      ActiveMap thenTrueMap =
        (lastCondExpr == iff->ifTrue) ? std::move(lastCondTrueMap) : activeMap;
      ActiveMap thenFalseMap =
        (lastCondExpr == iff->ifTrue) ? std::move(lastCondFalseMap) : activeMap;
      if (iff->type == Type::i32 && iff->ifFalse) {
        recordConditionFacts(iff->ifTrue, thenTrueMap, thenFalseMap);
      }
      ActiveMap afterTrue = std::move(activeMap);
      lastCondExpr = nullptr;

      ActiveMap afterFalse;
      ActiveMap elseTrueMap;
      ActiveMap elseFalseMap;
      if (iff->ifFalse) {
        activeMap = std::move(condFalseMap);
        walkExpr(iff->ifFalse);
        elseTrueMap = (lastCondExpr == iff->ifFalse)
                        ? std::move(lastCondTrueMap)
                        : activeMap;
        elseFalseMap = (lastCondExpr == iff->ifFalse)
                         ? std::move(lastCondFalseMap)
                         : activeMap;
        if (iff->type == Type::i32) {
          recordConditionFacts(iff->ifFalse, elseTrueMap, elseFalseMap);
        }
        afterFalse = std::move(activeMap);
        lastCondExpr = nullptr;
      } else {
        elseTrueMap = condFalseMap;
        elseFalseMap = condFalseMap;
        afterFalse = std::move(condFalseMap);
      }

      for (Index k = origStart; k < origRecords.size(); k++) {
        origRecords[k].exitedArmScope = true;
      }

      bool thenNR = doesNotReturn(iff->ifTrue);
      bool elseNR = iff->ifFalse && doesNotReturn(iff->ifFalse);

      if (iff->type == Type::unreachable || (thenNR && elseNR)) {
        activeMap.clear();
        return {};
      }

      if (thenNR && !elseNR) {
        activeMap = std::move(afterFalse);
      } else if (elseNR && !thenNR) {
        activeMap = std::move(afterTrue);
      } else {
        activeMap = std::move(afterCond);
        EffectAnalyzer thenEff(options, wasm, iff->ifTrue);
        thenEff.trap = false;
        invalidateByEffects(thenEff, iff->ifTrue);
        if (iff->ifFalse) {
          EffectAnalyzer elseEff(options, wasm, iff->ifFalse);
          elseEff.trap = false;
          invalidateByEffects(elseEff, iff->ifFalse);
        }
      }

      if (iff->type == Type::i32 && iff->ifFalse) {
        bool thenAlwaysZero = thenNR || isConstZeroI32(iff->ifTrue);
        bool elseAlwaysZero = elseNR || isConstZeroI32(iff->ifFalse);
        bool thenAlwaysNonZero = thenNR || isConstNonZeroI32(iff->ifTrue);
        bool elseAlwaysNonZero = elseNR || isConstNonZeroI32(iff->ifFalse);

        if (elseAlwaysZero && !thenAlwaysZero) {
          lastCondTrueMap = std::move(thenTrueMap);
        } else if (thenAlwaysZero && !elseAlwaysZero) {
          lastCondTrueMap = std::move(elseTrueMap);
        } else {
          lastCondTrueMap = activeMap;
        }

        if (thenAlwaysNonZero && !elseAlwaysNonZero) {
          lastCondFalseMap = std::move(elseFalseMap);
        } else if (elseAlwaysNonZero && !thenAlwaysNonZero) {
          lastCondFalseMap = std::move(thenFalseMap);
        } else {
          lastCondFalseMap = activeMap;
        }
        lastCondExpr = iff;
      }
      return {};
    }
    if (auto* loop = curr->dynCast<Loop>()) {
      EffectAnalyzer loopEff(options, wasm, loop->body);
      loopEff.trap = false;
      invalidateByEffects(loopEff, loop->body);
      ActiveMap beforeLoop = activeMap;
      walkExpr(loop->body);
      if (loop->type == Type::unreachable) {
        activeMap.clear();
      } else {
        activeMap = std::move(beforeLoop);
      }
      return {};
    }
    if (auto* tryy = curr->dynCast<Try>()) {
      ActiveMap beforeTry = activeMap;
      walkExpr(tryy->body);
      EffectAnalyzer bodyEff(options, wasm, tryy->body);
      bodyEff.trap = false;
      for (auto* cb : tryy->catchBodies) {
        activeMap = beforeTry;
        invalidateByEffects(bodyEff, tryy->body);
        walkExpr(cb);
      }
      if (tryy->type == Type::unreachable) {
        activeMap.clear();
      } else {
        activeMap = std::move(beforeTry);
        EffectAnalyzer tryEff(options, wasm, tryy);
        tryEff.trap = false;
        invalidateByEffects(tryEff, tryy);
      }
      return {};
    }
    if (auto* tt = curr->dynCast<TryTable>()) {
      ActiveMap beforeTry = activeMap;
      walkExpr(tt->body);
      if (tt->type == Type::unreachable) {
        activeMap.clear();
      } else {
        activeMap = std::move(beforeTry);
        EffectAnalyzer ttEff(options, wasm, tt);
        ttEff.trap = false;
        invalidateByEffects(ttEff, tt);
      }
      return {};
    }
    if (auto* br = curr->dynCast<Break>()) {
      if (br->value) {
        walkExpr(br->value);
      }
      if (br->condition) {
        walkExpr(br->condition);
        ActiveMap afterCond = activeMap;
        ActiveMap condTrueMap = (lastCondExpr == br->condition)
                                  ? std::move(lastCondTrueMap)
                                  : afterCond;
        ActiveMap condFalseMap = (lastCondExpr == br->condition)
                                   ? std::move(lastCondFalseMap)
                                   : afterCond;
        lastCondExpr = nullptr;
        recordConditionFacts(br->condition, condTrueMap, condFalseMap);
        hasLastBrExitMap = true;
        lastBrExitMap = std::move(afterCond);
        lastBrTakenMap = std::move(condTrueMap);
        activeMap = std::move(condFalseMap);
      }
      if (!br->condition) {
        activeMap.clear();
      }
      lastCondExpr = nullptr;
      return {};
    }
    if (auto* bron = curr->dynCast<BrOn>()) {
      walkExpr(bron->ref);
      if (bron->desc) {
        walkExpr(bron->desc);
      }
      lastCondExpr = nullptr;
      ActiveMap afterRef = activeMap;
      ActiveMap takenMap = afterRef;
      ActiveMap fallthroughMap = std::move(activeMap);
      if (bron->op == BrOnNull || bron->op == BrOnNonNull) {
        int8_t takenNullness = (bron->op == BrOnNull) ? 1 : -1;
        int8_t fallNullness = (bron->op == BrOnNull) ? -1 : 1;
        addFactForExpr(takenMap, bron->ref, Literal(), takenNullness);
        addFactForExpr(fallthroughMap, bron->ref, Literal(), fallNullness);
      } else if (bron->op == BrOnCast) {
        if (bron->castType.isNonNullable()) {
          addFactForExpr(takenMap, bron->ref, Literal(), -1);
        } else {
          addFactForExpr(fallthroughMap, bron->ref, Literal(), -1);
        }
      } else if (bron->op == BrOnCastFail) {
        if (bron->castType.isNonNullable()) {
          addFactForExpr(fallthroughMap, bron->ref, Literal(), -1);
        } else {
          addFactForExpr(takenMap, bron->ref, Literal(), -1);
        }
      }
      hasLastBrExitMap = true;
      lastBrExitMap = std::move(afterRef);
      lastBrTakenMap = std::move(takenMap);
      activeMap = std::move(fallthroughMap);
      return {};
    }
    if (auto* sw = curr->dynCast<Switch>()) {
      if (sw->value) {
        walkExpr(sw->value);
      }
      walkExpr(sw->condition);
      activeMap.clear();
      lastCondExpr = nullptr;
      return {};
    }
    if (auto* ret = curr->dynCast<Return>()) {
      if (ret->value) {
        walkExpr(ret->value);
      }
      activeMap.clear();
      lastCondExpr = nullptr;
      return {};
    }
    if (auto* th = curr->dynCast<Throw>()) {
      for (auto* op : th->operands) {
        walkExpr(op);
      }
      activeMap.clear();
      lastCondExpr = nullptr;
      return {};
    }
    if (curr->is<Rethrow>() || curr->is<Unreachable>()) {
      activeMap.clear();
      return {};
    }
    if (auto* lg = curr->dynCast<LocalGet>()) {
      if (func->isVar(lg->index)) {
        activeVarTypes.insert(getBinaryVarType(func->getLocalType(lg->index)));
      }
      NodeInfo info{wasm::hash(lg->index), 2, true, false};
      exprInfoMap[curr] = info;
      auto bucketIt = activeMap.find(info.hash);
      if (bucketIt != activeMap.end()) {
        for (Index origIdx : bucketIt->second) {
          const auto& rec = origRecords[origIdx];
          if (rec.isFactOnly && rec.knownConst.type != Type::none &&
              getConstEstBytes(rec.knownConst) <= 2 &&
              factMatches(rec, curr)) {
            repeatToOrigIdx[curr] = origIdx;
            origRecords[origIdx].reuses++;
            break;
          }
        }
      }
      return info;
    }
    if (auto* ls = curr->dynCast<LocalSet>()) {
      if (func->isVar(ls->index)) {
        activeVarTypes.insert(getBinaryVarType(func->getLocalType(ls->index)));
      }
      NodeInfo valInfo = walkExpr(ls->value);
      bool valWasLastCond = (lastCondExpr == ls->value);
      invalidateByLocalWrite(ls->index);
      if (valWasLastCond) {
        invalidateMapByLocalWrite(lastCondTrueMap, ls->index);
        invalidateMapByLocalWrite(lastCondFalseMap, ls->index);
        if (ls->isTee()) {
          lastCondExpr = ls;
        } else {
          lastCondExpr = nullptr;
        }
      } else {
        lastCondExpr = nullptr;
      }
      if (valInfo.possible && !origRecords.empty() &&
          origRecords.back().orig == ls->value &&
          !origRecords.back().isFactOnly &&
          origRecords.back().reuses == 0) {
        Type localType = func->getLocalType(ls->index);
        Type valType = ls->value->type;
        bool exactMatch = (localType == valType);
        bool nullableMatch =
          !exactMatch && localType.isRef() && valType.isRef() &&
          localType.getHeapType() == valType.getHeapType() &&
          localType.isNullable() && valType.isNonNullable() &&
          localType.getExactness() == valType.getExactness();
        if (exactMatch || nullableMatch) {
          Index valOrigIdx = origRecords.size() - 1;
          auto& rec = origRecords[valOrigIdx];
          if (std::find(rec.localsRead.begin(),
                        rec.localsRead.end(),
                        ls->index) == rec.localsRead.end()) {
            rec.existingLocal = ls->index;
            rec.existingLocalNeedsNonNull = nullableMatch;
            rec.localsRead.push_back(ls->index);
          }
        }
      }
      if (ls->isTee()) {
        NodeInfo info{wasm::hash(ls->index), 2, true, true};
        exprInfoMap[curr] = info;
        return info;
      }
      return {};
    }

    bool childrenPossible = true;
    bool hasTee = false;
    size_t hash = ExpressionAnalyzer::shallowHash(curr);
    Index estBytes = getShallowEstBytes(curr);
    for (auto* child : ChildIterator(curr)) {
      NodeInfo cInfo = walkExpr(child);
      if (!cInfo.possible) {
        childrenPossible = false;
      } else {
        hash_combine(hash, cInfo.hash);
        estBytes += cInfo.estBytes;
        hasTee = hasTee || cInfo.hasTee;
      }
    }
    if (auto* u = curr->dynCast<Unary>()) {
      if (u->op == EqZInt32 && lastCondExpr == u->value) {
        std::swap(lastCondTrueMap, lastCondFalseMap);
        lastCondExpr = u;
      } else {
        lastCondExpr = nullptr;
      }
    } else {
      lastCondExpr = nullptr;
    }

    bool idempotentCall = isIdempotent(curr, wasm);
    bool pureReadOnlyCall = isPureReadOnlyCall(curr, wasm, options);
    ShallowEffectAnalyzer shallow(options, wasm, curr);
    shallow.trap = false;
    if (idempotentCall) {
      if (auto* target = wasm.getFunctionOrNull(curr->cast<Call>()->target)) {
        if (target->effects && target->effects->onlyTouchesInitGlobals) {
          shallow.calls = false;
          shallow.writesArray = false;
          shallow.writesSharedArray = false;
        }
      }
    }
    if (shallow.hasNonTrapSideEffects()) {
      invalidateByEffects(shallow, curr);
    }
    if (shallow.branchesOut || curr->type == Type::unreachable ||
        (curr->is<Call>() && doesNotReturn(curr))) {
      activeMap.clear();
      return {};
    }

    if (auto* ra = curr->dynCast<RefAs>()) {
      if (ra->op == RefAsNonNull) {
        recordNonNullFact(ra->value);
      }
    } else if (auto* sg = curr->dynCast<StructGet>()) {
      recordNonNullFact(sg->ref);
    } else if (auto* ss = curr->dynCast<StructSet>()) {
      recordNonNullFact(ss->ref);
    } else if (auto* ag = curr->dynCast<ArrayGet>()) {
      recordNonNullFact(ag->ref);
    } else if (auto* as = curr->dynCast<ArraySet>()) {
      recordNonNullFact(as->ref);
    } else if (auto* al = curr->dynCast<ArrayLen>()) {
      recordNonNullFact(al->ref);
    } else if (auto* rc = curr->dynCast<RefCast>()) {
      if (rc->type.isNonNullable()) {
        recordNonNullFact(rc->ref);
      }
    }

    bool shallowPossible =
      idempotentCall || pureReadOnlyCall ||
      (!shallow.hasNonTrapSideEffects() &&
       !Properties::isShallowlyGenerative(curr, func, wasm));
    bool possible = childrenPossible && shallowPossible;
    if (possible && hasTee && !isMultiChildTeeSafe(curr)) {
      possible = false;
    }

    NodeInfo info{hash, estBytes, possible, hasTee};
    exprInfoMap[curr] = info;
    if (!possible || !curr->type.isConcrete() ||
        Properties::isConstantExpression(curr)) {
      return info;
    }

    if (!hasTee) {
      auto foldWithConstFact = [&](Literal val) -> bool {
        if (getConstEstBytes(val) > estBytes) {
          return false;
        }
        Index fIdx = origRecords.size();
        OrigRecord fRec;
        fRec.orig = curr;
        fRec.hash = hash;
        fRec.estBytes = estBytes;
        fRec.knownConst = val;
        fRec.isFactOnly = true;
        fRec.reuses = 1;
        origRecords.push_back(std::move(fRec));
        repeatToOrigIdx[curr] = fIdx;
        for (auto* child : ChildIterator(curr)) {
          auto cIt = repeatToOrigIdx.find(child);
          if (cIt != repeatToOrigIdx.end()) {
            Index cOrigIdx = cIt->second;
            unmatchedChildren[curr].push_back({child, cOrigIdx});
            assert(origRecords[cOrigIdx].reuses > 0);
            origRecords[cOrigIdx].reuses--;
            repeatToOrigIdx.erase(cIt);
          }
        }
        return true;
      };

      if (auto* rn = curr->dynCast<RefIsNull>()) {
        auto vIt = exprInfoMap.find(rn->value);
        if (vIt != exprInfoMap.end() && vIt->second.possible) {
          auto bIt = activeMap.find(vIt->second.hash);
          if (bIt != activeMap.end()) {
            for (Index origIdx : bIt->second) {
              const auto& rec = origRecords[origIdx];
              if (rec.knownNullness != 0 && factMatches(rec, rn->value)) {
                int32_t res = (rec.knownNullness > 0) ? 1 : 0;
                if (foldWithConstFact(Literal(res))) {
                  return info;
                }
                break;
              }
            }
          }
        }
      } else if (auto* rt = curr->dynCast<RefTest>()) {
        auto vIt = exprInfoMap.find(rt->ref);
        if (vIt != exprInfoMap.end() && vIt->second.possible) {
          auto bIt = activeMap.find(vIt->second.hash);
          if (bIt != activeMap.end()) {
            for (Index origIdx : bIt->second) {
              const auto& rec = origRecords[origIdx];
              if (rec.knownNullness > 0 && factMatches(rec, rt->ref)) {
                int32_t res = rt->castType.isNullable() ? 1 : 0;
                if (foldWithConstFact(Literal(res))) {
                  return info;
                }
                break;
              }
            }
          }
        }
      } else if (auto* u = curr->dynCast<Unary>()) {
        if (u->op == EqZInt32 || u->op == EqZInt64) {
          auto vIt = exprInfoMap.find(u->value);
          if (vIt != exprInfoMap.end() && vIt->second.possible) {
            auto bIt = activeMap.find(vIt->second.hash);
            if (bIt != activeMap.end()) {
              for (Index origIdx : bIt->second) {
                const auto& rec = origRecords[origIdx];
                if ((rec.knownNonZero || rec.knownConst.type != Type::none) &&
                    factMatches(rec, u->value)) {
                  bool isZero = false;
                  bool known = false;
                  if (rec.knownNonZero) {
                    isZero = false;
                    known = true;
                  } else if (rec.knownConst.type == Type::i32) {
                    isZero = (rec.knownConst.geti32() == 0);
                    known = true;
                  } else if (rec.knownConst.type == Type::i64) {
                    isZero = (rec.knownConst.geti64() == 0);
                    known = true;
                  }
                  if (known &&
                      foldWithConstFact(Literal(int32_t(isZero ? 1 : 0)))) {
                    return info;
                  }
                  break;
                }
              }
            }
          }
        }
      } else if (auto* bin = curr->dynCast<Binary>()) {
        if (bin->isRelational()) {
          Const* cRight = bin->right->dynCast<Const>();
          Const* cLeft = bin->left->dynCast<Const>();
          if (cRight || cLeft) {
            Expression* other = cRight ? bin->left : bin->right;
            auto oIt = exprInfoMap.find(other);
            if (oIt != exprInfoMap.end() && oIt->second.possible) {
              auto bIt = activeMap.find(oIt->second.hash);
              if (bIt != activeMap.end()) {
                for (Index origIdx : bIt->second) {
                  const auto& rec = origRecords[origIdx];
                  if (rec.knownConst.type != Type::none &&
                      factMatches(rec, other)) {
                    Literal lVal = cLeft ? cLeft->value : rec.knownConst;
                    Literal rVal = cRight ? cRight->value : rec.knownConst;
                    Literal folded;
                    if (evalIntRelOp(bin->op, lVal, rVal, folded) &&
                        foldWithConstFact(folded)) {
                      return info;
                    }
                    break;
                  }
                  if (rec.knownNonZero &&
                      (bin->op == EqInt32 || bin->op == NeInt32 ||
                       bin->op == EqInt64 || bin->op == NeInt64)) {
                    Literal cVal = cRight ? cRight->value : cLeft->value;
                    bool isZeroC =
                      (cVal.type == Type::i32 && cVal.geti32() == 0) ||
                      (cVal.type == Type::i64 && cVal.geti64() == 0);
                    if (isZeroC && factMatches(rec, other)) {
                      int32_t res =
                        (bin->op == NeInt32 || bin->op == NeInt64) ? 1 : 0;
                      if (foldWithConstFact(Literal(res))) {
                        return info;
                      }
                      break;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    auto& bucket = activeMap[hash];
    if (!hasTee) {
      for (int pass = 0; pass < 2; pass++) {
        bool wantFact = (pass == 0);
        for (Index origIdx : bucket) {
          const auto& rec = origRecords[origIdx];
          if (rec.isFactOnly != wantFact || !factMatches(rec, curr)) {
            continue;
          }
          if (rec.isFactOnly) {
            if (rec.knownConst.type != Type::none) {
              if (getConstEstBytes(rec.knownConst) > estBytes) {
                continue;
              }
            } else if (rec.knownNullness > 0 && curr->type.isNullable()) {
              if (2 > estBytes) {
                continue;
              }
            } else {
              continue;
            }
          } else if (estBytes < 4) {
            continue;
          }
          repeatToOrigIdx[curr] = origIdx;
          origRecords[origIdx].reuses++;
          if (origRecords[origIdx].exitedArmScope) {
            origRecords[origIdx].crossedScope = true;
          }
          for (auto* child : ChildIterator(curr)) {
            auto cIt = repeatToOrigIdx.find(child);
            if (cIt != repeatToOrigIdx.end()) {
              Index cOrigIdx = cIt->second;
              unmatchedChildren[curr].push_back({child, cOrigIdx});
              assert(origRecords[cOrigIdx].reuses > 0);
              origRecords[cOrigIdx].reuses--;
              repeatToOrigIdx.erase(cIt);
            }
          }
          return info;
        }
      }
    }

    if (estBytes < 4) {
      return info;
    }

    for (Index origIdx : bucket) {
      if (!origRecords[origIdx].isFactOnly &&
          normEqual(origRecords[origIdx].orig, curr)) {
        return info;
      }
    }

    Index newOrigIdx = origRecords.size();
    OrigRecord rec;
    rec.orig = curr;
    rec.hash = hash;
    rec.estBytes = estBytes;
    if (curr->type.isRef() && curr->type.isNonNullable()) {
      rec.knownNullness = -1;
    }
    populateNormReadEffects(curr, rec);
    origRecords.push_back(std::move(rec));
    exprToOrigIdx[curr] = newOrigIdx;
    bucket.push_back(newOrigIdx);
    return info;
  }

  void run() {
    if (!func->body) {
      return;
    }
    walkExpr(func->body);
    if (repeatToOrigIdx.empty()) {
      return;
    }

    std::vector<SmallVector<Expression*, 2>> origRepeats(origRecords.size());
    for (auto& [rep, origIdx] : repeatToOrigIdx) {
      origRepeats[origIdx].push_back(rep);
    }

    auto getHeaderBytes = [&](Type t) -> Index {
      Type binType = getBinaryVarType(t);
      if (activeVarTypes.contains(binType)) {
        return 0;
      }
      if (t.isRef() && !t.getHeapType().isBasic()) {
        return 4;
      }
      if (options.shrinkLevel >= 2) {
        for (Index p = 0; p < func->getNumParams(); p++) {
          if (getBinaryVarType(func->getLocalType(p)) == binType) {
            return 0;
          }
        }
        return 2;
      }
      return 0;
    };

    auto passesByteCostGate = [&](const OrigRecord& rec,
                                  Index effBytes) -> bool {
      if (rec.reuses == 0) {
        return false;
      }
      if (rec.isFactOnly) {
        if (rec.knownConst.type != Type::none) {
          return effBytes >= getConstEstBytes(rec.knownConst);
        }
        if (rec.knownNullness > 0 && rec.orig->type.isNullable()) {
          return effBytes >= 2;
        }
        return false;
      }
      if (effBytes <= 2) {
        return false;
      }
      if (rec.existingLocal != Index(-1)) {
        Index repBytes = rec.existingLocalNeedsNonNull ? 3 : 2;
        return effBytes > repBytes && effBytes >= 4;
      }
      Index nonNullPenalty =
        (rec.crossedScope && rec.orig->type.isNonNullable()) ? 1 : 0;
      if (effBytes <= 2 + nonNullPenalty) {
        return false;
      }
      Index headerBytes = getHeaderBytes(rec.orig->type);
      return rec.reuses * (effBytes - 2 - nonNullPenalty) >=
             2 + nonNullPenalty + headerBytes + 1;
    };

    if (options.shrinkLevel >= 2) {
      std::unordered_map<Type, Index> typeBodyGains;
      std::unordered_map<Type, Index> typeCandCounts;
      for (int idx = int(origRecords.size()) - 1; idx >= 0; idx--) {
        const auto& rec = origRecords[idx];
        if (rec.reuses == 0 || rec.isFactOnly ||
            rec.existingLocal != Index(-1)) {
          continue;
        }
        if (rec.orig->type.isRef() && !rec.orig->type.getHeapType().isBasic()) {
          continue;
        }
        Type binType = getBinaryVarType(rec.orig->type);
        if (activeVarTypes.contains(binType)) {
          continue;
        }
        Index nonNullPenalty =
          (rec.crossedScope && rec.orig->type.isNonNullable()) ? 1 : 0;
        if (rec.estBytes <= 2 + nonNullPenalty) {
          continue;
        }
        Index lhs = rec.reuses * (rec.estBytes - 2 - nonNullPenalty);
        Index rhsNoHeader = 2 + nonNullPenalty;
        if (lhs > rhsNoHeader) {
          typeBodyGains[binType] += (lhs - rhsNoHeader);
          typeCandCounts[binType]++;
        }
      }
      for (auto& [binType, gain] : typeBodyGains) {
        if (typeCandCounts[binType] >= 2 && gain >= 3) {
          activeVarTypes.insert(binType);
        }
      }
    }

    // Evaluate originals in reverse postorder (parents before children).
    for (int idx = int(origRecords.size()) - 1; idx >= 0; idx--) {
      auto& rec = origRecords[idx];
      if (rec.reuses == 0) {
        continue;
      }
      if (rec.isFactOnly) {
        continue;
      }
      Index effBytes = rec.estBytes;
      for (auto* child : ChildIterator(rec.orig)) {
        auto cIt = exprToOrigIdx.find(child);
        if (cIt != exprToOrigIdx.end()) {
          const auto& cRec = origRecords[cIt->second];
          if (!cRec.isFactOnly &&
              passesByteCostGate(cRec, cRec.estBytes) &&
              cRec.estBytes > 2 && effBytes > cRec.estBytes - 2) {
            effBytes -= (cRec.estBytes - 2);
          }
        }
      }
      if (!passesByteCostGate(rec, effBytes)) {
        rec.reuses = 0;
        for (Expression* rep : origRepeats[idx]) {
          repeatToOrigIdx.erase(rep);
          auto uIt = unmatchedChildren.find(rep);
          if (uIt != unmatchedChildren.end()) {
            for (auto& [childExpr, cOrigIdx] : uIt->second) {
              repeatToOrigIdx[childExpr] = cOrigIdx;
              origRecords[cOrigIdx].reuses++;
              origRepeats[cOrigIdx].push_back(childExpr);
            }
          }
        }
      } else if (rec.existingLocal == Index(-1)) {
        activeVarTypes.insert(getBinaryVarType(rec.orig->type));
      }
    }

    std::unordered_map<Expression*, Index> origLocalMap;
    std::unordered_map<Expression*, Index> needsTeeWrap;
    bool hasAnyTransform = false;
    for (Index idx = 0; idx < origRecords.size(); idx++) {
      const auto& rec = origRecords[idx];
      if (rec.reuses == 0) {
        continue;
      }
      if (rec.isFactOnly) {
        hasAnyTransform = true;
        continue;
      }
      hasAnyTransform = true;
      if (rec.existingLocal != Index(-1)) {
        origLocalMap[rec.orig] = rec.existingLocal;
      } else {
        Index loc = Builder::addVar(func, rec.orig->type);
        origLocalMap[rec.orig] = loc;
        needsTeeWrap[rec.orig] = loc;
      }
    }

    if (!hasAnyTransform) {
      return;
    }

    struct DomApplier
      : public PostWalker<DomApplier, UnifiedExpressionVisitor<DomApplier>> {
      Module& wasm;
      const std::unordered_map<Expression*, Index>& repeatToOrigIdx;
      const std::vector<OrigRecord>& origRecords;
      const std::unordered_map<Expression*, Index>& origLocalMap;
      const std::unordered_map<Expression*, Index>& needsTeeWrap;

      DomApplier(Module& wasm,
                 const std::unordered_map<Expression*, Index>& repeatToOrigIdx,
                 const std::vector<OrigRecord>& origRecords,
                 const std::unordered_map<Expression*, Index>& origLocalMap,
                 const std::unordered_map<Expression*, Index>& needsTeeWrap)
        : wasm(wasm), repeatToOrigIdx(repeatToOrigIdx),
          origRecords(origRecords), origLocalMap(origLocalMap),
          needsTeeWrap(needsTeeWrap) {}

      void visitExpression(Expression* curr) {
        if (auto rIt = repeatToOrigIdx.find(curr);
            rIt != repeatToOrigIdx.end()) {
          const auto& rec = origRecords[rIt->second];
          if (rec.isFactOnly) {
            if (rec.knownConst.type != Type::none) {
              replaceCurrent(Builder(wasm).makeConst(rec.knownConst));
              return;
            }
            if (rec.knownNullness > 0 && curr->type.isNullable()) {
              replaceCurrent(Builder(wasm).makeRefNull(
                curr->type.getHeapType().getBottom()));
              return;
            }
          }
          Expression* orig = rec.orig;
          auto lIt = origLocalMap.find(orig);
          if (lIt != origLocalMap.end()) {
            Expression* rep = nullptr;
            if (rec.existingLocal != Index(-1) &&
                rec.existingLocalNeedsNonNull &&
                curr->type.isNonNullable()) {
              Type localType = Type(curr->type.getHeapType(),
                                    Nullable,
                                    curr->type.getExactness());
              rep = Builder(wasm).makeRefAs(
                RefAsNonNull,
                Builder(wasm).makeLocalGet(lIt->second, localType));
            } else {
              rep = Builder(wasm).makeLocalGet(lIt->second, curr->type);
            }
            replaceCurrent(rep);
            return;
          }
        }
        if (auto tIt = needsTeeWrap.find(curr); tIt != needsTeeWrap.end()) {
          replaceCurrent(
            Builder(wasm).makeLocalTee(tIt->second, curr, curr->type));
        }
      }
    } applier(wasm, repeatToOrigIdx, origRecords, origLocalMap, needsTeeWrap);
    applier.walk(func->body);
  }
};

static bool unCseSingleWriteTees(Function* func,
                                 Module& wasm,
                                 PassOptions& options,
                                 bool allowNonTeeAndMultiWrite = false) {
  if (!func->body || func->getNumLocals() == 0) {
    return false;
  }

  ScopedDominatorCSE helper(func, wasm, options);
  bool anyChanged = false;
  std::unordered_set<Index> allUnCsedLocals;

  auto isDuplicableValue = [&](Expression* root,
                               ScopedDominatorCSE::OrigRecord& rec,
                               Index& estBytesOut,
                               LocalSet*& innerTeeOut) -> bool {
    estBytesOut = 0;
    innerTeeOut = nullptr;
    bool ok = true;
    std::function<void(Expression*, bool)> walkVal = [&](Expression* e,
                                                         bool onLeftmostPath) {
      if (!ok || !e) {
        return;
      }
      if (auto* ls = e->dynCast<LocalSet>()) {
        if (e != root && onLeftmostPath && !innerTeeOut && ls->isTee() &&
            ls->value->type.isConcrete() &&
            Type::isSubType(ls->value->type, func->getLocalType(ls->index))) {
          innerTeeOut = ls;
          estBytesOut += 2;
          if (std::find(rec.localsRead.begin(),
                        rec.localsRead.end(),
                        ls->index) == rec.localsRead.end()) {
            rec.localsRead.push_back(ls->index);
          }
          return;
        }
        ok = false;
        return;
      }
      if (e->is<Block>() || e->is<If>() || e->is<Loop>() || e->is<Try>() ||
          e->is<TryTable>() || e->is<Break>() || e->is<BrOn>() ||
          e->is<Switch>() || e->is<Return>() || e->is<Throw>() ||
          e->is<Rethrow>() || e->is<Unreachable>() || e->is<CallRef>() ||
          e->is<CallIndirect>() || e->is<Pop>() || e->is<TupleMake>() ||
          e->is<TupleExtract>()) {
        ok = false;
        return;
      }
      estBytesOut += helper.getShallowEstBytes(e);
      if (auto* lg = e->dynCast<LocalGet>()) {
        if (std::find(rec.localsRead.begin(),
                      rec.localsRead.end(),
                      lg->index) == rec.localsRead.end()) {
          rec.localsRead.push_back(lg->index);
        }
        return;
      }
      if (auto* call = e->dynCast<Call>()) {
        if (call->isReturn) {
          ok = false;
          return;
        }
        auto* target = wasm.getFunctionOrNull(call->target);
        if (!target || !target->effects) {
          ok = false;
          return;
        }
        const auto& tEff = *target->effects;
        bool idempotentInit =
          tEff.isIdempotent && tEff.onlyTouchesInitGlobals && !tEff.throws();
        bool pureReadOnly =
          !tEff.hasNonTrapSideEffects() && !tEff.generative && !tEff.throws();
        if (!idempotentInit && !pureReadOnly) {
          ok = false;
          return;
        }
        if (!options.trapsNeverHappen && tEff.trap) {
          ok = false;
          return;
        }
        if (!idempotentInit) {
          for (Name g : tEff.mutableGlobalsRead) {
            if (std::find(rec.mutableGlobalsRead.begin(),
                          rec.mutableGlobalsRead.end(),
                          g) == rec.mutableGlobalsRead.end()) {
              rec.mutableGlobalsRead.push_back(g);
            }
          }
          if (tEff.readsMutableStruct || tEff.readsSharedMutableStruct) {
            rec.readsMutableStruct = true;
            if (tEff.readsSharedMutableStruct ||
                !helper.collectPureCallStructFieldsRead(
                  target, rec.mutableStructFieldsRead)) {
              rec.readsUnknownMutableStruct = true;
            }
          }
          if (tEff.readsMutableArray || tEff.readsSharedMutableArray) {
            rec.readsMutableArray = true;
          }
          if (tEff.readsMemory || tEff.readsSharedMemory) {
            rec.readsMemory = true;
          }
          if (tEff.readsTable) {
            rec.readsTable = true;
          }
        }
        bool firstChild = onLeftmostPath;
        for (auto* ch : ChildIterator(e)) {
          walkVal(ch, firstChild);
          firstChild = false;
        }
        return;
      }
      ShallowEffectAnalyzer eff(options, wasm, e);
      if (!options.trapsNeverHappen && eff.trap) {
        ok = false;
        return;
      }
      eff.trap = false;
      if (eff.hasNonTrapSideEffects() ||
          Properties::isShallowlyGenerative(e, func, wasm)) {
        ok = false;
        return;
      }
      for (Name g : eff.mutableGlobalsRead) {
        if (std::find(rec.mutableGlobalsRead.begin(),
                      rec.mutableGlobalsRead.end(),
                      g) == rec.mutableGlobalsRead.end()) {
          rec.mutableGlobalsRead.push_back(g);
        }
      }
      if (eff.readsMutableStruct || eff.readsSharedMutableStruct) {
        rec.readsMutableStruct = true;
        if (auto* sg = e->dynCast<StructGet>()) {
          if (!eff.readsSharedMutableStruct && sg->ref->type.isRef()) {
            std::pair<HeapType, Index> key{sg->ref->type.getHeapType(),
                                           sg->index};
            if (std::find(rec.mutableStructFieldsRead.begin(),
                          rec.mutableStructFieldsRead.end(),
                          key) == rec.mutableStructFieldsRead.end()) {
              rec.mutableStructFieldsRead.push_back(key);
            }
          } else {
            rec.readsUnknownMutableStruct = true;
          }
        } else {
          rec.readsUnknownMutableStruct = true;
        }
      }
      if (eff.readsMutableArray || eff.readsSharedMutableArray) {
        rec.readsMutableArray = true;
      }
      if (eff.readsMemory || eff.readsSharedMemory) {
        rec.readsMemory = true;
      }
      if (eff.readsTable) {
        rec.readsTable = true;
      }
      bool firstChild = onLeftmostPath;
      for (auto* ch : ChildIterator(e)) {
        walkVal(ch, firstChild);
        firstChild = false;
      }
    };
    walkVal(root, true);
    return ok;
  };

  for (int iter = 0; iter < 4; iter++) {
    Index numLocals = func->getNumLocals();
    Index varBase = func->getVarIndexBase();
    std::vector<SmallVector<LocalSet*, 2>> writes(numLocals);
    std::vector<SmallVector<LocalGet*, 4>> reads(numLocals);
    std::vector<SmallVector<LocalGet*, 4>> readsAfterLastWrite(numLocals);
    std::vector<Index> teePostOrder(numLocals, 0);
    std::vector<Index> writeLoopDepth(numLocals, 0);
    std::vector<bool> inConcreteParent(numLocals, false);
    std::unordered_set<Index> coupledCallRefLocals;

    Index postCounter = 0;
    Index scanLoopDepth = 0;
    std::vector<Expression*> parentStack;
    std::function<void(Expression*)> scanAST = [&](Expression* curr) {
      if (!curr) {
        return;
      }
      bool isLoop = curr->is<Loop>();
      if (isLoop) {
        scanLoopDepth++;
      }
      parentStack.push_back(curr);
      for (auto* ch : ChildIterator(curr)) {
        scanAST(ch);
      }
      parentStack.pop_back();
      if (isLoop) {
        scanLoopDepth--;
      }
      Index myPost = ++postCounter;
      if (auto* lg = curr->dynCast<LocalGet>()) {
        reads[lg->index].push_back(lg);
        readsAfterLastWrite[lg->index].push_back(lg);
      } else if (auto* ls = curr->dynCast<LocalSet>()) {
        writes[ls->index].push_back(ls);
        readsAfterLastWrite[ls->index].clear();
        teePostOrder[ls->index] = myPost;
        writeLoopDepth[ls->index] = scanLoopDepth;
        if (ls->isTee() && !parentStack.empty() &&
            parentStack.back()->type.isConcrete()) {
          inConcreteParent[ls->index] = true;
        }
      } else if (auto* cr = curr->dynCast<CallRef>()) {
        if (!cr->operands.empty() && cr->target) {
          for (auto* ls : FindAll<LocalSet>(cr->operands[0]).list) {
            if (ls->isTee()) {
              for (auto* lg : FindAll<LocalGet>(cr->target).list) {
                if (lg->index == ls->index) {
                  coupledCallRefLocals.insert(ls->index);
                  break;
                }
              }
            }
          }
          for (auto* lg0 : FindAll<LocalGet>(cr->operands[0]).list) {
            for (auto* lgT : FindAll<LocalGet>(cr->target).list) {
              if (lg0->index == lgT->index) {
                coupledCallRefLocals.insert(lg0->index);
                break;
              }
            }
          }
        }
      }
    };
    scanAST(func->body);

    std::unordered_map<Type, Index> varTypeCounts;
    for (Index v = varBase; v < numLocals; v++) {
      if (!writes[v].empty() || !reads[v].empty()) {
        varTypeCounts[ScopedDominatorCSE::getBinaryVarType(
          func->getLocalType(v))]++;
      }
    }

    std::vector<bool> isCandidate(numLocals, false);
    std::vector<ScopedDominatorCSE::OrigRecord> candRec(numLocals);
    std::vector<Index> candEstBytes(numLocals, 0);
    std::vector<LocalSet*> candInnerTee(numLocals, nullptr);
    bool anyCand = false;

    for (Index x = 0; x < numLocals; x++) {
      if (writes[x].empty()) {
        continue;
      }
      bool isSingleWriteVar = (x >= varBase && writes[x].size() == 1);
      if (!allowNonTeeAndMultiWrite &&
          (!isSingleWriteVar || !writes[x][0]->isTee())) {
        continue;
      }
      if (!isSingleWriteVar && writeLoopDepth[x] > 0) {
        continue;
      }
      const auto& targetReads =
        isSingleWriteVar ? reads[x] : readsAfterLastWrite[x];
      if (targetReads.size() > 3) {
        continue;
      }
      LocalSet* wx = writes[x].back();
      Expression* V = wx->value;
      if (!V->type.isConcrete() ||
          !Type::isSubType(V->type, func->getLocalType(x))) {
        continue;
      }
      if (coupledCallRefLocals.contains(x) && !V->is<LocalGet>()) {
        continue;
      }
      if (targetReads.empty()) {
        if (isSingleWriteVar && wx->isTee()) {
          isCandidate[x] = true;
          anyCand = true;
        }
        continue;
      }
      candRec[x].orig = V;
      Index estV = 0;
      LocalSet* innerTee = nullptr;
      if (!isDuplicableValue(V, candRec[x], estV, innerTee)) {
        continue;
      }
      if (innerTee &&
          (!allowNonTeeAndMultiWrite ||
           EffectAnalyzer(options, wasm, innerTee->value)
             .localsWritten.contains(x))) {
        continue;
      }
      if (std::find(candRec[x].localsRead.begin(),
                    candRec[x].localsRead.end(),
                    x) != candRec[x].localsRead.end()) {
        continue;
      }
      candEstBytes[x] = estV;
      candInnerTee[x] = innerTee;
      isCandidate[x] = true;
      anyCand = true;
    }

    if (!anyCand) {
      break;
    }

    using TeeMap = std::unordered_map<Index, Index>;
    TeeMap activeTees;
    Expression* lastCondExpr = nullptr;
    TeeMap lastCondTrueMap;
    TeeMap lastCondFalseMap;
    bool hasLastBrExitMap = false;
    TeeMap lastBrExitMap;
    std::vector<Index> validReads(numLocals, 0);
    Index currentLoopDepth = 0;

    auto invalidateActiveByLocalWrite = [&](Index localIdx) {
      for (auto it = activeTees.begin(); it != activeTees.end();) {
        const auto& rec = candRec[it->first];
        if (std::find(rec.localsRead.begin(),
                      rec.localsRead.end(),
                      localIdx) != rec.localsRead.end()) {
          it = activeTees.erase(it);
        } else {
          ++it;
        }
      }
    };

    auto invalidateActiveByEffects = [&](const EffectAnalyzer& eff,
                                         Expression* killerExpr = nullptr) {
      if (!eff.hasNonTrapSideEffects()) {
        return;
      }
      for (auto it = activeTees.begin(); it != activeTees.end();) {
        if (helper.isKilledByEffects(candRec[it->first], eff, killerExpr)) {
          it = activeTees.erase(it);
        } else {
          ++it;
        }
      }
    };

    std::function<void(Expression*)> walkDom = [&](Expression* curr) {
      lastCondExpr = nullptr;
      if (!curr) {
        return;
      }
      if (auto* block = curr->dynCast<Block>()) {
        Index firstBranchIdx = block->list.size();
        if (block->name.is()) {
          for (Index i = 0; i < block->list.size(); i++) {
            if (SubtreeTargetsName(block->list[i], block->name)) {
              firstBranchIdx = i;
              break;
            }
          }
        }
        TeeMap savedExitMap;
        bool capturedAfterDirectBr = false;
        for (Index i = 0; i < block->list.size(); i++) {
          if (i == firstBranchIdx) {
            bool isDirectCondBr = false;
            Expression* brStmt = block->list[i];
            if (auto* d = brStmt->dynCast<Drop>()) {
              brStmt = d->value;
            }
            if (auto* br = brStmt->dynCast<Break>()) {
              isDirectCondBr = br->name == block->name && br->condition &&
                               !SubtreeTargetsName(br->value, block->name) &&
                               !SubtreeTargetsName(br->condition, block->name);
            } else if (auto* bron = brStmt->dynCast<BrOn>()) {
              isDirectCondBr = bron->name == block->name &&
                               !SubtreeTargetsName(bron->ref, block->name) &&
                               !SubtreeTargetsName(bron->desc, block->name);
            }
            if (!isDirectCondBr) {
              savedExitMap = activeTees;
            } else {
              capturedAfterDirectBr = true;
            }
          }
          hasLastBrExitMap = false;
          walkDom(block->list[i]);
          if (i == firstBranchIdx && capturedAfterDirectBr) {
            savedExitMap =
              hasLastBrExitMap ? std::move(lastBrExitMap) : activeTees;
            hasLastBrExitMap = false;
          }
        }
        if (block->type == Type::unreachable) {
          activeTees.clear();
          lastCondExpr = nullptr;
        } else if (firstBranchIdx < block->list.size()) {
          bool onlyExitsViaFirstDirectBr =
            capturedAfterDirectBr && firstBranchIdx + 1 < block->list.size();
          bool tailDoesNotReturn = false;
          if (onlyExitsViaFirstDirectBr) {
            for (Index i = firstBranchIdx + 1; i < block->list.size(); i++) {
              if (SubtreeTargetsName(block->list[i], block->name)) {
                onlyExitsViaFirstDirectBr = false;
                break;
              }
              if (helper.doesNotReturn(block->list[i])) {
                tailDoesNotReturn = true;
              }
            }
          }
          activeTees = std::move(savedExitMap);
          if (!(onlyExitsViaFirstDirectBr && tailDoesNotReturn)) {
            for (Index i = firstBranchIdx; i < block->list.size(); i++) {
              if (i == firstBranchIdx && capturedAfterDirectBr) {
                continue;
              }
              EffectAnalyzer eff(options, wasm, block->list[i]);
              eff.trap = false;
              invalidateActiveByEffects(eff, block->list[i]);
            }
          }
          lastCondExpr = nullptr;
        } else if (!block->list.empty() && lastCondExpr == block->list.back()) {
          lastCondExpr = block;
        } else {
          lastCondExpr = nullptr;
        }
        return;
      }
      if (auto* iff = curr->dynCast<If>()) {
        walkDom(iff->condition);
        auto afterCond = activeTees;
        TeeMap condTrueMap = (lastCondExpr == iff->condition)
                               ? std::move(lastCondTrueMap)
                               : afterCond;
        TeeMap condFalseMap = (lastCondExpr == iff->condition)
                                ? std::move(lastCondFalseMap)
                                : afterCond;
        lastCondExpr = nullptr;

        activeTees = std::move(condTrueMap);
        walkDom(iff->ifTrue);
        TeeMap thenTrueMap =
          (lastCondExpr == iff->ifTrue) ? std::move(lastCondTrueMap) : activeTees;
        TeeMap thenFalseMap = (lastCondExpr == iff->ifTrue)
                                ? std::move(lastCondFalseMap)
                                : activeTees;
        TeeMap afterTrue = std::move(activeTees);
        lastCondExpr = nullptr;

        TeeMap afterFalse;
        TeeMap elseTrueMap;
        TeeMap elseFalseMap;
        if (iff->ifFalse) {
          activeTees = std::move(condFalseMap);
          walkDom(iff->ifFalse);
          elseTrueMap = (lastCondExpr == iff->ifFalse)
                          ? std::move(lastCondTrueMap)
                          : activeTees;
          elseFalseMap = (lastCondExpr == iff->ifFalse)
                           ? std::move(lastCondFalseMap)
                           : activeTees;
          afterFalse = std::move(activeTees);
          lastCondExpr = nullptr;
        } else {
          elseTrueMap = condFalseMap;
          elseFalseMap = condFalseMap;
          afterFalse = std::move(condFalseMap);
        }

        bool thenNR = helper.doesNotReturn(iff->ifTrue);
        bool elseNR = iff->ifFalse && helper.doesNotReturn(iff->ifFalse);

        if (iff->type == Type::unreachable || (thenNR && elseNR)) {
          activeTees.clear();
          return;
        }

        if (thenNR && !elseNR) {
          activeTees = std::move(afterFalse);
        } else if (elseNR && !thenNR) {
          activeTees = std::move(afterTrue);
        } else {
          activeTees = std::move(afterCond);
          EffectAnalyzer thenEff(options, wasm, iff->ifTrue);
          thenEff.trap = false;
          invalidateActiveByEffects(thenEff, iff->ifTrue);
          if (iff->ifFalse) {
            EffectAnalyzer elseEff(options, wasm, iff->ifFalse);
            elseEff.trap = false;
            invalidateActiveByEffects(elseEff, iff->ifFalse);
          }
        }

        if (iff->type == Type::i32 && iff->ifFalse) {
          bool thenAlwaysZero =
            thenNR || ScopedDominatorCSE::isConstZeroI32(iff->ifTrue);
          bool elseAlwaysZero =
            elseNR || ScopedDominatorCSE::isConstZeroI32(iff->ifFalse);
          bool thenAlwaysNonZero =
            thenNR || ScopedDominatorCSE::isConstNonZeroI32(iff->ifTrue);
          bool elseAlwaysNonZero =
            elseNR || ScopedDominatorCSE::isConstNonZeroI32(iff->ifFalse);

          if (elseAlwaysZero && !thenAlwaysZero) {
            lastCondTrueMap = std::move(thenTrueMap);
          } else if (thenAlwaysZero && !elseAlwaysZero) {
            lastCondTrueMap = std::move(elseTrueMap);
          } else {
            lastCondTrueMap = activeTees;
          }

          if (thenAlwaysNonZero && !elseAlwaysNonZero) {
            lastCondFalseMap = std::move(elseFalseMap);
          } else if (elseAlwaysNonZero && !thenAlwaysNonZero) {
            lastCondFalseMap = std::move(thenFalseMap);
          } else {
            lastCondFalseMap = activeTees;
          }
          lastCondExpr = iff;
        }
        return;
      }
      if (auto* loop = curr->dynCast<Loop>()) {
        EffectAnalyzer loopEff(options, wasm, loop->body);
        loopEff.trap = false;
        invalidateActiveByEffects(loopEff, loop->body);
        auto beforeLoop = activeTees;
        currentLoopDepth++;
        walkDom(loop->body);
        currentLoopDepth--;
        if (loop->type == Type::unreachable) {
          activeTees.clear();
        } else {
          activeTees = std::move(beforeLoop);
        }
        lastCondExpr = nullptr;
        return;
      }
      if (auto* tryy = curr->dynCast<Try>()) {
        auto beforeTry = activeTees;
        walkDom(tryy->body);
        EffectAnalyzer bodyEff(options, wasm, tryy->body);
        bodyEff.trap = false;
        for (auto* cb : tryy->catchBodies) {
          activeTees = beforeTry;
          invalidateActiveByEffects(bodyEff, tryy->body);
          walkDom(cb);
        }
        if (tryy->type == Type::unreachable) {
          activeTees.clear();
        } else {
          activeTees = std::move(beforeTry);
          EffectAnalyzer tryEff(options, wasm, tryy);
          tryEff.trap = false;
          invalidateActiveByEffects(tryEff, tryy);
        }
        lastCondExpr = nullptr;
        return;
      }
      if (auto* tt = curr->dynCast<TryTable>()) {
        auto beforeTry = activeTees;
        walkDom(tt->body);
        if (tt->type == Type::unreachable) {
          activeTees.clear();
        } else {
          activeTees = std::move(beforeTry);
          EffectAnalyzer ttEff(options, wasm, tt);
          ttEff.trap = false;
          invalidateActiveByEffects(ttEff, tt);
        }
        lastCondExpr = nullptr;
        return;
      }
      if (auto* br = curr->dynCast<Break>()) {
        if (br->value) {
          walkDom(br->value);
        }
        if (br->condition) {
          walkDom(br->condition);
          if (lastCondExpr == br->condition) {
            hasLastBrExitMap = true;
            lastBrExitMap = activeTees;
            activeTees = std::move(lastCondFalseMap);
            lastCondExpr = nullptr;
          }
        }
        if (!br->condition) {
          activeTees.clear();
        }
        lastCondExpr = nullptr;
        return;
      }
      if (auto* bron = curr->dynCast<BrOn>()) {
        walkDom(bron->ref);
        if (bron->desc) {
          walkDom(bron->desc);
        }
        lastCondExpr = nullptr;
        return;
      }
      if (auto* sw = curr->dynCast<Switch>()) {
        if (sw->value) {
          walkDom(sw->value);
        }
        walkDom(sw->condition);
        activeTees.clear();
        lastCondExpr = nullptr;
        return;
      }
      if (auto* ret = curr->dynCast<Return>()) {
        if (ret->value) {
          walkDom(ret->value);
        }
        activeTees.clear();
        lastCondExpr = nullptr;
        return;
      }
      if (auto* th = curr->dynCast<Throw>()) {
        for (auto* op : th->operands) {
          walkDom(op);
        }
        activeTees.clear();
        lastCondExpr = nullptr;
        return;
      }
      if (curr->is<Rethrow>() || curr->is<Unreachable>()) {
        activeTees.clear();
        return;
      }
      if (auto* lg = curr->dynCast<LocalGet>()) {
        auto it = activeTees.find(lg->index);
        if (it != activeTees.end()) {
          Expression* origV = candRec[lg->index].orig;
          bool isLoopSafeLeaf = origV &&
                                (origV->is<Const>() || origV->is<RefNull>() ||
                                 origV->is<RefFunc>() ||
                                 origV->is<LocalGet>() ||
                                 origV->is<GlobalGet>());
          if (it->second == currentLoopDepth ||
              (isLoopSafeLeaf && it->second < currentLoopDepth)) {
            validReads[lg->index]++;
          }
        }
        return;
      }
      if (auto* ls = curr->dynCast<LocalSet>()) {
        walkDom(ls->value);
        lastCondExpr = nullptr;
        invalidateActiveByLocalWrite(ls->index);
        activeTees.erase(ls->index);
        if (isCandidate[ls->index] && writes[ls->index].back() == ls) {
          activeTees[ls->index] = currentLoopDepth;
        }
        return;
      }

      for (auto* child : ChildIterator(curr)) {
        walkDom(child);
      }
      if (auto* u = curr->dynCast<Unary>()) {
        if (u->op == EqZInt32 && lastCondExpr == u->value) {
          std::swap(lastCondTrueMap, lastCondFalseMap);
          lastCondExpr = u;
        } else {
          lastCondExpr = nullptr;
        }
      } else {
        lastCondExpr = nullptr;
      }

      bool idempotentCall = isIdempotent(curr, wasm);
      ShallowEffectAnalyzer shallow(options, wasm, curr);
      shallow.trap = false;
      if (idempotentCall) {
        if (auto* target = wasm.getFunctionOrNull(curr->cast<Call>()->target)) {
          if (target->effects && target->effects->onlyTouchesInitGlobals) {
            shallow.calls = false;
            shallow.writesArray = false;
            shallow.writesSharedArray = false;
          }
        }
      }
      if (shallow.hasNonTrapSideEffects()) {
        invalidateActiveByEffects(shallow, curr);
      }
      if (shallow.branchesOut || curr->type == Type::unreachable ||
          (curr->is<Call>() && helper.doesNotReturn(curr))) {
        activeTees.clear();
      }
    };

    walkDom(func->body);

    std::vector<Index> domCandidates;
    for (Index x = 0; x < numLocals; x++) {
      if (!isCandidate[x]) {
        continue;
      }
      bool isSingleWriteVar = (x >= varBase && writes[x].size() == 1);
      const auto& targetReads =
        isSingleWriteVar ? reads[x] : readsAfterLastWrite[x];
      if (validReads[x] == targetReads.size()) {
        domCandidates.push_back(x);
      }
    }
    if (domCandidates.empty()) {
      break;
    }

    // Evaluate inner/later writes first (descending post-order of wx).
    std::sort(domCandidates.begin(),
              domCandidates.end(),
              [&](Index a, Index b) {
                return teePostOrder[a] > teePostOrder[b];
              });

    std::vector<Index> currentReadCount(numLocals, 0);
    for (Index i = 0; i < numLocals; i++) {
      currentReadCount[i] = reads[i].size();
    }

    std::unordered_set<Index> selectedLocals;
    std::unordered_set<Index> readBySelectedLocals;
    std::unordered_map<LocalSet*, Expression*> selectedWriteNodes;
    std::unordered_map<LocalSet*, LocalSet*> selectedWriteInnerTee;
    std::unordered_map<LocalGet*, Expression*> selectedGetNodes;

    auto makeNormCopy = [&](Expression* V, LocalSet* innerTee) -> Expression* {
      if (!innerTee) {
        return V;
      }
      Expression* copyV = ExpressionManipulator::copy(V, wasm);
      struct TeeToGet : public PostWalker<TeeToGet> {
        Module& wasm;
        Function* func;
        Index targetIdx;
        bool done = false;
        TeeToGet(Module& wasm, Function* func, Index targetIdx)
          : wasm(wasm), func(func), targetIdx(targetIdx) {}
        void visitLocalSet(LocalSet* curr) {
          if (!done && curr->isTee() && curr->index == targetIdx) {
            done = true;
            replaceCurrent(Builder(wasm).makeLocalGet(
              targetIdx, func->getLocalType(targetIdx)));
          }
        }
      } converter(wasm, func, innerTee->index);
      converter.walk(copyV);
      return copyV;
    };

    for (Index x : domCandidates) {
      bool isSingleWriteVar = (x >= varBase && writes[x].size() == 1);
      LocalSet* wx = writes[x].back();
      LocalSet* innerTee = candInnerTee[x];
      const auto& targetReads =
        isSingleWriteVar ? reads[x] : readsAfterLastWrite[x];
      Index k = isSingleWriteVar ? currentReadCount[x] : targetReads.size();
      if (k > 3) {
        continue;
      }
      if ((!isSingleWriteVar || innerTee) && k == 0) {
        continue;
      }
      if (selectedLocals.contains(x) || readBySelectedLocals.contains(x)) {
        continue;
      }
      if (k > 0) {
        bool readsSelected = false;
        for (Index r : candRec[x].localsRead) {
          if (selectedLocals.contains(r)) {
            readsSelected = true;
            break;
          }
        }
        if (readsSelected) {
          continue;
        }
        Type xType = func->getLocalType(x);
        Type binType = ScopedDominatorCSE::getBinaryVarType(xType);
        Index headerBytes = 0;
        if (isSingleWriteVar && varTypeCounts[binType] == 1) {
          if (xType.isRef() && !xType.getHeapType().isBasic()) {
            headerBytes = 4;
          } else if (options.shrinkLevel >= 2) {
            bool sharedWithParam = false;
            for (Index p = 0; p < varBase; p++) {
              if (ScopedDominatorCSE::getBinaryVarType(
                    func->getLocalType(p)) == binType) {
                sharedWithParam = true;
                break;
              }
            }
            if (!sharedWithParam) {
              headerBytes = 2;
            }
          }
        }
        Index estV = candEstBytes[x];
        if (wx->isTee()) {
          Index lhs = estV > 2 ? k * (estV - 2) : 0;
          Index rhs = 2 + headerBytes;
          if (lhs > rhs) {
            continue;
          }
          if (lhs == rhs && !inConcreteParent[x]) {
            continue;
          }
        } else {
          Index lhs = (k - 1) * estV;
          Index rhs = (innerTee ? 0 : 2) +
                      2 * (options.trapsNeverHappen ? k : (k - 1)) +
                      headerBytes;
          if (lhs > rhs) {
            continue;
          }
          if (lhs == rhs && !inConcreteParent[x]) {
            continue;
          }
        }
      }

      selectedLocals.insert(x);
      for (Index r : candRec[x].localsRead) {
        readBySelectedLocals.insert(r);
      }
      Expression* V = wx->value;
      Expression* normV = makeNormCopy(V, innerTee);
      selectedWriteNodes[wx] = V;
      if (innerTee && !wx->isTee()) {
        selectedWriteInnerTee[wx] = innerTee;
      }
      for (LocalGet* lg : targetReads) {
        selectedGetNodes[lg] = normV;
      }
      if (isSingleWriteVar) {
        allUnCsedLocals.insert(x);
        Type binType =
          ScopedDominatorCSE::getBinaryVarType(func->getLocalType(x));
        if (varTypeCounts[binType] > 0) {
          varTypeCounts[binType]--;
        }
      }
      if (k > 0) {
        for (auto* lg : FindAll<LocalGet>(normV).list) {
          if (innerTee && lg->index == innerTee->index) {
            currentReadCount[lg->index] += k;
          } else {
            currentReadCount[lg->index] += (wx->isTee() ? k : (k - 1));
          }
        }
      }
    }

    if (selectedWriteNodes.empty()) {
      break;
    }

    struct UnCseReplacer : public PostWalker<UnCseReplacer> {
      Module& wasm;
      const std::unordered_map<LocalSet*, Expression*>& selectedWriteNodes;
      const std::unordered_map<LocalSet*, LocalSet*>& selectedWriteInnerTee;
      const std::unordered_map<LocalGet*, Expression*>& selectedGetNodes;

      UnCseReplacer(
        Module& wasm,
        const std::unordered_map<LocalSet*, Expression*>& selectedWriteNodes,
        const std::unordered_map<LocalSet*, LocalSet*>& selectedWriteInnerTee,
        const std::unordered_map<LocalGet*, Expression*>& selectedGetNodes)
        : wasm(wasm), selectedWriteNodes(selectedWriteNodes),
          selectedWriteInnerTee(selectedWriteInnerTee),
          selectedGetNodes(selectedGetNodes) {}

      void visitLocalGet(LocalGet* curr) {
        auto it = selectedGetNodes.find(curr);
        if (it != selectedGetNodes.end()) {
          replaceCurrent(ExpressionManipulator::copy(it->second, wasm));
        }
      }

      void visitLocalSet(LocalSet* curr) {
        auto it = selectedWriteNodes.find(curr);
        if (it != selectedWriteNodes.end()) {
          if (curr->isTee()) {
            replaceCurrent(curr->value);
          } else {
            auto tIt = selectedWriteInnerTee.find(curr);
            if (tIt != selectedWriteInnerTee.end()) {
              LocalSet* innerTee = tIt->second;
              innerTee->makeSet();
              replaceCurrent(innerTee);
            } else {
              ExpressionManipulator::nop(curr);
            }
          }
        }
      }

      void visitBlock(Block* curr) {
        curr->list.erase(
          std::remove_if(curr->list.begin(),
                         curr->list.end(),
                         [](Expression* e) { return e->is<Nop>(); }),
          curr->list.end());
      }
    } replacer(
      wasm, selectedWriteNodes, selectedWriteInnerTee, selectedGetNodes);
    replacer.walk(func->body);
    anyChanged = true;
  }

  if (anyChanged && !allUnCsedLocals.empty()) {
    // Strip the un-CSEd locals from func->vars and reindex local references.
    Index numParams = func->getNumParams();
    Index oldNumLocals = func->getNumLocals();
    std::vector<Index> oldToNew(oldNumLocals, 0);
    std::vector<Index> newToOld;
    newToOld.reserve(oldNumLocals);
    for (Index i = 0; i < numParams; i++) {
      oldToNew[i] = i;
      newToOld.push_back(i);
    }
    std::vector<Type> newVars;
    newVars.reserve(func->vars.size());
    for (Index v = numParams; v < oldNumLocals; v++) {
      if (allUnCsedLocals.contains(v)) {
        continue;
      }
      Index newIdx = numParams + newVars.size();
      oldToNew[v] = newIdx;
      newToOld.push_back(v);
      newVars.push_back(func->vars[v - numParams]);
    }
    func->vars = std::move(newVars);

    struct LocalReindexer : public PostWalker<LocalReindexer> {
      const std::vector<Index>& oldToNew;
      LocalReindexer(const std::vector<Index>& oldToNew)
        : oldToNew(oldToNew) {}
      void visitLocalGet(LocalGet* curr) {
        curr->index = oldToNew[curr->index];
      }
      void visitLocalSet(LocalSet* curr) {
        curr->index = oldToNew[curr->index];
      }
    } reindexer(oldToNew);
    reindexer.walk(func->body);

    if (!func->localNames.empty()) {
      auto oldLocalNames = std::move(func->localNames);
      func->localNames.clear();
      func->localIndices.clear();
      for (Index newIdx = 0; newIdx < newToOld.size(); newIdx++) {
        auto it = oldLocalNames.find(newToOld[newIdx]);
        if (it != oldLocalNames.end()) {
          func->localNames[newIdx] = it->second;
          func->localIndices[it->second] = newIdx;
        }
      }
    }
  }

  return anyChanged;
}

struct OnceCallOptimizer {
  Module& wasm;
  const PassOptions& options;

  std::unordered_set<Name> onceFuncs;
  std::unordered_map<Name, std::unordered_set<Name>> impliedOnce;
  std::unordered_map<Name, std::unordered_set<Name>> onceWrittenGlobals;

  std::unordered_set<Name> pureLazyOnce;
  std::unordered_set<Name> safePureOnce;
  std::unordered_map<Name, std::unordered_set<Name>> pureTransitiveGlobals;
  std::unordered_set<Name> allSafePureGlobals;

  std::unordered_set<Name> nonCodeGlobals;
  std::unordered_set<Name> resetZeroGlobals;
  std::unordered_map<Name, std::unordered_set<Name>> funcGets;
  std::unordered_map<Name, std::unordered_set<Name>> funcSets;
  std::unordered_map<Name, std::unordered_map<Name, Index>> funcSetCounts;
  std::unordered_map<Name, std::unordered_set<Name>> funcCalls;
  std::unordered_map<Name, std::unordered_set<Name>> globalWriterFuncs;
  std::unordered_map<Name, std::unordered_set<Name>> globalReaderFuncs;

  OnceCallOptimizer(Module& wasm, const PassOptions& options)
    : wasm(wasm), options(options) {}

  static bool extractGuardGlobal(Expression* cond, Name& gOut) {
    if (auto* r = cond->dynCast<RefIsNull>()) {
      if (auto* gg = r->value->dynCast<GlobalGet>()) {
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
  }

  static bool isZeroOrNullExpr(Expression* e) {
    if (!e) {
      return false;
    }
    while (auto* ls = e->dynCast<LocalSet>()) {
      e = ls->value;
    }
    if (e->is<RefNull>()) {
      return true;
    }
    if (auto* c = e->dynCast<Const>()) {
      return c->value.isZero();
    }
    return false;
  }

  static bool doesNotReturn(Expression* e) {
    if (!e) {
      return false;
    }
    if (e->type == Type::unreachable || e->is<Return>() || e->is<Throw>() ||
        e->is<Rethrow>() || e->is<Unreachable>() || e->is<Switch>()) {
      return true;
    }
    if (auto* br = e->dynCast<Break>()) {
      return !br->condition;
    }
    return false;
  }

  bool isValidGuardGlobal(Name gName) const {
    auto* g = wasm.getGlobalOrNull(gName);
    if (!g || g->imported() || !g->mutable_ || !g->init) {
      return false;
    }
    if (nonCodeGlobals.contains(gName) || resetZeroGlobals.contains(gName)) {
      return false;
    }
    return isZeroOrNullExpr(g->init);
  }

  static bool itemMayExitBlockOrFunc(Expression* item, Name blockName) {
    if (!FindAll<Return>(item).list.empty()) {
      return true;
    }
    if (blockName.is()) {
      for (auto* br : FindAll<Break>(item).list) {
        if (br->name == blockName) {
          return true;
        }
      }
      for (auto* sw : FindAll<Switch>(item).list) {
        if (sw->default_ == blockName) {
          return true;
        }
        for (auto t : sw->targets) {
          if (t == blockName) {
            return true;
          }
        }
      }
    }
    return false;
  }

  bool topLevelSetsGlobal(Expression* body, Name gName, int depth = 0) const {
    if (!body || depth > 1) {
      return false;
    }
    if (auto* gs = body->dynCast<GlobalSet>()) {
      return gs->name == gName;
    }
    if (auto* c = body->dynCast<Call>()) {
      if (auto* callee = wasm.getFunctionOrNull(c->target)) {
        if (!callee->imported() && callee->body) {
          return topLevelSetsGlobal(callee->body, gName, depth + 1);
        }
      }
      return false;
    }
    if (auto* b = body->dynCast<Block>()) {
      for (auto* item : b->list) {
        if (topLevelSetsGlobal(item, gName, depth)) {
          return true;
        }
        if (itemMayExitBlockOrFunc(item, b->name)) {
          return false;
        }
      }
    }
    return false;
  }

  bool isPureInitValue(Expression* val) const {
    struct Validator : public PostWalker<Validator> {
      const OnceCallOptimizer& parent;
      bool ok = true;
      Validator(const OnceCallOptimizer& parent) : parent(parent) {}
      void visitExpression(Expression* curr) {
        if (!ok) {
          return;
        }
        if (auto* c = curr->dynCast<Call>()) {
          if (c->isReturn) {
            ok = false;
            return;
          }
          if (c->type == Type::none) {
            if (!c->operands.empty() ||
                !parent.pureLazyOnce.contains(c->target)) {
              ok = false;
            }
            return;
          }
          auto* targetFn = parent.wasm.getFunctionOrNull(c->target);
          if (!targetFn || targetFn->imported()) {
            ok = false;
            return;
          }
          auto sIt = parent.funcSets.find(c->target);
          if (sIt != parent.funcSets.end()) {
            for (Name hg : sIt->second) {
              auto wIt = parent.globalWriterFuncs.find(hg);
              auto rIt = parent.globalReaderFuncs.find(hg);
              if ((wIt != parent.globalWriterFuncs.end() &&
                   wIt->second.size() > 1) ||
                  (rIt != parent.globalReaderFuncs.end() &&
                   (rIt->second.size() > 1 ||
                    (rIt->second.size() == 1 &&
                     !rIt->second.contains(c->target))))) {
                ok = false;
                return;
              }
            }
          }
          return;
        }
        if (curr->is<Const>() || curr->is<RefNull>() || curr->is<RefFunc>() ||
            curr->is<RefI31>() || curr->is<StringConst>() ||
            curr->is<GlobalGet>() || curr->is<StructNew>() ||
            curr->is<ArrayNew>() || curr->is<ArrayNewData>() ||
            curr->is<ArrayNewElem>() || curr->is<ArrayNewFixed>() ||
            curr->is<StructGet>() || curr->is<ArrayGet>() ||
            curr->is<ArrayLen>() || curr->is<RefCast>() ||
            curr->is<RefTest>() || curr->is<RefAs>() ||
            curr->is<RefIsNull>() || curr->is<RefEq>() || curr->is<I31Get>() ||
            curr->is<Unary>() || curr->is<Binary>() || curr->is<Select>() ||
            curr->is<Block>()) {
          return;
        }
        ok = false;
      }
    } validator(*this);
    validator.walk(val);
    return validator.ok;
  }

  void analyze() {
    for (auto& exp : wasm.exports) {
      if (exp->kind == ExternalKind::Global) {
        nonCodeGlobals.insert(*exp->getInternalName());
      }
    }
    for (auto& g : wasm.globals) {
      if (!g->imported() && g->init) {
        for (auto* gg : FindAll<GlobalGet>(g->init).list) {
          nonCodeGlobals.insert(gg->name);
        }
      }
    }
    ModuleUtils::iterActiveElementSegments(wasm, [&](ElementSegment* seg) {
      for (auto* gg : FindAll<GlobalGet>(seg->offset).list) {
        nonCodeGlobals.insert(gg->name);
      }
    });
    ModuleUtils::iterActiveDataSegments(wasm, [&](DataSegment* seg) {
      for (auto* gg : FindAll<GlobalGet>(seg->offset).list) {
        nonCodeGlobals.insert(gg->name);
      }
    });

    struct FuncSummary {
      std::unordered_set<Name> gets;
      std::unordered_set<Name> sets;
      std::unordered_map<Name, Index> setCounts;
      std::unordered_set<Name> calls;
      std::unordered_set<Name> resetZero;
    };
    ModuleUtils::ParallelFunctionAnalysis<FuncSummary> analysis(
      wasm, [&](Function* func, FuncSummary& summary) {
        if (func->imported() || !func->body) {
          return;
        }
        struct Scanner : public PostWalker<Scanner> {
          FuncSummary& summary;
          Scanner(FuncSummary& summary) : summary(summary) {}
          void visitGlobalGet(GlobalGet* curr) {
            summary.gets.insert(curr->name);
          }
          void visitGlobalSet(GlobalSet* curr) {
            summary.sets.insert(curr->name);
            summary.setCounts[curr->name]++;
            if (OnceCallOptimizer::isZeroOrNullExpr(curr->value)) {
              summary.resetZero.insert(curr->name);
            }
          }
          void visitCall(Call* curr) { summary.calls.insert(curr->target); }
        } scanner(summary);
        scanner.walk(func->body);
      });

    for (auto& [func, summary] : analysis.map) {
      if (func->imported()) {
        continue;
      }
      Name fn = func->name;
      for (Name g : summary.gets) {
        globalReaderFuncs[g].insert(fn);
      }
      for (Name g : summary.sets) {
        globalWriterFuncs[g].insert(fn);
      }
      for (Name g : summary.resetZero) {
        resetZeroGlobals.insert(g);
      }
      funcGets[fn] = std::move(summary.gets);
      funcSets[fn] = std::move(summary.sets);
      funcSetCounts[fn] = std::move(summary.setCounts);
      funcCalls[fn] = std::move(summary.calls);
    }

    // 1. Discover all void 0-param idempotent once-functions (fixed point).
    std::unordered_map<Name, std::vector<Name>> directImplied;
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto& fPtr : wasm.functions) {
        Function* f = fPtr.get();
        if (f->imported() || !f->body || f->getNumParams() != 0 ||
            f->getResults() != Type::none || onceFuncs.contains(f->name)) {
          continue;
        }
        // Shape B: (block $label (br_if $label (global.get $g)) (global.set $g
        // 1) ...)
        if (auto* b = f->body->dynCast<Block>()) {
          if (b->list.size() >= 2) {
            Name guardG;
            if (auto* br = b->list[0]->dynCast<Break>()) {
              if (br->condition && !br->value && br->name == b->name) {
                if (auto* gg = br->condition->dynCast<GlobalGet>()) {
                  guardG = gg->name;
                }
              }
            } else if (auto* iff = b->list[0]->dynCast<If>()) {
              if (!iff->ifFalse) {
                if (auto* gg = iff->condition->dynCast<GlobalGet>()) {
                  bool exits = iff->ifTrue->is<Return>();
                  if (!exits) {
                    if (auto* br = iff->ifTrue->dynCast<Break>()) {
                      exits = !br->condition && br->name == b->name;
                    }
                  }
                  if (exits) {
                    guardG = gg->name;
                  }
                }
              }
            }
            if (guardG.is() && isValidGuardGlobal(guardG)) {
              auto* gs = b->list[1]->dynCast<GlobalSet>();
              if (gs && gs->name == guardG) {
                onceFuncs.insert(f->name);
                auto& imp = directImplied[f->name];
                for (Index i = 2; i < b->list.size(); i++) {
                  if (auto* c = b->list[i]->dynCast<Call>()) {
                    imp.push_back(c->target);
                  }
                  if (itemMayExitBlockOrFunc(b->list[i], b->name)) {
                    break;
                  }
                }
                changed = true;
                continue;
              }
            }
          }
        }

        // Shape A: (if (i32.eqz/ref.is_null (global.get $g)) (then ...))
        // optionally preceded/followed by calls to other onceFuncs.
        If* mainIf = f->body->dynCast<If>();
        std::vector<Name> outerOnceCalls;
        if (!mainIf) {
          if (auto* b = f->body->dynCast<Block>()) {
            bool allValid = true;
            for (auto* item : b->list) {
              if (auto* iff = item->dynCast<If>()) {
                if (mainIf) {
                  allValid = false;
                  break;
                }
                mainIf = iff;
              } else if (auto* c = item->dynCast<Call>()) {
                if (onceFuncs.contains(c->target)) {
                  outerOnceCalls.push_back(c->target);
                } else {
                  allValid = false;
                  break;
                }
              } else {
                allValid = false;
                break;
              }
            }
            if (!allValid) {
              mainIf = nullptr;
            }
          }
        }
        if (mainIf && !mainIf->ifFalse) {
          Name guardG;
          if (extractGuardGlobal(mainIf->condition, guardG) &&
              isValidGuardGlobal(guardG) &&
              topLevelSetsGlobal(mainIf->ifTrue, guardG)) {
            onceFuncs.insert(f->name);
            auto& imp = directImplied[f->name];
            for (Name oc : outerOnceCalls) {
              imp.push_back(oc);
            }
            if (auto* c = mainIf->ifTrue->dynCast<Call>()) {
              imp.push_back(c->target);
            } else if (auto* tb = mainIf->ifTrue->dynCast<Block>()) {
              for (auto* item : tb->list) {
                if (auto* c = item->dynCast<Call>()) {
                  imp.push_back(c->target);
                }
                if (itemMayExitBlockOrFunc(item, tb->name)) {
                  break;
                }
              }
            }
            changed = true;
          }
        }
      }
    }

    if (onceFuncs.empty()) {
      return;
    }

    // 2. Compute transitive impliedOnce[f] and onceWrittenGlobals[f].
    for (Name fn : onceFuncs) {
      auto& impSet = impliedOnce[fn];
      impSet.insert(fn);
      std::vector<Name> worklist = {fn};
      while (!worklist.empty()) {
        Name cur = worklist.back();
        worklist.pop_back();
        auto dIt = directImplied.find(cur);
        if (dIt != directImplied.end()) {
          for (Name nxt : dIt->second) {
            if (onceFuncs.contains(nxt) && impSet.insert(nxt).second) {
              worklist.push_back(nxt);
            }
          }
        }
      }

      auto& written = onceWrittenGlobals[fn];
      std::unordered_set<Name> visitedCallees = {fn};
      std::vector<Name> callWorklist = {fn};
      while (!callWorklist.empty()) {
        Name cur = callWorklist.back();
        callWorklist.pop_back();
        auto sIt = funcSets.find(cur);
        if (sIt != funcSets.end()) {
          written.insert(sIt->second.begin(), sIt->second.end());
        }
        auto cIt = funcCalls.find(cur);
        if (cIt != funcCalls.end()) {
          for (Name callee : cIt->second) {
            if (visitedCallees.insert(callee).second) {
              callWorklist.push_back(callee);
            }
          }
        }
      }
    }

    // 3. Discover pureLazyOnce (fixed point).
    std::unordered_map<Name, std::unordered_set<Name>> pureDirectGlobals;
    std::unordered_map<Name, std::unordered_set<Name>> pureDirectPrereqs;
    changed = true;
    while (changed) {
      changed = false;
      for (Name fn : onceFuncs) {
        if (pureLazyOnce.contains(fn)) {
          continue;
        }
        Function* f = wasm.getFunction(fn);
        if (!f->vars.empty()) {
          continue;
        }
        auto* iff = f->body->dynCast<If>();
        Name guardG;
        if (!iff || iff->ifFalse ||
            !extractGuardGlobal(iff->condition, guardG)) {
          continue;
        }
        std::vector<Expression*> stmts;
        if (auto* tb = iff->ifTrue->dynCast<Block>()) {
          if (tb->name.is()) {
            continue;
          }
          for (auto* item : tb->list) {
            stmts.push_back(item);
          }
        } else {
          stmts.push_back(iff->ifTrue);
        }

        std::unordered_set<Name> writtenGs;
        std::unordered_set<Name> prereqs;
        bool ok = true;
        auto scIt = funcSetCounts.find(fn);
        for (Expression* stmt : stmts) {
          if (auto* c = stmt->dynCast<Call>()) {
            if (!c->operands.empty() || !pureLazyOnce.contains(c->target)) {
              ok = false;
              break;
            }
            prereqs.insert(c->target);
          } else if (auto* gs = stmt->dynCast<GlobalSet>()) {
            Name wg = gs->name;
            if (!isValidGuardGlobal(wg)) {
              ok = false;
              break;
            }
            auto wIt = globalWriterFuncs.find(wg);
            if (wIt == globalWriterFuncs.end() || wIt->second.size() != 1 ||
                !wIt->second.contains(fn)) {
              ok = false;
              break;
            }
            if (scIt == funcSetCounts.end() ||
                scIt->second.find(wg)->second != 1) {
              ok = false;
              break;
            }
            if (!isPureInitValue(gs->value)) {
              ok = false;
              break;
            }
            for (auto* subCall : FindAll<Call>(gs->value).list) {
              if (subCall->type == Type::none &&
                  pureLazyOnce.contains(subCall->target)) {
                prereqs.insert(subCall->target);
              }
            }
            writtenGs.insert(wg);
          } else {
            ok = false;
            break;
          }
        }
        if (ok && writtenGs.contains(guardG)) {
          pureLazyOnce.insert(fn);
          pureDirectGlobals[fn] = std::move(writtenGs);
          pureDirectPrereqs[fn] = std::move(prereqs);
          changed = true;
        }
      }
    }

    if (pureLazyOnce.empty()) {
      return;
    }

    for (Name fn : pureLazyOnce) {
      auto& tg = pureTransitiveGlobals[fn];
      std::unordered_set<Name> visited = {fn};
      std::vector<Name> worklist = {fn};
      while (!worklist.empty()) {
        Name cur = worklist.back();
        worklist.pop_back();
        for (Name g : pureDirectGlobals[cur]) {
          tg.insert(g);
        }
        for (Name p : pureDirectPrereqs[cur]) {
          if (visited.insert(p).second) {
            worklist.push_back(p);
          }
        }
      }
    }

    // 4. Verify module-wide Self-Protected Reader Invariant for each fn in
    // pureLazyOnce.
    for (Name fn : pureLazyOnce) {
      const auto& tg = pureTransitiveGlobals[fn];
      bool allReadersProtected = true;
      for (Name g : tg) {
        auto rIt = globalReaderFuncs.find(g);
        if (rIt == globalReaderFuncs.end()) {
          continue;
        }
        for (Name readerName : rIt->second) {
          if (pureLazyOnce.contains(readerName) &&
              pureTransitiveGlobals[readerName].contains(g)) {
            continue;
          }
          Function* readerFn = wasm.getFunction(readerName);
          if (!verifyReaderDominatesGlobal(readerFn, g)) {
            allReadersProtected = false;
            break;
          }
        }
        if (!allReadersProtected) {
          break;
        }
      }
      if (allReadersProtected) {
        safePureOnce.insert(fn);
      }
    }

    // Close safePureOnce under prerequisites.
    changed = true;
    while (changed) {
      changed = false;
      for (auto it = safePureOnce.begin(); it != safePureOnce.end();) {
        bool ok = true;
        for (Name p : pureDirectPrereqs[*it]) {
          if (!safePureOnce.contains(p)) {
            ok = false;
            break;
          }
        }
        if (!ok) {
          it = safePureOnce.erase(it);
          changed = true;
        } else {
          ++it;
        }
      }
    }

    for (Name fn : safePureOnce) {
      for (Name g : pureTransitiveGlobals[fn]) {
        allSafePureGlobals.insert(g);
      }
    }
  }

  bool verifyReaderDominatesGlobal(Function* readerFn, Name targetG) const {
    if (!readerFn || !readerFn->body) {
      return false;
    }
    std::unordered_map<Name, std::vector<bool>> breakStates;
    bool allDominated = true;

    std::function<void(Expression*, bool&)> walk = [&](Expression* curr,
                                                       bool& isInit) {
      if (!curr || !allDominated) {
        return;
      }
      if (auto* gg = curr->dynCast<GlobalGet>()) {
        if (gg->name == targetG && !isInit) {
          allDominated = false;
        }
        return;
      }
      if (auto* c = curr->dynCast<Call>()) {
        for (auto* op : c->operands) {
          walk(op, isInit);
        }
        auto it = pureTransitiveGlobals.find(c->target);
        if (it != pureTransitiveGlobals.end() &&
            it->second.contains(targetG)) {
          isInit = true;
        }
        return;
      }
      if (auto* iff = curr->dynCast<If>()) {
        walk(iff->condition, isInit);
        bool entry = isInit;
        bool trueInit = entry;
        walk(iff->ifTrue, trueInit);
        bool trueReach = !doesNotReturn(iff->ifTrue);
        if (iff->ifFalse) {
          bool falseInit = entry;
          walk(iff->ifFalse, falseInit);
          bool falseReach = !doesNotReturn(iff->ifFalse);
          if (trueReach && falseReach) {
            isInit = trueInit && falseInit;
          } else if (trueReach) {
            isInit = trueInit;
          } else if (falseReach) {
            isInit = falseInit;
          }
        } else {
          isInit = entry;
        }
        return;
      }
      if (auto* b = curr->dynCast<Block>()) {
        if (b->name.is()) {
          breakStates[b->name].clear();
        }
        bool fallthroughReach = true;
        for (auto* item : b->list) {
          walk(item, isInit);
          if (doesNotReturn(item)) {
            fallthroughReach = false;
            break;
          }
        }
        if (b->name.is()) {
          auto& bStates = breakStates[b->name];
          if (!bStates.empty()) {
            bool joined = fallthroughReach ? isInit : true;
            for (bool bs : bStates) {
              joined = joined && bs;
            }
            isInit = joined;
          }
          breakStates.erase(b->name);
        }
        return;
      }
      if (auto* br = curr->dynCast<Break>()) {
        if (br->value) {
          walk(br->value, isInit);
        }
        if (br->condition) {
          walk(br->condition, isInit);
        }
        auto it = breakStates.find(br->name);
        if (it != breakStates.end()) {
          it->second.push_back(isInit);
        }
        return;
      }
      if (auto* sw = curr->dynCast<Switch>()) {
        if (sw->value) {
          walk(sw->value, isInit);
        }
        walk(sw->condition, isInit);
        for (Name t : sw->targets) {
          auto it = breakStates.find(t);
          if (it != breakStates.end()) {
            it->second.push_back(isInit);
          }
        }
        auto it = breakStates.find(sw->default_);
        if (it != breakStates.end()) {
          it->second.push_back(isInit);
        }
        return;
      }
      if (auto* loop = curr->dynCast<Loop>()) {
        bool entry = isInit;
        bool bodyInit = entry;
        walk(loop->body, bodyInit);
        isInit = entry;
        return;
      }
      if (auto* tryy = curr->dynCast<Try>()) {
        bool entry = isInit;
        bool bodyInit = entry;
        walk(tryy->body, bodyInit);
        for (auto* cb : tryy->catchBodies) {
          bool cbInit = entry;
          walk(cb, cbInit);
        }
        isInit = entry;
        return;
      }
      for (auto* child : ChildIterator(curr)) {
        walk(child, isInit);
      }
    };

    bool init = false;
    walk(readerFn->body, init);
    return allDominated;
  }

  bool optimizeFunction(Function* func) const {
    if (!func || func->imported() || !func->body || onceFuncs.empty()) {
      return false;
    }

    // Quick check: does this function contain any Call to onceFuncs?
    bool hasOnceCall = false;
    for (auto* c : FindAll<Call>(func->body).list) {
      if (onceFuncs.contains(c->target)) {
        hasOnceCall = true;
        break;
      }
    }
    if (!hasOnceCall) {
      return false;
    }

    bool anyChanged = false;

    // Step 1: Forward dominator elimination of duplicate once-calls and
    // vestigial safePureOnce calls.
    struct State {
      std::unordered_set<Name> activeOnce;
      std::unordered_map<Name, std::unordered_set<Call*>> activeProviders;
    };

    auto joinStates = [](const std::vector<State>& states) -> State {
      State res;
      if (states.empty()) {
        return res;
      }
      res.activeOnce = states[0].activeOnce;
      for (size_t i = 1; i < states.size(); i++) {
        for (auto it = res.activeOnce.begin(); it != res.activeOnce.end();) {
          if (!states[i].activeOnce.contains(*it)) {
            it = res.activeOnce.erase(it);
          } else {
            ++it;
          }
        }
      }
      for (const auto& st : states) {
        for (const auto& [g, calls] : st.activeProviders) {
          res.activeProviders[g].insert(calls.begin(), calls.end());
        }
      }
      return res;
    };

    auto runDominatorPass = [&]() -> bool {
      std::unordered_map<Name, std::vector<State>> breakStates;
      std::unordered_set<Call*> dominatedCalls;
      std::unordered_set<Call*> allSafePureCallNodes;
      std::unordered_set<Call*> neededSafePureCallNodes;

      State initialState;
      if (onceFuncs.contains(func->name)) {
        initialState.activeOnce.insert(func->name);
      }

      std::function<void(Expression*, State&)> walk = [&](Expression* curr,
                                                          State& state) {
        if (!curr) {
          return;
        }
        if (auto* gg = curr->dynCast<GlobalGet>()) {
          if (allSafePureGlobals.contains(gg->name)) {
            auto pIt = state.activeProviders.find(gg->name);
            if (pIt != state.activeProviders.end()) {
              neededSafePureCallNodes.insert(pIt->second.begin(),
                                             pIt->second.end());
            }
          }
          return;
        }
        if (auto* c = curr->dynCast<Call>()) {
          for (auto* op : c->operands) {
            walk(op, state);
          }
          if (onceFuncs.contains(c->target)) {
            if (state.activeOnce.contains(c->target)) {
              dominatedCalls.insert(c);
            } else {
              auto impIt = impliedOnce.find(c->target);
              if (impIt != impliedOnce.end()) {
                state.activeOnce.insert(impIt->second.begin(),
                                        impIt->second.end());
              } else {
                state.activeOnce.insert(c->target);
              }
              if (safePureOnce.contains(c->target)) {
                allSafePureCallNodes.insert(c);
                auto tgIt = pureTransitiveGlobals.find(c->target);
                if (tgIt != pureTransitiveGlobals.end()) {
                  for (Name g : tgIt->second) {
                    auto& prov = state.activeProviders[g];
                    if (prov.empty()) {
                      prov.insert(c);
                    }
                  }
                }
              }
            }
          }
          return;
        }
        if (auto* iff = curr->dynCast<If>()) {
          walk(iff->condition, state);
          State entry = state;
          State trueState = entry;
          walk(iff->ifTrue, trueState);
          bool trueReach = !doesNotReturn(iff->ifTrue);
          if (iff->ifFalse) {
            State falseState = entry;
            walk(iff->ifFalse, falseState);
            bool falseReach = !doesNotReturn(iff->ifFalse);
            std::vector<State> branches;
            if (trueReach) {
              branches.push_back(std::move(trueState));
            }
            if (falseReach) {
              branches.push_back(std::move(falseState));
            }
            if (!branches.empty()) {
              state = joinStates(branches);
            }
          } else {
            state = std::move(entry);
          }
          return;
        }
        if (auto* b = curr->dynCast<Block>()) {
          if (b->name.is()) {
            breakStates[b->name].clear();
          }
          bool fallthroughReach = true;
          for (auto* item : b->list) {
            walk(item, state);
            if (doesNotReturn(item)) {
              fallthroughReach = false;
              break;
            }
          }
          if (b->name.is()) {
            auto& bStates = breakStates[b->name];
            if (!bStates.empty()) {
              std::vector<State> branches;
              if (fallthroughReach) {
                branches.push_back(std::move(state));
              }
              for (auto& bs : bStates) {
                branches.push_back(std::move(bs));
              }
              state = joinStates(branches);
            }
            breakStates.erase(b->name);
          }
          return;
        }
        if (auto* br = curr->dynCast<Break>()) {
          if (br->value) {
            walk(br->value, state);
          }
          if (br->condition) {
            walk(br->condition, state);
          }
          auto it = breakStates.find(br->name);
          if (it != breakStates.end()) {
            it->second.push_back(state);
          }
          return;
        }
        if (auto* sw = curr->dynCast<Switch>()) {
          if (sw->value) {
            walk(sw->value, state);
          }
          walk(sw->condition, state);
          for (Name t : sw->targets) {
            auto it = breakStates.find(t);
            if (it != breakStates.end()) {
              it->second.push_back(state);
            }
          }
          auto it = breakStates.find(sw->default_);
          if (it != breakStates.end()) {
            it->second.push_back(state);
          }
          return;
        }
        if (auto* loop = curr->dynCast<Loop>()) {
          State entry = state;
          State bodyState = entry;
          walk(loop->body, bodyState);
          state = std::move(entry);
          return;
        }
        if (auto* tryy = curr->dynCast<Try>()) {
          State entry = state;
          State bodyState = entry;
          walk(tryy->body, bodyState);
          for (auto* cb : tryy->catchBodies) {
            State cbState = entry;
            walk(cb, cbState);
          }
          state = std::move(entry);
          return;
        }
        for (auto* child : ChildIterator(curr)) {
          walk(child, state);
        }
      };

      walk(func->body, initialState);

      bool passChanged = false;
      bool removeUnusedSafePure = !pureLazyOnce.contains(func->name);
      for (Call* c : dominatedCalls) {
        ExpressionManipulator::nop(c);
        passChanged = true;
      }
      if (removeUnusedSafePure) {
        for (Call* c : allSafePureCallNodes) {
          if (!neededSafePureCallNodes.contains(c) &&
              !dominatedCalls.contains(c)) {
            ExpressionManipulator::nop(c);
            passChanged = true;
          }
        }
      }
      return passChanged;
    };

    auto cleanNops = [&]() {
      struct NopCleaner : public PostWalker<NopCleaner> {
        Module& wasm;
        const PassOptions& options;
        NopCleaner(Module& wasm, const PassOptions& options)
          : wasm(wasm), options(options) {}
        void visitBlock(Block* curr) {
          if (curr->list.size() <= 1) {
            if (curr->list.size() == 1 && curr->list[0]->is<Nop>() &&
                !curr->name.is()) {
              ExpressionManipulator::nop(curr);
            }
            return;
          }
          curr->list.erase(
            std::remove_if(curr->list.begin(),
                           curr->list.end(),
                           [](Expression* e) { return e->is<Nop>(); }),
            curr->list.end());
          if (curr->list.empty()) {
            if (!curr->name.is()) {
              ExpressionManipulator::nop(curr);
            } else {
              curr->list.push_back(Builder(wasm).makeNop());
            }
          }
        }
        void visitIf(If* curr) {
          if (curr->ifFalse && curr->ifFalse->is<Nop>()) {
            curr->ifFalse = nullptr;
          }
          if (!curr->ifFalse && curr->ifTrue->is<Nop>()) {
            EffectAnalyzer condEff(options, wasm, curr->condition);
            if (!condEff.hasSideEffects()) {
              ExpressionManipulator::nop(curr);
            }
          }
        }
      } cleaner(wasm, options);
      cleaner.walk(func->body);
    };

    if (runDominatorPass()) {
      cleanNops();
      anyChanged = true;
    }

    // Step 2: Bottom-up hoisting of common leading once-calls that remain at
    // the start of both If and Else branches.
    struct IfHoister : public PostWalker<IfHoister> {
      const OnceCallOptimizer& parent;
      bool changed = false;
      IfHoister(const OnceCallOptimizer& parent) : parent(parent) {}

      Name getLeadingOnceCall(Expression* branch) const {
        if (auto* c = branch->dynCast<Call>()) {
          if (parent.onceFuncs.contains(c->target)) {
            return c->target;
          }
        } else if (auto* b = branch->dynCast<Block>()) {
          if (!b->list.empty()) {
            if (auto* c = b->list[0]->dynCast<Call>()) {
              if (parent.onceFuncs.contains(c->target)) {
                return c->target;
              }
            }
          }
        }
        return Name();
      }

      Call* popLeadingOnceCall(Expression*& branch) {
        if (auto* c = branch->dynCast<Call>()) {
          branch = Builder(parent.wasm).makeNop();
          return c;
        }
        auto* b = branch->cast<Block>();
        auto* c = b->list[0]->cast<Call>();
        if (b->list.size() == 1) {
          b->list[0] = Builder(parent.wasm).makeNop();
        } else {
          b->list.erase(b->list.begin());
        }
        return c;
      }

      void visitIf(If* curr) {
        if (!curr->ifFalse) {
          return;
        }
        std::vector<Call*> hoistedCalls;
        while (true) {
          Name t1 = getLeadingOnceCall(curr->ifTrue);
          if (!t1.is()) {
            break;
          }
          Name t2 = getLeadingOnceCall(curr->ifFalse);
          if (t1 != t2) {
            break;
          }
          EffectAnalyzer condEff(parent.options, parent.wasm, curr->condition);
          if (condEff.transfersControlFlow() || condEff.trap ||
              condEff.throws() || condEff.calls ||
              !condEff.globalsWritten.empty()) {
            break;
          }
          bool readsWrittenGlobal = false;
          auto wIt = parent.onceWrittenGlobals.find(t1);
          if (wIt != parent.onceWrittenGlobals.end()) {
            for (const auto& g : condEff.mutableGlobalsRead) {
              if (wIt->second.contains(g)) {
                readsWrittenGlobal = true;
                break;
              }
            }
          }
          if (readsWrittenGlobal) {
            break;
          }
          Call* keptCall = popLeadingOnceCall(curr->ifTrue);
          popLeadingOnceCall(curr->ifFalse);
          hoistedCalls.push_back(keptCall);
        }
        if (!hoistedCalls.empty()) {
          Builder builder(parent.wasm);
          auto* seq = builder.makeBlock();
          for (Call* c : hoistedCalls) {
            seq->list.push_back(c);
          }
          seq->list.push_back(curr);
          seq->finalize(curr->type);
          replaceCurrent(seq);
          changed = true;
        }
      }
    } hoister(*this);
    hoister.walk(func->body);
    if (hoister.changed) {
      anyChanged = true;
      runDominatorPass();
      cleanNops();
    }

    return anyChanged;
  }
};

} // anonymous namespace

struct LocalCSE : public WalkerPass<PostWalker<LocalCSE>> {
  std::shared_ptr<const OnceCallOptimizer> onceOpt;

  LocalCSE() = default;
  explicit LocalCSE(std::shared_ptr<const OnceCallOptimizer> onceOpt)
    : onceOpt(std::move(onceOpt)) {}

  bool isFunctionParallel() override { return true; }

  // FIXME DWARF updating does not handle local changes yet.
  bool invalidatesDWARF() override { return true; }

  bool hasPrepareToRun() override { return true; }

  void prepareToRun(Module* module) override {
    auto opt = std::make_shared<OnceCallOptimizer>(*module, getPassOptions());
    opt->analyze();
    onceOpt = std::move(opt);
  }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<LocalCSE>(onceOpt);
  }

  void doWalkFunction(Function* func) {
    auto& options = getPassOptions();
    bool onceChanged = onceOpt && onceOpt->optimizeFunction(func);

    if (options.shrinkLevel > 0) {
      bool changed = onceChanged;
      if (forwardStructNewFields(func, *getModule(), options)) {
        changed = true;
      }
      if (unaliasSingleFieldProjectionLocals(func, *getModule(), options)) {
        changed = true;
      }
      if (unCseSingleWriteTees(func, *getModule(), options, false)) {
        changed = true;
      }
      if (changed) {
        ReFinalize().walkFunctionInModule(func, getModule());
      }
      ScopedDominatorCSE(func, *getModule(), options).run();
      if (unCseSingleWriteTees(func,
                               *getModule(),
                               options,
                               options.shrinkLevel >= 2 ||
                                 options.trapsNeverHappen)) {
        ReFinalize().walkFunctionInModule(func, getModule());
      }
      TypeUpdating::handleNonDefaultableLocals(func, *getModule());
      return;
    }

    if (onceChanged) {
      ReFinalize().walkFunctionInModule(func, getModule());
    }

    RequestInfoMap requestInfos;

    Scanner scanner(options, requestInfos);
    scanner.walkFunctionInModule(func, getModule());
    if (requestInfos.empty()) {
      // We did not find any repeated expressions at all.
      return;
    }

    Checker checker(options, requestInfos);
    checker.walkFunctionInModule(func, getModule());
    if (requestInfos.empty()) {
      // No repeated expressions remain after checking for effects.
      return;
    }

    Applier applier(requestInfos);
    applier.walkFunctionInModule(func, getModule());
  }
};

Pass* createLocalCSEPass() { return new LocalCSE(); }

} // namespace wasm
