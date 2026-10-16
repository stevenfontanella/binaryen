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

static bool forwardStructNewFields(Function* func,
                                   Module& wasm,
                                   PassOptions& options) {
  if (!wasm.features.hasGC() || !func->body) {
    return false;
  }

  Index numLocals = func->getNumLocals();
  Index numParams = func->getNumParams();
  std::vector<Index> setCounts(numLocals, 0);
  std::unordered_set<HeapType> writtenStructTypes;
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
    std::unordered_set<HeapType>& writtenStructTypes;
    bool& hasUnknownStructWrite;
    bool& hasCandidateStructNew;

    PreScanner(Module& wasm,
               PassOptions& options,
               Index numParams,
               std::vector<Index>& setCounts,
               std::unordered_set<HeapType>& writtenStructTypes,
               bool& hasUnknownStructWrite,
               bool& hasCandidateStructNew)
      : wasm(wasm), options(options), numParams(numParams),
        setCounts(setCounts), writtenStructTypes(writtenStructTypes),
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
      if (auto* ss = curr->dynCast<StructSet>()) {
        if (ss->ref->type.isRef()) {
          writtenStructTypes.insert(ss->ref->type.getHeapType());
        } else {
          hasUnknownStructWrite = true;
        }
        return;
      }
      if (! hasUnknownStructWrite) {
        ShallowEffectAnalyzer eff(options, wasm, curr);
        if (eff.writesStruct || eff.writesSharedStruct || eff.calls) {
          hasUnknownStructWrite = true;
        }
      }
    }
  } preScanner(wasm,
               options,
               numParams,
               setCounts,
               writtenStructTypes,
               hasUnknownStructWrite,
               hasCandidateStructNew);
  preScanner.walk(func->body);

  if (!hasCandidateStructNew) {
    return false;
  }

  auto isFieldUnmodified = [&](HeapType ht, const Field& field) -> bool {
    if (field.mutable_ == Immutable) {
      return true;
    }
    if (hasUnknownStructWrite) {
      return false;
    }
    for (HeapType wht : writtenStructTypes) {
      if (HeapType::isSubType(wht, ht) || HeapType::isSubType(ht, wht)) {
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
      if (iff->ifFalse) {
        activeStructNews = savedSN;
        activeSingleSets = savedSS;
        walkDom(&iff->ifFalse);
      }
      activeStructNews = std::move(savedSN);
      activeSingleSets = std::move(savedSS);
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
      if (!isFieldUnmodified(ht, field)) {
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
    std::unordered_set<HeapType> writtenStructTypes;
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
        if (iff->ifFalse) {
          activeSingleSets = savedSS;
          scanDom(&iff->ifFalse);
        }
        activeSingleSets = std::move(savedSS);
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
      if (auto* ss = curr->dynCast<StructSet>()) {
        if (ss->ref->type.isRef()) {
          writtenStructTypes.insert(ss->ref->type.getHeapType());
        } else {
          hasUnknownStructWrite = true;
        }
        return;
      }
      if (!hasUnknownStructWrite) {
        ShallowEffectAnalyzer eff(options, wasm, curr);
        if (eff.writesStruct || eff.writesSharedStruct || eff.calls) {
          hasUnknownStructWrite = true;
        }
      }
    };

    scanDom(&func->body);

    auto isFieldUnmodified = [&](HeapType ht, const Field& field) -> bool {
      if (field.mutable_ == Immutable) {
        return true;
      }
      if (hasUnknownStructWrite) {
        return false;
      }
      for (HeapType wht : writtenStructTypes) {
        if (HeapType::isSubType(wht, ht) || HeapType::isSubType(ht, wht)) {
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
      if (!isFieldUnmodified(ht, field)) {
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
    Index reuses = 0;
    SmallVector<Index, 4> localsRead;
    SmallVector<Name, 2> mutableGlobalsRead;
    bool readsMutableStruct = false;
    bool readsMutableArray = false;
    bool readsMemory = false;
    bool readsTable = false;
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

  static Index getShallowEstBytes(Expression* curr) {
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
    if (curr->is<StructGet>() || curr->is<RefCast>() || curr->is<RefTest>()) {
      return 4;
    }
    if (curr->is<ArrayGet>() || curr->is<Load>()) {
      return 3;
    }
    if (auto* call = curr->dynCast<Call>()) {
      return call->operands.empty() ? 3 : 4;
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
      return true;
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

  void invalidateByLocalWrite(Index localIdx) {
    for (auto it = activeMap.begin(); it != activeMap.end();) {
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
        it = activeMap.erase(it);
      } else {
        ++it;
      }
    }
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
        if (isKilledByEffects(childNormRecs[i], jEff)) {
          return false;
        }
      }
    }
    return true;
  }

  NodeInfo walkExpr(Expression* curr) {
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
      bool capturedAfterDirectBr = false;
      for (Index i = 0; i < block->list.size(); i++) {
        if (i == firstBranchIdx) {
          bool isDirectCondBr = false;
          if (auto* br = block->list[i]->dynCast<Break>()) {
            isDirectCondBr = br->name == block->name && br->condition &&
                             !SubtreeTargetsName(br->value, block->name) &&
                             !SubtreeTargetsName(br->condition, block->name);
          } else if (auto* bron = block->list[i]->dynCast<BrOn>()) {
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
        walkExpr(block->list[i]);
        if (i == firstBranchIdx && capturedAfterDirectBr) {
          savedExitMap = activeMap;
        }
      }
      if (block->type == Type::unreachable) {
        activeMap.clear();
      } else if (firstBranchIdx < block->list.size()) {
        activeMap = std::move(savedExitMap);
        for (Index i = firstBranchIdx; i < block->list.size(); i++) {
          if (i == firstBranchIdx && capturedAfterDirectBr) {
            continue;
          }
          EffectAnalyzer eff(options, wasm, block->list[i]);
          eff.trap = false;
          invalidateByEffects(eff);
        }
      }
      return {};
    }
    if (auto* iff = curr->dynCast<If>()) {
      walkExpr(iff->condition);
      ActiveMap afterCond = activeMap;
      walkExpr(iff->ifTrue);
      if (iff->ifFalse) {
        activeMap = afterCond;
        walkExpr(iff->ifFalse);
      }
      if (iff->type == Type::unreachable) {
        activeMap.clear();
      } else {
        activeMap = std::move(afterCond);
        EffectAnalyzer thenEff(options, wasm, iff->ifTrue);
        thenEff.trap = false;
        invalidateByEffects(thenEff);
        if (iff->ifFalse) {
          EffectAnalyzer elseEff(options, wasm, iff->ifFalse);
          elseEff.trap = false;
          invalidateByEffects(elseEff);
        }
      }
      return {};
    }
    if (auto* loop = curr->dynCast<Loop>()) {
      EffectAnalyzer loopEff(options, wasm, loop->body);
      loopEff.trap = false;
      invalidateByEffects(loopEff);
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
        invalidateByEffects(bodyEff);
        walkExpr(cb);
      }
      if (tryy->type == Type::unreachable) {
        activeMap.clear();
      } else {
        activeMap = std::move(beforeTry);
        EffectAnalyzer tryEff(options, wasm, tryy);
        tryEff.trap = false;
        invalidateByEffects(tryEff);
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
        invalidateByEffects(ttEff);
      }
      return {};
    }
    if (auto* br = curr->dynCast<Break>()) {
      if (br->value) {
        walkExpr(br->value);
      }
      if (br->condition) {
        walkExpr(br->condition);
      }
      if (!br->condition) {
        activeMap.clear();
      }
      return {};
    }
    if (auto* bron = curr->dynCast<BrOn>()) {
      walkExpr(bron->ref);
      if (bron->desc) {
        walkExpr(bron->desc);
      }
      return {};
    }
    if (auto* sw = curr->dynCast<Switch>()) {
      if (sw->value) {
        walkExpr(sw->value);
      }
      walkExpr(sw->condition);
      activeMap.clear();
      return {};
    }
    if (auto* ret = curr->dynCast<Return>()) {
      if (ret->value) {
        walkExpr(ret->value);
      }
      activeMap.clear();
      return {};
    }
    if (auto* th = curr->dynCast<Throw>()) {
      for (auto* op : th->operands) {
        walkExpr(op);
      }
      activeMap.clear();
      return {};
    }
    if (curr->is<Rethrow>() || curr->is<Unreachable>()) {
      activeMap.clear();
      return {};
    }
    if (auto* lg = curr->dynCast<LocalGet>()) {
      return {wasm::hash(lg->index), 2, true, false};
    }
    if (auto* ls = curr->dynCast<LocalSet>()) {
      NodeInfo valInfo = walkExpr(ls->value);
      invalidateByLocalWrite(ls->index);
      if (valInfo.possible && !origRecords.empty() &&
          origRecords.back().orig == ls->value &&
          origRecords.back().reuses == 0 &&
          func->getLocalType(ls->index) == ls->value->type) {
        Index valOrigIdx = origRecords.size() - 1;
        auto& rec = origRecords[valOrigIdx];
        if (std::find(rec.localsRead.begin(),
                      rec.localsRead.end(),
                      ls->index) == rec.localsRead.end()) {
          rec.existingLocal = ls->index;
          rec.localsRead.push_back(ls->index);
        }
      }
      if (ls->isTee()) {
        return {wasm::hash(ls->index), 2, true, true};
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
      invalidateByEffects(shallow, curr);
    }
    if (shallow.branchesOut || curr->type == Type::unreachable) {
      activeMap.clear();
      return {};
    }

    bool shallowPossible =
      idempotentCall ||
      (!shallow.hasNonTrapSideEffects() &&
       !Properties::isShallowlyGenerative(curr, func, wasm));
    bool possible = childrenPossible && shallowPossible;
    if (possible && hasTee && !isMultiChildTeeSafe(curr)) {
      possible = false;
    }

    NodeInfo info{hash, estBytes, possible, hasTee};
    if (!possible || !curr->type.isConcrete() ||
        Properties::isConstantExpression(curr) || estBytes < 4) {
      return info;
    }

    auto& bucket = activeMap[hash];
    if (!hasTee) {
      for (Index origIdx : bucket) {
        if (normEqual(origRecords[origIdx].orig, curr)) {
          repeatToOrigIdx[curr] = origIdx;
          origRecords[origIdx].reuses++;
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

    for (Index origIdx : bucket) {
      if (normEqual(origRecords[origIdx].orig, curr)) {
        return info;
      }
    }

    Index newOrigIdx = origRecords.size();
    OrigRecord rec;
    rec.orig = curr;
    rec.hash = hash;
    rec.estBytes = estBytes;
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

    auto passesByteCostGate = [&](const OrigRecord& rec,
                                  Index effBytes) -> bool {
      if (rec.reuses == 0 || effBytes <= 2) {
        return false;
      }
      if (rec.existingLocal != Index(-1)) {
        return effBytes >= 4;
      }
      return rec.reuses * (effBytes - 2) >= 3;
    };

    // Evaluate originals in reverse postorder (parents before children).
    for (int idx = int(origRecords.size()) - 1; idx >= 0; idx--) {
      auto& rec = origRecords[idx];
      if (rec.reuses == 0) {
        continue;
      }
      Index effBytes = rec.estBytes;
      for (auto* child : ChildIterator(rec.orig)) {
        auto cIt = exprToOrigIdx.find(child);
        if (cIt != exprToOrigIdx.end()) {
          const auto& cRec = origRecords[cIt->second];
          if (passesByteCostGate(cRec, cRec.estBytes) &&
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
      }
    }

    std::unordered_map<Expression*, Index> origLocalMap;
    std::unordered_map<Expression*, Index> needsTeeWrap;
    for (Index idx = 0; idx < origRecords.size(); idx++) {
      const auto& rec = origRecords[idx];
      if (rec.reuses == 0) {
        continue;
      }
      if (rec.existingLocal != Index(-1)) {
        origLocalMap[rec.orig] = rec.existingLocal;
      } else {
        Index loc = Builder::addVar(func, rec.orig->type);
        origLocalMap[rec.orig] = loc;
        needsTeeWrap[rec.orig] = loc;
      }
    }

    if (origLocalMap.empty()) {
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
          Expression* orig = origRecords[rIt->second].orig;
          auto lIt = origLocalMap.find(orig);
          if (lIt != origLocalMap.end()) {
            replaceCurrent(Builder(wasm).makeLocalGet(lIt->second, curr->type));
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

} // anonymous namespace

struct LocalCSE : public WalkerPass<PostWalker<LocalCSE>> {
  bool isFunctionParallel() override { return true; }

  // FIXME DWARF updating does not handle local changes yet.
  bool invalidatesDWARF() override { return true; }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<LocalCSE>();
  }

  void doWalkFunction(Function* func) {
    auto& options = getPassOptions();

    if (options.shrinkLevel > 0) {
      bool changed = forwardStructNewFields(func, *getModule(), options);
      if (unaliasSingleFieldProjectionLocals(func, *getModule(), options)) {
        changed = true;
      }
      if (changed) {
        ReFinalize().walkFunctionInModule(func, getModule());
      }
      ScopedDominatorCSE(func, *getModule(), options).run();
      TypeUpdating::handleNonDefaultableLocals(func, *getModule());
      return;
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
