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
// Outlines duplicated expression trees into shared helper functions.
//
// Aggressive inlining leaves many identical copies of small expression trees
// in function bodies, for example
//
//   (if (result i32)
//     (ref.test (ref $T) (struct.get $U 2 (local.get $x)))
//     (then (call $foo (local.get $x)))
//     (else (i32.const 2147483647)))
//
// This pass finds trees that are identical except for the locals they read,
// and replaces them with calls to a helper function containing a single copy
// of the tree, when that is estimated to shrink the binary. The local.gets are
// the "holes" of the tree: they become parameters of the helper, passed in
// evaluation order.
//
// A tree can be moved into a helper verbatim if it has no local writes and
// nothing in it branches out of it: that is, it contains no local.set/tee,
// br/br_if/br_table/br_on_*, return, return_call, try, loop, pop, or stack
// switching. Control flow like if, block, select and drop is fine. Reading the
// holes before the tree (at the call) instead of inside it is fine as they have
// no effects, and nothing in the tree can change what they read.
//
// Trees whose type is unreachable (e.g. ones rooted in a throw) become a call
// to a helper returning nothing, followed by an unreachable.
//
// Turning cheap code into a call costs speed, so what we outline is tiered.
// The tier is chosen with --outline-expressions=MODE (or, equivalently,
// --pass-arg=outline-expressions-mode@MODE):
//
//   calls:  trees that contain a call, allocation or throw, except single
//           instructions with trivial operands (the inliner would inline such
//           a helper right back when optimizing for speed) and once-function
//           guards. The default without a shrink level.
//   guards: as calls, but also the single instructions and the guards.
//   trees:  as guards, but also pure trees of more than one instruction.
//   all:    anything, including pure single instructions with trivial
//           operands. The default with a shrink level.
//
// The helpers may themselves contain duplicated trees (two helpers can share
// a sub-tree that also appears elsewhere), so the pass repeats itself a few
// times until nothing more is outlined.
//
// This pass is not part of the default pipelines: it works best once at the
// very end, after all inlining and other optimizations are done, as the
// helpers it creates hide information from passes that run later.
//

#include <deque>

#include "ir/find_all.h"
#include "ir/iteration.h"
#include "ir/manipulation.h"
#include "ir/module-utils.h"
#include "ir/names.h"
#include "ir/properties.h"
#include "ir/utils.h"
#include "pass.h"
#include "support/hash.h"
#include "support/small_vector.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

// Rough estimates of binary sizes, in bytes.

// A call opcode plus a typical two-byte function index.
const Index CallBytes = 3;
// A local.get in the helper, for each hole, and at the site.
const Index LocalGetBytes = 2;
// Function section entry, code size, local declarations, end opcode, and a
// share of a type section entry.
const Index HelperOverheadBytes = 6;

// The number of bytes used to encode a value as a signed LEB128.
int getSLEBSize(int64_t value) {
  int size = 0;
  while (true) {
    uint8_t byte = value & 0x7f;
    value >>= 7;
    size++;
    if ((value == 0 && !(byte & 0x40)) || (value == -1 && (byte & 0x40))) {
      return size;
    }
  }
}

// The estimated binary size of an expression, not including its children.
Index getShallowBytes(Expression* curr) {
  switch (curr->_id) {
    case Expression::BlockId:
      // Unnamed blocks are not emitted at all.
      return curr->cast<Block>()->name.is() ? 3 : 0;
    case Expression::IfId:
      return curr->cast<If>()->ifFalse ? 4 : 3;
    case Expression::LoopId:
      return 3;
    case Expression::BreakId:
      return 2;
    case Expression::CallId:
    case Expression::CallRefId:
      return 3;
    case Expression::CallIndirectId:
      return 4;
    case Expression::LocalGetId:
    case Expression::LocalSetId:
      return 2;
    case Expression::GlobalGetId:
    case Expression::GlobalSetId:
      return 3;
    case Expression::LoadId:
    case Expression::StoreId:
      return 3;
    case Expression::ConstId: {
      auto* c = curr->cast<Const>();
      switch (c->type.getBasic()) {
        case Type::i32:
          return 1 + getSLEBSize(c->value.geti32());
        case Type::i64:
          return 1 + getSLEBSize(c->value.geti64());
        case Type::f32:
          return 5;
        case Type::f64:
          return 9;
        case Type::v128:
          return 18;
        default:
          WASM_UNREACHABLE("unexpected const type");
      }
    }
    case Expression::UnaryId:
    case Expression::BinaryId:
    case Expression::DropId:
    case Expression::ReturnId:
    case Expression::NopId:
    case Expression::UnreachableId:
    case Expression::RefIsNullId:
    case Expression::RefEqId:
    case Expression::RefAsId:
      return 1;
    case Expression::SelectId:
      return curr->type.isRef() ? 3 : 1;
    case Expression::ThrowId:
    case Expression::RefNullId:
    case Expression::RefI31Id:
    case Expression::I31GetId:
    case Expression::ArrayLenId:
      return 2;
    case Expression::RefFuncId:
    case Expression::StringConstId:
      return 3;
    case Expression::StructNewId:
    case Expression::StructGetId:
    case Expression::StructSetId:
    case Expression::ArrayNewId:
    case Expression::ArrayGetId:
    case Expression::ArraySetId:
    case Expression::ArrayFillId:
    case Expression::RefCastId:
    case Expression::RefTestId:
      return 4;
    case Expression::ArrayNewFixedId:
      return 5;
    case Expression::ArrayNewDataId:
    case Expression::ArrayNewElemId:
    case Expression::ArrayCopyId:
    case Expression::ArrayInitDataId:
    case Expression::ArrayInitElemId:
      return 6;
    default:
      return 2;
  }
}

// A group of sites with the same shape.
struct KeyInfo {
  std::vector<struct Site*> sites;
  // The function that performs this tree, if we found or created one.
  Function* helper = nullptr;
  // Whether we decided to outline the sites with this key.
  bool outline = false;
};

// An expression tree in a function body that we may outline.
struct Site {
  Expression* expr;
  // A hash of the tree's shape, with holes abstracted away. Trees with equal
  // shapes can share a helper.
  size_t hash = 0;
  // The estimated binary size of the tree.
  Index bytes = 0;
  // The number of holes, that is, of distinct locals the tree reads.
  Index holes = 0;
  // The nearest enclosing site, if any. Outlining that makes this site vanish.
  Site* parent = nullptr;
  // The key this site belongs to.
  KeyInfo* key = nullptr;
  // Whether this site is the entire body of the function that we use as the
  // helper for its shape (either one we created earlier, or by coincidence).
  bool helperBody = false;
};

// Information about a tree, computed bottom-up.
struct NodeInfo {
  size_t hash = 0;
  Index bytes = 0;
  // The distinct locals the tree reads, in order of first use.
  SmallVector<Index, 4> locals;
  // Whether the root has children, and whether all of them are leaves.
  bool hasChildren = false;
  bool trivialChildren = true;
  // Whether the tree can be moved into a helper.
  bool movable = true;
  // Whether the tree contains a call, allocation, or throw.
  bool interesting = false;
  // Whether the tree contains a once-function guard, see below.
  bool hasGuard = false;
};

struct FunctionInfo {
  // A deque so that pointers to sites remain valid as we add more.
  std::deque<Site> sites;
};

// What to outline, see the top of the file.
enum class Mode { Calls, Guards, Trees, All };

struct Scanner : public PostWalker<Scanner, UnifiedExpressionVisitor<Scanner>> {
  Module& wasm;
  FunctionInfo& info;
  Mode mode;

  // Infos of the already-visited children of the expressions we are in the
  // middle of visiting.
  std::vector<NodeInfo> stack;

  Scanner(Module& wasm, FunctionInfo& info, Mode mode)
    : wasm(wasm), info(info), mode(mode) {}

  void visitExpression(Expression* curr) {
    NodeInfo node;
    node.hash = getShallowHash(curr);
    node.bytes = getShallowBytes(curr);
    Index numChildren = ChildIterator(curr).getNumChildren();
    assert(stack.size() >= numChildren);
    node.hasChildren = numChildren > 0;
    for (Index i = stack.size() - numChildren; i < stack.size(); i++) {
      auto& child = stack[i];
      if (child.hasChildren) {
        node.trivialChildren = false;
      }
      hash_combine(node.hash, child.hash);
      node.bytes += child.bytes;
      for (auto local : child.locals) {
        noteLocal(node, local);
      }
      node.movable = node.movable && child.movable;
      node.interesting = node.interesting || child.interesting;
      node.hasGuard = node.hasGuard || child.hasGuard;
    }
    stack.resize(stack.size() - numChildren);
    if (auto* get = curr->dynCast<LocalGet>()) {
      noteLocal(node, get->index);
    }
    if (!isMovable(curr)) {
      node.movable = false;
    }
    if (isInteresting(curr)) {
      node.interesting = true;
    }
    if (isGuard(curr)) {
      node.hasGuard = true;
    }
    stack.push_back(node);
    if (!node.movable) {
      return;
    }
    // A single instruction with trivial operands, like a call with constant
    // arguments or a struct.get of a local.
    bool single =
      node.trivialChildren && !Properties::isControlFlowStructure(curr);
    switch (mode) {
      case Mode::Calls:
        if (!node.interesting || node.hasGuard || single) {
          return;
        }
        break;
      case Mode::Guards:
        if (!node.interesting) {
          return;
        }
        break;
      case Mode::Trees:
        if (!node.interesting && single) {
          return;
        }
        break;
      case Mode::All:
        break;
    }
    // Nothing to gain without saving at least a byte per site.
    if (node.bytes <= LocalGetBytes * node.locals.size() + CallBytes) {
      return;
    }
    // A constant expression is better left for other passes to hoist (and for
    // allocations, OutlineAllocations handles them).
    if (!node.interesting &&
        Properties::isValidConstantExpression(wasm, curr)) {
      return;
    }
    info.sites.emplace_back();
    auto& site = info.sites.back();
    site.expr = curr;
    site.hash = node.hash;
    site.bytes = node.bytes;
    site.holes = node.locals.size();
  }

  void noteLocal(NodeInfo& node, Index local) {
    for (auto other : node.locals) {
      if (other == local) {
        return;
      }
    }
    node.locals.push_back(local);
  }

  size_t getShallowHash(Expression* curr) {
    if (curr->is<LocalGet>()) {
      // A hole. Only the type matters.
      auto digest = hash(curr->_id);
      rehash(digest, curr->type.getID());
      return digest;
    }
    return ExpressionAnalyzer::shallowHash(curr);
  }

  // Whether an expression can be moved verbatim into a helper function,
  // assuming its children can be.
  bool isMovable(Expression* curr) {
    if (curr->type.isTuple()) {
      return false;
    }
    switch (curr->_id) {
      case Expression::BreakId:
      case Expression::SwitchId:
      case Expression::ReturnId:
      case Expression::LocalSetId:
      case Expression::LoopId:
      case Expression::TryId:
      case Expression::TryTableId:
      case Expression::RethrowId:
      case Expression::PopId:
      case Expression::BrOnId:
      case Expression::TupleMakeId:
      case Expression::TupleExtractId:
      case Expression::ContNewId:
      case Expression::ContBindId:
      case Expression::SuspendId:
      case Expression::ResumeId:
      case Expression::ResumeThrowId:
      case Expression::StackSwitchId:
        return false;
      case Expression::CallId:
        return !curr->cast<Call>()->isReturn;
      case Expression::CallIndirectId:
        return !curr->cast<CallIndirect>()->isReturn;
      case Expression::CallRefId:
        return !curr->cast<CallRef>()->isReturn;
      default:
        return true;
    }
  }

  bool isInteresting(Expression* curr) {
    switch (curr->_id) {
      case Expression::CallId:
      case Expression::CallIndirectId:
      case Expression::CallRefId:
      case Expression::StructNewId:
      case Expression::ArrayNewId:
      case Expression::ArrayNewFixedId:
      case Expression::ArrayNewDataId:
      case Expression::ArrayNewElemId:
      case Expression::ThrowId:
        return true;
      default:
        return false;
    }
  }

  // A guard of a "once" function,
  //
  //   (if (i32.eqz (global.get $initialized)) (call $init))
  //
  // is almost always cheap and should stay inline for speed.
  bool isGuard(Expression* curr) {
    auto* iff = curr->dynCast<If>();
    if (!iff || iff->ifFalse) {
      return false;
    }
    auto* unary = iff->condition->dynCast<Unary>();
    if (!unary || unary->op != EqZInt32 || !unary->value->is<GlobalGet>()) {
      return false;
    }
    auto* call = iff->ifTrue->dynCast<Call>();
    return call && call->operands.empty();
  }
};

// The holes of a tree: the distinct locals it reads, in order of first use,
// with the first local.get of each.
std::vector<std::pair<Index, LocalGet*>> getHoles(Expression* curr) {
  std::vector<std::pair<Index, LocalGet*>> holes;
  for (auto* get : FindAll<LocalGet>(curr).list) {
    bool seen = false;
    for (auto& [local, _] : holes) {
      if (local == get->index) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      holes.emplace_back(get->index, get);
    }
  }
  return holes;
}

// Information about a replacement of a site with a call to a helper.
struct Replacement {
  Name helper;
  const Site* site;
};

using Replacements = std::unordered_map<Expression*, Replacement>;

struct Replacer : public WalkerPass<
                    PostWalker<Replacer, UnifiedExpressionVisitor<Replacer>>> {
  bool isFunctionParallel() override { return true; }

  bool requiresNonNullableLocalFixups() override { return false; }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<Replacer>(replacements);
  }

  Replacer(const Replacements& replacements) : replacements(replacements) {}

  // We do not replace a site inside another replaced site, so we can use a
  // simple postorder walk here (the holes we pass to the helper are the
  // original local.gets, which no replacement of a child could touch).
  void visitExpression(Expression* curr) {
    auto iter = replacements.find(curr);
    if (iter == replacements.end()) {
      return;
    }
    auto& replacement = iter->second;
    Builder builder(*getModule());
    std::vector<Expression*> args;
    for (auto& [local, get] : getHoles(curr)) {
      args.push_back(get);
    }
    assert(args.size() == replacement.site->holes);
    if (curr->type == Type::unreachable) {
      replaceCurrent(builder.makeSequence(
        builder.makeCall(replacement.helper, args, Type::none),
        builder.makeUnreachable()));
    } else {
      replaceCurrent(builder.makeCall(replacement.helper, args, curr->type));
    }
  }

private:
  const Replacements& replacements;
};

struct OutlineExpressions : public Pass {
  // The helpers perform exactly the code they replace, so the effects of
  // existing functions do not change.
  bool addsEffects() override { return false; }

  bool requiresNonNullableLocalFixups() override { return false; }

  Module* wasm;

  // A deque so that pointers to keys remain valid, in creation order so that
  // the helpers are created in a deterministic order.
  std::deque<KeyInfo> keys;
  // Sites with the same hash usually have the same shape, but we verify that,
  // so a hash may map to several keys.
  std::unordered_map<size_t, std::vector<KeyInfo*>> keysByHash;

  // The helpers we create may themselves contain duplicated trees (two
  // different helpers can share a sub-tree, which can also appear on its own
  // elsewhere), and a site nested inside an outlined site is not considered
  // again in the same round. Repeat until nothing more is outlined, with a
  // limit, as later rounds find little.
  static const Index MaxRounds = 4;

  void run(Module* module) override {
    wasm = module;

    auto mode = getMode();

    for (Index round = 0; round < MaxRounds; round++) {
      keys.clear();
      keysByHash.clear();
      if (!runRound(mode)) {
        break;
      }
    }
  }

  // Scans, decides, and replaces once. Returns whether anything was replaced.
  bool runRound(Mode mode) {
    ModuleUtils::ParallelFunctionAnalysis<FunctionInfo> analysis(
      *wasm, [&](Function* func, FunctionInfo& info) {
        if (func->imported()) {
          return;
        }
        Scanner(*wasm, info, mode).walk(func->body);
        noteParents(func->body, info);
      });

    // Group the sites by key, and find existing helpers.
    for (auto& func : wasm->functions) {
      auto& info = analysis.map[func.get()];
      for (auto& site : info.sites) {
        site.key = getKey(&site);
        site.key->sites.push_back(&site);
      }
      if (!info.sites.empty() && info.sites.back().expr == func->body) {
        noteExistingHelper(func.get(), info.sites.back());
      }
    }

    // Decide what to outline. Nested sites complicate this, as a site inside
    // an outlined site vanishes along with it, and so must not be counted for
    // its own key, which may change the decision for that key, and so forth.
    // Iterate to a fixed point (or just stop after a while, as any decision is
    // safe, just not optimal).
    for (Index i = 0; i < 10; i++) {
      bool changed = false;
      for (auto& info : keys) {
        Index count = 0;
        for (auto* site : info.sites) {
          if (isCandidate(site)) {
            count++;
          }
        }
        bool outline;
        if (info.helper) {
          // The helper already exists, so any use of it is a win.
          outline = count > 0;
        } else {
          // Each site shrinks to a call with the holes as arguments, and the
          // helper contains a copy of the tree plus overhead.
          auto* site = info.sites[0];
          int64_t saved =
            int64_t(count) * (int64_t(site->bytes) - getCallBytes(site) -
                              LocalGetBytes * int64_t(site->holes));
          int64_t cost = int64_t(site->bytes) + HelperOverheadBytes;
          outline = saved > cost;
        }
        if (outline != info.outline) {
          info.outline = outline;
          changed = true;
        }
      }
      if (!changed) {
        break;
      }
    }

    // Create the helpers and gather the replacements.
    Replacements replacements;
    for (auto& info : keys) {
      if (!info.outline) {
        continue;
      }
      for (auto* site : info.sites) {
        if (!isReplaced(site)) {
          continue;
        }
        if (!info.helper) {
          info.helper = createHelper(site);
        }
        replacements[site->expr] = Replacement{info.helper->name, site};
      }
    }
    if (replacements.empty()) {
      return false;
    }

    Replacer(replacements).run(getPassRunner(), wasm);
    return true;
  }

  Mode getMode() {
    // --outline-expressions=MODE or --pass-arg=outline-expressions-mode@MODE.
    auto name = getArgumentOrDefault(
      "outline-expressions",
      getArgumentOrDefault("outline-expressions-mode", ""));
    if (name.empty()) {
      return getPassOptions().shrinkLevel == 0 ? Mode::Calls : Mode::All;
    }
    if (name == "calls") {
      return Mode::Calls;
    }
    if (name == "guards") {
      return Mode::Guards;
    }
    if (name == "trees") {
      return Mode::Trees;
    }
    if (name == "all") {
      return Mode::All;
    }
    Fatal() << "unknown outline-expressions-mode: " << name
            << " (expected calls, guards, trees, or all)";
  }

  // Link each site to the nearest site that contains it.
  void noteParents(Expression* body, FunctionInfo& info) {
    if (info.sites.empty()) {
      return;
    }
    std::unordered_map<Expression*, Site*> siteMap;
    for (auto& site : info.sites) {
      siteMap[site.expr] = &site;
    }
    std::vector<std::pair<Expression*, Site*>> stack;
    stack.emplace_back(body, nullptr);
    while (!stack.empty()) {
      auto [curr, parent] = stack.back();
      stack.pop_back();
      if (auto iter = siteMap.find(curr); iter != siteMap.end()) {
        iter->second->parent = parent;
        parent = iter->second;
      }
      for (auto* child : ChildIterator(curr)) {
        stack.emplace_back(child, parent);
      }
    }
  }

  // Two trees have the same shape if they are equal apart from the locals that
  // their holes read, where the same local must be used in the same places in
  // both trees.
  bool sameShape(Expression* left, Expression* right) {
    std::unordered_map<Index, Index> leftToRight, rightToLeft;
    bool mismatch = false;
    auto comparer = [&](Expression* left, Expression* right) {
      auto* leftGet = left->dynCast<LocalGet>();
      auto* rightGet = right->dynCast<LocalGet>();
      if (!leftGet || !rightGet) {
        return false;
      }
      if (leftGet->type != rightGet->type) {
        mismatch = true;
        return true;
      }
      auto [l, lInserted] =
        leftToRight.try_emplace(leftGet->index, rightGet->index);
      auto [r, rInserted] =
        rightToLeft.try_emplace(rightGet->index, leftGet->index);
      if (l->second != rightGet->index || r->second != leftGet->index) {
        mismatch = true;
      }
      return true;
    };
    return ExpressionAnalyzer::flexibleEqual(left, right, comparer) &&
           !mismatch;
  }

  KeyInfo* getKey(Site* site) {
    auto& candidates = keysByHash[site->hash];
    for (auto* key : candidates) {
      if (sameShape(key->sites[0]->expr, site->expr)) {
        return key;
      }
    }
    keys.emplace_back();
    auto* key = &keys.back();
    candidates.push_back(key);
    return key;
  }

  // The bytes a site costs after outlining: a call, plus an unreachable if the
  // tree is unreachable.
  Index getCallBytes(Site* site) {
    return site->expr->type == Type::unreachable ? CallBytes + 1 : CallBytes;
  }

  // Whether a site is eligible to be outlined, given the current decisions.
  bool isEligible(Site* site) {
    return !site->helperBody && site->key->outline;
  }

  // Whether a site could be replaced, given the current decisions for the
  // other keys: it is not inside another site that will be replaced instead
  // (which would make it vanish).
  bool isCandidate(Site* site) {
    if (site->helperBody) {
      return false;
    }
    for (auto* parent = site->parent; parent; parent = parent->parent) {
      if (isEligible(parent)) {
        return false;
      }
    }
    return true;
  }

  // Whether a site will be replaced, given the current decisions.
  bool isReplaced(Site* site) { return isEligible(site) && isCandidate(site); }

  Type getHelperResults(Expression* expr) {
    return expr->type == Type::unreachable ? Type::none : expr->type;
  }

  // If the function's body is just a tree whose holes are its params, in
  // order, note it as the helper for the tree's key.
  void noteExistingHelper(Function* func, Site& site) {
    if (site.holes != func->getNumParams() ||
        func->getResults() != getHelperResults(site.expr)) {
      return;
    }
    Index i = 0;
    for (auto& [local, get] : getHoles(site.expr)) {
      if (local != i || get->type != func->getLocalType(i)) {
        return;
      }
      i++;
    }
    if (site.key->helper) {
      // We already have a helper for this key. This function can just call it
      // like any other site.
      return;
    }
    site.helperBody = true;
    site.key->helper = func;
  }

  Function* createHelper(Site* site) {
    auto* body = ExpressionManipulator::copy(site->expr, *wasm);
    std::vector<Type> params;
    std::unordered_map<Index, Index> paramOfLocal;
    for (auto& [local, get] : getHoles(body)) {
      paramOfLocal[local] = params.size();
      params.push_back(get->type);
    }
    assert(params.size() == site->holes);
    for (auto* get : FindAll<LocalGet>(body).list) {
      get->index = paramOfLocal.at(get->index);
    }
    auto name = Names::getValidFunctionName(*wasm, "outlined");
    return wasm->addFunction(Builder::makeFunction(
      name, Signature(Type(params), getHelperResults(site->expr)), {}, body));
  }
};

} // anonymous namespace

Pass* createOutlineExpressionsPass() { return new OutlineExpressions(); }

} // namespace wasm
