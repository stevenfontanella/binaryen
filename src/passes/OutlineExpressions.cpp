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
// Once that converges, a final round also treats constants (i32.const etc.,
// global.get of an immutable global, ref.func) as holes, so that trees that
// differ only in their constants share a helper, for example
//
//   (struct.new $T (global.get $vtable) (i32.const 0) (f64.const 1.5))
//   (struct.new $T (global.get $vtable) (i32.const 0) (f64.const 2.5))
//
// A constant on which all the sites agree (the vtable and the 0 here) is
// "fixed": it stays inline in the helper. The others become parameters, passed
// at each site, in evaluation order with the local holes.
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

// Whether an expression is a constant that a round with constants as holes
// treats as a hole.
bool isConstHole(Module& wasm, Expression* curr) {
  if (curr->is<Const>() || curr->is<RefFunc>()) {
    return true;
  }
  if (auto* get = curr->dynCast<GlobalGet>()) {
    return !wasm.getGlobal(get->name)->mutable_;
  }
  return false;
}

// A leaf of a tree that may become a parameter of the helper: a local.get (the
// same local read several times is a single hole), or, in a round with
// constants as holes, a constant.
struct Hole {
  Expression* expr;
  bool isLocal;
};

// A group of sites with the same shape.
struct KeyInfo {
  std::vector<struct Site*> sites;
  // The function that performs this tree, if we found or created one.
  Function* helper = nullptr;
  // Whether we decided to outline the sites with this key.
  bool outline = false;
  // Whether the fields below have been computed.
  bool prepared = false;
  // For each hole position, whether all sites have the same constant there, so
  // that it stays inline in the helper instead of becoming a parameter.
  std::vector<bool> fixed;
  // The number of parameters the helper has: the local holes, and the
  // constant holes that are not fixed.
  Index numLocals = 0;
  Index numVarying = 0;
};

// An expression tree in a function body that we may outline.
struct Site {
  Expression* expr;
  // A hash of the tree's shape, with holes abstracted away. Trees with equal
  // shapes can share a helper.
  size_t hash = 0;
  // The estimated binary size of the tree.
  Index bytes = 0;
  // The nearest enclosing site, if any. Outlining that makes this site vanish.
  Site* parent = nullptr;
  // The key this site belongs to.
  KeyInfo* key = nullptr;
  // Whether this site is the entire body of the function that we use as the
  // helper for its shape (either one we created earlier, or by coincidence).
  bool helperBody = false;
  // The holes, in evaluation order, computed when the key is prepared.
  std::vector<Hole> holes;
  // The estimated size of the constants at positions that are not fixed, which
  // the site must pass as arguments.
  Index varyingBytes = 0;
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
  // Whether constants are holes in this round.
  bool constHoles;

  // Infos of the already-visited children of the expressions we are in the
  // middle of visiting.
  std::vector<NodeInfo> stack;

  Scanner(Module& wasm, FunctionInfo& info, Mode mode, bool constHoles)
    : wasm(wasm), info(info), mode(mode), constHoles(constHoles) {}

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
    if (curr->is<LocalGet>() || (constHoles && isConstHole(wasm, curr))) {
      // A hole. Only the kind and type matter.
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

// Finds the holes of a tree, in evaluation order: the distinct locals it reads
// (with the first local.get of each) and, if constants are holes, each
// constant.
struct HoleFinder
  : public PostWalker<HoleFinder, UnifiedExpressionVisitor<HoleFinder>> {
  Module& wasm;
  bool constHoles;
  std::vector<Hole> holes;

  HoleFinder(Module& wasm, bool constHoles)
    : wasm(wasm), constHoles(constHoles) {}

  void visitExpression(Expression* curr) {
    if (auto* get = curr->dynCast<LocalGet>()) {
      for (auto& hole : holes) {
        if (hole.isLocal && hole.expr->cast<LocalGet>()->index == get->index) {
          return;
        }
      }
      holes.push_back({curr, true});
    } else if (constHoles && isConstHole(wasm, curr)) {
      holes.push_back({curr, false});
    }
  }
};

std::vector<Hole> getHoles(Module& wasm, Expression* expr, bool constHoles) {
  HoleFinder finder(wasm, constHoles);
  finder.walk(expr);
  return std::move(finder.holes);
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
  // original leaves, which no replacement of a child could touch).
  void visitExpression(Expression* curr) {
    auto iter = replacements.find(curr);
    if (iter == replacements.end()) {
      return;
    }
    auto& replacement = iter->second;
    auto* site = replacement.site;
    auto& fixed = site->key->fixed;
    Builder builder(*getModule());
    std::vector<Expression*> args;
    for (Index i = 0; i < site->holes.size(); i++) {
      if (!fixed[i]) {
        args.push_back(site->holes[i].expr);
      }
    }
    assert(args.size() == site->key->numLocals + site->key->numVarying);
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
  // Then do the same with constants as holes.
  static const Index MaxConstRounds = 2;

  // Whether constants are holes in the current round.
  bool constHoles = false;

  void run(Module* module) override {
    wasm = module;

    auto mode = getMode();

    constHoles = false;
    for (Index round = 0; round < MaxRounds; round++) {
      if (!runRound(mode)) {
        break;
      }
    }
    constHoles = true;
    for (Index round = 0; round < MaxConstRounds; round++) {
      if (!runRound(mode)) {
        break;
      }
    }
  }

  // Scans, decides, and replaces once. Returns whether anything was replaced.
  bool runRound(Mode mode) {
    keys.clear();
    keysByHash.clear();

    ModuleUtils::ParallelFunctionAnalysis<FunctionInfo> analysis(
      *wasm, [&](Function* func, FunctionInfo& info) {
        if (func->imported()) {
          return;
        }
        Scanner(*wasm, info, mode, constHoles).walk(func->body);
        noteParents(func->body, info);
      });

    // Group the sites by key.
    for (auto& func : wasm->functions) {
      auto& info = analysis.map[func.get()];
      for (auto& site : info.sites) {
        site.key = getKey(&site);
        site.key->sites.push_back(&site);
      }
    }

    // Find existing helpers. This needs the keys to be complete, as whether a
    // function can be the helper for its key depends on all the sites.
    for (auto& func : wasm->functions) {
      auto& info = analysis.map[func.get()];
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
        bool outline = false;
        if (info.helper) {
          // The helper already exists, so any use of it is a win.
          for (auto* site : info.sites) {
            if (isCandidate(site)) {
              outline = true;
              break;
            }
          }
        } else if (info.sites.size() >= 2) {
          // Each site shrinks to a call with the holes as arguments, and the
          // helper contains a copy of the tree plus overhead.
          prepareKey(info);
          int64_t saved = 0;
          for (auto* site : info.sites) {
            if (isCandidate(site)) {
              saved += int64_t(site->bytes) - getCallBytes(site) -
                       LocalGetBytes * int64_t(info.numLocals) -
                       int64_t(site->varyingBytes);
            }
          }
          auto* site = info.sites[0];
          int64_t cost = int64_t(site->bytes) - int64_t(site->varyingBytes) +
                         LocalGetBytes * int64_t(info.numVarying) +
                         HelperOverheadBytes;
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
  // both trees (and, with constants as holes, apart from the constants, which
  // must have the same type).
  bool sameShape(Expression* left, Expression* right) {
    std::unordered_map<Index, Index> leftToRight, rightToLeft;
    bool mismatch = false;
    auto comparer = [&](Expression* left, Expression* right) {
      if (constHoles) {
        bool leftConst = isConstHole(*wasm, left);
        bool rightConst = isConstHole(*wasm, right);
        if (leftConst || rightConst) {
          if (!leftConst || !rightConst || left->_id != right->_id ||
              left->type != right->type) {
            mismatch = true;
          }
          return true;
        }
      }
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

  // Computes the holes of a key's sites, which constant holes are fixed (all
  // sites have the same constant there), and the resulting parameter counts
  // and per-site argument sizes.
  void prepareKey(KeyInfo& key) {
    if (key.prepared) {
      return;
    }
    key.prepared = true;
    for (auto* site : key.sites) {
      site->holes = getHoles(*wasm, site->expr, constHoles);
    }
    auto& first = key.sites[0]->holes;
    key.fixed.resize(first.size());
    for (Index i = 0; i < first.size(); i++) {
      key.fixed[i] = !first[i].isLocal;
    }
    for (auto* site : key.sites) {
      assert(site->holes.size() == first.size());
      for (Index i = 0; i < first.size(); i++) {
        if (key.fixed[i] &&
            !ExpressionAnalyzer::equal(first[i].expr, site->holes[i].expr)) {
          key.fixed[i] = false;
        }
      }
    }
    for (Index i = 0; i < first.size(); i++) {
      if (first[i].isLocal) {
        key.numLocals++;
      } else if (!key.fixed[i]) {
        key.numVarying++;
      }
    }
    for (auto* site : key.sites) {
      for (Index i = 0; i < first.size(); i++) {
        auto& hole = site->holes[i];
        if (!hole.isLocal && !key.fixed[i]) {
          site->varyingBytes += getShallowBytes(hole.expr);
        }
      }
    }
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
  // order, note it as the helper for the tree's key. With constants as holes,
  // all the constants in the body must be fixed, as the function has them
  // baked in.
  void noteExistingHelper(Function* func, Site& site) {
    if (func->getResults() != getHelperResults(site.expr)) {
      return;
    }
    auto& key = *site.key;
    prepareKey(key);
    if (key.numVarying != 0 || key.numLocals != func->getNumParams()) {
      return;
    }
    Index i = 0;
    for (auto& hole : site.holes) {
      if (!hole.isLocal) {
        continue;
      }
      auto* get = hole.expr->cast<LocalGet>();
      if (get->index != i || get->type != func->getLocalType(i)) {
        return;
      }
      i++;
    }
    if (key.helper) {
      // We already have a helper for this key. This function can just call it
      // like any other site.
      return;
    }
    site.helperBody = true;
    key.helper = func;
  }

  // Turns the non-fixed holes of a copied tree into local.gets of the params.
  struct HoleReplacer
    : public PostWalker<HoleReplacer, UnifiedExpressionVisitor<HoleReplacer>> {
    Module& wasm;
    std::unordered_map<Index, Index> paramOfLocal;
    std::unordered_map<Expression*, Index> paramOfConst;

    HoleReplacer(Module& wasm) : wasm(wasm) {}

    void visitExpression(Expression* curr) {
      if (auto* get = curr->dynCast<LocalGet>()) {
        get->index = paramOfLocal.at(get->index);
      } else if (auto iter = paramOfConst.find(curr);
                 iter != paramOfConst.end()) {
        replaceCurrent(Builder(wasm).makeLocalGet(iter->second, curr->type));
      }
    }
  };

  Function* createHelper(Site* site) {
    auto& key = *site->key;
    auto* body = ExpressionManipulator::copy(site->expr, *wasm);
    auto holes = getHoles(*wasm, body, constHoles);
    assert(holes.size() == site->holes.size());
    std::vector<Type> params;
    HoleReplacer replacer(*wasm);
    for (Index i = 0; i < holes.size(); i++) {
      if (key.fixed[i]) {
        continue;
      }
      auto& hole = holes[i];
      if (hole.isLocal) {
        replacer.paramOfLocal[hole.expr->cast<LocalGet>()->index] =
          params.size();
      } else {
        replacer.paramOfConst[hole.expr] = params.size();
      }
      params.push_back(hole.expr->type);
    }
    assert(params.size() == key.numLocals + key.numVarying);
    replacer.walk(body);
    auto name = Names::getValidFunctionName(*wasm, "outlined");
    return wasm->addFunction(Builder::makeFunction(
      name, Signature(Type(params), getHelperResults(site->expr)), {}, body));
  }
};

} // anonymous namespace

Pass* createOutlineExpressionsPass() { return new OutlineExpressions(); }

} // namespace wasm
