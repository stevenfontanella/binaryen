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
// return, return_call, try, pop, or stack switching, and no br/br_if/br_table/
// br_on_* to a label outside of it. Control flow like if, block, loop, select
// and drop is fine, and so are branches to labels inside the tree. Reading the
// holes before the tree (at the call) instead of inside it is fine as they have
// no effects, and nothing in the tree can change what they read.
//
// A tree of type none that writes a single local, like the loop
//
//   (loop $l
//     (if (i32.lt_s (local.get $i) (i32.const 3))
//       (then
//         (call $foo (local.get $x))
//         (local.set $i (i32.add (local.get $i) (i32.const 1)))
//         (br $l))))
//
// can also be outlined: the helper returns the final value of that local, and
// the site becomes (local.set $i (call $helper (local.get $i) (local.get $x))).
// If the tree reads the local, its incoming value is a parameter (as here);
// otherwise it is just a local of the helper. The local must have a defaultable
// type, so that reading it at the site is always valid.
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

#include <algorithm>
#include <cctype>
#include <deque>
#include <map>

#include "ir/effects.h"
#include "ir/find_all.h"
#include "ir/intrinsics.h"
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
// The local.set at a site whose tree writes a local, and the local.get of the
// final value in the helper.
const Index LocalSetBytes = 2;
// Function section entry, code size, local declarations, end opcode, and a
// share of a type section entry.
const Index HelperOverheadBytes = 6;
// Maximum number of live-out locals a site may write (1 returned on the Wasm
// stack, up to 3 returned via module-level scratch globals).
const Index MaxWrittenLocals = 4;

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
// same local read several times is a single hole), a passthrough live-out
// written local (expr == nullptr), or, in a round with constants as holes, a
// constant.
struct Hole {
  Expression* expr = nullptr;
  Index localIndex = 0;
  Type type = Type::none;
  bool isLocal = false;
};

// A live-out local a tree writes.
struct WrittenLocal {
  Index index;
  Type type;
};

// Read and write counts for a local inside a subtree or block slice.
struct LocalCount {
  Index index;
  Index reads = 0;
  Index writes = 0;
};

// A group of sites with the same shape.
struct KeyInfo {
  std::vector<struct Site*> sites;
  // The function that performs this tree, if we found or created one.
  Function* helper = nullptr;
  // For multi-write helpers (written.size() >= 2), the module-level scratch
  // globals used to pass written[1 .. k-1] back to the caller.
  SmallVector<Name, 3> scratchGlobals;
  // Whether we decided to outline the sites with this key.
  bool outline = false;
  // Whether this key holds leftover sites that were rejected during constant
  // sub-key splitting.
  bool disabled = false;
  // Whether the fields below have been computed.
  bool prepared = false;
  // For each hole position, whether all sites have the same constant there, so
  // that it stays inline in the helper instead of becoming a parameter.
  std::vector<bool> fixed;
  // For each varying constant hole position, the index of an earlier varying
  // constant hole that always has the equal constant across all sites in this
  // key, or Index(-1) if unique.
  std::vector<Index> dupOf;
  // The number of parameters the helper has: the local holes, and the unique
  // constant holes that are not fixed.
  Index numLocals = 0;
  Index numVarying = 0;
  Index numDupVarying = 0;
};

// An expression tree (or contiguous block slice) in a function body that we may
// outline.
struct Site {
  Expression* expr;
  Function* func;
  // If sliceLen > 0, this site is the contiguous slice
  // expr->cast<Block>()->list[sliceOffset .. sliceOffset + sliceLen - 1]
  // rather than expr itself.
  Index sliceOffset = 0;
  Index sliceLen = 0;
  // If true, the last statement of the slice is a LocalSet whose value is
  // returned by the helper while the LocalSet itself stays at the call site.
  bool trailingSet = false;
  // A hash of the tree's shape, with holes abstracted away. Trees with equal
  // shapes can share a helper.
  size_t hash = 0;
  // The estimated binary size of the tree.
  Index bytes = 0;
  // The live-out locals the tree writes, if any. The tree then has type none,
  // and the helper returns written[0]'s final value on the stack and writes
  // written[1..] into module-level scratch globals.
  SmallVector<WrittenLocal, 4> written;
  // Defaultable non-parameter locals whose reads and writes in the entire
  // function occur 100% inside this site. Allocated as local vars in the helper
  // with 0 call-site overhead.
  SmallVector<Index, 2> internalLocals;
  // Live-out written locals that are read inside the site only after being
  // unconditionally written on all paths. Allocated as local vars in the helper
  // rather than passed as input parameters.
  SmallVector<Index, 2> overwriteLocals;
  // Live-out written locals that are not read in the site, are not
  // unconditionally written on all paths through the site, and may hold a
  // non-default value before the site. Passed as trailing input parameters so
  // their incoming values are preserved on untaken paths.
  SmallVector<Index, 2> passthroughLocals;
  // The nearest enclosing single-expression site, if any. Outlining that makes
  // this site vanish.
  Site* parent = nullptr;
  // Slice sites that enclose this site with no intermediate single-expression
  // site in between.
  SmallVector<Site*, 4> enclosingSlices;
  // Higher-priority slice sites of the same Block that overlap this slice site.
  SmallVector<Site*, 4> overlappingSlices;
  // The key this site belongs to.
  KeyInfo* key = nullptr;
  // Whether this site is the entire body of the function that we use as the
  // helper for its shape (either one we created earlier, or by coincidence).
  bool helperBody = false;
  // Updated during the fixed-point decision loop.
  bool killedByAncestor = false;
  bool candidate = false;
  bool replaced = false;
  bool selectedForKey = false;
  // The holes, in evaluation order, computed when the key is prepared.
  std::vector<Hole> holes;
  bool holesComputed = false;
  // The estimated size of the constants at unique varying positions, which
  // the site must pass as arguments.
  Index varyingBytes = 0;
  // The estimated size of the constants at all non-fixed positions, which are
  // replaced by local.get in the helper body.
  Index helperVaryingBytes = 0;

  bool isInternalLocal(Index idx) const {
    for (Index internalIdx : internalLocals) {
      if (internalIdx == idx) {
        return true;
      }
    }
    return false;
  }

  bool isOverwriteLocal(Index idx) const {
    for (Index overIdx : overwriteLocals) {
      if (overIdx == idx) {
        return true;
      }
    }
    return false;
  }

  bool isPassthroughLocal(Index idx) const {
    for (Index passIdx : passthroughLocals) {
      if (passIdx == idx) {
        return true;
      }
    }
    return false;
  }

  bool isWrittenLocal(Index idx) const {
    for (auto& w : written) {
      if (w.index == idx) {
        return true;
      }
    }
    return false;
  }

  Expression* getSliceExpr(Index idx) const {
    Expression* e = expr->cast<Block>()->list[sliceOffset + idx];
    if (trailingSet && idx == sliceLen - 1) {
      return e->cast<LocalSet>()->value;
    }
    return e;
  }

  Type getType() const {
    if (sliceLen == 0) {
      return expr->type;
    }
    return getSliceExpr(sliceLen - 1)->type;
  }
};

// Information about a tree, computed bottom-up.
struct NodeInfo {
  size_t hash = 0;
  size_t setValueHash = 0;
  Index bytes = 0;
  // Per-local read and write counts within this subtree.
  SmallVector<LocalCount, 4> localCounts;
  // The distinct locals the tree reads, in order of first use.
  SmallVector<Index, 4> locals;
  // The distinct locals the tree writes.
  SmallVector<Index, 2> written;
  // Locals unconditionally written on every normal-exit path of this subtree.
  SmallVector<Index, 2> mustWrite;
  // Locals that may be read before being written in this subtree.
  SmallVector<Index, 4> readBeforeWrite;
  // The labels of branches in the tree that no block or loop in the tree
  // defines.
  SmallVector<Name, 2> openLabels;
  // Whether the root has children, and whether all of them are leaves.
  bool hasChildren = false;
  bool trivialChildren = true;
  // Whether the tree can be moved into a helper.
  bool movable = true;
  // Whether the tree contains a call, allocation, or throw.
  bool interesting = false;
  // Whether the tree contains a once-function guard, see below.
  bool hasGuard = false;
  // Whether this expression is a LocalSet whose value is movable and has no
  // local writes or open labels.
  bool hasCleanSetValue = false;
};

struct FunctionInfo {
  // A deque so that pointers to sites remain valid as we add more.
  std::deque<Site> sites;
  // Demanded types for LocalGet nodes when canonicalizing local holes.
  std::unordered_map<LocalGet*, Type> demandedTypes;
};

Type getChildDemandedType(Module& wasm, Expression* parent, Expression* child) {
  if (!parent || !child || child->type == Type::unreachable ||
      !child->type.isRef() || child->type.isNull()) {
    return child ? child->type : Type::none;
  }
  if (auto* call = parent->dynCast<Call>()) {
    auto* callee = wasm.getFunctionOrNull(call->target);
    if (!callee || call->operands.size() != callee->getNumParams()) {
      return child->type;
    }
    for (Index i = 0; i < call->operands.size(); i++) {
      if (call->operands[i] == child) {
        return callee->getLocalType(i);
      }
    }
    return child->type;
  }
  if (auto* cr = parent->dynCast<CallRef>()) {
    if (!cr->target || !cr->target->type.isSignature() ||
        cr->target->type.isNull()) {
      return child->type;
    }
    if (cr->target == child) {
      return Type(cr->target->type.getHeapType(), Nullable);
    }
    auto params = cr->target->type.getHeapType().getSignature().params;
    if (cr->operands.size() != params.size()) {
      return child->type;
    }
    for (Index i = 0; i < cr->operands.size(); i++) {
      if (cr->operands[i] == child) {
        return params[i];
      }
    }
    return child->type;
  }
  if (auto* ci = parent->dynCast<CallIndirect>()) {
    if (!ci->heapType.isSignature()) {
      return child->type;
    }
    auto params = ci->heapType.getSignature().params;
    if (ci->operands.size() != params.size()) {
      return child->type;
    }
    for (Index i = 0; i < ci->operands.size(); i++) {
      if (ci->operands[i] == child) {
        return params[i];
      }
    }
    return child->type;
  }
  if (auto* sn = parent->dynCast<StructNew>()) {
    if (!sn->isWithDefault() && sn->type.isStruct()) {
      const auto& fields = sn->type.getHeapType().getStruct().fields;
      if (sn->operands.size() == fields.size()) {
        for (Index i = 0; i < sn->operands.size(); i++) {
          if (sn->operands[i] == child) {
            return fields[i].type;
          }
        }
      }
    }
    return child->type;
  }
  if (auto* sg = parent->dynCast<StructGet>()) {
    if (sg->ref == child && sg->ref->type.isStruct() &&
        !sg->ref->type.isNull()) {
      return Type(sg->ref->type.getHeapType(), Nullable);
    }
    return child->type;
  }
  if (auto* ss = parent->dynCast<StructSet>()) {
    if (!ss->ref || !ss->ref->type.isStruct() || ss->ref->type.isNull()) {
      return child->type;
    }
    if (ss->ref == child) {
      return Type(ss->ref->type.getHeapType(), Nullable);
    }
    if (ss->value == child) {
      return ss->ref->type.getHeapType().getStruct().fields[ss->index].type;
    }
    return child->type;
  }
  if (auto* an = parent->dynCast<ArrayNew>()) {
    if (an->init == child && an->type.isArray()) {
      return an->type.getHeapType().getArray().element.type;
    }
    return child->type;
  }
  if (auto* anf = parent->dynCast<ArrayNewFixed>()) {
    if (anf->type.isArray()) {
      Type elemType = anf->type.getHeapType().getArray().element.type;
      for (auto* v : anf->values) {
        if (v == child) {
          return elemType;
        }
      }
    }
    return child->type;
  }
  if (auto* ag = parent->dynCast<ArrayGet>()) {
    if (ag->ref == child && ag->ref->type.isArray() &&
        !ag->ref->type.isNull()) {
      return Type(ag->ref->type.getHeapType(), Nullable);
    }
    return child->type;
  }
  if (auto* as = parent->dynCast<ArraySet>()) {
    if (!as->ref || !as->ref->type.isArray() || as->ref->type.isNull()) {
      return child->type;
    }
    if (as->ref == child) {
      return Type(as->ref->type.getHeapType(), Nullable);
    }
    if (as->value == child) {
      return as->ref->type.getHeapType().getArray().element.type;
    }
    return child->type;
  }
  if (auto* al = parent->dynCast<ArrayLen>()) {
    if (al->ref == child) {
      return Type(HeapType::array, Nullable);
    }
    return child->type;
  }
  if (auto* gs = parent->dynCast<GlobalSet>()) {
    if (gs->value == child) {
      if (auto* g = wasm.getGlobalOrNull(gs->name)) {
        return g->type;
      }
    }
    return child->type;
  }
  if (auto* ra = parent->dynCast<RefAs>()) {
    if (ra->op == RefAsNonNull && ra->value == child) {
      return Type(
        child->type.getHeapType(), Nullable, child->type.getExactness());
    }
    return child->type;
  }
  if (auto* rin = parent->dynCast<RefIsNull>()) {
    if (rin->value == child) {
      return Type(child->type.getHeapType().getTop(), Nullable);
    }
    return child->type;
  }
  if (auto* req = parent->dynCast<RefEq>()) {
    if (req->left == child || req->right == child) {
      return Type(HeapType::eq, Nullable);
    }
    return child->type;
  }
  if (auto* rt = parent->dynCast<RefTest>()) {
    if (rt->ref == child && rt->castType.isRef()) {
      return Type(rt->castType.getHeapType().getTop(), Nullable);
    }
    return child->type;
  }
  if (auto* rc = parent->dynCast<RefCast>()) {
    if (!rc->desc && rc->ref == child && rc->type.isRef() &&
        rc->type.isInexact()) {
      return Type(rc->type.getHeapType().getTop(), Nullable);
    }
    return child->type;
  }
  return child->type;
}

// Counts total reads and writes per local across an entire function body.
struct LocalUseCounter : public PostWalker<LocalUseCounter> {
  std::vector<Index> reads;
  std::vector<Index> writes;

  LocalUseCounter(Index numLocals)
    : reads(numLocals, 0), writes(numLocals, 0) {}

  void visitLocalGet(LocalGet* curr) { reads[curr->index]++; }
  void visitLocalSet(LocalSet* curr) { writes[curr->index]++; }
};

// What to outline, see the top of the file.
enum class Mode { Calls, Guards, Trees, All };

struct Scanner : public PostWalker<Scanner, UnifiedExpressionVisitor<Scanner>> {
  Module& wasm;
  Function* func;
  FunctionInfo& info;
  Mode mode;
  // Whether constants are holes in this round.
  bool constHoles;
  // Whether reference-typed local holes are canonicalized to their demanded
  // types.
  bool canonicalizeLocalHoles;
  std::vector<Index> totalReads;
  std::vector<Index> totalWrites;
  std::vector<Expression*> parentStack;
  Index loopDepth = 0;

  // Infos of the already-visited children of the expressions we are in the
  // middle of visiting.
  std::vector<NodeInfo> stack;

  Scanner(Module& wasm,
          Function* func,
          FunctionInfo& info,
          Mode mode,
          bool constHoles,
          bool canonicalizeLocalHoles)
    : wasm(wasm), func(func), info(info), mode(mode), constHoles(constHoles),
      canonicalizeLocalHoles(canonicalizeLocalHoles) {
    LocalUseCounter counter(func->getNumLocals());
    counter.walk(func->body);
    totalReads = std::move(counter.reads);
    totalWrites = std::move(counter.writes);
  }

  static void scan(Scanner* self, Expression** currp) {
    PostWalker<Scanner, UnifiedExpressionVisitor<Scanner>>::scan(self, currp);
    self->parentStack.push_back(*currp);
    if ((*currp)->is<Loop>()) {
      self->loopDepth++;
    }
  }

  void visitExpression(Expression* curr) {
    assert(!parentStack.empty() && parentStack.back() == curr);
    parentStack.pop_back();
    if (curr->is<Loop>()) {
      assert(loopDepth > 0);
      loopDepth--;
    }
    if (canonicalizeLocalHoles) {
      if (auto* get = curr->dynCast<LocalGet>()) {
        Expression* par = parentStack.empty() ? nullptr : parentStack.back();
        Type demanded = getChildDemandedType(wasm, par, get);
        if (!Type::isSubType(get->type, demanded)) {
          demanded = get->type;
        }
        info.demandedTypes[get] = demanded;
      }
    }
    NodeInfo node;
    node.hash = getShallowHash(curr);
    node.bytes = getShallowBytes(curr);
    Index numChildren = ChildIterator(curr).getNumChildren();
    assert(stack.size() >= numChildren);
    Index stackBase = stack.size() - numChildren;
    node.hasChildren = numChildren > 0;
    for (Index i = stackBase; i < stack.size(); i++) {
      auto& child = stack[i];
      if (child.hasChildren) {
        node.trivialChildren = false;
      }
      hash_combine(node.hash, child.hash);

      node.bytes += child.bytes;
      addLocalCounts(node.localCounts, child.localCounts);
      for (auto local : child.locals) {
        noteUnique(node.locals, local);
      }
      for (auto local : child.written) {
        noteUnique(node.written, local);
      }
      for (auto label : child.openLabels) {
        noteUnique(node.openLabels, label);
      }
      node.movable = node.movable && child.movable;
      node.interesting = node.interesting || child.interesting;
      node.hasGuard = node.hasGuard || child.hasGuard;
    }
    if (auto* iff = curr->dynCast<If>()) {
      auto& condInfo = stack[stackBase];
      node.readBeforeWrite = condInfo.readBeforeWrite;
      if (condInfo.openLabels.empty()) {
        node.mustWrite = condInfo.mustWrite;
      }
      auto& trueInfo = stack[stackBase + 1];
      for (Index x : trueInfo.readBeforeWrite) {
        if (!contains(node.mustWrite, x)) {
          noteUnique(node.readBeforeWrite, x);
        }
      }
      if (iff->ifFalse) {
        auto& falseInfo = stack[stackBase + 2];
        for (Index x : falseInfo.readBeforeWrite) {
          if (!contains(node.mustWrite, x)) {
            noteUnique(node.readBeforeWrite, x);
          }
        }
        if (condInfo.openLabels.empty() && trueInfo.openLabels.empty() &&
            falseInfo.openLabels.empty()) {
          for (Index x : trueInfo.mustWrite) {
            if (contains(falseInfo.mustWrite, x)) {
              noteUnique(node.mustWrite, x);
            }
          }
        }
      }
    } else if (curr->is<Loop>()) {
      auto& bodyInfo = stack[stackBase];
      node.readBeforeWrite = bodyInfo.readBeforeWrite;
      if (bodyInfo.openLabels.empty()) {
        node.mustWrite = bodyInfo.mustWrite;
      }
    } else {
      bool seenBranch = false;
      for (Index i = stackBase; i < stack.size(); i++) {
        auto& child = stack[i];
        for (Index x : child.readBeforeWrite) {
          if (!contains(node.mustWrite, x)) {
            noteUnique(node.readBeforeWrite, x);
          }
        }
        if (!seenBranch && child.openLabels.empty()) {
          for (Index x : child.mustWrite) {
            noteUnique(node.mustWrite, x);
          }
        } else {
          seenBranch = true;
        }
      }
    }
    if (auto* set = curr->dynCast<LocalSet>()) {
      // We compare the values of local.sets separately (see sameShape), which
      // only works if they do not branch out of themselves.
      auto& valInfo = stack.back();
      if (!valInfo.openLabels.empty()) {
        node.movable = false;
      }
      if (valInfo.movable && valInfo.openLabels.empty() &&
          valInfo.written.empty() && set->value->type.isConcrete() &&
          !set->value->type.isTuple()) {
        node.hasCleanSetValue = true;
        node.setValueHash = valInfo.hash;
      }
      addLocalWrite(node.localCounts, set->index, 1);
      noteUnique(node.written, set->index);
      if (valInfo.openLabels.empty()) {
        noteUnique(node.mustWrite, set->index);
      }
    }
    bool recordedFullBlockSlice = false;
    if (auto* block = curr->dynCast<Block>()) {
      recordedFullBlockSlice = scanBlockSlices(block, stackBase);
    }
    stack.resize(stackBase);
    if (auto* get = curr->dynCast<LocalGet>()) {
      addLocalRead(node.localCounts, get->index, 1);
      noteUnique(node.locals, get->index);
      noteUnique(node.readBeforeWrite, get->index);
    }
    noteLabels(curr, node);
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
    if (!recordedFullBlockSlice) {
      recordSite(curr, 0, 0, false, curr->type, node);
    }
  }

  static bool isScratchGlobalReadStmt(Expression* e) {
    if (auto* set = e->dynCast<LocalSet>()) {
      if (auto* gget = set->value->dynCast<GlobalGet>()) {
        return gget->name.startsWith("oe$out_");
      }
    }
    return false;
  }

  bool scanBlockSlices(Block* block, Index stackBase) {
    bool recordedFull = false;
    Index n = block->list.size();
    for (Index k = 2; k <= 6; k++) {
      if (n < k) {
        continue;
      }
      for (Index i = n - k + 1; i > 0;) {
        i--;
        if (isScratchGlobalReadStmt(block->list[i]) ||
            (i + k < n && isScratchGlobalReadStmt(block->list[i + k]))) {
          continue;
        }
        Expression* lastExpr = block->list[i + k - 1];
        Type sliceType = lastExpr->type;
        if (sliceType.isTuple()) {
          continue;
        }
        if (i + k < n && sliceType != Type::none &&
            sliceType != Type::unreachable) {
          continue;
        }
        bool typesOk = true;
        bool prefixWrites = false;
        for (Index j = i; j < i + k - 1; j++) {
          if (block->list[j]->type != Type::none) {
            typesOk = false;
            break;
          }
          if (!stack[stackBase + j].written.empty()) {
            prefixWrites = true;
          }
        }
        if (!typesOk) {
          continue;
        }
        auto& lastChild = stack[stackBase + i + k - 1];
        bool trailingSet = !prefixWrites && lastChild.hasCleanSetValue;
        if (n == k && !trailingSet) {
          continue;
        }
        if (trailingSet) {
          sliceType = lastExpr->cast<LocalSet>()->value->type;
        }
        NodeInfo sliceNode;
        sliceNode.hash = wasm::hash(0);
        rehash(sliceNode.hash, Expression::Id(Expression::BlockId));
        rehash(sliceNode.hash, sliceType.getID());
        sliceNode.hasChildren = true;
        sliceNode.trivialChildren = false;
        for (Index j = i; j < i + k; j++) {
          auto& child = stack[stackBase + j];
          if (!child.movable || !child.openLabels.empty()) {
            sliceNode.movable = false;
            break;
          }
          addLocalCounts(sliceNode.localCounts, child.localCounts);
          for (Index x : child.readBeforeWrite) {
            if (!contains(sliceNode.mustWrite, x)) {
              noteUnique(sliceNode.readBeforeWrite, x);
            }
          }
          if (trailingSet && j == i + k - 1) {
            auto* set = lastExpr->cast<LocalSet>();
            subLocalWrite(sliceNode.localCounts, set->index, 1);
            for (Index x : child.mustWrite) {
              if (x != set->index) {
                noteUnique(sliceNode.mustWrite, x);
              }
            }
            hash_combine(sliceNode.hash, child.setValueHash);
            assert(child.bytes >= LocalSetBytes);
            sliceNode.bytes += child.bytes - LocalSetBytes;
          } else {
            for (Index x : child.mustWrite) {
              noteUnique(sliceNode.mustWrite, x);
            }
            hash_combine(sliceNode.hash, child.hash);
            sliceNode.bytes += child.bytes;
            for (auto local : child.written) {
              noteUnique(sliceNode.written, local);
            }
          }
          for (auto local : child.locals) {
            noteUnique(sliceNode.locals, local);
          }
          sliceNode.interesting = sliceNode.interesting || child.interesting;
          sliceNode.hasGuard = sliceNode.hasGuard || child.hasGuard;
        }
        if (!sliceNode.movable) {
          continue;
        }
        recordSite(block, i, k, trailingSet, sliceType, sliceNode);
        if (n == k) {
          recordedFull = true;
        }
      }
    }
    return recordedFull;
  }

  static bool isBareMultiWriteCallSite(Expression* curr,
                                       Index sliceOffset,
                                       Index sliceLen) {
    auto* block = curr->dynCast<Block>();
    if (!block) {
      return false;
    }
    Index offset = sliceLen > 0 ? sliceOffset : 0;
    Index len = sliceLen > 0 ? sliceLen : block->list.size();
    if (len < 2 || !isScratchGlobalReadStmt(block->list[offset + 1])) {
      return false;
    }
    auto* firstSet = block->list[offset]->dynCast<LocalSet>();
    if (!firstSet || !firstSet->value->is<Call>()) {
      return false;
    }
    for (Index j = 2; j < len; j++) {
      if (!isScratchGlobalReadStmt(block->list[offset + j])) {
        return false;
      }
    }
    return true;
  }

  void recordSite(Expression* curr,
                  Index sliceOffset,
                  Index sliceLen,
                  bool trailingSet,
                  Type type,
                  const NodeInfo& node) {
    if (!node.movable || !node.openLabels.empty()) {
      return;
    }
    // A local.set itself is never worth outlining: its value is a better
    // tree (it does not need to pass and return the local).
    if (sliceLen == 0 && curr->is<LocalSet>()) {
      return;
    }
    if (isBareMultiWriteCallSite(curr, sliceOffset, sliceLen)) {
      return;
    }
    // A single instruction with trivial operands, like a call with constant
    // arguments or a struct.get of a local.
    bool single = sliceLen == 0 && node.trivialChildren &&
                  !Properties::isControlFlowStructure(curr);
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
    // Identify function-internal scratch locals whose reads and writes in the
    // enclosing function occur 100% inside this candidate (and cannot carry
    // state across an enclosing loop).
    SmallVector<Index, 2> internalLocals;
    for (auto& lc : node.localCounts) {
      if (func->isVar(lc.index) &&
          func->getLocalType(lc.index).isDefaultable() &&
          lc.reads == totalReads[lc.index] &&
          lc.writes == totalWrites[lc.index] &&
          (loopDepth == 0 || !contains(node.readBeforeWrite, lc.index))) {
        internalLocals.push_back(lc.index);
      }
    }
    SmallVector<WrittenLocal, 4> written;
    for (Index index : node.written) {
      if (contains(internalLocals, index)) {
        continue;
      }
      if (type != Type::none || written.size() >= MaxWrittenLocals) {
        return;
      }
      auto localType = func->getLocalType(index);
      if (!localType.isDefaultable()) {
        return;
      }
      written.push_back({index, localType});
    }
    SmallVector<Index, 2> overwriteLocals;
    Index numInputLocals = 0;
    for (Index index : node.locals) {
      if (contains(internalLocals, index)) {
        continue;
      }
      if (!contains(node.readBeforeWrite, index) &&
          contains(node.mustWrite, index)) {
        overwriteLocals.push_back(index);
        continue;
      }
      numInputLocals++;
    }
    SmallVector<Index, 2> passthroughLocals;
    for (auto& w : written) {
      if (!contains(node.locals, w.index) &&
          !contains(node.mustWrite, w.index)) {
        auto lc = getLocalCount(node.localCounts, w.index);
        if (func->isParam(w.index) || lc.writes < totalWrites[w.index] ||
            loopDepth > 0) {
          passthroughLocals.push_back(w.index);
          numInputLocals++;
        }
      }
    }
    // Nothing to gain without saving at least a byte per site.
    Index siteBytes = LocalGetBytes * numInputLocals + CallBytes;
    if (!written.empty()) {
      siteBytes += LocalSetBytes + 4 * (written.size() - 1);
    }
    if (node.bytes <= siteBytes) {
      return;
    }
    // A constant expression is better left for other passes to hoist (and for
    // allocations, OutlineAllocations handles them).
    if (sliceLen == 0 && !node.interesting &&
        Properties::isValidConstantExpression(wasm, curr)) {
      return;
    }
    size_t siteHash = node.hash;
    rehash(siteHash, written.size());
    rehash(siteHash, internalLocals.size());
    rehash(siteHash, passthroughLocals.size());

    info.sites.emplace_back();
    auto& site = info.sites.back();
    site.expr = curr;
    site.func = func;
    site.sliceOffset = sliceOffset;
    site.sliceLen = sliceLen;
    site.trailingSet = trailingSet;
    site.hash = siteHash;
    site.bytes = node.bytes;
    site.written = std::move(written);
    site.internalLocals = std::move(internalLocals);
    site.overwriteLocals = std::move(overwriteLocals);
    site.passthroughLocals = std::move(passthroughLocals);
  }

  template<typename T, typename V> bool contains(const T& vec, V value) {
    for (auto other : vec) {
      if (other == value) {
        return true;
      }
    }
    return false;
  }

  template<typename T, typename V> void noteUnique(T& vec, V value) {
    if (!contains(vec, value)) {
      vec.push_back(value);
    }
  }

  void addLocalRead(SmallVector<LocalCount, 4>& counts, Index index, Index n) {
    for (auto& lc : counts) {
      if (lc.index == index) {
        lc.reads += n;
        return;
      }
    }
    counts.push_back({index, n, 0});
  }

  void addLocalWrite(SmallVector<LocalCount, 4>& counts, Index index, Index n) {
    for (auto& lc : counts) {
      if (lc.index == index) {
        lc.writes += n;
        return;
      }
    }
    counts.push_back({index, 0, n});
  }

  void subLocalWrite(SmallVector<LocalCount, 4>& counts, Index index, Index n) {
    for (Index i = 0; i < counts.size(); i++) {
      if (counts[i].index == index) {
        assert(counts[i].writes >= n);
        counts[i].writes -= n;
        if (counts[i].reads == 0 && counts[i].writes == 0) {
          counts.erase(counts.begin() + i, counts.begin() + i + 1);
        }
        return;
      }
    }
    WASM_UNREACHABLE("missing local count");
  }

  void addLocalCounts(SmallVector<LocalCount, 4>& dst,
                      const SmallVector<LocalCount, 4>& src) {
    for (auto& lc : src) {
      bool found = false;
      for (auto& d : dst) {
        if (d.index == lc.index) {
          d.reads += lc.reads;
          d.writes += lc.writes;
          found = true;
          break;
        }
      }
      if (!found) {
        dst.push_back(lc);
      }
    }
  }

  LocalCount getLocalCount(const SmallVector<LocalCount, 4>& counts,
                           Index index) {
    for (auto& lc : counts) {
      if (lc.index == index) {
        return lc;
      }
    }
    return {index, 0, 0};
  }

  // Notes the labels a branch uses, and removes the one a block or loop
  // defines.
  void noteLabels(Expression* curr, NodeInfo& node) {
    Name defined;
    if (auto* block = curr->dynCast<Block>()) {
      defined = block->name;
    } else if (auto* loop = curr->dynCast<Loop>()) {
      defined = loop->name;
    } else if (auto* br = curr->dynCast<Break>()) {
      noteUnique(node.openLabels, br->name);
    } else if (auto* sw = curr->dynCast<Switch>()) {
      for (auto target : sw->targets) {
        noteUnique(node.openLabels, target);
      }
      noteUnique(node.openLabels, sw->default_);
    } else if (auto* brOn = curr->dynCast<BrOn>()) {
      noteUnique(node.openLabels, brOn->name);
    }
    if (defined.is()) {
      for (Index i = 0; i < node.openLabels.size(); i++) {
        if (node.openLabels[i] == defined) {
          node.openLabels[i] = node.openLabels.back();
          node.openLabels.pop_back();
          break;
        }
      }
    }
  }

  size_t getShallowHash(Expression* curr) {
    if (auto* get = curr->dynCast<LocalGet>()) {
      auto digest = hash(curr->_id);
      Type t = canonicalizeLocalHoles ? info.demandedTypes.at(get) : get->type;
      rehash(digest, t.getID());
      return digest;
    }
    if (curr->is<LocalSet>() || (constHoles && isConstHole(wasm, curr))) {
      // A hole (or a write to one). Only the kind and type matter.
      auto digest = hash(curr->_id);
      rehash(digest, curr->type.getID());
      return digest;
    }
    // Labels differ between sites, so hash branches without them (the
    // comparison checks they match up with the blocks and loops in the tree).
    if (auto* br = curr->dynCast<Break>()) {
      auto digest = hash(curr->_id);
      rehash(digest, curr->type.getID());
      rehash(digest, !!br->condition);
      rehash(digest, !!br->value);
      return digest;
    }
    if (auto* sw = curr->dynCast<Switch>()) {
      auto digest = hash(curr->_id);
      rehash(digest, curr->type.getID());
      rehash(digest, sw->targets.size());
      rehash(digest, !!sw->value);
      return digest;
    }
    if (auto* brOn = curr->dynCast<BrOn>()) {
      auto digest = hash(curr->_id);
      rehash(digest, curr->type.getID());
      rehash(digest, brOn->op);
      rehash(digest, brOn->castType.getID());
      return digest;
    }
    return ExpressionAnalyzer::shallowHash(curr);
  }

  // Whether an expression can be moved verbatim into a helper function,
  // assuming its children can be. Branches and local writes are checked
  // separately, see above.
  bool isMovable(Expression* curr) {
    if (curr->type.isTuple()) {
      return false;
    }
    switch (curr->_id) {
      case Expression::ReturnId:
      case Expression::TryId:
      case Expression::TryTableId:
      case Expression::RethrowId:
      case Expression::PopId:
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
      case Expression::GlobalSetId:
        return !curr->cast<GlobalSet>()->name.startsWith("oe$out_");
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

// Finds the holes of a tree, in evaluation order: the distinct non-internal
// locals it reads (with the first local.get of each), any passthrough written
// locals, and, if constants are holes, each constant.
struct HoleFinder
  : public PostWalker<HoleFinder, UnifiedExpressionVisitor<HoleFinder>> {
  Module& wasm;
  const Site* site;
  bool constHoles;
  const std::unordered_map<LocalGet*, Type>* demandedTypes;
  std::vector<Hole> holes;

  HoleFinder(Module& wasm,
             const Site* site,
             bool constHoles,
             const std::unordered_map<LocalGet*, Type>* demandedTypes = nullptr)
    : wasm(wasm), site(site), constHoles(constHoles),
      demandedTypes(demandedTypes) {}

  void visitExpression(Expression* curr) {
    if (auto* get = curr->dynCast<LocalGet>()) {
      if (site->isInternalLocal(get->index) ||
          site->isOverwriteLocal(get->index)) {
        return;
      }
      Type hType = get->type;
      if (demandedTypes && !site->isWrittenLocal(get->index)) {
        if (auto it = demandedTypes->find(get); it != demandedTypes->end()) {
          hType = it->second;
        }
      }
      for (auto& hole : holes) {
        if (hole.isLocal && hole.localIndex == get->index) {
          hole.type = Type::getGreatestLowerBound(hole.type, hType);
          return;
        }
      }
      holes.push_back({curr, get->index, hType, true});
    } else if (constHoles && isConstHole(wasm, curr)) {
      holes.push_back({curr, 0, curr->type, false});
    }
  }
};

void appendPassthroughHoles(const Site* site, std::vector<Hole>& holes) {
  for (Index passIdx : site->passthroughLocals) {
    holes.push_back(
      {nullptr, passIdx, site->func->getLocalType(passIdx), true});
  }
}

std::vector<Hole>
getHoles(Module& wasm,
         Expression* expr,
         const Site* site,
         bool constHoles,
         const std::unordered_map<LocalGet*, Type>* demandedTypes = nullptr) {
  HoleFinder finder(wasm, site, constHoles, demandedTypes);
  finder.walk(expr);
  appendPassthroughHoles(site, finder.holes);
  return std::move(finder.holes);
}

std::vector<Hole>
getHoles(Module& wasm,
         const Site* site,
         bool constHoles,
         const std::unordered_map<LocalGet*, Type>* demandedTypes = nullptr) {
  if (site->sliceLen == 0) {
    return getHoles(wasm, site->expr, site, constHoles, demandedTypes);
  }
  HoleFinder finder(wasm, site, constHoles, demandedTypes);
  for (Index i = 0; i < site->sliceLen; i++) {
    Expression* elem = site->getSliceExpr(i);
    finder.walk(elem);
  }
  appendPassthroughHoles(site, finder.holes);
  return std::move(finder.holes);
}

// Information about a replacement of a site with a call to a helper.
struct Replacement {
  Name helper;
  const Site* site;
};

using Replacements = std::unordered_map<Expression*, Replacement>;
using SliceReplacements = std::unordered_map<Block*, std::vector<Replacement>>;

struct Replacer : public WalkerPass<
                    PostWalker<Replacer, UnifiedExpressionVisitor<Replacer>>> {
  bool isFunctionParallel() override { return true; }

  bool requiresNonNullableLocalFixups() override { return false; }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<Replacer>(replacements, sliceReplacements);
  }

  Replacer(const Replacements& replacements,
           const SliceReplacements& sliceReplacements)
    : replacements(replacements), sliceReplacements(sliceReplacements) {}

  Expression* makeReplacementExpr(const Replacement& replacement, Type type) {
    auto* site = replacement.site;
    auto& fixed = site->key->fixed;
    auto& dupOf = site->key->dupOf;
    Builder builder(*getModule());
    std::vector<Expression*> args;
    for (Index i = 0; i < site->holes.size(); i++) {
      if (!fixed[i] && dupOf[i] == Index(-1)) {
        auto& hole = site->holes[i];
        args.push_back(hole.expr
                         ? hole.expr
                         : builder.makeLocalGet(hole.localIndex, hole.type));
      }
    }
    assert(args.size() == site->key->numLocals + site->key->numVarying);
    if (type == Type::unreachable) {
      return builder.makeSequence(
        builder.makeCall(replacement.helper, args, Type::none),
        builder.makeUnreachable());
    }
    if (!site->written.empty()) {
      auto& firstWritten = site->written[0];
      Expression* firstSet = builder.makeLocalSet(
        firstWritten.index,
        builder.makeCall(replacement.helper, args, firstWritten.type));
      if (site->written.size() == 1) {
        return firstSet;
      }
      assert(site->key->scratchGlobals.size() == site->written.size() - 1);
      std::vector<Expression*> stmts;
      stmts.reserve(site->written.size());
      stmts.push_back(firstSet);
      for (Index j = 1; j < site->written.size(); j++) {
        auto& w = site->written[j];
        stmts.push_back(builder.makeLocalSet(
          w.index,
          builder.makeGlobalGet(site->key->scratchGlobals[j - 1], w.type)));
      }
      return builder.makeBlock(stmts, Type::none);
    }
    return builder.makeCall(replacement.helper, args, type);
  }

  // We do not replace a site inside another replaced site, so we can use a
  // simple postorder walk here (the holes we pass to the helper are the
  // original leaves, which no replacement of a child could touch).
  void visitExpression(Expression* curr) {
    if (auto* block = curr->dynCast<Block>()) {
      if (auto sliceIter = sliceReplacements.find(block);
          sliceIter != sliceReplacements.end()) {
        auto sorted = sliceIter->second;
        std::sort(sorted.begin(),
                  sorted.end(),
                  [](const Replacement& a, const Replacement& b) {
                    return a.site->sliceOffset > b.site->sliceOffset;
                  });
        Builder builder(*getModule());
        for (auto& replacement : sorted) {
          auto* site = replacement.site;
          Index offset = site->sliceOffset;
          Index len = site->sliceLen;
          assert(len >= 2 && offset + len <= block->list.size());
          Type type = site->getType();
          if (type == Type::unreachable) {
            auto& fixed = site->key->fixed;
            auto& dupOf = site->key->dupOf;
            std::vector<Expression*> args;
            for (Index i = 0; i < site->holes.size(); i++) {
              if (!fixed[i] && dupOf[i] == Index(-1)) {
                auto& hole = site->holes[i];
                args.push_back(
                  hole.expr ? hole.expr
                            : builder.makeLocalGet(hole.localIndex, hole.type));
              }
            }
            block->list[offset] =
              builder.makeCall(replacement.helper, args, Type::none);
            block->list[offset + 1] = builder.makeUnreachable();
            block->list.erase(block->list.begin() + offset + 2,
                              block->list.begin() + offset + len);
          } else if (site->trailingSet) {
            auto* set = block->list[offset + len - 1]->cast<LocalSet>();
            set->value = makeReplacementExpr(replacement, type);
            block->list[offset] = set;
            block->list.erase(block->list.begin() + offset + 1,
                              block->list.begin() + offset + len);
          } else {
            block->list[offset] = makeReplacementExpr(replacement, type);
            block->list.erase(block->list.begin() + offset + 1,
                              block->list.begin() + offset + len);
          }
        }
      }
    }
    auto iter = replacements.find(curr);
    if (iter == replacements.end()) {
      return;
    }
    replaceCurrent(makeReplacementExpr(iter->second, curr->type));
  }

private:
  const Replacements& replacements;
  const SliceReplacements& sliceReplacements;
};

struct OutlineExpressions : public Pass {
  // The helpers perform the code they replace; any multi-write scratch globals
  // we introduce are added to existing function effect summaries in run().
  bool addsEffects() override { return false; }

  bool requiresNonNullableLocalFixups() override { return false; }

  Module* wasm;

  // A deque so that pointers to keys remain valid, in creation order so that
  // the helpers are created in a deterministic order.
  std::deque<KeyInfo> keys;
  // Sites with the same hash usually have the same shape, but we verify that,
  // so a hash may map to several keys.
  std::unordered_map<size_t, std::vector<KeyInfo*>> keysByHash;
  // Module-level mutable scratch globals created/reused for multi-write helpers
  // keyed by (Type, per-type rank).
  std::unordered_map<std::pair<Type, Index>, Name> scratchGlobals;

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
  // Whether reference-typed local holes are canonicalized to demanded types.
  bool canonicalizeLocalHoles = false;
  // Whether module functions have precomputed GlobalEffects summaries.
  bool hasModuleEffects = false;
  ModuleUtils::ParallelFunctionAnalysis<FunctionInfo>* currentAnalysis =
    nullptr;

  void run(Module* module) override {
    wasm = module;
    scratchGlobals.clear();
    hasModuleEffects = false;
    for (auto& f : wasm->functions) {
      if (f->effects) {
        hasModuleEffects = true;
        break;
      }
    }
    canonicalizeLocalHoles =
      getPassOptions().shrinkLevel > 0 && getPassOptions().trapsNeverHappen;

    auto mode = getMode();
    bool enableUnCseTees =
      getPassOptions().shrinkLevel > 0 || getPassOptions().trapsNeverHappen;

    unCseCallRefTargets();
    if (enableUnCseTees) {
      unCseOverCsedTees();
    }

    constHoles = false;
    for (Index round = 0; round < MaxRounds; round++) {
      if (!runRound(mode)) {
        break;
      }
      if (enableUnCseTees) {
        unCseOverCsedTees();
      }
    }
    foldCallRefDispatch(true);
    if (enableUnCseTees) {
      unCseOverCsedTees();
    }
    constHoles = true;
    for (Index round = 0; round < MaxConstRounds; round++) {
      if (!runRound(mode)) {
        break;
      }
      if (enableUnCseTees) {
        unCseOverCsedTees();
      }
    }
    foldCallRefDispatch(true);
    for (Index round = 0; round < 3; round++) {
      bool changed = sinkWrapperCalls();
      if (getPassOptions().shrinkLevel > 0) {
        changed |= sinkConstantArgs();
      }
      if (!changed) {
        break;
      }
    }
    if (enableUnCseTees) {
      unCseOverCsedTees();
    }

    if (!scratchGlobals.empty()) {
      for (auto& func : wasm->functions) {
        if (func->effects) {
          auto* mutableEffects =
            const_cast<EffectAnalyzer*>(func->effects.get());
          for (auto& [_, gName] : scratchGlobals) {
            mutableEffects->mutableGlobalsRead.insert(gName);
            mutableEffects->globalsWritten.insert(gName);
          }
        }
      }
      for (auto& [_, effects] : wasm->indirectCallEffects) {
        if (effects) {
          auto* mutableEffects = const_cast<EffectAnalyzer*>(effects.get());
          for (auto& [_, gName] : scratchGlobals) {
            mutableEffects->mutableGlobalsRead.insert(gName);
            mutableEffects->globalsWritten.insert(gName);
          }
        }
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
        Scanner(*wasm, func, info, mode, constHoles, canonicalizeLocalHoles)
          .walk(func->body);
        noteParents(func->body, info);
      });
    currentAnalysis = &analysis;

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
      if (!info.sites.empty() && info.sites.back().expr == func->body &&
          info.sites.back().sliceLen == 0) {
        noteExistingHelper(func.get(), info.sites.back());
      }
    }

    if (constHoles) {
      splitKeysByConstants();
    }

    // Decide what to outline. Nested sites complicate this, as a site inside
    // an outlined site vanishes along with it, and so must not be counted for
    // its own key, which may change the decision for that key, and so forth.
    // Iterate to a fixed point (or just stop after a while, as any decision is
    // safe, just not optimal).
    for (Index i = 0; i < 10; i++) {
      updateCandidates(analysis);
      bool changed = false;
      for (auto& info : keys) {
        if (info.disabled) {
          continue;
        }
        bool outline = false;
        if (info.helper) {
          // The helper already exists, so any use of it is a win.
          for (auto* site : info.sites) {
            if (site->candidate) {
              outline = true;
              break;
            }
          }
        } else if (info.sites.size() >= 2) {
          // Each site shrinks to a call with the holes as arguments, and the
          // helper contains a copy of the tree plus overhead.
          prepareKey(info);
          for (auto* site : info.sites) {
            site->selectedForKey = false;
          }
          int64_t saved = 0;
          for (auto it = info.sites.rbegin(); it != info.sites.rend(); ++it) {
            auto* site = *it;
            if (!site->candidate || isShadowedWithinKey(site)) {
              continue;
            }
            site->selectedForKey = true;
            saved += int64_t(site->bytes) - getCallBytes(site) -
                     LocalGetBytes * int64_t(info.numLocals) -
                     int64_t(site->varyingBytes);
          }
          auto* site = info.sites[0];
          int64_t cost =
            int64_t(site->bytes) - int64_t(site->helperVaryingBytes) +
            LocalGetBytes * int64_t(info.numVarying + info.numDupVarying) +
            HelperOverheadBytes +
            LocalSetBytes * int64_t(site->internalLocals.size());
          if (!site->written.empty()) {
            // The helper returns written[0]'s final value, writes written[1..]
            // into scratch globals, and declares any written locals that are
            // not parameters.
            cost += LocalSetBytes + 4 * int64_t(site->written.size() - 1);
            for (auto& w : site->written) {
              if (!isWrittenParam(site, w.index)) {
                cost += LocalSetBytes;
              }
            }
          }
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
    updateCandidates(analysis);

    // Create the helpers and gather the replacements.
    Replacements replacements;
    SliceReplacements sliceReplacements;
    for (auto& info : keys) {
      if (!info.outline) {
        continue;
      }
      for (auto* site : info.sites) {
        if (!site->replaced) {
          continue;
        }
        if (!info.helper) {
          info.helper = createHelper(site);
        }
        if (site->sliceLen == 0) {
          replacements[site->expr] = Replacement{info.helper->name, site};
        } else {
          sliceReplacements[site->expr->cast<Block>()].push_back(
            Replacement{info.helper->name, site});
        }
      }
    }
    if (replacements.empty() && sliceReplacements.empty()) {
      return false;
    }

    Replacer(replacements, sliceReplacements).run(getPassRunner(), wasm);
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
    WASM_UNREACHABLE("unknown mode");
  }

  // Link each site to the nearest single-expression site and slice sites that
  // contain or dominate it.
  void noteParents(Expression* body, FunctionInfo& info) {
    if (info.sites.empty()) {
      return;
    }
    std::unordered_map<Expression*, Site*> siteMap;
    std::unordered_map<Block*, SmallVector<Site*, 8>> sliceSitesByBlock;
    for (auto& site : info.sites) {
      if (site.sliceLen == 0) {
        siteMap[site.expr] = &site;
      } else {
        sliceSitesByBlock[site.expr->cast<Block>()].push_back(&site);
      }
    }
    struct StackEntry {
      Expression* curr;
      Site* parent;
      SmallVector<Site*, 4> enclosingSlices;
    };
    std::vector<StackEntry> stack;
    stack.push_back({body, nullptr, {}});
    while (!stack.empty()) {
      auto [curr, parent, enclosingSlices] = std::move(stack.back());
      stack.pop_back();
      if (auto iter = siteMap.find(curr); iter != siteMap.end()) {
        auto* site = iter->second;
        site->parent = parent;
        site->enclosingSlices = std::move(enclosingSlices);
        enclosingSlices.clear();
        parent = site;
      }
      if (auto* block = curr->dynCast<Block>()) {
        auto sliceIter = sliceSitesByBlock.find(block);
        if (sliceIter != sliceSitesByBlock.end()) {
          auto& slices = sliceIter->second;
          for (auto* slice : slices) {
            slice->parent = parent;
            slice->enclosingSlices = enclosingSlices;
            for (auto* other : slices) {
              if (other == slice) {
                continue;
              }
              bool overlaps =
                other->sliceOffset < slice->sliceOffset + slice->sliceLen &&
                slice->sliceOffset < other->sliceOffset + other->sliceLen;
              bool higherPriority =
                other->sliceLen > slice->sliceLen ||
                (other->sliceLen == slice->sliceLen &&
                  other->sliceOffset < slice->sliceOffset);
              if (overlaps && higherPriority) {
                slice->overlappingSlices.push_back(other);
              }
            }
          }
        }
        for (Index j = 0; j < block->list.size(); j++) {
          auto childSlices = enclosingSlices;
          if (sliceIter != sliceSitesByBlock.end()) {
            for (auto* slice : sliceIter->second) {
              if (slice->sliceOffset <= j &&
                  j < slice->sliceOffset + slice->sliceLen) {
                childSlices.push_back(slice);
              }
            }
          }
          stack.push_back({block->list[j], parent, std::move(childSlices)});
        }
      } else {
        for (auto* child : ChildIterator(curr)) {
          stack.push_back({child, parent, enclosingSlices});
        }
      }
    }
  }

  // Two trees have the same shape if they are equal apart from the locals that
  // their holes read and write, where the same local must be used in the same
  // places in both trees (and, with constants as holes, apart from the
  // constants, which must have the same type).
  bool sameShape(Site* leftSite, Site* rightSite) {
    if (leftSite->written.size() != rightSite->written.size() ||
        leftSite->internalLocals.size() != rightSite->internalLocals.size() ||
        leftSite->passthroughLocals.size() !=
          rightSite->passthroughLocals.size()) {
      return false;
    }
    std::unordered_map<Index, Index> leftToRight, rightToLeft;
    bool mismatch = false;
    auto mapLocals = [&](Index leftIndex, Index rightIndex) {
      auto [l, lInserted] = leftToRight.try_emplace(leftIndex, rightIndex);
      auto [r, rInserted] = rightToLeft.try_emplace(rightIndex, leftIndex);
      if (l->second != rightIndex || r->second != leftIndex) {
        mismatch = true;
        return;
      }
      if (lInserted) {
        if (leftSite->isInternalLocal(leftIndex) !=
              rightSite->isInternalLocal(rightIndex) ||
            leftSite->isPassthroughLocal(leftIndex) !=
              rightSite->isPassthroughLocal(rightIndex) ||
            leftSite->isOverwriteLocal(leftIndex) !=
              rightSite->isOverwriteLocal(rightIndex) ||
            leftSite->isWrittenLocal(leftIndex) !=
              rightSite->isWrittenLocal(rightIndex)) {
          mismatch = true;
        }
        if (leftSite->isInternalLocal(leftIndex) ||
            leftSite->isOverwriteLocal(leftIndex) ||
            leftSite->isPassthroughLocal(leftIndex) ||
            leftSite->isWrittenLocal(leftIndex)) {
          if (leftSite->func->getLocalType(leftIndex) !=
              rightSite->func->getLocalType(rightIndex)) {
            mismatch = true;
          }
        }
      }
    };
    const auto* leftDemanded =
      canonicalizeLocalHoles
        ? &currentAnalysis->map.at(leftSite->func).demandedTypes
        : nullptr;
    const auto* rightDemanded =
      canonicalizeLocalHoles
        ? &currentAnalysis->map.at(rightSite->func).demandedTypes
        : nullptr;
    ExpressionAnalyzer::ExprComparer comparer = [&](Expression* left,
                                                    Expression* right) {
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
      if (auto* leftGet = left->dynCast<LocalGet>()) {
        auto* rightGet = right->dynCast<LocalGet>();
        if (!rightGet) {
          return false;
        }
        Type leftType =
          leftDemanded ? leftDemanded->at(leftGet) : leftGet->type;
        Type rightType =
          rightDemanded ? rightDemanded->at(rightGet) : rightGet->type;
        if (leftType != rightType) {
          mismatch = true;
          return true;
        }
        mapLocals(leftGet->index, rightGet->index);
        return true;
      }
      if (auto* leftSet = left->dynCast<LocalSet>()) {
        auto* rightSet = right->dynCast<LocalSet>();
        if (!rightSet) {
          return false;
        }
        // The local's type matters as the helper returns or declares it.
        if (leftSet->type != rightSet->type ||
            leftSite->func->getLocalType(leftSet->index) !=
              rightSite->func->getLocalType(rightSet->index)) {
          mismatch = true;
          return true;
        }
        mapLocals(leftSet->index, rightSet->index);
        // Returning true skips the children, so compare them here. That is
        // fine as a value we consider does not branch out of itself, so all
        // the labels it uses are defined inside it.
        if (!ExpressionAnalyzer::flexibleEqual(
              leftSet->value, rightSet->value, comparer)) {
          mismatch = true;
        }
        return true;
      }
      return false;
    };
    auto getSliceLen = [](Site* site, Index& len) -> bool {
      if (site->sliceLen > 0) {
        len = site->sliceLen;
        return true;
      }
      if (auto* block = site->expr->dynCast<Block>()) {
        if (!block->name.is()) {
          len = block->list.size();
          return true;
        }
      }
      return false;
    };
    auto getSliceElem = [](Site* site, Index idx) -> Expression* {
      if (site->sliceLen > 0) {
        return site->getSliceExpr(idx);
      }
      return site->expr->cast<Block>()->list[idx];
    };
    Index leftLen = 0, rightLen = 0;
    bool leftIsSlice = getSliceLen(leftSite, leftLen);
    bool rightIsSlice = getSliceLen(rightSite, rightLen);
    if (leftIsSlice || rightIsSlice) {
      if (!leftIsSlice || !rightIsSlice || leftLen != rightLen ||
          leftSite->getType() != rightSite->getType()) {
        return false;
      }
      for (Index i = 0; i < leftLen; i++) {
        if (!ExpressionAnalyzer::flexibleEqual(
              getSliceElem(leftSite, i), getSliceElem(rightSite, i), comparer) ||
            mismatch) {
          return false;
        }
      }
      return true;
    }
    return ExpressionAnalyzer::flexibleEqual(
             leftSite->expr, rightSite->expr, comparer) &&
           !mismatch;
  }

  KeyInfo* getKey(Site* site) {
    auto& candidates = keysByHash[site->hash];
    for (auto* key : candidates) {
      if (sameShape(key->sites[0], site)) {
        return key;
      }
    }
    keys.emplace_back();
    auto* key = &keys.back();
    candidates.push_back(key);
    return key;
  }

  void ensureHoles(Site* site) {
    if (!site->holesComputed) {
      const auto* demanded =
        canonicalizeLocalHoles
          ? &currentAnalysis->map.at(site->func).demandedTypes
          : nullptr;
      site->holes = getHoles(*wasm, site, constHoles, demanded);
      site->holesComputed = true;
    }
  }

  // Computes the holes of a key's sites, which constant holes are fixed (all
  // sites have the same constant there) or duplicates of earlier varying
  // constant holes, and the resulting parameter counts and per-site argument
  // sizes.
  void prepareKey(KeyInfo& key) {
    if (key.prepared) {
      return;
    }
    key.prepared = true;
    for (auto* site : key.sites) {
      ensureHoles(site);
      site->varyingBytes = 0;
      site->helperVaryingBytes = 0;
    }
    auto& first = key.sites[0]->holes;
    Index numHoles = first.size();
    key.fixed.assign(numHoles, false);
    key.dupOf.assign(numHoles, Index(-1));
    key.numLocals = 0;
    key.numVarying = 0;
    key.numDupVarying = 0;
    for (Index i = 0; i < numHoles; i++) {
      key.fixed[i] = !first[i].isLocal;
    }
    for (auto* site : key.sites) {
      assert(site->holes.size() == numHoles);
      for (Index i = 0; i < numHoles; i++) {
        if (key.fixed[i] &&
            !ExpressionAnalyzer::equal(first[i].expr, site->holes[i].expr)) {
          key.fixed[i] = false;
        }
      }
    }
    for (Index i = 0; i < numHoles; i++) {
      if (first[i].isLocal) {
        key.numLocals++;
      } else if (!key.fixed[i]) {
        for (Index j = 0; j < i; j++) {
          if (!first[j].isLocal && !key.fixed[j] && key.dupOf[j] == Index(-1)) {
            bool allEq = true;
            for (auto* site : key.sites) {
              if (!ExpressionAnalyzer::equal(site->holes[i].expr,
                                             site->holes[j].expr)) {
                allEq = false;
                break;
              }
            }
            if (allEq) {
              key.dupOf[i] = j;
              break;
            }
          }
        }
        if (key.dupOf[i] == Index(-1)) {
          key.numVarying++;
        } else {
          key.numDupVarying++;
        }
      }
    }
    for (auto* site : key.sites) {
      for (Index i = 0; i < numHoles; i++) {
        auto& hole = site->holes[i];
        if (!hole.isLocal && !key.fixed[i]) {
          Index b = getShallowBytes(hole.expr);
          site->helperVaryingBytes += b;
          if (key.dupOf[i] == Index(-1)) {
            site->varyingBytes += b;
          }
        }
      }
    }
  }

  int64_t evalSubset(const std::vector<Site*>& subset) {
    if (subset.size() < 2) {
      return 0;
    }
    for (auto* site : subset) {
      site->selectedForKey = false;
    }
    std::vector<Site*> active;
    active.reserve(subset.size());
    for (auto it = subset.rbegin(); it != subset.rend(); ++it) {
      auto* site = *it;
      if (site->helperBody || isShadowedWithinKey(site)) {
        continue;
      }
      site->selectedForKey = true;
      active.push_back(site);
    }
    for (auto* site : subset) {
      site->selectedForKey = false;
    }
    if (active.size() < 2) {
      return 0;
    }
    auto& first = active[0]->holes;
    Index numHoles = first.size();
    std::vector<bool> fixed(numHoles, false);
    std::vector<Index> dupOf(numHoles, Index(-1));
    for (Index i = 0; i < numHoles; i++) {
      if (!first[i].isLocal) {
        bool allEq = true;
        for (Index s = 1; s < active.size(); s++) {
          if (!ExpressionAnalyzer::equal(first[i].expr,
                                         active[s]->holes[i].expr)) {
            allEq = false;
            break;
          }
        }
        fixed[i] = allEq;
      }
    }
    Index numLocals = 0;
    Index numVarying = 0;
    Index numDupVarying = 0;
    for (Index i = 0; i < numHoles; i++) {
      if (first[i].isLocal) {
        numLocals++;
      } else if (!fixed[i]) {
        for (Index j = 0; j < i; j++) {
          if (!first[j].isLocal && !fixed[j] && dupOf[j] == Index(-1)) {
            bool allEq = true;
            for (auto* site : active) {
              if (!ExpressionAnalyzer::equal(site->holes[i].expr,
                                             site->holes[j].expr)) {
                allEq = false;
                break;
              }
            }
            if (allEq) {
              dupOf[i] = j;
              break;
            }
          }
        }
        if (dupOf[i] == Index(-1)) {
          numVarying++;
        } else {
          numDupVarying++;
        }
      }
    }
    int64_t saved = 0;
    for (auto* site : active) {
      Index varBytes = 0;
      for (Index i = 0; i < numHoles; i++) {
        if (!first[i].isLocal && !fixed[i] && dupOf[i] == Index(-1)) {
          varBytes += getShallowBytes(site->holes[i].expr);
        }
      }
      saved += int64_t(site->bytes) - getCallBytes(site) -
               LocalGetBytes * int64_t(numLocals) - int64_t(varBytes);
    }
    auto* site0 = active[0];
    Index helperVarBytes = 0;
    for (Index i = 0; i < numHoles; i++) {
      if (!first[i].isLocal && !fixed[i]) {
        helperVarBytes += getShallowBytes(site0->holes[i].expr);
      }
    }
    int64_t cost = int64_t(site0->bytes) - int64_t(helperVarBytes) +
                   LocalGetBytes * int64_t(numVarying + numDupVarying) +
                   HelperOverheadBytes +
                   LocalSetBytes * int64_t(site0->internalLocals.size());
    if (!site0->written.empty()) {
      cost += LocalSetBytes + 4 * int64_t(site0->written.size() - 1);
      for (auto& w : site0->written) {
        if (!isWrittenParam(site0, w.index)) {
          cost += LocalSetBytes;
        }
      }
    }
    return saved > cost ? saved - cost : 0;
  }

  std::vector<std::vector<Site*>>
  partitionByHole(const std::vector<Site*>& subset, Index holeIdx) {
    std::vector<std::vector<Site*>> parts;
    std::unordered_map<size_t, SmallVector<Index, 2>> buckets;
    for (auto* site : subset) {
      Expression* c = site->holes[holeIdx].expr;
      size_t h = ExpressionAnalyzer::hash(c);
      auto& bucket = buckets[h];
      Index found = Index(-1);
      for (Index pIdx : bucket) {
        if (ExpressionAnalyzer::equal(parts[pIdx][0]->holes[holeIdx].expr, c)) {
          found = pIdx;
          break;
        }
      }
      if (found == Index(-1)) {
        found = parts.size();
        parts.push_back({site});
        bucket.push_back(found);
      } else {
        parts[found].push_back(site);
      }
    }
    return parts;
  }

  int64_t bestRecursiveSplit(const std::vector<Site*>& subset,
                             std::vector<std::vector<Site*>>& outGroups,
                             Index depth = 0) {
    outGroups.clear();
    if (subset.size() < 2) {
      return 0;
    }
    int64_t baseScore = evalSubset(subset);
    if (depth >= 4) {
      if (baseScore > 0) {
        outGroups.push_back(subset);
      }
      return baseScore;
    }
    Index numHoles = subset[0]->holes.size();
    int64_t bestScore = baseScore;
    Index bestHole = Index(-1);
    for (Index i = 0; i < numHoles; i++) {
      if (subset[0]->holes[i].isLocal) {
        continue;
      }
      auto parts = partitionByHole(subset, i);
      if (parts.size() <= 1) {
        continue;
      }
      int64_t subScore = 0;
      std::unordered_set<Site*> kept;
      for (auto& part : parts) {
        if (part.size() >= 2) {
          int64_t sc = evalSubset(part);
          if (sc > 0) {
            subScore += sc;
            for (auto* s : part) {
              kept.insert(s);
            }
          }
        }
      }
      if (kept.size() < subset.size() && subset.size() - kept.size() >= 2 &&
          !kept.empty()) {
        std::vector<Site*> leftovers;
        leftovers.reserve(subset.size() - kept.size());
        for (auto* s : subset) {
          if (!kept.contains(s)) {
            leftovers.push_back(s);
          }
        }
        subScore += evalSubset(leftovers);
      }
      if (subScore > bestScore) {
        bestScore = subScore;
        bestHole = i;
      }
    }
    if (bestHole == Index(-1)) {
      if (baseScore > 0) {
        outGroups.push_back(subset);
      }
      return baseScore;
    }
    auto parts = partitionByHole(subset, bestHole);
    int64_t totalScore = 0;
    std::vector<std::vector<Site*>> splitGroups;
    std::unordered_set<Site*> kept;
    for (auto& part : parts) {
      if (part.size() >= 2) {
        std::vector<std::vector<Site*>> subOut;
        int64_t sc = bestRecursiveSplit(part, subOut, depth + 1);
        if (sc > 0) {
          totalScore += sc;
          for (auto& g : subOut) {
            for (auto* s : g) {
              kept.insert(s);
            }
            splitGroups.push_back(std::move(g));
          }
        }
      }
    }
    if (kept.size() < subset.size() && subset.size() - kept.size() >= 2 &&
        !kept.empty()) {
      std::vector<Site*> leftovers;
      leftovers.reserve(subset.size() - kept.size());
      for (auto* s : subset) {
        if (!kept.contains(s)) {
          leftovers.push_back(s);
        }
      }
      std::vector<std::vector<Site*>> leftOut;
      int64_t sc = bestRecursiveSplit(leftovers, leftOut, depth + 1);
      if (sc > 0) {
        totalScore += sc;
        for (auto& g : leftOut) {
          splitGroups.push_back(std::move(g));
        }
      }
    }
    if (totalScore > baseScore) {
      outGroups = std::move(splitGroups);
      return totalScore;
    }
    if (baseScore > 0) {
      outGroups.push_back(subset);
    }
    return baseScore;
  }

  void splitKeysByConstants() {
    size_t origNumKeys = keys.size();
    for (size_t kIdx = 0; kIdx < origNumKeys; kIdx++) {
      auto& key = keys[kIdx];
      if (key.helper || key.sites.size() < 2) {
        continue;
      }
      for (auto* site : key.sites) {
        ensureHoles(site);
      }
      bool hasConstHole = false;
      for (auto& h : key.sites[0]->holes) {
        if (!h.isLocal) {
          hasConstHole = true;
          break;
        }
      }
      if (!hasConstHole) {
        continue;
      }
      std::vector<std::vector<Site*>> groups;
      int64_t score = bestRecursiveSplit(key.sites, groups);
      if (score <= 0 ||
          (groups.size() == 1 && groups[0].size() == key.sites.size())) {
        continue;
      }
      std::unordered_set<Site*> chosen;
      for (auto& g : groups) {
        for (auto* s : g) {
          chosen.insert(s);
        }
      }
      if (chosen.size() < key.sites.size()) {
        std::vector<Site*> unchosen;
        unchosen.reserve(key.sites.size() - chosen.size());
        for (auto* s : key.sites) {
          if (!chosen.contains(s)) {
            unchosen.push_back(s);
          }
        }
        keys.emplace_back();
        auto& deadKey = keys.back();
        deadKey.disabled = true;
        deadKey.sites = std::move(unchosen);
        for (auto* s : deadKey.sites) {
          s->key = &deadKey;
        }
      }
      key.sites = std::move(groups[0]);
      key.prepared = false;
      for (auto* s : key.sites) {
        s->key = &key;
      }
      for (size_t g = 1; g < groups.size(); g++) {
        keys.emplace_back();
        auto& subKey = keys.back();
        subKey.sites = std::move(groups[g]);
        for (auto* s : subKey.sites) {
          s->key = &subKey;
        }
      }
    }
  }

  // The bytes a site costs after outlining: a call, plus an unreachable if the
  // tree is unreachable, or local.sets (+ global.gets) if it writes locals.
  Index getCallBytes(Site* site) {
    if (site->getType() == Type::unreachable) {
      return CallBytes + 1;
    }
    if (!site->written.empty()) {
      return CallBytes + LocalSetBytes + 4 * (site->written.size() - 1);
    }
    return CallBytes;
  }

  // Whether a local the site writes is also a parameter of the helper (either
  // read in the tree or passed through to preserve its value on untaken paths).
  bool isWrittenParam(Site* site, Index writtenIndex) {
    for (auto& hole : site->holes) {
      if (hole.isLocal && hole.localIndex == writtenIndex) {
        return true;
      }
    }
    return false;
  }

  void updateCandidates(
    ModuleUtils::ParallelFunctionAnalysis<FunctionInfo>& analysis) {
    for (auto& func : wasm->functions) {
      auto& fnInfo = analysis.map[func.get()];
      for (auto it = fnInfo.sites.rbegin(); it != fnInfo.sites.rend(); ++it) {
        Site& site = *it;
        bool killed = false;
        if (site.parent &&
            (site.parent->replaced || site.parent->killedByAncestor)) {
          killed = true;
        }
        if (!killed) {
          for (auto* enc : site.enclosingSlices) {
            if (enc->replaced) {
              killed = true;
              break;
            }
          }
        }
        site.killedByAncestor = killed;
        bool cand = !site.helperBody && !killed;
        if (cand) {
          for (auto* ov : site.overlappingSlices) {
            if (ov->replaced) {
              cand = false;
              break;
            }
          }
        }
        site.candidate = cand;
        site.replaced = cand && site.key->outline;
      }
    }
  }

  bool isShadowedWithinKey(Site* site) {
    for (auto* ov : site->overlappingSlices) {
      if (ov->key == site->key && ov->selectedForKey) {
        return true;
      }
    }
    for (auto* curr = site; curr; curr = curr->parent) {
      if (curr != site && curr->key == site->key && curr->selectedForKey) {
        return true;
      }
      for (auto* enc : curr->enclosingSlices) {
        if (enc->key == site->key && enc->selectedForKey) {
          return true;
        }
      }
    }
    return false;
  }

  Type getHelperResults(Site* site) {
    if (!site->written.empty()) {
      return site->written[0].type;
    }
    Type type = site->getType();
    return type == Type::unreachable ? Type::none : type;
  }

  // If the function's body is just a tree whose holes are its params, in
  // order, note it as the helper for the tree's key. With constants as holes,
  // all the constants in the body must be fixed, as the function has them
  // baked in. (The helpers of trees that write a local are blocks that end in
  // a read of the local, which we do not try to recognize.)
  void noteExistingHelper(Function* func, Site& site) {
    if (!site.written.empty() ||
        (!site.internalLocals.empty() &&
         func->vars.size() != site.internalLocals.size()) ||
        func->getResults() != getHelperResults(&site)) {
      return;
    }
    auto& key = *site.key;
    prepareKey(key);
    if (key.numVarying != 0 || key.numLocals != func->getNumParams()) {
      return;
    }
    Index i = 0;
    for (Index hIdx = 0; hIdx < site.holes.size(); hIdx++) {
      auto& hole = site.holes[hIdx];
      if (!hole.isLocal) {
        continue;
      }
      if (hole.localIndex != i) {
        return;
      }
      Type paramType = func->getLocalType(i);
      if (hole.type != paramType) {
        if (!canonicalizeLocalHoles) {
          return;
        }
        bool allSub = true;
        for (auto* otherSite : key.sites) {
          auto& otherHole = otherSite->holes[hIdx];
          Type actualType =
            otherSite->func->getLocalType(otherHole.localIndex);
          if (!Type::isSubType(actualType, paramType)) {
            allSub = false;
            break;
          }
        }
        if (!allSub) {
          return;
        }
      }
      i++;
    }
    for (Index j = 0; j < site.internalLocals.size(); j++) {
      Index expectedIdx = func->getNumParams() + j;
      if (site.internalLocals[j] != expectedIdx ||
          func->getLocalType(expectedIdx) !=
            site.func->getLocalType(site.internalLocals[j])) {
        return;
      }
    }
    if (key.helper) {
      // We already have a helper for this key. This function can just call it
      // like any other site.
      return;
    }
    site.helperBody = true;
    key.helper = func;
  }

  // Turns the non-fixed holes of a copied tree into local.gets of the params,
  // and local.sets of the written/internal locals into local.sets of the
  // helper's locals for them.
  struct HoleReplacer
    : public PostWalker<HoleReplacer, UnifiedExpressionVisitor<HoleReplacer>> {
    Module& wasm;
    std::unordered_map<Index, Index> paramOfLocal;
    std::unordered_map<Index, Type> typeOfLocal;
    std::unordered_map<Expression*, Index> paramOfConst;

    HoleReplacer(Module& wasm) : wasm(wasm) {}

    void visitExpression(Expression* curr) {
      if (auto* get = curr->dynCast<LocalGet>()) {
        Index oldIdx = get->index;
        get->index = paramOfLocal.at(oldIdx);
        if (auto it = typeOfLocal.find(oldIdx); it != typeOfLocal.end()) {
          get->type = it->second;
        }
      } else if (auto* set = curr->dynCast<LocalSet>()) {
        set->index = paramOfLocal.at(set->index);
      } else if (auto iter = paramOfConst.find(curr);
                 iter != paramOfConst.end()) {
        replaceCurrent(Builder(wasm).makeLocalGet(iter->second, curr->type));
      }
    }
  };

  Name getScratchGlobal(Type type, Index rank) {
    auto key = std::make_pair(type, rank);
    if (auto iter = scratchGlobals.find(key); iter != scratchGlobals.end()) {
      return iter->second;
    }
    std::string base =
      "oe$out_" + type.toString() + "_" + std::to_string(rank + 1);
    for (char& c : base) {
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' &&
          c != '$') {
        c = '_';
      }
    }
    Name candidate(base);
    if (auto* existing = wasm->getGlobalOrNull(candidate)) {
      if (existing->mutable_ && existing->type == type &&
          !existing->imported()) {
        scratchGlobals[key] = candidate;
        return candidate;
      }
    }
    Name name = Names::getValidGlobalName(*wasm, base);
    Builder builder(*wasm);
    wasm->addGlobal(builder.makeGlobal(
      name,
      type,
      builder.makeConstantExpression(Literal::makeZero(type)),
      Builder::Mutable));
    scratchGlobals[key] = name;
    return name;
  }

  Function* createHelper(Site* site) {
    auto& key = *site->key;
    Builder builder(*wasm);
    Expression* body;
    if (site->sliceLen == 0) {
      body = ExpressionManipulator::copy(site->expr, *wasm);
    } else {
      std::vector<Expression*> copied;
      copied.reserve(site->sliceLen);
      for (Index i = 0; i < site->sliceLen; i++) {
        copied.push_back(
          ExpressionManipulator::copy(site->getSliceExpr(i), *wasm));
      }
      body = builder.makeBlock(copied, site->getType());
    }
    auto holes = getHoles(*wasm, body, site, constHoles);
    assert(holes.size() == site->holes.size());
    std::vector<Type> params;
    HoleReplacer replacer(*wasm);
    std::vector<Index> holeParamIdx(holes.size(), Index(-1));
    for (Index i = 0; i < holes.size(); i++) {
      if (key.fixed[i]) {
        continue;
      }
      if (key.dupOf[i] != Index(-1)) {
        Index primary = key.dupOf[i];
        assert(holeParamIdx[primary] != Index(-1));
        replacer.paramOfConst[holes[i].expr] = holeParamIdx[primary];
        continue;
      }
      auto& hole = holes[i];
      Type paramType = site->holes[i].type;
      Index pIdx = params.size();
      holeParamIdx[i] = pIdx;
      if (hole.isLocal) {
        replacer.paramOfLocal[hole.localIndex] = pIdx;
        replacer.typeOfLocal[hole.localIndex] = paramType;
      } else {
        replacer.paramOfConst[hole.expr] = pIdx;
      }
      params.push_back(paramType);
    }
    assert(params.size() == key.numLocals + key.numVarying);
    std::vector<Type> vars;
    for (Index internalIdx : site->internalLocals) {
      Index helperIdx = params.size() + vars.size();
      vars.push_back(site->func->getLocalType(internalIdx));
      replacer.paramOfLocal[internalIdx] = helperIdx;
    }
    if (!site->written.empty()) {
      SmallVector<Index, 4> writtenHelperIndices;
      for (auto& w : site->written) {
        auto iter = replacer.paramOfLocal.find(w.index);
        Index index;
        if (iter != replacer.paramOfLocal.end()) {
          index = iter->second;
        } else {
          index = params.size() + vars.size();
          vars.push_back(w.type);
          replacer.paramOfLocal[w.index] = index;
        }
        writtenHelperIndices.push_back(index);
      }
      if (site->written.size() >= 2) {
        std::unordered_map<Type, Index> typeCounts;
        for (Index j = 1; j < site->written.size(); j++) {
          Type t = site->written[j].type;
          Index rank = typeCounts[t]++;
          key.scratchGlobals.push_back(getScratchGlobal(t, rank));
        }
      }
      replacer.walk(body);
      auto& firstWritten = site->written[0];
      if (site->sliceLen > 0) {
        auto* block = body->cast<Block>();
        for (Index j = 1; j < site->written.size(); j++) {
          block->list.push_back(builder.makeGlobalSet(
            key.scratchGlobals[j - 1],
            builder.makeLocalGet(writtenHelperIndices[j],
                                 site->written[j].type)));
        }
        block->list.push_back(
          builder.makeLocalGet(writtenHelperIndices[0], firstWritten.type));
        block->type = firstWritten.type;
      } else {
        std::vector<Expression*> list;
        list.reserve(site->written.size() + 1);
        list.push_back(body);
        for (Index j = 1; j < site->written.size(); j++) {
          list.push_back(builder.makeGlobalSet(
            key.scratchGlobals[j - 1],
            builder.makeLocalGet(writtenHelperIndices[j],
                                 site->written[j].type)));
        }
        list.push_back(
          builder.makeLocalGet(writtenHelperIndices[0], firstWritten.type));
        body = builder.makeBlock(list, firstWritten.type);
      }
    } else {
      replacer.walk(body);
    }
    auto name = Names::getValidFunctionName(*wasm, "outlined");
    Function* newFunc = wasm->addFunction(
      Builder::makeFunction(name,
                            Signature(Type(params), getHelperResults(site)),
                            std::move(vars),
                            body));
    initHelperEffects(newFunc);
    return newFunc;
  }

  struct CallRefReplacer : public PostWalker<CallRefReplacer> {
    Module& wasm;
    struct RepInfo {
      Name helper;
      bool unwrapTee = false;
    };
    std::unordered_map<Expression*, RepInfo> replacements;
    std::unordered_set<LocalSet*> innerTeesToUnwrap;
    CallRefReplacer(Module& wasm) : wasm(wasm) {}

    void visitLocalSet(LocalSet* curr) {
      if (innerTeesToUnwrap.count(curr)) {
        assert(curr->isTee());
        replaceCurrent(curr->value);
      }
    }

    void visitCallRef(CallRef* curr) {
      auto it = replacements.find(curr);
      if (it == replacements.end()) {
        return;
      }
      const auto& info = it->second;
      Builder builder(wasm);
      Index numArgs = curr->operands.size();
      std::vector<Expression*> args(numArgs);
      for (Index i = 0; i < numArgs; i++) {
        args[i] = curr->operands[i];
      }
      if (info.unwrapTee && args[0]->is<LocalSet>()) {
        auto* tee = args[0]->cast<LocalSet>();
        assert(tee->isTee());
        args[0] = tee->value;
      }
      replaceCurrent(
        builder.makeCall(info.helper, args, curr->type, curr->isReturn));
    }
  };

  bool isPureVtableBodyExpr(Expression* e, int depth = 0) {
    if (!e || e->type == Type::unreachable || depth > 4) {
      return false;
    }
    if (auto* lg = e->dynCast<LocalGet>()) {
      return lg->index == 0;
    }
    if (e->is<Const>()) {
      return true;
    }
    if (auto* sg = e->dynCast<StructGet>()) {
      return sg->ref && sg->ref->type.isStruct() &&
             isPureVtableBodyExpr(sg->ref, depth);
    }
    if (auto* rc = e->dynCast<RefCast>()) {
      return !rc->desc && rc->ref && isPureVtableBodyExpr(rc->ref, depth);
    }
    if (auto* ra = e->dynCast<RefAs>()) {
      return ra->op == RefAsNonNull && ra->value &&
             isPureVtableBodyExpr(ra->value, depth);
    }
    if (auto* ag = e->dynCast<ArrayGet>()) {
      return ag->ref && ag->ref->type.isArray() && ag->index &&
             isPureVtableBodyExpr(ag->ref, depth) &&
             isPureVtableBodyExpr(ag->index, depth);
    }
    if (auto* call = e->dynCast<Call>()) {
      return !call->isReturn && call->operands.size() == 1 &&
             isPureVtableHelper(call->target, depth + 1) &&
             isPureVtableBodyExpr(call->operands[0], depth + 1);
    }
    return false;
  }

  bool isPureVtableHelper(Name name, int depth = 0) {
    if (depth > 4) {
      return false;
    }
    Function* f = wasm->getFunctionOrNull(name);
    if (!f || f->imported() || f->getNumParams() != 1 ||
        !f->getResults().isSingle() || !f->getResults().isRef()) {
      return false;
    }
    return isPureVtableBodyExpr(f->body, depth);
  }

  bool isPureVtableLookupOnLocal(Expression* e,
                                 Index recvLocal,
                                 Index& numGets) {
    if (!e || e->type == Type::unreachable) {
      return false;
    }
    if (auto* lg = e->dynCast<LocalGet>()) {
      if (lg->index != recvLocal) {
        return false;
      }
      numGets++;
      return true;
    }
    if (auto* sg = e->dynCast<StructGet>()) {
      return sg->ref && sg->ref->type.isStruct() &&
             isPureVtableLookupOnLocal(sg->ref, recvLocal, numGets);
    }
    if (auto* rc = e->dynCast<RefCast>()) {
      return !rc->desc && rc->ref &&
             isPureVtableLookupOnLocal(rc->ref, recvLocal, numGets);
    }
    if (auto* ra = e->dynCast<RefAs>()) {
      return ra->op == RefAsNonNull && ra->value &&
             isPureVtableLookupOnLocal(ra->value, recvLocal, numGets);
    }
    if (auto* call = e->dynCast<Call>()) {
      return !call->isReturn && call->operands.size() == 1 &&
             isPureVtableHelper(call->target) &&
             isPureVtableLookupOnLocal(call->operands[0], recvLocal, numGets);
    }
    return false;
  }

  struct TreeLayout {
    std::unordered_map<Expression*, Expression*> parent;
    std::unordered_map<Expression*, Index> depth;
    std::unordered_map<Expression*, Index> preOrder;
    std::unordered_map<Expression*, Index> postOrder;

    void build(Expression* root) {
      parent.clear();
      depth.clear();
      preOrder.clear();
      postOrder.clear();
      if (!root) {
        return;
      }
      struct Frame {
        Expression* expr;
        Expression* par;
        Index d;
        bool visited;
      };
      std::vector<Frame> stack;
      stack.push_back({root, nullptr, 0, false});
      Index pre = 0;
      Index post = 0;
      while (!stack.empty()) {
        auto& top = stack.back();
        Expression* e = top.expr;
        if (!top.visited) {
          top.visited = true;
          parent[e] = top.par;
          depth[e] = top.d;
          preOrder[e] = pre++;
          ChildIterator it(e);
          std::vector<Expression*> children(it.begin(), it.end());
          for (auto rit = children.rbegin(); rit != children.rend(); ++rit) {
            if (*rit) {
              stack.push_back({*rit, e, top.d + 1, false});
            }
          }
        } else {
          postOrder[e] = post++;
          stack.pop_back();
        }
      }
    }

    bool isDescendant(Expression* node, Expression* anc) const {
      auto itN = preOrder.find(node);
      auto itA = preOrder.find(anc);
      if (itN == preOrder.end() || itA == preOrder.end()) {
        return false;
      }
      return itA->second <= itN->second &&
             postOrder.at(node) <= postOrder.at(anc);
    }

    Expression* getLCA(Expression* a, Expression* b) const {
      if (!a || !b) {
        return nullptr;
      }
      Index da = depth.at(a);
      Index db = depth.at(b);
      while (da > db) {
        a = parent.at(a);
        da--;
      }
      while (db > da) {
        b = parent.at(b);
        db--;
      }
      while (a != b) {
        a = parent.at(a);
        b = parent.at(b);
      }
      return a;
    }

    bool hasLoopBetween(Expression* node, Expression* anc) const {
      Expression* curr = node;
      while (curr && curr != anc) {
        if (curr->is<Loop>()) {
          return true;
        }
        curr = parent.at(curr);
      }
      return false;
    }

    bool isUnconditionalFrom(Expression* node, Expression* anc) const {
      Expression* curr = node;
      while (curr && curr != anc) {
        Expression* par = parent.at(curr);
        if (!par) {
          return false;
        }
        if (auto* iff = par->dynCast<If>()) {
          if (iff->condition != curr) {
            return false;
          }
        } else if (par->is<Loop>() || par->is<Try>() || par->is<TryTable>()) {
          return false;
        } else if (auto* blk = par->dynCast<Block>()) {
          for (auto* item : blk->list) {
            if (item == curr) {
              break;
            }
            if (!FindAll<Break>(item).list.empty() ||
                !FindAll<Switch>(item).list.empty() ||
                !FindAll<BrOn>(item).list.empty() ||
                !FindAll<Return>(item).list.empty()) {
              return false;
            }
          }
        }
        curr = par;
      }
      return curr == anc;
    }
  };

  void unCseCallRefTargets() {
    for (auto& f : wasm->functions) {
      if (f->imported()) {
        continue;
      }
      Function* func = f.get();
      if (FindAll<CallRef>(func->body).list.empty()) {
        continue;
      }

      bool funcChanged = false;
      TreeLayout layout;
      layout.build(func->body);

      // Step 1A: Propagate LCA-safe single-write copy tees
      // (local.tee $x (local.get $y)) on non-param locals $x.
      {
        Index numLocals = func->getNumLocals();
        std::vector<SmallVector<LocalSet*, 2>> writes(numLocals);
        std::vector<SmallVector<LocalGet*, 4>> reads(numLocals);
        for (auto* ls : FindAll<LocalSet>(func->body).list) {
          writes[ls->index].push_back(ls);
        }
        for (auto* lg : FindAll<LocalGet>(func->body).list) {
          reads[lg->index].push_back(lg);
        }

        std::unordered_map<Index, Index> copyTarget;
        std::unordered_set<LocalSet*> teesToUnwrap;
        for (Index x = func->getVarIndexBase(); x < numLocals; x++) {
          if (writes[x].size() != 1 || reads[x].empty()) {
            continue;
          }
          LocalSet* wx = writes[x][0];
          if (!wx->isTee()) {
            continue;
          }
          auto* innerLg = wx->value->dynCast<LocalGet>();
          if (!innerLg || innerLg->index == x) {
            continue;
          }
          Index y = innerLg->index;
          if (func->getLocalType(x) != func->getLocalType(y)) {
            continue;
          }
          Expression* lca = wx;
          Index wxPre = layout.preOrder.at(wx);
          bool orderOk = true;
          for (auto* gx : reads[x]) {
            if (layout.preOrder.at(gx) <= wxPre) {
              orderOk = false;
              break;
            }
            lca = layout.getLCA(lca, gx);
          }
          if (!orderOk || !lca || !layout.isUnconditionalFrom(wx, lca)) {
            continue;
          }
          bool ySafe = true;
          for (auto* wy : writes[y]) {
            if (layout.isDescendant(wy, lca)) {
              if (layout.preOrder.at(wy) > wxPre ||
                  layout.hasLoopBetween(wy, lca)) {
                ySafe = false;
                break;
              }
              for (auto* gx : reads[x]) {
                if (layout.hasLoopBetween(gx, lca)) {
                  ySafe = false;
                  break;
                }
              }
              if (!ySafe) {
                break;
              }
            }
          }
          if (!ySafe) {
            continue;
          }
          copyTarget[x] = y;
          teesToUnwrap.insert(wx);
        }

        if (!copyTarget.empty()) {
          // Resolve chains x -> y -> z.
          for (auto& [x, y] : copyTarget) {
            Index curr = y;
            while (copyTarget.count(curr)) {
              curr = copyTarget[curr];
            }
            y = curr;
          }
          struct CopyPropagator : public PostWalker<CopyPropagator> {
            const std::unordered_map<Index, Index>& copyTarget;
            const std::unordered_set<LocalSet*>& teesToUnwrap;
            CopyPropagator(const std::unordered_map<Index, Index>& copyTarget,
                           const std::unordered_set<LocalSet*>& teesToUnwrap)
              : copyTarget(copyTarget), teesToUnwrap(teesToUnwrap) {}
            void visitLocalGet(LocalGet* curr) {
              auto it = copyTarget.find(curr->index);
              if (it != copyTarget.end()) {
                curr->index = it->second;
              }
            }
            void visitLocalSet(LocalSet* curr) {
              if (teesToUnwrap.count(curr)) {
                replaceCurrent(curr->value);
              }
            }
          };
          CopyPropagator(copyTarget, teesToUnwrap).walk(func->body);
          funcChanged = true;
          layout.build(func->body);
        }
      }

      // Step 1B: Un-CSE hoisted vtable locals $v where $v is written once via
      // (struct.get $vt slot0 (local.tee $v V($x))) as cr0->target and read
      // only via (struct.get $vt sloti (local.get $v)) as cri->target on the
      // same receiver local $x.
      {
        Index numLocals = func->getNumLocals();
        std::vector<SmallVector<LocalSet*, 2>> writes(numLocals);
        std::vector<SmallVector<LocalGet*, 4>> reads(numLocals);
        for (auto* ls : FindAll<LocalSet>(func->body).list) {
          writes[ls->index].push_back(ls);
        }
        for (auto* lg : FindAll<LocalGet>(func->body).list) {
          reads[lg->index].push_back(lg);
        }

        for (Index v = func->getVarIndexBase(); v < numLocals; v++) {
          if (writes[v].size() != 1 || reads[v].empty()) {
            continue;
          }
          LocalSet* wv = writes[v][0];
          if (!wv->isTee()) {
            continue;
          }
          Expression* wvPar = layout.parent[wv];
          auto* sg0 = wvPar ? wvPar->dynCast<StructGet>() : nullptr;
          CallRef* cr0 = nullptr;
          bool directTarget = false;
          if (sg0 && sg0->ref == wv) {
            Expression* sg0Par = layout.parent[sg0];
            cr0 = sg0Par ? sg0Par->dynCast<CallRef>() : nullptr;
            if (cr0 && cr0->target != sg0) {
              cr0 = nullptr;
            }
          } else if (wvPar && wvPar->is<CallRef>()) {
            cr0 = wvPar->cast<CallRef>();
            if (cr0->target == wv) {
              directTarget = true;
            } else {
              cr0 = nullptr;
            }
          }
          if (!cr0 || cr0->isReturn || cr0->operands.empty()) {
            continue;
          }
          // Determine receiver local $x from cr0->operands[0].
          Expression* op0 = cr0->operands[0];
          Index recvLocal = Index(-1);
          if (auto* lg0 = op0->dynCast<LocalGet>()) {
            recvLocal = lg0->index;
          } else if (auto* tee0 = op0->dynCast<LocalSet>()) {
            if (tee0->isTee()) {
              recvLocal = tee0->index;
            }
          }
          if (recvLocal == Index(-1) || recvLocal == v) {
            continue;
          }
          // Check that cr0's later operands do not write recvLocal.
          bool writtenInCr0Args = false;
          for (Index i = 1; i < cr0->operands.size(); i++) {
            for (auto* ls : FindAll<LocalSet>(cr0->operands[i]).list) {
              if (ls->index == recvLocal) {
                writtenInCr0Args = true;
                break;
              }
            }
          }
          if (writtenInCr0Args) {
            continue;
          }
          Index numRecvGets = 0;
          if (!isPureVtableLookupOnLocal(wv->value, recvLocal, numRecvGets) ||
              numRecvGets != 1) {
            continue;
          }

          SmallVector<StructGet*, 4> readStructGets;
          SmallVector<CallRef*, 4> readDirectCallRefs;
          Expression* lca = cr0;
          Index wvPre = layout.preOrder.at(wv);
          bool readsOk = true;
          for (auto* gv : reads[v]) {
            if (layout.preOrder.at(gv) <= wvPre) {
              readsOk = false;
              break;
            }
            Expression* gvPar = layout.parent[gv];
            CallRef* cri = nullptr;
            StructGet* sgi = nullptr;
            if (!directTarget) {
              sgi = gvPar ? gvPar->dynCast<StructGet>() : nullptr;
              if (!sgi || sgi->ref != gv) {
                readsOk = false;
                break;
              }
              Expression* sgiPar = layout.parent[sgi];
              cri = sgiPar ? sgiPar->dynCast<CallRef>() : nullptr;
              if (!cri || cri->target != sgi) {
                readsOk = false;
                break;
              }
            } else {
              cri = gvPar ? gvPar->dynCast<CallRef>() : nullptr;
              if (!cri || cri->target != gv) {
                readsOk = false;
                break;
              }
            }
            if (cri->isReturn || cri->operands.empty()) {
              readsOk = false;
              break;
            }
            auto* lgi = cri->operands[0]->dynCast<LocalGet>();
            if (!lgi || lgi->index != recvLocal) {
              readsOk = false;
              break;
            }
            for (Index i = 1; i < cri->operands.size(); i++) {
              for (auto* ls : FindAll<LocalSet>(cri->operands[i]).list) {
                if (ls->index == recvLocal) {
                  readsOk = false;
                  break;
                }
              }
            }
            if (!readsOk) {
              break;
            }
            if (directTarget) {
              readDirectCallRefs.push_back(cri);
            } else {
              readStructGets.push_back(sgi);
            }
            lca = layout.getLCA(lca, cri);
          }
          if (!readsOk || !lca || !layout.isUnconditionalFrom(op0, lca)) {
            continue;
          }

          // Ensure recvLocal is not overwritten inside lca after op0.
          Index op0Pre = layout.preOrder.at(op0);
          bool recvSafe = true;
          for (auto* wx : writes[recvLocal]) {
            if (wx == op0) {
              continue;
            }
            if (layout.isDescendant(wx, lca)) {
              if (layout.preOrder.at(wx) > op0Pre ||
                  layout.hasLoopBetween(wx, lca)) {
                recvSafe = false;
                break;
              }
              for (auto* gv : reads[v]) {
                if (layout.hasLoopBetween(gv, lca)) {
                  recvSafe = false;
                  break;
                }
              }
              if (!recvSafe) {
                break;
              }
            }
          }
          if (!recvSafe) {
            continue;
          }

          Expression* vExpr = wv->value;
          if (directTarget) {
            cr0->target = vExpr;
            for (auto* cri : readDirectCallRefs) {
              cri->target = ExpressionManipulator::copy(vExpr, *wasm);
            }
          } else {
            sg0->ref = vExpr;
            for (auto* sgi : readStructGets) {
              sgi->ref = ExpressionManipulator::copy(vExpr, *wasm);
            }
          }
          funcChanged = true;
        }
      }

      if (funcChanged) {
        ReFinalize().walkFunctionInModule(func, wasm);
        if (func->effects) {
          func->effects =
            std::make_unique<EffectAnalyzer>(getPassOptions(), *wasm, func);
        }
      }
    }
  }

  static bool isPureReceiverExpr(Expression* e,
                                 LocalSet*& outTee,
                                 Index& numTees) {
    if (!e || e->type == Type::unreachable) {
      return false;
    }
    if (e->is<LocalGet>()) {
      return true;
    }
    if (auto* ls = e->dynCast<LocalSet>()) {
      if (!ls->isTee() || numTees >= 1) {
        return false;
      }
      outTee = ls;
      numTees++;
      return true;
    }
    if (auto* sg = e->dynCast<StructGet>()) {
      return sg->ref && sg->ref->type.isStruct() &&
             isPureReceiverExpr(sg->ref, outTee, numTees);
    }
    if (auto* rc = e->dynCast<RefCast>()) {
      return !rc->desc && rc->ref &&
             isPureReceiverExpr(rc->ref, outTee, numTees);
    }
    if (auto* ra = e->dynCast<RefAs>()) {
      return ra->op == RefAsNonNull && ra->value &&
             isPureReceiverExpr(ra->value, outTee, numTees);
    }
    return false;
  }

  static bool matchesCoupledReceiver(Expression* a0, Expression* targetSub) {
    if (!a0 || !targetSub) {
      return false;
    }
    if (auto* ls = a0->dynCast<LocalSet>()) {
      if (!ls->isTee()) {
        return false;
      }
      if (auto* lg = targetSub->dynCast<LocalGet>()) {
        if (lg->index == ls->index) {
          return true;
        }
        if (auto* innerLg = ls->value->dynCast<LocalGet>()) {
          if (lg->index == innerLg->index) {
            return true;
          }
        }
      }
      return false;
    }
    if (auto* lgA = a0->dynCast<LocalGet>()) {
      auto* lgB = targetSub->dynCast<LocalGet>();
      return lgB && lgA->index == lgB->index;
    }
    if (auto* sgA = a0->dynCast<StructGet>()) {
      auto* sgB = targetSub->dynCast<StructGet>();
      if (!sgB || sgA->index != sgB->index || sgA->signed_ != sgB->signed_ ||
          !sgA->ref || !sgB->ref || !sgA->ref->type.isStruct() ||
          !sgB->ref->type.isStruct() ||
          sgA->ref->type.getHeapType() != sgB->ref->type.getHeapType()) {
        return false;
      }
      return matchesCoupledReceiver(sgA->ref, sgB->ref);
    }
    if (auto* rcA = a0->dynCast<RefCast>()) {
      auto* rcB = targetSub->dynCast<RefCast>();
      if (!rcB || rcA->desc || rcB->desc || rcA->type != rcB->type) {
        return false;
      }
      return matchesCoupledReceiver(rcA->ref, rcB->ref);
    }
    if (auto* raA = a0->dynCast<RefAs>()) {
      auto* raB = targetSub->dynCast<RefAs>();
      if (!raB || raA->op != raB->op) {
        return false;
      }
      return matchesCoupledReceiver(raA->value, raB->value);
    }
    return false;
  }

  bool buildInlineTargetSkel(Expression* e,
                             Expression* a0,
                             Type parentDemanded,
                             Type& demandedByTarget,
                             std::unordered_map<Name, uint32_t>& callIds,
                             const std::unordered_set<Name>& disallowedHelpers,
                             SmallVector<uint32_t, 16>& out,
                             unsigned& siteTargetBytes,
                             unsigned& helperTargetBytes,
                             unsigned& numRecvMatches) {
    if (!e || e->type == Type::unreachable) {
      return false;
    }
    if (matchesCoupledReceiver(a0, e)) {
      out.push_back(1);
      siteTargetBytes += unsigned(Measurer::measure(e));
      helperTargetBytes += 2;
      demandedByTarget =
        Type::getGreatestLowerBound(demandedByTarget, parentDemanded);
      numRecvMatches++;
      return demandedByTarget != Type::none && !demandedByTarget.isNull();
    }
    if (auto* sg = e->dynCast<StructGet>()) {
      if (!sg->ref || sg->ref->type == Type::unreachable ||
          !sg->ref->type.isStruct()) {
        return false;
      }
      uint64_t htId = sg->ref->type.getHeapType().getID();
      out.push_back(2);
      out.push_back(sg->index);
      out.push_back(sg->signed_ ? 1 : 0);
      out.push_back(uint32_t(htId));
      out.push_back(uint32_t(htId >> 32));
      siteTargetBytes += 4;
      helperTargetBytes += 4;
      Type childDemanded(sg->ref->type.getHeapType(), Nullable);
      return buildInlineTargetSkel(sg->ref,
                                   a0,
                                   childDemanded,
                                   demandedByTarget,
                                   callIds,
                                   disallowedHelpers,
                                   out,
                                   siteTargetBytes,
                                   helperTargetBytes,
                                   numRecvMatches);
    }
    if (auto* rc = e->dynCast<RefCast>()) {
      if (rc->desc || !rc->ref || rc->ref->type == Type::unreachable) {
        return false;
      }
      uint64_t tId = rc->type.getID();
      out.push_back(3);
      out.push_back(uint32_t(tId));
      out.push_back(uint32_t(tId >> 32));
      siteTargetBytes += 4;
      helperTargetBytes += 4;
      Type childDemanded(rc->ref->type.getHeapType().getTop(), Nullable);
      return buildInlineTargetSkel(rc->ref,
                                   a0,
                                   childDemanded,
                                   demandedByTarget,
                                   callIds,
                                   disallowedHelpers,
                                   out,
                                   siteTargetBytes,
                                   helperTargetBytes,
                                   numRecvMatches);
    }
    if (auto* ra = e->dynCast<RefAs>()) {
      if (ra->op != RefAsNonNull || !ra->value ||
          ra->value->type == Type::unreachable) {
        return false;
      }
      out.push_back(4);
      siteTargetBytes += 1;
      helperTargetBytes += 1;
      Type childDemanded =
        parentDemanded.isRef()
          ? Type(parentDemanded.getHeapType(), Nullable)
          : Type(ra->value->type.getHeapType().getTop(), Nullable);
      return buildInlineTargetSkel(ra->value,
                                   a0,
                                   childDemanded,
                                   demandedByTarget,
                                   callIds,
                                   disallowedHelpers,
                                   out,
                                   siteTargetBytes,
                                   helperTargetBytes,
                                   numRecvMatches);
    }
    if (auto* ag = e->dynCast<ArrayGet>()) {
      if (!ag->ref || ag->ref->type == Type::unreachable ||
          !ag->ref->type.isArray() || !ag->index) {
        return false;
      }
      uint64_t htId = ag->ref->type.getHeapType().getID();
      out.push_back(5);
      out.push_back(ag->signed_ ? 1 : 0);
      out.push_back(uint32_t(htId));
      out.push_back(uint32_t(htId >> 32));
      siteTargetBytes += 4;
      helperTargetBytes += 4;
      Type refDemanded(ag->ref->type.getHeapType(), Nullable);
      return buildInlineTargetSkel(ag->ref,
                                   a0,
                                   refDemanded,
                                   demandedByTarget,
                                   callIds,
                                   disallowedHelpers,
                                   out,
                                   siteTargetBytes,
                                   helperTargetBytes,
                                   numRecvMatches) &&
             buildInlineTargetSkel(ag->index,
                                   a0,
                                   Type::i32,
                                   demandedByTarget,
                                   callIds,
                                   disallowedHelpers,
                                   out,
                                   siteTargetBytes,
                                   helperTargetBytes,
                                   numRecvMatches);
    }
    if (auto* c = e->dynCast<Const>()) {
      if (c->value.type != Type::i32) {
        return false;
      }
      out.push_back(6);
      out.push_back(uint32_t(c->value.geti32()));
      unsigned b = getShallowBytes(c);
      siteTargetBytes += b;
      helperTargetBytes += b;
      return true;
    }
    if (auto* call = e->dynCast<Call>()) {
      if (call->isReturn || call->operands.size() != 1 ||
          disallowedHelpers.count(call->target) ||
          !isPureVtableHelper(call->target)) {
        return false;
      }
      Function* callee = wasm->getFunction(call->target);
      uint32_t id =
        callIds.try_emplace(call->target, uint32_t(callIds.size() + 1))
          .first->second;
      out.push_back(7);
      out.push_back(id);
      siteTargetBytes += 3;
      helperTargetBytes += 3;
      return buildInlineTargetSkel(call->operands[0],
                                   a0,
                                   callee->getLocalType(0),
                                   demandedByTarget,
                                   callIds,
                                   disallowedHelpers,
                                   out,
                                   siteTargetBytes,
                                   helperTargetBytes,
                                   numRecvMatches);
    }
    return false;
  }

  static Expression* replaceCoupledReceiverInCopy(Expression* orig,
                                                  Expression* copy,
                                                  Expression* a0,
                                                  Index paramIdx,
                                                  Type paramType,
                                                  Module& wasm) {
    if (matchesCoupledReceiver(a0, orig)) {
      return Builder(wasm).makeLocalGet(paramIdx, paramType);
    }
    if (auto* sgO = orig->dynCast<StructGet>()) {
      auto* sgC = copy->cast<StructGet>();
      sgC->ref = replaceCoupledReceiverInCopy(
        sgO->ref, sgC->ref, a0, paramIdx, paramType, wasm);
      return sgC;
    }
    if (auto* rcO = orig->dynCast<RefCast>()) {
      auto* rcC = copy->cast<RefCast>();
      rcC->ref = replaceCoupledReceiverInCopy(
        rcO->ref, rcC->ref, a0, paramIdx, paramType, wasm);
      return rcC;
    }
    if (auto* raO = orig->dynCast<RefAs>()) {
      auto* raC = copy->cast<RefAs>();
      raC->value = replaceCoupledReceiverInCopy(
        raO->value, raC->value, a0, paramIdx, paramType, wasm);
      return raC;
    }
    if (auto* agO = orig->dynCast<ArrayGet>()) {
      auto* agC = copy->cast<ArrayGet>();
      agC->ref = replaceCoupledReceiverInCopy(
        agO->ref, agC->ref, a0, paramIdx, paramType, wasm);
      agC->index = replaceCoupledReceiverInCopy(
        agO->index, agC->index, a0, paramIdx, paramType, wasm);
      return agC;
    }
    if (auto* cO = orig->dynCast<Call>()) {
      auto* cC = copy->cast<Call>();
      for (Index i = 0; i < cO->operands.size(); i++) {
        cC->operands[i] = replaceCoupledReceiverInCopy(
          cO->operands[i], cC->operands[i], a0, paramIdx, paramType, wasm);
      }
      return cC;
    }
    return copy;
  }

  bool foldCallRefDispatch(bool allowInlinePart2 = true) {
    // Collect referenced functions (exports, tables, globals, ref.func) and
    // total direct call counts across the module.
    std::unordered_set<Name> disallowedHelpers;
    for (auto& ex : wasm->exports) {
      if (ex->kind == ExternalKind::Function) {
        disallowedHelpers.insert(*ex->getInternalName());
      }
    }
    for (auto& seg : wasm->elementSegments) {
      for (auto* expr : seg->data) {
        for (auto* rf : FindAll<RefFunc>(expr).list) {
          disallowedHelpers.insert(rf->func);
        }
      }
    }
    for (auto& g : wasm->globals) {
      if (!g->imported() && g->init) {
        for (auto* rf : FindAll<RefFunc>(g->init).list) {
          disallowedHelpers.insert(rf->func);
        }
      }
    }
    std::unordered_map<Name, Index> totalCalls;
    for (auto& f : wasm->functions) {
      if (f->imported()) {
        continue;
      }
      for (auto* rf : FindAll<RefFunc>(f->body).list) {
        disallowedHelpers.insert(rf->func);
      }
      for (auto* c : FindAll<Call>(f->body).list) {
        totalCalls[c->target]++;
      }
    }

    auto isDispatchHelperBody = [](Function* func, CallRef* cr) -> bool {
      if (!cr || func->body != cr || !func->vars.empty() ||
          func->getNumParams() != cr->operands.size() || cr->operands.empty()) {
        return false;
      }
      for (Index i = 0; i < cr->operands.size(); i++) {
        auto* lg = cr->operands[i]->dynCast<LocalGet>();
        if (!lg || lg->index != i) {
          return false;
        }
      }
      return true;
    };

    struct CRSite {
      Function* func;
      CallRef* cr;
      HeapType sigHt;
      Index teeLocal;
      LocalSet* teeExpr;
      bool isDirectTee;
      unsigned numTeeGetsInTarget;
      unsigned siteTargetBytes;
      unsigned helperTargetBytes;
      Type demandedByTarget;
    };

    // Part 1: (helperName, sigHeapTypeID) -> sites
    std::map<std::pair<Name, uint64_t>, std::vector<CRSite>> part1Groups;
    std::map<std::pair<Name, uint64_t>, Function*> existingPart1Helpers;
    // Part 2: (sigHeapTypeID, numArgs, skel) -> sites
    using Part2Key = std::tuple<uint64_t, Index, std::vector<uint32_t>>;
    std::map<Part2Key, std::vector<CRSite>> part2Groups;
    std::map<Part2Key, Function*> existingPart2Helpers;

    std::unordered_map<Name, uint32_t> callIds;
    std::unordered_map<Function*, std::vector<Index>> funcLocalReads;

    for (auto& f : wasm->functions) {
      if (f->imported()) {
        continue;
      }
      Function* func = f.get();
      auto allCallRefs = FindAll<CallRef>(func->body).list;
      if (allCallRefs.empty()) {
        continue;
      }
      std::vector<Index> reads(func->getNumLocals(), 0);
      for (auto* lg : FindAll<LocalGet>(func->body).list) {
        reads[lg->index]++;
      }
      funcLocalReads[func] = std::move(reads);

      for (auto* cr : allCallRefs) {
        if (cr->isReturn || cr->type == Type::unreachable ||
            cr->operands.empty() || !cr->target ||
            cr->target->type == Type::unreachable ||
            !cr->target->type.isSignature()) {
          continue;
        }
        bool hasUnreachableOp = false;
        for (auto* op : cr->operands) {
          if (op->type == Type::unreachable) {
            hasUnreachableOp = true;
            break;
          }
        }
        if (hasUnreachableOp) {
          continue;
        }
        HeapType sigHt = cr->target->type.getHeapType();
        Signature sig = sigHt.getSignature();
        if (sig.params.size() != cr->operands.size() ||
            sig.results != cr->type) {
          continue;
        }

        if (isDispatchHelperBody(func, cr)) {
          if (!disallowedHelpers.count(func->name) &&
              func->getResults() == sig.results) {
            bool paramsMatch = true;
            for (Index i = 1; i < sig.params.size(); i++) {
              if (func->getLocalType(i) != sig.params[i]) {
                paramsMatch = false;
                break;
              }
            }
            if (paramsMatch) {
              Expression* a0 = cr->operands[0];
              if (auto* targetCall = cr->target->dynCast<Call>()) {
                if (!targetCall->isReturn &&
                    targetCall->operands.size() == 1 &&
                    matchesCoupledReceiver(a0, targetCall->operands[0])) {
                  existingPart1Helpers.try_emplace(
                    {targetCall->target, sigHt.getID()}, func);
                }
              } else if (allowInlinePart2 && (cr->target->is<StructGet>() ||
                                              cr->target->is<RefCast>())) {
                SmallVector<uint32_t, 16> skel;
                unsigned sBytes = 0, hBytes = 0, nMatches = 0;
                Type demanded = sig.params[0];
                if (buildInlineTargetSkel(cr->target,
                                          a0,
                                          sig.params[0],
                                          demanded,
                                          callIds,
                                          disallowedHelpers,
                                          skel,
                                          sBytes,
                                          hBytes,
                                          nMatches) &&
                    nMatches > 0) {
                  std::vector<uint32_t> skelVec(skel.begin(), skel.end());
                  existingPart2Helpers.try_emplace(
                    {sigHt.getID(),
                     Index(cr->operands.size()),
                     std::move(skelVec)},
                    func);
                }
              }
            }
          }
          continue;
        }

        Expression* a0 = cr->operands[0];
        LocalSet* teeExpr = nullptr;
        Index numTees = 0;
        if (!isPureReceiverExpr(a0, teeExpr, numTees)) {
          continue;
        }
        bool isCompoundReceiver = !a0->is<LocalGet>() && !a0->is<LocalSet>();

        // Check that later operands do not clobber any local read in a0 or
        // have side effects if a0 is a compound receiver (e.g. StructGet).
        bool laterArgsSafe = true;
        std::unordered_set<Index> a0Locals;
        for (auto* lg : FindAll<LocalGet>(a0).list) {
          a0Locals.insert(lg->index);
        }
        if (teeExpr) {
          a0Locals.insert(teeExpr->index);
        }
        for (Index i = 1; i < cr->operands.size(); i++) {
          for (auto* ls : FindAll<LocalSet>(cr->operands[i]).list) {
            if (a0Locals.count(ls->index)) {
              laterArgsSafe = false;
              break;
            }
          }
          if (!laterArgsSafe) {
            break;
          }
          if (isCompoundReceiver &&
              EffectAnalyzer(getPassOptions(), *wasm, cr->operands[i])
                .hasAnything()) {
            laterArgsSafe = false;
            break;
          }
        }
        if (!laterArgsSafe) {
          continue;
        }

        auto countTeeGets = [&](Expression* targetExpr) -> unsigned {
          if (!teeExpr) {
            return 0;
          }
          unsigned cnt = 0;
          for (auto* lg : FindAll<LocalGet>(targetExpr).list) {
            if (lg->index == teeExpr->index) {
              cnt++;
            }
          }
          return cnt;
        };

        if (auto* targetCall = cr->target->dynCast<Call>()) {
          if (targetCall->isReturn || targetCall->operands.size() != 1) {
            continue;
          }
          if (!matchesCoupledReceiver(a0, targetCall->operands[0])) {
            continue;
          }
          Function* h = wasm->getFunctionOrNull(targetCall->target);
          if (!h || h->imported() || h->getNumParams() != 1 ||
              !h->getResults().isSingle() || !h->getResults().isRef() ||
              disallowedHelpers.count(h->name)) {
            continue;
          }
          Type param0Type =
            Type::getGreatestLowerBound(sig.params[0], h->getLocalType(0));
          if (param0Type == Type::none || param0Type.isNull() ||
              !Type::isSubType(a0->type, param0Type)) {
            continue;
          }
          unsigned tGets = countTeeGets(cr->target);
          bool isDirectTee = (a0 == teeExpr) && tGets > 0;
          unsigned sBytes = unsigned(Measurer::measure(cr->target));
          part1Groups[{h->name, sigHt.getID()}].push_back(
            {func,
             cr,
             sigHt,
             teeExpr ? teeExpr->index : Index(-1),
             tGets > 0 ? teeExpr : nullptr,
             isDirectTee,
             tGets,
             sBytes,
             5,
             param0Type});
        } else if (allowInlinePart2 &&
                   (cr->target->is<StructGet>() || cr->target->is<RefCast>())) {
          SmallVector<uint32_t, 16> skel;
          unsigned sBytes = 0;
          unsigned hBytes = 0;
          unsigned numMatches = 0;
          Type demanded = sig.params[0];
          if (!buildInlineTargetSkel(cr->target,
                                     a0,
                                     sig.params[0],
                                     demanded,
                                     callIds,
                                     disallowedHelpers,
                                     skel,
                                     sBytes,
                                     hBytes,
                                     numMatches) ||
              numMatches == 0) {
            continue;
          }
          if (!Type::isSubType(a0->type, demanded)) {
            continue;
          }
          unsigned tGets = countTeeGets(cr->target);
          bool isDirectTee = (a0 == teeExpr) && tGets > 0;
          std::vector<uint32_t> skelVec(skel.begin(), skel.end());
          part2Groups[{sigHt.getID(),
                       Index(cr->operands.size()),
                       std::move(skelVec)}]
            .push_back({func,
                        cr,
                        sigHt,
                        teeExpr ? teeExpr->index : Index(-1),
                        tGets > 0 ? teeExpr : nullptr,
                        isDirectTee,
                        tGets,
                        sBytes,
                        hBytes,
                        demanded});
        }
      }
    }

    Builder builder(*wasm);
    struct PendingSite {
      CRSite site;
      Name helperName;
    };
    std::vector<PendingSite> acceptedSites;

    // Process Part 1 groups.
    for (auto& [keyPair, sites] : part1Groups) {
      Name hName = keyPair.first;
      HeapType sigHt = sites[0].sigHt;
      Function* h = wasm->getFunction(hName);
      Signature sig = sigHt.getSignature();
      Index nArgs = sig.params.size();
      Type param0Type =
        Type::getGreatestLowerBound(sig.params[0], h->getLocalType(0));
      if (param0Type == Type::none || param0Type.isNull()) {
        continue;
      }

      unsigned siteSavings = 0;
      for (const auto& s : sites) {
        unsigned sv = s.siteTargetBytes;
        if (s.teeExpr &&
            funcLocalReads[s.func][s.teeLocal] == s.numTeeGetsInTarget) {
          sv += 2;
        }
        siteSavings += sv;
      }

      auto exIt = existingPart1Helpers.find(keyPair);
      if (exIt != existingPart1Helpers.end() &&
          exIt->second->getLocalType(0) == param0Type) {
        for (const auto& s : sites) {
          acceptedSites.push_back({s, exIt->second->name});
        }
        continue;
      }

      bool canModifyInPlace =
        sites.size() == totalCalls[hName] && h->vars.empty() &&
        FindAll<LocalSet>(h->body).list.empty() &&
        FindAll<Return>(h->body).list.empty() &&
        FindAll<CallRef>(h->body).list.empty();

      if (canModifyInPlace) {
        unsigned hCost = 3 + 2 * nArgs;
        if (siteSavings <= hCost) {
          continue;
        }
        for (auto* lg : FindAll<LocalGet>(h->body).list) {
          assert(lg->index == 0);
          lg->type = param0Type;
        }
        std::vector<Type> newParams(nArgs);
        newParams[0] = param0Type;
        for (Index i = 1; i < nArgs; i++) {
          newParams[i] = sig.params[i];
        }
        std::vector<Expression*> callArgs(nArgs);
        for (Index i = 0; i < nArgs; i++) {
          callArgs[i] = builder.makeLocalGet(i, newParams[i]);
        }
        h->setParams(Type(newParams));
        h->setResults(sig.results);
        h->body = builder.makeCallRef(h->body, callArgs, sig.results, false);
        ReFinalize().walkFunctionInModule(h, wasm);
        if (h->effects) {
          h->effects =
            std::make_unique<EffectAnalyzer>(getPassOptions(), *wasm, h);
        }
        for (const auto& s : sites) {
          acceptedSites.push_back({s, hName});
        }
      } else if (sites.size() >= 3) {
        unsigned hCost = 14 + 2 * nArgs;
        if (siteSavings <= hCost) {
          continue;
        }
        std::vector<Type> newParams(nArgs);
        newParams[0] = param0Type;
        for (Index i = 1; i < nArgs; i++) {
          newParams[i] = sig.params[i];
        }
        std::vector<Expression*> callArgs(nArgs);
        for (Index i = 0; i < nArgs; i++) {
          callArgs[i] = builder.makeLocalGet(i, newParams[i]);
        }
        Expression* callH = builder.makeCall(
          hName, {builder.makeLocalGet(0, param0Type)}, h->getResults());
        Expression* newBody =
          builder.makeCallRef(callH, callArgs, sig.results, false);
        Name newName = Names::getValidFunctionName(*wasm, "outlined_cr");
        Function* newFunc = wasm->addFunction(Builder::makeFunction(
          newName, Signature(Type(newParams), sig.results), {}, newBody));
        initHelperEffects(newFunc);
        for (const auto& s : sites) {
          acceptedSites.push_back({s, newName});
        }
      }
    }

    // Process Part 2 groups.
    for (auto& [keyTuple, sites] : part2Groups) {
      HeapType sigHt = sites[0].sigHt;
      Signature sig = sigHt.getSignature();
      Index nArgs = sig.params.size();
      unsigned hTargetBytes = sites[0].helperTargetBytes;

      Type param0Type = sites[0].demandedByTarget;
      for (const auto& s : sites) {
        param0Type = Type::getGreatestLowerBound(param0Type, s.demandedByTarget);
      }
      if (param0Type == Type::none || param0Type.isNull()) {
        continue;
      }
      bool allTypeOk = true;
      for (const auto& s : sites) {
        if (!Type::isSubType(s.cr->operands[0]->type, param0Type)) {
          allTypeOk = false;
          break;
        }
      }
      if (!allTypeOk) {
        continue;
      }

      auto exIt = existingPart2Helpers.find(keyTuple);
      if (exIt != existingPart2Helpers.end() &&
          Type::isSubType(param0Type, exIt->second->getLocalType(0))) {
        Type exParam0 = exIt->second->getLocalType(0);
        bool exOk = true;
        for (const auto& s : sites) {
          if (!Type::isSubType(s.cr->operands[0]->type, exParam0)) {
            exOk = false;
            break;
          }
        }
        if (exOk) {
          for (const auto& s : sites) {
            acceptedSites.push_back({s, exIt->second->name});
          }
          continue;
        }
      }

      if (sites.size() < 2) {
        continue;
      }

      unsigned siteSavings = 0;
      for (const auto& s : sites) {
        unsigned sv = s.siteTargetBytes;
        if (s.teeExpr &&
            funcLocalReads[s.func][s.teeLocal] == s.numTeeGetsInTarget) {
          sv += 2;
        }
        siteSavings += sv;
      }
      unsigned hCost = 9 + 2 * nArgs + hTargetBytes;
      if (siteSavings <= hCost) {
        continue;
      }

      Expression* targetCopy =
        ExpressionManipulator::copy(sites[0].cr->target, *wasm);
      targetCopy = replaceCoupledReceiverInCopy(sites[0].cr->target,
                                                targetCopy,
                                                sites[0].cr->operands[0],
                                                0,
                                                param0Type,
                                                *wasm);
      std::vector<Type> newParams(nArgs);
      newParams[0] = param0Type;
      for (Index i = 1; i < nArgs; i++) {
        newParams[i] = sig.params[i];
      }
      std::vector<Expression*> callArgs(nArgs);
      for (Index i = 0; i < nArgs; i++) {
        callArgs[i] = builder.makeLocalGet(i, newParams[i]);
      }
      Expression* newBody =
        builder.makeCallRef(targetCopy, callArgs, sig.results, false);
      Name newName = Names::getValidFunctionName(*wasm, "outlined_cr");
      Function* newFunc = wasm->addFunction(Builder::makeFunction(
        newName, Signature(Type(newParams), sig.results), {}, newBody));
      ReFinalize().walkFunctionInModule(newFunc, wasm);
      initHelperEffects(newFunc);
      for (const auto& s : sites) {
        acceptedSites.push_back({s, newName});
      }
    }

    if (acceptedSites.empty()) {
      return false;
    }

    // Count total removed LocalGet reads per (func, local) across accepted
    // sites so we know when a local has zero remaining reads in func.
    std::unordered_map<Function*, std::unordered_map<Index, Index>>
      removedReads;
    for (const auto& ps : acceptedSites) {
      if (ps.site.teeExpr) {
        removedReads[ps.site.func][ps.site.teeLocal] +=
          ps.site.numTeeGetsInTarget;
      }
    }

    std::unordered_map<
      Function*,
      std::unordered_map<Expression*, CallRefReplacer::RepInfo>>
      funcReplacements;
    std::unordered_map<Function*, std::unordered_set<LocalSet*>>
      funcInnerTees;
    for (const auto& ps : acceptedSites) {
      const auto& s = ps.site;
      bool canUnwrap =
        s.teeExpr &&
        funcLocalReads[s.func][s.teeLocal] ==
          removedReads[s.func][s.teeLocal];
      funcReplacements[s.func][s.cr] = {ps.helperName,
                                        canUnwrap && s.isDirectTee};
      if (canUnwrap && !s.isDirectTee) {
        funcInnerTees[s.func].insert(s.teeExpr);
      }
    }

    for (auto& [func, reps] : funcReplacements) {
      CallRefReplacer replacer(*wasm);
      replacer.replacements = std::move(reps);
      if (auto it = funcInnerTees.find(func); it != funcInnerTees.end()) {
        replacer.innerTeesToUnwrap = std::move(it->second);
      }
      replacer.walk(func->body);
      ReFinalize().walkFunctionInModule(func, wasm);
      if (func->effects) {
        func->effects =
          std::make_unique<EffectAnalyzer>(getPassOptions(), *wasm, func);
      }
    }
    return true;
  }

  static Expression* unwrapRefAsNonNull(Expression* e) {
    while (e) {
      if (auto* ra = e->dynCast<RefAs>()) {
        if (ra->op == RefAsNonNull) {
          e = ra->value;
          continue;
        }
      }
      break;
    }
    return e;
  }

  void refreshFuncEffects(Function* func) {
    if (!func || !func->effects) {
      return;
    }
    bool wasIdempotent = func->effects->isIdempotent;
    bool wasOnlyInit = func->effects->onlyTouchesInitGlobals;
    auto newEff =
      std::make_unique<EffectAnalyzer>(getPassOptions(), *wasm, func);
    if (wasIdempotent && wasOnlyInit && !newEff->throws()) {
      newEff->isIdempotent = true;
      newEff->onlyTouchesInitGlobals = true;
    }
    func->effects = std::move(newEff);
  }

  void initHelperEffects(Function* newFunc) {
    if (!hasModuleEffects || !newFunc || newFunc->imported()) {
      return;
    }
    auto newEff =
      std::make_unique<EffectAnalyzer>(getPassOptions(), *wasm, newFunc);
    if (getPassOptions().shrinkLevel > 0 && !newEff->throws() &&
        !newEff->suspends && !newEff->calls && !newEff->mayNotReturn &&
        !newEff->danglingPop && !newEff->hasSynchronization() &&
        !newEff->writesMemory && !newEff->writesSharedMemory &&
        !newEff->writesTable && !newEff->writesStruct &&
        !newEff->writesSharedStruct && !newEff->writesSharedArray) {
      if (!newEff->hasNonTrapSideEffects() && !newEff->generative &&
          !newEff->readsMutableGlobalState()) {
        newEff->isIdempotent = true;
        newEff->onlyTouchesInitGlobals = true;
      } else {
        bool onlyIdempotentInitCalls = true;
        std::function<void(Expression*)> checkNode = [&](Expression* e) {
          if (!onlyIdempotentInitCalls || !e) {
            return;
          }
          if (auto* c = e->dynCast<Call>()) {
            auto* t = wasm->getFunctionOrNull(c->target);
            if (!t || !t->effects || !t->effects->isIdempotent ||
                !t->effects->onlyTouchesInitGlobals || c->isReturn) {
              onlyIdempotentInitCalls = false;
              return;
            }
            for (auto* op : c->operands) {
              checkNode(op);
            }
            return;
          }
          ShallowEffectAnalyzer sEff(getPassOptions(), *wasm, e);
          sEff.localsRead.clear();
          sEff.localsWritten.clear();
          sEff.breakTargets.clear();
          if (sEff.hasNonTrapSideEffects() || sEff.generative ||
              sEff.readsMutableGlobalState()) {
            onlyIdempotentInitCalls = false;
            return;
          }
          for (auto* ch : ChildIterator(e)) {
            checkNode(ch);
          }
        };
        checkNode(newFunc->body);
        if (onlyIdempotentInitCalls) {
          newEff->generative = false;
          newEff->isIdempotent = true;
          newEff->onlyTouchesInitGlobals = true;
        }
      }
    }
    newFunc->effects = std::move(newEff);
  }

  static Type getBinaryVarType(Type t) {
    if (t.isRef()) {
      return Type(t.getHeapType(), Nullable, Inexact);
    }
    return t;
  }

  static bool subtreeTargetsLabel(Expression* root, Name target) {
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

  bool isIdempotentCallExpr(Expression* curr) const {
    if (auto* call = curr->dynCast<Call>()) {
      if (call->isReturn) {
        return false;
      }
      auto* target = wasm->getFunctionOrNull(call->target);
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
    if (curr->is<StructGet>() || curr->is<RefCast>() || curr->is<RefTest>()) {
      return 4;
    }
    if (curr->is<ArrayGet>() || curr->is<Load>()) {
      return 3;
    }
    if (auto* call = curr->dynCast<Call>()) {
      if (auto* f = wasm->getFunctionOrNull(call->target)) {
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

  struct TeeReadRecord {
    Expression* orig = nullptr;
    SmallVector<Index, 4> localsRead;
    SmallVector<Name, 2> mutableGlobalsRead;
    SmallVector<std::pair<HeapType, Index>, 2> mutableStructFieldsRead;
    bool readsMutableStruct = false;
    bool readsUnknownMutableStruct = false;
    bool readsMutableArray = false;
    bool readsMemory = false;
    bool readsTable = false;
  };

  bool isTeeKilledByEffects(const TeeReadRecord& rec,
                            const EffectAnalyzer& eff,
                            Expression* killerExpr = nullptr) const {
    if (killerExpr && isIdempotentCallExpr(killerExpr) &&
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
          eff.calls || !killerExpr) {
        return true;
      }
      if (!FindAll<CallRef>(killerExpr).list.empty() ||
          !FindAll<CallIndirect>(killerExpr).list.empty()) {
        return true;
      }
      bool aliasFound = false;
      for (auto* call : FindAll<Call>(killerExpr).list) {
        auto* target = wasm->getFunctionOrNull(call->target);
        if (!target || !target->effects) {
          aliasFound = true;
          break;
        }
        const auto& tEff = *target->effects;
        if ((tEff.writesStruct || tEff.writesSharedStruct || tEff.calls) &&
            !(tEff.isIdempotent && tEff.onlyTouchesInitGlobals)) {
          aliasFound = true;
          break;
        }
      }
      if (aliasFound) {
        return true;
      }
      for (auto* ss : FindAll<StructSet>(killerExpr).list) {
        if (!ss->ref->type.isRef()) {
          aliasFound = true;
          break;
        }
        HeapType wHT = ss->ref->type.getHeapType();
        for (auto& [rHT, rIdx] : rec.mutableStructFieldsRead) {
          if (ss->index == rIdx &&
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

  void unCseOverCsedTees() {
    auto& options = getPassOptions();
    for (auto& f : wasm->functions) {
      if (f->imported()) {
        continue;
      }
      Function* func = f.get();
      if (!func->body || func->getNumLocals() == 0) {
        continue;
      }

      bool anyChanged = false;
      std::unordered_set<Index> allUnCsedLocals;

      auto isDuplicableValue = [&](Expression* root,
                                   TeeReadRecord& rec,
                                   Index& estBytesOut,
                                   LocalSet*& innerTeeOut) -> bool {
        estBytesOut = 0;
        innerTeeOut = nullptr;
        bool ok = true;
        std::function<void(Expression*, bool)> walkVal =
          [&](Expression* e, bool onLeftmostPath) {
          if (!ok || !e) {
            return;
          }
          if (auto* ls = e->dynCast<LocalSet>()) {
            if (e != root && onLeftmostPath && !innerTeeOut && ls->isTee() &&
                ls->value->type.isConcrete() &&
                Type::isSubType(ls->value->type,
                                func->getLocalType(ls->index))) {
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
          estBytesOut += getShallowEstBytes(e);
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
            auto* target = wasm->getFunctionOrNull(call->target);
            if (!target || (target->imported() && !target->effects)) {
              ok = false;
              return;
            }
            std::optional<EffectAnalyzer> localEff;
            const EffectAnalyzer* tEffPtr = target->effects.get();
            if (!tEffPtr && target->body &&
                Measurer::measure(target->body) <= 16) {
              localEff.emplace(options, *wasm, target);
              tEffPtr = &*localEff;
            }
            if (!tEffPtr) {
              ok = false;
              return;
            }
            const auto& tEff = *tEffPtr;
            bool idempotentInit =
              tEff.isIdempotent && tEff.onlyTouchesInitGlobals &&
              !tEff.throws();
            bool pureReadOnly = !tEff.hasNonTrapSideEffects() &&
                                !tEff.generative && !tEff.throws();
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
                rec.readsUnknownMutableStruct = true;
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
          ShallowEffectAnalyzer eff(options, *wasm, e);
          if (!options.trapsNeverHappen && eff.trap) {
            ok = false;
            return;
          }
          eff.trap = false;
          if (eff.hasNonTrapSideEffects() ||
              Properties::isShallowlyGenerative(e, func, *wasm)) {
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
            varTypeCounts[getBinaryVarType(func->getLocalType(v))]++;
          }
        }

        std::vector<bool> isCandidate(numLocals, false);
        std::vector<TeeReadRecord> candRec(numLocals);
        std::vector<Index> candEstBytes(numLocals, 0);
        std::vector<LocalSet*> candInnerTee(numLocals, nullptr);
        bool anyCand = false;

        for (Index x = 0; x < numLocals; x++) {
          if (writes[x].empty()) {
            continue;
          }
          bool isSingleWriteVar = (x >= varBase && writes[x].size() == 1);
          if (options.shrinkLevel < 2 &&
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
              (options.shrinkLevel < 2 ||
               EffectAnalyzer(options, *wasm, innerTee->value)
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

        std::unordered_map<Index, Index> activeTees;
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
            if (isTeeKilledByEffects(candRec[it->first], eff, killerExpr)) {
              it = activeTees.erase(it);
            } else {
              ++it;
            }
          }
        };

        std::function<void(Expression*)> walkDom = [&](Expression* curr) {
          if (!curr) {
            return;
          }
          if (auto* block = curr->dynCast<Block>()) {
            Index firstBranchIdx = block->list.size();
            if (block->name.is()) {
              for (Index i = 0; i < block->list.size(); i++) {
                if (subtreeTargetsLabel(block->list[i], block->name)) {
                  firstBranchIdx = i;
                  break;
                }
              }
            }
            std::unordered_map<Index, Index> savedExitMap;
            bool capturedAfterDirectBr = false;
            for (Index i = 0; i < block->list.size(); i++) {
              if (i == firstBranchIdx) {
                bool isDirectCondBr = false;
                if (auto* br = block->list[i]->dynCast<Break>()) {
                  isDirectCondBr =
                    br->name == block->name && br->condition &&
                    !subtreeTargetsLabel(br->value, block->name) &&
                    !subtreeTargetsLabel(br->condition, block->name);
                } else if (auto* bron = block->list[i]->dynCast<BrOn>()) {
                  isDirectCondBr =
                    bron->name == block->name &&
                    !subtreeTargetsLabel(bron->ref, block->name) &&
                    !subtreeTargetsLabel(bron->desc, block->name);
                }
                if (!isDirectCondBr) {
                  savedExitMap = activeTees;
                } else {
                  capturedAfterDirectBr = true;
                }
              }
              walkDom(block->list[i]);
              if (i == firstBranchIdx && capturedAfterDirectBr) {
                savedExitMap = activeTees;
              }
            }
            if (block->type == Type::unreachable) {
              activeTees.clear();
            } else if (firstBranchIdx < block->list.size()) {
              activeTees = std::move(savedExitMap);
              for (Index i = firstBranchIdx; i < block->list.size(); i++) {
                if (i == firstBranchIdx && capturedAfterDirectBr) {
                  continue;
                }
                EffectAnalyzer eff(options, *wasm, block->list[i]);
                eff.trap = false;
                invalidateActiveByEffects(eff, block->list[i]);
              }
            }
            return;
          }
          if (auto* iff = curr->dynCast<If>()) {
            walkDom(iff->condition);
            auto afterCond = activeTees;
            walkDom(iff->ifTrue);
            if (iff->ifFalse) {
              activeTees = afterCond;
              walkDom(iff->ifFalse);
            }
            if (iff->type == Type::unreachable) {
              activeTees.clear();
            } else {
              activeTees = std::move(afterCond);
              bool thenExitsFunc =
                iff->ifTrue->type == Type::unreachable &&
                FindAll<Break>(iff->ifTrue).list.empty() &&
                FindAll<Switch>(iff->ifTrue).list.empty() &&
                FindAll<BrOn>(iff->ifTrue).list.empty();
              if (!thenExitsFunc) {
                EffectAnalyzer thenEff(options, *wasm, iff->ifTrue);
                thenEff.trap = false;
                invalidateActiveByEffects(thenEff, iff->ifTrue);
              }
              if (iff->ifFalse) {
                bool elseExitsFunc =
                  iff->ifFalse->type == Type::unreachable &&
                  FindAll<Break>(iff->ifFalse).list.empty() &&
                  FindAll<Switch>(iff->ifFalse).list.empty() &&
                  FindAll<BrOn>(iff->ifFalse).list.empty();
                if (!elseExitsFunc) {
                  EffectAnalyzer elseEff(options, *wasm, iff->ifFalse);
                  elseEff.trap = false;
                  invalidateActiveByEffects(elseEff, iff->ifFalse);
                }
              }
            }
            return;
          }
          if (auto* loop = curr->dynCast<Loop>()) {
            EffectAnalyzer loopEff(options, *wasm, loop->body);
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
            return;
          }
          if (auto* tryy = curr->dynCast<Try>()) {
            auto beforeTry = activeTees;
            walkDom(tryy->body);
            EffectAnalyzer bodyEff(options, *wasm, tryy->body);
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
              EffectAnalyzer tryEff(options, *wasm, tryy);
              tryEff.trap = false;
              invalidateActiveByEffects(tryEff, tryy);
            }
            return;
          }
          if (auto* tt = curr->dynCast<TryTable>()) {
            auto beforeTry = activeTees;
            walkDom(tt->body);
            if (tt->type == Type::unreachable) {
              activeTees.clear();
            } else {
              activeTees = std::move(beforeTry);
              EffectAnalyzer ttEff(options, *wasm, tt);
              ttEff.trap = false;
              invalidateActiveByEffects(ttEff, tt);
            }
            return;
          }
          if (auto* br = curr->dynCast<Break>()) {
            if (br->value) {
              walkDom(br->value);
            }
            if (br->condition) {
              walkDom(br->condition);
            }
            if (!br->condition) {
              activeTees.clear();
            }
            return;
          }
          if (auto* bron = curr->dynCast<BrOn>()) {
            walkDom(bron->ref);
            if (bron->desc) {
              walkDom(bron->desc);
            }
            return;
          }
          if (auto* sw = curr->dynCast<Switch>()) {
            if (sw->value) {
              walkDom(sw->value);
            }
            walkDom(sw->condition);
            activeTees.clear();
            return;
          }
          if (auto* ret = curr->dynCast<Return>()) {
            if (ret->value) {
              walkDom(ret->value);
            }
            activeTees.clear();
            return;
          }
          if (auto* th = curr->dynCast<Throw>()) {
            for (auto* op : th->operands) {
              walkDom(op);
            }
            activeTees.clear();
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
                                    (origV->is<Const>() ||
                                     origV->is<RefNull>() ||
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

          bool idempotentCall = isIdempotentCallExpr(curr);
          ShallowEffectAnalyzer shallow(options, *wasm, curr);
          shallow.trap = false;
          if (idempotentCall) {
            if (auto* target =
                  wasm->getFunctionOrNull(curr->cast<Call>()->target)) {
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
          if (shallow.branchesOut || curr->type == Type::unreachable) {
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

        auto makeNormCopy = [&](Expression* V,
                                LocalSet* innerTee) -> Expression* {
          if (!innerTee) {
            return V;
          }
          Expression* copyV = ExpressionManipulator::copy(V, *wasm);
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
          } converter(*wasm, func, innerTee->index);
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
            Type binType = getBinaryVarType(xType);
            Index headerBytes = 0;
            if (isSingleWriteVar && varTypeCounts[binType] == 1) {
              headerBytes =
                (xType.isRef() && !xType.getHeapType().isBasic()) ? 4 : 2;
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
              Index rhs = (innerTee ? 0 : 2) + 2 * k + headerBytes;
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
            Type binType = getBinaryVarType(func->getLocalType(x));
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
          const std::unordered_map<LocalSet*, LocalSet*>&
            selectedWriteInnerTee;
          const std::unordered_map<LocalGet*, Expression*>& selectedGetNodes;

          UnCseReplacer(
            Module& wasm,
            const std::unordered_map<LocalSet*, Expression*>&
              selectedWriteNodes,
            const std::unordered_map<LocalSet*, LocalSet*>&
              selectedWriteInnerTee,
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
          *wasm, selectedWriteNodes, selectedWriteInnerTee, selectedGetNodes);
        replacer.walk(func->body);
        anyChanged = true;
      }

      if (anyChanged) {
        if (!allUnCsedLocals.empty()) {
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
          if (newVars.size() != func->vars.size()) {
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
              std::unordered_map<Index, Name> newLocalNames;
              std::unordered_map<Name, Index> newLocalIndices;
              for (Index newIdx = 0; newIdx < newToOld.size(); newIdx++) {
                Index oldIdx = newToOld[newIdx];
                auto it = func->localNames.find(oldIdx);
                if (it != func->localNames.end()) {
                  newLocalNames[newIdx] = it->second;
                  newLocalIndices[it->second] = newIdx;
                }
              }
              func->localNames = std::move(newLocalNames);
              func->localIndices = std::move(newLocalIndices);
            }
          }
        }
        ReFinalize().walkFunctionInModule(func, wasm);
        refreshFuncEffects(func);
      }
    }
  }

  bool isNonMutatingWrapper(Function* wFunc, int depth = 0) {
    if (!wFunc || wFunc->imported() || depth > 4) {
      return false;
    }
    if (wFunc->effects && wFunc->effects->isIdempotent &&
        wFunc->effects->onlyTouchesInitGlobals && !wFunc->effects->throws()) {
      return true;
    }
    EffectAnalyzer wEff(getPassOptions(), *wasm, wFunc);
    wEff.localsRead.clear();
    wEff.localsWritten.clear();
    if (!wEff.hasSideEffects() && !wEff.throws()) {
      return true;
    }
    Expression* body = unwrapRefAsNonNull(wFunc->body);
    if (!body) {
      return false;
    }
    // Pattern 1: Lazy cached constant-pool/factory wrapper starting with
    // (block $l (br_on_non_null $l (array.get/global.get ...)) ...)
    // (optionally preceded by lazy-init If guards or wrapped in Drop).
    if (auto* blk = body->dynCast<Block>()) {
      Index firstIdx = 0;
      while (firstIdx < blk->list.size()) {
        if (auto* iff = blk->list[firstIdx]->dynCast<If>()) {
          if (!iff->ifFalse) {
            firstIdx++;
            continue;
          }
        }
        break;
      }
      if (firstIdx < blk->list.size()) {
        Expression* firstStmt = unwrapRefAsNonNull(blk->list[firstIdx]);
        if (auto* innerBlk = firstStmt->dynCast<Block>()) {
          if (!innerBlk->list.empty()) {
            firstStmt = innerBlk->list[0];
          }
        }
        if (auto* drop = firstStmt->dynCast<Drop>()) {
          firstStmt = drop->value;
        }
        if (auto* brOn = firstStmt->dynCast<BrOn>()) {
          if (brOn->op == BrOnNonNull) {
            return true;
          }
        }
      }
      // Pattern 2: Idempotent once-init getter:
      // (block (if (i32.eqz/ref.is_null (global.get $init)) ...)
      //        (global.get $val)).
      if (blk->list.size() == 2 &&
          unwrapRefAsNonNull(blk->list[1])->is<GlobalGet>()) {
        if (auto* iff = blk->list[0]->dynCast<If>()) {
          if (!iff->ifFalse) {
            if (auto* un = iff->condition->dynCast<Unary>()) {
              if (un->op == EqZInt32 && un->value->is<GlobalGet>()) {
                return true;
              }
            }
            if (auto* rin = iff->condition->dynCast<RefIsNull>()) {
              if (rin->value->is<GlobalGet>()) {
                return true;
              }
            }
          }
        }
      }
    }
    // Pattern 3: Pure allocation / read tree whose only mutations (if any) are
    // initializing a freshly allocated struct/array in a local tee, and whose
    // calls (if any) are to non-mutating wrappers.
    if (!FindAll<Throw>(body).list.empty() ||
        !FindAll<Rethrow>(body).list.empty() ||
        !FindAll<GlobalSet>(body).list.empty() ||
        !FindAll<Store>(body).list.empty() ||
        !FindAll<CallRef>(body).list.empty() ||
        !FindAll<CallIndirect>(body).list.empty()) {
      return false;
    }
    for (auto* ss : FindAll<StructSet>(body).list) {
      auto* tee = ss->ref->dynCast<LocalSet>();
      if (!tee || !tee->isTee() || !tee->value->is<StructNew>()) {
        return false;
      }
    }
    for (auto* as : FindAll<ArraySet>(body).list) {
      auto* tee = as->ref->dynCast<LocalSet>();
      if (!tee || !tee->isTee() ||
          (!tee->value->is<ArrayNew>() && !tee->value->is<ArrayNewFixed>())) {
        return false;
      }
    }
    for (auto* c : FindAll<Call>(body).list) {
      if (c->isReturn || c->target == wFunc->name ||
          !isNonMutatingWrapper(wasm->getFunctionOrNull(c->target),
                                depth + 1)) {
        return false;
      }
    }
    return true;
  }

  bool canExprThrow(Expression* expr, const EffectAnalyzer& eff) {
    if (!eff.throws()) {
      return false;
    }
    if (eff.suspends || !FindAll<Throw>(expr).list.empty() ||
        !FindAll<Rethrow>(expr).list.empty() ||
        !FindAll<CallRef>(expr).list.empty() ||
        !FindAll<CallIndirect>(expr).list.empty()) {
      return true;
    }
    for (auto* c : FindAll<Call>(expr).list) {
      Function* f = wasm->getFunctionOrNull(c->target);
      if (!f || !isNonMutatingWrapper(f)) {
        return true;
      }
    }
    return false;
  }

  static bool hasReturnOrTailCall(Expression* body) {
    if (!body || !FindAll<Return>(body).list.empty()) {
      return true;
    }
    for (auto* c : FindAll<Call>(body).list) {
      if (c->isReturn) {
        return true;
      }
    }
    for (auto* c : FindAll<CallIndirect>(body).list) {
      if (c->isReturn) {
        return true;
      }
    }
    for (auto* c : FindAll<CallRef>(body).list) {
      if (c->isReturn) {
        return true;
      }
    }
    return false;
  }

  bool canMoveWrapperPast(Function* wFunc, Expression* laterOp) {
    EffectAnalyzer opEff(getPassOptions(), *wasm, laterOp);
    // wFunc is a separate function and cannot access the caller's locals.
    opEff.localsRead.clear();
    opEff.localsWritten.clear();
    if (!opEff.hasAnything()) {
      return true;
    }
    if (!wFunc || wFunc->imported()) {
      return false;
    }
    if (isNonMutatingWrapper(wFunc)) {
      return !opEff.branchesOut && !opEff.hasExternalBreakTargets();
    }
    EffectAnalyzer wEff(getPassOptions(), *wasm, wFunc);
    wEff.localsRead.clear();
    wEff.localsWritten.clear();
    return !wEff.invalidates(opEff);
  }

  bool isSingleEntryReadSafe(Function* callee,
                             Index paramIdx,
                             Function* wFunc,
                             LocalGet*& outGet) {
    outGet = nullptr;
    if (!wFunc || wFunc->imported()) {
      return false;
    }
    TreeLayout layout;
    layout.build(callee->body);
    auto allGets = FindAll<LocalGet>(callee->body).list;
    LocalGet* targetGet = nullptr;
    for (auto* lg : allGets) {
      if (lg->index == paramIdx) {
        if (targetGet) {
          return false;
        }
        targetGet = lg;
      }
    }
    if (!targetGet || !layout.isUnconditionalFrom(targetGet, callee->body)) {
      return false;
    }
    // Check that every expression evaluated before targetGet on the path from
    // callee->body to targetGet can be moved past wFunc.
    Expression* curr = targetGet;
    while (curr && curr != callee->body) {
      Expression* par = layout.parent.at(curr);
      if (!par) {
        return false;
      }
      for (auto* sib : ChildIterator(par)) {
        if (sib == curr) {
          break;
        }
        if (sib && !canMoveWrapperPast(wFunc, sib)) {
          return false;
        }
      }
      curr = par;
    }
    outGet = targetGet;
    return true;
  }

  bool sinkWrapperCalls() {
    std::unordered_set<Name> disallowedTargets;
    for (auto& ex : wasm->exports) {
      if (ex->kind == ExternalKind::Function) {
        disallowedTargets.insert(*ex->getInternalName());
      }
    }
    for (auto& seg : wasm->elementSegments) {
      for (auto* expr : seg->data) {
        for (auto* rf : FindAll<RefFunc>(expr).list) {
          disallowedTargets.insert(rf->func);
        }
      }
    }
    for (auto& g : wasm->globals) {
      if (!g->imported() && g->init) {
        for (auto* rf : FindAll<RefFunc>(g->init).list) {
          disallowedTargets.insert(rf->func);
        }
      }
    }

    struct CallSiteInfo {
      Function* caller;
      Call* call;
    };
    std::unordered_map<Name, std::vector<CallSiteInfo>> callsByTarget;
    std::unordered_map<Function*, Index> importIndex;
    Index nextImportIdx = 0;
    ModuleUtils::iterImportedFunctions(
      *wasm, [&](Function* f) { importIndex[f] = nextImportIdx++; });
    auto getFuncCallBytes = [&](Function* f) -> unsigned {
      if (f && f->imported()) {
        auto it = importIndex.find(f);
        if (it != importIndex.end() && it->second < 128) {
          return 2;
        }
      }
      return 3;
    };

    for (auto& f : wasm->functions) {
      if (f->imported()) {
        continue;
      }
      Function* caller = f.get();
      for (auto* rf : FindAll<RefFunc>(caller->body).list) {
        disallowedTargets.insert(rf->func);
      }
      for (auto* c : FindAll<Call>(caller->body).list) {
        callsByTarget[c->target].push_back({caller, c});
      }
    }

    Builder builder(*wasm);
    std::unordered_set<Function*> modifiedFuncs;
    std::unordered_set<Function*> modifiedCallees;
    std::unordered_set<Function*> calleesWithStaleCallSites;
    std::unordered_set<Expression*> touchedCalls;
    bool anyChanged = false;

    // Part A: In-place wrapper sinking when 100% of callers of a non-escaped
    // function pass (call $w X_0 ... X_{r-1}) at parameter k.
    for (auto& f : wasm->functions) {
      if (f->imported()) {
        continue;
      }
      Function* callee = f.get();
      if (disallowedTargets.count(callee->name) ||
          modifiedCallees.count(callee) ||
          calleesWithStaleCallSites.count(callee) ||
          callee->getNumParams() == 0) {
        continue;
      }
      auto it = callsByTarget.find(callee->name);
      if (it == callsByTarget.end() || it->second.size() < 2) {
        continue;
      }
      auto& sites = it->second;
      Index N = sites.size();
      bool validSites = true;
      for (const auto& s : sites) {
        if (s.caller == callee || touchedCalls.count(s.call) ||
            s.call->operands.size() != callee->getNumParams()) {
          validSites = false;
          break;
        }
        for (auto* op : s.call->operands) {
          if (op->type == Type::unreachable) {
            validSites = false;
            break;
          }
        }
        if (!validSites) {
          break;
        }
      }
      if (!validSites) {
        continue;
      }

      std::vector<Index> localReads(callee->getNumLocals(), 0);
      std::vector<Index> localWrites(callee->getNumLocals(), 0);
      for (auto* lg : FindAll<LocalGet>(callee->body).list) {
        localReads[lg->index]++;
      }
      for (auto* ls : FindAll<LocalSet>(callee->body).list) {
        localWrites[ls->index]++;
      }

      Index m = callee->getNumParams();
      std::vector<std::vector<Type>> newParamTypes(m);
      for (Index k = 0; k < m; k++) {
        newParamTypes[k] = {callee->getLocalType(k)};
      }
      std::vector<bool> paramSunk(m, false);
      std::vector<Index> reuseSubIdx(m, 0);
      std::vector<std::vector<LocalGet*>> expandedCallGets(m);
      std::vector<LocalSet*> expandedReuseSet(m, nullptr);
      bool calleeModified = false;

      for (Index revK = 0; revK < m; revK++) {
        Index k = m - 1 - revK;
        if (localReads[k] == 0) {
          continue;
        }
        auto* firstW = sites[0].call->operands[k]->dynCast<Call>();
        if (!firstW || firstW->isReturn || firstW->target == callee->name) {
          continue;
        }
        Index wArity = firstW->operands.size();
        Name wName = firstW->target;
        Function* wFunc = wasm->getFunctionOrNull(wName);
        if (!wFunc || modifiedCallees.count(wFunc)) {
          continue;
        }
        Type oldParamType = callee->getLocalType(k);
        Type wResultType = wFunc->getResults();
        if (wFunc->getNumParams() != wArity || !wResultType.isSingle() ||
            !Type::isSubType(wResultType, oldParamType)) {
          continue;
        }

        bool allMatch = true;
        for (const auto& s : sites) {
          auto* wCall = s.call->operands[k]->dynCast<Call>();
          if (!wCall || touchedCalls.count(wCall) || wCall->isReturn ||
              wCall->target != wName || wCall->operands.size() != wArity) {
            allMatch = false;
            break;
          }
          for (Index wi = 0; wi < wArity; wi++) {
            if (wCall->operands[wi]->type == Type::unreachable ||
                !Type::isSubType(wCall->operands[wi]->type,
                                 wFunc->getLocalType(wi))) {
              allMatch = false;
              break;
            }
          }
          if (!allMatch) {
            break;
          }
          for (Index j = k + 1; j < m; j++) {
            if (!paramSunk[j]) {
              if (!canMoveWrapperPast(wFunc, s.call->operands[j])) {
                allMatch = false;
                break;
              }
            } else {
              auto* laterW = s.call->operands[j]->cast<Call>();
              for (auto* laterOp : laterW->operands) {
                if (!canMoveWrapperPast(wFunc, laterOp)) {
                  allMatch = false;
                  break;
                }
              }
              if (!allMatch) {
                break;
              }
            }
          }
          if (!allMatch) {
            break;
          }
        }
        if (!allMatch) {
          continue;
        }

        if (wArity == 0) {
          // 0-arg wrapper call: either substitute (call $w) directly at all
          // reads of param k (when pure or single-entry read) or compute once
          // into a fresh local at callee entry, then remove param k.
          LocalGet* singleGet = nullptr;
          bool canSubDirectly =
            (localWrites[k] == 0 &&
             ((localReads[k] == 1 &&
               isSingleEntryReadSafe(callee, k, wFunc, singleGet)) ||
              isNonMutatingWrapper(wFunc)));
          unsigned callWSize = getFuncCallBytes(wFunc);
          unsigned costBytes = 0;
          bool useDirectSub = false;
          if (canSubDirectly &&
              localReads[k] * (callWSize - 2) <= callWSize + 4) {
            useDirectSub = true;
            costBytes = localReads[k] * (callWSize - 2);
          } else if (oldParamType.isDefaultable()) {
            useDirectSub = false;
            costBytes = callWSize + 4;
          } else if (canSubDirectly) {
            useDirectSub = true;
            costBytes = localReads[k] * (callWSize - 2);
          } else {
            continue;
          }
          if (callWSize * N <= costBytes) {
            continue;
          }

          if (useDirectSub) {
            struct ParamGetReplacer : public PostWalker<ParamGetReplacer> {
              Index targetIdx;
              Name wName;
              Type wResultType;
              Builder& builder;
              ParamGetReplacer(Index targetIdx,
                               Name wName,
                               Type wResultType,
                               Builder& builder)
                : targetIdx(targetIdx), wName(wName), wResultType(wResultType),
                  builder(builder) {}
              void visitLocalGet(LocalGet* curr) {
                if (curr->index == targetIdx) {
                  replaceCurrent(builder.makeCall(wName, {}, wResultType));
                }
              }
            };
            ParamGetReplacer(k, wName, wResultType, builder).walk(callee->body);
          } else {
            Index tmpLocal = Builder::addVar(callee, oldParamType);
            for (auto* lg : FindAll<LocalGet>(callee->body).list) {
              if (lg->index == k) {
                lg->index = tmpLocal;
              }
            }
            for (auto* ls : FindAll<LocalSet>(callee->body).list) {
              if (ls->index == k) {
                ls->index = tmpLocal;
              }
            }
            Expression* setExpr = builder.makeLocalSet(
              tmpLocal, builder.makeCall(wName, {}, wResultType));
            callee->body = builder.makeSequence(setExpr, callee->body);
          }

          newParamTypes[k].clear();
          paramSunk[k] = true;
          for (const auto& s : sites) {
            auto* wCall = s.call->operands[k]->cast<Call>();
            touchedCalls.insert(wCall);
            modifiedFuncs.insert(s.caller);
          }
          modifiedCallees.insert(callee);
          calleesWithStaleCallSites.insert(wFunc);
          calleeModified = true;
          anyChanged = true;
          continue;
        }

        LocalGet* singleGet = nullptr;
        bool directSub = localWrites[k] == 0 && localReads[k] == 1 &&
                         isSingleEntryReadSafe(callee, k, wFunc, singleGet);
        unsigned callWOverhead = getFuncCallBytes(wFunc);
        unsigned costBytes = callWOverhead + 2 * unsigned(wArity - 1);
        Index reuseSub = Index(-1);
        for (Index wi = 0; wi < wArity; wi++) {
          if (wFunc->getLocalType(wi) == oldParamType) {
            reuseSub = wi;
            break;
          }
        }
        if (!directSub) {
          if (reuseSub != Index(-1)) {
            costBytes = callWOverhead + 2 * unsigned(wArity) + 2;
          } else if (oldParamType.isDefaultable()) {
            costBytes = callWOverhead + 2 * unsigned(wArity) + 4;
          } else {
            continue;
          }
        }
        if (callWOverhead * N <= costBytes) {
          continue;
        }
        // If wFunc is also exclusive to (callee, k) and absorbing callee into
        // wFunc in Part A2 is strictly cheaper, defer to Part A2.
        if ((wArity >= 2 || !directSub) && !calleeModified &&
            !wFunc->imported() && !disallowedTargets.count(wName) &&
            !calleesWithStaleCallSites.count(wFunc) &&
            callsByTarget[wName].size() == N &&
            wFunc->body->type != Type::unreachable &&
            !hasReturnOrTailCall(wFunc->body)) {
          unsigned cat1Cost = getFuncCallBytes(callee) + 2 * unsigned(m - 1);
          if (cat1Cost < costBytes) {
            continue;
          }
        }

        std::vector<Type> wInputTypes(wArity);
        std::vector<Expression*> wCallArgs(wArity);
        expandedCallGets[k].resize(wArity);
        for (Index wi = 0; wi < wArity; wi++) {
          wInputTypes[wi] = wFunc->getLocalType(wi);
          LocalGet* lg = builder.makeLocalGet(k, wInputTypes[wi]);
          expandedCallGets[k][wi] = lg;
          wCallArgs[wi] = lg;
        }
        Expression* wCallExpr = builder.makeCall(wName, wCallArgs, wResultType);

        if (directSub) {
          struct SingleGetReplacer : public PostWalker<SingleGetReplacer> {
            LocalGet* target;
            Expression* rep;
            SingleGetReplacer(LocalGet* target, Expression* rep)
              : target(target), rep(rep) {}
            void visitLocalGet(LocalGet* curr) {
              if (curr == target) {
                replaceCurrent(rep);
              }
            }
          };
          SingleGetReplacer(singleGet, wCallExpr).walk(callee->body);
        } else if (reuseSub != Index(-1)) {
          reuseSubIdx[k] = reuseSub;
          LocalSet* setExpr = builder.makeLocalSet(k, wCallExpr);
          expandedReuseSet[k] = setExpr;
          callee->body = builder.makeSequence(setExpr, callee->body);
        } else {
          Index tmpLocal = Builder::addVar(callee, oldParamType);
          for (auto* lg : FindAll<LocalGet>(callee->body).list) {
            if (lg->index == k) {
              lg->index = tmpLocal;
            }
          }
          for (auto* ls : FindAll<LocalSet>(callee->body).list) {
            if (ls->index == k) {
              ls->index = tmpLocal;
            }
          }
          Expression* setExpr = builder.makeLocalSet(tmpLocal, wCallExpr);
          callee->body = builder.makeSequence(setExpr, callee->body);
        }

        newParamTypes[k] = std::move(wInputTypes);
        paramSunk[k] = true;
        for (const auto& s : sites) {
          auto* wCall = s.call->operands[k]->cast<Call>();
          touchedCalls.insert(wCall);
          modifiedFuncs.insert(s.caller);
        }
        modifiedCallees.insert(callee);
        calleesWithStaleCallSites.insert(wFunc);
        calleeModified = true;
        anyChanged = true;
      }

      if (calleeModified) {
        Index totalLocals = callee->getNumLocals();
        std::vector<Index> indexMap(totalLocals, 0);
        std::vector<Index> newParamStart(m, 0);
        std::vector<Type> keptParams;
        Index nextIdx = 0;
        bool paramCountChanged = false;
        for (Index k = 0; k < m; k++) {
          newParamStart[k] = nextIdx;
          if (newParamTypes[k].size() != 1) {
            paramCountChanged = true;
          }
          if (!newParamTypes[k].empty()) {
            indexMap[k] = nextIdx + reuseSubIdx[k];
            for (Type pt : newParamTypes[k]) {
              keptParams.push_back(pt);
              nextIdx++;
            }
          }
        }
        for (Index v = m; v < totalLocals; v++) {
          indexMap[v] = nextIdx++;
        }
        for (auto* lg : FindAll<LocalGet>(callee->body).list) {
          lg->index = indexMap[lg->index];
          if (lg->index < keptParams.size()) {
            lg->type = keptParams[lg->index];
          }
        }
        for (auto* ls : FindAll<LocalSet>(callee->body).list) {
          ls->index = indexMap[ls->index];
        }
        for (Index k = 0; k < m; k++) {
          for (Index wi = 0; wi < expandedCallGets[k].size(); wi++) {
            Index pIdx = newParamStart[k] + wi;
            expandedCallGets[k][wi]->index = pIdx;
            expandedCallGets[k][wi]->type = keptParams[pIdx];
          }
          if (expandedReuseSet[k]) {
            expandedReuseSet[k]->index = newParamStart[k] + reuseSubIdx[k];
          }
        }
        if (paramCountChanged && !callee->localNames.empty()) {
          std::unordered_map<Index, Name> newLocalNames;
          std::unordered_map<Name, Index> newLocalIndices;
          for (auto& [oldIdx, name] : callee->localNames) {
            if (oldIdx < totalLocals &&
                (oldIdx >= m || !newParamTypes[oldIdx].empty())) {
              Index newIdx = indexMap[oldIdx];
              newLocalNames[newIdx] = name;
              newLocalIndices[name] = newIdx;
            }
          }
          callee->localNames = std::move(newLocalNames);
          callee->localIndices = std::move(newLocalIndices);
        }
        callee->setParams(Type(keptParams));
        for (const auto& s : sites) {
          ExpressionList newOps(wasm->allocator);
          for (Index k = 0; k < m; k++) {
            if (!paramSunk[k]) {
              newOps.push_back(s.call->operands[k]);
            } else {
              auto* wCall = s.call->operands[k]->cast<Call>();
              for (auto* wOp : wCall->operands) {
                newOps.push_back(wOp);
              }
            }
          }
          s.call->operands = std::move(newOps);
          touchedCalls.insert(s.call);
        }
        modifiedFuncs.insert(callee);
      }
    }

    // Part A2: Absorb outer callee into exclusive inner wFunc in-place when
    // 100% of direct calls (N >= 2) to a non-escaped defined function wFunc
    // occur as operand k of (call $callee a_0 ... (call $wFunc b_0 ...) ...).
    {
      struct ParentCallInfo {
        Function* caller;
        Call* outerCall;
        Function* callee;
        Index k;
      };
      std::unordered_map<Call*, ParentCallInfo> parentCallOf;
      for (auto& f : wasm->functions) {
        if (f->imported()) {
          continue;
        }
        Function* caller = f.get();
        for (auto* c : FindAll<Call>(caller->body).list) {
          if (c->isReturn || touchedCalls.count(c)) {
            continue;
          }
          Function* callee = wasm->getFunctionOrNull(c->target);
          if (!callee || modifiedCallees.count(callee) ||
              c->operands.size() != callee->getNumParams()) {
            continue;
          }
          bool hasUnreachable = false;
          for (auto* op : c->operands) {
            if (op->type == Type::unreachable) {
              hasUnreachable = true;
              break;
            }
          }
          if (hasUnreachable) {
            continue;
          }
          for (Index k = 0; k < c->operands.size(); k++) {
            if (auto* wCall = c->operands[k]->dynCast<Call>()) {
              parentCallOf[wCall] = {caller, c, callee, k};
            }
          }
        }
      }

      struct AbsorbCandidate {
        Function* wFunc;
        Function* callee;
        Index k;
        Index numSites;
        int netSaved;
      };
      std::vector<AbsorbCandidate> absorbCandidates;

      for (auto& f : wasm->functions) {
        if (f->imported()) {
          continue;
        }
        Function* wFunc = f.get();
        if (disallowedTargets.count(wFunc->name) ||
            modifiedCallees.count(wFunc) ||
            calleesWithStaleCallSites.count(wFunc) ||
            !wFunc->getResults().isSingle() ||
            wFunc->body->type == Type::unreachable) {
          continue;
        }
        auto it = callsByTarget.find(wFunc->name);
        if (it == callsByTarget.end() || it->second.size() < 2) {
          continue;
        }
        const auto& sites = it->second;
        Index N = sites.size();
        Index r = wFunc->getNumParams();

        Function* commonCallee = nullptr;
        Index commonK = 0;
        bool valid = true;
        for (const auto& s : sites) {
          Call* wCall = s.call;
          if (wCall->isReturn || touchedCalls.count(wCall) ||
              wCall->operands.size() != r) {
            valid = false;
            break;
          }
          for (Index wi = 0; wi < r; wi++) {
            if (wCall->operands[wi]->type == Type::unreachable ||
                !Type::isSubType(wCall->operands[wi]->type,
                                 wFunc->getLocalType(wi))) {
              valid = false;
              break;
            }
          }
          if (!valid) {
            break;
          }
          auto pIt = parentCallOf.find(wCall);
          if (pIt == parentCallOf.end()) {
            valid = false;
            break;
          }
          const auto& pInfo = pIt->second;
          if (pInfo.caller == wFunc || pInfo.caller == pInfo.callee ||
              pInfo.callee == wFunc || touchedCalls.count(pInfo.outerCall) ||
              modifiedCallees.count(pInfo.callee)) {
            valid = false;
            break;
          }
          if (!commonCallee) {
            commonCallee = pInfo.callee;
            commonK = pInfo.k;
            if (!Type::isSubType(wFunc->getResults(),
                                 commonCallee->getLocalType(commonK))) {
              valid = false;
              break;
            }
          } else if (pInfo.callee != commonCallee || pInfo.k != commonK) {
            valid = false;
            break;
          }
          Index m = commonCallee->getNumParams();
          for (Index j = 0; j < m; j++) {
            if (j != commonK &&
                !Type::isSubType(pInfo.outerCall->operands[j]->type,
                                 commonCallee->getLocalType(j))) {
              valid = false;
              break;
            }
          }
          if (!valid) {
            break;
          }
          for (Index j = commonK + 1; j < m; j++) {
            if (!canMoveWrapperPast(wFunc, pInfo.outerCall->operands[j])) {
              valid = false;
              break;
            }
          }
          if (!valid) {
            break;
          }
        }
        if (!valid || !commonCallee) {
          continue;
        }
        Index m = commonCallee->getNumParams();
        unsigned outerCallBytes = getFuncCallBytes(commonCallee);
        unsigned savedBytes = outerCallBytes * unsigned(N);
        unsigned costBytes = outerCallBytes + 2 * unsigned(m - 1);
        if (savedBytes <= costBytes) {
          continue;
        }
        if (hasReturnOrTailCall(wFunc->body)) {
          continue;
        }
        absorbCandidates.push_back(
          {wFunc, commonCallee, commonK, N, int(savedBytes) - int(costBytes)});
      }

      std::sort(absorbCandidates.begin(),
                absorbCandidates.end(),
                [](const AbsorbCandidate& a, const AbsorbCandidate& b) {
                  if (a.netSaved != b.netSaved) {
                    return a.netSaved > b.netSaved;
                  }
                  if (a.numSites != b.numSites) {
                    return a.numSites > b.numSites;
                  }
                  return a.wFunc->name < b.wFunc->name;
                });

      for (const auto& cand : absorbCandidates) {
        Function* wFunc = cand.wFunc;
        Function* callee = cand.callee;
        Index k = cand.k;
        if (modifiedCallees.count(wFunc) ||
            calleesWithStaleCallSites.count(wFunc) ||
            modifiedCallees.count(callee)) {
          continue;
        }
        const auto& sites = callsByTarget[wFunc->name];
        bool stillValid = true;
        for (const auto& s : sites) {
          auto pIt = parentCallOf.find(s.call);
          if (pIt == parentCallOf.end() || touchedCalls.count(s.call) ||
              touchedCalls.count(pIt->second.outerCall)) {
            stillValid = false;
            break;
          }
        }
        if (!stillValid) {
          continue;
        }

        Index r = wFunc->getNumParams();
        Index m = callee->getNumParams();
        std::vector<Type> newWParams;
        newWParams.reserve(m + r - 1);
        for (Index j = 0; j < k; j++) {
          newWParams.push_back(callee->getLocalType(j));
        }
        for (Index wi = 0; wi < r; wi++) {
          newWParams.push_back(wFunc->getLocalType(wi));
        }
        for (Index j = k + 1; j < m; j++) {
          newWParams.push_back(callee->getLocalType(j));
        }

        if (m > 1) {
          for (auto* lg : FindAll<LocalGet>(wFunc->body).list) {
            lg->index =
              (lg->index < r) ? (lg->index + k) : (lg->index + (m - 1));
          }
          for (auto* ls : FindAll<LocalSet>(wFunc->body).list) {
            ls->index =
              (ls->index < r) ? (ls->index + k) : (ls->index + (m - 1));
          }
          if (!wFunc->localNames.empty()) {
            std::unordered_map<Index, Name> newLocalNames;
            std::unordered_map<Name, Index> newLocalIndices;
            for (auto& [oldIdx, name] : wFunc->localNames) {
              Index newIdx = (oldIdx < r) ? (oldIdx + k) : (oldIdx + (m - 1));
              newLocalNames[newIdx] = name;
              newLocalIndices[name] = newIdx;
            }
            wFunc->localNames = std::move(newLocalNames);
            wFunc->localIndices = std::move(newLocalIndices);
          }
        }

        std::vector<Expression*> callArgs(m);
        for (Index j = 0; j < m; j++) {
          if (j < k) {
            callArgs[j] = builder.makeLocalGet(j, newWParams[j]);
          } else if (j == k) {
            callArgs[j] = wFunc->body;
          } else {
            Index pIdx = r + j - 1;
            callArgs[j] = builder.makeLocalGet(pIdx, newWParams[pIdx]);
          }
        }
        wFunc->setParams(Type(newWParams));
        wFunc->setResults(callee->getResults());
        wFunc->body =
          builder.makeCall(callee->name, callArgs, callee->getResults(), false);
        if (wFunc->effects && !isNonMutatingWrapper(callee)) {
          wFunc->effects =
            std::make_unique<EffectAnalyzer>(getPassOptions(), *wasm, wFunc);
        }

        for (const auto& s : sites) {
          Call* wCall = s.call;
          const auto& pInfo = parentCallOf[wCall];
          Call* outerCall = pInfo.outerCall;
          ExpressionList newOps(wasm->allocator);
          for (Index j = 0; j < m; j++) {
            if (j == k) {
              for (auto* wOp : wCall->operands) {
                newOps.push_back(wOp);
              }
            } else {
              newOps.push_back(outerCall->operands[j]);
            }
          }
          outerCall->target = wFunc->name;
          outerCall->operands = std::move(newOps);
          touchedCalls.insert(outerCall);
          touchedCalls.insert(wCall);
          modifiedFuncs.insert(pInfo.caller);
        }
        modifiedCallees.insert(wFunc);
        calleesWithStaleCallSites.insert(callee);
        modifiedFuncs.insert(wFunc);
        anyChanged = true;
      }
    }

    // Step 1B: Reuse existing 0-var wrapper-forwarding helpers $H matching
    // (call $callee (local.get 0) ... (call $w (local.get k) ...) ...).
    {
      struct WrappedArg {
        Index targetPos;
        Function* wFunc;
        Index wArity;
      };
      struct ParamSource {
        Index targetPos;
        Function* wFunc; // non-null if wrapped at targetPos
        Index wArgIdx = 0;
      };
      struct WrapperForwarder {
        Function* helper;
        Function* target;
        std::vector<WrappedArg> wrappedArgs;
        std::vector<ParamSource> paramSources;
        unsigned savedBytesPerCall = 0;
      };
      std::unordered_map<Name, std::vector<WrapperForwarder>>
        wrapperHelpersByTarget;
      std::unordered_set<Function*> allWrapperHelpers;

      for (auto& f : wasm->functions) {
        if (f->imported() || f->getNumVars() != 0) {
          continue;
        }
        Function* h = f.get();
        if (modifiedCallees.count(h)) {
          continue;
        }
        auto* innerCall = h->body->dynCast<Call>();
        if (!innerCall || innerCall->isReturn || innerCall->target == h->name) {
          continue;
        }
        Function* target = wasm->getFunctionOrNull(innerCall->target);
        if (!target || modifiedCallees.count(target) ||
            h->getResults() != target->getResults() ||
            innerCall->operands.size() != target->getNumParams()) {
          continue;
        }
        Index m = innerCall->operands.size();
        Index expectedLocal = 0;
        std::vector<WrappedArg> wrappedArgs;
        std::vector<ParamSource> paramSources;
        unsigned totalCallsBytes = getFuncCallBytes(target);
        bool validForwarder = true;
        for (Index j = 0; j < m; j++) {
          Expression* op = innerCall->operands[j];
          if (auto* lg = op->dynCast<LocalGet>()) {
            if (lg->index != expectedLocal) {
              validForwarder = false;
              break;
            }
            expectedLocal++;
            paramSources.push_back({j, nullptr, 0});
          } else if (auto* wCall = op->dynCast<Call>()) {
            Index wArity = wCall->operands.size();
            if (wCall->isReturn || (wArity > 1 && j != m - 1) ||
                wCall->target == h->name || wCall->target == target->name) {
              validForwarder = false;
              break;
            }
            Function* wFunc = wasm->getFunctionOrNull(wCall->target);
            if (!wFunc || wFunc->imported() || modifiedCallees.count(wFunc) ||
                wFunc->getNumParams() != wArity ||
                !wFunc->getResults().isSingle() ||
                !Type::isSubType(wFunc->getResults(),
                                 target->getLocalType(j))) {
              validForwarder = false;
              break;
            }
            for (Index wi = 0; wi < wArity; wi++) {
              auto* lg = wCall->operands[wi]->dynCast<LocalGet>();
              if (!lg || lg->index != expectedLocal ||
                  expectedLocal >= h->getNumParams() ||
                  !Type::isSubType(h->getLocalType(expectedLocal),
                                   wFunc->getLocalType(wi))) {
                validForwarder = false;
                break;
              }
              expectedLocal++;
              paramSources.push_back({j, wFunc, wi});
            }
            if (!validForwarder) {
              break;
            }
            wrappedArgs.push_back({j, wFunc, wArity});
            totalCallsBytes += getFuncCallBytes(wFunc);
          } else {
            validForwarder = false;
            break;
          }
        }
        unsigned hCallBytes = getFuncCallBytes(h);
        if (!validForwarder || expectedLocal != h->getNumParams() ||
            wrappedArgs.empty() || totalCallsBytes <= hCallBytes) {
          continue;
        }
        unsigned savedBytes = totalCallsBytes - hCallBytes;
        wrapperHelpersByTarget[target->name].push_back(
          {h, target, std::move(wrappedArgs), std::move(paramSources), savedBytes});
        allWrapperHelpers.insert(h);
      }

      for (auto& [targetName, helperList] : wrapperHelpersByTarget) {
        auto it = callsByTarget.find(targetName);
        if (it == callsByTarget.end()) {
          continue;
        }
        Function* targetFunc = wasm->getFunction(targetName);
        std::sort(helperList.begin(),
                  helperList.end(),
                  [](const WrapperForwarder& a, const WrapperForwarder& b) {
                    if (a.savedBytesPerCall != b.savedBytesPerCall) {
                      return a.savedBytesPerCall > b.savedBytesPerCall;
                    }
                    return a.helper->name < b.helper->name;
                  });

        for (const auto& s : it->second) {
          if (s.call->isReturn || touchedCalls.count(s.call) ||
              allWrapperHelpers.count(s.caller) ||
              s.call->operands.size() != targetFunc->getNumParams()) {
            continue;
          }
          bool hasUnreachable = false;
          for (auto* op : s.call->operands) {
            if (op->type == Type::unreachable) {
              hasUnreachable = true;
              break;
            }
          }
          if (hasUnreachable) {
            continue;
          }

          Index m = s.call->operands.size();
          for (auto& wh : helperList) {
            if (s.caller == wh.helper) {
              continue;
            }
            bool wrappersMatch = true;
            for (const auto& wa : wh.wrappedArgs) {
              auto* wCall = s.call->operands[wa.targetPos]->dynCast<Call>();
              if (!wCall || wCall->isReturn || touchedCalls.count(wCall) ||
                  wCall->target != wa.wFunc->name ||
                  wCall->operands.size() != wa.wArity) {
                wrappersMatch = false;
                break;
              }
              for (Index wi = 0; wi < wa.wArity; wi++) {
                if (wCall->operands[wi]->type == Type::unreachable ||
                    !Type::isSubType(wCall->operands[wi]->type,
                                     wa.wFunc->getLocalType(wi))) {
                  wrappersMatch = false;
                  break;
                }
              }
              if (!wrappersMatch) {
                break;
              }
              for (Index j = wa.targetPos + 1; j < m; j++) {
                if (!canMoveWrapperPast(wa.wFunc, s.call->operands[j])) {
                  wrappersMatch = false;
                  break;
                }
              }
              if (!wrappersMatch) {
                break;
              }
            }
            if (!wrappersMatch) {
              continue;
            }

            bool typesMatch = true;
            bool needsWiden = false;
            for (Index p = 0; p < wh.paramSources.size(); p++) {
              const auto& src = wh.paramSources[p];
              Expression* newArg =
                src.wFunc
                  ? s.call->operands[src.targetPos]
                      ->cast<Call>()
                      ->operands[src.wArgIdx]
                  : s.call->operands[src.targetPos];
              Type maxAllowed = src.wFunc
                                  ? src.wFunc->getLocalType(src.wArgIdx)
                                  : wh.target->getLocalType(src.targetPos);
              Type argType = newArg->type;
              Type hParamType = wh.helper->getLocalType(p);
              if (!Type::isSubType(argType, hParamType)) {
                if (!disallowedTargets.count(wh.helper->name) &&
                    Type::isSubType(argType, maxAllowed)) {
                  needsWiden = true;
                } else {
                  typesMatch = false;
                  break;
                }
              }
            }
            if (!typesMatch) {
              continue;
            }

            if (needsWiden) {
              std::vector<Type> widenedParams(wh.paramSources.size());
              auto* hCall = wh.helper->body->cast<Call>();
              for (Index p = 0; p < wh.paramSources.size(); p++) {
                const auto& src = wh.paramSources[p];
                Expression* newArg =
                  src.wFunc
                    ? s.call->operands[src.targetPos]
                        ->cast<Call>()
                        ->operands[src.wArgIdx]
                    : s.call->operands[src.targetPos];
                Type lub = Type::getLeastUpperBound(wh.helper->getLocalType(p),
                                                    newArg->type);
                widenedParams[p] = lub;
                if (src.wFunc) {
                  hCall->operands[src.targetPos]
                    ->cast<Call>()
                    ->operands[src.wArgIdx]
                    ->cast<LocalGet>()
                    ->type = lub;
                } else {
                  hCall->operands[src.targetPos]->cast<LocalGet>()->type = lub;
                }
              }
              wh.helper->setParams(Type(widenedParams));
              modifiedFuncs.insert(wh.helper);
            }

            ExpressionList newOps(wasm->allocator);
            for (Index p = 0; p < wh.paramSources.size(); p++) {
              const auto& src = wh.paramSources[p];
              Expression* newArg =
                src.wFunc
                  ? s.call->operands[src.targetPos]
                      ->cast<Call>()
                      ->operands[src.wArgIdx]
                  : s.call->operands[src.targetPos];
              newOps.push_back(newArg);
            }
            for (const auto& wa : wh.wrappedArgs) {
              touchedCalls.insert(s.call->operands[wa.targetPos]);
            }
            s.call->target = wh.helper->name;
            s.call->operands = std::move(newOps);
            touchedCalls.insert(s.call);
            modifiedFuncs.insert(s.caller);
            anyChanged = true;
            break;
          }
        }
      }
    }

    // Part B: Partial wrapper specialization when shrinkLevel > 0 and a subset
    // M >= 5 of callers of callee pass (call $w X_0 ... X_{r-1}) at parameter k.
    if (getPassOptions().shrinkLevel > 0) {
      struct PartialKey {
        Name calleeName;
        Index paramIdx;
        Name wName;
        bool operator<(const PartialKey& o) const {
          return std::tie(calleeName, paramIdx, wName) <
                 std::tie(o.calleeName, o.paramIdx, o.wName);
        }
      };
      std::map<PartialKey, std::vector<CallSiteInfo>> partialGroups;

      for (auto& f : wasm->functions) {
        if (f->imported()) {
          continue;
        }
        Function* caller = f.get();
        for (auto* c : FindAll<Call>(caller->body).list) {
          if (c->isReturn || touchedCalls.count(c)) {
            continue;
          }
          Function* callee = wasm->getFunctionOrNull(c->target);
          if (!callee || modifiedCallees.count(callee) ||
              c->operands.size() != callee->getNumParams()) {
            continue;
          }
          bool hasUnreachable = false;
          for (auto* op : c->operands) {
            if (op->type == Type::unreachable) {
              hasUnreachable = true;
              break;
            }
          }
          if (hasUnreachable) {
            continue;
          }
          Index m = c->operands.size();
          for (Index k = 0; k < m; k++) {
            auto* wCall = c->operands[k]->dynCast<Call>();
            if (!wCall || wCall->isReturn || touchedCalls.count(wCall) ||
                wCall->target == callee->name) {
              continue;
            }
            Index wArity = wCall->operands.size();
            if (wArity > 1 && k != m - 1) {
              continue;
            }
            Function* wFunc = wasm->getFunctionOrNull(wCall->target);
            if (!wFunc || wFunc->imported() || modifiedCallees.count(wFunc) ||
                wFunc->getNumParams() != wArity ||
                !wFunc->getResults().isSingle() ||
                !Type::isSubType(wFunc->getResults(),
                                 callee->getLocalType(k))) {
              continue;
            }
            bool wArgsValid = true;
            for (Index wi = 0; wi < wArity; wi++) {
              if (wCall->operands[wi]->type == Type::unreachable ||
                  !Type::isSubType(wCall->operands[wi]->type,
                                   wFunc->getLocalType(wi))) {
                wArgsValid = false;
                break;
              }
            }
            if (!wArgsValid) {
              continue;
            }
            bool laterSafe = true;
            for (Index j = k + 1; j < m; j++) {
              if (!canMoveWrapperPast(wFunc, c->operands[j])) {
                laterSafe = false;
                break;
              }
            }
            if (!laterSafe) {
              continue;
            }
            partialGroups[{callee->name, k, wCall->target}].push_back(
              {caller, c});
          }
        }
      }

      // Sort candidate groups by descending site count so the largest wins are
      // applied first.
      std::vector<std::pair<PartialKey, std::vector<CallSiteInfo>>> sortedGroups(
        partialGroups.begin(), partialGroups.end());
      std::sort(sortedGroups.begin(),
                sortedGroups.end(),
                [](const auto& a, const auto& b) {
                  if (a.second.size() != b.second.size()) {
                    return a.second.size() > b.second.size();
                  }
                  return a.first < b.first;
                });

      for (auto& [pKey, sites] : sortedGroups) {
        Function* callee = wasm->getFunction(pKey.calleeName);
        Function* wFunc = wasm->getFunction(pKey.wName);
        Index k = pKey.paramIdx;
        Index wArity = wFunc->getNumParams();
        std::vector<CallSiteInfo> active;
        for (const auto& s : sites) {
          if (touchedCalls.count(s.call)) {
            continue;
          }
          auto* wCall = s.call->operands[k]->dynCast<Call>();
          if (!wCall || touchedCalls.count(wCall) ||
              wCall->target != pKey.wName || wCall->operands.size() != wArity) {
            continue;
          }
          bool wArgsOk = true;
          for (Index wi = 0; wi < wArity; wi++) {
            if (wCall->operands[wi]->type == Type::unreachable ||
                !Type::isSubType(wCall->operands[wi]->type,
                                 wFunc->getLocalType(wi))) {
              wArgsOk = false;
              break;
            }
          }
          if (!wArgsOk) {
            continue;
          }
          active.push_back(s);
        }
        if (active.size() < 5) {
          continue;
        }
        Index m = callee->getNumParams();
        Index numHelperParams = m - 1 + wArity;
        unsigned calleeCallBytes = getFuncCallBytes(callee);
        unsigned wCallBytes = getFuncCallBytes(wFunc);
        if (calleeCallBytes + wCallBytes <= 3) {
          continue;
        }
        unsigned savedBytes =
          (calleeCallBytes + wCallBytes - 3) * unsigned(active.size());
        unsigned wrapperOverhead =
          7 + calleeCallBytes + wCallBytes + 2 * unsigned(numHelperParams);
        if (savedBytes <= wrapperOverhead) {
          continue;
        }

        std::vector<Type> newParams;
        newParams.reserve(numHelperParams);
        for (Index j = 0; j < m; j++) {
          if (j == k) {
            for (Index wi = 0; wi < wArity; wi++) {
              newParams.push_back(wFunc->getLocalType(wi));
            }
          } else {
            newParams.push_back(callee->getLocalType(j));
          }
        }
        std::vector<Expression*> forwardArgs(m);
        Index nextParamIdx = 0;
        for (Index j = 0; j < m; j++) {
          if (j == k) {
            std::vector<Expression*> wArgs(wArity);
            for (Index wi = 0; wi < wArity; wi++) {
              wArgs[wi] =
                builder.makeLocalGet(nextParamIdx++, wFunc->getLocalType(wi));
            }
            forwardArgs[j] =
              builder.makeCall(wFunc->name, wArgs, wFunc->getResults());
          } else {
            Expression* lg =
              builder.makeLocalGet(nextParamIdx++, callee->getLocalType(j));
            forwardArgs[j] = lg;
          }
        }
        Expression* newBody = builder.makeCall(
          callee->name, forwardArgs, callee->getResults(), false);
        Name newName = Names::getValidFunctionName(*wasm, "outlined_w");
        Function* newFunc = wasm->addFunction(Builder::makeFunction(
          newName, Signature(Type(newParams), callee->getResults()), {}, newBody));
        ReFinalize().walkFunctionInModule(newFunc, wasm);
        initHelperEffects(newFunc);

        for (const auto& s : active) {
          auto* wCall = s.call->operands[k]->cast<Call>();
          s.call->target = newName;
          if (wArity == 1) {
            s.call->operands[k] = wCall->operands[0];
          } else {
            ExpressionList newOps(wasm->allocator);
            for (Index j = 0; j < m; j++) {
              if (j == k) {
                for (auto* wOp : wCall->operands) {
                  newOps.push_back(wOp);
                }
              } else {
                newOps.push_back(s.call->operands[j]);
              }
            }
            s.call->operands = std::move(newOps);
          }
          touchedCalls.insert(s.call);
          touchedCalls.insert(wCall);
          modifiedFuncs.insert(s.caller);
        }
        anyChanged = true;
      }
    }

    for (Function* func : modifiedFuncs) {
      ReFinalize().walkFunctionInModule(func, wasm);
      refreshFuncEffects(func);
    }
    return anyChanged;
  }

  // Phase 2: Constant-Argument Sinking, Existing Constant-Specialized Helper
  // Reuse, and Partial Constant-Argument Specialization.
  //
  // 1. Part A (100%-caller in-place constant parameter sinking):
  //    For any non-escaped function $F (including 0/1-arg wrappers that DAE
  //    skips in early stages) where 100% of N >= 2 callers pass the same
  //    constant C (Const, GlobalGet of immutable global, RefFunc, RefNull) at
  //    parameter k, substitute C for every local.get $k in $F and remove
  //    parameter k when N * sizeof(C) > reads(k) * (sizeof(C) - 2).
  // 2. Part B (Existing constant-specialized forwarding helper reuse):
  //    Discover all existing functions $H whose body is a single (call $F ...)
  //    forwarding its parameters in order and filling one or more parameters of
  //    $F with constants. Rewrite any remaining direct call to $F with matching
  //    constant arguments to call $H instead (widening $H's parameter types in
  //    place when $H is non-escaped and a caller passes a subtype of $F's
  //    parameter type).
  // 3. Part C (Partial constant-argument specialization):
  //    Group remaining call sites of each target $F by subsets of constant
  //    arguments and greedily emit new forwarding helpers $H for any subset
  //    that yields net binary size savings >= 6 bytes.
  bool sinkConstantArgs() {
    std::unordered_set<Name> immutableGlobals;
    for (auto& g : wasm->globals) {
      if (!g->mutable_) {
        immutableGlobals.insert(g->name);
      }
    }

    auto isSinkableConstLeaf = [&](Expression* expr) -> bool {
      if (expr->is<Const>() || expr->is<RefNull>() || expr->is<RefFunc>()) {
        return true;
      }
      if (auto* gg = expr->dynCast<GlobalGet>()) {
        return immutableGlobals.count(gg->name) > 0;
      }
      return false;
    };

    auto isSinkableConst = [&](Expression* expr) -> bool {
      if (isSinkableConstLeaf(expr)) {
        return true;
      }
      if (auto* c = expr->dynCast<Call>()) {
        if (c->isReturn || c->operands.size() > 1) {
          return false;
        }
        Function* wFunc = wasm->getFunctionOrNull(c->target);
        if (!wFunc || !isNonMutatingWrapper(wFunc)) {
          return false;
        }
        if (c->operands.empty()) {
          return true;
        }
        return isSinkableConstLeaf(c->operands[0]);
      }
      return false;
    };

    auto getSinkableConstBytes = [&](Expression* expr) -> unsigned {
      if (auto* c = expr->dynCast<Call>()) {
        unsigned b = getShallowBytes(c);
        for (auto* op : c->operands) {
          b += getShallowBytes(op);
        }
        return b;
      }
      return getShallowBytes(expr);
    };

    std::unordered_set<Name> disallowedTargets;
    for (auto& ex : wasm->exports) {
      if (ex->kind == ExternalKind::Function) {
        disallowedTargets.insert(*ex->getInternalName());
      }
    }
    for (auto& seg : wasm->elementSegments) {
      for (auto* expr : seg->data) {
        for (auto* rf : FindAll<RefFunc>(expr).list) {
          disallowedTargets.insert(rf->func);
        }
      }
    }
    for (auto& g : wasm->globals) {
      if (!g->imported() && g->init) {
        for (auto* rf : FindAll<RefFunc>(g->init).list) {
          disallowedTargets.insert(rf->func);
        }
      }
    }

    struct CallSiteInfo {
      Function* caller;
      Call* call;
    };
    std::unordered_map<Name, std::vector<CallSiteInfo>> callsByTarget;
    for (auto& f : wasm->functions) {
      if (f->imported()) {
        continue;
      }
      Function* caller = f.get();
      for (auto* rf : FindAll<RefFunc>(caller->body).list) {
        disallowedTargets.insert(rf->func);
      }
      for (auto* c : FindAll<Call>(caller->body).list) {
        callsByTarget[c->target].push_back({caller, c});
      }
    }

    Builder builder(*wasm);
    std::unordered_set<Function*> modifiedFuncs;
    std::unordered_set<Function*> modifiedCallees;
    std::unordered_set<Function*> calleesWithStaleCallSites;
    bool anyChanged = false;

    // Step 2A: In-place 100%-caller constant parameter sinking.
    for (auto& f : wasm->functions) {
      if (f->imported()) {
        continue;
      }
      Function* callee = f.get();
      if (disallowedTargets.count(callee->name) ||
          modifiedCallees.count(callee) ||
          calleesWithStaleCallSites.count(callee) ||
          callee->getNumParams() == 0) {
        continue;
      }
      auto it = callsByTarget.find(callee->name);
      if (it == callsByTarget.end() || it->second.size() < 2) {
        continue;
      }
      auto& sites = it->second;
      Index N = sites.size();
      Index m = callee->getNumParams();
      bool validSites = true;
      for (const auto& s : sites) {
        if (s.caller == callee || s.call->isReturn ||
            s.call->operands.size() != m) {
          validSites = false;
          break;
        }
        for (auto* op : s.call->operands) {
          if (op->type == Type::unreachable) {
            validSites = false;
            break;
          }
        }
        if (!validSites) {
          break;
        }
      }
      if (!validSites) {
        continue;
      }

      std::vector<Index> localReads(callee->getNumLocals(), 0);
      std::vector<Index> localWrites(callee->getNumLocals(), 0);
      for (auto* lg : FindAll<LocalGet>(callee->body).list) {
        localReads[lg->index]++;
      }
      for (auto* ls : FindAll<LocalSet>(callee->body).list) {
        localWrites[ls->index]++;
      }

      std::vector<bool> removedParam(m, false);
      bool calleeModified = false;

      for (Index revK = 0; revK < m; revK++) {
        Index k = m - 1 - revK;
        if (localReads[k] == 0 || localWrites[k] > 0) {
          continue;
        }
        Expression* firstArg = sites[0].call->operands[k];
        if (!isSinkableConst(firstArg) ||
            !Type::isSubType(firstArg->type, callee->getLocalType(k))) {
          continue;
        }
        Function* wFunc = nullptr;
        if (auto* wCall = firstArg->dynCast<Call>()) {
          if (wCall->target == callee->name) {
            continue;
          }
          wFunc = wasm->getFunctionOrNull(wCall->target);
          if (!wFunc || modifiedCallees.count(wFunc)) {
            continue;
          }
        }
        bool allSame = true;
        for (Index i = 1; i < N; i++) {
          if (!ExpressionAnalyzer::equal(firstArg, sites[i].call->operands[k])) {
            allSame = false;
            break;
          }
        }
        if (!allSame) {
          continue;
        }
        // Check that later arguments at all call sites have no side effects so
        // removing argument k does not alter trap/side-effect ordering.
        bool laterSafe = true;
        for (const auto& s : sites) {
          for (Index j = k + 1; j < m; j++) {
            if (removedParam[j]) {
              continue;
            }
            EffectAnalyzer eff(getPassOptions(), *wasm, s.call->operands[j]);
            eff.localsRead.clear();
            eff.localsWritten.clear();
            if (eff.hasSideEffects() || canExprThrow(s.call->operands[j], eff)) {
              laterSafe = false;
              break;
            }
          }
          if (!laterSafe) {
            break;
          }
        }
        if (!laterSafe) {
          continue;
        }

        unsigned constSize = getSinkableConstBytes(firstArg);
        int savedBytes = int(N) * int(constSize);
        int addedBytes = int(localReads[k]) * (int(constSize) - 2);
        if (savedBytes <= addedBytes) {
          continue;
        }

        struct ConstParamReplacer : public PostWalker<ConstParamReplacer> {
          Index targetIdx;
          Expression* proto;
          Module& wasm;
          ConstParamReplacer(Index targetIdx, Expression* proto, Module& wasm)
            : targetIdx(targetIdx), proto(proto), wasm(wasm) {}
          void visitLocalGet(LocalGet* curr) {
            if (curr->index == targetIdx) {
              replaceCurrent(ExpressionManipulator::copy(proto, wasm));
            }
          }
        };
        ConstParamReplacer(k, firstArg, *wasm).walk(callee->body);
        removedParam[k] = true;
        calleeModified = true;
        modifiedCallees.insert(callee);
        if (wFunc) {
          calleesWithStaleCallSites.insert(wFunc);
        }
        anyChanged = true;
      }

      if (calleeModified) {
        Index totalLocals = callee->getNumLocals();
        std::vector<Index> indexMap(totalLocals, 0);
        std::vector<Type> keptParams;
        Index nextIdx = 0;
        for (Index k = 0; k < m; k++) {
          if (!removedParam[k]) {
            indexMap[k] = nextIdx++;
            keptParams.push_back(callee->getLocalType(k));
          }
        }
        for (Index v = m; v < totalLocals; v++) {
          indexMap[v] = nextIdx++;
        }
        for (auto* lg : FindAll<LocalGet>(callee->body).list) {
          lg->index = indexMap[lg->index];
        }
        for (auto* ls : FindAll<LocalSet>(callee->body).list) {
          ls->index = indexMap[ls->index];
        }
        callee->setParams(Type(keptParams));
        for (const auto& s : sites) {
          ExpressionList newOps(wasm->allocator);
          for (Index k = 0; k < m; k++) {
            if (!removedParam[k]) {
              newOps.push_back(s.call->operands[k]);
            }
          }
          s.call->operands = std::move(newOps);
          modifiedFuncs.insert(s.caller);
        }
        modifiedFuncs.insert(callee);
      }
    }

    if (anyChanged) {
      for (Function* func : modifiedFuncs) {
        ReFinalize().walkFunctionInModule(func, wasm);
        refreshFuncEffects(func);
      }
      modifiedFuncs.clear();
      callsByTarget.clear();
      for (auto& f : wasm->functions) {
        if (!f->imported()) {
          for (auto* c : FindAll<Call>(f->body).list) {
            callsByTarget[c->target].push_back({f.get(), c});
          }
        }
      }
    }

    // Step 2B: Reuse existing constant-specialized forwarding helpers $H.
    // A function $H is a constant-specialized forwarding helper for $F if:
    // - $H->getNumVars() == 0 and $H->getResults() == $F->getResults()
    // - $H->body is a non-return (call $F arg_0 ... arg_{m-1})
    // - Each arg_j is either (local.get $p) with strictly increasing $p
    //   covering 0 .. H_params-1, or a sinkable constant expression.
    struct ForwardingHelper {
      Function* helper;
      Function* target;
      std::vector<std::pair<Index, Expression*>> constArgs;
      std::vector<Index> keptParamPositions;
      unsigned savedBytesPerCall = 0;
    };
    std::unordered_map<Name, std::vector<ForwardingHelper>> helpersByTarget;
    std::unordered_set<Function*> allForwardingHelpers;

    for (auto& f : wasm->functions) {
      if (f->imported() || f->getNumVars() != 0) {
        continue;
      }
      Function* h = f.get();
      auto* innerCall = h->body->dynCast<Call>();
      if (!innerCall || innerCall->isReturn || innerCall->target == h->name) {
        continue;
      }
      Function* target = wasm->getFunctionOrNull(innerCall->target);
      if (!target || target->imported() ||
          h->getResults() != target->getResults() ||
          innerCall->operands.size() != target->getNumParams()) {
        continue;
      }
      Index m = innerCall->operands.size();
      Index expectedLocal = 0;
      std::vector<std::pair<Index, Expression*>> constArgs;
      std::vector<Index> keptPos;
      unsigned savedBytes = 0;
      bool validForwarder = true;
      for (Index j = 0; j < m; j++) {
        Expression* op = innerCall->operands[j];
        if (auto* lg = op->dynCast<LocalGet>()) {
          if (lg->index != expectedLocal) {
            validForwarder = false;
            break;
          }
          expectedLocal++;
          keptPos.push_back(j);
        } else if (isSinkableConst(op)) {
          constArgs.push_back({j, op});
          savedBytes += getSinkableConstBytes(op);
        } else {
          validForwarder = false;
          break;
        }
      }
      if (!validForwarder || expectedLocal != h->getNumParams() ||
          constArgs.empty()) {
        continue;
      }
      helpersByTarget[target->name].push_back(
        {h, target, std::move(constArgs), std::move(keptPos), savedBytes});
      allForwardingHelpers.insert(h);
    }

    std::unordered_set<Call*> rewrittenCalls;
    for (auto& [targetName, helperList] : helpersByTarget) {
      auto it = callsByTarget.find(targetName);
      if (it == callsByTarget.end()) {
        continue;
      }
      Function* targetFunc = wasm->getFunction(targetName);
      // Prefer helpers that save the most bytes per call site.
      std::sort(helperList.begin(),
                helperList.end(),
                [](const ForwardingHelper& a, const ForwardingHelper& b) {
                  if (a.savedBytesPerCall != b.savedBytesPerCall) {
                    return a.savedBytesPerCall > b.savedBytesPerCall;
                  }
                  return a.helper->name < b.helper->name;
                });

      for (const auto& s : it->second) {
        if (s.call->isReturn || rewrittenCalls.count(s.call) ||
            allForwardingHelpers.count(s.caller) ||
            s.call->operands.size() != targetFunc->getNumParams()) {
          continue;
        }
        bool hasUnreachable = false;
        for (auto* op : s.call->operands) {
          if (op->type == Type::unreachable) {
            hasUnreachable = true;
            break;
          }
        }
        if (hasUnreachable) {
          continue;
        }

        for (auto& fh : helperList) {
          if (s.caller == fh.helper) {
            continue;
          }
          bool constsMatch = true;
          for (const auto& [pos, cExpr] : fh.constArgs) {
            if (!ExpressionAnalyzer::equal(s.call->operands[pos], cExpr)) {
              constsMatch = false;
              break;
            }
          }
          if (!constsMatch) {
            continue;
          }
          // Check that non-constant operands following any removed constant
          // position do not have side effects / traps if the helper were to
          // evaluate anything out of order (constants are pure so dropping them
          // at the call site preserves evaluation order of remaining operands).
          bool typesMatch = true;
          bool needsWiden = false;
          for (Index p = 0; p < fh.keptParamPositions.size(); p++) {
            Index origPos = fh.keptParamPositions[p];
            Type argType = s.call->operands[origPos]->type;
            Type hParamType = fh.helper->getLocalType(p);
            if (!Type::isSubType(argType, hParamType)) {
              if (!disallowedTargets.count(fh.helper->name) &&
                  Type::isSubType(argType, fh.target->getLocalType(origPos))) {
                needsWiden = true;
              } else {
                typesMatch = false;
                break;
              }
            }
          }
          if (!typesMatch) {
            continue;
          }
          if (needsWiden) {
            std::vector<Type> widenedParams(fh.keptParamPositions.size());
            auto* hCall = fh.helper->body->cast<Call>();
            for (Index p = 0; p < fh.keptParamPositions.size(); p++) {
              Index origPos = fh.keptParamPositions[p];
              Type lub = Type::getLeastUpperBound(fh.helper->getLocalType(p),
                                                  s.call->operands[origPos]->type);
              widenedParams[p] = lub;
              hCall->operands[origPos]->cast<LocalGet>()->type = lub;
            }
            fh.helper->setParams(Type(widenedParams));
            modifiedFuncs.insert(fh.helper);
          }

          ExpressionList newOps(wasm->allocator);
          for (Index origPos : fh.keptParamPositions) {
            newOps.push_back(s.call->operands[origPos]);
          }
          s.call->target = fh.helper->name;
          s.call->operands = std::move(newOps);
          rewrittenCalls.insert(s.call);
          modifiedFuncs.insert(s.caller);
          anyChanged = true;
          break;
        }
      }
    }

    // Step 2C: Partial constant-argument specialization for remaining calls.
    // For each target function $F with m >= 1 parameters, find subsets of
    // constant arguments shared by M >= 3 call sites and greedily emit new
    // forwarding helpers when net binary savings >= 6 bytes.
    struct ConstValKey {
      enum Kind : uint8_t {
        KConst,
        KGlobal,
        KRefFunc,
        KRefNull,
        KCall0,
      } kind;
      Name callTarget;
      Literal lit;
      Name name;
      Type type;
      bool operator<(const ConstValKey& o) const {
        if (callTarget != o.callTarget) {
          return callTarget < o.callTarget;
        }
        if (kind != o.kind) {
          return kind < o.kind;
        }
        if (kind == KConst) {
          if (lit.type != o.lit.type) {
            return lit.type.getID() < o.lit.type.getID();
          }
          uint8_t bufA[16] = {0};
          uint8_t bufB[16] = {0};
          lit.getBits(bufA);
          o.lit.getBits(bufB);
          return std::memcmp(bufA, bufB, 16) < 0;
        }
        if (kind == KGlobal || kind == KRefFunc) {
          return name < o.name;
        }
        return type.getID() < o.type.getID();
      }
      bool operator==(const ConstValKey& o) const {
        return !(*this < o) && !(o < *this);
      }
    };

    auto makeLeafConstKey = [&](Expression* expr) -> ConstValKey {
      if (auto* c = expr->dynCast<Const>()) {
        return {ConstValKey::KConst, Name(), c->value, Name(), c->type};
      }
      if (auto* gg = expr->dynCast<GlobalGet>()) {
        return {ConstValKey::KGlobal, Name(), Literal(), gg->name, gg->type};
      }
      if (auto* rf = expr->dynCast<RefFunc>()) {
        return {ConstValKey::KRefFunc, Name(), Literal(), rf->func, rf->type};
      }
      auto* rn = expr->cast<RefNull>();
      return {ConstValKey::KRefNull, Name(), Literal(), Name(), rn->type};
    };

    auto makeConstKey = [&](Expression* expr) -> ConstValKey {
      if (auto* call = expr->dynCast<Call>()) {
        if (call->operands.empty()) {
          return {
            ConstValKey::KCall0, call->target, Literal(), Name(), call->type};
        }
        ConstValKey k = makeLeafConstKey(call->operands[0]);
        k.callTarget = call->target;
        return k;
      }
      return makeLeafConstKey(expr);
    };

    for (auto& f : wasm->functions) {
      if (f->imported()) {
        continue;
      }
      Function* target = f.get();
      Index m = target->getNumParams();
      if (m == 0 || m > 8) {
        continue;
      }
      auto it = callsByTarget.find(target->name);
      if (it == callsByTarget.end() || it->second.size() < 3) {
        continue;
      }

      std::vector<Index> targetReads(target->getNumLocals(), 0);
      for (auto* lg : FindAll<LocalGet>(target->body).list) {
        targetReads[lg->index]++;
      }

      std::vector<CallSiteInfo> validCalls;
      for (const auto& s : it->second) {
        if (s.call->isReturn || rewrittenCalls.count(s.call) ||
            s.call->operands.size() != m) {
          continue;
        }
        bool ok = true;
        for (Index j = 0; j < m; j++) {
          if (s.call->operands[j]->type == Type::unreachable ||
              !Type::isSubType(s.call->operands[j]->type,
                               target->getLocalType(j))) {
            ok = false;
            break;
          }
        }
        if (ok) {
          validCalls.push_back(s);
        }
      }
      if (validCalls.size() < 3) {
        continue;
      }

      // Greedily specialize subsets for this target until no profitable subset
      // remains.
      std::vector<bool> callUsed(validCalls.size(), false);
      while (true) {
        using SubsetPattern = std::vector<std::pair<Index, ConstValKey>>;
        struct PatternInfo {
          std::vector<Index> callIndices;
          std::vector<Expression*> protoConsts;
          unsigned constBytesSum = 0;
        };
        std::map<SubsetPattern, PatternInfo> patternMap;

        for (Index ci = 0; ci < validCalls.size(); ci++) {
          if (callUsed[ci]) {
            continue;
          }
          Call* c = validCalls[ci].call;
          std::vector<Index> constPositions;
          std::vector<ConstValKey> constKeys;
          std::vector<Expression*> constExprs;
          std::vector<unsigned> constSizes;
          for (Index j = 0; j < m; j++) {
            if (targetReads[j] > 0 && isSinkableConst(c->operands[j])) {
              constPositions.push_back(j);
              constKeys.push_back(makeConstKey(c->operands[j]));
              constExprs.push_back(c->operands[j]);
              constSizes.push_back(getSinkableConstBytes(c->operands[j]));
            }
          }
          Index k = constPositions.size();
          if (k == 0) {
            continue;
          }
          // Enumerate all non-empty subsets of up to 3 constant positions, plus
          // the full constant set if k > 3.
          auto addMask = [&](uint32_t mask) {
            SubsetPattern pat;
            std::vector<Expression*> protos;
            unsigned bytesSum = 0;
            for (Index bit = 0; bit < k; bit++) {
              if (mask & (1u << bit)) {
                pat.push_back({constPositions[bit], constKeys[bit]});
                protos.push_back(constExprs[bit]);
                bytesSum += constSizes[bit];
              }
            }
            auto& info = patternMap[pat];
            if (info.callIndices.empty()) {
              info.protoConsts = std::move(protos);
              info.constBytesSum = bytesSum;
            }
            info.callIndices.push_back(ci);
          };

          if (k <= 4) {
            for (uint32_t mask = 1; mask < (1u << k); mask++) {
              addMask(mask);
            }
          } else {
            for (Index a = 0; a < k; a++) {
              addMask(1u << a);
              for (Index b = a + 1; b < k; b++) {
                addMask((1u << a) | (1u << b));
              }
            }
            addMask((1u << k) - 1);
          }
        }

        int bestNetSavings = 5; // Require at least 6 bytes net savings
        SubsetPattern bestPat;
        for (const auto& [pat, info] : patternMap) {
          Index M = info.callIndices.size();
          if (M < 3) {
            continue;
          }
          Index numRemoved = pat.size();
          Index numKept = m - numRemoved;
          int grossSaved = int(M) * int(info.constBytesSum);
          int helperOverhead =
            10 + int(info.constBytesSum) + 2 * int(numKept);
          int netSavings = grossSaved - helperOverhead;
          if (netSavings > bestNetSavings) {
            bestNetSavings = netSavings;
            bestPat = pat;
          }
        }

        if (bestPat.empty()) {
          break;
        }

        const auto& bestInfo = patternMap[bestPat];
        std::vector<Expression*> constAtPos(m, nullptr);
        for (Index idx = 0; idx < bestPat.size(); idx++) {
          constAtPos[bestPat[idx].first] = bestInfo.protoConsts[idx];
        }

        std::vector<Type> newParams;
        std::vector<Index> keptPositions;
        for (Index j = 0; j < m; j++) {
          if (!constAtPos[j]) {
            keptPositions.push_back(j);
            // Compute LUB of argument types across the selected call sites so
            // the specialized helper has the tightest valid parameter type.
            Type lub = validCalls[bestInfo.callIndices[0]].call->operands[j]->type;
            for (Index idx = 1; idx < bestInfo.callIndices.size(); idx++) {
              lub = Type::getLeastUpperBound(
                lub,
                validCalls[bestInfo.callIndices[idx]].call->operands[j]->type);
            }
            if (!Type::isSubType(lub, target->getLocalType(j))) {
              lub = target->getLocalType(j);
            }
            newParams.push_back(lub);
          }
        }

        std::vector<Expression*> forwardArgs(m, nullptr);
        Index nextParamIdx = 0;
        for (Index j = 0; j < m; j++) {
          if (constAtPos[j]) {
            forwardArgs[j] =
              ExpressionManipulator::copy(constAtPos[j], *wasm);
          } else {
            Type pType = newParams[nextParamIdx];
            forwardArgs[j] = builder.makeLocalGet(nextParamIdx, pType);
            nextParamIdx++;
          }
        }

        Expression* newBody = builder.makeCall(
          target->name, forwardArgs, target->getResults(), false);
        Name newName = Names::getValidFunctionName(*wasm, "outlined_c");
        Function* newFunc = wasm->addFunction(Builder::makeFunction(
          newName, Signature(Type(newParams), target->getResults()), {}, newBody));
        ReFinalize().walkFunctionInModule(newFunc, wasm);
        initHelperEffects(newFunc);

        for (Index ci : bestInfo.callIndices) {
          callUsed[ci] = true;
          Call* c = validCalls[ci].call;
          ExpressionList newOps(wasm->allocator);
          for (Index origPos : keptPositions) {
            newOps.push_back(c->operands[origPos]);
          }
          c->target = newName;
          c->operands = std::move(newOps);
          rewrittenCalls.insert(c);
          modifiedFuncs.insert(validCalls[ci].caller);
        }
        anyChanged = true;
      }
    }

    for (Function* func : modifiedFuncs) {
      ReFinalize().walkFunctionInModule(func, wasm);
      refreshFuncEffects(func);
    }
    return anyChanged;
  }
};

} // anonymous namespace

Pass* createOutlineExpressionsPass() { return new OutlineExpressions(); }

} // namespace wasm
