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
};

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
  std::vector<Index> totalReads;
  std::vector<Index> totalWrites;
  Index loopDepth = 0;

  // Infos of the already-visited children of the expressions we are in the
  // middle of visiting.
  std::vector<NodeInfo> stack;

  Scanner(Module& wasm,
          Function* func,
          FunctionInfo& info,
          Mode mode,
          bool constHoles)
    : wasm(wasm), func(func), info(info), mode(mode), constHoles(constHoles) {
    LocalUseCounter counter(func->getNumLocals());
    counter.walk(func->body);
    totalReads = std::move(counter.reads);
    totalWrites = std::move(counter.writes);
  }

  static void scan(Scanner* self, Expression** currp) {
    PostWalker<Scanner, UnifiedExpressionVisitor<Scanner>>::scan(self, currp);
    if ((*currp)->is<Loop>()) {
      self->loopDepth++;
    }
  }

  void visitExpression(Expression* curr) {
    if (curr->is<Loop>()) {
      assert(loopDepth > 0);
      loopDepth--;
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
    if (curr->is<LocalGet>() || curr->is<LocalSet>() ||
        (constHoles && isConstHole(wasm, curr))) {
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
  std::vector<Hole> holes;

  HoleFinder(Module& wasm, const Site* site, bool constHoles)
    : wasm(wasm), site(site), constHoles(constHoles) {}

  void visitExpression(Expression* curr) {
    if (auto* get = curr->dynCast<LocalGet>()) {
      if (site->isInternalLocal(get->index) ||
          site->isOverwriteLocal(get->index)) {
        return;
      }
      for (auto& hole : holes) {
        if (hole.isLocal && hole.localIndex == get->index) {
          return;
        }
      }
      holes.push_back({curr, get->index, get->type, true});
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
getHoles(Module& wasm, Expression* expr, const Site* site, bool constHoles) {
  HoleFinder finder(wasm, site, constHoles);
  finder.walk(expr);
  appendPassthroughHoles(site, finder.holes);
  return std::move(finder.holes);
}

std::vector<Hole> getHoles(Module& wasm, const Site* site, bool constHoles) {
  if (site->sliceLen == 0) {
    return getHoles(wasm, site->expr, site, constHoles);
  }
  HoleFinder finder(wasm, site, constHoles);
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

  void run(Module* module) override {
    wasm = module;
    scratchGlobals.clear();

    auto mode = getMode();

    unCseCallRefTargets();

    constHoles = false;
    for (Index round = 0; round < MaxRounds; round++) {
      if (!runRound(mode)) {
        break;
      }
    }
    foldCallRefDispatch(true);
    constHoles = true;
    for (Index round = 0; round < MaxConstRounds; round++) {
      if (!runRound(mode)) {
        break;
      }
    }
    foldCallRefDispatch(true);
    for (Index round = 0; round < 2; round++) {
      if (!sinkWrapperCalls()) {
        break;
      }
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
        Scanner(*wasm, func, info, mode, constHoles).walk(func->body);
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
              rightSite->isPassthroughLocal(rightIndex)) {
          mismatch = true;
        }
      }
    };
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
        if (leftGet->type != rightGet->type) {
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
      site->holes = getHoles(*wasm, site, constHoles);
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
        func->vars.size() != site.internalLocals.size() ||
        func->getResults() != getHelperResults(&site)) {
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
      if (hole.localIndex != i || hole.type != func->getLocalType(i)) {
        return;
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
    std::unordered_map<Expression*, Index> paramOfConst;

    HoleReplacer(Module& wasm) : wasm(wasm) {}

    void visitExpression(Expression* curr) {
      if (auto* get = curr->dynCast<LocalGet>()) {
        get->index = paramOfLocal.at(get->index);
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
      Index pIdx = params.size();
      holeParamIdx[i] = pIdx;
      if (hole.isLocal) {
        replacer.paramOfLocal[hole.localIndex] = pIdx;
      } else {
        replacer.paramOfConst[hole.expr] = pIdx;
      }
      params.push_back(hole.type);
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
    return wasm->addFunction(
      Builder::makeFunction(name,
                            Signature(Type(params), getHelperResults(site)),
                            std::move(vars),
                            body));
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

  static bool isPureVtableBodyExpr(Expression* e) {
    if (!e || e->type == Type::unreachable) {
      return false;
    }
    if (e->is<LocalGet>() || e->is<Const>()) {
      return true;
    }
    if (auto* sg = e->dynCast<StructGet>()) {
      return sg->ref && sg->ref->type.isStruct() &&
             isPureVtableBodyExpr(sg->ref);
    }
    if (auto* rc = e->dynCast<RefCast>()) {
      return !rc->desc && rc->ref && isPureVtableBodyExpr(rc->ref);
    }
    if (auto* ra = e->dynCast<RefAs>()) {
      return ra->op == RefAsNonNull && ra->value &&
             isPureVtableBodyExpr(ra->value);
    }
    if (auto* ag = e->dynCast<ArrayGet>()) {
      return ag->ref && ag->ref->type.isArray() && ag->index &&
             isPureVtableBodyExpr(ag->ref) && isPureVtableBodyExpr(ag->index);
    }
    return false;
  }

  bool isPureVtableHelper(Name name) {
    Function* f = wasm->getFunctionOrNull(name);
    if (!f || f->imported() || f->getNumParams() != 1 || !f->vars.empty() ||
        !f->getResults().isSingle() || !f->getResults().isRef()) {
      return false;
    }
    return isPureVtableBodyExpr(f->body);
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
          if (!sg0 || sg0->ref != wv) {
            continue;
          }
          Expression* sg0Par = layout.parent[sg0];
          auto* cr0 = sg0Par ? sg0Par->dynCast<CallRef>() : nullptr;
          if (!cr0 || cr0->target != sg0 || cr0->isReturn ||
              cr0->operands.empty()) {
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
          Expression* lca = cr0;
          Index wvPre = layout.preOrder.at(wv);
          bool readsOk = true;
          for (auto* gv : reads[v]) {
            if (layout.preOrder.at(gv) <= wvPre) {
              readsOk = false;
              break;
            }
            Expression* gvPar = layout.parent[gv];
            auto* sgi = gvPar ? gvPar->dynCast<StructGet>() : nullptr;
            if (!sgi || sgi->ref != gv) {
              readsOk = false;
              break;
            }
            Expression* sgiPar = layout.parent[sgi];
            auto* cri = sgiPar ? sgiPar->dynCast<CallRef>() : nullptr;
            if (!cri || cri->target != sgi || cri->isReturn ||
                cri->operands.empty()) {
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
            readStructGets.push_back(sgi);
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
          sg0->ref = vExpr;
          for (auto* sgi : readStructGets) {
            sgi->ref = ExpressionManipulator::copy(vExpr, *wasm);
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
        wasm->addFunction(Builder::makeFunction(
          newName, Signature(Type(newParams), sig.results), {}, newBody));
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

  bool canMoveWrapperPast(Function* wFunc, Expression* laterOp) {
    EffectAnalyzer opEff(getPassOptions(), *wasm, laterOp);
    if (!opEff.hasAnything()) {
      return true;
    }
    if (!wFunc || wFunc->imported()) {
      return false;
    }
    EffectAnalyzer wEff(getPassOptions(), *wasm, wFunc);
    if (!wEff.invalidates(opEff)) {
      return true;
    }
    // Many constant-pool / string-literal wrappers (like $0, $1, $3) lazily
    // populate a private cache array/global on first miss, while laterOp is a
    // pure call-free read (e.g. struct.get of a field).
    if (!opEff.hasSideEffects() && !opEff.calls && !opEff.throws() &&
        FindAll<StructSet>(wFunc->body).list.empty()) {
      bool globalsDisjoint = true;
      for (auto* gg : FindAll<GlobalGet>(laterOp).list) {
        if (wEff.globalsWritten.count(gg->name)) {
          globalsDisjoint = false;
          break;
        }
      }
      if (globalsDisjoint &&
          (FindAll<ArrayGet>(laterOp).list.empty() ||
           FindAll<ArraySet>(wFunc->body).list.empty())) {
        return true;
      }
    }
    return false;
  }

  bool isSingleEntryReadSafe(Function* callee,
                             Index paramIdx,
                             Function* wFunc,
                             LocalGet*& outGet) {
    outGet = nullptr;
    if (!wFunc || wFunc->imported()) {
      return false;
    }
    EffectAnalyzer wEff(getPassOptions(), *wasm, wFunc);
    if (wEff.hasAnything()) {
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
    std::unordered_set<Expression*> touchedCalls;
    bool anyChanged = false;

    // Part A: In-place wrapper sinking when 100% of callers of a non-escaped
    // function pass (call $w X_k) at parameter k.
    for (auto& f : wasm->functions) {
      if (f->imported()) {
        continue;
      }
      Function* callee = f.get();
      if (disallowedTargets.count(callee->name) ||
          modifiedCallees.count(callee) || callee->getNumParams() == 0) {
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
      std::vector<Type> currentParams(m);
      for (Index k = 0; k < m; k++) {
        currentParams[k] = callee->getLocalType(k);
      }
      bool calleeModified = false;

      for (Index revK = 0; revK < m; revK++) {
        Index k = m - 1 - revK;
        if (localReads[k] == 0) {
          continue;
        }
        auto* firstW = sites[0].call->operands[k]->dynCast<Call>();
        if (!firstW || firstW->isReturn || firstW->operands.size() != 1 ||
            firstW->target == callee->name ||
            firstW->operands[0]->type == Type::unreachable) {
          continue;
        }
        Name wName = firstW->target;
        Function* wFunc = wasm->getFunctionOrNull(wName);
        if (!wFunc || modifiedCallees.count(wFunc)) {
          continue;
        }
        Type wInputType = wFunc->getParams();
        Type wResultType = wFunc->getResults();
        if (!wInputType.isSingle() || !wResultType.isSingle() ||
            !Type::isSubType(wResultType, currentParams[k])) {
          continue;
        }

        bool allMatch = true;
        for (const auto& s : sites) {
          auto* wCall = s.call->operands[k]->dynCast<Call>();
          if (!wCall || touchedCalls.count(wCall) || wCall->isReturn ||
              wCall->target != wName || wCall->operands.size() != 1 ||
              wCall->operands[0]->type == Type::unreachable ||
              !Type::isSubType(wCall->operands[0]->type, wInputType)) {
            allMatch = false;
            break;
          }
          for (Index j = k + 1; j < m; j++) {
            if (!canMoveWrapperPast(wFunc, s.call->operands[j])) {
              allMatch = false;
              break;
            }
          }
          if (!allMatch) {
            break;
          }
        }
        if (!allMatch) {
          continue;
        }

        Type oldParamType = currentParams[k];
        LocalGet* singleGet = nullptr;
        bool directSub = localWrites[k] == 0 && localReads[k] == 1 &&
                         isSingleEntryReadSafe(callee, k, wFunc, singleGet);
        unsigned costBytes = 3;
        if (!directSub) {
          if (wInputType == oldParamType) {
            costBytes = 7;
          } else if (oldParamType.isDefaultable()) {
            costBytes = 9;
          } else {
            continue;
          }
        }
        if (3 * N <= costBytes) {
          continue;
        }

        // Apply in-place sinking on callee and all N call sites.
        currentParams[k] = wInputType;
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
          Expression* wCallExpr = builder.makeCall(
            wName, {builder.makeLocalGet(k, wInputType)}, wResultType);
          SingleGetReplacer(singleGet, wCallExpr).walk(callee->body);
        } else if (wInputType == oldParamType) {
          Expression* setExpr = builder.makeLocalSet(
            k,
            builder.makeCall(
              wName, {builder.makeLocalGet(k, wInputType)}, wResultType));
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
          Expression* setExpr = builder.makeLocalSet(
            tmpLocal,
            builder.makeCall(
              wName, {builder.makeLocalGet(k, wInputType)}, wResultType));
          callee->body = builder.makeSequence(setExpr, callee->body);
        }

        for (const auto& s : sites) {
          auto* wCall = s.call->operands[k]->cast<Call>();
          touchedCalls.insert(s.call);
          touchedCalls.insert(wCall);
          s.call->operands[k] = wCall->operands[0];
          modifiedFuncs.insert(s.caller);
        }
        modifiedCallees.insert(callee);
        modifiedCallees.insert(wFunc);
        calleeModified = true;
        anyChanged = true;
      }

      if (calleeModified) {
        callee->setParams(Type(currentParams));
        modifiedFuncs.insert(callee);
      }
    }

    // Part B: Partial wrapper specialization when shrinkLevel > 0 and a subset
    // M >= 6 of callers of callee pass (call $w X_k) at parameter k.
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
          if (!callee || callee->imported() || modifiedCallees.count(callee) ||
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
                wCall->operands.size() != 1 ||
                wCall->operands[0]->type == Type::unreachable) {
              continue;
            }
            Function* wFunc = wasm->getFunctionOrNull(wCall->target);
            if (!wFunc || modifiedCallees.count(wFunc) ||
                !wFunc->getParams().isSingle() ||
                !wFunc->getResults().isSingle() ||
                !Type::isSubType(wFunc->getResults(),
                                 callee->getLocalType(k)) ||
                !Type::isSubType(wCall->operands[0]->type,
                                 wFunc->getParams())) {
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
        std::vector<CallSiteInfo> active;
        for (const auto& s : sites) {
          if (touchedCalls.count(s.call)) {
            continue;
          }
          auto* wCall = s.call->operands[k]->dynCast<Call>();
          if (!wCall || touchedCalls.count(wCall) ||
              wCall->target != pKey.wName || wCall->operands.size() != 1 ||
              !Type::isSubType(wCall->operands[0]->type, wFunc->getParams())) {
            continue;
          }
          active.push_back(s);
        }
        if (active.size() < 6) {
          continue;
        }
        Index m = callee->getNumParams();
        unsigned savedBytes = 3 * unsigned(active.size());
        unsigned helperCost = 15 + 2 * unsigned(m);
        if (savedBytes <= helperCost + 6) {
          continue;
        }

        std::vector<Type> newParams(m);
        for (Index j = 0; j < m; j++) {
          newParams[j] =
            (j == k) ? wFunc->getParams() : callee->getLocalType(j);
        }
        std::vector<Expression*> forwardArgs(m);
        for (Index j = 0; j < m; j++) {
          Expression* lg = builder.makeLocalGet(j, newParams[j]);
          if (j == k) {
            forwardArgs[j] =
              builder.makeCall(wFunc->name, {lg}, wFunc->getResults());
          } else {
            forwardArgs[j] = lg;
          }
        }
        Expression* newBody = builder.makeCall(
          callee->name, forwardArgs, callee->getResults(), false);
        Name newName = Names::getValidFunctionName(*wasm, "outlined_w");
        Function* newFunc = wasm->addFunction(Builder::makeFunction(
          newName, Signature(Type(newParams), callee->getResults()), {}, newBody));
        ReFinalize().walkFunctionInModule(newFunc, wasm);

        for (const auto& s : active) {
          auto* wCall = s.call->operands[k]->cast<Call>();
          s.call->target = newName;
          s.call->operands[k] = wCall->operands[0];
          touchedCalls.insert(s.call);
          touchedCalls.insert(wCall);
          modifiedFuncs.insert(s.caller);
        }
        anyChanged = true;
      }
    }

    for (Function* func : modifiedFuncs) {
      ReFinalize().walkFunctionInModule(func, wasm);
      if (func->effects) {
        func->effects =
          std::make_unique<EffectAnalyzer>(getPassOptions(), *wasm, func);
      }
    }
    return anyChanged;
  }
};

} // anonymous namespace

Pass* createOutlineExpressionsPass() { return new OutlineExpressions(); }

} // namespace wasm
