/*
 * Copyright 2016 WebAssembly Community Group participants
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
// Inlining.
//
// Two versions are provided: inlining and inlining-optimizing. You
// probably want the optimizing version, which will optimize locations
// we inlined into, as inlining by itself creates a block to house the
// inlined code, some temp locals, etc., which can usually be removed
// by optimizations. Note that the two versions use the same heuristics,
// so we don't take into account the overhead if you don't optimize
// afterwards. The non-optimizing version is mainly useful for debugging,
// or if you intend to run a full set of optimizations anyhow on
// everything later.
//

#include <atomic>
#include <sstream>

#include "ir/branch-hints.h"
#include "ir/branch-utils.h"
#include "ir/drop.h"
#include "ir/effects.h"
#include "ir/eh-utils.h"
#include "ir/element-utils.h"
#include "ir/find_all.h"
#include "ir/literal-utils.h"
#include "ir/localize.h"
#include "ir/metadata.h"
#include "ir/module-utils.h"
#include "ir/names.h"
#include "ir/properties.h"
#include "ir/return-utils.h"
#include "ir/type-updating.h"
#include "ir/utils.h"
#include "parsing.h"
#include "pass.h"
#include "passes/opt-utils.h"
#include "wasm-builder.h"
#include "wasm.h"

#ifndef INLINING_DEBUG
#define INLINING_DEBUG 0
#endif

namespace wasm {

namespace {

enum class InliningMode {
  // We do not know yet if this function can be inlined, as that has
  // not been computed yet.
  Unknown,
  // This function cannot be inlined in any way.
  Uninlineable,
  // This function can be inlined fully, that is, normally: the entire function
  // can be inlined. This is in contrast to split/partial inlining, see below.
  Full,
  // This function cannot be inlined normally, but we can use split inlining,
  // using pattern "A" or "B" (see below).
  SplitPatternA,
  SplitPatternB
};

// Whether a function is just one instruction that always shrinks when inlined.
enum class TrivialInstruction {
  // Function is not a single instruction, or it may not shrink when inlined.
  NotTrivial,

  // Function is just one instruction, with `local.get`s as arguments, and with
  // each `local` is used exactly once, and in the order they appear in the
  // argument list.
  //
  // In this case, inlining the function generates smaller code, and it is also
  // good for runtime. (Note that in theory inlining an instruction might grow
  // the size, if before we had a call - one byte+LEB - and after we have
  // something like a prefixed instruction - two bytes - with some other LEB
  // like a type index. Figuring out when the LEBs will cause growth here is
  // hard, and probably not worth it, since doing a call to run a single
  // instruction is almost always going to be larger and slower.)
  Shrinks,

  // Function is a single instruction, but maybe with constant arguments, or
  // maybe some locals are used more than once. In these cases code size does
  // not always shrink: at the call sites, omitted locals can create `drop`
  // instructions, a local used multiple times can create new locals, and
  // encoding of constants may be larger than just a `local.get` with a small
  // index. In these cases we still want to inline with `-O3`, but the code size
  // may increase when inlined.
  MayNotShrink,
};

// Useful info on a function, helping us decide if we can inline it.
struct FunctionInfo {
  std::atomic<Index> refs;
  Index size;
  Index nonParamBodyBytes;
  Index numParamGets;
  bool hasCalls;
  bool hasLoops;
  bool hasTryDelegate;
  // Something is used globally if there is a reference to it in a table or
  // export etc.
  bool usedGlobally;
  // Whether some control flow in the function (an if, select, br_if, br_table
  // or br_on*) depends on a parameter. If so, then calling the function with
  // constant arguments is likely to let that control flow be folded away after
  // inlining.
  bool hasParamDependentBranch;
  bool paramsUsedOnceInOrder;
  bool trivialDropCall;
  TrivialInstruction trivialInstruction;
  InliningMode inliningMode;
  std::optional<uint8_t> toolchainInlineHint;

  FunctionInfo() { clear(); }

  void clear() {
    refs = 0;
    size = 0;
    nonParamBodyBytes = 0;
    numParamGets = 0;
    hasCalls = false;
    hasLoops = false;
    hasTryDelegate = false;
    usedGlobally = false;
    hasParamDependentBranch = false;
    paramsUsedOnceInOrder = false;
    trivialDropCall = false;
    trivialInstruction = TrivialInstruction::NotTrivial;
    inliningMode = InliningMode::Unknown;
    toolchainInlineHint = std::nullopt;
  }

  // Provide an explicit = operator as the |refs| field lacks one by default.
  FunctionInfo& operator=(const FunctionInfo& other) {
    refs = other.refs.load();
    size = other.size;
    nonParamBodyBytes = other.nonParamBodyBytes;
    numParamGets = other.numParamGets;
    hasCalls = other.hasCalls;
    hasLoops = other.hasLoops;
    hasTryDelegate = other.hasTryDelegate;
    usedGlobally = other.usedGlobally;
    hasParamDependentBranch = other.hasParamDependentBranch;
    paramsUsedOnceInOrder = other.paramsUsedOnceInOrder;
    trivialDropCall = other.trivialDropCall;
    trivialInstruction = other.trivialInstruction;
    inliningMode = other.inliningMode;
    toolchainInlineHint = other.toolchainInlineHint;
    return *this;
  }

  // See pass.h for how defaults for these options were chosen.
  bool worthFullInlining(PassOptions& options, bool byteBudgetMode = false) {
    // Until we have proper support for try-delegate, ignore such functions.
    // FIXME https://github.com/WebAssembly/binaryen/issues/3634
    if (hasTryDelegate) {
      return false;
    }
    // Follow the toolchain hint, if present.
    if (toolchainInlineHint == CodeAnnotation::NeverInline) {
      return false;
    }
    if (toolchainInlineHint == CodeAnnotation::AlwaysInline) {
      return true;
    }
    // If it's small enough that we always want to inline such things, do so.
    if ((!byteBudgetMode || options.shrinkLevel == 0 || refs <= 1 ||
         usedGlobally) &&
        size <= options.inlining.alwaysInlineMaxSize) {
      return true;
    }
    // If it has one use, then inlining it would likely reduce code size, at
    // least for reasonable function sizes.
    if (refs == 1 && !usedGlobally &&
        size <= options.inlining.oneCallerInlineMaxSize) {
      return true;
    }
    // If the function calls another one in a way that always shrinks when
    // inlined, inline it in all optimization and shrink modes.
    if ((!byteBudgetMode || options.shrinkLevel == 0 || usedGlobally) &&
        trivialInstruction == TrivialInstruction::Shrinks) {
      return true;
    }
    // When every parameter is read once in evaluation order with no local
    // variables, writes, loops, or branches, each parameter local.get vanishes
    // when replaced by the caller's argument expression. Inline into all
    // callers if eliminating the function saves more bytes than the duplicated
    // non-parameter body costs.
    if (!byteBudgetMode && !usedGlobally && !hasLoops &&
        paramsUsedOnceInOrder && refs >= 2 && options.shrinkLevel > 0 &&
        (hasCalls || trivialInstruction != TrivialInstruction::MayNotShrink)) {
      if ((refs - 1) * nonParamBodyBytes <
          3 * refs + 2 * numParamGets + 5) {
        return true;
      }
    }
    // If it's so big that we have no flexible options that could allow it,
    // do not inline.
    if (size > options.inlining.flexibleInlineMaxSize) {
      return false;
    }
    // More than one use, so we can't eliminate it after inlining, and inlining
    // it will hurt code size. Stop if we are focused on size or not heavily
    // focused on speed.
    if (options.shrinkLevel > 0 || options.optimizeLevel < 3) {
      return false;
    }
    // The function is just one instruction, but the code size may increase when
    // inlined. We only inline it fully with `-O3`.
    if (trivialInstruction == TrivialInstruction::MayNotShrink) {
      return true;
    }
    // Trivial instructions are already handled. Inline if
    // 1. The function doesn't have calls, and
    // 2. The function doesn't have loops, or we allow inlining with loops.
    return !hasCalls && (!hasLoops || options.inlining.allowFunctionsWithLoops);
  }

  // Whether the function is worth inlining at call sites where every argument
  // is a constant (see Planner::hasConstantArgs). We only do that when the
  // function branches on a parameter, so that constant arguments are likely to
  // fold away a large part of the inlined code (and possibly all of it, which
  // can then enable further optimizations in the caller).
  bool worthConstantArgInlining(PassOptions& options) {
    if (hasTryDelegate) {
      return false;
    }
    if (toolchainInlineHint == CodeAnnotation::NeverInline) {
      return false;
    }
    if (options.shrinkLevel > 0 || options.optimizeLevel < 3) {
      return false;
    }
    if (size > options.inlining.constantArgInlineMaxSize) {
      return false;
    }
    if (hasLoops && !options.inlining.allowFunctionsWithLoops) {
      return false;
    }
    return hasParamDependentBranch;
  }
};

using NameInfoMap = std::unordered_map<Name, FunctionInfo>;
using GlobalRefMap = std::unordered_map<Name, std::atomic<Index>>;

struct FunctionInfoScanner
  : public WalkerPass<PostWalker<FunctionInfoScanner>> {
  bool isFunctionParallel() override { return true; }

  bool modifiesBinaryenIR() override { return false; }

  FunctionInfoScanner(NameInfoMap& infos, GlobalRefMap* globalRefs = nullptr)
    : infos(infos), globalRefs(globalRefs) {}

  std::unique_ptr<Pass> create() override {
    return std::make_unique<FunctionInfoScanner>(infos, globalRefs);
  }

  void visitLoop(Loop* curr) {
    // having a loop
    infos[getFunction()->name].hasLoops = true;
  }

  void visitCall(Call* curr) {
    // can't add a new element in parallel
    assert(infos.contains(curr->target));
    infos[curr->target].refs++;
    // having a call
    infos[getFunction()->name].hasCalls = true;
  }

  void visitCallIndirect(CallIndirect* curr) {
    infos[getFunction()->name].hasCalls = true;
  }

  void visitCallRef(CallRef* curr) {
    infos[getFunction()->name].hasCalls = true;
  }

  void visitGlobalGet(GlobalGet* curr) {
    if (globalRefs) {
      auto it = globalRefs->find(curr->name);
      if (it != globalRefs->end()) {
        it->second++;
      }
    }
  }

  void visitGlobalSet(GlobalSet* curr) {
    if (globalRefs) {
      auto it = globalRefs->find(curr->name);
      if (it != globalRefs->end()) {
        it->second++;
      }
    }
  }

  void visitTry(Try* curr) {
    if (curr->isDelegate()) {
      infos[getFunction()->name].hasTryDelegate = true;
    }
  }

  void visitRefFunc(RefFunc* curr) {
    assert(infos.contains(curr->func));
    infos[curr->func].refs++;
    infos[curr->func].usedGlobally = true;
  }

  // Note control flow that depends on a parameter. (We do not check that the
  // parameter is never written to; that is rare and only makes this a less
  // precise heuristic.)
  void noteCondition(Expression* condition) {
    if (!getFunction()) {
      // Module-level code (global initializers etc.).
      return;
    }
    auto& info = infos[getFunction()->name];
    if (info.hasParamDependentBranch) {
      return;
    }
    auto numParams = getFunction()->getNumParams();
    for (auto* get : FindAll<LocalGet>(condition).list) {
      if (get->index < numParams) {
        info.hasParamDependentBranch = true;
        return;
      }
    }
  }

  void visitIf(If* curr) { noteCondition(curr->condition); }

  void visitSelect(Select* curr) { noteCondition(curr->condition); }

  void visitSwitch(Switch* curr) { noteCondition(curr->condition); }

  void visitBreak(Break* curr) {
    if (curr->condition) {
      noteCondition(curr->condition);
    }
  }

  void visitBrOn(BrOn* curr) { noteCondition(curr->ref); }

  static int getULEBSize(uint64_t value) {
    int size = 0;
    do {
      value >>= 7;
      size++;
    } while (value != 0);
    return size;
  }

  static int getSLEBSize(int64_t value) {
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

  static Index getShallowBytes(Expression* curr) {
    switch (curr->_id) {
      case Expression::BlockId:
        return curr->cast<Block>()->name.is() ? 3 : 0;
      case Expression::IfId:
        return curr->cast<If>()->ifFalse ? 4 : 3;
      case Expression::LoopId:
        return 3;
      case Expression::BreakId:
        return 2;
      case Expression::CallId:
      case Expression::CallRefId:
      case Expression::GlobalGetId:
      case Expression::GlobalSetId:
      case Expression::RefFuncId:
      case Expression::StringConstId:
        return 3;
      case Expression::CallIndirectId:
        return 4;
      case Expression::LocalGetId:
      case Expression::LocalSetId:
        return 2;
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

  void visitFunction(Function* curr) {
    auto& info = infos[curr->name];

    struct BodyScanner
      : public PostWalker<BodyScanner, UnifiedExpressionVisitor<BodyScanner>> {
      Name funcName;
      Index size = 0;
      Index nonParamBodyBytes = 0;
      Index nextParamIndex = 0;
      bool canInlineByBytes = true;

      void visitExpression(Expression* curr) {
        size++;
        if (!canInlineByBytes) {
          return;
        }
        if (auto* get = curr->dynCast<LocalGet>()) {
          if (get->index == nextParamIndex) {
            nextParamIndex++;
          } else {
            canInlineByBytes = false;
          }
          return;
        }
        switch (curr->_id) {
          case Expression::LoopId:
          case Expression::TryId:
          case Expression::TryTableId:
          case Expression::ReturnId:
          case Expression::LocalSetId:
          case Expression::BreakId:
          case Expression::SwitchId:
          case Expression::BrOnId:
          case Expression::PopId:
            canInlineByBytes = false;
            return;
          case Expression::CallId: {
            auto* call = curr->cast<Call>();
            if (call->isReturn || call->target == funcName) {
              canInlineByBytes = false;
              return;
            }
            break;
          }
          case Expression::CallIndirectId:
            if (curr->cast<CallIndirect>()->isReturn) {
              canInlineByBytes = false;
              return;
            }
            break;
          case Expression::CallRefId:
            if (curr->cast<CallRef>()->isReturn) {
              canInlineByBytes = false;
              return;
            }
            break;
          default:
            break;
        }
        nonParamBodyBytes += getShallowBytes(curr);
      }
    } scanner;

    scanner.funcName = curr->name;
    scanner.canInlineByBytes = curr->vars.empty();
    scanner.walk(curr->body);

    info.size = scanner.size;
    if (scanner.canInlineByBytes &&
        scanner.nextParamIndex == curr->getNumParams()) {
      info.paramsUsedOnceInOrder = true;
      info.numParamGets = scanner.nextParamIndex;
      info.nonParamBodyBytes = scanner.nonParamBodyBytes;
    }
    info.toolchainInlineHint = curr->funcAnnotations.toolchainInline;

    // If the body is a simple instruction with roughly the same encoded size as
    // a `call` instruction, and arguments are function locals read in order,
    // then the code size always shrinks when the call is inlined.
    //
    // Note that skipping arguments can create `drop` instructions, and using
    // arguments multiple times can create new locals, at the call sites. So we
    // don't consider the function as "always shrinks" in these cases.
    // TODO: Consider allowing drops, as at least in traps-never-happen mode
    //       they can usually be removed.
    auto* body = curr->body;
    bool isDropCall = false;
    if (auto* drop = body->dynCast<Drop>()) {
      if (drop->value->is<Call>()) {
        body = drop->value;
        isDropCall = true;
      }
    }
    // Skip control flow as those can be substantially larger (middle and end
    // bytes in an If), or no situation exists where we can optimize them (a
    // Block with only LocalGets would have been removed by other passes).
    if (!Properties::isControlFlowStructure(body)) {
      bool shrinks = true;
      Index nextLocalGetIndex = 0;
      for (auto* operand : ChildIterator(body)) {
        if (auto* localGet = operand->dynCast<LocalGet>()) {
          if (localGet->index == nextLocalGetIndex) {
            nextLocalGetIndex++;
          } else {
            shrinks = false;
            break;
          }
        } else {
          shrinks = false;
          break;
        }
      }

      if (shrinks && nextLocalGetIndex == curr->getNumParams()) {
        info.trivialInstruction = TrivialInstruction::Shrinks;
        info.trivialDropCall = isDropCall;
        return;
      }

      // If the operands are trivial (size 1) like LocalGet or Const, we still
      // consider this as trivial instruction, but the size may not shrink when
      // inlined.
      uint32_t numOperands = ChildIterator(curr->body).children.size();
      if (info.size == numOperands + 1) {
        info.trivialInstruction = TrivialInstruction::MayNotShrink;
      }
    }
  }

private:
  NameInfoMap& infos;
  GlobalRefMap* globalRefs;
};

enum class ParamStrategy : uint8_t {
  DropSilent,
  DirectSingle,
  DirectDup,
  ReuseArgTee,
  ScratchTee,
  PrologueSet,
  PrologueReuseTee,
};

struct ByteBudgetSitePlan {
  bool valid = false;
  int siteDeltaBytes = 0;
  std::vector<ParamStrategy> paramStrategies;
};

struct RankTables {
  std::unordered_map<Name, Index> funcRank;
  std::unordered_map<Name, Index> globalRank;
  std::unordered_map<HeapType, Index> typeRank;
  std::unordered_set<HeapType> soleUseFuncTypes;

  Index getFuncRank(Name name) const {
    auto it = funcRank.find(name);
    return it != funcRank.end() ? it->second : 200;
  }

  Index getGlobalRank(Name name) const {
    auto it = globalRank.find(name);
    return it != globalRank.end() ? it->second : 200;
  }

  Index getTypeRank(HeapType ht) const {
    auto it = typeRank.find(ht);
    return it != typeRank.end() ? it->second : 200;
  }

  int getFuncIndexBytes(Name name) const {
    return FunctionInfoScanner::getULEBSize(getFuncRank(name));
  }

  int getGlobalIndexBytes(Name name) const {
    return FunctionInfoScanner::getULEBSize(getGlobalRank(name));
  }

  int getTypeIndexBytes(HeapType ht, bool isS33) const {
    if (ht.isBasic()) {
      return 1;
    }
    Index rank = getTypeRank(ht);
    if (isS33) {
      return FunctionInfoScanner::getSLEBSize(int64_t(rank));
    }
    return FunctionInfoScanner::getULEBSize(rank);
  }

  int getTypeBytes(Type t) const {
    if (t == Type::none || t == Type::unreachable) {
      return 0;
    }
    if (t.isTuple()) {
      int sum = 0;
      for (auto elem : t) {
        sum += getTypeBytes(elem);
      }
      return sum;
    }
    if (!t.isRef()) {
      return 1;
    }
    HeapType ht = t.getHeapType();
    if (ht.isBasic() && t.isNullable() && !t.isExact()) {
      return 1;
    }
    return 1 + getTypeIndexBytes(ht, /*isS33=*/true);
  }

  int getExactShallowBytes(Expression* curr) const {
    switch (curr->_id) {
      case Expression::BlockId: {
        auto* b = curr->cast<Block>();
        if (!b->name.is()) {
          return 0;
        }
        int bt = (b->type == Type::none || b->type == Type::unreachable)
                   ? 1
                   : getTypeBytes(b->type);
        return 2 + bt;
      }
      case Expression::IfId: {
        auto* iff = curr->cast<If>();
        int bt = (iff->type == Type::none || iff->type == Type::unreachable)
                   ? 1
                   : getTypeBytes(iff->type);
        return (iff->ifFalse ? 3 : 2) + bt;
      }
      case Expression::LoopId: {
        auto* loop = curr->cast<Loop>();
        int bt = (loop->type == Type::none || loop->type == Type::unreachable)
                   ? 1
                   : getTypeBytes(loop->type);
        return 2 + bt;
      }
      case Expression::BreakId:
        return 2;
      case Expression::CallId:
        return 1 + getFuncIndexBytes(curr->cast<Call>()->target);
      case Expression::CallRefId:
        return 2 + getTypeIndexBytes(
                     curr->cast<CallRef>()->target->type.getHeapType(), false);
      case Expression::CallIndirectId:
        return 2 + getTypeIndexBytes(curr->cast<CallIndirect>()->heapType,
                                     false);
      case Expression::GlobalGetId:
        return 1 + getGlobalIndexBytes(curr->cast<GlobalGet>()->name);
      case Expression::GlobalSetId:
        return 1 + getGlobalIndexBytes(curr->cast<GlobalSet>()->name);
      case Expression::RefFuncId:
        return 1 + getFuncIndexBytes(curr->cast<RefFunc>()->func);
      case Expression::StringConstId:
        return 3;
      case Expression::LocalGetId:
        return 1 +
               FunctionInfoScanner::getULEBSize(curr->cast<LocalGet>()->index);
      case Expression::LocalSetId:
        return 1 +
               FunctionInfoScanner::getULEBSize(curr->cast<LocalSet>()->index);
      case Expression::LoadId:
        return 2 +
               FunctionInfoScanner::getULEBSize(curr->cast<Load>()->offset.addr);
      case Expression::StoreId:
        return 2 + FunctionInfoScanner::getULEBSize(
                     curr->cast<Store>()->offset.addr);
      case Expression::ConstId: {
        auto* c = curr->cast<Const>();
        switch (c->type.getBasic()) {
          case Type::i32:
            return 1 + FunctionInfoScanner::getSLEBSize(c->value.geti32());
          case Type::i64:
            return 1 + FunctionInfoScanner::getSLEBSize(c->value.geti64());
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
      case Expression::UnaryId: {
        auto op = curr->cast<Unary>()->op;
        if (op == TruncSatSFloat32ToInt32 || op == TruncSatUFloat32ToInt32 ||
            op == TruncSatSFloat64ToInt32 || op == TruncSatUFloat64ToInt32 ||
            op == TruncSatSFloat32ToInt64 || op == TruncSatUFloat32ToInt64 ||
            op == TruncSatSFloat64ToInt64 || op == TruncSatUFloat64ToInt64) {
          return 2;
        }
        return 1;
      }
      case Expression::BinaryId:
      case Expression::DropId:
      case Expression::ReturnId:
      case Expression::NopId:
      case Expression::UnreachableId:
      case Expression::RefIsNullId:
      case Expression::RefEqId:
        return 1;
      case Expression::RefAsId: {
        auto op = curr->cast<RefAs>()->op;
        return (op == AnyConvertExtern || op == ExternConvertAny) ? 2 : 1;
      }
      case Expression::SelectId:
        return curr->type.isRef() ? 2 + getTypeBytes(curr->type) : 1;
      case Expression::RefNullId:
        return 1 + getTypeIndexBytes(curr->type.getHeapType(), /*isS33=*/true);
      case Expression::ThrowId:
      case Expression::RefI31Id:
      case Expression::I31GetId:
      case Expression::ArrayLenId:
        return 2;
      case Expression::StructNewId:
        return 2 + getTypeIndexBytes(curr->type.getHeapType(), false);
      case Expression::StructGetId: {
        auto* sg = curr->cast<StructGet>();
        return 2 + getTypeIndexBytes(sg->ref->type.getHeapType(), false) +
               FunctionInfoScanner::getULEBSize(sg->index);
      }
      case Expression::StructSetId: {
        auto* ss = curr->cast<StructSet>();
        return 2 + getTypeIndexBytes(ss->ref->type.getHeapType(), false) +
               FunctionInfoScanner::getULEBSize(ss->index);
      }
      case Expression::ArrayNewId:
        return 2 + getTypeIndexBytes(curr->type.getHeapType(), false);
      case Expression::ArrayGetId:
        return 2 + getTypeIndexBytes(
                     curr->cast<ArrayGet>()->ref->type.getHeapType(), false);
      case Expression::ArraySetId:
        return 2 + getTypeIndexBytes(
                     curr->cast<ArraySet>()->ref->type.getHeapType(), false);
      case Expression::ArrayFillId:
        return 2 + getTypeIndexBytes(
                     curr->cast<ArrayFill>()->ref->type.getHeapType(), false);
      case Expression::RefCastId:
        return 2 + getTypeIndexBytes(curr->type.getHeapType(), /*isS33=*/true);
      case Expression::RefTestId:
        return 2 + getTypeIndexBytes(curr->cast<RefTest>()->castType.getHeapType(),
                                     /*isS33=*/true);
      case Expression::ArrayNewFixedId: {
        auto* anf = curr->cast<ArrayNewFixed>();
        return 2 + getTypeIndexBytes(anf->type.getHeapType(), false) +
               FunctionInfoScanner::getULEBSize(anf->values.size());
      }
      case Expression::ArrayCopyId: {
        auto* ac = curr->cast<ArrayCopy>();
        return 2 + getTypeIndexBytes(ac->destRef->type.getHeapType(), false) +
               getTypeIndexBytes(ac->srcRef->type.getHeapType(), false);
      }
      case Expression::ArrayNewDataId:
      case Expression::ArrayNewElemId:
      case Expression::ArrayInitDataId:
      case Expression::ArrayInitElemId:
        return 6;
      default:
        return 2;
    }
  }

  int getTreeBytes(Expression* root) const {
    int total = 0;
    struct TreeBytesScanner
      : public PostWalker<TreeBytesScanner,
                          UnifiedExpressionVisitor<TreeBytesScanner>> {
      const RankTables& ranks;
      int& total;
      TreeBytesScanner(const RankTables& ranks, int& total)
        : ranks(ranks), total(total) {}
      void visitExpression(Expression* curr) {
        total += ranks.getExactShallowBytes(curr);
      }
    } scanner(*this, total);
    scanner.walk(root);
    return total;
  }
};

struct ByteBudgetFuncInfo {
  bool valid = false;
  int savedFuncBytes = 0;
  int nonParamExprBytes = 0;
  Index totalVarAccesses = 0;
  std::vector<Index> paramGetCount;
  std::vector<Index> firstReadOrd;
  std::vector<bool> paramAllowsNullBottom;
  std::vector<bool> paramNeedsExactHeapType;
  std::vector<bool> paramInBranch;
  std::vector<EffectAnalyzer> prefixEffectsBeforeFirstGet;
  std::vector<Index> varReads;
  std::vector<Index> varWrites;
};

struct InliningAction {
  Expression** callSite;
  Function* contents;
  bool insideATry;

  // An optional name hint can be provided, which will then be used in the name
  // of the block we put the inlined code in. Using a unique name hint in each
  // inlining can reduce the risk of name overlaps (which cause fixup work in
  // UniqueNameMapper::uniquify).
  Index nameHint = 0;

  // Whether to inline the entire function regardless of its inlining mode.
  // This is set for constant-argument call sites, see
  // FunctionInfo::worthConstantArgInlining.
  bool forceFull = false;

  bool useDirectInlining = false;
  ByteBudgetSitePlan directPlan;
  const ByteBudgetFuncInfo* bbFuncInfo = nullptr;

  InliningAction(Expression** callSite,
                 Function* contents,
                 bool insideATry,
                 Index nameHint = 0)
    : callSite(callSite), contents(contents), insideATry(insideATry),
      nameHint(nameHint) {}
};

struct InliningState {
  // Maps functions worth inlining to the mode with which we can inline them.
  std::unordered_map<Name, InliningMode> inlinableFunctions;
  // Functions that are worth fully inlining at call sites where all the
  // arguments are constants (these are not fully inlineable in general).
  std::unordered_set<Name> constantArgFunctions;
  // Immutable globals whose initializer is a constant, or an allocation of
  // constants (which has a fixed identity). A global.get of such a global
  // counts as a constant argument.
  std::unordered_set<Name> constantGlobals;
  // function name => actions that can be performed in it
  std::unordered_map<Name, std::vector<InliningAction>> actionsForFunction;
  // Byte-budget candidate analysis per function.
  std::unordered_map<Name, ByteBudgetFuncInfo> byteBudgetInfos;
  // Functions added to inlinableFunctions solely via byte-budget mode.
  std::unordered_set<Name> byteBudgetOnlyFunctions;
};

struct Planner : public WalkerPass<TryDepthWalker<Planner>> {
  bool isFunctionParallel() override { return true; }

  bool modifiesBinaryenIR() override { return false; }

  Planner(InliningState* state) : state(state) {}

  std::unique_ptr<Pass> create() override {
    return std::make_unique<Planner>(state);
  }

  void visitCall(Call* curr) {
    // plan to inline if we know this is valid to inline, and if the call is
    // actually performed - if it is dead code, it's pointless to inline.
    // we also cannot inline ourselves.
    bool isUnreachable;
    if (curr->isReturn) {
      // Tail calls are only actually unreachable if an argument is
      isUnreachable = std::any_of(
        curr->operands.begin(), curr->operands.end(), [](Expression* op) {
          return op->type == Type::unreachable;
        });
    } else {
      isUnreachable = curr->type == Type::unreachable;
    }
    if (isUnreachable || curr->target == getFunction()->name) {
      return;
    }
    bool forceFull = false;
    if (state->constantArgFunctions.contains(curr->target) &&
        hasConstantArgs(curr)) {
      forceFull = true;
    } else if (!state->inlinableFunctions.contains(curr->target)) {
      return;
    }
    // can't add a new element in parallel
    assert(state->actionsForFunction.contains(getFunction()->name));
    auto& actions = state->actionsForFunction[getFunction()->name];
    actions.emplace_back(
      getCurrentPointer(), getModule()->getFunction(curr->target), tryDepth > 0);
    actions.back().forceFull = forceFull;
  }

private:
  InliningState* state;

  bool isConstantArg(Expression* curr) {
    if (Properties::isSingleConstantExpression(curr)) {
      return true;
    }
    if (auto* get = curr->dynCast<GlobalGet>()) {
      return state->constantGlobals.contains(get->name);
    }
    if (auto* i31 = curr->dynCast<RefI31>()) {
      return i31->value->is<Const>();
    }
    return false;
  }

  bool hasConstantArgs(Call* curr) {
    // A call without arguments has nothing to specialize on.
    if (curr->operands.empty()) {
      return false;
    }
    for (auto* operand : curr->operands) {
      if (!isConstantArg(operand)) {
        return false;
      }
    }
    return true;
  }
};

struct Updater : public TryDepthWalker<Updater> {
  Module* module;
  std::map<Index, Index> localMapping;
  Name returnName;
  Type resultType;
  bool isReturn;
  Builder* builder;
  PassOptions& options;

  struct ReturnCallInfo {
    // The original `return_call` or `return_call_indirect` or `return_call_ref`
    // with its operands replaced with `local.get`s.
    Expression* call;
    // The branch that is serving as the "return" part of the original
    // `return_call`.
    Break* branch;
  };

  // Collect information on return_calls in the inlined body. Each will be
  // turned into branches out of the original inlined body followed by
  // non-return version of the original `return_call`, followed by a branch out
  // to the caller. The branch labels will be filled in at the end of the walk.
  std::vector<ReturnCallInfo> returnCallInfos;

  Updater(PassOptions& options) : options(options) {}

  void visitReturn(Return* curr) {
    replaceCurrent(builder->makeBreak(returnName, curr->value));
  }

  template<typename T> void handleReturnCall(T* curr, Signature sig) {
    if (isReturn || !curr->isReturn) {
      // If the inlined callsite was already a return_call, then we can keep
      // return_calls in the inlined function rather than downgrading them.
      // That is, if A->B and B->C and both those calls are return_calls
      // then after inlining A->B we want to now have A->C be a
      // return_call.
      return;
    }

    if (tryDepth == 0) {
      // Return calls in inlined functions should only break out of
      // the scope of the inlined code, not the entire function they
      // are being inlined into. To achieve this, make the call a
      // non-return call and add a break. This does not cause
      // unbounded stack growth because inlining and return calling
      // both avoid creating a new stack frame.
      curr->isReturn = false;
      curr->type = sig.results;
      // There might still be unreachable children causing this to be
      // unreachable.
      curr->finalize();
      if (sig.results.isConcrete()) {
        replaceCurrent(builder->makeBreak(returnName, curr));
      } else {
        replaceCurrent(builder->blockify(curr, builder->makeBreak(returnName)));
      }
    } else {
      // Set the children to locals as necessary, then add a branch out of the
      // inlined body. The branch label will be set later when we create branch
      // targets for the calls.
      Block* childBlock = ChildLocalizer(curr, getFunction(), *module, options)
                            .getChildrenReplacement();
      Break* branch = builder->makeBreak(Name());
      childBlock->list.push_back(branch);
      childBlock->type = Type::unreachable;
      replaceCurrent(childBlock);

      curr->isReturn = false;
      curr->type = sig.results;
      returnCallInfos.push_back({curr, branch});
    }
  }

  void visitCall(Call* curr) {
    handleReturnCall(curr, module->getFunction(curr->target)->getSig());
  }

  void visitCallIndirect(CallIndirect* curr) {
    handleReturnCall(curr, curr->heapType.getSignature());
  }

  void visitCallRef(CallRef* curr) {
    Type targetType = curr->target->type;
    if (!targetType.isSignature()) {
      // We don't know what type the call should return, but it will also never
      // be reached, so we don't need to do anything here.
      return;
    }
    handleReturnCall(curr, targetType.getHeapType().getSignature());
  }

  void visitLocalGet(LocalGet* curr) {
    curr->index = localMapping[curr->index];
  }

  void visitLocalSet(LocalSet* curr) {
    curr->index = localMapping[curr->index];
  }

  void walk(Expression*& curr) {
    PostWalker<Updater>::walk(curr);
    if (returnCallInfos.empty()) {
      return;
    }

    Block* body = builder->blockify(curr);
    curr = body;
    auto blockNames = BranchUtils::BranchAccumulator::get(body);

    for (Index i = 0; i < returnCallInfos.size(); ++i) {
      auto& info = returnCallInfos[i];

      // Add a block containing the previous body and a branch up to the caller.
      // Give the block a name that will allow this return_call's original
      // callsite to branch out of it then execute the call before returning to
      // the caller.
      auto name = Names::getValidName(
        "__return_call",
        [&](Name test) { return !blockNames.contains(test); },
        i);
      blockNames.insert(name);
      info.branch->name = name;
      Block* oldBody = builder->makeBlock(body->list, body->type);
      body->list.clear();

      if (resultType.isConcrete()) {
        body->list.push_back(builder->makeBlock(
          name, {builder->makeBreak(returnName, oldBody)}, Type::none));
      } else {
        oldBody->list.push_back(builder->makeBreak(returnName));
        oldBody->name = name;
        oldBody->type = Type::none;
        body->list.push_back(oldBody);
      }
      body->list.push_back(info.call);
      body->finalize(resultType);
    }
  }
};

// Core inlining logic. Modifies the outside function (adding locals as
// needed) by copying the inlined code into it.
static void doCodeInlining(Module* module,
                           Function* into,
                           const InliningAction& action,
                           PassOptions& options) {
  Function* from = action.contents;
  auto* call = (*action.callSite)->cast<Call>();

  // Works for return_call, too
  Type retType = module->getFunction(call->target)->getResults();

  // Build the block that will contain the inlined contents.
  Builder builder(*module);
  auto* block = builder.makeBlock();
  auto name = std::string("__inlined_func$") + from->name.toString();
  if (action.nameHint) {
    name += '$' + std::to_string(action.nameHint);
  }
  block->name = Name(name);

  // In the unlikely event that the function already has a branch target with
  // this name, fix that up, as otherwise we can get unexpected capture of our
  // branches, that is, we could end up with this:
  //
  //  (block $X             ;; a new block we add as the target of returns
  //    (from's contents
  //      (block $X         ;; a block in from's contents with a colliding name
  //        (br $X          ;; a new br we just added that replaces a return
  //
  // Here the br wants to go to the very outermost block, to represent a
  // return from the inlined function's code, but it ends up captured by an
  // internal block. We also need to be careful of the call's children:
  //
  //  (block $X             ;; a new block we add as the target of returns
  //    (local.set $param
  //      (call's first parameter
  //        (br $X)         ;; nested br in call's first parameter
  //      )
  //    )
  //
  // (In this case we could use a second block and define the named block $X
  // after the call's parameters, but that adds work for an extremely rare
  // situation.) The latter case does not apply if the call is a
  // return_call inside a try, because in that case the call's
  // children do not appear inside the same block as the inlined body.
  bool hoistCall = call->isReturn && action.insideATry;
  if (BranchUtils::hasBranchTarget(from->body, block->name) ||
      (!hoistCall && BranchUtils::BranchSeeker::has(call, block->name))) {
    auto fromNames = BranchUtils::getBranchTargets(from->body);
    auto callNames = hoistCall ? BranchUtils::NameSet{}
                               : BranchUtils::BranchAccumulator::get(call);
    block->name = Names::getValidName(block->name, [&](Name test) {
      return !fromNames.contains(test) && !callNames.contains(test);
    });
  }

  // Prepare to update the inlined code's locals and other things.
  Updater updater(options);
  updater.setFunction(into);
  updater.module = module;
  updater.resultType = from->getResults();
  updater.returnName = block->name;
  updater.isReturn = call->isReturn;
  updater.builder = &builder;
  // Set up a locals mapping
  for (Index i = 0; i < from->getNumLocals(); i++) {
    updater.localMapping[i] = builder.addVar(into, from->getLocalType(i));
  }

  if (hoistCall) {
    // Wrap the existing function body in a block we can branch out of before
    // entering the inlined function body. This block must have a name that is
    // different from any other block name above the branch.
    auto intoNames = BranchUtils::BranchAccumulator::get(into->body);
    auto bodyName =
      Names::getValidName(Name("__original_body"),
                          [&](Name test) { return !intoNames.contains(test); });
    if (retType.isConcrete()) {
      into->body = builder.makeBlock(
        bodyName, {builder.makeReturn(into->body)}, Type::none);
    } else {
      into->body = builder.makeBlock(
        bodyName, {into->body, builder.makeReturn()}, Type::none);
    }

    // Sequence the inlined function body after the original caller body.
    into->body = builder.makeSequence(into->body, block, retType);

    // Replace the original callsite with an expression that assigns the
    // operands into the params and branches out of the original body.
    auto numParams = from->getParams().size();
    if (numParams) {
      auto* branchBlock = builder.makeBlock();
      for (Index i = 0; i < numParams; i++) {
        branchBlock->list.push_back(
          builder.makeLocalSet(updater.localMapping[i], call->operands[i]));
      }
      branchBlock->list.push_back(builder.makeBreak(bodyName));
      branchBlock->finalize(Type::unreachable);
      *action.callSite = branchBlock;
    } else {
      *action.callSite = builder.makeBreak(bodyName);
    }
  } else {
    // Assign the operands into the params
    for (Index i = 0; i < from->getParams().size(); i++) {
      block->list.push_back(
        builder.makeLocalSet(updater.localMapping[i], call->operands[i]));
    }
    // Zero out the vars (as we may be in a loop, and may depend on their
    // zero-init value
    for (Index i = 0; i < from->vars.size(); i++) {
      auto type = from->vars[i];
      if (!LiteralUtils::canMakeZero(type)) {
        // Non-zeroable locals do not need to be zeroed out. As they have no
        // zero value they by definition should not be used before being written
        // to, so any value we set here would not be observed anyhow.
        continue;
      }
      block->list.push_back(
        builder.makeLocalSet(updater.localMapping[from->getVarIndexBase() + i],
                             LiteralUtils::makeZero(type, *module)));
    }
    if (call->isReturn) {
      assert(!action.insideATry);
      if (retType.isConcrete()) {
        *action.callSite = builder.makeReturn(block);
      } else {
        *action.callSite = builder.makeSequence(block, builder.makeReturn());
      }
    } else {
      *action.callSite = block;
    }
  }

  // Generate and update the inlined contents
  auto* contents = ExpressionManipulator::copy(from->body, *module);
  metadata::copyBetweenFunctions(from->body, contents, from, into);
  updater.walk(contents);
  block->list.push_back(contents);
  block->type = retType;

  // The ReFinalize below will handle propagating unreachability if we need to
  // do so, that is, if the call was reachable but now the inlined content we
  // replaced it with was unreachable. The opposite case requires special
  // handling: ReFinalize works under the assumption that code can become
  // unreachable, but it does not go back from that state. But inlining can
  // cause that:
  //
  //  (call $A                               ;; an unreachable call
  //    (unreachable)
  //  )
  // =>
  //  (block $__inlined_A_body (result i32)  ;; reachable code after inlining
  //    (unreachable)
  //  )
  //
  // That is, if the called function wraps the input parameter in a block with a
  // declared type, then the block is not unreachable. And then we might error
  // if the outside expects the code to be unreachable - perhaps it only
  // validates that way. To fix this, if the call was unreachable then we make
  // the inlined code unreachable as well. That also maximizes DCE
  // opportunities by propagating unreachability as much as possible.
  //
  // (Note that we don't need to do this for a return_call, which is always
  // unreachable anyhow.)
  if (call->type == Type::unreachable && !call->isReturn) {
    // Make the replacement code unreachable. Note that we can't just add an
    // unreachable at the end, as the block might have breaks to it (returns are
    // transformed into those).
    Expression* old = block;
    if (old->type.isConcrete()) {
      old = builder.makeDrop(old);
    }
    *action.callSite = builder.makeSequence(old, builder.makeUnreachable());
  }
}

using ScratchLocalPool = std::unordered_map<Type, std::vector<Index>>;

static void doDirectInlining(Module* module,
                             Function* into,
                             const InliningAction& action,
                             ScratchLocalPool& scratchPool) {
  Function* from = action.contents;
  auto* call = (*action.callSite)->cast<Call>();
  const auto& plan = action.directPlan;
  const auto& bbInfo = *action.bbFuncInfo;

  Builder builder(*module);

  std::unordered_set<Index> busyLocals;
  for (auto* lg : FindAll<LocalGet>(call).list) {
    busyLocals.insert(lg->index);
  }
  for (auto* ls : FindAll<LocalSet>(call).list) {
    busyLocals.insert(ls->index);
  }

  std::unordered_set<Index> claimedByThisCall;
  auto getScratchLocal = [&](Type t) -> Index {
    auto& vec = scratchPool[t];
    for (Index idx : vec) {
      if (!busyLocals.contains(idx) && !claimedByThisCall.contains(idx)) {
        claimedByThisCall.insert(idx);
        return idx;
      }
    }
    Index idx = builder.addVar(into, t);
    vec.push_back(idx);
    claimedByThisCall.insert(idx);
    return idx;
  };

  Index numParams = from->getNumParams();
  std::vector<Index> varLocalMap(from->vars.size(), Index(-1));
  for (Index v = 0; v < from->vars.size(); ++v) {
    if (bbInfo.varReads[v] > 0 || bbInfo.varWrites[v] > 0) {
      varLocalMap[v] = getScratchLocal(from->vars[v]);
    }
  }

  std::vector<Index> paramScratchLocals(numParams, Index(-1));
  for (Index p = 0; p < numParams; ++p) {
    Expression* arg = call->operands[p];
    Type paramType = from->getLocalType(p);
    if (arg->is<RefNull>() && !bbInfo.paramAllowsNullBottom[p] &&
        arg->type != paramType) {
      arg->type = Type(paramType.getHeapType(), Nullable);
    }
    if (plan.paramStrategies[p] == ParamStrategy::ScratchTee ||
        plan.paramStrategies[p] == ParamStrategy::PrologueSet) {
      paramScratchLocals[p] = getScratchLocal(paramType);
    } else if (plan.paramStrategies[p] == ParamStrategy::DirectSingle &&
               bbInfo.paramNeedsExactHeapType[p] && !arg->is<RefNull>() &&
               arg->type.getHeapType() != paramType.getHeapType()) {
      paramScratchLocals[p] = getScratchLocal(paramType);
    }
  }

  auto* contents = ExpressionManipulator::copy(from->body, *module);
  metadata::copyBetweenFunctions(from->body, contents, from, into);

  struct DirectUpdater : public PostWalker<DirectUpdater> {
    Module* module;
    Function* into;
    Function* from;
    Call* call;
    Builder& builder;
    const ByteBudgetSitePlan& plan;
    const std::vector<Index>& paramScratchLocals;
    const std::vector<Index>& varLocalMap;
    std::vector<Index> seenParamGets;

    DirectUpdater(Module* module,
                  Function* into,
                  Function* from,
                  Call* call,
                  Builder& builder,
                  const ByteBudgetSitePlan& plan,
                  const std::vector<Index>& paramScratchLocals,
                  const std::vector<Index>& varLocalMap)
      : module(module), into(into), from(from), call(call), builder(builder),
        plan(plan), paramScratchLocals(paramScratchLocals),
        varLocalMap(varLocalMap), seenParamGets(from->getNumParams(), 0) {}

    void visitLocalGet(LocalGet* curr) {
      Index numParams = from->getNumParams();
      if (curr->index >= numParams) {
        Index v = curr->index - numParams;
        assert(varLocalMap[v] != Index(-1));
        curr->index = varLocalMap[v];
        return;
      }

      Index p = curr->index;
      Index seen = seenParamGets[p]++;
      Expression* arg = call->operands[p];
      Type paramType = from->getLocalType(p);

      switch (plan.paramStrategies[p]) {
        case ParamStrategy::DirectSingle: {
          assert(seen == 0);
          if (paramScratchLocals[p] != Index(-1)) {
            replaceCurrent(
              builder.makeLocalTee(paramScratchLocals[p], arg, paramType));
          } else {
            replaceCurrent(arg);
          }
          return;
        }
        case ParamStrategy::DirectDup: {
          if (seen == 0) {
            replaceCurrent(arg);
          } else {
            replaceCurrent(ExpressionManipulator::copy(arg, *module));
          }
          return;
        }
        case ParamStrategy::ReuseArgTee: {
          if (seen == 0) {
            replaceCurrent(arg);
          } else {
            Index teeLocal = arg->cast<LocalSet>()->index;
            replaceCurrent(
              builder.makeLocalGet(teeLocal, into->getLocalType(teeLocal)));
          }
          return;
        }
        case ParamStrategy::ScratchTee: {
          Index tmp = paramScratchLocals[p];
          assert(tmp != Index(-1));
          if (seen == 0) {
            replaceCurrent(builder.makeLocalTee(tmp, arg, paramType));
          } else {
            replaceCurrent(builder.makeLocalGet(tmp, paramType));
          }
          return;
        }
        case ParamStrategy::PrologueSet: {
          Index tmp = paramScratchLocals[p];
          assert(tmp != Index(-1));
          replaceCurrent(builder.makeLocalGet(tmp, paramType));
          return;
        }
        case ParamStrategy::PrologueReuseTee: {
          Index teeLocal = arg->cast<LocalSet>()->index;
          replaceCurrent(
            builder.makeLocalGet(teeLocal, into->getLocalType(teeLocal)));
          return;
        }
        case ParamStrategy::DropSilent:
          WASM_UNREACHABLE("unexpected get of silent-dropped param");
      }
    }

    void visitLocalSet(LocalSet* curr) {
      Index numParams = from->getNumParams();
      assert(curr->index >= numParams);
      Index v = curr->index - numParams;
      assert(varLocalMap[v] != Index(-1));
      curr->index = varLocalMap[v];
    }
  } updater(module,
            into,
            from,
            call,
            builder,
            plan,
            paramScratchLocals,
            varLocalMap);

  updater.walk(contents);
  std::vector<Expression*> prologue;
  for (Index p = 0; p < numParams; ++p) {
    if (plan.paramStrategies[p] == ParamStrategy::PrologueSet) {
      prologue.push_back(
        builder.makeLocalSet(paramScratchLocals[p], call->operands[p]));
    } else if (plan.paramStrategies[p] == ParamStrategy::PrologueReuseTee) {
      auto* ls = call->operands[p]->cast<LocalSet>();
      prologue.push_back(builder.makeLocalSet(ls->index, ls->value));
    }
  }
  if (!prologue.empty()) {
    if (auto* b = contents->dynCast<Block>(); b && !b->name.is()) {
      for (auto* item : b->list) {
        prologue.push_back(item);
      }
      b->list.set(prologue);
    } else {
      prologue.push_back(contents);
      contents = builder.makeBlock(prologue, contents->type);
    }
  }
  *action.callSite = contents;
}

// Updates the outer function after we inline into it. This is a general
// operation that does not depend on what we inlined, it just makes sure that we
// refinalize everything, have no duplicate break labels, etc.
static void updateAfterInlining(Module* module, Function* into) {
  // Anything we inlined into may now have non-unique label names, fix it up.
  // Note that we must do this before refinalization, as otherwise duplicate
  // block labels can lead to errors (the IR must be valid before we
  // refinalize).
  wasm::UniqueNameMapper::uniquify(into->body);
  // Inlining unreachable contents can make things in the function we inlined
  // into unreachable.
  ReFinalize().walkFunctionInModule(into, module);
  // New locals we added may require fixups for nondefaultability. We do this
  // here and not in the main pass (or its subpasses) so that we only do it
  // where needed.
  TypeUpdating::handleNonDefaultableLocals(into, *module);
}

static void doInlining(Module* module,
                       Function* into,
                       const InliningAction& action,
                       PassOptions& options) {
  doCodeInlining(module, into, action, options);
  updateAfterInlining(module, into);
}

// A map of function names to the inlining actions we've decided to actually
// perform in them.
using ChosenActions = std::unordered_map<Name, std::vector<InliningAction>>;

// A pass that calls doInlining() on a bunch of actions that were chosen to
// perform.
struct DoInlining : public Pass {
  bool isFunctionParallel() override { return true; }

  // We do this only where we inline, inside updateAfterInlining().
  bool requiresNonNullableLocalFixups() override { return false; }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<DoInlining>(chosenActions);
  }

  DoInlining(const ChosenActions& chosenActions)
    : chosenActions(chosenActions) {}

  void runOnFunction(Module* module, Function* func) override {
    auto iter = chosenActions.find(func->name);
    // We must be called on a function that we actually want to inline into.
    assert(iter != chosenActions.end());
    const auto& actions = iter->second;
    assert(!actions.empty());

    ScratchLocalPool scratchPool;
    // Inline all the code first, then update func once at the end (which saves
    // e.g. running ReFinalize after each action, of which there might be many).
    for (const auto& action : actions) {
      if (action.useDirectInlining) {
        doDirectInlining(module, func, action, scratchPool);
      } else {
        doCodeInlining(module, func, action, getPassOptions());
      }
    }
    updateAfterInlining(module, func);
  }

private:
  const ChosenActions& chosenActions;
};

//
// Function splitting / partial inlining / inlining of conditions.
//
// A function may be too costly to inline, but it may be profitable to
// *partially* inline it. The specific cases optimized here are functions with a
// condition,
//
//  function foo(x) {
//    if (x) return;
//    ..lots and lots of other code..
//  }
//
// If the other code after the if is large enough or costly enough, then we will
// not inline the entire function. But it is useful to inline the condition.
// Consider this caller:
//
//  function caller(x) {
//    foo(0);
//    foo(x);
//  }
//
// If we inline the condition, we end up with this:
//
//  function caller(x) {
//    if (0) foo(0);
//    if (x) foo(x);
//  }
//
// By inlining the condition out of foo() we gain two benefits:
//
//  * In the first call here the condition is zero, which means we can
//    statically optimize out the call entirely.
//  * Even if we can't do that (as in the second call) if at runtime we see the
//    condition is false then we avoid the call. That means we perform what is
//    hopefully a cheap branch instead of a call and then a branch.
//
// The cost to doing this is an increase in code size, and so this is only done
// when optimizing heavily for speed.
//
// To implement partial inlining we split the function to be inlined. Starting
// with
//
//  function foo(x) {
//    if (x) return;
//    ..lots and lots of other code..
//  }
//
// We create the "inlineable" part of it, and the code that is "outlined":
//
//  function foo$inlineable(x) {
//    if (x) return;
//    foo$outlined(x);
//  }
//  function foo$outlined(x) {
//    ..lots and lots of other code..
//  }
//
// (Note how if the second function were inlined into the first, we would end
// up where we started, with the original function.) After splitting the
// function in this way, we simply inline the inlineable part using the normal
// mechanism for that. That ends up replacing  foo(x);  with
//
//  if (x) foo$outlined(x);
//
// which is what we wanted.
//
// To reduce the complexity of this feature, it is implemented almost entirely
// in its own class, FunctionSplitter. The main inlining logic just calls out to
// the splitter to check if a function is worth splitting, and to get the split
// part if so.
//

struct FunctionSplitter {
  Module* module;
  PassOptions& options;

  FunctionSplitter(Module* module, PassOptions& options)
    : module(module), options(options) {}

  // Check if an function could be split in order to at least inline part of it,
  // in a worthwhile manner.
  //
  // Even if this returns a split inlining mode, we may not end up inlining the
  // function, as the main inlining logic has a few other considerations to take
  // into account (like limitations on which functions can be inlined into in
  // each iteration, the number of iterations, etc.). Therefore this function
  // may only find out if we *can* split, but not actually do any splitting.
  //
  // Note that to avoid wasteful work, this function may return "Full" inlining
  // mode instead of a split inlining. That is, if it detects that a partial
  // inlining will trigger a follow up full inline of the split function then it
  // will instead return "InliningMode::Full" directly. In more detail, imagine
  // we have
  //
  //   foo(10);
  //
  // which calls
  //
  //   func foo(x) {
  //     if (x) {
  //       CODE
  //     }
  //   }
  //
  // If we partially inline then the call becomes
  //
  //   if (10) {
  //     outlined-CODE()
  //   }
  //
  // That is, we've only inlined the |if| out of foo(), and we call a function
  // that contains the code in CODE. But if CODE is small enough to be inlined
  // then a later iteration of the inliner will do just that. The result of this
  // split inlining and then full inlining of the outlined code is exactly the
  // same as if we normally inlined from the beginning, but it adds more work,
  // so we'd like to avoid it, which we do by seeing when a split would be
  // followed by inlining the remainder, and then we return Full here and just
  // inline it all normally.
  //
  // Note that the above issue shows that we may in some cases inline more than
  // the normal inlining limit. That is, if the inlining limit is 20, and CODE
  // in the example above was 19, then foo()'s size was 21 (when we add the if
  // and get of |x|). That leads to the situation where foo() is too big to
  // normally inline, but the outlined CODE can then be inlined. And this shows
  // that we end up inlining a function of size 21, even though it is larger
  // than our inlining limit. We consider this acceptable because it only
  // occurs on functions slightly larger than the inlining limit, and only if
  // they have the simple forms that split inlining recognizes, that we think
  // are useful to inline. Alternatively, if we wanted to avoid this, it would
  // add complexity because we'd need to recognize the outlined function with
  // CODE in it and *not* inline it even though it is small enough, after
  // splitting, to be inlined. That is, splitting creates smaller functions, so
  // it can lead to more inlining (but, again, that makes sense since we only
  // split on very specific patterns we believe are worth handling in that
  // manner).
  InliningMode getSplitDrivenInliningMode(Function* func, FunctionInfo& info) {
    const Index MaxIfs = options.inlining.partialInliningIfs;
    assert(MaxIfs > 0);

    auto* body = func->body;

    // If the body is a block, and we have breaks to that block, then we cannot
    // outline any code - we can't outline a break without the break's target.
    if (auto* block = body->dynCast<Block>()) {
      if (BranchUtils::BranchSeeker::has(block, block->name)) {
        return InliningMode::Uninlineable;
      }
    }

    // All the patterns we look for right now start with an if at the very top
    // of the function.
    auto* iff = getIf(body);
    if (!iff) {
      return InliningMode::Uninlineable;
    }

    // If the condition is not very simple, the benefits of this optimization
    // are not obvious.
    if (!isSimple(iff->condition)) {
      return InliningMode::Uninlineable;
    }

    // Pattern A: Check if the function begins with
    //
    //  if (simple) return;
    //
    // TODO: support a return value
    if (!iff->ifFalse && func->getResults() == Type::none &&
        iff->ifTrue->is<Return>()) {
      // The body must be a block, because if it were not then the function
      // would be easily inlineable (just an if with a simple condition and a
      // return), and we would not even attempt to do splitting.
      assert(body->is<Block>());

      auto outlinedFunctionSize = info.size - Measurer::measure(iff);
      // If outlined function will be worth normal inline, skip the intermediate
      // state and inline fully now. Note that if full inlining is disabled we
      // will not do this, and instead inline partially.
      if (!func->noFullInline &&
          outlinedFunctionWorthInlining(info, outlinedFunctionSize)) {
        return InliningMode::Full;
      }

      return InliningMode::SplitPatternA;
    }

    // Pattern B: Represents a function whose entire body looks like
    //
    //  if (A_1) {
    //    ..heavy work..
    //  }
    //  ..
    //  if (A_k) {
    //    ..heavy work..
    //  }
    //  B; // an optional final value (which can be a return value)
    //
    // where there is a small number of such ifs with arguments A1..A_k, and
    // A_1..A_k and B (if the final value B exists) are very simple.
    //
    // Also, each if body must either be unreachable, or it must have type none
    // and have no returns. If it is unreachable, for example because it is a
    // return, then we will just return a value in the inlineable function:
    //
    //  if (A_i) {
    //    return outlined(..);
    //  }
    //
    // Or, if an if body has type none, then for now we assume that we do not
    // need to return a value from there, which makes things simpler, and in
    // that case we just do this, which continues onward in the function:
    //
    //  if (A_i) {
    //    outlined(..);
    //  }
    //
    // TODO: handle a possible returned value in this case as well.
    //
    // Note that the if body type must be unreachable or none, as this is an if
    // without an else.

    // Find the number of ifs.
    Index numIfs = 0;
    while (getIf(body, numIfs) && numIfs <= MaxIfs) {
      numIfs++;
    }
    if (numIfs == 0 || numIfs > MaxIfs) {
      return InliningMode::Uninlineable;
    }

    // Look for a final item after the ifs.
    auto* finalItem = getItem(body, numIfs);

    // The final item must be simple (or not exist, which is simple enough).
    if (finalItem && !isSimple(finalItem)) {
      return InliningMode::Uninlineable;
    }

    // There must be no other items after the optional final one.
    if (finalItem && getItem(body, numIfs + 1)) {
      return InliningMode::Uninlineable;
    }
    // This has the general shape we seek. Check each if: it must be in the
    // form mentioned above (simple condition, no returns in body). We must also
    // have no sets of locals that the final item notices, as then we could
    // have this:
    //
    //  if (A) {
    //    x = 10;
    //  }
    //  return x;
    //
    // We cannot split out the if in such a case because of the local
    // dependency.
    std::unordered_set<Index> writtenLocals;
    for (Index i = 0; i < numIfs; i++) {
      auto* iff = getIf(body, i);
      // The if must have a simple condition and no else arm.
      if (!isSimple(iff->condition) || iff->ifFalse) {
        return InliningMode::Uninlineable;
      }
      if (iff->ifTrue->type == Type::none) {
        // This must have no returns.
        if (ReturnUtils::getInfo(iff->ifTrue).hasReturn) {
          return InliningMode::Uninlineable;
        }
      } else {
        // This is an if without an else, and so the type is either none or
        // unreachable, and we ruled out none before.
        assert(iff->ifTrue->type == Type::unreachable);
      }
      if (finalItem) {
        for (auto* set : FindAll<LocalSet>(iff).list) {
          writtenLocals.insert(set->index);
        }
      }
    }
    // Finish the locals check mentioned above.
    if (finalItem) {
      for (auto* get : FindAll<LocalGet>(finalItem).list) {
        if (writtenLocals.contains(get->index)) {
          return InliningMode::Uninlineable;
        }
      }
    }

    // Success, this matches the pattern.

    // If the outlined function will be worth inlining normally, skip the
    // intermediate state and inline fully now. (As above, if full inlining is
    // disabled, we only partially inline.)
    if (numIfs == 1) {
      auto outlinedFunctionSize = Measurer::measure(iff->ifTrue);
      if (!func->noFullInline &&
          outlinedFunctionWorthInlining(info, outlinedFunctionSize)) {
        return InliningMode::Full;
      }
    }

    return InliningMode::SplitPatternB;
  }

  // Returns the function we should inline, after we split the function into two
  // pieces as described above (that is, in the example above, this would return
  // foo$inlineable).
  //
  // This is called when we are definitely inlining the function, and so it will
  // perform the splitting (if that has not already been done before).
  Function* getInlineableSplitFunction(Function* func,
                                       InliningMode inliningMode) {
    assert(inliningMode == InliningMode::SplitPatternA ||
           inliningMode == InliningMode::SplitPatternB);
    auto& split = splits[func->name];

    if (!split.inlineable) {
      // We haven't performed the split, do it now.
      split.inlineable = doSplit(func, inliningMode);
    }

    return split.inlineable;
  }

  // Clean up. When we are done we no longer need the inlineable functions on
  // the module, as they have been inlined into all the places we wanted them
  // for.
  //
  // Returns a list of the names of the functions we split.
  std::vector<Name> finish() {
    std::vector<Name> ret;
    std::unordered_set<Name> inlineableNames;
    for (auto& [func, split] : splits) {
      auto* inlineable = split.inlineable;
      if (inlineable) {
        inlineableNames.insert(inlineable->name);
        ret.push_back(func);
      }
    }
    module->removeFunctions([&](Function* func) {
      return inlineableNames.find(func->name) != inlineableNames.end();
    });
    return ret;
  }

private:
  // Information about splitting a function.
  struct Split {
    // The inlineable function out of the two that we generate by splitting.
    // That is, foo$inlineable from above.
    Function* inlineable = nullptr;

    // The outlined function, that is, foo$outlined from above.
    Function* outlined = nullptr;
  };

  // All the splitting we have already performed.
  //
  // Note that this maps from function names, and not Function*, as the main
  // inlining code can remove functions as it goes, but we can rely on names
  // staying constant.
  std::unordered_map<Name, Split> splits;

  bool outlinedFunctionWorthInlining(FunctionInfo& origin, Index sizeEstimate) {
    FunctionInfo info;
    // Start with a copy of the origin's info, and apply the size estimate.
    // This is not accurate, for example the origin function may have
    // loop or calls even though this section may not have.
    // This is a conservative estimate, that is, it will return true only when
    // it should, but might return false when a more precise analysis would
    // return true. And it is a practical estimation to avoid extra future work.
    info = origin;
    info.size = sizeEstimate;
    info.paramsUsedOnceInOrder = false;
    return info.worthFullInlining(options);
  }

  Function* doSplit(Function* func, InliningMode inliningMode) {
    Builder builder(*module);

    if (inliningMode == InliningMode::SplitPatternA) {
      // Note that "A" in the name here identifies this as being a split from
      // pattern A. The second pattern B will have B in the name.
      Function* inlineable = copyFunction(func, "inlineable-A");
      auto* outlined = copyFunction(func, "outlined-A");

      // The inlineable function should only have the if, which will call the
      // outlined function with a flipped condition.
      auto* inlineableIf = getIf(inlineable->body);
      inlineableIf->condition =
        builder.makeUnary(EqZInt32, inlineableIf->condition);
      BranchHints::flip(inlineableIf, inlineable);
      inlineableIf->ifTrue = builder.makeCall(
        outlined->name, getForwardedArgs(func, builder), Type::none);
      inlineable->body = inlineableIf;

      // The outlined function no longer needs the initial if.
      auto& outlinedList = outlined->body->cast<Block>()->list;
      outlinedList.erase(outlinedList.begin());

      return inlineable;
    }

    assert(inliningMode == InliningMode::SplitPatternB);

    Function* inlineable = copyFunction(func, "inlineable-B");

    const Index MaxIfs = options.inlining.partialInliningIfs;
    assert(MaxIfs > 0);

    // The inlineable function should only have the ifs, which will call the
    // outlined heavy work.
    for (Index i = 0; i < MaxIfs; i++) {
      // For each if, create an outlined function with the body of that if,
      // and call that from the if.
      auto* inlineableIf = getIf(inlineable->body, i);
      if (!inlineableIf) {
        break;
      }
      auto* outlined = copyFunction(func, "outlined-B");
      outlined->body = inlineableIf->ifTrue;

      // The outlined function either returns the same results as the original
      // one, or nothing, depending on if a value is returned here.
      auto valueReturned =
        func->getResults() != Type::none && outlined->body->type != Type::none;
      outlined->setResults(valueReturned ? func->getResults() : Type::none);
      inlineableIf->ifTrue = builder.makeCall(outlined->name,
                                              getForwardedArgs(func, builder),
                                              outlined->getResults());
      if (valueReturned) {
        inlineableIf->ifTrue = builder.makeReturn(inlineableIf->ifTrue);
      }
    }

    return inlineable;
  }

  Function* copyFunction(Function* func, std::string prefix) {
    // TODO: We copy quite a lot more than we need here, and throw stuff out.
    //       It is simple to just copy the entire thing to get the params and
    //       results and all that, but we could be more efficient.
    prefix = "byn-split-" + prefix;
    return ModuleUtils::copyFunction(
      func,
      *module,
      Names::getValidFunctionName(*module,
                                  prefix + '$' + func->name.toString()));
  }

  // Get the i-th item in a sequence of initial items in an expression. That is,
  // if the item is a block, it may have several such items, and otherwise there
  // is a single item, that item itself. This basically provides a simpler
  // interface than checking if something is a block or not when there is just
  // one item.
  //
  // Returns nullptr if there is no such item.
  static Expression* getItem(Expression* curr, Index i = 0) {
    if (auto* block = curr->dynCast<Block>()) {
      auto& list = block->list;
      if (i < list.size()) {
        return list[i];
      }
    }
    if (i == 0) {
      return curr;
    }
    return nullptr;
  }

  // Get the i-th if in a sequence of initial ifs in an expression. If no such
  // if exists, returns nullptr.
  static If* getIf(Expression* curr, Index i = 0) {
    auto* item = getItem(curr, i);
    if (!item) {
      return nullptr;
    }
    if (auto* iff = item->dynCast<If>()) {
      return iff;
    }
    return nullptr;
  }

  // Checks if an expression is very simple - something simple enough that we
  // are willing to inline it in this optimization. This should basically take
  // almost no cost at all to compute.
  bool isSimple(Expression* curr) {
    // For now, support local and global gets, and unary operations.
    // TODO: Generalize? Use costs.h?
    if (curr->type == Type::unreachable) {
      return false;
    }
    if (curr->is<GlobalGet>() || curr->is<LocalGet>()) {
      return true;
    }
    if (auto* unary = curr->dynCast<Unary>()) {
      return isSimple(unary->value);
    }
    if (auto* is = curr->dynCast<RefIsNull>()) {
      return isSimple(is->value);
    }
    return false;
  }

  // Returns a list of local.gets, one for each of the parameters to the
  // function. This forwards the arguments passed to the inlineable function to
  // the outlined one.
  std::vector<Expression*> getForwardedArgs(Function* func, Builder& builder) {
    std::vector<Expression*> args;
    for (Index i = 0; i < func->getNumParams(); i++) {
      args.push_back(builder.makeLocalGet(i, func->getLocalType(i)));
    }
    return args;
  }
};

struct Inlining : public Pass {
  // This pass changes locals and parameters.
  // FIXME DWARF updating does not handle local changes yet.
  bool invalidatesDWARF() override { return true; }

  // We do this only where we inline, inside updateAfterInlining().
  bool requiresNonNullableLocalFixups() override { return false; }

  // whether to optimize where we inline
  bool optimize = false;

  // the information for each function. recomputed in each interaction
  NameInfoMap infos;
  GlobalRefMap globalRefs;

  std::unique_ptr<FunctionSplitter> functionSplitter;

  Module* module = nullptr;

  bool isByteBudgetMode() {
    if (hasArgument("inlining-byte-budget")) {
      auto val = getArgument("inlining-byte-budget", "");
      return val.empty() || val != "0";
    }
    if (hasArgument("inlining") &&
        getArgument("inlining", "") == "byte-budget") {
      return true;
    }
    if (hasArgument("inlining-optimizing") &&
        getArgument("inlining-optimizing", "") == "byte-budget") {
      return true;
    }
    return false;
  }

  void run(Module* module_) override {
    module = module_;

    // No point to do more iterations than the number of functions, as it means
    // we are infinitely recursing (which should be very rare in practice, but
    // it is possible that a recursive call can look like it is worth inlining).
    Index iterationNumber = 0;

    auto numOriginalFunctions = module->functions.size();

    // Track in how many iterations a function was inlined into. We are willing
    // to inline many times into a function within an iteration, as e.g. that
    // helps the case of many calls of a small getter. However, if we only do
    // more inlining in separate iterations then it is likely code that was the
    // result of previous inlinings that is now being inlined into. That is, an
    // old inlining added a call to somewhere, and now we are inlining into that
    // call. This is typically recursion, which to some extent can help, but
    // then like loop unrolling it loses its benefit quickly, so set a limit
    // here.
    //
    // In addition to inlining into a function, we track how many times we do
    // other potentially repetitive operations like splitting a function before
    // inlining, as any such repetitive operation should be limited in how many
    // times we perform it. (An exception is how many times we inlined a
    // function, which we do not want to limit - it can be profitable to inline
    // a call into a great many callsites, over many iterations.)
    //
    // (Track names here, and not Function pointers, as we can remove functions
    // while inlining, and it may be confusing during debugging to have a
    // pointer to something that was removed.)
    std::unordered_map<Name, Index> iterationCounts;

    const size_t MaxIterationsForFunc = 5;

    while (iterationNumber <= numOriginalFunctions) {
#if INLINING_DEBUG
      std::cout << "inlining loop iter " << iterationNumber
                << " (numFunctions: " << module->functions.size() << ")\n";
#endif
      iterationNumber++;

      std::unordered_set<Function*> inlinedInto;

      prepare();
      iteration(inlinedInto);

      if (inlinedInto.empty()) {
        return;
      }

#if INLINING_DEBUG
      std::cout << "  inlined into " << inlinedInto.size() << " funcs.\n";
#endif

      for (auto* func : inlinedInto) {
        EHUtils::handleBlockNestedPops(func, *module);
      }

      for (auto* func : inlinedInto) {
        if (++iterationCounts[func->name] >= MaxIterationsForFunc) {
          return;
        }
      }

      if (functionSplitter) {
        auto splitNames = functionSplitter->finish();
        for (auto name : splitNames) {
          if (++iterationCounts[name] >= MaxIterationsForFunc) {
            return;
          }
        }
      }
    }
  }

  void prepare() {
    infos.clear();
    globalRefs.clear();
    // fill in info, as we operate on it in parallel (each function to its own
    // entry)
    for (auto& func : module->functions) {
      infos.try_emplace(func->name);
    }
    bool byteBudget = isByteBudgetMode();
    if (byteBudget) {
      for (auto& g : module->globals) {
        globalRefs.try_emplace(g->name, 0);
      }
    }
    {
      FunctionInfoScanner scanner(infos, byteBudget ? &globalRefs : nullptr);
      scanner.run(getPassRunner(), module);
      scanner.walkModuleCode(module);
    }
    for (auto& ex : module->exports) {
      if (ex->kind == ExternalKind::Function) {
        infos[*ex->getInternalName()].usedGlobally = true;
      }
    }
    if (module->start.is()) {
      infos[module->start].usedGlobally = true;
    }

    // When optimizing heavily for size, we may potentially split functions in
    // order to inline parts of them, if partialInliningIfs is enabled.
    auto& options = getPassOptions();
    if (options.optimizeLevel >= 3 && !options.shrinkLevel &&
        options.inlining.partialInliningIfs) {
      functionSplitter = std::make_unique<FunctionSplitter>(module, options);
    }
  }

  RankTables computeRankTables() {
    RankTables ranks;

    Index nextFuncRank = 0;
    std::vector<Name> definedFuncs;
    definedFuncs.reserve(module->functions.size());
    for (auto& func : module->functions) {
      if (func->imported()) {
        ranks.funcRank[func->name] = nextFuncRank++;
      } else {
        definedFuncs.push_back(func->name);
      }
    }
    std::sort(definedFuncs.begin(),
              definedFuncs.end(),
              [&](const Name& a, const Name& b) {
                Index ra = infos[a].refs.load();
                Index rb = infos[b].refs.load();
                if (ra != rb) {
                  return ra > rb;
                }
                return a > b;
              });
    for (auto& name : definedFuncs) {
      ranks.funcRank[name] = nextFuncRank++;
    }

    Index nextGlobalRank = 0;
    std::vector<Name> definedGlobals;
    definedGlobals.reserve(module->globals.size());
    for (auto& g : module->globals) {
      if (g->imported()) {
        ranks.globalRank[g->name] = nextGlobalRank++;
      } else {
        definedGlobals.push_back(g->name);
      }
    }
    std::sort(definedGlobals.begin(),
              definedGlobals.end(),
              [&](const Name& a, const Name& b) {
                Index ra = globalRefs[a].load();
                Index rb = globalRefs[b].load();
                if (ra != rb) {
                  return ra > rb;
                }
                return a < b;
              });
    for (auto& name : definedGlobals) {
      ranks.globalRank[name] = nextGlobalRank++;
    }

    if (module->features.hasGC()) {
      auto typeCounts = ModuleUtils::collectHeapTypeInfo(
        *module, WorldMode::Open, ModuleUtils::TypeInclusion::BinaryTypes);
      std::unordered_map<HeapType, double> weights;
      std::vector<HeapType> allTypes;
      allTypes.reserve(typeCounts.size());
      for (auto& [ht, info] : typeCounts) {
        weights[ht] = info.useCount;
        allTypes.push_back(ht);
        if (ht.isSignature() && info.useCount == 1 &&
            !ht.getDeclaredSuperType()) {
          ranks.soleUseFuncTypes.insert(ht);
        }
      }
      for (auto it = allTypes.rbegin(); it != allTypes.rend(); ++it) {
        if (auto sup = it->getSuperType()) {
          if (weights.contains(*sup)) {
            weights[*sup] = std::max(weights[*sup], weights[*it] + 0.5);
          }
        }
      }
      std::stable_sort(allTypes.begin(),
                       allTypes.end(),
                       [&](const HeapType& a, const HeapType& b) {
                         return weights[a] > weights[b];
                       });
      for (Index i = 0; i < allTypes.size(); ++i) {
        ranks.typeRank[allTypes[i]] = i;
      }
    }

    return ranks;
  }

  bool analyzeByteBudgetCandidate(Function* func,
                                  const RankTables& ranks,
                                  const PassOptions& options,
                                  ByteBudgetFuncInfo& bbInfo) {
    if (!func->body || func->imported() || func->noFullInline) {
      return false;
    }
    Index numParams = func->getNumParams();
    Index numVars = func->vars.size();
    bbInfo.paramGetCount.assign(numParams, 0);
    bbInfo.firstReadOrd.assign(numParams, 0);
    bbInfo.paramAllowsNullBottom.assign(numParams, true);
    bbInfo.paramNeedsExactHeapType.assign(numParams, false);
    bbInfo.paramInBranch.assign(numParams, false);
    bbInfo.prefixEffectsBeforeFirstGet.clear();
    bbInfo.prefixEffectsBeforeFirstGet.reserve(numParams);
    for (Index p = 0; p < numParams; ++p) {
      bbInfo.prefixEffectsBeforeFirstGet.emplace_back(options, *module);
    }
    bbInfo.varReads.assign(numVars, 0);
    bbInfo.varWrites.assign(numVars, 0);
    bbInfo.nonParamExprBytes = 0;
    bbInfo.totalVarAccesses = 0;

    int totalExprBytes = 0;
    Index nextReadOrd = 0;
    EffectAnalyzer runningEffects(options, *module);

    std::function<bool(Expression*, int, bool, bool)> walkExpr =
      [&](Expression* curr,
          int branchDepth,
          bool allowNullBottomHere,
          bool needsExactHeapTypeHere) -> bool {
      if (!curr) {
        return true;
      }
      if (curr->type == Type::unreachable) {
        return false;
      }
      switch (curr->_id) {
        case Expression::LoopId:
        case Expression::TryId:
        case Expression::TryTableId:
        case Expression::ReturnId:
        case Expression::BreakId:
        case Expression::SwitchId:
        case Expression::BrOnId:
        case Expression::PopId:
        case Expression::UnreachableId:
        case Expression::ThrowId:
        case Expression::RethrowId:
        case Expression::ThrowRefId:
        case Expression::TupleMakeId:
        case Expression::TupleExtractId:
        case Expression::ContNewId:
        case Expression::ContBindId:
        case Expression::SuspendId:
        case Expression::ResumeId:
        case Expression::ResumeThrowId:
        case Expression::StackSwitchId:
          return false;
        case Expression::BlockId: {
          auto* b = curr->cast<Block>();
          if (b->name.is()) {
            return false;
          }
          for (size_t i = 0; i < b->list.size(); ++i) {
            bool isLast = (i + 1 == b->list.size());
            if (!walkExpr(b->list[i],
                          branchDepth,
                          isLast ? allowNullBottomHere : true,
                          isLast ? needsExactHeapTypeHere : false)) {
              return false;
            }
          }
          return true;
        }
        case Expression::IfId: {
          auto* iff = curr->cast<If>();
          if (!walkExpr(iff->condition, branchDepth, true, false)) {
            return false;
          }
          if (!walkExpr(iff->ifTrue,
                        branchDepth + 1,
                        allowNullBottomHere,
                        needsExactHeapTypeHere)) {
            return false;
          }
          if (iff->ifFalse && !walkExpr(iff->ifFalse,
                                        branchDepth + 1,
                                        allowNullBottomHere,
                                        needsExactHeapTypeHere)) {
            return false;
          }
          int shallow = ranks.getExactShallowBytes(iff);
          bbInfo.nonParamExprBytes += shallow;
          totalExprBytes += shallow;
          ShallowEffectAnalyzer sea(options, *module, iff);
          if (options.trapsNeverHappen) {
            sea.trap = false;
            sea.implicitTrap = false;
          }
          sea.localsRead.clear();
          sea.localsWritten.clear();
          runningEffects.mergeIn(sea);
          return true;
        }
        case Expression::LocalGetId: {
          auto* lg = curr->cast<LocalGet>();
          totalExprBytes += ranks.getExactShallowBytes(lg);
          if (lg->index < numParams) {
            Index p = lg->index;
            if (bbInfo.paramGetCount[p] == 0) {
              bbInfo.firstReadOrd[p] = nextReadOrd++;
              bbInfo.prefixEffectsBeforeFirstGet[p].mergeIn(runningEffects);
            }
            bbInfo.paramGetCount[p]++;
            if (!allowNullBottomHere) {
              bbInfo.paramAllowsNullBottom[p] = false;
            }
            if (needsExactHeapTypeHere) {
              bbInfo.paramNeedsExactHeapType[p] = true;
            }
            if (branchDepth > 0) {
              bbInfo.paramInBranch[p] = true;
            }
          } else {
            Index v = lg->index - numParams;
            if (bbInfo.varWrites[v] == 0 || branchDepth > 0) {
              return false;
            }
            bbInfo.varReads[v]++;
            bbInfo.totalVarAccesses++;
          }
          return true;
        }
        case Expression::LocalSetId: {
          auto* ls = curr->cast<LocalSet>();
          if (ls->index < numParams || branchDepth > 0) {
            return false;
          }
          if (!walkExpr(ls->value, branchDepth, true, false)) {
            return false;
          }
          Index v = ls->index - numParams;
          bbInfo.varWrites[v]++;
          bbInfo.totalVarAccesses++;
          totalExprBytes += ranks.getExactShallowBytes(ls);
          return true;
        }
        default:
          break;
      }

      if (auto* c = curr->dynCast<Call>()) {
        if (c->isReturn || c->target == func->name) {
          return false;
        }
      } else if (auto* ci = curr->dynCast<CallIndirect>()) {
        if (ci->isReturn) {
          return false;
        }
      } else if (auto* cr = curr->dynCast<CallRef>()) {
        if (cr->isReturn) {
          return false;
        }
      }

      for (auto* child : ChildIterator(curr)) {
        bool childNeedsExactHT = false;
        if ((curr->is<StructSet>() && child == curr->cast<StructSet>()->ref) ||
            (curr->is<ArrayGet>() && child == curr->cast<ArrayGet>()->ref) ||
            (curr->is<ArraySet>() && child == curr->cast<ArraySet>()->ref) ||
            (curr->is<ArrayFill>() && child == curr->cast<ArrayFill>()->ref) ||
            (curr->is<ArrayCopy>() &&
             (child == curr->cast<ArrayCopy>()->destRef ||
              child == curr->cast<ArrayCopy>()->srcRef)) ||
            (curr->is<CallRef>() && child == curr->cast<CallRef>()->target)) {
          childNeedsExactHT = true;
        }
        bool childAllowNullBottom = false;
        if (!childNeedsExactHT) {
          if (curr->is<Call>() || curr->is<ArrayNewFixed>() ||
              curr->is<GlobalSet>() || curr->is<RefIsNull>() ||
              curr->is<RefEq>() || curr->is<Drop>() ||
              (curr->is<CallIndirect>() &&
               child != curr->cast<CallIndirect>()->target) ||
              (curr->is<CallRef>() && child != curr->cast<CallRef>()->target) ||
              (curr->is<StructNew>() &&
               child != curr->cast<StructNew>()->desc) ||
              (curr->is<StructSet>() &&
               child == curr->cast<StructSet>()->value) ||
              (curr->is<ArrayNew>() && child == curr->cast<ArrayNew>()->init) ||
              (curr->is<ArraySet>() &&
               child == curr->cast<ArraySet>()->value)) {
            childAllowNullBottom = true;
          }
        }
        if (!walkExpr(
              child, branchDepth, childAllowNullBottom, childNeedsExactHT)) {
          return false;
        }
      }

      int shallow = ranks.getExactShallowBytes(curr);
      bbInfo.nonParamExprBytes += shallow;
      totalExprBytes += shallow;

      ShallowEffectAnalyzer sea(options, *module, curr);
      if (options.trapsNeverHappen) {
        sea.trap = false;
        sea.implicitTrap = false;
      }
      sea.localsRead.clear();
      sea.localsWritten.clear();
      runningEffects.mergeIn(sea);
      return true;
    };

    if (!walkExpr(func->body,
                  /*branchDepth=*/0,
                  /*allowNullBottomHere=*/false,
                  /*needsExactHeapTypeHere=*/false)) {
      return false;
    }

    int localDeclBytes = 1;
    if (!func->vars.empty()) {
      std::unordered_map<Type, Index> varCounts;
      for (auto vt : func->vars) {
        varCounts[vt]++;
      }
      for (auto& [vt, cnt] : varCounts) {
        localDeclBytes +=
          FunctionInfoScanner::getULEBSize(cnt) + ranks.getTypeBytes(vt);
      }
    }
    int codeBodyBytes = localDeclBytes + totalExprBytes + 1;
    HeapType funcHT = func->type.getHeapType();
    int funcSecBytes = ranks.getTypeIndexBytes(funcHT, /*isS33=*/false);
    int soleSigBytes = 0;
    if (ranks.soleUseFuncTypes.contains(funcHT)) {
      soleSigBytes = 3;
      for (auto pt : func->getParams()) {
        soleSigBytes += ranks.getTypeBytes(pt);
      }
      if (func->getResults() != Type::none) {
        soleSigBytes += 1 + ranks.getTypeBytes(func->getResults());
      }
    }
    bbInfo.savedFuncBytes = funcSecBytes + soleSigBytes +
                            FunctionInfoScanner::getULEBSize(codeBodyBytes) +
                            codeBodyBytes;
    bbInfo.valid = true;
    return true;
  }

  bool evalByteBudgetCallSite(
    Call* call,
    Function* callee,
    Function* caller,
    const ByteBudgetFuncInfo& bbInfo,
    const RankTables& ranks,
    const PassOptions& options,
    ByteBudgetSitePlan& plan,
    std::unordered_map<Type, Index>& callerMaxScratch) {
    if (call->isReturn || call->type == Type::unreachable) {
      return false;
    }
    Index numParams = callee->getNumParams();
    if (call->operands.size() != numParams) {
      return false;
    }

    std::vector<EffectAnalyzer> argEffects;
    argEffects.reserve(numParams);
    for (Index p = 0; p < numParams; ++p) {
      Expression* arg = call->operands[p];
      if (arg->type == Type::unreachable) {
        return false;
      }
      argEffects.emplace_back(options, *module, arg);
      if (argEffects[p].branchesOut ||
          argEffects[p].hasExternalBreakTargets() ||
          argEffects[p].danglingPop || argEffects[p].mayNotReturn) {
        return false;
      }
      if (options.trapsNeverHappen) {
        argEffects[p].trap = false;
        argEffects[p].implicitTrap = false;
      }
    }

    std::vector<bool> mustPrologue(numParams, false);
    for (Index p = 0; p < numParams; ++p) {
      if (bbInfo.paramGetCount[p] == 0) {
        if (argEffects[p].hasUnremovableSideEffects()) {
          return false;
        }
        continue;
      }
      if (bbInfo.paramInBranch[p] && argEffects[p].hasSideEffects()) {
        mustPrologue[p] = true;
      }
      if (argEffects[p].invalidates(bbInfo.prefixEffectsBeforeFirstGet[p])) {
        mustPrologue[p] = true;
      }
      for (Index q = p + 1; q < numParams; ++q) {
        if (bbInfo.paramGetCount[q] == 0) {
          continue;
        }
        if (bbInfo.firstReadOrd[q] < bbInfo.firstReadOrd[p]) {
          if (argEffects[p].invalidates(argEffects[q])) {
            mustPrologue[p] = true;
          }
        }
      }
    }

    std::unordered_map<Type, Index> neededScratchByType;
    Index totalScratchCount = 0;
    for (Index v = 0; v < callee->vars.size(); ++v) {
      if (bbInfo.varReads[v] > 0 || bbInfo.varWrites[v] > 0) {
        neededScratchByType[callee->vars[v]]++;
        totalScratchCount++;
      }
    }

    int callerLocalBytes =
      (caller->getNumLocals() + totalScratchCount + numParams >= 128) ? 3 : 2;
    plan.paramStrategies.resize(numParams);
    std::vector<int> paramDeltas(numParams, 0);

    for (Index p = 0; p < numParams; ++p) {
      Expression* arg = call->operands[p];
      Index k = bbInfo.paramGetCount[p];
      Type paramType = callee->getLocalType(p);

      if (k == 0) {
        plan.paramStrategies[p] = ParamStrategy::DropSilent;
        paramDeltas[p] = -ranks.getTreeBytes(arg);
        continue;
      }

      if (mustPrologue[p]) {
        continue;
      }

      if (k == 1) {
        if (!arg->is<RefNull>() &&
            ((bbInfo.paramNeedsExactHeapType[p] &&
              arg->type.getHeapType() != paramType.getHeapType()) ||
             (!bbInfo.paramAllowsNullBottom[p] && arg->type.isNull()))) {
          if (bbInfo.paramInBranch[p]) {
            mustPrologue[p] = true;
            continue;
          }
          plan.paramStrategies[p] = ParamStrategy::ScratchTee;
          paramDeltas[p] = callerLocalBytes;
        } else {
          plan.paramStrategies[p] = ParamStrategy::DirectSingle;
          if (arg->is<RefNull>() && !bbInfo.paramAllowsNullBottom[p] &&
              arg->type != paramType) {
            int oldB = ranks.getExactShallowBytes(arg);
            int newB = 1 + ranks.getTypeIndexBytes(paramType.getHeapType(),
                                                   /*isS33=*/true);
            paramDeltas[p] = (newB - oldB);
          }
        }
        continue;
      }

      bool exactHTOk =
        (!bbInfo.paramNeedsExactHeapType[p] || arg->is<RefNull>() ||
         arg->type.getHeapType() == paramType.getHeapType()) &&
        (bbInfo.paramAllowsNullBottom[p] || arg->is<RefNull>() ||
         !arg->type.isNull());
      bool chosen = false;

      if (exactHTOk) {
        if (auto* lg = arg->dynCast<LocalGet>()) {
          bool clobbered = false;
          for (Index q = 0; q < numParams; ++q) {
            if (q != p && argEffects[q].localsWritten.contains(lg->index)) {
              clobbered = true;
              break;
            }
          }
          if (!clobbered) {
            plan.paramStrategies[p] = ParamStrategy::DirectDup;
            paramDeltas[p] = int(k - 1) * ranks.getExactShallowBytes(lg);
            chosen = true;
          }
        } else if (auto* ra = arg->dynCast<RefAs>();
                   ra && ra->op == RefAsNonNull && ra->value->is<LocalGet>()) {
          auto* lg = ra->value->cast<LocalGet>();
          bool clobbered = false;
          for (Index q = 0; q < numParams; ++q) {
            if (q != p && argEffects[q].localsWritten.contains(lg->index)) {
              clobbered = true;
              break;
            }
          }
          int treeB = 1 + ranks.getExactShallowBytes(lg);
          if (!clobbered && int(k - 1) * treeB <= int(k) * callerLocalBytes) {
            plan.paramStrategies[p] = ParamStrategy::DirectDup;
            paramDeltas[p] = int(k - 1) * treeB;
            chosen = true;
          }
        } else if (arg->is<Const>() || arg->is<RefNull>() ||
                   (arg->is<GlobalGet>() &&
                    !module->getGlobal(arg->cast<GlobalGet>()->name)
                       ->mutable_)) {
          int oldB = ranks.getExactShallowBytes(arg);
          int newB = oldB;
          if (arg->is<RefNull>() && !bbInfo.paramAllowsNullBottom[p] &&
              arg->type != paramType) {
            newB = 1 + ranks.getTypeIndexBytes(paramType.getHeapType(),
                                               /*isS33=*/true);
          }
          if (int(k) * newB - oldB <= int(k) * callerLocalBytes) {
            plan.paramStrategies[p] = ParamStrategy::DirectDup;
            paramDeltas[p] = int(k) * newB - oldB;
            chosen = true;
          }
        }
      }

      if (!chosen && exactHTOk && !bbInfo.paramInBranch[p]) {
        if (auto* ls = arg->dynCast<LocalSet>(); ls && ls->isTee()) {
          Type teeLocalType = caller->getLocalType(ls->index);
          bool teeTypeOk =
            Type::isSubType(teeLocalType, paramType) &&
            (!bbInfo.paramNeedsExactHeapType[p] ||
             teeLocalType.getHeapType() == paramType.getHeapType()) &&
            (bbInfo.paramAllowsNullBottom[p] || !teeLocalType.isNull());
          if (teeTypeOk) {
            bool conflict = false;
            for (Index q = 0; q < numParams; ++q) {
              if (q != p && (argEffects[q].localsWritten.contains(ls->index) ||
                             argEffects[q].localsRead.contains(ls->index))) {
                conflict = true;
                break;
              }
            }
            if (!conflict) {
              plan.paramStrategies[p] = ParamStrategy::ReuseArgTee;
              int getB = 1 + FunctionInfoScanner::getULEBSize(ls->index);
              paramDeltas[p] = int(k - 1) * getB;
              chosen = true;
            }
          }
        }
      }

      if (!chosen) {
        if (bbInfo.paramInBranch[p]) {
          mustPrologue[p] = true;
          continue;
        }
        plan.paramStrategies[p] = ParamStrategy::ScratchTee;
        paramDeltas[p] = int(k) * callerLocalBytes;
      }
    }

    for (Index p = numParams; p > 0;) {
      --p;
      if (!mustPrologue[p]) {
        continue;
      }
      for (Index r = 0; r < p; ++r) {
        if (bbInfo.paramGetCount[r] > 0 && !mustPrologue[r] &&
            argEffects[r].invalidates(argEffects[p])) {
          mustPrologue[r] = true;
        }
      }
    }

    int paramDelta = 0;
    for (Index p = 0; p < numParams; ++p) {
      Index k = bbInfo.paramGetCount[p];
      Type paramType = callee->getLocalType(p);
      if (!mustPrologue[p]) {
        if (k > 0 && plan.paramStrategies[p] == ParamStrategy::ScratchTee) {
          neededScratchByType[paramType]++;
        }
        paramDelta += paramDeltas[p];
        continue;
      }

      Expression* arg = call->operands[p];
      bool chosen = false;
      if (auto* ls = arg->dynCast<LocalSet>(); ls && ls->isTee()) {
        Type teeLocalType = caller->getLocalType(ls->index);
        bool teeTypeOk =
          Type::isSubType(teeLocalType, paramType) &&
          (!bbInfo.paramNeedsExactHeapType[p] ||
           teeLocalType.getHeapType() == paramType.getHeapType()) &&
          (bbInfo.paramAllowsNullBottom[p] || !teeLocalType.isNull());
        if (teeTypeOk) {
          bool conflict = false;
          for (Index q = 0; q < numParams; ++q) {
            if (q == p || bbInfo.paramGetCount[q] == 0) {
              continue;
            }
            if (q > p && argEffects[q].localsWritten.contains(ls->index)) {
              conflict = true;
              break;
            }
            if (q < p && !mustPrologue[q] &&
                (argEffects[q].localsWritten.contains(ls->index) ||
                 argEffects[q].localsRead.contains(ls->index))) {
              conflict = true;
              break;
            }
          }
          if (!conflict) {
            plan.paramStrategies[p] = ParamStrategy::PrologueReuseTee;
            int getB = 1 + FunctionInfoScanner::getULEBSize(ls->index);
            paramDelta += int(k) * getB;
            chosen = true;
          }
        }
      }
      if (!chosen) {
        plan.paramStrategies[p] = ParamStrategy::PrologueSet;
        neededScratchByType[paramType]++;
        paramDelta += int(k + 1) * callerLocalBytes;
      }
    }

    for (auto& [vt, need] : neededScratchByType) {
      callerMaxScratch[vt] = std::max(callerMaxScratch[vt], need);
    }

    int callInstBytes = 1 + ranks.getFuncIndexBytes(callee->name);
    plan.siteDeltaBytes = bbInfo.nonParamExprBytes +
                          int(bbInfo.totalVarAccesses) * callerLocalBytes -
                          callInstBytes + paramDelta;
    plan.valid = true;
    return true;
  }

  void iteration(std::unordered_set<Function*>& inlinedInto) {
    // decide which to inline
    InliningState state;
    auto& options = getPassOptions();
    bool byteBudget = isByteBudgetMode();
    RankTables ranks;
    if (byteBudget) {
      ranks = computeRankTables();
    }

    ModuleUtils::iterDefinedFunctions(*module, [&](Function* func) {
      InliningMode inliningMode = getInliningMode(func->name);
      assert(inliningMode != InliningMode::Unknown);
      if (inliningMode != InliningMode::Uninlineable) {
        state.inlinableFunctions[func->name] = inliningMode;
      }
      if (byteBudget) {
        auto& info = infos[func->name];
        if (!info.usedGlobally && !info.hasTryDelegate && !func->noFullInline &&
            info.toolchainInlineHint != CodeAnnotation::NeverInline &&
            info.refs >= 1) {
          ByteBudgetFuncInfo bbInfo;
          if (analyzeByteBudgetCandidate(func, ranks, options, bbInfo)) {
            state.byteBudgetInfos[func->name] = std::move(bbInfo);
            if (!state.inlinableFunctions.contains(func->name)) {
              info.inliningMode = InliningMode::Full;
              state.inlinableFunctions[func->name] = InliningMode::Full;
              state.byteBudgetOnlyFunctions.insert(func->name);
            }
          }
        }
      }
      // Functions that are not fully inlined everywhere may still be worth
      // fully inlining at call sites with constant arguments. That is only
      // useful if we optimize after inlining, as the benefit comes from the
      // inlined code folding away.
      if (optimize && inliningMode != InliningMode::Full &&
          !func->noFullInline &&
          infos[func->name].worthConstantArgInlining(options)) {
        state.constantArgFunctions.insert(func->name);
      }
    });
    if (state.inlinableFunctions.empty() &&
        state.constantArgFunctions.empty()) {
      return;
    }
    if (!state.constantArgFunctions.empty()) {
      findConstantGlobals(state.constantGlobals);
    }
    // Fill in actionsForFunction, as we operate on it in parallel (each
    // function to its own entry). Also generate a vector of the function names
    // so that in the later loop we can iterate on it deterministically and
    // without iterator invalidation.
    std::vector<Name> funcNames;
    for (auto& func : module->functions) {
      state.actionsForFunction.try_emplace(func->name);
      funcNames.push_back(func->name);
    }

    // Find and plan inlinings in parallel. This discovers inlining
    // opportunities, by themselves, but does not yet take into account
    // interactions between them (e.g. we don't want to both inline into a
    // function and then inline it as well).
    Planner(&state).run(getPassRunner(), module);

    std::unordered_set<Name> rejectedByteBudgetFunctions;
    if (byteBudget && !state.byteBudgetInfos.empty()) {
      struct SiteRef {
        Function* caller;
        InliningAction* action;
      };
      std::unordered_map<Name, std::vector<SiteRef>> sitesByCallee;
      for (auto callerName : funcNames) {
        auto* callerFunc = module->getFunction(callerName);
        for (auto& action : state.actionsForFunction[callerName]) {
          Name calleeName = action.contents->name;
          if (state.byteBudgetInfos.contains(calleeName)) {
            sitesByCallee[calleeName].push_back({callerFunc, &action});
          }
        }
      }

      for (auto& [calleeName, bbInfo] : state.byteBudgetInfos) {
        auto& sites = sitesByCallee[calleeName];
        bool isByteBudgetOnly =
          state.byteBudgetOnlyFunctions.contains(calleeName);
        if (sites.size() != infos[calleeName].refs) {
          if (isByteBudgetOnly) {
            rejectedByteBudgetFunctions.insert(calleeName);
          }
          continue;
        }

        auto* calleeFunc = module->getFunction(calleeName);
        bool allValid = true;
        int totalSiteDeltaBytes = 0;
        std::vector<ByteBudgetSitePlan> plans(sites.size());
        std::unordered_map<Function*, std::unordered_map<Type, Index>>
          callerMaxScratch;
        for (size_t i = 0; i < sites.size(); ++i) {
          auto* call = (*sites[i].action->callSite)->cast<Call>();
          if (!evalByteBudgetCallSite(call,
                                      calleeFunc,
                                      sites[i].caller,
                                      bbInfo,
                                      ranks,
                                      options,
                                      plans[i],
                                      callerMaxScratch[sites[i].caller])) {
            allValid = false;
            break;
          }
          totalSiteDeltaBytes += plans[i].siteDeltaBytes;
        }

        if (!allValid) {
          if (isByteBudgetOnly) {
            rejectedByteBudgetFunctions.insert(calleeName);
          }
          continue;
        }

        for (auto& [callerFunc, scratchByType] : callerMaxScratch) {
          std::unordered_set<Type> existingVarTypes(callerFunc->vars.begin(),
                                                    callerFunc->vars.end());
          for (auto& [vt, need] : scratchByType) {
            if (need > 0 && !existingVarTypes.contains(vt)) {
              totalSiteDeltaBytes +=
                FunctionInfoScanner::getULEBSize(need) + ranks.getTypeBytes(vt);
            }
          }
        }

        if (isByteBudgetOnly && bbInfo.savedFuncBytes <= totalSiteDeltaBytes) {
          rejectedByteBudgetFunctions.insert(calleeName);
          continue;
        }

        // Profitable (or already inlined via worthFullInlining and valid for
        // direct substitution): attach direct inlining plans.
        for (size_t i = 0; i < sites.size(); ++i) {
          sites[i].action->useDirectInlining = true;
          sites[i].action->directPlan = std::move(plans[i]);
          sites[i].action->bbFuncInfo = &bbInfo;
        }
      }
    }

    // Choose which inlinings to perform. We do this sequentially so that we
    // can consider interactions between them, and avoid nondeterminism.
    ChosenActions chosenActions;

    // How many uses (calls of the function) we inlined.
    std::unordered_map<Name, Index> inlinedUses;

    for (auto name : funcNames) {
      auto* func = module->getFunction(name);
      // if we've inlined a function, don't inline into it in this iteration,
      // avoid risk of races
      // note that we do not risk stalling progress, as each iteration() will
      // inline at least one call before hitting this
      if (inlinedUses.contains(func->name)) {
        continue;
      }
      for (auto& action : state.actionsForFunction[name]) {
        auto* inlinedFunction = action.contents;
        Name inlinedName = inlinedFunction->name;
        if (rejectedByteBudgetFunctions.contains(inlinedName)) {
          continue;
        }
        // if we've inlined into a function, don't inline it in this iteration,
        // avoid risk of races
        // note that we do not risk stalling progress, as each iteration() will
        // inline at least one call before hitting this
        if (inlinedInto.contains(inlinedFunction)) {
          continue;
        }
        if (!isUnderSizeLimit(func->name, inlinedName)) {
          continue;
        }
        if (action.forceFull &&
            !constantArgInliningShrinks((*action.callSite)->cast<Call>(),
                                        inlinedFunction)) {
          continue;
        }

        // Success - we can inline.
#if INLINING_DEBUG
        std::cout << "inline " << inlinedName << " into " << func->name << '\n';
#endif

        // Update the action for the actual inlining we have chosen to perform
        // (when splitting, we will actually inline one of the split pieces and
        // not the original function itself; note how even if we do that then
        // we are still removing a call to the original function here, and so
        // we do not need to change anything else lower down - we still want to
        // note that we got rid of one use of the original function).
        if (!action.forceFull) {
          action.contents = getActuallyInlinedFunction(action.contents);
        }
        action.nameHint = inlinedNameHint++;
        chosenActions[func->name].push_back(action);
        inlinedUses[inlinedName]++;
        inlinedInto.insert(func);
        assert(inlinedUses[inlinedName] <= infos[inlinedName].refs);
      }
    }

    // Enforce all-or-nothing inlining for multi-caller functions selected only
    // via byteBudgetMode: if any call site was skipped in this iteration (e.g.
    // due to inlinedUses/inlinedInto ordering conflicts), defer all call sites
    // of that function to the next iteration so it is never partially inlined.
    if (byteBudget && !state.byteBudgetOnlyFunctions.empty()) {
      std::unordered_set<Name> incompleteCallee;
      for (auto& [calleeName, count] : inlinedUses) {
        if (state.byteBudgetOnlyFunctions.contains(calleeName) &&
            count != infos[calleeName].refs) {
          incompleteCallee.insert(calleeName);
        }
      }
      if (!incompleteCallee.empty()) {
        for (auto calleeName : incompleteCallee) {
          inlinedUses.erase(calleeName);
        }
        inlinedInto.clear();
        for (auto it = chosenActions.begin(); it != chosenActions.end();) {
          auto& vec = it->second;
          vec.erase(std::remove_if(vec.begin(),
                                   vec.end(),
                                   [&](const InliningAction& a) {
                                     return incompleteCallee.contains(
                                       a.contents->name);
                                   }),
                    vec.end());
          if (vec.empty()) {
            it = chosenActions.erase(it);
          } else {
            inlinedInto.insert(module->getFunction(it->first));
            ++it;
          }
        }
      }
    }

    if (chosenActions.empty()) {
      // We found nothing to do.
      return;
    }

    // Perform the inlinings in parallel (sequentially inside each function we
    // inline into, but in parallel between them). If we are optimizing, do so
    // as well.
    {
      PassUtils::FilteredPassRunner runner(
        module, inlinedInto, getPassRunner()->options);
      runner.setIsNested(true);
      runner.add(std::make_unique<DoInlining>(chosenActions));
      if (optimize) {
        OptUtils::addUsefulPassesAfterInlining(runner);
      }
      runner.run();
    }

    // remove functions that we no longer need after inlining
    module->removeFunctions([&](Function* func) {
      auto name = func->name;
      auto& info = infos[name];
      return inlinedUses.contains(name) && inlinedUses[name] == info.refs &&
             !info.usedGlobally;
    });
  }

  // See explanation in InliningAction.
  Index inlinedNameHint = 0;

  // Decide for a given function whether to inline, and if so in what mode.
  InliningMode getInliningMode(Name name) {
    auto* func = module->getFunction(name);
    auto& info = infos[name];

    if (info.inliningMode != InliningMode::Unknown) {
      return info.inliningMode;
    }

    // Check if the function itself is worth inlining as it is.
    if (!func->noFullInline &&
        info.worthFullInlining(getPassOptions(), isByteBudgetMode())) {
      return info.inliningMode = InliningMode::Full;
    }

    // Otherwise, check if we can at least inline part of it, if we are
    // interested in such things.
    if (!func->noPartialInline && functionSplitter) {
      info.inliningMode = functionSplitter->getSplitDrivenInliningMode(
        module->getFunction(name), info);
      return info.inliningMode;
    }

    // Cannot be fully or partially inlined => uninlineable.
    info.inliningMode = InliningMode::Uninlineable;
    return info.inliningMode;
  }

  // Gets the actual function to be inlined. Normally this is the function
  // itself, but if it is a function that we must first split (i.e., we only
  // want to partially inline it) then it will be the inlineable part of the
  // split.
  //
  // This is called right before actually performing the inlining, that is, we
  // are guaranteed to inline after this.
  Function* getActuallyInlinedFunction(Function* func) {
    InliningMode inliningMode = infos[func->name].inliningMode;
    // If we want to inline this function itself, do so.
    if (inliningMode == InliningMode::Full) {
      return func;
    }

    // Otherwise, this is a case where we want to inline part of it, after
    // splitting.
    assert(functionSplitter);
    return functionSplitter->getInlineableSplitFunction(func, inliningMode);
  }

  // Checks if the combined size of the code after inlining is under the
  // absolute size limit.
  bool isUnderSizeLimit(Name target, Name source) {
    // Estimate the combined binary size from the number of instructions.
    auto combinedSize = infos[target].size + infos[source].size;
    auto estimatedBinarySize = Measurer::BytesPerExpr * combinedSize;
    auto& options = getPassRunner()->options;
    return estimatedBinarySize < options.inlining.maxCombinedBinarySize;
  }

  // Finds immutable globals whose value is a constant, or an allocation whose
  // fields are all constants (such a value has a fixed identity and contents,
  // so comparisons and immutable field reads of it can be precomputed). Globals
  // can only refer to earlier globals, so a single pass in order suffices.
  void findConstantGlobals(std::unordered_set<Name>& constantGlobals) {
    std::function<bool(Expression*)> isConstant = [&](Expression* curr) {
      if (Properties::isSingleConstantExpression(curr)) {
        return true;
      }
      if (auto* get = curr->dynCast<GlobalGet>()) {
        return constantGlobals.contains(get->name);
      }
      if (curr->is<StructNew>() || curr->is<ArrayNewFixed>() ||
          curr->is<ArrayNew>() || curr->is<RefI31>()) {
        for (auto* child : ChildIterator(curr)) {
          if (!isConstant(child)) {
            return false;
          }
        }
        return true;
      }
      return false;
    };
    for (auto& global : module->globals) {
      if (!global->imported() && !global->mutable_ && isConstant(global->init)) {
        constantGlobals.insert(global->name);
      }
    }
  }

  // Memoized results of constant-argument trial inlinings, keyed by the callee
  // and the constant arguments (which fully determine the result).
  std::unordered_map<std::string, bool> constantArgTrialResults;

  // Checks whether inlining the callee at a call site whose arguments are all
  // constants would shrink the code, by actually doing that in a scratch
  // function, optimizing it, and measuring. This is the crucial check for
  // constant-argument inlining: inlining a large callee is only worthwhile when
  // the constant arguments let the inlined code fold away.
  bool constantArgInliningShrinks(Call* call, Function* callee) {
    std::stringstream key;
    key << callee->name;
    for (auto* operand : call->operands) {
      key << ' ' << *operand;
    }
    auto [iter, inserted] =
      constantArgTrialResults.try_emplace(key.str(), false);
    if (inserted) {
      iter->second = runConstantArgTrial(call, callee);
    }
    return iter->second;
  }

  bool runConstantArgTrial(Call* call, Function* callee) {
    auto& options = getPassOptions();
    Function trial;
    trial.name = Name("constant-arg-trial");
    trial.setResults(callee->getResults());
    auto* copy = ExpressionManipulator::copy(call, *module)->cast<Call>();
    copy->isReturn = false;
    copy->type = callee->getResults();
    trial.body = copy;
    doInlining(module, &trial, InliningAction(&trial.body, callee, false), options);
    PassRunner runner(module, options);
    runner.setIsNested(true);
    OptUtils::addUsefulPassesAfterInlining(runner);
    runner.runOnFunction(&trial);
    return Measurer::measure(trial.body) <= Measurer::measure(call);
  }
};

} // anonymous namespace

//
// InlineMain
//
// Inline __original_main into main, if they exist. This works around the odd
// thing that clang/llvm currently do, where __original_main contains the user's
// actual main (this is done as a workaround for main having two different
// possible signatures).
//

static const char* MAIN = "main";
static const char* ORIGINAL_MAIN = "__original_main";

struct InlineMainPass : public Pass {
  void run(Module* module) override {
    auto* main = module->getFunctionOrNull(MAIN);
    auto* originalMain = module->getFunctionOrNull(ORIGINAL_MAIN);
    if (!main || main->imported() || !originalMain ||
        originalMain->imported()) {
      return;
    }
    FindAllPointers<Call> calls(main->body);
    Expression** callSite = nullptr;
    for (auto* call : calls.list) {
      if ((*call)->cast<Call>()->target == ORIGINAL_MAIN) {
        if (callSite) {
          // More than one call site.
          return;
        }
        callSite = call;
      }
    }
    if (!callSite) {
      // No call at all.
      return;
    }
    doInlining(module,
               main,
               InliningAction(callSite, originalMain, true),
               getPassOptions());
  }
};

Pass* createInliningPass() { return new Inlining(); }

Pass* createInliningOptimizingPass() {
  auto* ret = new Inlining();
  ret->optimize = true;
  return ret;
}

Pass* createInlineMainPass() { return new InlineMainPass(); }

} // namespace wasm
