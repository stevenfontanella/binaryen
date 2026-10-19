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
// Rolls runs of similar consecutive statements into loops over data columns.
//
// Generated initialization code (registrations, table fills, map puts) often
// consists of long runs of statements that are identical except for some
// constants, for example
//
//   (drop (call $put (local.get $m) (call $str (i32.const 10)) (f64.const 1)))
//   (drop (call $put (local.get $m) (call $str (i32.const 13)) (f64.const 2)))
//   ...
//
// This pass finds maximal runs of such statements in each block: consecutive
// statements with the same tree shape, whose leaves are constants (i32.const
// etc., or global.get of an immutable global) and local.gets of the same
// locals. Each run is replaced by a single copy of the statement inside a loop
// over a counter. A constant leaf that varies along the run becomes
//
//   - an arithmetic progression of the counter (base + step * i), or
//   - a read from a column of values in a passive data segment, which is
//     loaded into an array local (array.new_data) before the run, or
//   - for references to immutable globals, a read from an immutable global
//     array of the distinct globals, indexed by a byte column.
//
// Only the leaves change: the statements still execute in the same order with
// the same values, so this is valid whatever the effects of the statements.
//
// All the columns of a function share one array per element type, loaded in
// the innermost block that contains all of its rolled runs; all functions
// share one passive data segment.
//
// A call of a function on a single i32 constant that returns a reference
// (typically a factory of lazily-allocated constants) is also a leaf, which
// may share a hole with global.gets: such a hole becomes a column of indexes
// that select either an entry of the table or an argument for the call.
//
// This is a size optimization that costs a little speed per statement; it is
// not part of the default pipelines. It is best run late, before passes that
// outline duplicated code (which would otherwise shrink the statements first
// and hide the runs).
//

#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "ir/branch-utils.h"
#include "ir/find_all.h"
#include "ir/iteration.h"
#include "ir/manipulation.h"
#include "ir/names.h"
#include "ir/utils.h"
#include "pass.h"
#include "support/hash.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

// Rough estimates of binary sizes, in bytes.

// The shortest run we consider.
const Index MinRun = 4;
// local.set i 0, loop, end, and the br_if with the counter update and compare.
const Index LoopBytes = 20;
// local.set of an array.new_data, plus the local declaration.
const Index ArraySetupBytes = 15;
// A new global holding an array.new_fixed of global.gets.
const Index TableBytes = 8;
const Index TableEntryBytes = 3;
// Below this many values a byte column of differences does not pay for its
// accumulator (a local, its initialization, and the add and tee per read).
const Index MinDeltaRun = 10;
const Index CallBytes = 3;
const Index LocalGetBytes = 2;
const Index GlobalGetBytes = 3;
const Index ArrayGetBytes = 3;
const Index MaxDepth = 64;
// The largest function body we inline while scanning.
const Index MaxInlinedNodes = 18;
const Index MaxInlineDepth = 5;

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

Index getConstBytes(const Literal& value) {
  switch (value.type.getBasic()) {
    case Type::i32:
      return 1 + getSLEBSize(value.geti32());
    case Type::i64:
      return 1 + getSLEBSize(value.geti64());
    case Type::f32:
      return 5;
    case Type::f64:
      return 9;
    default:
      return 2;
  }
}

// The estimated binary size of an expression, not including its children.
Index getShallowBytes(Expression* curr) {
  switch (curr->_id) {
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
    case Expression::ConstId:
      return getConstBytes(curr->cast<Const>()->value);
    case Expression::DropId:
    case Expression::UnaryId:
    case Expression::BinaryId:
    case Expression::RefIsNullId:
    case Expression::RefEqId:
      return 1;
    case Expression::SelectId:
      return curr->type.isRef() ? 3 : 1;
    case Expression::StructNewId:
    case Expression::StructGetId:
    case Expression::StructSetId:
    case Expression::ArrayNewId:
    case Expression::ArrayGetId:
    case Expression::ArraySetId:
    case Expression::RefCastId:
    case Expression::RefTestId:
    case Expression::RefFuncId:
      return 3;
    case Expression::ArrayNewFixedId:
      return 4;
    case Expression::RefNullId:
    case Expression::RefAsId:
    case Expression::RefI31Id:
    case Expression::I31GetId:
      return 2;
    case Expression::BlockId:
      return 0;
    default:
      return 2;
  }
}

Index getTreeBytes(Expression* curr) {
  Index total = getShallowBytes(curr);
  for (auto* child : ChildIterator(curr)) {
    total += getTreeBytes(child);
  }
  return total;
}

// Const: a numeric constant (or a global.get of an immutable global that is
//        initialized to one).
// Global: a global.get of an immutable reference global (or a closed local-free
//         reference expression).
// IndexedCall: a call of a function on an i32 constant, returning a reference.
enum class LeafKind { Const, Global, IndexedCall };

// A leaf of a statement that may vary along a run.
struct Leaf {
  Expression** ptr;
  LeafKind kind;
  // The value of a Const leaf, or the argument of an IndexedCall leaf.
  Literal value;
  // The global of a Global leaf.
  Name global;
  // The target of an IndexedCall leaf.
  Name callee;
  Type type;
  // The encoded size of the original leaf.
  Index bytes;
  // Whether the leaf comes from an inlined call (see Statement::inlined).
  bool inlined = false;
  // For a closed local-free reference expression leaf (LeafKind::Global).
  Expression* refExpr = nullptr;
};

bool sameGlobalLeaf(const Leaf& a, const Leaf& b) {
  if ((a.refExpr != nullptr) != (b.refExpr != nullptr)) {
    return false;
  }
  if (a.refExpr) {
    return ExpressionAnalyzer::equal(a.refExpr, b.refExpr);
  }
  return a.global == b.global;
}

struct Statement {
  Expression* expr = nullptr;
  bool valid = false;
  // The shape: a pre-order encoding of the tree with the leaves abstracted.
  // The loose key abstracts all reference leaves (Global and IndexedCall) the
  // same way, so that statements that differ only in such leaves can share a
  // run; the strict key distinguishes them and the call targets.
  std::vector<size_t> key;
  std::vector<size_t> strictKey;
  std::vector<Leaf> leaves;
  // Bytes of the tree excluding the leaves.
  Index fixedBytes = 0;
  // Calls of small pure functions without parameters (e.g. outlined
  // allocations of constants) are inlined while scanning, so that they take
  // the shape of their body; they are restored if the statement is not
  // rolled. The original bytes of the call, minus the bytes of the inlined
  // tree, so that the original size can be computed.
  std::vector<std::pair<Expression**, Expression*>> inlined;
  int inlinedAdjust = 0;
};

// Element kinds of the column arrays.
enum class ColumnKind { I8, I16, I32, I64, F32, F64, Count };

Index getColumnWidth(ColumnKind kind) {
  switch (kind) {
    case ColumnKind::I8:
      return 1;
    case ColumnKind::I16:
      return 2;
    case ColumnKind::I32:
    case ColumnKind::F32:
      return 4;
    case ColumnKind::I64:
    case ColumnKind::F64:
      return 8;
    case ColumnKind::Count:
      WASM_UNREACHABLE("bad kind");
  }
  WASM_UNREACHABLE("bad kind");
}

HeapType getColumnHeapType(ColumnKind kind) {
  switch (kind) {
    case ColumnKind::I8:
      return HeapTypes::getMutI8Array();
    case ColumnKind::I16:
      return HeapTypes::getMutI16Array();
    case ColumnKind::I32:
      return HeapType(Array(Field(Type::i32, Mutable)));
    case ColumnKind::I64:
      return HeapType(Array(Field(Type::i64, Mutable)));
    case ColumnKind::F32:
      return HeapType(Array(Field(Type::f32, Mutable)));
    case ColumnKind::F64:
      return HeapType(Array(Field(Type::f64, Mutable)));
    case ColumnKind::Count:
      WASM_UNREACHABLE("bad kind");
  }
  WASM_UNREACHABLE("bad kind");
}

Type getColumnValueType(ColumnKind kind) {
  switch (kind) {
    case ColumnKind::I8:
    case ColumnKind::I16:
    case ColumnKind::I32:
      return Type::i32;
    case ColumnKind::I64:
      return Type::i64;
    case ColumnKind::F32:
      return Type::f32;
    case ColumnKind::F64:
      return Type::f64;
    case ColumnKind::Count:
      WASM_UNREACHABLE("bad kind");
  }
  WASM_UNREACHABLE("bad kind");
}

struct CalleePlan {
  Name name;
  uint32_t minArg = UINT32_MAX;
  uint32_t maxArg = 0;
  uint32_t baseArg = 0;
  Index bound = 0;
};

// How a varying leaf is computed in the loop.
struct HolePlan {
  // Mixed: a column of indexes; an index below tableSize selects an entry of
  // the table, and otherwise selects a factory call and argument.
  // Delta: a byte column of differences, accumulated in a local that starts
  // at base.
  enum Kind { Fixed, Progression, Delta, Column, Table, Mixed } kind = Fixed;
  // Progression: base + step * i. Delta: the first value.
  uint32_t base = 0;
  uint32_t step = 0;
  // If set, the computed value is passed to this function.
  Name callee;
  // Mixed.
  Index tableSize = 0;
  Type mixedType;
  std::vector<CalleePlan> callees;
  // Column (and the index column of a Table).
  ColumnKind column = ColumnKind::I8;
  bool signed_ = false;
  std::vector<uint8_t> data;
  // Assigned when the function's arrays are laid out.
  Index offset = 0;
  // Whether this hole carries the 0 gap sentinel for a switch with gaps, and
  // the offset added to non-gap column values so they are non-zero.
  bool isGapHole = false;
  Index gapOffset = 0;
  // Table: the index of the table in the function state.
  Index table = 0;
  std::vector<Name> newGlobals;
  // Dictionary-indexed column (e.g. I8 index column into an I32 dictionary).
  bool hasDict = false;
  ColumnKind dictColumn = ColumnKind::I32;
  std::vector<uint8_t> dictData;
  Index dictCount = 0;
  Index dictOffset = 0;
};

struct RunPlan {
  Block* block = nullptr;
  Index start = 0;
  Index size = 0;
  Statement first;
  std::vector<HolePlan> holes;
  // For switch ladders:
  Expression** switchPtr = nullptr;
  Expression* switchCond = nullptr;
  // 0: Stmt, 1: Return, 2: BrVal
  int switchKind = -1;
  Name switchExitLabel = Name();
  bool hasGaps = false;
  // The expression that replaced the run.
  Expression* replacement = nullptr;
  int saved = 0;
};

struct Table {
  Type elemType;
  std::vector<Name> globals;
  std::vector<Expression*> exprs;
  std::map<Name, Index> indexes;
  Name globalName;
  Index localIndex = 0;
  bool hasExprs = false;
};

struct FuncState {
  Function* func = nullptr;
  std::map<ColumnKind, Index> arrayLocals;
  std::set<ColumnKind> usedKinds;
  std::vector<ColumnKind> kindOrder;
  std::map<ColumnKind, std::vector<uint8_t>> arrayData;
  std::map<ColumnKind, Index> arrayCounts;
  std::vector<Table> tables;
  std::unordered_map<Type, Index> tablesByType;
  std::vector<RunPlan> plans;
  std::set<Name> scopeNames;
  std::unordered_set<Block*> plannedBlocks;
  std::unordered_map<Type, Index> scratchLocals;
  bool inArrayNewFixed = false;
};

void appendLE(std::vector<uint8_t>& data, uint64_t value, Index width) {
  for (Index i = 0; i < width; i++) {
    data.push_back(uint8_t(value >> (8 * i)));
  }
}

struct RollRepeatedCalls : public Pass {
  Module* module = nullptr;
  std::vector<char> segmentData;
  Name segmentName;
  std::map<std::pair<Name, Name>, Name> composedWrappers;

  void run(Module* module_) override {
    module = module_;
    if (!module->features.hasGC() || !module->features.hasBulkMemory()) {
      return;
    }
    for (auto& f : module->functions) {
      if (!f->imported() && f->getNumParams() == 1 && f->getNumVars() == 0 &&
          f->getParams() == Type::i32 && f->getResults().isRef()) {
        if (auto* c1 = f->body->dynCast<Call>()) {
          if (!c1->isReturn && c1->operands.size() == 1) {
            if (auto* c2 = c1->operands[0]->dynCast<Call>()) {
              if (!c2->isReturn && c2->operands.size() == 1) {
                if (auto* lg = c2->operands[0]->dynCast<LocalGet>()) {
                  if (lg->index == 0) {
                    composedWrappers.emplace(
                      std::make_pair(c1->target, c2->target), f->name);
                  }
                }
              }
            }
          }
        }
      }
    }
    DataSegment* existingSegment = module->getDataSegmentOrNull("roll");
    if (existingSegment && existingSegment->isPassive()) {
      segmentName = existingSegment->name;
      segmentData = std::move(existingSegment->data);
    } else {
      existingSegment = nullptr;
      segmentName = Names::getValidDataSegmentName(*module, "roll");
    }
    for (auto& func : module->functions) {
      if (!func->imported()) {
        processFunction(func.get());
      }
    }
    if (existingSegment) {
      existingSegment->data = std::move(segmentData);
    } else if (!segmentData.empty()) {
      module->addDataSegment(Builder::makeDataSegment(
        segmentName, Name(), nullptr, segmentData.data(), segmentData.size()));
    }
  }

  // Scanning statements.

  struct Scanner {
    Module& wasm;
    Statement& stmt;
    RollRepeatedCalls& pass;
    Function* func;
    FuncState* state = nullptr;
    Index inlineDepth = 0;

    Scanner(Module& wasm,
            Statement& stmt,
            RollRepeatedCalls& pass,
            Function* func,
            FuncState* state = nullptr)
      : wasm(wasm), stmt(stmt), pass(pass), func(func), state(state) {}

    void push(size_t value) {
      stmt.key.push_back(value);
      stmt.strictKey.push_back(value);
    }

    // A reference leaf: the loose key only notes that there is one.
    void pushRefLeaf(size_t strict) {
      stmt.key.push_back(hash(size_t(LeafKind::Global)));
      stmt.strictKey.push_back(strict);
    }

    bool getConstI32(Expression* expr, Literal& out, Index& bytes) {
      if (auto* c = expr->dynCast<Const>()) {
        if (c->type == Type::i32) {
          out = c->value;
          bytes = getConstBytes(c->value);
          return true;
        }
      } else if (auto* get = expr->dynCast<GlobalGet>()) {
        auto* global = wasm.getGlobal(get->name);
        if (!global->mutable_ && !global->imported()) {
          if (auto* c = global->init->dynCast<Const>()) {
            if (c->type == Type::i32) {
              out = c->value;
              bytes = GlobalGetBytes;
              return true;
            }
          }
        }
      }
      return false;
    }

    bool isClosedRefExpr(Expression* e, Index d = 0) {
      if (d > 5 || e->type == Type::unreachable) {
        return false;
      }
      switch (e->_id) {
        case Expression::ConstId:
        case Expression::RefNullId:
          return true;
        case Expression::GlobalGetId: {
          auto* g = wasm.getGlobal(e->cast<GlobalGet>()->name);
          return !g->mutable_;
        }
        case Expression::ArrayNewId:
        case Expression::ArrayNewFixedId: {
          for (auto* child : ChildIterator(e)) {
            if (!isClosedRefExpr(child, d + 1)) {
              return false;
            }
          }
          return true;
        }
        case Expression::CallId: {
          auto* c = e->cast<Call>();
          if (c->isReturn || !c->type.isRef()) {
            return false;
          }
          for (auto* op : c->operands) {
            if (!isClosedRefExpr(op, d + 1)) {
              return false;
            }
          }
          return true;
        }
        default:
          return false;
      }
    }

    bool isSideEffectFreeArg(Expression* e) {
      if (e->type == Type::unreachable) {
        return false;
      }
      switch (e->_id) {
        case Expression::ConstId:
        case Expression::RefNullId:
        case Expression::LocalGetId:
          return true;
        case Expression::GlobalGetId: {
          auto* g = wasm.getGlobal(e->cast<GlobalGet>()->name);
          return !g->mutable_;
        }
        case Expression::RefCastId:
        case Expression::StructNewId:
        case Expression::ArrayNewId:
        case Expression::ArrayNewFixedId: {
          for (auto* child : ChildIterator(e)) {
            if (!isSideEffectFreeArg(child)) {
              return false;
            }
          }
          return true;
        }
        case Expression::LocalSetId: {
          auto* ls = e->cast<LocalSet>();
          if (!ls->isTee() || !state) {
            return false;
          }
          auto it = state->scratchLocals.find(func->getLocalType(ls->index));
          return it != state->scratchLocals.end() && it->second == ls->index &&
                 isSideEffectFreeArg(ls->value);
        }
        case Expression::CallId: {
          auto* c = e->cast<Call>();
          if (c->isReturn || !c->type.isRef()) {
            return false;
          }
          for (auto* op : c->operands) {
            if (!isSideEffectFreeArg(op)) {
              return false;
            }
          }
          return true;
        }
        default:
          return false;
      }
    }

    void substInCopy(Expression** p,
                     const ExpressionList& args,
                     Index scratchIdx,
                     Index callerScratch) {
      Expression* e = *p;
      if (auto* lg = e->dynCast<LocalGet>()) {
        if (lg->index < args.size()) {
          *p = ExpressionManipulator::copy(args[lg->index], wasm);
        } else if (lg->index == scratchIdx) {
          lg->index = callerScratch;
          lg->type = func->getLocalType(callerScratch);
        }
        return;
      }
      if (auto* ls = e->dynCast<LocalSet>()) {
        if (ls->index == scratchIdx) {
          ls->index = callerScratch;
          if (ls->isTee()) {
            ls->type = func->getLocalType(callerScratch);
          }
        }
      }
      ChildIterator it(e);
      for (Index i = 0; i < it.getNumChildren(); i++) {
        substInCopy(it.children[i], args, scratchIdx, callerScratch);
      }
      if (auto* rc = (*p)->dynCast<RefCast>()) {
        if (Type::isSubType(rc->ref->type, rc->type)) {
          *p = rc->ref;
        }
      }
      if (auto* b = (*p)->dynCast<Block>()) {
        if (!b->name.is() && !b->list.empty() && b->type.isRef()) {
          b->type = b->list.back()->type;
        }
      }
      if (auto* cr = (*p)->dynCast<CallRef>()) {
        if (!cr->isReturn) {
          Expression* t = cr->target;
          while (auto* rc = t->dynCast<RefCast>()) {
            t = rc->ref;
          }
          if (auto* rf = t->dynCast<RefFunc>()) {
            Builder builder(wasm);
            *p = builder.makeCall(
              rf->func,
              std::vector<Expression*>(cr->operands.begin(), cr->operands.end()),
              cr->type);
          }
        }
      }
    }

    Expression* tryInlineCall(Call* call) {
      if (call->target == func->name) {
        return nullptr;
      }
      auto* callee = wasm.getFunction(call->target);
      if (callee->imported() || callee->getNumVars() > 1) {
        return nullptr;
      }
      Index numParams = callee->getNumParams();
      if (call->operands.size() != numParams) {
        return nullptr;
      }
      if (!(state && state->inArrayNewFixed)) {
        if (numParams > 0 || callee->getNumVars() > 0) {
          return nullptr;
        }
      }
      Index scratchIdx = numParams;
      Index callerScratch = 0;
      if (callee->getNumVars() == 1) {
        if (!state) {
          return nullptr;
        }
        Type varType = callee->getLocalType(scratchIdx);
        auto* b = callee->body->dynCast<Block>();
        if (!b || b->name.is() || b->list.size() != 2) {
          return nullptr;
        }
        auto* endGet = b->list[1]->dynCast<LocalGet>();
        if (!endGet || endGet->index != scratchIdx) {
          return nullptr;
        }
        Index scratchGets = 0;
        Index scratchTees = 0;
        Index scratchSets = 0;
        LocalSet* scratchTee = nullptr;
        for (auto* lg : FindAll<LocalGet>(b->list[0]).list) {
          if (lg->index == scratchIdx) {
            scratchGets++;
          }
        }
        for (auto* ls : FindAll<LocalSet>(b->list[0]).list) {
          if (ls->index == scratchIdx) {
            scratchSets++;
            if (ls->isTee()) {
              scratchTees++;
              scratchTee = ls;
            }
          }
        }
        if (scratchGets != 0 || scratchSets != 1 || scratchTees != 1 ||
            !scratchTee) {
          return nullptr;
        }
        Type bestType = Type::none;
        bool foundScratch = false;
        for (auto& [candType, candIdx] : state->scratchLocals) {
          if (Type::isSubType(scratchTee->value->type, candType) &&
              Type::isSubType(candType, varType)) {
            if (!foundScratch || Type::isSubType(candType, bestType)) {
              bestType = candType;
              callerScratch = candIdx;
              foundScratch = true;
            }
          }
        }
        if (!foundScratch) {
          return nullptr;
        }
      }
      if (Measurer::measure(callee->body) > MaxInlinedNodes) {
        return nullptr;
      }
      std::vector<Index> paramGets(numParams, 0);
      for (auto* lg : FindAll<LocalGet>(callee->body).list) {
        if (lg->index < numParams) {
          paramGets[lg->index]++;
        }
      }
      for (auto* ls : FindAll<LocalSet>(callee->body).list) {
        if (ls->index < numParams) {
          return nullptr;
        }
      }
      for (Index i = 0; i < numParams; i++) {
        Expression* arg = call->operands[i];
        if (!isSideEffectFreeArg(arg)) {
          return nullptr;
        }
        if (paramGets[i] > 1 && !arg->is<Const>() && !arg->is<GlobalGet>() &&
            !arg->is<RefNull>() && !arg->is<RefFunc>()) {
          return nullptr;
        }
      }
      Expression* copy = ExpressionManipulator::copy(callee->body, wasm);
      substInCopy(&copy, call->operands, scratchIdx, callerScratch);
      return copy;
    }

    bool scan(Expression** ptr, Index depth) {
      auto* curr = *ptr;
      if (depth > MaxDepth || curr->type == Type::unreachable) {
        return false;
      }
      switch (curr->_id) {
        case Expression::LocalGetId: {
          size_t digest = hash(curr->_id);
          rehash(digest, curr->cast<LocalGet>()->index);
          push(digest);
          stmt.fixedBytes += LocalGetBytes;
          return true;
        }
        case Expression::ConstId: {
          auto* c = curr->cast<Const>();
          size_t digest = hash(curr->_id);
          rehash(digest, c->type.getID());
          push(digest);
          stmt.leaves.push_back({ptr,
                                 LeafKind::Const,
                                 c->value,
                                 Name(),
                                 Name(),
                                 c->type,
                                 getConstBytes(c->value)});
          return true;
        }
        case Expression::RefNullId: {
          if (curr->type.getHeapType().getBottom() == HeapType::none) {
            size_t digest = hash(Expression::GlobalGetId);
            rehash(digest, size_t(2));
            pushRefLeaf(digest);
            stmt.leaves.push_back({ptr,
                                   LeafKind::Global,
                                   Literal(),
                                   Name(),
                                   Name(),
                                   curr->type,
                                   2});
            return true;
          }
          break;
        }
        case Expression::GlobalGetId: {
          auto* get = curr->cast<GlobalGet>();
          auto* global = wasm.getGlobal(get->name);
          size_t digest = hash(curr->_id);
          if (!global->mutable_ && !global->imported()) {
            if (auto* c = global->init->dynCast<Const>()) {
              // A constant in disguise: the same shape as a Const.
              digest = hash(Expression::ConstId);
              rehash(digest, c->type.getID());
              push(digest);
              stmt.leaves.push_back({ptr,
                                     LeafKind::Const,
                                     c->value,
                                     Name(),
                                     Name(),
                                     c->type,
                                     GlobalGetBytes});
              return true;
            }
          }
          if (!global->mutable_ && global->type.isRef()) {
            rehash(digest, size_t(2));
            pushRefLeaf(digest);
            stmt.leaves.push_back({ptr,
                                   LeafKind::Global,
                                   Literal(),
                                   get->name,
                                   Name(),
                                   global->type,
                                   GlobalGetBytes});
            return true;
          }
          rehash(digest, get->name);
          push(digest);
          stmt.fixedBytes += GlobalGetBytes;
          return true;
        }
        case Expression::RefCastId: {
          auto* rc = curr->cast<RefCast>();
          if (state && state->inArrayNewFixed &&
              Type::isSubType(rc->ref->type, rc->type)) {
            stmt.inlined.push_back({ptr, curr});
            *ptr = rc->ref;
            return scan(ptr, depth);
          }
          break;
        }
        case Expression::BlockId: {
          auto* block = curr->cast<Block>();
          if (!state || !state->inArrayNewFixed || block->name.is() ||
              state->plannedBlocks.count(block)) {
            return false;
          }
          if (block->list.size() == 2) {
            if (auto* endGet = block->list[1]->dynCast<LocalGet>()) {
              Type curType = func->getLocalType(endGet->index);
              auto it = state->scratchLocals.find(curType);
              if (it != state->scratchLocals.end() &&
                  it->second == endGet->index) {
                LocalSet* scratchTee = nullptr;
                for (auto* ls : FindAll<LocalSet>(block->list[0]).list) {
                  if (ls->index == endGet->index && ls->isTee()) {
                    scratchTee = ls;
                    break;
                  }
                }
                if (scratchTee) {
                  Type bestType = curType;
                  Index bestIdx = endGet->index;
                  for (auto& [candType, candIdx] : state->scratchLocals) {
                    if (Type::isSubType(scratchTee->value->type, candType) &&
                        Type::isSubType(candType, bestType) &&
                        candType != bestType) {
                      bestType = candType;
                      bestIdx = candIdx;
                    }
                  }
                  if (bestIdx != endGet->index) {
                    auto* copy =
                      ExpressionManipulator::copy(block, wasm)->cast<Block>();
                    for (auto* ls : FindAll<LocalSet>(copy->list[0]).list) {
                      if (ls->index == endGet->index && ls->isTee()) {
                        ls->index = bestIdx;
                        ls->type = bestType;
                      }
                    }
                    auto* copyGet = copy->list[1]->cast<LocalGet>();
                    copyGet->index = bestIdx;
                    copyGet->type = bestType;
                    copy->type = bestType;
                    stmt.inlined.push_back({ptr, curr});
                    *ptr = copy;
                    curr = copy;
                  }
                }
              }
            }
          }
          break;
        }
        // Control flow and anything that must not be moved into a loop body.
        case Expression::IfId:
        case Expression::LoopId:
        case Expression::BreakId:
        case Expression::SwitchId:
        case Expression::ReturnId:
        case Expression::TryId:
        case Expression::TryTableId:
        case Expression::ThrowId:
        case Expression::RethrowId:
        case Expression::ThrowRefId:
        case Expression::PopId:
        case Expression::UnreachableId:
        case Expression::BrOnId:
        case Expression::ContNewId:
        case Expression::ContBindId:
        case Expression::SuspendId:
        case Expression::ResumeId:
        case Expression::ResumeThrowId:
        case Expression::StackSwitchId:
          return false;
        case Expression::CallId: {
          auto* call = curr->cast<Call>();
          if (call->isReturn) {
            return false;
          }
          if (call->operands.size() == 1 && call->type.isRef()) {
            Literal val;
            Index argBytes = 0;
            if (getConstI32(call->operands[0], val, argBytes)) {
              size_t digest = hash(curr->_id);
              rehash(digest, call->target);
              pushRefLeaf(digest);
              stmt.leaves.push_back({ptr,
                                     LeafKind::IndexedCall,
                                     val,
                                     Name(),
                                     call->target,
                                     call->type,
                                     CallBytes + argBytes});
              return true;
            }
            if (auto* innerCall = call->operands[0]->dynCast<Call>()) {
              if (!innerCall->isReturn && innerCall->operands.size() == 1 &&
                  getConstI32(innerCall->operands[0], val, argBytes)) {
                auto it = pass.composedWrappers.find(
                  {call->target, innerCall->target});
                if (it != pass.composedWrappers.end()) {
                  Name wrapper = it->second;
                  size_t digest = hash(curr->_id);
                  rehash(digest, wrapper);
                  pushRefLeaf(digest);
                  stmt.leaves.push_back(
                    {ptr,
                     LeafKind::IndexedCall,
                     val,
                     Name(),
                     wrapper,
                     wasm.getFunction(wrapper)->getResults(),
                     CallBytes + CallBytes + argBytes});
                  return true;
                }
              }
            }
          }
          if (state && state->inArrayNewFixed && depth >= 2 &&
              call->type.isRef() && !call->operands.empty()) {
            bool allRefArgs = true;
            for (auto* op : call->operands) {
              if (!op->type.isRef()) {
                allRefArgs = false;
                break;
              }
            }
            if (allRefArgs && isClosedRefExpr(call)) {
              size_t digest = hash(Expression::GlobalGetId);
              rehash(digest, size_t(2));
              pushRefLeaf(digest);
              stmt.leaves.push_back({ptr,
                                     LeafKind::Global,
                                     Literal(),
                                     Name(),
                                     Name(),
                                     call->type,
                                     getTreeBytes(call),
                                     false,
                                     call});
              return true;
            }
          }
          if (inlineDepth < MaxInlineDepth) {
            if (Expression* inlinedBody = tryInlineCall(call)) {
              Index origBytes = getTreeBytes(curr);
              stmt.inlined.push_back({ptr, curr});
              *ptr = inlinedBody;
              Index leavesBefore = stmt.leaves.size();
              Index fixedBefore = stmt.fixedBytes;
              int adjustBefore = stmt.inlinedAdjust;
              inlineDepth++;
              bool ok = scan(ptr, depth);
              inlineDepth--;
              if (!ok) {
                return false;
              }
              for (Index i = leavesBefore; i < stmt.leaves.size(); i++) {
                stmt.leaves[i].inlined = true;
              }
              stmt.inlinedAdjust =
                adjustBefore + int(origBytes) - int(stmt.fixedBytes - fixedBefore);
              return true;
            }
          }
          break;
        }
        case Expression::CallIndirectId:
          if (curr->cast<CallIndirect>()->isReturn) {
            return false;
          }
          break;
        case Expression::CallRefId:
          if (curr->cast<CallRef>()->isReturn) {
            return false;
          }
          break;
        default:
          break;
      }
      push(ExpressionAnalyzer::shallowHash(curr));
      stmt.fixedBytes += getShallowBytes(curr);
      ChildIterator children(curr);
      push(children.getNumChildren());
      for (Index i = 0; i < children.getNumChildren(); i++) {
        if (!scan(children.children[children.getNumChildren() - 1 - i],
                  depth + 1)) {
          return false;
        }
      }
      return true;
    }
  };

  Statement scanStatement(FuncState& state, Expression** ptr) {
    Statement stmt;
    stmt.expr = *ptr;
    if (stmt.expr->type != Type::none) {
      return stmt;
    }
    Scanner scanner(*module, stmt, *this, state.func, &state);
    stmt.valid = scanner.scan(ptr, 0);
    stmt.expr = *ptr;
    if (!stmt.valid) {
      for (auto it = stmt.inlined.rbegin(); it != stmt.inlined.rend(); ++it) {
        *it->first = it->second;
      }
      stmt.inlined.clear();
      stmt.expr = *ptr;
    }
    return stmt;
  }

  // Planning.

  void findScratchLocals(FuncState& state) {
    auto* func = state.func;
    Index numParams = func->getNumParams();
    Index numLocals = func->getNumLocals();
    if (numLocals <= numParams) {
      return;
    }
    std::vector<Index> totalGets(numLocals, 0);
    std::vector<Index> totalSets(numLocals, 0);
    std::vector<Index> validBlockUses(numLocals, 0);
    for (auto* lg : FindAll<LocalGet>(func->body).list) {
      totalGets[lg->index]++;
    }
    for (auto* ls : FindAll<LocalSet>(func->body).list) {
      totalSets[ls->index]++;
    }
    for (auto* b : FindAll<Block>(func->body).list) {
      if (!b->name.is() &&
          (b->list.size() == 2 ||
           (b->list.size() == 3 && isLazyInitPrefix(b->list[0])))) {
        Expression* bodyStmt = b->list[b->list.size() - 2];
        if (auto* endGet = b->list.back()->dynCast<LocalGet>()) {
          Index idx = endGet->index;
          if (idx >= numParams) {
            Index getsIn0 = 0;
            Index teesIn0 = 0;
            Index setsIn0 = 0;
            for (auto* lg : FindAll<LocalGet>(bodyStmt).list) {
              if (lg->index == idx) {
                getsIn0++;
              }
            }
            for (auto* ls : FindAll<LocalSet>(bodyStmt).list) {
              if (ls->index == idx) {
                setsIn0++;
                if (ls->isTee()) {
                  teesIn0++;
                }
              }
            }
            if (getsIn0 == 0 && setsIn0 == 1 && teesIn0 == 1) {
              validBlockUses[idx]++;
            }
          }
        }
      }
    }
    for (Index idx = numParams; idx < numLocals; idx++) {
      if (totalGets[idx] == validBlockUses[idx] &&
          totalSets[idx] == validBlockUses[idx]) {
        Type t = func->getLocalType(idx);
        if (!state.scratchLocals.count(t)) {
          state.scratchLocals[t] = idx;
        }
      }
    }
  }

  void processFunction(Function* func) {
    FuncState state;
    state.func = func;
    findScratchLocals(state);
    for (auto* block : FindAll<Block>(func->body).list) {
      planBlock(state, block);
    }
    planSwitchLadders(state);
    planArrayNewFixed(state);
    if (state.plans.empty()) {
      return;
    }
    applyPlans(state);
  }

  int planBlock(FuncState& state, Block* block) {
    auto& list = block->list;
    std::vector<Statement> stmts;
    stmts.reserve(list.size());
    for (Index i = 0; i < list.size(); i++) {
      stmts.push_back(scanStatement(state, &list[i]));
    }
    Index plansBefore = state.plans.size();
    int totalSaved = 0;
    auto findRuns = [&](Index from, Index to, bool strict, auto onRun) {
      Index i = from;
      while (i < to) {
        if (!stmts[i].valid) {
          i++;
          continue;
        }
        Index j = i + 1;
        while (j < to && stmts[j].valid &&
               (strict ? stmts[j].strictKey == stmts[i].strictKey
                       : stmts[j].key == stmts[i].key) &&
               stmts[j].leaves.size() == stmts[i].leaves.size()) {
          j++;
        }
        if (j - i >= MinRun) {
          onRun(i, j);
        }
        i = j;
      }
    };
    findRuns(0, list.size(), false, [&](Index i, Index j) {
      bool isStrict = true;
      for (Index k = i + 1; k < j; k++) {
        if (stmts[k].strictKey != stmts[i].strictKey) {
          isStrict = false;
          break;
        }
      }
      if (isStrict) {
        totalSaved += planStrictRun(state, block, stmts, i, j);
        return;
      }
      FuncState looseState = state;
      int looseSaved = 0;
      if (planRun(looseState, block, stmts, i, j - i, &looseSaved) != Rolled) {
        looseSaved = 0;
      }
      FuncState strictState = state;
      int strictSaved = 0;
      findRuns(i, j, true, [&](Index si, Index sj) {
        if (si != i || sj != j) {
          strictSaved += planStrictRun(strictState, block, stmts, si, sj);
        }
      });
      if (looseSaved > 0 && looseSaved >= strictSaved) {
        state = std::move(looseState);
        totalSaved += looseSaved;
      } else if (strictSaved > 0) {
        state = std::move(strictState);
        totalSaved += strictSaved;
      }
    });
    if (state.plans.size() > plansBefore) {
      state.plannedBlocks.insert(block);
    }
    // Undo the inlining in the statements we do not keep.
    std::set<Expression*> kept;
    for (Index p = plansBefore; p < state.plans.size(); p++) {
      kept.insert(state.plans[p].first.expr);
    }
    for (auto& stmt : stmts) {
      if (!kept.count(stmt.expr)) {
        for (auto it = stmt.inlined.rbegin(); it != stmt.inlined.rend(); ++it) {
          *it->first = it->second;
        }
      }
    }
    return totalSaved;
  }

  int planStrictRun(FuncState& state,
                    Block* block,
                    std::vector<Statement>& stmts,
                    Index si,
                    Index sj) {
    FuncState fullState = state;
    int fullSaved = 0;
    if (planRun(fullState, block, stmts, si, sj - si, &fullSaved) != Rolled) {
      fullSaved = 0;
    } else {
      bool allProg = true;
      for (auto& h : fullState.plans.back().holes) {
        if (h.kind != HolePlan::Fixed && h.kind != HolePlan::Progression) {
          allProg = false;
          break;
        }
      }
      if (allProg) {
        state = std::move(fullState);
        return fullSaved;
      }
    }

    Index numLeaves = stmts[si].leaves.size();
    bool canSplitAP = numLeaves > 0;
    for (Index h = 0; h < numLeaves && canSplitAP; h++) {
      auto& leaf = stmts[si].leaves[h];
      if (leaf.kind == LeafKind::Const && leaf.value.type != Type::i32) {
        canSplitAP = false;
      }
    }
    int splitSaved = 0;
    FuncState splitState = state;
    if (canSplitAP) {
      std::vector<std::pair<Index, Index>> apRuns;
      Index a = si;
      while (a + MinRun <= sj) {
        Index b = a + 2;
        bool stepOk = true;
        for (Index h = 0; h < numLeaves; h++) {
          if (stmts[a].leaves[h].kind == LeafKind::Global &&
              !sameGlobalLeaf(stmts[a + 1].leaves[h], stmts[a].leaves[h])) {
            stepOk = false;
            break;
          }
        }
        if (!stepOk) {
          a++;
          continue;
        }
        while (b < sj) {
          bool ok = true;
          for (Index h = 0; h < numLeaves; h++) {
            auto& la = stmts[a].leaves[h];
            auto& lb = stmts[b].leaves[h];
            if (la.kind == LeafKind::Global) {
              if (!sameGlobalLeaf(lb, la)) {
                ok = false;
                break;
              }
            } else {
              uint32_t base = la.value.geti32();
              uint32_t step =
                uint32_t(stmts[a + 1].leaves[h].value.geti32()) - base;
              if (uint32_t(lb.value.geti32()) != base + step * (b - a)) {
                ok = false;
                break;
              }
            }
          }
          if (!ok) {
            break;
          }
          b++;
        }
        if (b - a >= MinRun) {
          apRuns.push_back({a, b});
          a = b;
        } else {
          a++;
        }
      }
      if (!apRuns.empty()) {
        Index cur = si;
        for (auto [apA, apB] : apRuns) {
          if (apA - cur >= MinRun) {
            int s = 0;
            if (planRun(splitState, block, stmts, cur, apA - cur, &s) ==
                Rolled) {
              splitSaved += s;
            }
          }
          int s = 0;
          if (planRun(splitState, block, stmts, apA, apB - apA, &s) == Rolled) {
            splitSaved += s;
          }
          cur = apB;
        }
        if (sj - cur >= MinRun) {
          int s = 0;
          if (planRun(splitState, block, stmts, cur, sj - cur, &s) == Rolled) {
            splitSaved += s;
          }
        }
      }
    }

    if (fullSaved > 0 && fullSaved >= splitSaved) {
      state = std::move(fullState);
      return fullSaved;
    }
    if (splitSaved > 0) {
      state = std::move(splitState);
      return splitSaved;
    }
    return 0;
  }

  enum PlanResult { Rolled, NotWorthIt, Mismatch };

  bool findTableEntry(const Table& table, const Leaf& leaf, Index& outIdx) {
    if (leaf.refExpr) {
      for (Index i = 0; i < table.exprs.size(); i++) {
        if (table.exprs[i] &&
            ExpressionAnalyzer::equal(table.exprs[i], leaf.refExpr)) {
          outIdx = i;
          return true;
        }
      }
      return false;
    }
    auto it = table.indexes.find(leaf.global);
    if (it != table.indexes.end()) {
      outIdx = it->second;
      return true;
    }
    return false;
  }

  PlanResult planRun(FuncState& state,
                     Block* block,
                     std::vector<Statement>& stmts,
                     Index start,
                     Index size,
                     int* savedOut = nullptr) {
    auto& first = stmts[start];
    Index numLeaves = first.leaves.size();
    std::vector<HolePlan> holes(numLeaves);
    // The bytes of the statements, versus the bytes of the loop.
    int oldBytes = 0;
    for (Index k = 0; k < size; k++) {
      oldBytes += int(first.fixedBytes) + stmts[start + k].inlinedAdjust;
    }
    int newBytes = LoopBytes + first.fixedBytes;
    std::set<ColumnKind> newKinds;
    // Tables are shared across runs; work on a copy until we commit.
    std::unordered_map<Type, Index> newTables;
    std::vector<Table> tempTables = state.tables;

    auto getTable = [&](Type lub) {
      auto iter = state.tablesByType.find(lub);
      if (iter != state.tablesByType.end()) {
        return iter->second;
      }
      auto newIter = newTables.find(lub);
      if (newIter != newTables.end()) {
        return newIter->second;
      }
      Index tableIndex = tempTables.size();
      tempTables.push_back(Table{lub, {}, {}, {}, Name()});
      newTables[lub] = tableIndex;
      return tableIndex;
    };
    auto getTableIndex =
      [&](Table& table, HolePlan& plan, const Leaf& other) -> Index {
      Index idx = 0;
      if (findTableEntry(table, other, idx)) {
        return idx;
      }
      idx = table.globals.size();
      Name name = other.refExpr ? Name("$expr$" + std::to_string(idx))
                                : other.global;
      table.indexes[name] = idx;
      table.globals.push_back(name);
      table.exprs.push_back(other.refExpr);
      if (other.refExpr) {
        table.hasExprs = true;
      }
      plan.newGlobals.push_back(name);
      Index entryBytes = other.refExpr ? other.bytes : TableEntryBytes;
      if (idx == 1) {
        Index firstBytes = table.exprs[0] ? getTreeBytes(table.exprs[0])
                                          : TableEntryBytes;
        newBytes += TableBytes + firstBytes + entryBytes;
      } else if (idx > 1) {
        newBytes += entryBytes;
      }
      return idx;
    };
    auto useColumn = [&](HolePlan& plan) {
      if (!state.usedKinds.count(plan.column)) {
        newKinds.insert(plan.column);
      }
      if (plan.hasDict && !state.usedKinds.count(plan.dictColumn)) {
        newKinds.insert(plan.dictColumn);
      }
      newBytes += plan.data.size() + plan.dictData.size() + getReadBytes(plan);
    };

    for (Index h = 0; h < numLeaves; h++) {
      auto& plan = holes[h];
      auto& leaf = first.leaves[h];
      for (Index k = 0; k < size; k++) {
        auto& other = stmts[start + k].leaves[h];
        if (!other.inlined) {
          oldBytes += other.bytes;
        }
      }
      // Classify the leaves of this hole.
      bool varies = false;
      bool hasGlobal = false;
      std::vector<CalleePlan> callees;
      for (Index k = 0; k < size; k++) {
        auto& other = stmts[start + k].leaves[h];
        if (other.kind == LeafKind::Const) {
          if (leaf.kind != LeafKind::Const ||
              other.value.type != leaf.value.type) {
            return Mismatch;
          }
          if (other.value != leaf.value) {
            varies = true;
          }
        } else if (other.kind == LeafKind::Global) {
          if (leaf.kind == LeafKind::Const) {
            return Mismatch;
          }
          hasGlobal = true;
          if (other.kind != leaf.kind || !sameGlobalLeaf(other, leaf)) {
            varies = true;
          }
        } else {
          if (leaf.kind == LeafKind::Const) {
            return Mismatch;
          }
          Index calleeIdx = callees.size();
          for (Index c = 0; c < callees.size(); c++) {
            if (callees[c].name == other.callee) {
              calleeIdx = c;
              break;
            }
          }
          if (calleeIdx == callees.size()) {
            if (callees.size() >= 3) {
              return Mismatch;
            }
            callees.push_back({other.callee});
          }
          if (other.kind != leaf.kind || other.callee != leaf.callee ||
              other.value != leaf.value) {
            varies = true;
          }
        }
      }
      if (!varies) {
        plan.kind = HolePlan::Fixed;
        newBytes += leaf.bytes;
        continue;
      }
      if (leaf.kind == LeafKind::Const ||
          (callees.size() == 1 && !hasGlobal)) {
        // A numeric column, possibly passed to the factory.
        std::vector<Literal> values;
        for (Index k = 0; k < size; k++) {
          values.push_back(stmts[start + k].leaves[h].value);
        }
        if (!planConst(plan, values)) {
          return Mismatch;
        }
        if (!callees.empty()) {
          plan.callee = callees[0].name;
          newBytes += CallBytes;
        }
        if (plan.kind == HolePlan::Column || plan.kind == HolePlan::Delta) {
          useColumn(plan);
        } else {
          newBytes += getReadBytes(plan);
        }
        continue;
      }
      // A table of the distinct globals, shared per element type.
      Type lub = Type::none;
      Index tableIndex = 0;
      if (hasGlobal) {
        for (Index k = 0; k < size; k++) {
          auto& other = stmts[start + k].leaves[h];
          if (other.kind != LeafKind::Global) {
            continue;
          }
          lub = lub == Type::none ? other.type
                                  : Type::getLeastUpperBound(lub, other.type);
          if (lub == Type::none) {
            return Mismatch;
          }
        }
        tableIndex = getTable(lub);
        auto& table = tempTables[tableIndex];
        if (callees.empty()) {
          std::vector<uint32_t> indexes;
          for (Index k = 0; k < size; k++) {
            indexes.push_back(
              getTableIndex(table, plan, stmts[start + k].leaves[h]));
          }
          if (table.globals.size() > 65536) {
            return Mismatch;
          }
          plan.kind = HolePlan::Table;
          plan.table = tableIndex;
          uint32_t base = indexes[0];
          uint32_t step = indexes[1] - base;
          bool isProgression = true;
          for (Index k = 0; k < size; k++) {
            if (indexes[k] != base + step * k) {
              isProgression = false;
              break;
            }
          }
          if (isProgression) {
            plan.column = ColumnKind::Count;
            plan.base = base;
            plan.step = step;
            newBytes += getReadBytes(plan);
            continue;
          }
          uint32_t maxIndex = *std::max_element(indexes.begin(), indexes.end());
          plan.column = maxIndex <= 255 ? ColumnKind::I8 : ColumnKind::I16;
          for (auto index : indexes) {
            appendLE(plan.data, index, getColumnWidth(plan.column));
          }
          useColumn(plan);
          continue;
        }
        for (Index k = 0; k < size; k++) {
          auto& other = stmts[start + k].leaves[h];
          if (other.kind == LeafKind::Global) {
            getTableIndex(table, plan, other);
          }
        }
        if (table.globals.size() > 65536) {
          return Mismatch;
        }
      }
      // Mixed: table entries (if any) and one or more factory calls.
      for (Index k = 0; k < size; k++) {
        auto& other = stmts[start + k].leaves[h];
        if (other.kind != LeafKind::Global) {
          int32_t arg = other.value.geti32();
          if (arg < 0) {
            return Mismatch;
          }
          for (auto& c : callees) {
            if (c.name == other.callee) {
              c.minArg = std::min(c.minArg, uint32_t(arg));
              c.maxArg = std::max(c.maxArg, uint32_t(arg));
              break;
            }
          }
        }
      }
      Type mixedType = lub;
      for (auto& c : callees) {
        Type retType = module->getFunction(c.name)->getResults();
        mixedType = mixedType == Type::none
                      ? retType
                      : Type::getLeastUpperBound(mixedType, retType);
        if (mixedType == Type::none) {
          return Mismatch;
        }
      }
      Index tableSize = hasGlobal ? tempTables[tableIndex].globals.size() : 0;
      uint64_t nextBound = tableSize;
      for (auto& c : callees) {
        c.bound = Index(nextBound);
        nextBound += uint64_t(c.maxArg) + 1;
      }
      if (nextBound - 1 > 255) {
        std::stable_sort(callees.begin(),
                         callees.end(),
                         [](const CalleePlan& a, const CalleePlan& b) {
                           return a.maxArg < b.maxArg;
                         });
        uint64_t unshiftedBound = tableSize;
        uint64_t shiftedBound = tableSize;
        for (Index c = 0; c < callees.size(); c++) {
          unshiftedBound += uint64_t(callees[c].maxArg) + 1;
          uint32_t base =
            (c > 0 || hasGlobal || callees[c].maxArg > 255) ? callees[c].minArg
                                                            : 0;
          shiftedBound += uint64_t(callees[c].maxArg - base) + 1;
        }
        bool useShift =
          (shiftedBound - 1 <= 255 && unshiftedBound - 1 > 255) ||
          (shiftedBound - 1 <= 65535 && unshiftedBound - 1 > 65535);
        nextBound = tableSize;
        for (Index c = 0; c < callees.size(); c++) {
          callees[c].baseArg =
            useShift && (c > 0 || hasGlobal || callees[c].maxArg > 255)
              ? callees[c].minArg
              : 0;
          callees[c].bound = Index(nextBound);
          nextBound += uint64_t(callees[c].maxArg - callees[c].baseArg) + 1;
        }
      }
      if (nextBound > 0x80000000ULL ||
          (callees.size() > 1 && nextBound - 1 > 65535)) {
        return Mismatch;
      }
      std::vector<uint32_t> indexes;
      indexes.reserve(size);
      uint32_t maxIndex = 0;
      for (Index k = 0; k < size; k++) {
        auto& other = stmts[start + k].leaves[h];
        uint32_t index = 0;
        if (other.kind == LeafKind::Global) {
          findTableEntry(tempTables[tableIndex], other, index);
        } else {
          Index c = 0;
          while (callees[c].name != other.callee) {
            c++;
          }
          index = callees[c].bound + uint32_t(other.value.geti32()) -
                  callees[c].baseArg;
        }
        maxIndex = std::max(maxIndex, index);
        indexes.push_back(index);
      }
      plan.kind = HolePlan::Mixed;
      plan.table = tableIndex;
      plan.tableSize = tableSize;
      plan.callees = std::move(callees);
      plan.mixedType = mixedType;
      plan.column = maxIndex <= 255     ? ColumnKind::I8
                    : maxIndex <= 65535 ? ColumnKind::I16
                                        : ColumnKind::I32;
      for (auto index : indexes) {
        appendLE(plan.data, index, getColumnWidth(plan.column));
      }
      useColumn(plan);
    }
    newBytes += newKinds.size() * ArraySetupBytes;
    if (newBytes >= oldBytes) {
      return NotWorthIt;
    }
    int saved = oldBytes - newBytes;
    if (savedOut) {
      *savedOut = saved;
    }
    // Commit.
    state.tables = std::move(tempTables);
    for (auto& [type, index] : newTables) {
      state.tablesByType[type] = index;
    }
    for (auto kind : newKinds) {
      state.usedKinds.insert(kind);
      state.kindOrder.push_back(kind);
    }
    for (auto& plan : holes) {
      if (plan.hasDict) {
        auto& dData = state.arrayData[plan.dictColumn];
        plan.dictOffset = state.arrayCounts[plan.dictColumn];
        state.arrayCounts[plan.dictColumn] += plan.dictCount;
        dData.insert(dData.end(), plan.dictData.begin(), plan.dictData.end());
      }
      if (plan.kind == HolePlan::Column || plan.kind == HolePlan::Delta ||
          (plan.kind == HolePlan::Table && plan.column != ColumnKind::Count) ||
          plan.kind == HolePlan::Mixed) {
        auto& data = state.arrayData[plan.column];
        plan.offset = state.arrayCounts[plan.column];
        state.arrayCounts[plan.column] += size;
        data.insert(data.end(), plan.data.begin(), plan.data.end());
      }
    }
    state.plans.push_back(RunPlan{block,
                                  start,
                                  size,
                                  first,
                                  std::move(holes),
                                  nullptr,
                                  nullptr,
                                  0,
                                  Name(),
                                  false,
                                  nullptr,
                                  saved});
    return Rolled;
  }

  // Plans a varying constant leaf. Returns false if it cannot be handled.
  bool planConst(HolePlan& plan,
                 const std::vector<Literal>& values,
                 bool allowDelta = true,
                 bool allowProgression = true) {
    auto type = values[0].type;
    Index size = values.size();
    if (type == Type::i32) {
      if (allowProgression) {
        // An arithmetic progression needs no data at all.
        uint32_t base = values[0].geti32();
        uint32_t step = uint32_t(values[1].geti32()) - base;
        bool progression = true;
        for (Index k = 0; k < size; k++) {
          if (uint32_t(values[k].geti32()) != base + step * k) {
            progression = false;
            break;
          }
        }
        if (progression) {
          plan.kind = HolePlan::Progression;
          plan.base = base;
          plan.step = step;
          return true;
        }
      }
      int32_t min = values[0].geti32();
      int32_t max = min;
      for (auto& value : values) {
        min = std::min(min, value.geti32());
        max = std::max(max, value.geti32());
      }
      plan.kind = HolePlan::Column;
      if (min >= 0 && max <= 255) {
        plan.column = ColumnKind::I8;
      } else if (min >= -128 && max <= 127) {
        plan.column = ColumnKind::I8;
        plan.signed_ = true;
      } else if (allowDelta && planDelta(plan, values)) {
        return true;
      } else if (min >= 0 && max <= 65535) {
        plan.column = ColumnKind::I16;
      } else if (min >= -32768 && max <= 32767) {
        plan.column = ColumnKind::I16;
        plan.signed_ = true;
      } else {
        plan.column = ColumnKind::I32;
      }
      if (plan.column == ColumnKind::I32 && size >= 8) {
        std::vector<uint32_t> distinct;
        std::unordered_map<uint32_t, uint32_t> valToIdx;
        std::vector<uint32_t> idxs;
        idxs.reserve(size);
        bool okDict = true;
        for (auto& value : values) {
          uint32_t v = uint32_t(value.geti32());
          auto it = valToIdx.find(v);
          if (it == valToIdx.end()) {
            if (distinct.size() >= 256) {
              okDict = false;
              break;
            }
            uint32_t idx = distinct.size();
            valToIdx[v] = idx;
            distinct.push_back(v);
            idxs.push_back(idx);
          } else {
            idxs.push_back(it->second);
          }
        }
        if (okDict && 4 * distinct.size() + size + 10 < 4 * size) {
          plan.hasDict = true;
          plan.dictColumn = ColumnKind::I32;
          plan.column = ColumnKind::I8;
          plan.signed_ = false;
          plan.dictCount = distinct.size();
          for (uint32_t v : distinct) {
            appendLE(plan.dictData, v, 4);
          }
          for (uint32_t idx : idxs) {
            appendLE(plan.data, idx, 1);
          }
          return true;
        }
      }
      for (auto& value : values) {
        appendLE(
          plan.data, uint32_t(value.geti32()), getColumnWidth(plan.column));
      }
      return true;
    }
    plan.kind = HolePlan::Column;
    if (type == Type::i64) {
      plan.column = ColumnKind::I64;
      for (auto& value : values) {
        appendLE(plan.data, uint64_t(value.geti64()), 8);
      }
    } else if (type == Type::f32) {
      plan.column = ColumnKind::F32;
      for (auto& value : values) {
        appendLE(plan.data, uint32_t(value.reinterpreti32()), 4);
      }
    } else if (type == Type::f64) {
      plan.column = ColumnKind::F64;
      for (auto& value : values) {
        appendLE(plan.data, uint64_t(value.reinterpreti64()), 8);
      }
    } else {
      return false;
    }
    return true;
  }

  struct ChildPtrFinder
    : public PostWalker<ChildPtrFinder,
                        UnifiedExpressionVisitor<ChildPtrFinder>> {
    std::unordered_map<Expression*, Expression*> parents;
    std::unordered_map<Expression*, Expression**> parentPtrs;
    void visitExpression(Expression* curr) {
      ChildIterator children(curr);
      for (Index i = 0; i < children.getNumChildren(); i++) {
        Expression** ptr = children.children[i];
        parents[*ptr] = curr;
        parentPtrs[*ptr] = ptr;
      }
    }
  };

  void planSwitchLadders(FuncState& state) {
    auto switches = FindAll<Switch>(state.func->body).list;
    if (switches.empty()) {
      return;
    }
    ChildPtrFinder finder;
    finder.parentPtrs[state.func->body] = &state.func->body;
    finder.walk(state.func->body);
    for (auto* sw : switches) {
      planSwitch(state, sw, finder);
    }
  }

  void planSwitch(FuncState& state, Switch* sw, ChildPtrFinder& finder) {
    if (sw->value || !sw->default_.is() ||
        sw->condition->type == Type::unreachable ||
        sw->targets.size() < MinRun) {
      return;
    }
    std::vector<Block*> chain;
    std::set<Name> chainNames;
    Expression* curr = sw;
    Block* bDef = nullptr;
    while (true) {
      auto it = finder.parents.find(curr);
      if (it == finder.parents.end()) {
        break;
      }
      auto* block = it->second->dynCast<Block>();
      if (!block || block->list.empty() || block->list[0] != curr ||
          block->type != Type::none) {
        break;
      }
      chain.push_back(block);
      if (block->name.is()) {
        chainNames.insert(block->name);
      }
      if (block->name == sw->default_) {
        bDef = block;
        break;
      }
      curr = block;
    }
    if (!bDef || chain.size() < 2) {
      return;
    }
    auto ptrIt = finder.parentPtrs.find(bDef);
    if (ptrIt == finder.parentPtrs.end()) {
      return;
    }
    std::reverse(chain.begin(), chain.end());
    if (chain.back()->list.size() != 1) {
      return;
    }
    bool condBranchesToChain = false;
    BranchUtils::operateOnScopeNameUses(sw->condition, [&](Name& name) {
      if (chainNames.count(name)) {
        condBranchesToChain = true;
      }
    });
    if (condBranchesToChain) {
      return;
    }

    std::unordered_map<Name, Index> nameToIdx;
    for (Index k = 0; k < chain.size(); k++) {
      if (chain[k]->name.is()) {
        nameToIdx[chain[k]->name] = k;
      }
    }
    Index M = sw->targets.size();
    std::vector<Index> targetEffK(M);
    std::vector<Index> distinctEffKs;
    std::unordered_map<Index, Index> effKToArmIdx;
    bool hasGaps = false;
    for (Index i = 0; i < M; i++) {
      auto it = nameToIdx.find(sw->targets[i]);
      if (it == nameToIdx.end()) {
        return;
      }
      Index effK = it->second;
      while (effK >= 1 && chain[effK - 1]->list.size() == 1) {
        effK--;
      }
      targetEffK[i] = effK;
      if (effK == 0) {
        hasGaps = true;
      } else if (!effKToArmIdx.count(effK)) {
        effKToArmIdx[effK] = distinctEffKs.size();
        distinctEffKs.push_back(effK);
      }
    }
    if (distinctEffKs.size() < MinRun) {
      return;
    }
    for (Index k = 1; k < chain.size(); k++) {
      if (chain[k - 1]->list.size() > 1 && !effKToArmIdx.count(k)) {
        return;
      }
    }

    int switchKind = -1;
    Name switchExitLabel;
    std::vector<Statement> armStmts;
    armStmts.reserve(distinctEffKs.size());
    bool committed = false;
    auto cleanup = [&]() {
      if (!committed) {
        for (auto& s : armStmts) {
          for (auto it = s.inlined.rbegin(); it != s.inlined.rend(); ++it) {
            *it->first = it->second;
          }
        }
      }
    };

    for (Index d = 0; d < distinctEffKs.size(); d++) {
      Index effK = distinctEffKs[d];
      Block* armBlock = chain[effK - 1];
      int armKind = -1;
      Name armExit;
      Index armFixed = 0;
      Expression** valPtr = nullptr;
      if (armBlock->list.size() == 2) {
        if (auto* ret = armBlock->list[1]->dynCast<Return>()) {
          if (ret->value) {
            armKind = 1;
            armFixed = 1;
            valPtr = &ret->value;
          }
        } else if (auto* br = armBlock->list[1]->dynCast<Break>()) {
          if (!br->condition && br->value && !chainNames.count(br->name)) {
            armKind = 2;
            armExit = br->name;
            armFixed = 2;
            valPtr = &br->value;
          }
        } else if (effK == 1 && armBlock->list[1]->type == Type::none) {
          armKind = 0;
          armExit = bDef->name;
          armFixed = 0;
          valPtr = &armBlock->list[1];
        }
      } else if (armBlock->list.size() == 3 &&
                 armBlock->list[1]->type == Type::none) {
        if (auto* br = armBlock->list[2]->dynCast<Break>()) {
          if (!br->condition && !br->value &&
              (br->name == bDef->name || !chainNames.count(br->name))) {
            armKind = 0;
            armExit = br->name;
            armFixed = 2;
            valPtr = &armBlock->list[1];
          }
        }
      }
      if (armKind < 0) {
        cleanup();
        return;
      }
      if (d == 0) {
        switchKind = armKind;
        switchExitLabel = armExit;
      } else if (armKind != switchKind || armExit != switchExitLabel) {
        cleanup();
        return;
      }
      Statement stmt;
      stmt.fixedBytes = armFixed;
      Scanner scanner(*module, stmt, *this, state.func, &state);
      stmt.valid = scanner.scan(valPtr, 0);
      stmt.expr = *valPtr;
      armStmts.push_back(std::move(stmt));
      if (!armStmts.back().valid || armStmts.back().key != armStmts[0].key ||
          armStmts.back().leaves.size() != armStmts[0].leaves.size()) {
        cleanup();
        return;
      }
    }

    auto& first = armStmts[0];
    Index numLeaves = first.leaves.size();
    std::vector<HolePlan> holes(numLeaves);
    int oldBytes = int(3 * chain.size() + 3 + M);
    for (auto& s : armStmts) {
      oldBytes += int(s.fixedBytes) + s.inlinedAdjust;
      for (auto& leaf : s.leaves) {
        if (!leaf.inlined) {
          oldBytes += int(leaf.bytes);
        }
      }
    }
    Index rolledFixedBytes = first.fixedBytes;
    if (switchKind == 0 && switchExitLabel == bDef->name &&
        rolledFixedBytes >= 2) {
      rolledFixedBytes -= 2;
    }
    int newBytes = 8 + int(getConstBytes(Literal(int32_t(M)))) +
                   int(rolledFixedBytes) + (hasGaps ? 10 : 0);
    std::set<ColumnKind> newKinds;
    std::unordered_map<Type, Index> newTables;
    std::vector<Table> tempTables = state.tables;

    auto getTable = [&](Type lub) {
      auto iter = state.tablesByType.find(lub);
      if (iter != state.tablesByType.end()) {
        return iter->second;
      }
      auto newIter = newTables.find(lub);
      if (newIter != newTables.end()) {
        return newIter->second;
      }
      Index tableIndex = tempTables.size();
      tempTables.push_back(Table{lub, {}, {}, {}, Name()});
      newTables[lub] = tableIndex;
      return tableIndex;
    };
    auto getTableIndex =
      [&](Table& table, HolePlan& plan, const Leaf& other) -> Index {
      Index idx = 0;
      if (findTableEntry(table, other, idx)) {
        return idx;
      }
      idx = table.globals.size();
      Name name = other.refExpr ? Name("$expr$" + std::to_string(idx))
                                : other.global;
      table.indexes[name] = idx;
      table.globals.push_back(name);
      table.exprs.push_back(other.refExpr);
      if (other.refExpr) {
        table.hasExprs = true;
      }
      plan.newGlobals.push_back(name);
      Index entryBytes = other.refExpr ? other.bytes : TableEntryBytes;
      if (idx == 1) {
        Index firstBytes = table.exprs[0] ? getTreeBytes(table.exprs[0])
                                          : TableEntryBytes;
        newBytes += TableBytes + firstBytes + entryBytes;
      } else if (idx > 1) {
        newBytes += entryBytes;
      }
      return idx;
    };
    auto useColumn = [&](HolePlan& plan) {
      if (!state.usedKinds.count(plan.column)) {
        newKinds.insert(plan.column);
      }
      if (plan.hasDict && !state.usedKinds.count(plan.dictColumn)) {
        newKinds.insert(plan.dictColumn);
      }
      newBytes += plan.data.size() + plan.dictData.size() + getReadBytes(plan);
    };

    bool foundGapHole = !hasGaps;
    for (Index h = 0; h < numLeaves; h++) {
      auto& plan = holes[h];
      auto& leaf = first.leaves[h];
      bool varies = false;
      bool hasGlobal = false;
      std::vector<CalleePlan> callees;
      for (auto& s : armStmts) {
        auto& other = s.leaves[h];
        if (other.kind == LeafKind::Const) {
          if (leaf.kind != LeafKind::Const ||
              other.value.type != leaf.value.type) {
            cleanup();
            return;
          }
          if (other.value != leaf.value) {
            varies = true;
          }
        } else if (other.kind == LeafKind::Global) {
          if (leaf.kind == LeafKind::Const) {
            cleanup();
            return;
          }
          hasGlobal = true;
          if (other.kind != leaf.kind || !sameGlobalLeaf(other, leaf)) {
            varies = true;
          }
        } else {
          if (leaf.kind == LeafKind::Const) {
            cleanup();
            return;
          }
          Index calleeIdx = callees.size();
          for (Index c = 0; c < callees.size(); c++) {
            if (callees[c].name == other.callee) {
              calleeIdx = c;
              break;
            }
          }
          if (calleeIdx == callees.size()) {
            if (callees.size() >= 3) {
              cleanup();
              return;
            }
            callees.push_back({other.callee});
          }
          if (other.kind != leaf.kind || other.callee != leaf.callee ||
              other.value != leaf.value) {
            varies = true;
          }
        }
      }
      if (!varies) {
        plan.kind = HolePlan::Fixed;
        newBytes += leaf.bytes;
        continue;
      }
      if (leaf.kind == LeafKind::Const ||
          (callees.size() == 1 && !hasGlobal)) {
        if (!foundGapHole && leaf.value.type == Type::i32) {
          int32_t min = armStmts[0].leaves[h].value.geti32();
          int32_t max = min;
          for (auto& s : armStmts) {
            int32_t v = s.leaves[h].value.geti32();
            min = std::min(min, v);
            max = std::max(max, v);
          }
          if (min >= 1 && max <= 255) {
            plan.isGapHole = true;
            plan.column = ColumnKind::I8;
            plan.gapOffset = 0;
          } else if (min >= 0 && max <= 254) {
            plan.isGapHole = true;
            plan.column = ColumnKind::I8;
            plan.gapOffset = 1;
          } else if (min >= 1 && max <= 65535) {
            plan.isGapHole = true;
            plan.column = ColumnKind::I16;
            plan.gapOffset = 0;
          } else if (min >= 0 && max <= 65534) {
            plan.isGapHole = true;
            plan.column = ColumnKind::I16;
            plan.gapOffset = 1;
          }
          if (plan.isGapHole) {
            foundGapHole = true;
            plan.kind = HolePlan::Column;
            for (Index i = 0; i < M; i++) {
              uint32_t v =
                targetEffK[i] == 0
                  ? 0
                  : uint32_t(armStmts[effKToArmIdx[targetEffK[i]]]
                               .leaves[h]
                               .value.geti32()) +
                      plan.gapOffset;
              appendLE(plan.data, v, getColumnWidth(plan.column));
            }
          }
        }
        if (!plan.isGapHole) {
          std::vector<Literal> values;
          values.reserve(M);
          for (Index i = 0; i < M; i++) {
            if (targetEffK[i] == 0) {
              values.push_back(Literal::makeZero(leaf.value.type));
            } else {
              values.push_back(
                armStmts[effKToArmIdx[targetEffK[i]]].leaves[h].value);
            }
          }
          if (!planConst(plan, values, false, !hasGaps)) {
            cleanup();
            return;
          }
        }
        if (!callees.empty()) {
          plan.callee = callees[0].name;
          newBytes += CallBytes;
        }
        if (plan.kind == HolePlan::Column) {
          useColumn(plan);
        } else {
          newBytes += getReadBytes(plan);
        }
        continue;
      }

      Type lub = Type::none;
      Index tableIndex = 0;
      if (hasGlobal) {
        for (auto& s : armStmts) {
          auto& other = s.leaves[h];
          if (other.kind != LeafKind::Global) {
            continue;
          }
          lub = lub == Type::none ? other.type
                                  : Type::getLeastUpperBound(lub, other.type);
          if (lub == Type::none) {
            cleanup();
            return;
          }
        }
        tableIndex = getTable(lub);
        auto& table = tempTables[tableIndex];
        if (callees.empty()) {
          for (auto& s : armStmts) {
            getTableIndex(table, plan, s.leaves[h]);
          }
          if (!foundGapHole) {
            plan.isGapHole = true;
            foundGapHole = true;
            plan.gapOffset = 1;
          }
          if (table.globals.size() + plan.gapOffset > 65536) {
            cleanup();
            return;
          }
          std::vector<uint32_t> indexes;
          indexes.reserve(M);
          uint32_t maxIndex = 0;
          for (Index i = 0; i < M; i++) {
            uint32_t idx = 0;
            if (targetEffK[i] != 0) {
              findTableEntry(
                table, armStmts[effKToArmIdx[targetEffK[i]]].leaves[h], idx);
              idx += plan.gapOffset;
            }
            maxIndex = std::max(maxIndex, idx);
            indexes.push_back(idx);
          }
          plan.kind = HolePlan::Table;
          plan.table = tableIndex;
          plan.column = maxIndex <= 255 ? ColumnKind::I8 : ColumnKind::I16;
          for (auto index : indexes) {
            appendLE(plan.data, index, getColumnWidth(plan.column));
          }
          useColumn(plan);
          continue;
        }
        for (auto& s : armStmts) {
          auto& other = s.leaves[h];
          if (other.kind == LeafKind::Global) {
            getTableIndex(table, plan, other);
          }
        }
      }
      for (auto& s : armStmts) {
        auto& other = s.leaves[h];
        if (other.kind != LeafKind::Global) {
          int32_t arg = other.value.geti32();
          if (arg < 0) {
            cleanup();
            return;
          }
          for (auto& c : callees) {
            if (c.name == other.callee) {
              c.minArg = std::min(c.minArg, uint32_t(arg));
              c.maxArg = std::max(c.maxArg, uint32_t(arg));
              break;
            }
          }
        }
      }
      Type mixedType = lub;
      for (auto& c : callees) {
        Type retType = module->getFunction(c.name)->getResults();
        mixedType = mixedType == Type::none
                      ? retType
                      : Type::getLeastUpperBound(mixedType, retType);
        if (mixedType == Type::none) {
          cleanup();
          return;
        }
      }
      if (!foundGapHole) {
        plan.isGapHole = true;
        foundGapHole = true;
        plan.gapOffset = (hasGlobal || callees[0].minArg == 0) ? 1 : 0;
      }
      Index tableSize = hasGlobal ? tempTables[tableIndex].globals.size() : 0;
      if (tableSize + plan.gapOffset > 65536) {
        cleanup();
        return;
      }
      uint64_t nextBound = tableSize + plan.gapOffset;
      for (auto& c : callees) {
        c.bound = Index(nextBound);
        nextBound += uint64_t(c.maxArg) + 1;
      }
      if (nextBound - 1 > 255) {
        std::stable_sort(callees.begin(),
                         callees.end(),
                         [](const CalleePlan& a, const CalleePlan& b) {
                           return a.maxArg < b.maxArg;
                         });
        if (plan.isGapHole) {
          plan.gapOffset = (hasGlobal || callees[0].minArg == 0) ? 1 : 0;
        }
        uint64_t unshiftedBound = tableSize + plan.gapOffset;
        uint64_t shiftedBound = tableSize + plan.gapOffset;
        for (Index c = 0; c < callees.size(); c++) {
          unshiftedBound += uint64_t(callees[c].maxArg) + 1;
          uint32_t base = (c > 0 || hasGlobal || plan.gapOffset > 0 ||
                           (!plan.isGapHole && callees[c].maxArg > 255))
                            ? callees[c].minArg
                            : 0;
          shiftedBound += uint64_t(callees[c].maxArg - base) + 1;
        }
        bool useShift =
          (shiftedBound - 1 <= 255 && unshiftedBound - 1 > 255) ||
          (shiftedBound - 1 <= 65535 && unshiftedBound - 1 > 65535);
        nextBound = tableSize + plan.gapOffset;
        for (Index c = 0; c < callees.size(); c++) {
          callees[c].baseArg =
            useShift && (c > 0 || hasGlobal || plan.gapOffset > 0 ||
                         (!plan.isGapHole && callees[c].maxArg > 255))
              ? callees[c].minArg
              : 0;
          callees[c].bound = Index(nextBound);
          nextBound += uint64_t(callees[c].maxArg - callees[c].baseArg) + 1;
        }
      }
      if (nextBound > 0x80000000ULL ||
          (callees.size() > 1 && nextBound - 1 > 65535)) {
        cleanup();
        return;
      }
      std::vector<uint32_t> indexes;
      indexes.reserve(M);
      uint32_t maxIndex = 0;
      for (Index i = 0; i < M; i++) {
        uint32_t index = 0;
        if (targetEffK[i] != 0) {
          auto& other = armStmts[effKToArmIdx[targetEffK[i]]].leaves[h];
          if (other.kind == LeafKind::Global) {
            findTableEntry(tempTables[tableIndex], other, index);
            index += plan.gapOffset;
          } else {
            Index c = 0;
            while (callees[c].name != other.callee) {
              c++;
            }
            index = callees[c].bound + uint32_t(other.value.geti32()) -
                    callees[c].baseArg;
          }
        }
        maxIndex = std::max(maxIndex, index);
        indexes.push_back(index);
      }
      plan.kind = HolePlan::Mixed;
      plan.table = tableIndex;
      plan.tableSize = tableSize;
      plan.callees = std::move(callees);
      plan.mixedType = mixedType;
      plan.column = maxIndex <= 255     ? ColumnKind::I8
                    : maxIndex <= 65535 ? ColumnKind::I16
                                        : ColumnKind::I32;
      for (auto index : indexes) {
        appendLE(plan.data, index, getColumnWidth(plan.column));
      }
      useColumn(plan);
    }
    if (!foundGapHole) {
      cleanup();
      return;
    }
    newBytes += newKinds.size() * ArraySetupBytes;
    if (newBytes >= oldBytes) {
      cleanup();
      return;
    }

    committed = true;
    state.tables = std::move(tempTables);
    for (auto& [type, index] : newTables) {
      state.tablesByType[type] = index;
    }
    for (auto kind : newKinds) {
      state.usedKinds.insert(kind);
      state.kindOrder.push_back(kind);
    }
    for (auto& plan : holes) {
      if (plan.hasDict) {
        auto& dData = state.arrayData[plan.dictColumn];
        plan.dictOffset = state.arrayCounts[plan.dictColumn];
        state.arrayCounts[plan.dictColumn] += plan.dictCount;
        dData.insert(dData.end(), plan.dictData.begin(), plan.dictData.end());
      }
      if (plan.kind == HolePlan::Column || plan.kind == HolePlan::Table ||
          plan.kind == HolePlan::Mixed) {
        auto& data = state.arrayData[plan.column];
        plan.offset = state.arrayCounts[plan.column];
        state.arrayCounts[plan.column] += M;
        data.insert(data.end(), plan.data.begin(), plan.data.end());
      }
    }
    Name effectiveExitLabel =
      (switchKind == 0 && switchExitLabel == bDef->name) ? Name()
                                                         : switchExitLabel;
    state.plans.push_back(RunPlan{nullptr,
                                  0,
                                  M,
                                  std::move(first),
                                  std::move(holes),
                                  ptrIt->second,
                                  sw->condition,
                                  switchKind,
                                  effectiveExitLabel,
                                  hasGaps});
  }

  bool isDefaultValue(Expression* expr, Index& bytesOut) {
    if (expr->is<RefNull>()) {
      bytesOut = 2;
      return true;
    }
    if (auto* c = expr->dynCast<Const>()) {
      if (c->value.isZero()) {
        bytesOut = getConstBytes(c->value);
        return true;
      }
    }
    if (auto* get = expr->dynCast<GlobalGet>()) {
      auto* global = module->getGlobal(get->name);
      if (!global->mutable_ && !global->imported()) {
        Index unused = 0;
        if (isDefaultValue(global->init, unused)) {
          bytesOut = GlobalGetBytes;
          return true;
        }
      }
    }
    return false;
  }

  void planArrayNewFixed(FuncState& state) {
    auto anfs = FindAll<ArrayNewFixed>(state.func->body).list;
    if (anfs.empty()) {
      return;
    }
    ChildPtrFinder finder;
    finder.parentPtrs[state.func->body] = &state.func->body;
    finder.walk(state.func->body);
    std::unordered_set<Expression*> coveredRoots;
    for (auto& plan : state.plans) {
      if (plan.block) {
        for (Index k = 0; k < plan.size; k++) {
          coveredRoots.insert(plan.block->list[plan.start + k]);
        }
      } else if (plan.switchPtr) {
        coveredRoots.insert(*plan.switchPtr);
      }
    }
    for (auto* anf : anfs) {
      planOneArrayNewFixed(state, anf, finder, coveredRoots);
    }
  }

  bool isHoistableInit(Expression* e) {
    if (!e || e->type == Type::unreachable) {
      return false;
    }
    switch (e->_id) {
      case Expression::LocalGetId:
      case Expression::ConstId:
      case Expression::RefNullId:
      case Expression::RefFuncId:
        return true;
      case Expression::GlobalGetId:
        return !module->getGlobal(e->cast<GlobalGet>()->name)->mutable_;
      case Expression::RefCastId:
        return isHoistableInit(e->cast<RefCast>()->ref);
      case Expression::CallId: {
        auto* c = e->cast<Call>();
        if (c->isReturn || c->operands.size() != 1 ||
            !isHoistableInit(c->operands[0])) {
          return false;
        }
        auto* callee = module->getFunction(c->target);
        if (callee->imported() || callee->getNumParams() != 1 ||
            callee->getNumVars() != 0) {
          return false;
        }
        if (auto* rc = callee->body->dynCast<RefCast>()) {
          if (auto* lg = rc->ref->dynCast<LocalGet>()) {
            return lg->index == 0;
          }
        }
        return false;
      }
      default:
        return false;
    }
  }

  bool isLazyInitPrefix(Expression* e) {
    if (auto* iff = e->dynCast<If>()) {
      return !iff->ifFalse && iff->type == Type::none;
    }
    if (auto* c = e->dynCast<Call>()) {
      if (!c->isReturn && c->operands.empty() && c->type == Type::none) {
        auto* callee = module->getFunction(c->target);
        if (!callee->imported() && callee->getNumLocals() == 0) {
          if (auto* iff = callee->body->dynCast<If>()) {
            return !iff->ifFalse && iff->type == Type::none;
          }
        }
      }
    }
    return false;
  }

  void planOneArrayNewFixed(FuncState& state,
                            ArrayNewFixed* anf,
                            ChildPtrFinder& finder,
                            const std::unordered_set<Expression*>& covered) {
    const Index MinArrayNewFixed = MinRun;
    const int WrapperOverhead = 8;
    Index N = anf->values.size();
    if (N < MinArrayNewFixed || anf->type == Type::unreachable ||
        !anf->type.isRef() || !anf->type.getHeapType().isArray()) {
      return;
    }
    auto ptrIt = finder.parentPtrs.find(anf);
    if (ptrIt == finder.parentPtrs.end()) {
      return;
    }
    Expression* cur = anf;
    while (cur) {
      if (covered.count(cur)) {
        return;
      }
      auto pIt = finder.parents.find(cur);
      cur = pIt != finder.parents.end() ? pIt->second : nullptr;
    }
    HeapType arrHeapType = anf->type.getHeapType();
    Field field = arrHeapType.getArray().element;
    if (field.mutable_ != Mutable || !field.type.isDefaultable()) {
      return;
    }
    for (auto* v : anf->values) {
      if (v->type == Type::unreachable) {
        return;
      }
    }

    Index teeElemIdx = N;
    LocalSet* unwrappedTee = nullptr;
    for (Index k = 0; k < N; k++) {
      if (auto* set = anf->values[k]->dynCast<LocalSet>()) {
        if (set->isTee()) {
          if (unwrappedTee) {
            unwrappedTee = nullptr;
            teeElemIdx = N;
            break;
          }
          unwrappedTee = set;
          teeElemIdx = k;
        }
      }
    }
    if (unwrappedTee &&
        (field.packedType != Field::NotPacked ||
         !Type::isSubType(field.type,
                          state.func->getLocalType(unwrappedTee->index)))) {
      unwrappedTee = nullptr;
      teeElemIdx = N;
    }
    if (unwrappedTee) {
      Index teeLocal = unwrappedTee->index;
      Index uses = 0;
      for (auto* get : FindAll<LocalGet>(anf).list) {
        if (get->index == teeLocal) {
          uses++;
        }
      }
      for (auto* set : FindAll<LocalSet>(anf).list) {
        if (set->index == teeLocal) {
          uses++;
        }
      }
      if (uses != 1) {
        unwrappedTee = nullptr;
        teeElemIdx = N;
      }
    }

    LocalSet* nestedTee = nullptr;
    Expression** nestedTeePtr = nullptr;
    if (!unwrappedTee) {
      for (auto* set : FindAll<LocalSet>(anf->values[0]).list) {
        if (!set->isTee() || !isHoistableInit(set->value)) {
          continue;
        }
        auto pIt = finder.parentPtrs.find(set);
        if (pIt == finder.parentPtrs.end()) {
          continue;
        }
        Index localIdx = set->index;
        Index getsIn0 = 0;
        for (auto* lg : FindAll<LocalGet>(anf->values[0]).list) {
          if (lg->index == localIdx) {
            getsIn0++;
          }
        }
        if (getsIn0 != 0) {
          continue;
        }
        Index setsInAnf = 0;
        Index getsInAnf = 0;
        for (auto* ls : FindAll<LocalSet>(anf).list) {
          if (ls->index == localIdx) {
            setsInAnf++;
          }
        }
        for (auto* lg : FindAll<LocalGet>(anf).list) {
          if (lg->index == localIdx) {
            getsInAnf++;
          }
        }
        if (setsInAnf == 1 && getsInAnf > 0) {
          nestedTee = set;
          nestedTeePtr = pIt->second;
          break;
        }
      }
    }

    Builder builder(*module);
    if (nestedTee) {
      *nestedTeePtr = builder.makeLocalGet(
        nestedTee->index, state.func->getLocalType(nestedTee->index));
    }

    std::vector<Expression*> hoistedPrefixes;
    Type arrRefType = anf->type;
    Index placeholderLocal = state.func->getNumLocals();
    Block* arrBlock = builder.makeBlock();
    std::vector<LocalGet*> arrGets;
    arrGets.reserve(N);
    std::vector<Index> stmtToElemIdx;
    stmtToElemIdx.reserve(N);
    std::vector<int> stmtPenalty;
    stmtPenalty.reserve(N);
    int defaultSaved = 0;
    int unrolledPenalty = 0;
    Index teeStmtIdx = N;
    for (Index k = 0; k < N; k++) {
      Expression* val =
        (k == teeElemIdx) ? unwrappedTee->value : anf->values[k];
      if (k != teeElemIdx) {
        if (auto* b = val->dynCast<Block>()) {
          if (!b->name.is() && b->list.size() == 2 &&
              b->list[0]->type == Type::none && !b->list[1]->is<LocalGet>() &&
              isLazyInitPrefix(b->list[0])) {
            hoistedPrefixes.push_back(b->list[0]);
            val = b->list[1];
          } else if (!b->name.is() && b->list.size() == 3 &&
                     b->list[0]->type == Type::none &&
                     b->list[2]->is<LocalGet>() &&
                     isLazyInitPrefix(b->list[0])) {
            hoistedPrefixes.push_back(b->list[0]);
            val = builder.makeBlock({b->list[1], b->list[2]}, b->type);
          }
        }
      }
      Index defBytes = 0;
      if (k != teeElemIdx && isDefaultValue(val, defBytes)) {
        defaultSaved += int(defBytes);
        continue;
      }
      if (k == teeElemIdx) {
        teeStmtIdx = arrBlock->list.size();
      }
      auto* arrGet = builder.makeLocalGet(placeholderLocal, arrRefType);
      arrGets.push_back(arrGet);
      stmtToElemIdx.push_back(k);
      int pen = 5 + int(getConstBytes(Literal(int32_t(k))));
      stmtPenalty.push_back(pen);
      arrBlock->list.push_back(builder.makeArraySet(
        arrGet, builder.makeConst(int32_t(k)), val, MemoryOrder::Unordered));
      unrolledPenalty += pen;
    }
    if (arrBlock->list.size() < MinRun) {
      if (nestedTee) {
        *nestedTeePtr = nestedTee;
      }
      return;
    }

    FuncState tentativeState = state;
    tentativeState.inArrayNewFixed = true;
    Index plansBefore = tentativeState.plans.size();
    int totalBlockSaved = planBlock(tentativeState, arrBlock);
    tentativeState.inArrayNewFixed = false;

    bool teeRolled = false;
    int teePenalty = 0;
    if (unwrappedTee) {
      for (Index p = plansBefore; p < tentativeState.plans.size(); p++) {
        auto& plan = tentativeState.plans[p];
        if (teeStmtIdx >= plan.start && teeStmtIdx < plan.start + plan.size) {
          teeRolled = true;
          break;
        }
      }
      if (teeRolled) {
        teePenalty = 5 + int(getConstBytes(Literal(int32_t(teeElemIdx))));
      }
    }

    Index totalRolled = 0;
    bool hasInnerBlock = false;
    for (Index p = plansBefore; p < tentativeState.plans.size(); p++) {
      totalRolled += tentativeState.plans[p].size;
      if (!FindAll<Block>(tentativeState.plans[p].first.expr).list.empty()) {
        hasInnerBlock = true;
      }
    }
    bool allowSplit = !hasInnerBlock ||
                      (tentativeState.plans.size() - plansBefore == 1 &&
                       totalRolled == N);

    int nestedTeeOverhead = nestedTee ? 2 : 0;
    int netSaved1 = (!allowSplit || (N >= 64 && totalRolled * 4 < N * 3))
                      ? -1
                      : totalBlockSaved + defaultSaved - unrolledPenalty -
                          teePenalty - WrapperOverhead - nestedTeeOverhead;

    bool canPartial = allowSplit && !unwrappedTee && !nestedTee &&
                      hoistedPrefixes.empty() &&
                      tentativeState.plans.size() > plansBefore &&
                      (N < 64 || totalRolled * 4 >= N * 3);
    int rolledPenalty = 0;
    Index rolledCount = 0;
    if (canPartial) {
      for (Index p = plansBefore; p < tentativeState.plans.size(); p++) {
        auto& plan = tentativeState.plans[p];
        int pPen = 0;
        for (Index s = plan.start; s < plan.start + plan.size; s++) {
          pPen += stmtPenalty[s];
        }
        if (plan.saved <= pPen + 2 * int(plan.size)) {
          canPartial = false;
          break;
        }
        rolledPenalty += pPen;
        rolledCount += plan.size;
      }
      if (rolledCount == N) {
        canPartial = false;
      }
    }
    int netSaved2 = canPartial ? (totalBlockSaved - rolledPenalty) -
                                   2 * int(rolledCount) - 5 - nestedTeeOverhead
                               : -1;

    if ((netSaved1 <= 0 && netSaved2 <= 0) ||
        tentativeState.plans.size() == plansBefore) {
      for (Index p = plansBefore; p < tentativeState.plans.size(); p++) {
        for (auto it = tentativeState.plans[p].first.inlined.rbegin();
             it != tentativeState.plans[p].first.inlined.rend();
             ++it) {
          *it->first = it->second;
        }
      }
      if (nestedTee) {
        *nestedTeePtr = nestedTee;
      }
      return;
    }

    Index arrLocal = Builder::addVar(state.func, arrRefType);
    for (auto* get : arrGets) {
      get->index = arrLocal;
    }

    if (canPartial && netSaved2 > netSaved1) {
      auto makeDefaultElem = [&]() -> Expression* {
        if (field.type.isRef()) {
          return builder.makeRefNull(field.type.getHeapType().getBottom());
        }
        return builder.makeConst(Literal::makeZero(field.type));
      };
      std::vector<Expression*> fullList;
      if (nestedTee) {
        nestedTee->makeSet();
        fullList.push_back(nestedTee);
      }
      fullList.push_back(builder.makeLocalSet(arrLocal, anf));
      for (Index p = plansBefore; p < tentativeState.plans.size(); p++) {
        auto& plan = tentativeState.plans[p];
        Index newStart = fullList.size();
        for (Index s = plan.start; s < plan.start + plan.size; s++) {
          anf->values[stmtToElemIdx[s]] = makeDefaultElem();
          fullList.push_back(arrBlock->list[s]);
        }
        plan.start = newStart;
      }
      fullList.push_back(builder.makeLocalGet(arrLocal, arrRefType));
      arrBlock->list.set(fullList);
      arrBlock->type = anf->type;
      *ptrIt->second = arrBlock;
      state = std::move(tentativeState);
      return;
    }

    if (unwrappedTee && !teeRolled) {
      auto* setStmt = arrBlock->list[teeStmtIdx]->cast<ArraySet>();
      unwrappedTee->value = setStmt->value;
      setStmt->value = unwrappedTee;
    }
    std::vector<Expression*> fullList;
    fullList.reserve(arrBlock->list.size() + hoistedPrefixes.size() + 4);
    if (nestedTee) {
      nestedTee->makeSet();
      fullList.push_back(nestedTee);
    }
    for (auto* prefix : hoistedPrefixes) {
      fullList.push_back(prefix);
    }
    fullList.push_back(builder.makeLocalSet(
      arrLocal,
      builder.makeArrayNew(arrHeapType, builder.makeConst(int32_t(N)))));
    Index prefixCount = fullList.size();
    for (auto* expr : arrBlock->list) {
      fullList.push_back(expr);
    }
    if (unwrappedTee && teeRolled) {
      fullList.push_back(builder.makeLocalSet(
        unwrappedTee->index,
        builder.makeArrayGet(builder.makeLocalGet(arrLocal, arrRefType),
                             builder.makeConst(int32_t(teeElemIdx)),
                             MemoryOrder::Unordered,
                             field.type)));
    }
    fullList.push_back(builder.makeLocalGet(arrLocal, arrRefType));
    arrBlock->list.set(fullList);
    arrBlock->type = anf->type;
    for (Index p = plansBefore; p < tentativeState.plans.size(); p++) {
      tentativeState.plans[p].start += prefixCount;
    }
    *ptrIt->second = arrBlock;
    state = std::move(tentativeState);
  }

  // Plans a column of byte differences, if the values are close to their
  // predecessors and there are enough of them to pay for the accumulator.
  bool planDelta(HolePlan& plan, const std::vector<Literal>& values) {
    if (values.size() <= MinDeltaRun) {
      return false;
    }
    int64_t min = 0;
    int64_t max = 0;
    for (Index k = 1; k < values.size(); k++) {
      int64_t delta =
        int64_t(values[k].geti32()) - int64_t(values[k - 1].geti32());
      min = std::min(min, delta);
      max = std::max(max, delta);
    }
    if (max <= 255 && min >= 0) {
      plan.signed_ = false;
    } else if (max <= 127 && min >= -128) {
      plan.signed_ = true;
    } else {
      return false;
    }
    plan.kind = HolePlan::Delta;
    plan.column = ColumnKind::I8;
    plan.base = values[0].geti32();
    plan.data.push_back(0);
    for (Index k = 1; k < values.size(); k++) {
      plan.data.push_back(uint8_t(uint32_t(values[k].geti32()) -
                                  uint32_t(values[k - 1].geti32())));
    }
    return true;
  }

  Index getProgressionIndexBytes(uint32_t base, uint32_t step) {
    Index bytes = LocalGetBytes;
    if (step != 1) {
      bytes += 1 + getConstBytes(Literal(int32_t(step)));
    }
    if (base != 0) {
      bytes += 1 + getConstBytes(Literal(int32_t(base)));
    }
    return bytes;
  }

  // The bytes of the expression that computes a hole in the loop body.
  Index getReadBytes(const HolePlan& plan) {
    switch (plan.kind) {
      case HolePlan::Fixed:
        return 0;
      case HolePlan::Progression: {
        if (plan.base == 0 && plan.step == 1) {
          return LocalGetBytes;
        }
        Index bytes =
          LocalGetBytes + 1 + getConstBytes(Literal(int32_t(plan.base)));
        if (plan.step != 1) {
          bytes += 1 + getConstBytes(Literal(int32_t(plan.step)));
        }
        return bytes;
      }
      case HolePlan::Delta:
      case HolePlan::Column:
      case HolePlan::Table:
      case HolePlan::Mixed: {
        if (plan.kind == HolePlan::Table && plan.column == ColumnKind::Count) {
          return GlobalGetBytes +
                 getProgressionIndexBytes(plan.base, plan.step) + ArrayGetBytes;
        }
        // The offset is not known yet; assume a short one is needed.
        Index bytes = ArrayGetBytes + LocalGetBytes + LocalGetBytes + 1 + 2;
        if (plan.hasDict) {
          bytes += ArrayGetBytes + LocalGetBytes;
        }
        if (plan.kind == HolePlan::Delta) {
          // tee, add, get, and the initialization of the accumulator.
          bytes +=
            2 + 1 + 2 + 2 + 2 + getConstBytes(Literal(int32_t(plan.base)));
        } else if (plan.kind == HolePlan::Table) {
          bytes += ArrayGetBytes + GlobalGetBytes;
        } else if (plan.kind == HolePlan::Mixed) {
          Index numBranches =
            (plan.tableSize > 0 ? 1 : 0) + plan.callees.size();
          bytes += (numBranches - 1) * (3 + 1 + LocalGetBytes + 3);
          if (plan.tableSize == 1) {
            bytes += GlobalGetBytes;
          } else if (plan.tableSize > 1) {
            bytes += ArrayGetBytes + GlobalGetBytes + LocalGetBytes;
          }
          for (auto& callee : plan.callees) {
            bytes += CallBytes + LocalGetBytes;
            int32_t subOffset = int32_t(callee.bound) - int32_t(callee.baseArg);
            if (subOffset != 0) {
              bytes += 1 + getConstBytes(Literal(subOffset));
            }
          }
        }
        return bytes;
      }
    }
    WASM_UNREACHABLE("bad kind");
  }

  // Applying.

  struct ParentFinder
    : public PostWalker<ParentFinder, UnifiedExpressionVisitor<ParentFinder>> {
    std::unordered_map<Expression*, Expression*> parents;
    void visitExpression(Expression* curr) {
      for (auto* child : ChildIterator(curr)) {
        parents[child] = curr;
      }
    }
  };

  struct ScopeNameFinder
    : public PostWalker<ScopeNameFinder,
                        UnifiedExpressionVisitor<ScopeNameFinder>> {
    std::set<Name>& names;
    ScopeNameFinder(std::set<Name>& names) : names(names) {}
    void visitExpression(Expression* curr) {
      BranchUtils::operateOnScopeNameDefs(curr, [&](Name& name) {
        if (name.is()) {
          names.insert(name);
        }
      });
    }
  };

  void applyPlans(FuncState& state) {
    auto* func = state.func;
    Builder builder(*module);
    ScopeNameFinder(state.scopeNames).walk(func->body);

    for (auto kind : state.kindOrder) {
      state.arrayLocals[kind] =
        Builder::addVar(func, Type(getColumnHeapType(kind), NonNullable));
    }

    // Create the table globals or locals (skip 1-element tables).
    std::vector<Expression*> tableSetups;
    for (auto& table : state.tables) {
      if (table.globals.size() <= 1) {
        continue;
      }
      HeapType arrayType(Array(Field(table.elemType, Immutable)));
      std::vector<Expression*> gets;
      gets.reserve(table.globals.size());
      for (Index i = 0; i < table.globals.size(); i++) {
        if (i < table.exprs.size() && table.exprs[i]) {
          gets.push_back(ExpressionManipulator::copy(table.exprs[i], *module));
          continue;
        }
        auto name = table.globals[i];
        if (name.is()) {
          gets.push_back(
            builder.makeGlobalGet(name, module->getGlobal(name)->type));
        } else {
          gets.push_back(
            builder.makeRefNull(table.elemType.getHeapType().getBottom()));
        }
      }
      if (table.hasExprs) {
        table.localIndex =
          Builder::addVar(func, Type(arrayType, NonNullable));
        tableSetups.push_back(builder.makeLocalSet(
          table.localIndex, builder.makeArrayNewFixed(arrayType, gets)));
      } else {
        table.globalName = Names::getValidGlobalName(*module, "roll$table");
        module->addGlobal(
          builder.makeGlobal(table.globalName,
                             Type(arrayType, NonNullable),
                             builder.makeArrayNewFixed(arrayType, gets),
                             Builder::Immutable));
      }
    }

    // Replace the runs, last first so that earlier indexes stay valid.
    Index loopCounter = 0;
    for (auto it = state.plans.rbegin(); it != state.plans.rend(); ++it) {
      auto& plan = *it;
      for (auto& leaf : plan.first.leaves) {
        if (*leaf.ptr == plan.first.expr) {
          leaf.ptr = &plan.first.expr;
        }
      }
      auto i32 = Type::i32;
      Index counter = Builder::addVar(func, i32);
      auto getCounter = [&]() { return builder.makeLocalGet(counter, i32); };
      auto getIndex = [&](Index offset) -> Expression* {
        if (offset == 0) {
          return getCounter();
        }
        return builder.makeBinary(
          AddInt32, getCounter(), builder.makeConst(int32_t(offset)));
      };
      auto readColumn = [&](const HolePlan& hole) -> Expression* {
        auto heapType = getColumnHeapType(hole.column);
        Expression* rawRead = builder.makeArrayGet(
          builder.makeLocalGet(state.arrayLocals[hole.column],
                               Type(heapType, NonNullable)),
          getIndex(hole.offset),
          MemoryOrder::Unordered,
          getColumnValueType(hole.column),
          hole.signed_);
        if (!hole.hasDict) {
          return rawRead;
        }
        auto dictHeapType = getColumnHeapType(hole.dictColumn);
        Expression* dictIdx =
          hole.dictOffset == 0
            ? rawRead
            : builder.makeBinary(
                AddInt32, rawRead, builder.makeConst(int32_t(hole.dictOffset)));
        return builder.makeArrayGet(
          builder.makeLocalGet(state.arrayLocals[hole.dictColumn],
                               Type(dictHeapType, NonNullable)),
          dictIdx,
          MemoryOrder::Unordered,
          getColumnValueType(hole.dictColumn),
          false);
      };
      auto readTable = [&](const HolePlan& hole,
                           Expression* index) -> Expression* {
        auto& table = state.tables[hole.table];
        if (table.globals.size() == 1) {
          if (!table.exprs.empty() && table.exprs[0]) {
            return ExpressionManipulator::copy(table.exprs[0], *module);
          }
          Name name = table.globals[0];
          if (name.is()) {
            return builder.makeGlobalGet(name, module->getGlobal(name)->type);
          }
          return builder.makeRefNull(table.elemType.getHeapType().getBottom());
        }
        HeapType arrayType(Array(Field(table.elemType, Immutable)));
        Expression* tableRef =
          table.hasExprs
            ? static_cast<Expression*>(builder.makeLocalGet(
                table.localIndex, Type(arrayType, NonNullable)))
            : static_cast<Expression*>(builder.makeGlobalGet(
                table.globalName, Type(arrayType, NonNullable)));
        return builder.makeArrayGet(
          tableRef, index, MemoryOrder::Unordered, table.elemType);
      };
      auto makeMixed = [&](const HolePlan& hole,
                           Expression* firstRaw,
                           Index rawLocal) {
        auto getRaw = [&]() -> Expression* {
          return builder.makeLocalGet(rawLocal, i32);
        };
        struct Branch {
          Index upperBound;
          Expression* body;
        };
        std::vector<Branch> branches;
        if (hole.tableSize > 0) {
          Expression* tableIdx =
            (hole.tableSize == 1 || hole.gapOffset == 0)
              ? getRaw()
              : builder.makeBinary(SubInt32,
                                   getRaw(),
                                   builder.makeConst(int32_t(hole.gapOffset)));
          branches.push_back(
            {hole.tableSize + hole.gapOffset, readTable(hole, tableIdx)});
        }
        for (Index c = 0; c < hole.callees.size(); c++) {
          auto& callee = hole.callees[c];
          int32_t subOffset = int32_t(callee.bound) - int32_t(callee.baseArg);
          Expression* arg =
            subOffset == 0
              ? getRaw()
              : builder.makeBinary(
                  SubInt32, getRaw(), builder.makeConst(subOffset));
          Expression* call = builder.makeCall(
            callee.name,
            {arg},
            module->getFunction(callee.name)->getResults());
          Index upperBound =
            c + 1 < hole.callees.size() ? hole.callees[c + 1].bound : 0;
          branches.push_back({upperBound, call});
        }
        Expression* curr = branches.back().body;
        for (Index b = branches.size() - 1; b > 0; b--) {
          Index idx = b - 1;
          Expression* condRaw = idx == 0 ? firstRaw : getRaw();
          curr = builder.makeIf(
            builder.makeBinary(
              LtUInt32,
              condRaw,
              builder.makeConst(int32_t(branches[idx].upperBound))),
            branches[idx].body,
            curr,
            hole.mixedType);
        }
        return curr;
      };
      Index gapTemp = 0;
      Expression* gapRead = nullptr;
      if (plan.switchPtr && plan.hasGaps) {
        gapTemp = Builder::addVar(func, i32);
      }
      // Statements before the loop: the counter and the accumulators.
      std::vector<Expression*> inits;
      if (!plan.switchPtr) {
        inits.push_back(
          builder.makeLocalSet(counter, builder.makeConst(int32_t(0))));
      }
      for (Index h = 0; h < plan.holes.size(); h++) {
        auto& hole = plan.holes[h];
        auto& leaf = plan.first.leaves[h];
        if (hole.kind == HolePlan::Fixed) {
          continue;
        }
        if (hole.isGapHole) {
          gapRead = readColumn(hole);
          auto getRaw = [&]() { return builder.makeLocalGet(gapTemp, i32); };
          auto getAdj = [&]() -> Expression* {
            if (hole.gapOffset == 0) {
              return getRaw();
            }
            return builder.makeBinary(
              SubInt32, getRaw(), builder.makeConst(int32_t(hole.gapOffset)));
          };
          Expression* value = nullptr;
          if (hole.kind == HolePlan::Column) {
            value = getAdj();
            if (hole.callee.is()) {
              value = builder.makeCall(
                hole.callee,
                {value},
                module->getFunction(hole.callee)->getResults());
            }
          } else if (hole.kind == HolePlan::Table) {
            value = readTable(hole, getAdj());
          } else if (hole.kind == HolePlan::Mixed) {
            value = makeMixed(hole, getRaw(), gapTemp);
          } else {
            WASM_UNREACHABLE("unexpected gap hole kind");
          }
          *leaf.ptr = value;
          continue;
        }
        Expression* value = nullptr;
        switch (hole.kind) {
          case HolePlan::Fixed:
            WASM_UNREACHABLE("handled above");
          case HolePlan::Progression: {
            value = getCounter();
            if (hole.step != 1) {
              value = builder.makeBinary(
                MulInt32, value, builder.makeConst(int32_t(hole.step)));
            }
            if (hole.base != 0) {
              value = builder.makeBinary(
                AddInt32, value, builder.makeConst(int32_t(hole.base)));
            }
            break;
          }
          case HolePlan::Column:
            value = readColumn(hole);
            break;
          case HolePlan::Delta: {
            Index acc = Builder::addVar(func, i32);
            inits.push_back(
              builder.makeLocalSet(acc, builder.makeConst(int32_t(hole.base))));
            value = builder.makeLocalTee(
              acc,
              builder.makeBinary(
                AddInt32, builder.makeLocalGet(acc, i32), readColumn(hole)),
              i32);
            break;
          }
          case HolePlan::Table: {
            Expression* idxExpr = nullptr;
            if (hole.column == ColumnKind::Count) {
              idxExpr = getCounter();
              if (hole.step != 1) {
                idxExpr = builder.makeBinary(
                  MulInt32, idxExpr, builder.makeConst(int32_t(hole.step)));
              }
              if (hole.base != 0) {
                idxExpr = builder.makeBinary(
                  AddInt32, idxExpr, builder.makeConst(int32_t(hole.base)));
              }
            } else {
              idxExpr = readColumn(hole);
            }
            value = readTable(hole, idxExpr);
            break;
          }
          case HolePlan::Mixed: {
            Index temp = Builder::addVar(func, i32);
            value = makeMixed(
              hole, builder.makeLocalTee(temp, readColumn(hole), i32), temp);
            *leaf.ptr = value;
            continue;
          }
        }
        if (hole.callee.is()) {
          value =
            builder.makeCall(hole.callee,
                             {value},
                             module->getFunction(hole.callee)->getResults());
        }
        *leaf.ptr = value;
      }
      if (plan.switchPtr) {
        Expression* armBody = plan.first.expr;
        if (plan.switchKind == 1) {
          armBody = builder.makeReturn(armBody);
        } else if (plan.switchKind == 2) {
          armBody = builder.makeBreak(plan.switchExitLabel, armBody);
        } else if (plan.switchExitLabel.is()) {
          armBody = builder.makeSequence(
            armBody, builder.makeBreak(plan.switchExitLabel));
        }
        if (plan.hasGaps) {
          armBody = builder.makeIf(
            builder.makeLocalTee(gapTemp, gapRead, i32), armBody);
        }
        plan.replacement = builder.makeIf(
          builder.makeBinary(
            LtUInt32,
            builder.makeLocalTee(counter, plan.switchCond, i32),
            builder.makeConst(int32_t(plan.size))),
          armBody);
        *plan.switchPtr = plan.replacement;
        continue;
      }
      Name loopName = Names::getValidName(
        "roll",
        [&](Name name) { return !state.scopeNames.count(name); },
        loopCounter++);
      state.scopeNames.insert(loopName);
      auto* next = builder.makeBreak(
        loopName,
        nullptr,
        builder.makeBinary(LtUInt32,
                           builder.makeLocalTee(
                             counter,
                             builder.makeBinary(AddInt32,
                                                getCounter(),
                                                builder.makeConst(int32_t(1))),
                             i32),
                           builder.makeConst(int32_t(plan.size))));
      auto* loop =
        builder.makeLoop(loopName, builder.makeBlock({plan.first.expr, next}));
      inits.push_back(loop);
      plan.replacement = builder.makeBlock(inits);
      // Splice.
      auto& list = plan.block->list;
      std::vector<Expression*> newList(list.begin(), list.begin() + plan.start);
      newList.push_back(plan.replacement);
      newList.insert(
        newList.end(), list.begin() + plan.start + plan.size, list.end());
      list.set(newList);
    }

    // Lay out the arrays in the segment and load them in the innermost block
    // that contains all the runs.
    std::vector<Expression*> setups = std::move(tableSetups);
    for (auto& [kind, data] : state.arrayData) {
      Index offset = segmentData.size();
      segmentData.insert(segmentData.end(), data.begin(), data.end());
      setups.push_back(builder.makeLocalSet(
        state.arrayLocals[kind],
        builder.makeArrayNewData(
          getColumnHeapType(kind),
          segmentName,
          builder.makeConst(int32_t(offset)),
          builder.makeConst(int32_t(state.arrayCounts[kind])))));
    }
    if (!setups.empty()) {
      insertSetups(state, setups);
    }
    ReFinalize().walkFunctionInModule(func, module);
  }

  void insertSetups(FuncState& state, std::vector<Expression*>& setups) {
    auto* func = state.func;
    ParentFinder parents;
    parents.walk(func->body);
    // For each run, the chain of enclosing blocks from the root, each with
    // the child on the path to the run.
    using Chain = std::vector<std::pair<Block*, Expression*>>;
    std::vector<Chain> chains;
    for (auto& plan : state.plans) {
      Chain chain;
      Expression* curr = plan.replacement;
      while (true) {
        auto iter = parents.parents.find(curr);
        if (iter == parents.parents.end()) {
          break;
        }
        auto* parent = iter->second;
        if (auto* block = parent->dynCast<Block>()) {
          chain.push_back({block, curr});
        }
        curr = parent;
      }
      std::reverse(chain.begin(), chain.end());
      chains.push_back(std::move(chain));
    }
    // The longest common prefix of blocks.
    Index common = 0;
    while (true) {
      bool same = true;
      for (auto& chain : chains) {
        if (common >= chain.size() ||
            chain[common].first != chains[0][common].first) {
          same = false;
          break;
        }
      }
      if (!same) {
        break;
      }
      common++;
    }
    Builder builder(*module);
    if (common == 0) {
      // No common block: wrap the body.
      setups.push_back(func->body);
      func->body = builder.makeBlock(setups);
      return;
    }
    auto* target = chains[0][common - 1].first;
    auto& list = target->list;
    Index insertAt = list.size();
    for (auto& chain : chains) {
      auto* child = chain[common - 1].second;
      for (Index i = 0; i < list.size(); i++) {
        if (list[i] == child) {
          insertAt = std::min(insertAt, i);
          break;
        }
      }
    }
    std::vector<Expression*> newList(list.begin(), list.begin() + insertAt);
    newList.insert(newList.end(), setups.begin(), setups.end());
    newList.insert(newList.end(), list.begin() + insertAt, list.end());
    list.set(newList);
  }
};

} // anonymous namespace

Pass* createRollRepeatedCallsPass() { return new RollRepeatedCalls(); }

} // namespace wasm
