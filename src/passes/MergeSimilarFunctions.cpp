/*
 * Copyright 2021 WebAssembly Community Group participants
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
// Merge similar functions that only differ in constant values (like immediate
// operands of const, immutable global.get, ref.func, and call instructions) by
// parameterization.
//

#include "ir/find_all.h"
#include "ir/hashed.h"
#include "ir/iteration.h"
#include "ir/manipulation.h"
#include "ir/module-utils.h"
#include "ir/names.h"
#include "ir/public-type-validator.h"
#include "ir/utils.h"
#include "opt-utils.h"
#include "pass.h"
#include "support/hash.h"
#include "support/small_vector.h"
#include "support/utilities.h"
#include "wasm-builder.h"
#include "wasm-limits.h"
#include "wasm.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace wasm {

namespace {

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

Index getShallowBytes(Expression* curr) {
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

Index getTreeBytes(Expression* root) {
  if (!root) {
    return 0;
  }
  Index total = 0;
  SmallVector<Expression*, 32> worklist;
  worklist.push_back(root);
  while (!worklist.empty()) {
    Expression* curr = worklist.back();
    worklist.pop_back();
    total += getShallowBytes(curr);
    for (auto* child : ChildIterator(curr)) {
      worklist.push_back(child);
    }
  }
  return total;
}

Index getExtraLocalDeclBytes(Function* f) {
  if (f->vars.empty()) {
    return 0;
  }
  std::unordered_set<Type> distinctTypes;
  for (auto vt : f->vars) {
    distinctTypes.insert(vt);
  }
  Index bytes = 0;
  for (auto vt : distinctTypes) {
    bytes += vt.isRef() ? 3 : 2;
  }
  return bytes;
}

bool isPureConstLeaf(Expression* expr, Module* module) {
  if (expr->is<Const>() || expr->is<RefFunc>() || expr->is<RefNull>()) {
    return true;
  }
  if (auto* gg = expr->dynCast<GlobalGet>()) {
    return !module->getGlobal(gg->name)->mutable_;
  }
  return false;
}

struct LeafValue {
  enum Kind : uint8_t { LiteralVal, GlobalVal, FuncVal } kind = LiteralVal;
  Type type = Type::none;
  Literal lit;
  Name name;

  bool operator==(const LeafValue& other) const {
    if (kind != other.kind || type != other.type) {
      return false;
    }
    if (kind == LiteralVal) {
      return lit == other.lit;
    }
    return name == other.name;
  }
  bool operator!=(const LeafValue& other) const { return !(*this == other); }
};

struct ParamInfo {
  std::vector<LeafValue> values;
  std::vector<Expression**> uses;

  ParamInfo(std::vector<LeafValue> values, std::vector<Expression**> uses)
    : values(std::move(values)), uses(std::move(uses)) {}

  Type getValueType(Module* module) const { return values[0].type; }

  Index getArgBytes(size_t index) const {
    const auto& v = values[index];
    if (v.kind == LeafValue::LiteralVal) {
      switch (v.type.getBasic()) {
        case Type::i32:
          return 1 + getSLEBSize(v.lit.geti32());
        case Type::i64:
          return 1 + getSLEBSize(v.lit.geti64());
        case Type::f32:
          return 5;
        case Type::f64:
          return 9;
        case Type::v128:
          return 18;
        default:
          return 3;
      }
    }
    return 3;
  }

  Expression*
  lowerToExpression(Builder& builder, Module* module, size_t index) const {
    const auto& v = values[index];
    switch (v.kind) {
      case LeafValue::LiteralVal:
        return builder.makeConst(v.lit);
      case LeafValue::GlobalVal:
        return builder.makeGlobalGet(v.name, v.type);
      case LeafValue::FuncVal:
        return builder.makeRefFunc(v.name);
    }
    WASM_UNREACHABLE("unexpected const value type");
  }
};

struct ConstThunkInfo {
  Name baseTarget;
  Index numOrigParams = 0;
  Index totalBaseArgs = 0;
  // For each operand j in 0..totalBaseArgs-1 of baseTarget:
  // If isParam[j] is true, paramIdx[j] is 0..numOrigParams-1 (in strict order).
  // Otherwise constExpr[j] is the constant leaf expression from the thunk.
  std::vector<bool> isParam;
  std::vector<Expression*> constExpr;
  Index insertedConstBytes = 0;
};

struct UnwrappedCallRecord {
  Call* callExpr = nullptr;
  Name origThunkTarget;
  std::vector<Index> callerArgIndices;
  std::vector<Expression*> insertedConstExprs;
  Index insertedConstBytes = 0;
};

struct FuncNormalizationState {
  bool localsPermuted = false;
  std::vector<Type> origVars;
  std::vector<Index> newToOldLocal;
  std::vector<Expression*> swappedCommutative;
  std::vector<UnwrappedCallRecord> unwrappedCalls;
  Index totalUnwrappedConstBytes = 0;
};

struct ModuleCallStats {
  std::unordered_map<Name, Index> callCounts;
  std::unordered_map<Name, Index> refFuncCounts;
  std::unordered_set<Name> exportedFuncs;
  std::unordered_set<Signature> existingSignatures;
};

} // anonymous namespace

// Describes the set of functions which are considered as "equivalent" (i.e.
// only differing by some constants).
struct EquivalentClass {
  Function* primaryFunction;
  std::vector<Function*> functions;

  EquivalentClass(Function* primaryFunction, std::vector<Function*> functions)
    : primaryFunction(primaryFunction), functions(std::move(functions)) {}

  bool isEligibleToMerge() const { return this->functions.size() >= 2; }

  void merge(
    Module* module,
    const std::vector<ParamInfo>& params,
    const std::unordered_map<Function*, FuncNormalizationState>* normStates,
    std::vector< std::pair<Function*, Function*>>* mergedThunkPairs);

  int evaluateMergeSavings(
    Module* module,
    const std::vector<ParamInfo>& params,
    const ModuleCallStats& stats,
    const std::unordered_map<Function*, FuncNormalizationState>* normStates,
    bool shrinkMode,
    bool earlyMode = false) const;

  bool hasMergeBenefit(
    Module* module,
    const std::vector<ParamInfo>& params,
    const ModuleCallStats& stats,
    const std::unordered_map<Function*, FuncNormalizationState>* normStates,
    bool shrinkMode,
    bool earlyMode = false) const {
    return evaluateMergeSavings(
             module, params, stats, normStates, shrinkMode, earlyMode) > 0;
  }

  Function* createShared(
    Module* module,
    const std::vector<ParamInfo>& params,
    const std::unordered_map<Function*, FuncNormalizationState>* normStates);

  Function* replaceWithThunk(Builder& builder,
                             Function* target,
                             Function* shared,
                             const std::vector<ParamInfo>& params,
                             const std::vector<Expression*>& extraArgs,
                             bool isReturn);

  bool deriveParams(Module* module,
                    std::vector<ParamInfo>& params,
                    bool isIndirectionEnabled,
                    bool earlyMode = false) const;
};

struct MergeSimilarFunctions : public Pass {
  bool invalidatesDWARF() override { return true; }

  std::unordered_map<Name, ConstThunkInfo> constThunks;
  std::unordered_set<Name> activeConstThunks;
  std::unordered_map<Function*, FuncNormalizationState> normStates;
  ModuleCallStats stats;
  bool earlyMode = false;

  void run(Module* module) override {
    earlyMode =
      getArgumentOrDefault("merge-similar-functions", "") == "early";
    int numRounds = earlyMode ? 2 : 3;
    for (int round = 0; round < numRounds; ++round) {
      constThunks.clear();
      activeConstThunks.clear();
      normStates.clear();
      stats = ModuleCallStats{};

      collectModuleStats(module);
      identifyConstThunks(module);
      if (round == 2) {
        activeConstThunks.clear();
      }
      normalizeModuleFunctions(module);

      std::vector<EquivalentClass> rawClasses;
      collectEquivalentClasses(rawClasses, module);
      std::sort(rawClasses.begin(),
                rawClasses.end(),
                [](const auto& left, const auto& right) {
                  return left.primaryFunction->name <
                         right.primaryFunction->name;
                });

      bool callIndirection = isCallIndirectionEnabled(module);
      bool shrinkMode = getPassOptions().shrinkLevel > 0;

      std::vector<std::pair<Function*, Function*>> mergedThunkPairs;
      std::unordered_set<Function*> mergedFunctions;

      // Track functions that became identical (0 differing params) after
      // local/commutative normalization so we keep their canonical form for
      // DuplicateFunctionElimination.
      std::unordered_set<Function*> canonicalizedIdenticalFuncs;

      for (auto& rawClass : rawClasses) {
        if (!rawClass.isEligibleToMerge()) {
          continue;
        }
        std::vector<EquivalentClass> subClasses;
        splitIntoSubClusters(rawClass,
                             module,
                             callIndirection,
                             shrinkMode,
                             subClasses,
                             canonicalizedIdenticalFuncs);

        for (auto& clazz : subClasses) {
          if (!clazz.isEligibleToMerge()) {
            continue;
          }
          std::vector<ParamInfo> params;
          if (!clazz.deriveParams(module, params, callIndirection, earlyMode)) {
            continue;
          }
          if (params.empty()) {
            for (auto* f : clazz.functions) {
              canonicalizedIdenticalFuncs.insert(f);
            }
            continue;
          }
          if (!clazz.hasMergeBenefit(
                module, params, stats, &normStates, shrinkMode, earlyMode)) {
            continue;
          }
          for (auto* f : clazz.functions) {
            mergedFunctions.insert(f);
          }
          clazz.merge(module, params, &normStates, &mergedThunkPairs);
        }
      }

      // Restore unmerged functions to their exact pre-normalization state
      // before forwarding any direct callers. For functions that became
      // identical to another function via local/commutative canonicalization,
      // only restore unwrapped calls so DFE can merge them.
      for (auto& [func, state] : normStates) {
        if (!mergedFunctions.count(func)) {
          restoreFunction(
            func, state, canonicalizedIdenticalFuncs.count(func) > 0);
        }
      }

      if (mergedThunkPairs.empty()) {
        if (earlyMode || activeConstThunks.empty()) {
          break;
        }
        if (round == 0) {
          round = 1;
        }
        continue;
      }

      // Forward non-escaped 1..2 caller merged thunks directly to their shared
      // helper and remove the eliminated thunk functions.
      forwardDirectCallers(module, mergedThunkPairs);
    }
  }

  bool isCallIndirectionEnabled(Module* module) const {
    return module->features.hasReferenceTypes() && module->features.hasGC();
  }

  void collectModuleStats(Module* module);
  void identifyConstThunks(Module* module);
  void normalizeModuleFunctions(Module* module);
  void restoreFunction(Function* func,
                       const FuncNormalizationState& state,
                       bool keepLocalAndCommutativeCanon = false);
  void splitIntoSubClusters(
    const EquivalentClass& rawClass,
    Module* module,
    bool callIndirection,
    bool shrinkMode,
    std::vector<EquivalentClass>& outClusters,
    std::unordered_set<Function*>& canonicalizedIdenticalFuncs);
  void forwardDirectCallers(
    Module* module,
    const std::vector<std::pair<Function*, Function*>>& mergedThunkPairs);

  bool areInEquvalentClass(Function* lhs, Function* rhs, Module* module);
  void collectEquivalentClasses(std::vector<EquivalentClass>& classes,
                                Module* module);
};

void MergeSimilarFunctions::collectModuleStats(Module* module) {
  for (auto& exp : module->exports) {
    if (exp->kind == ExternalKind::Function) {
      stats.exportedFuncs.insert(*exp->getInternalName());
    }
  }
  if (module->start.is()) {
    stats.exportedFuncs.insert(module->start);
  }
  for (auto& seg : module->elementSegments) {
    for (auto* expr : seg->data) {
      SmallVector<Expression*, 8> worklist;
      worklist.push_back(expr);
      while (!worklist.empty()) {
        Expression* curr = worklist.back();
        worklist.pop_back();
        if (auto* rf = curr->dynCast<RefFunc>()) {
          stats.refFuncCounts[rf->func]++;
        }
        for (auto* child : ChildIterator(curr)) {
          worklist.push_back(child);
        }
      }
    }
  }
  for (auto& glob : module->globals) {
    if (!glob->imported() && glob->init) {
      SmallVector<Expression*, 8> worklist;
      worklist.push_back(glob->init);
      while (!worklist.empty()) {
        Expression* curr = worklist.back();
        worklist.pop_back();
        if (auto* rf = curr->dynCast<RefFunc>()) {
          stats.refFuncCounts[rf->func]++;
        }
        for (auto* child : ChildIterator(curr)) {
          worklist.push_back(child);
        }
      }
    }
  }
  ModuleUtils::iterDefinedFunctions(*module, [&](Function* func) {
    if (func->type.isSignature()) {
      stats.existingSignatures.insert(func->getSig());
    }
    if (!func->body) {
      return;
    }
    SmallVector<Expression*, 32> worklist;
    worklist.push_back(func->body);
    while (!worklist.empty()) {
      Expression* curr = worklist.back();
      worklist.pop_back();
      if (auto* call = curr->dynCast<Call>()) {
        stats.callCounts[call->target]++;
      } else if (auto* rf = curr->dynCast<RefFunc>()) {
        stats.refFuncCounts[rf->func]++;
      }
      for (auto* child : ChildIterator(curr)) {
        worklist.push_back(child);
      }
    }
  });
}

void MergeSimilarFunctions::identifyConstThunks(Module* module) {
  ModuleUtils::iterDefinedFunctions(*module, [&](Function* func) {
    if (func->getNumVars() != 0 || !func->body) {
      return;
    }
    auto* call = func->body->dynCast<Call>();
    if (!call || call->target == func->name ||
        call->type != func->getResults()) {
      return;
    }
    Index numParams = func->getNumParams();
    Index numArgs = call->operands.size();
    if (numArgs <= numParams || numArgs - numParams > 3) {
      return;
    }
    ConstThunkInfo info;
    info.baseTarget = call->target;
    info.numOrigParams = numParams;
    info.totalBaseArgs = numArgs;
    info.isParam.resize(numArgs, false);
    info.constExpr.resize(numArgs, nullptr);

    Index expectedParam = 0;
    bool valid = true;
    for (Index j = 0; j < numArgs; ++j) {
      Expression* op = call->operands[j];
      if (auto* lg = op->dynCast<LocalGet>()) {
        if (expectedParam >= numParams || lg->index != expectedParam) {
          valid = false;
          break;
        }
        bool typeOk =
          earlyMode
            ? (lg->type == func->getLocalType(expectedParam))
            : Type::isSubType(lg->type, func->getLocalType(expectedParam));
        if (!typeOk) {
          valid = false;
          break;
        }
        info.isParam[j] = true;
        expectedParam++;
      } else if (isPureConstLeaf(op, module)) {
        if (auto* rf = op->dynCast<RefFunc>()) {
          if (!isCallIndirectionEnabled(module)) {
            valid = false;
            break;
          }
          if (getPassOptions().worldMode == WorldMode::Open &&
              !PublicTypeValidator(module->features)
                 .isValidPublicType(
                   module->getFunction(rf->func)->type.getHeapType())) {
            valid = false;
            break;
          }
        }
        if (auto* gg = op->dynCast<GlobalGet>()) {
          if (getPassOptions().worldMode == WorldMode::Open &&
              !PublicTypeValidator(module->features)
                 .isValidPublicType(gg->type)) {
            valid = false;
            break;
          }
        }
        info.isParam[j] = false;
        info.constExpr[j] = op;
        info.insertedConstBytes += getShallowBytes(op);
      } else {
        valid = false;
        break;
      }
    }
    if (valid && expectedParam == numParams) {
      constThunks[func->name] = std::move(info);
    }
  });

  // Resolve 2-level constant-forwarding thunks transitively if total added
  // constants <= 3.
  for (auto& [name, info] : constThunks) {
    auto it = constThunks.find(info.baseTarget);
    if (it == constThunks.end() || it->first == name) {
      continue;
    }
    const auto& parent = it->second;
    if (parent.numOrigParams != info.totalBaseArgs) {
      continue;
    }
    Index combinedAdded = parent.totalBaseArgs - info.numOrigParams;
    if (combinedAdded > 3) {
      continue;
    }
    ConstThunkInfo merged;
    merged.baseTarget = parent.baseTarget;
    merged.numOrigParams = info.numOrigParams;
    merged.totalBaseArgs = parent.totalBaseArgs;
    merged.isParam.resize(parent.totalBaseArgs, false);
    merged.constExpr.resize(parent.totalBaseArgs, nullptr);
    merged.insertedConstBytes =
      info.insertedConstBytes + parent.insertedConstBytes;
    Index childIdx = 0;
    for (Index j = 0; j < parent.totalBaseArgs; ++j) {
      if (parent.isParam[j]) {
        assert(childIdx < info.totalBaseArgs);
        merged.isParam[j] = info.isParam[childIdx];
        merged.constExpr[j] = info.constExpr[childIdx];
        childIdx++;
      } else {
        merged.isParam[j] = false;
        merged.constExpr[j] = parent.constExpr[j];
      }
    }
    info = std::move(merged);
  }

  // Count how many distinct non-thunk functions call each baseTarget (either
  // directly or through a constant-forwarding thunk).
  std::unordered_map<Name, Index> baseCallerFuncCount;
  ModuleUtils::iterDefinedFunctions(*module, [&](Function* func) {
    if (constThunks.count(func->name) || !func->body) {
      return;
    }
    std::unordered_set<Name> seenBases;
    SmallVector<Expression*, 32> worklist;
    worklist.push_back(func->body);
    while (!worklist.empty()) {
      Expression* curr = worklist.back();
      worklist.pop_back();
      if (auto* call = curr->dynCast<Call>()) {
        auto it = constThunks.find(call->target);
        Name base = (it != constThunks.end()) ? it->second.baseTarget
                                              : call->target;
        seenBases.insert(base);
      }
      for (auto* child : ChildIterator(curr)) {
        worklist.push_back(child);
      }
    }
    for (Name base : seenBases) {
      baseCallerFuncCount[base]++;
    }
  });

  for (const auto& [thunkName, info] : constThunks) {
    bool hasRefFuncConst = false;
    if (!earlyMode) {
      for (Expression* ce : info.constExpr) {
        if (ce && ce->is<RefFunc>()) {
          hasRefFuncConst = true;
          break;
        }
      }
    }
    if (!hasRefFuncConst && baseCallerFuncCount[info.baseTarget] >= 2) {
      activeConstThunks.insert(thunkName);
    }
  }
}

void MergeSimilarFunctions::normalizeModuleFunctions(Module* module) {
  ModuleUtils::iterDefinedFunctions(*module, [&](Function* func) {
    if (!func->body || constThunks.count(func->name)) {
      return;
    }
    FuncNormalizationState state;
    Index numParams = func->getNumParams();
    Index numVars = func->getNumVars();
    std::vector<Index> oldToNew(numVars, Index(-1));
    Index nextVar = 0;

    // Preorder traversal using a stack (pushing children in reverse order so
    // they are visited in forward ChildIterator order).
    SmallVector<Expression*, 32> stack;
    stack.push_back(func->body);
    while (!stack.empty()) {
      Expression* curr = stack.back();
      stack.pop_back();

      // 1. Unwrap calls to active constant-forwarding thunks.
      if (auto* call = curr->dynCast<Call>()) {
        auto it = constThunks.find(call->target);
        if (it != constThunks.end() && activeConstThunks.count(call->target) &&
            call->operands.size() == it->second.numOrigParams &&
            call->type != Type::unreachable) {
          bool anyUnreachableArg = false;
          for (auto* op : call->operands) {
            if (op->type == Type::unreachable) {
              anyUnreachableArg = true;
              break;
            }
          }
          if (!anyUnreachableArg) {
            const auto& info = it->second;
            UnwrappedCallRecord rec;
            rec.callExpr = call;
            rec.origThunkTarget = call->target;
            rec.insertedConstBytes = info.insertedConstBytes;

            std::vector<Expression*> origArgs(call->operands.begin(),
                                              call->operands.end());
            call->operands.resize(info.totalBaseArgs);
            Index origIdx = 0;
            for (Index j = 0; j < info.totalBaseArgs; ++j) {
              if (info.isParam[j]) {
                call->operands[j] = origArgs[origIdx++];
                rec.callerArgIndices.push_back(j);
              } else {
                Expression* clonedConst =
                  ExpressionManipulator::copy(info.constExpr[j], *module);
                call->operands[j] = clonedConst;
                rec.insertedConstExprs.push_back(clonedConst);
              }
            }
            call->target = info.baseTarget;
            state.totalUnwrappedConstBytes += rec.insertedConstBytes;
            state.unwrappedCalls.push_back(std::move(rec));
          }
        }
      }

      // 2. Canonicalize commutative i32.eq / i32.ne / ref.eq when left is a
      // pure constant leaf and right is not.
      if (auto* bin = curr->dynCast<Binary>()) {
        if ((bin->op == EqInt32 || bin->op == NeInt32) &&
            isPureConstLeaf(bin->left, module) &&
            !isPureConstLeaf(bin->right, module)) {
          std::swap(bin->left, bin->right);
          state.swappedCommutative.push_back(curr);
        }
      } else if (auto* re = curr->dynCast<RefEq>()) {
        if (isPureConstLeaf(re->left, module) &&
            !isPureConstLeaf(re->right, module)) {
          std::swap(re->left, re->right);
          state.swappedCommutative.push_back(curr);
        }
      }

      // 3. Record first-appearance order of non-parameter locals.
      if (numVars >= 2) {
        if (auto* lg = curr->dynCast<LocalGet>()) {
          if (lg->index >= numParams) {
            Index v = lg->index - numParams;
            if (oldToNew[v] == Index(-1)) {
              oldToNew[v] = numParams + nextVar++;
            }
          }
        } else if (auto* ls = curr->dynCast<LocalSet>()) {
          if (ls->index >= numParams) {
            Index v = ls->index - numParams;
            if (oldToNew[v] == Index(-1)) {
              oldToNew[v] = numParams + nextVar++;
            }
          }
        }
      }

      ChildIterator childIt(curr);
      for (auto* childPtr : childIt.children) {
        stack.push_back(*childPtr);
      }
    }

    if (numVars >= 2) {
      for (Index v = 0; v < numVars; ++v) {
        if (oldToNew[v] == Index(-1)) {
          oldToNew[v] = numParams + nextVar++;
        }
      }
      bool anyChanged = false;
      for (Index v = 0; v < numVars; ++v) {
        if (oldToNew[v] != numParams + v) {
          anyChanged = true;
          break;
        }
      }
      if (anyChanged) {
        state.localsPermuted = true;
        state.origVars = func->vars;
        state.newToOldLocal.resize(numParams + numVars);
        for (Index p = 0; p < numParams; ++p) {
          state.newToOldLocal[p] = p;
        }
        for (Index v = 0; v < numVars; ++v) {
          Index newIdx = oldToNew[v];
          state.newToOldLocal[newIdx] = numParams + v;
          func->vars[newIdx - numParams] = state.origVars[v];
        }
        stack.push_back(func->body);
        while (!stack.empty()) {
          Expression* curr = stack.back();
          stack.pop_back();
          if (auto* lg = curr->dynCast<LocalGet>()) {
            if (lg->index >= numParams) {
              lg->index = oldToNew[lg->index - numParams];
            }
          } else if (auto* ls = curr->dynCast<LocalSet>()) {
            if (ls->index >= numParams) {
              ls->index = oldToNew[ls->index - numParams];
            }
          }
          for (auto* child : ChildIterator(curr)) {
            stack.push_back(child);
          }
        }
      }
    }

    if (state.localsPermuted || !state.swappedCommutative.empty() ||
        !state.unwrappedCalls.empty()) {
      normStates[func] = std::move(state);
    }
  });
}

void MergeSimilarFunctions::restoreFunction(
  Function* func,
  const FuncNormalizationState& state,
  bool keepLocalAndCommutativeCanon) {
  if (!func->body) {
    return;
  }
  if (!keepLocalAndCommutativeCanon) {
    if (state.localsPermuted) {
      Index numParams = func->getNumParams();
      func->vars = state.origVars;
      SmallVector<Expression*, 32> stack;
      stack.push_back(func->body);
      while (!stack.empty()) {
        Expression* curr = stack.back();
        stack.pop_back();
        if (auto* lg = curr->dynCast<LocalGet>()) {
          if (lg->index >= numParams) {
            lg->index = state.newToOldLocal[lg->index];
          }
        } else if (auto* ls = curr->dynCast<LocalSet>()) {
          if (ls->index >= numParams) {
            ls->index = state.newToOldLocal[ls->index];
          }
        }
        for (auto* child : ChildIterator(curr)) {
          stack.push_back(child);
        }
      }
    }
    for (Expression* curr : state.swappedCommutative) {
      if (auto* bin = curr->dynCast<Binary>()) {
        std::swap(bin->left, bin->right);
      } else if (auto* re = curr->dynCast<RefEq>()) {
        std::swap(re->left, re->right);
      }
    }
  }
  for (const auto& rec : state.unwrappedCalls) {
    SmallVector<Expression*, 8> origArgs;
    for (Index idx : rec.callerArgIndices) {
      origArgs.push_back(rec.callExpr->operands[idx]);
    }
    rec.callExpr->operands.clear();
    for (auto* arg : origArgs) {
      rec.callExpr->operands.push_back(arg);
    }
    rec.callExpr->target = rec.origThunkTarget;
  }
}

// Determine if two functions are equivalent ignoring constants.
bool MergeSimilarFunctions::areInEquvalentClass(Function* lhs,
                                                Function* rhs,
                                                Module* module) {
  if (lhs->imported() || rhs->imported()) {
    return false;
  }
  if (lhs->type != rhs->type) {
    return false;
  }
  if (lhs->vars != rhs->vars) {
    return false;
  }

  ExpressionAnalyzer::ExprComparer comparer = [&](Expression* lhsExpr,
                                                  Expression* rhsExpr) {
    if (lhsExpr->_id != rhsExpr->_id) {
      return false;
    }
    if (lhsExpr->type != rhsExpr->type) {
      return false;
    }
    if (lhsExpr->is<Call>()) {
      auto* lhsCast = lhsExpr->cast<Call>();
      auto* rhsCast = rhsExpr->cast<Call>();
      if (lhsCast->isReturn != rhsCast->isReturn) {
        return false;
      }
      if (lhsCast->operands.size() != rhsCast->operands.size()) {
        return false;
      }
      if (lhsCast->target != rhsCast->target) {
        if (!this->isCallIndirectionEnabled(module)) {
          return false;
        }
        auto* lhsCallee = module->getFunction(lhsCast->target);
        auto* rhsCallee = module->getFunction(rhsCast->target);
        if (lhsCallee->type != rhsCallee->type) {
          return false;
        }
        if (getPassOptions().worldMode == WorldMode::Open &&
            !PublicTypeValidator(module->features)
               .isValidPublicType(lhsCallee->type.getHeapType())) {
          return false;
        }
      }
      for (Index i = 0; i < lhsCast->operands.size(); i++) {
        if (!ExpressionAnalyzer::flexibleEqual(
              lhsCast->operands[i], rhsCast->operands[i], comparer)) {
          return false;
        }
      }
      return true;
    }

    if (lhsExpr->is<Const>()) {
      auto* lhsCast = lhsExpr->cast<Const>();
      auto* rhsCast = rhsExpr->cast<Const>();
      return lhsCast->value.type == rhsCast->value.type;
    }

    if (lhsExpr->is<GlobalGet>()) {
      auto* lhsCast = lhsExpr->cast<GlobalGet>();
      auto* rhsCast = rhsExpr->cast<GlobalGet>();
      if (lhsCast->name == rhsCast->name) {
        return true;
      }
      if (module->getGlobal(lhsCast->name)->mutable_ ||
          module->getGlobal(rhsCast->name)->mutable_) {
        return false;
      }
      if (getPassOptions().worldMode == WorldMode::Open &&
          !PublicTypeValidator(module->features)
             .isValidPublicType(lhsCast->type)) {
        return false;
      }
      return true;
    }

    if (lhsExpr->is<RefFunc>()) {
      auto* lhsCast = lhsExpr->cast<RefFunc>();
      auto* rhsCast = rhsExpr->cast<RefFunc>();
      if (lhsCast->func == rhsCast->func) {
        return true;
      }
      if (!this->isCallIndirectionEnabled(module)) {
        return false;
      }
      auto* lhsCallee = module->getFunction(lhsCast->func);
      auto* rhsCallee = module->getFunction(rhsCast->func);
      if (lhsCallee->type != rhsCallee->type) {
        return false;
      }
      if (lhsCast->type.getHeapType() != lhsCallee->type.getHeapType() ||
          rhsCast->type.getHeapType() != rhsCallee->type.getHeapType() ||
          ((lhsCast->type.isExact() || rhsCast->type.isExact()) &&
           module->features.hasCustomDescriptors())) {
        return false;
      }
      if (getPassOptions().worldMode == WorldMode::Open &&
          !PublicTypeValidator(module->features)
             .isValidPublicType(lhsCallee->type.getHeapType())) {
        return false;
      }
      return true;
    }

    return false;
  };
  return ExpressionAnalyzer::flexibleEqual(lhs->body, rhs->body, comparer);
}

// Collect all equivalent classes to be merged.
void MergeSimilarFunctions::collectEquivalentClasses(
  std::vector<EquivalentClass>& classes, Module* module) {
  auto hashes = FunctionHasher::createMap(module);
  PassRunner runner(module);
  bool callIndirection = isCallIndirectionEnabled(module);

  std::function<bool(Expression*, size_t&)> ignoringConsts =
    [&](Expression* expr, size_t& digest) {
      if (expr->is<Const>()) {
        return true;
      }
      if (auto* gg = expr->dynCast<GlobalGet>()) {
        if (!module->getGlobal(gg->name)->mutable_) {
          return true;
        }
        return false;
      }
      if (auto* rf = expr->dynCast<RefFunc>()) {
        if (callIndirection) {
          rehash(digest, module->getFunction(rf->func)->type.getID());
          return true;
        }
        return false;
      }
      if (auto* call = expr->dynCast<Call>()) {
        for (auto operand : call->operands) {
          rehash(digest,
                 ExpressionAnalyzer::flexibleHash(operand, ignoringConsts));
        }
        rehash(digest, call->isReturn);
        if (callIndirection) {
          rehash(digest, module->getFunction(call->target)->type.getID());
        } else {
          rehash(digest, call->target);
        }
        return true;
      }
      return false;
    };
  FunctionHasher(&hashes, ignoringConsts).run(&runner, module);

  std::map<size_t, std::vector<Function*>> hashGroups;
  ModuleUtils::iterDefinedFunctions(*module, [&](Function* func) {
    if (constThunks.count(func->name)) {
      return;
    }
    hashGroups[hashes[func]].push_back(func);
  });

  for (auto& [_, hashGroup] : hashGroups) {
    if (hashGroup.size() < 2) {
      continue;
    }

    std::vector<EquivalentClass> classesInGroup = {
      EquivalentClass(hashGroup[0], {hashGroup[0]})};

    for (Index i = 1; i < hashGroup.size(); i++) {
      auto* func = hashGroup[i];
      bool found = false;
      for (auto& newClass : classesInGroup) {
        if (areInEquvalentClass(newClass.primaryFunction, func, module)) {
          newClass.functions.push_back(func);
          found = true;
          break;
        }
      }

      if (!found) {
        classesInGroup.push_back(EquivalentClass(func, {func}));
      }
    }
    std::copy(classesInGroup.begin(),
              classesInGroup.end(),
              std::back_inserter(classes));
  }
}

bool EquivalentClass::deriveParams(Module* module,
                                   std::vector<ParamInfo>& params,
                                   bool isCallIndirectionEnabled,
                                   bool earlyMode) const {
  struct DeepValueIterator {
    SmallVector<Expression**, 10> tasks;

    DeepValueIterator(Expression** root) { tasks.push_back(root); }

    void operator++() {
      ChildIterator it(*tasks.back());
      tasks.pop_back();
      for (Expression*& child : it) {
        tasks.push_back(&child);
      }
    }

    Expression*& operator*() {
      assert(!empty());
      return *tasks.back();
    }
    bool empty() { return tasks.empty(); }
  };

  if (primaryFunction->imported()) {
    return false;
  }
  DeepValueIterator primaryIt(&primaryFunction->body);
  std::vector<DeepValueIterator> siblingIterators;
  assert(functions.size() >= 2);
  for (auto func = functions.begin() + 1; func != functions.end(); ++func) {
    siblingIterators.emplace_back(&(*func)->body);
  }

  auto extractValueLeaf = [&](Expression* expr, LeafValue& out) -> bool {
    if (auto* c = expr->dynCast<Const>()) {
      out = LeafValue{LeafValue::LiteralVal, c->value.type, c->value, Name()};
      return true;
    }
    if (auto* gg = expr->dynCast<GlobalGet>()) {
      if (module->getGlobal(gg->name)->mutable_) {
        return false;
      }
      out = LeafValue{LeafValue::GlobalVal, gg->type, Literal(), gg->name};
      return true;
    }
    if (isCallIndirectionEnabled) {
      if (auto* rf = expr->dynCast<RefFunc>()) {
        auto* callee = module->getFunction(rf->func);
        Type paramType(callee->type.getHeapType(), NonNullable);
        bool match =
          earlyMode
            ? (paramType == rf->type)
            : (rf->type.getHeapType() == callee->type.getHeapType() &&
               (!rf->type.isExact() || !module->features.hasCustomDescriptors()));
        if (match) {
          out = LeafValue{LeafValue::FuncVal, paramType, Literal(), rf->func};
          return true;
        }
      }
    }
    return false;
  };

  for (; !primaryIt.empty(); ++primaryIt) {
    Expression*& primary = *primaryIt;
    std::vector<LeafValue> values;
    values.reserve(functions.size());

    bool isAllSame = true;
    LeafValue v0;
    if (extractValueLeaf(primary, v0)) {
      values.push_back(v0);
      for (auto& it : siblingIterators) {
        Expression*& sibling = *it;
        ++it;
        LeafValue vi;
        bool ok = extractValueLeaf(sibling, vi);
        assert(ok);
        (void)ok;
        isAllSame &= (v0 == vi);
        values.push_back(vi);
      }
    } else if (isCallIndirectionEnabled && primary->is<Call>()) {
      auto* primaryCall = primary->cast<Call>();
      auto* callee0 = module->getFunction(primaryCall->target);
      Type paramType(callee0->type.getHeapType(), NonNullable);
      v0 = LeafValue{
        LeafValue::FuncVal, paramType, Literal(), primaryCall->target};
      values.push_back(v0);
      for (auto& it : siblingIterators) {
        Expression*& sibling = *it;
        ++it;
        auto* siblingCall = sibling->cast<Call>();
        LeafValue vi{
          LeafValue::FuncVal, paramType, Literal(), siblingCall->target};
        isAllSame &= (v0 == vi);
        values.push_back(vi);
      }
    } else {
      for (auto& it : siblingIterators) {
        assert((*it)->_id == primary->_id);
        ++it;
      }
      continue;
    }

    if (isAllSame) {
      continue;
    }

    bool paramReused = false;
    for (auto& param : params) {
      if (param.values == values) {
        param.uses.push_back(&primary);
        paramReused = true;
        break;
      }
    }
    if (!paramReused) {
      params.push_back(ParamInfo(std::move(values), {&primary}));
    }
  }
  return true;
}

int EquivalentClass::evaluateMergeSavings(
  Module* module,
  const std::vector<ParamInfo>& params,
  const ModuleCallStats& stats,
  const std::unordered_map<Function*, FuncNormalizationState>* normStates,
  bool shrinkMode,
  bool earlyMode) const {
  if (params.empty()) {
    return -1;
  }
  if (params.size() + primaryFunction->getNumParams() >
      MaxSyntheticFunctionParams) {
    return -1;
  }

  std::unordered_set<Expression*> parameterizedExprs;
  int paramUseDeltas = 0;
  std::unordered_set<Name> newlyReferencedFuncs;

  for (const auto& param : params) {
    for (auto* usePtr : param.uses) {
      Expression* use = *usePtr;
      parameterizedExprs.insert(use);
      if (use->is<Call>()) {
        // call (3 B) -> local.get (2 B) + call_ref (2..3 B)
        paramUseDeltas += shrinkMode ? 2 : 1;
      } else {
        paramUseDeltas += 2 - int(getShallowBytes(use));
      }
    }
    if (!param.values.empty() &&
        param.values[0].kind == LeafValue::FuncVal) {
      for (const auto& v : param.values) {
        auto it = stats.refFuncCounts.find(v.name);
        if (it == stats.refFuncCounts.end() || it->second == 0) {
          newlyReferencedFuncs.insert(v.name);
        }
      }
    }
  }

  int primaryTreeBytes = int(getTreeBytes(primaryFunction->body));
  int rewrappedPrimarySaved = 0;
  if (normStates) {
    auto it = normStates->find(primaryFunction);
    if (it != normStates->end()) {
      for (const auto& rec : it->second.unwrappedCalls) {
        bool recParameterized = parameterizedExprs.count(rec.callExpr) > 0;
        if (!recParameterized) {
          for (Expression* ce : rec.insertedConstExprs) {
            if (parameterizedExprs.count(ce) > 0) {
              recParameterized = true;
              break;
            }
          }
        }
        if (!recParameterized) {
          rewrappedPrimarySaved += int(rec.insertedConstBytes);
        }
      }
    }
  }

  int origPrimaryBodyBytes = primaryTreeBytes - rewrappedPrimarySaved;
  if (origPrimaryBodyBytes < 12) {
    return -1;
  }

  int sharedBodyBytes = origPrimaryBodyBytes + paramUseDeltas;
  int sharedLocalDeclBytes = int(getExtraLocalDeclBytes(primaryFunction));
  int sharedOverhead =
    4 + (shrinkMode ? int(newlyReferencedFuncs.size()) * 2 : 0);
  std::vector<Type> sigParams;
  for (const auto& p : primaryFunction->getParams()) {
    sigParams.push_back(p);
  }
  for (const auto& p : params) {
    sigParams.push_back(p.getValueType(module));
  }
  Signature newSig(Type(sigParams), primaryFunction->getResults());
  bool isNewSig = !stats.existingSignatures.count(newSig);
  if (shrinkMode && isNewSig) {
    sharedOverhead += 3;
  }

  int newBytes = sharedBodyBytes + sharedLocalDeclBytes + sharedOverhead;
  int oldBytes = 0;
  int numOrigParams = int(primaryFunction->getNumParams());
  int numForwarded = 0;

  for (size_t i = 0; i < functions.size(); ++i) {
    Function* f = functions[i];
    int fTreeBytes = int(getTreeBytes(f->body));
    if (normStates) {
      auto it = normStates->find(f);
      if (it != normStates->end()) {
        fTreeBytes -= int(it->second.totalUnwrappedConstBytes);
      }
    }
    int fOldBytes = fTreeBytes + int(getExtraLocalDeclBytes(f));

    int extraArgBytes = 0;
    for (const auto& param : params) {
      extraArgBytes += int(param.getArgBytes(i));
    }
    int thunkInstrBytes = 2 * numOrigParams + extraArgBytes + 3;

    auto callIt = stats.callCounts.find(f->name);
    Index callCount = (callIt != stats.callCounts.end()) ? callIt->second : 0;
    auto rfIt = stats.refFuncCounts.find(f->name);
    Index rfCount = (rfIt != stats.refFuncCounts.end()) ? rfIt->second : 0;
    bool canForward =
      (callCount >= 1 && rfCount == 0 &&
       (earlyMode || !newlyReferencedFuncs.count(f->name)) &&
       !stats.exportedFuncs.count(f->name));
    bool forwardSavingsOk =
      earlyMode ? (int(callCount) * extraArgBytes < thunkInstrBytes + 5)
                : (int(callCount) * extraArgBytes <= thunkInstrBytes + 5);

    if (canForward && forwardSavingsOk) {
      oldBytes += fOldBytes + 5;
      newBytes += int(callCount) * extraArgBytes;
      numForwarded++;
    } else {
      oldBytes += fOldBytes;
      newBytes += thunkInstrBytes;
    }
  }

  if (numForwarded > 0 && !shrinkMode && isNewSig) {
    newBytes += 3;
  }

  return oldBytes - newBytes;
}

void MergeSimilarFunctions::splitIntoSubClusters(
  const EquivalentClass& rawClass,
  Module* module,
  bool callIndirection,
  bool shrinkMode,
  std::vector<EquivalentClass>& outClusters,
  std::unordered_set<Function*>& canonicalizedIdenticalFuncs) {
  if (rawClass.functions.size() <= 2) {
    outClusters.push_back(rawClass);
    return;
  }

  std::vector<ParamInfo> allParams;
  bool allValid =
    rawClass.deriveParams(module, allParams, callIndirection, earlyMode);
  if (allValid && allParams.empty()) {
    for (auto* f : rawClass.functions) {
      canonicalizedIdenticalFuncs.insert(f);
    }
    return;
  }
  int allSavings =
    allValid ? rawClass.evaluateMergeSavings(
                 module, allParams, stats, &normStates, shrinkMode, earlyMode)
             : -1;

  // If the entire group already needs <= 1 parameter and is profitable, no
  // sub-clustering can improve on merging the whole group.
  if (allValid && allParams.size() <= 1 && allSavings > 0) {
    outClusters.push_back(rawClass);
    return;
  }

  size_t N = rawClass.functions.size();
  if (!allValid || N > 180 || allParams.size() > 160) {
    if (allSavings > 0) {
      outClusters.push_back(rawClass);
    }
    return;
  }

  // Pre-group functions in rawClass that have identical hole values across all
  // varying parameters so identical functions stay together and reduce N.
  std::vector<std::vector<size_t>> clusters;
  for (size_t i = 0; i < N; ++i) {
    bool joined = false;
    for (auto& c : clusters) {
      size_t rep = c[0];
      bool sameHoles = true;
      for (const auto& p : allParams) {
        if (p.values[rep] != p.values[i]) {
          sameHoles = false;
          break;
        }
      }
      if (sameHoles) {
        c.push_back(i);
        joined = true;
        break;
      }
    }
    if (!joined) {
      clusters.push_back({i});
    }
  }
  for (const auto& c : clusters) {
    if (c.size() >= 2) {
      for (size_t idx : c) {
        canonicalizedIdenticalFuncs.insert(rawClass.functions[idx]);
      }
    }
  }

  auto evalSubset = [&](const std::vector<size_t>& idxs) -> std::pair<int, int> {
    if (idxs.size() < 2) {
      return {-999, 0};
    }
    std::vector<Function*> subFuncs;
    subFuncs.reserve(idxs.size());
    for (size_t idx : idxs) {
      subFuncs.push_back(rawClass.functions[idx]);
    }
    std::vector<ParamInfo> subParams;
    for (const auto& p : allParams) {
      bool same = true;
      const auto& v0 = p.values[idxs[0]];
      std::vector<LeafValue> subVals;
      subVals.reserve(idxs.size());
      subVals.push_back(v0);
      for (size_t k = 1; k < idxs.size(); ++k) {
        const auto& vk = p.values[idxs[k]];
        same &= (v0 == vk);
        subVals.push_back(vk);
      }
      if (same) {
        continue;
      }
      bool reused = false;
      for (auto& sp : subParams) {
        if (sp.values == subVals) {
          sp.uses.insert(sp.uses.end(), p.uses.begin(), p.uses.end());
          reused = true;
          break;
        }
      }
      if (!reused) {
        subParams.emplace_back(std::move(subVals), p.uses);
      }
    }
    if (subParams.empty()) {
      return {0, 0};
    }
    EquivalentClass subClass(rawClass.primaryFunction, std::move(subFuncs));
    int sav = subClass.evaluateMergeSavings(
      module, subParams, stats, &normStates, shrinkMode, earlyMode);
    return {sav, int(subParams.size())};
  };

  int bestTotalSavings = std::max(0, allSavings);
  std::vector<std::vector<size_t>> bestClusters;
  if (allSavings > 0) {
    std::vector<size_t> allIdxs(N);
    for (size_t i = 0; i < N; ++i) {
      allIdxs[i] = i;
    }
    bestClusters.push_back(std::move(allIdxs));
  }

  std::vector<int> clusterScore(clusters.size(), 0);
  while (clusters.size() > 1) {
    std::pair<int, int> bestKey{-999999, -999999};
    int bestI = -1;
    int bestJ = -1;
    int bestMergedScore = 0;

    for (size_t i = 0; i < clusters.size(); ++i) {
      int si = std::max(0, clusterScore[i]);
      for (size_t j = i + 1; j < clusters.size(); ++j) {
        int sj = std::max(0, clusterScore[j]);
        std::vector<size_t> merged = clusters[i];
        merged.insert(merged.end(), clusters[j].begin(), clusters[j].end());
        auto [sij, pij] = evalSubset(merged);
        if (pij > 6) {
          continue;
        }
        int gain = sij - (si + sj);
        std::pair<int, int> key{gain, -pij};
        if (key > bestKey) {
          bestKey = key;
          bestI = int(i);
          bestJ = int(j);
          bestMergedScore = sij;
        }
      }
    }
    if (bestI < 0) {
      break;
    }
    clusters[bestI].insert(clusters[bestI].end(),
                           clusters[bestJ].begin(),
                           clusters[bestJ].end());
    clusterScore[bestI] = bestMergedScore;
    clusters.erase(clusters.begin() + bestJ);
    clusterScore.erase(clusterScore.begin() + bestJ);

    int curTotal = 0;
    for (size_t i = 0; i < clusters.size(); ++i) {
      if (clusterScore[i] > 0) {
        curTotal += clusterScore[i];
      }
    }
    if (curTotal > bestTotalSavings) {
      bestTotalSavings = curTotal;
      bestClusters.clear();
      for (size_t i = 0; i < clusters.size(); ++i) {
        if (clusterScore[i] > 0) {
          bestClusters.push_back(clusters[i]);
        }
      }
    }
  }

  for (auto& c : bestClusters) {
    if (c.size() < 2) {
      continue;
    }
    std::sort(c.begin(), c.end());
    std::vector<Function*> subFuncs;
    subFuncs.reserve(c.size());
    for (size_t idx : c) {
      subFuncs.push_back(rawClass.functions[idx]);
    }
    outClusters.emplace_back(subFuncs[0], std::move(subFuncs));
  }
}

void EquivalentClass::merge(
  Module* module,
  const std::vector<ParamInfo>& params,
  const std::unordered_map<Function*, FuncNormalizationState>* normStates,
  std::vector<std::pair<Function*, Function*>>* mergedThunkPairs) {
  Function* sharedFn = createShared(module, params, normStates);
  for (size_t i = 0; i < functions.size(); ++i) {
    Builder builder(*module);
    auto* func = functions[i];
    std::vector<Expression*> extraArgs;
    for (auto& param : params) {
      extraArgs.push_back(param.lowerToExpression(builder, module, i));
    }
    replaceWithThunk(builder,
                     func,
                     sharedFn,
                     params,
                     extraArgs,
                     module->features.hasTailCall());
    if (mergedThunkPairs) {
      mergedThunkPairs->emplace_back(func, sharedFn);
    }
  }
}

Function* EquivalentClass::createShared(
  Module* module,
  const std::vector<ParamInfo>& params,
  const std::unordered_map<Function*, FuncNormalizationState>* normStates) {
  Name fnName = Names::getValidFunctionName(*module,
                                            std::string("byn$mgfn-shared$") +
                                              primaryFunction->name.toString());
  Builder builder(*module);
  std::vector<Type> sigParams;
  Index extraParamBase = primaryFunction->getNumParams();
  Index newVarBase = primaryFunction->getNumParams() + params.size();

  for (const auto& param : primaryFunction->getParams()) {
    sigParams.push_back(param);
  }
  for (const auto& param : params) {
    sigParams.push_back(param.getValueType(module));
  }

  std::unordered_set<Expression*> parameterizedExprs;
  for (const auto& param : params) {
    for (auto* usePtr : param.uses) {
      parameterizedExprs.insert(*usePtr);
    }
  }

  std::unordered_map<Call*, const UnwrappedCallRecord*> rewrapInPrimary;
  if (normStates) {
    auto it = normStates->find(primaryFunction);
    if (it != normStates->end()) {
      for (const auto& rec : it->second.unwrappedCalls) {
        if (parameterizedExprs.count(rec.callExpr)) {
          continue;
        }
        bool anyConstParameterized = false;
        for (Expression* ce : rec.insertedConstExprs) {
          if (parameterizedExprs.count(ce)) {
            anyConstParameterized = true;
            break;
          }
        }
        if (!anyConstParameterized) {
          rewrapInPrimary[rec.callExpr] = &rec;
        }
      }
    }
  }

  Signature sig(Type(sigParams), primaryFunction->getResults());
  ExpressionManipulator::CustomCopier copier =
    [&](Expression* expr) -> Expression* {
    if (!expr) {
      return nullptr;
    }
    for (Index paramIdx = 0; paramIdx < params.size(); paramIdx++) {
      for (auto& use : params[paramIdx].uses) {
        if (*use != expr) {
          continue;
        }
        auto* paramExpr = builder.makeLocalGet(
          extraParamBase + paramIdx, params[paramIdx].getValueType(module));
        if (auto* call = expr->dynCast<Call>()) {
          ExpressionList operands(module->allocator);
          for (auto* operand : call->operands) {
            operands.push_back(
              ExpressionManipulator::flexibleCopy(operand, *module, copier));
          }
          auto returnType = module->getFunction(call->target)->getResults();
          return builder.makeCallRef(
            paramExpr, operands, returnType, call->isReturn);
        } else {
          return paramExpr;
        }
      }
    }
    if (auto* call = expr->dynCast<Call>()) {
      auto it = rewrapInPrimary.find(call);
      if (it != rewrapInPrimary.end()) {
        const auto* rec = it->second;
        ExpressionList operands(module->allocator);
        for (Index idx : rec->callerArgIndices) {
          operands.push_back(ExpressionManipulator::flexibleCopy(
            call->operands[idx], *module, copier));
        }
        return builder.makeCall(
          rec->origThunkTarget, operands, call->type, call->isReturn);
      }
    }
    if (auto* localGet = expr->dynCast<LocalGet>()) {
      if (primaryFunction->isVar(localGet->index)) {
        localGet->index =
          newVarBase + (localGet->index - primaryFunction->getNumParams());
        localGet->finalize();
        return localGet;
      }
    }
    if (auto* localSet = expr->dynCast<LocalSet>()) {
      if (primaryFunction->isVar(localSet->index)) {
        auto operand =
          ExpressionManipulator::flexibleCopy(localSet->value, *module, copier);
        localSet->index =
          newVarBase + (localSet->index - primaryFunction->getNumParams());
        localSet->value = operand;
        localSet->finalize();
        return localSet;
      }
    }
    return nullptr;
  };
  Expression* body =
    ExpressionManipulator::flexibleCopy(primaryFunction->body, *module, copier);

  bool anyUpgraded = false;
  if (module->features.hasCustomDescriptors()) {
    std::unordered_set<Type> demandedExactTypes;
    for (const auto& t : primaryFunction->getResults()) {
      if (t.isExact()) {
        demandedExactTypes.insert(t);
      }
    }
    for (const auto& t : primaryFunction->vars) {
      if (t.isExact()) {
        demandedExactTypes.insert(t);
      }
    }
    for (const auto& t : primaryFunction->getParams()) {
      if (t.isExact()) {
        demandedExactTypes.insert(t);
      }
    }
    SmallVector<Expression*, 32> exactWorklist;
    exactWorklist.push_back(body);
    while (!exactWorklist.empty()) {
      Expression* curr = exactWorklist.back();
      exactWorklist.pop_back();
      if (auto* call = curr->dynCast<Call>()) {
        for (const auto& t : module->getFunction(call->target)->getParams()) {
          if (t.isExact()) {
            demandedExactTypes.insert(t);
          }
        }
      } else if (auto* cr = curr->dynCast<CallRef>()) {
        if (cr->target->type.isSignature()) {
          for (const auto& t :
               cr->target->type.getHeapType().getSignature().params) {
            if (t.isExact()) {
              demandedExactTypes.insert(t);
            }
          }
        }
      } else if (auto* gs = curr->dynCast<GlobalSet>()) {
        Type gt = module->getGlobal(gs->name)->type;
        if (gt.isExact()) {
          demandedExactTypes.insert(gt);
        }
      } else if (auto* sn = curr->dynCast<StructNew>()) {
        if (sn->type.isStruct()) {
          for (const auto& field : sn->type.getHeapType().getStruct().fields) {
            if (field.type.isExact()) {
              demandedExactTypes.insert(field.type);
            }
          }
        }
      } else if (auto* ss = curr->dynCast<StructSet>()) {
        if (ss->ref->type.isStruct()) {
          Type ft =
            ss->ref->type.getHeapType().getStruct().fields[ss->index].type;
          if (ft.isExact()) {
            demandedExactTypes.insert(ft);
          }
        }
      }
      for (auto* child : ChildIterator(curr)) {
        exactWorklist.push_back(child);
      }
    }

    if (!demandedExactTypes.empty()) {
      for (Index paramIdx = 0; paramIdx < params.size(); ++paramIdx) {
        if (params[paramIdx].values.empty() ||
            params[paramIdx].values[0].kind != LeafValue::FuncVal) {
          continue;
        }
        Type curType = sigParams[extraParamBase + paramIdx];
        Type exactType(curType.getHeapType(), NonNullable, Exact);
        if (demandedExactTypes.count(exactType)) {
          sigParams[extraParamBase + paramIdx] = exactType;
          anyUpgraded = true;
        }
      }
      if (anyUpgraded) {
        sig = Signature(Type(sigParams), primaryFunction->getResults());
        exactWorklist.push_back(body);
        while (!exactWorklist.empty()) {
          Expression* curr = exactWorklist.back();
          exactWorklist.pop_back();
          if (auto* lg = curr->dynCast<LocalGet>()) {
            if (lg->index >= extraParamBase && lg->index < newVarBase) {
              lg->type = sigParams[lg->index];
            }
          }
          for (auto* child : ChildIterator(curr)) {
            exactWorklist.push_back(child);
          }
        }
      }
    }
  }

  auto vars = primaryFunction->vars;
  std::unique_ptr<Function> f = builder.makeFunction(
    fnName, Type(sig, NonNullable, Exact), std::move(vars), body);
  if (anyUpgraded) {
    ReFinalize().walkFunctionInModule(f.get(), module);
  }
  return module->addFunction(std::move(f));
}

Function*
EquivalentClass::replaceWithThunk(Builder& builder,
                                  Function* target,
                                  Function* shared,
                                  const std::vector<ParamInfo>& params,
                                  const std::vector<Expression*>& extraArgs,
                                  bool isReturn) {
  std::vector<Expression*> callOperands;
  Type targetParams = target->getParams();
  for (Index i = 0; i < targetParams.size(); i++) {
    callOperands.push_back(builder.makeLocalGet(i, targetParams[i]));
  }

  for (const auto& value : extraArgs) {
    callOperands.push_back(value);
  }

  auto ret = builder.makeCall(
    shared->name, callOperands, target->getResults(), isReturn);
  target->vars.clear();
  target->body = ret;
  return target;
}

void MergeSimilarFunctions::forwardDirectCallers(
  Module* module,
  const std::vector<std::pair<Function*, Function*>>& mergedThunkPairs) {
  std::unordered_map<Name, std::vector<Call*>> liveCallSites;
  std::unordered_map<Name, Index> liveRefFuncs;

  for (auto& seg : module->elementSegments) {
    for (auto* expr : seg->data) {
      SmallVector<Expression*, 8> worklist;
      worklist.push_back(expr);
      while (!worklist.empty()) {
        Expression* curr = worklist.back();
        worklist.pop_back();
        if (auto* rf = curr->dynCast<RefFunc>()) {
          liveRefFuncs[rf->func]++;
        }
        for (auto* child : ChildIterator(curr)) {
          worklist.push_back(child);
        }
      }
    }
  }
  for (auto& glob : module->globals) {
    if (!glob->imported() && glob->init) {
      SmallVector<Expression*, 8> worklist;
      worklist.push_back(glob->init);
      while (!worklist.empty()) {
        Expression* curr = worklist.back();
        worklist.pop_back();
        if (auto* rf = curr->dynCast<RefFunc>()) {
          liveRefFuncs[rf->func]++;
        }
        for (auto* child : ChildIterator(curr)) {
          worklist.push_back(child);
        }
      }
    }
  }
  ModuleUtils::iterDefinedFunctions(*module, [&](Function* func) {
    if (!func->body) {
      return;
    }
    SmallVector<Expression*, 32> worklist;
    worklist.push_back(func->body);
    while (!worklist.empty()) {
      Expression* curr = worklist.back();
      worklist.pop_back();
      if (auto* call = curr->dynCast<Call>()) {
        liveCallSites[call->target].push_back(call);
      } else if (auto* rf = curr->dynCast<RefFunc>()) {
        liveRefFuncs[rf->func]++;
      }
      for (auto* child : ChildIterator(curr)) {
        worklist.push_back(child);
      }
    }
  });

  std::vector<Name> funcsToRemove;
  for (const auto& [thunkFunc, sharedFn] : mergedThunkPairs) {
    Name thunkName = thunkFunc->name;
    if (stats.exportedFuncs.count(thunkName) || liveRefFuncs[thunkName] > 0) {
      continue;
    }
    auto origCallIt = stats.callCounts.find(thunkName);
    if (origCallIt == stats.callCounts.end() || origCallIt->second == 0) {
      // Never delete functions that had 0 callers prior to the pass (e.g.
      // standalone functions in lit tests).
      continue;
    }
    const auto& sites = liveCallSites[thunkName];
    auto* thunkCall = thunkFunc->body->dynCast<Call>();
    if (!thunkCall || thunkCall->target != sharedFn->name) {
      continue;
    }
    Index numOrigParams = thunkFunc->getNumParams();
    int extraArgBytes = 0;
    for (Index k = numOrigParams; k < thunkCall->operands.size(); ++k) {
      extraArgBytes += int(getShallowBytes(thunkCall->operands[k]));
    }
    int thunkTotalBytes = 8 + 2 * int(numOrigParams) + extraArgBytes;
    bool skipForward =
      earlyMode ? (int(sites.size()) * extraArgBytes >= thunkTotalBytes)
                : (int(sites.size()) * extraArgBytes > thunkTotalBytes);
    if (skipForward) {
      continue;
    }
    bool allSitesValid = true;
    for (Call* site : sites) {
      if (site->operands.size() != numOrigParams ||
          site->type != thunkFunc->getResults()) {
        allSitesValid = false;
        break;
      }
      for (auto* op : site->operands) {
        if (op->type == Type::unreachable) {
          allSitesValid = false;
          break;
        }
      }
      if (!allSitesValid) {
        break;
      }
    }
    if (!allSitesValid) {
      continue;
    }

    for (Call* site : sites) {
      for (Index k = numOrigParams; k < thunkCall->operands.size(); ++k) {
        site->operands.push_back(
          ExpressionManipulator::copy(thunkCall->operands[k], *module));
      }
      site->target = sharedFn->name;
    }
    funcsToRemove.push_back(thunkName);
  }

  for (Name name : funcsToRemove) {
    module->removeFunction(name);
  }
}

Pass* createMergeSimilarFunctionsPass() { return new MergeSimilarFunctions(); }

} // namespace wasm
