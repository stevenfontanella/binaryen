/*
 * Copyright 2023 WebAssembly Community Group participants
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
// Optimize J2CL specifics construct to simplify them and enable further
// optimizations by other passes.
//

#include "ir/find_all.h"
#include "ir/global-utils.h"
#include "ir/utils.h"
#include "opt-utils.h"
#include "pass.h"
#include "passes.h"
#include "wasm.h"

#ifndef J2CL_OPT_DEBUG
#define J2CL_OPT_DEBUG 0
#endif

namespace wasm {

namespace {

using AssignmentCountMap = std::unordered_map<Name, Index>;
using TrivialFunctionMap = std::unordered_map<Name, Expression*>;

bool isOnceFunction(Name name) { return name.hasSubstring("_<once>_"); }
bool isOnceFunction(Function* func) { return isOnceFunction(func->name); }

Name getEnclosingClass(Name name) {
  auto pos = name.view().find_last_of('@');
  if (pos == std::string_view::npos) {
    return Name();
  }
  return Name(name.view().substr(pos));
}

// Returns the function body if it is a trivial function, null otherwise.
Expression* getTrivialFunctionBody(Function* func) {
  auto* body = func->body;

  // Only consider trivial the following instructions which can be safely
  // inlined and note that their size is at most 2.
  if (body->is<Nop>() || body->is<GlobalGet>() || body->is<Const>() ||
      (body->is<Block>() && body->cast<Block>()->list.empty()) ||
      // Call with no arguments.
      (body->is<Call>() && body->dynCast<Call>()->operands.size() == 0) ||
      // Simple global.set with a constant.
      (body->is<GlobalSet>() &&
       body->dynCast<GlobalSet>()->value->is<Const>())) {
    return body;
  }
  return nullptr;
}

// Adds the function to the map if it is trivial.
void maybeCollectTrivialFunction(Function* func,
                                 TrivialFunctionMap& trivialFunctionMap) {
  auto* body = getTrivialFunctionBody(func);
  if (body == nullptr) {
    return;
  }

  trivialFunctionMap[func->name] = body;
}

// Cleans up a once function that has been modified in the hopes it
// becomes trivial.
void cleanupFunction(Module* module, Function* func) {
  PassRunner runner(module);
  runner.add("precompute");
  runner.add("vacuum");
  // Run after vacuum to remove the extra returns that vacuum might
  // leave when reducing a block that ends up with just one instruction.
  runner.add("remove-unused-brs");
  runner.setIsNested(true);
  runner.runOnFunction(func);
}

// Extract the GlobalSet* from a lazy null-guarded initialization pattern:
//   (if (ref.is_null (global.get $g)) (then (global.set $g V)))
// allowing unnamed single-element Block wrappers around the GlobalSet.
GlobalSet* getLazyNullGuardSet(If* iff) {
  if (!iff || iff->ifFalse) {
    return nullptr;
  }
  auto* isNull = iff->condition->dynCast<RefIsNull>();
  if (!isNull) {
    return nullptr;
  }
  auto* condGet = isNull->value->dynCast<GlobalGet>();
  if (!condGet) {
    return nullptr;
  }
  Expression* thenBody = iff->ifTrue;
  while (auto* block = thenBody->dynCast<Block>()) {
    if (!block->name.is() && block->list.size() == 1) {
      thenBody = block->list[0];
    } else {
      break;
    }
  }
  auto* set = thenBody->dynCast<GlobalSet>();
  if (!set || set->name != condGet->name) {
    return nullptr;
  }
  return set;
}

int hoistLazyNullGlobals(Module* module,
                         PassRunner* passRunner,
                         AssignmentCountMap& assignmentCounts,
                         TrivialFunctionMap& trivialFunctionMap) {
  bool hasCfp = false;
  bool hasTypeMerging = false;
  bool hasDeferGlobals = false;
  if (passRunner) {
    for (auto& pass : passRunner->passes) {
      if (pass->name == "cfp" || pass->name == "cfp-reftest") {
        hasCfp = true;
      } else if (pass->name == "type-merging") {
        hasTypeMerging = true;
      } else if (pass->name == "defer-global-allocations") {
        hasDeferGlobals = true;
      }
    }
  }
  bool onlyRefFunc = hasCfp && !hasDeferGlobals;
  if (onlyRefFunc && !hasTypeMerging) {
    return 0;
  }

  std::unordered_set<Name> exportedGlobals;
  for (auto& ex : module->exports) {
    if (ex->kind == ExternalKind::Global) {
      exportedGlobals.insert(*ex->getInternalName());
    }
  }

  std::unordered_set<Name> candidates;
  for (auto& global : module->globals) {
    if (!global->imported() && global->mutable_ &&
        !exportedGlobals.contains(global->name) &&
        !global->name.startsWith("$class-initialized@") &&
        global->type.isRef() && global->type.isNullable() && global->init &&
        global->init->is<RefNull>()) {
      candidates.insert(global->name);
    }
  }
  if (candidates.empty()) {
    return 0;
  }

  struct CandidateInfo {
    Index totalSets = 0;
    std::vector<std::pair<Function*, If*>> guardIfs;
    std::vector<std::pair<Function*, GlobalSet*>> onceNullSets;
    std::vector<Function*> readerFuncs;
    Expression* initValue = nullptr;
    bool disqualified = false;
  };
  std::unordered_map<Name, CandidateInfo> infos;

  struct Scanner : public ExpressionStackWalker<Scanner> {
    const std::unordered_set<Name>& candidates;
    std::unordered_map<Name, CandidateInfo>& infos;

    Scanner(const std::unordered_set<Name>& candidates,
            std::unordered_map<Name, CandidateInfo>& infos)
      : candidates(candidates), infos(infos) {}

    void visitGlobalSet(GlobalSet* curr) {
      if (!candidates.contains(curr->name)) {
        return;
      }
      auto& info = infos[curr->name];
      info.totalSets++;
      if (curr->value->is<RefNull>()) {
        if (auto* func = getFunction()) {
          if (isOnceFunction(func)) {
            Name funcClass = getEnclosingClass(func->name);
            if (funcClass.is() && funcClass == getEnclosingClass(curr->name)) {
              bool isTopLevel =
                expressionStack.size() == 1 ||
                (expressionStack.size() == 2 &&
                 expressionStack[0] == func->body && func->body->is<Block>());
              if (isTopLevel) {
                info.onceNullSets.emplace_back(func, curr);
              }
            }
          }
        }
      }
    }

    void visitGlobalGet(GlobalGet* curr) {
      if (!candidates.contains(curr->name)) {
        return;
      }
      auto& info = infos[curr->name];
      if (auto* func = getFunction()) {
        if (info.readerFuncs.empty() || info.readerFuncs.back() != func) {
          info.readerFuncs.push_back(func);
        }
      } else {
        info.disqualified = true;
        return;
      }
      if (expressionStack.size() >= 2) {
        auto* parent = expressionStack[expressionStack.size() - 2];
        if (parent->is<RefIsNull>()) {
          bool isGuardCond = false;
          if (expressionStack.size() >= 3) {
            if (auto* iff =
                  expressionStack[expressionStack.size() - 3]->dynCast<If>()) {
              if (iff->condition == parent) {
                if (auto* set = getLazyNullGuardSet(iff)) {
                  isGuardCond = (set->name == curr->name);
                }
              }
            }
          }
          if (!isGuardCond) {
            info.disqualified = true;
          }
        } else if (parent->is<RefEq>() || parent->is<BrOn>() ||
                   parent->is<RefTest>()) {
          info.disqualified = true;
        }
      }
    }

    void visitIf(If* curr) {
      auto* set = getLazyNullGuardSet(curr);
      if (!set || !candidates.contains(set->name)) {
        return;
      }
      auto& info = infos[set->name];
      info.guardIfs.emplace_back(getFunction(), curr);
      auto* val = set->value;
      if (!val->type.isRef() || !val->type.isNonNullable() ||
          !GlobalUtils::canInitializeGlobal(*getModule(), val)) {
        info.disqualified = true;
        return;
      }
      if (!info.initValue) {
        info.initValue = val;
      } else if (!ExpressionAnalyzer::equal(val, info.initValue)) {
        info.disqualified = true;
      }
    }
  };

  Scanner scanner(candidates, infos);
  scanner.walkModule(module);

  std::unordered_set<Name> promoted;
  std::unordered_set<Function*> funcsWithNoppedGuards;
  std::unordered_set<Function*> funcsToUpdate;

  for (auto& global : module->globals) {
    auto it = infos.find(global->name);
    if (it == infos.end()) {
      continue;
    }
    auto& info = it->second;
    if (info.disqualified || info.guardIfs.empty() ||
        info.totalSets != info.guardIfs.size() + info.onceNullSets.size() ||
        !info.initValue) {
      continue;
    }
    if (onlyRefFunc && FindAll<RefFunc>(info.initValue).list.empty()) {
      continue;
    }
    global->init = ExpressionManipulator::copy(info.initValue, *module);
    global->mutable_ = false;
    global->type = global->init->type;
    assignmentCounts[global->name] = 1;
    promoted.insert(global->name);
    for (auto& [func, iff] : info.guardIfs) {
      ExpressionManipulator::nop(iff);
      if (func) {
        funcsWithNoppedGuards.insert(func);
        funcsToUpdate.insert(func);
      }
    }
    for (auto& [func, set] : info.onceNullSets) {
      ExpressionManipulator::nop(set);
      if (func) {
        funcsWithNoppedGuards.insert(func);
        funcsToUpdate.insert(func);
      }
    }
    for (auto* func : info.readerFuncs) {
      funcsToUpdate.insert(func);
    }
  }

  if (promoted.empty()) {
    return 0;
  }

  struct GetUpdater : public PostWalker<GetUpdater> {
    Module& wasm;
    const std::unordered_set<Name>& promoted;
    GetUpdater(Module& wasm, const std::unordered_set<Name>& promoted)
      : wasm(wasm), promoted(promoted) {}
    void visitGlobalGet(GlobalGet* curr) {
      if (promoted.contains(curr->name)) {
        curr->type = wasm.getGlobal(curr->name)->type;
      }
    }
  } updater(*module, promoted);

  for (auto& func : module->functions) {
    if (!funcsToUpdate.contains(func.get())) {
      continue;
    }
    updater.walkFunctionInModule(func.get(), module);
    ReFinalize().walkFunctionInModule(func.get(), module);
    if (funcsWithNoppedGuards.contains(func.get())) {
      cleanupFunction(module, func.get());
      if (isOnceFunction(func.get())) {
        maybeCollectTrivialFunction(func.get(), trivialFunctionMap);
      }
    }
  }

  return promoted.size();
}

// A visitor to count the number of GlobalSets of each global so we can later
// identify the number of assignments of the global.
// TODO: parallelize
class GlobalAssignmentCollector
  : public WalkerPass<PostWalker<GlobalAssignmentCollector>> {
public:
  GlobalAssignmentCollector(AssignmentCountMap& assignmentCounts)
    : assignmentCounts(assignmentCounts) {}

  void visitGlobal(Global* curr) {
    // Ignore imported globals, e.g., JS prototypes.
    if (curr->imported()) {
      return;
    }
    if (isInitialValue(curr->init)) {
      return;
    }
    // J2CL normally doesn't set non-default initial values, however, just in
    // case other passes in binaryen do something and set a value to the global
    // we should back off by recording this as an assignment.
    recordGlobalAssignment(curr->name);
  }
  void visitGlobalSet(GlobalSet* curr) { recordGlobalAssignment(curr->name); }

private:
  bool isInitialValue(Expression* expr) {
    if (auto* constExpr = expr->dynCast<Const>()) {
      return constExpr->value.isZero();
    } else {
      return expr->is<RefNull>();
    }
  }

  void recordGlobalAssignment(Name name) {
    // Avoid optimizing class initialization condition variable itself. If we
    // were optimizing it then it would become "true" and would defeat its
    // functionality and the clinit would never trigger during runtime.
    if (name.startsWith("$class-initialized@")) {
      return;
    }
    assignmentCounts[name]++;
  }

  AssignmentCountMap& assignmentCounts;
};

// A visitor that moves initialization of constant-like globals from "once"
// functions to global init.
// TODO: parallelize
class ConstantHoister : public WalkerPass<PostWalker<ConstantHoister>> {
public:
  ConstantHoister(AssignmentCountMap& assignmentCounts,
                  TrivialFunctionMap& trivialFunctionMap)
    : assignmentCounts(assignmentCounts),
      trivialFunctionMap(trivialFunctionMap) {}

  int optimized = 0;

  void visitFunction(Function* curr) {
    if (!isOnceFunction(curr)) {
      return;
    }
    Name enclosingClassName = getEnclosingClass(curr->name);
    int optimizedBefore = optimized;
    if (auto* block = curr->body->dynCast<Block>()) {
      for (auto*& expr : block->list) {
        maybeHoistConstant(expr, enclosingClassName);
      }
    } else {
      maybeHoistConstant(curr->body, enclosingClassName);
    }

    if (optimized != optimizedBefore) {
      cleanupFunction(getModule(), curr);
      maybeCollectTrivialFunction(curr, trivialFunctionMap);
    }
  }

private:
  void maybeHoistConstant(Expression* expr, Name enclosingClassName) {
    auto set = expr->dynCast<GlobalSet>();
    if (!set) {
      return;
    }

    if (assignmentCounts[set->name] != 1) {
      // The global assigned in multiple places, so it is not safe to
      // hoist them as global constants.
      return;
    }

    if (getEnclosingClass(set->name) != enclosingClassName) {
      // Only hoist fields initialized by its own class.
      // If it is only initialized once but by another class (although it is
      // very uncommon / edge scenario) then we cannot be sure if the clinit was
      // triggered before the field access so it is better to leave it alone.
      return;
    }

    if (!GlobalUtils::canInitializeGlobal(*getModule(), set->value)) {
      // It is not a valid constant expression so cannot be hoisted to
      // global init.
      return;
    }

    // Move it to global init and mark it as immutable.
    auto global = getModule()->getGlobal(set->name);
    global->init = set->value;
    global->mutable_ = false;
    ExpressionManipulator::nop(expr);

    optimized++;
  }

  AssignmentCountMap& assignmentCounts;
  TrivialFunctionMap& trivialFunctionMap;
};

// Class to collect functions that are already trivial before the pass is run.
// When this pass is run, other optimizations that preceded it might have left
// the body of some of these functions trivial.
// Since the loop in this pass will only inline the functions that are made
// trivial by this pass, the functions that were already trivial before would
// not be inlined if they were not collected by this visitor.
class TrivialOnceFunctionCollector
  : public WalkerPass<PostWalker<TrivialOnceFunctionCollector>> {
public:
  TrivialOnceFunctionCollector(TrivialFunctionMap& trivialFunctionMap)
    : trivialFunctionMap(trivialFunctionMap) {}

  void visitFunction(Function* curr) {
    if (!isOnceFunction(curr)) {
      return;
    }
    maybeCollectTrivialFunction(curr, trivialFunctionMap);
  }

private:
  TrivialFunctionMap& trivialFunctionMap;
};

// A visitor that inlines trivial once functions.
// TODO: parallelize
class InlineTrivialOnceFunctions
  : public WalkerPass<PostWalker<InlineTrivialOnceFunctions>> {
public:
  InlineTrivialOnceFunctions(TrivialFunctionMap& trivialFunctionMap)
    : trivialFunctionMap(trivialFunctionMap) {}

  void visitCall(Call* curr) {
    if (curr->operands.size() != 0 || !isOnceFunction(curr->target)) {
      return;
    }

    auto iter = trivialFunctionMap.find(curr->target);
    if (iter == trivialFunctionMap.end()) {
      return;
    }
    auto* expr = iter->second;

    // The call was to a trivial once function which consists of the expression
    // in <expr>; replace the call with it.
    Builder builder(*getModule());
    auto* replacement = ExpressionManipulator::copy(expr, *getModule());
    replaceCurrent(replacement);

    lastModifiedFunction = getFunction();
    inlined++;
  }

  void visitFunction(Function* curr) {
    // Since the traversal is in post-order, we only need to check if the
    // current function is the function that was last inlined into.
    // We also do not want to do any cleanup for a non-once function (we leave
    // that for other passes, as it will not end up helping further work here).
    if (lastModifiedFunction != curr || !isOnceFunction(curr)) {
      return;
    }

    cleanupFunction(getModule(), curr);
    maybeCollectTrivialFunction(curr, trivialFunctionMap);
  }

  int inlined = 0;

private:
  TrivialFunctionMap& trivialFunctionMap;
  Function* lastModifiedFunction = nullptr;
};

struct J2CLOpts : public Pass {
  void hoistConstants(Module* module) {
    AssignmentCountMap assignmentCounts;
    TrivialFunctionMap trivialFunctionMap;

    GlobalAssignmentCollector collector(assignmentCounts);
    collector.run(getPassRunner(), module);

    TrivialOnceFunctionCollector trivialFunctionCollector(trivialFunctionMap);
    trivialFunctionCollector.run(getPassRunner(), module);

    while (1) {
      ConstantHoister hoister(assignmentCounts, trivialFunctionMap);
      hoister.run(getPassRunner(), module);
      int optimized = hoister.optimized;
      optimized += hoistLazyNullGlobals(
        module, getPassRunner(), assignmentCounts, trivialFunctionMap);

      InlineTrivialOnceFunctions inliner(trivialFunctionMap);
      inliner.run(getPassRunner(), module);
      int inlined = inliner.inlined;

#if J2CL_OPT_DEBUG
      std::cout << "Optimized " << optimized << " global fields\n";
#endif
      if (optimized == 0 && inlined == 0) {
        break;
      }
    }
  }

  void run(Module* module) override {
    if (!module->features.hasGC()) {
      return;
    }
    // Move constant like properties set by the once functions to global
    // initialization.
    hoistConstants(module);

    // We might have introduced new globals depending on other globals. Reorder
    // order them so they follow initialization order.
    // TODO: do this only if have introduced a new global.
    PassRunner runner(module);
    runner.add("reorder-globals-always");
    runner.setIsNested(true);
    runner.run();
  }
};

} // anonymous namespace

Pass* createJ2CLOptsPass() { return new J2CLOpts(); }

} // namespace wasm
