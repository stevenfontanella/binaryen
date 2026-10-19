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
// Hoists repeated wide constants in function bodies into immutable globals.
//
// An f64.const is 9 bytes, an i64.const up to 11, and an i32.const of a large
// value up to 6, while a global.get is typically 2-3 bytes. When the same value
// appears often enough in the code, it is smaller to define it once as an
// immutable global and read that global everywhere. Existing immutable globals
// whose initializer is exactly the constant are reused.
//
// Constants in global initializers are never touched (they must remain
// constant expressions, and nothing is gained there).
//
// Later optimization passes (precompute, simplify-globals) propagate constant
// globals back into the code, so this pass is meant to run at the very end of
// a pipeline (e.g. before reorder-globals, which then assigns the new globals
// small indexes).
//

#include <algorithm>
#include <queue>

#include "ir/module-utils.h"
#include "ir/names.h"
#include "ir/utils.h"
#include "pass.h"
#include "support/insert_ordered.h"
#include "wasm-binary.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

struct HoistConstants : public Pass {
  using ExprPtrs = std::vector<Expression**>;

  struct SiteList {
    ExprPtrs consts;
    ExprPtrs refFuncs;
    std::vector<GlobalGet*> globalGets;
    std::unordered_map<Name, Index> globalSetCounts;
  };

  static bool isPlainGlobalInit(Expression* init) {
    return init->is<Const>() || init->is<RefFunc>() || init->is<RefNull>();
  }

  void run(Module* module) override {
    auto arg = getArgumentOrDefault("hoist-constants", "");
    bool functionsOnly =
      arg == "functions-only" || hasArgument("hoist-constants-functions-only");
    bool top128 = arg == "top128" || hasArgument("hoist-constants-top128");
    bool collectRefFuncs = top128 && module->features.hasGC() &&
                           !module->features.hasCustomDescriptors();
    bool walkGlobals = module->features.hasGC() && !functionsOnly;

    // Collect all Const (and optionally RefFunc) expressions and GlobalGet/Set
    // uses in function bodies, in parallel.
    struct SiteWalker : public PostWalker<SiteWalker> {
      SiteList& sites;
      bool collectRefFuncs;
      SiteWalker(SiteList& sites, bool collectRefFuncs)
        : sites(sites), collectRefFuncs(collectRefFuncs) {}
      void visitConst(Const* curr) {
        sites.consts.push_back(getCurrentPointer());
      }
      void visitRefFunc(RefFunc* curr) {
        if (collectRefFuncs) {
          sites.refFuncs.push_back(getCurrentPointer());
        }
      }
      void visitGlobalGet(GlobalGet* curr) {
        sites.globalGets.push_back(curr);
      }
      void visitGlobalSet(GlobalSet* curr) {
        sites.globalSetCounts[curr->name]++;
      }
    };

    ModuleUtils::ParallelFunctionAnalysis<SiteList> analysis(
      *module, [&](Function* func, SiteList& sites) {
        if (!func->imported()) {
          SiteWalker(sites, collectRefFuncs).walk(func->body);
        }
      });

    // Group the uses by value, in a deterministic order (function order, then
    // composite global order, then use order within each).
    InsertOrderedMap<Literal, ExprPtrs> uses;
    InsertOrderedMap<Name, ExprPtrs> refFuncUses;
    std::unordered_map<Name, std::vector<Function*>> refFuncUsers;
    std::unordered_map<Name, Index> existingGlobalUses;
    for (auto& func : module->functions) {
      auto& sites = analysis.map[func.get()];
      for (auto** ptr : sites.consts) {
        auto& value = (*ptr)->cast<Const>()->value;
        if (getEncodedSize(value) > 0) {
          uses[value].push_back(ptr);
        }
      }
      for (auto** ptr : sites.refFuncs) {
        auto target = (*ptr)->cast<RefFunc>()->func;
        refFuncUses[target].push_back(ptr);
        auto& vec = refFuncUsers[target];
        if (vec.empty() || vec.back() != func.get()) {
          vec.push_back(func.get());
        }
      }
      for (auto* gg : sites.globalGets) {
        existingGlobalUses[gg->name]++;
      }
      for (auto& [name, count] : sites.globalSetCounts) {
        existingGlobalUses[name] += count;
      }
    }

    bool hasCompositeGlobals = false;
    std::vector<GlobalGet*> globalInitGlobalGets;
    for (auto& global : module->globals) {
      if (global->imported() || isPlainGlobalInit(global->init)) {
        continue;
      }
      bool canHoistFromGlobal = walkGlobals && !global->type.isExact();
      SiteList sites;
      SiteWalker(sites, canHoistFromGlobal && collectRefFuncs)
        .walk(global->init);
      for (auto* gg : sites.globalGets) {
        globalInitGlobalGets.push_back(gg);
        existingGlobalUses[gg->name]++;
      }
      if (!canHoistFromGlobal) {
        continue;
      }
      hasCompositeGlobals = true;
      for (auto** ptr : sites.consts) {
        auto& value = (*ptr)->cast<Const>()->value;
        if (getEncodedSize(value) > 0) {
          uses[value].push_back(ptr);
        }
      }
      for (auto** ptr : sites.refFuncs) {
        auto target = (*ptr)->cast<RefFunc>()->func;
        refFuncUses[target].push_back(ptr);
      }
    }

    std::unordered_set<Name> exportedGlobals;
    for (auto& exp : module->exports) {
      if (exp->kind == ExternalKind::Global) {
        exportedGlobals.insert(*exp->getInternalName());
      }
    }

    // Existing immutable globals that are initialized to a plain constant or
    // ref.func can be reused for free, and duplicate immutable Const/RefFunc
    // globals can be deduplicated.
    std::unordered_map<Literal, Name> existingGlobals;
    std::unordered_map<Name, Name> existingRefFuncGlobals;
    std::unordered_map<Name, Name> globalRenameMap;
    std::unordered_set<Name> globalsToRemove;
    std::unordered_map<Name, Type> upgradedGlobalTypes;
    for (auto& global : module->globals) {
      if (global->imported() || global->mutable_) {
        continue;
      }
      if (auto* c = global->init->dynCast<Const>()) {
        if (global->type == c->type && getEncodedSize(c->value) > 0) {
          auto [it, inserted] =
            existingGlobals.insert({c->value, global->name});
          if (!inserted) {
            globalRenameMap[global->name] = it->second;
            existingGlobalUses[it->second] += existingGlobalUses[global->name];
            existingGlobalUses.erase(global->name);
            if (!exportedGlobals.count(global->name)) {
              globalsToRemove.insert(global->name);
            }
          }
        }
      } else if (collectRefFuncs) {
        if (auto* rf = global->init->dynCast<RefFunc>()) {
          if (auto* f = module->getFunctionOrNull(rf->func)) {
            if (global->type.isRef() && f->type.isRef() &&
                global->type.getHeapType() == f->type.getHeapType() &&
                global->type.isNullable() == f->type.isNullable()) {
              auto [it, inserted] =
                existingRefFuncGlobals.insert({rf->func, global->name});
              if (inserted) {
                if (global->type != f->type) {
                  global->type = f->type;
                  rf->type = f->type;
                  upgradedGlobalTypes[global->name] = f->type;
                }
              } else {
                globalRenameMap[global->name] = it->second;
                existingGlobalUses[it->second] +=
                  existingGlobalUses[global->name];
                existingGlobalUses.erase(global->name);
                if (!exportedGlobals.count(global->name)) {
                  globalsToRemove.insert(global->name);
                }
              }
            }
          }
        }
      }
    }

    std::unordered_set<Function*> funcsToRefinalize;
    bool modifiedGlobals = false;
    if (!globalRenameMap.empty() || !upgradedGlobalTypes.empty()) {
      for (auto& func : module->functions) {
        bool needsRefinalize = false;
        for (auto* gg : analysis.map[func.get()].globalGets) {
          if (auto it = globalRenameMap.find(gg->name);
              it != globalRenameMap.end()) {
            gg->name = it->second;
            Type newType = module->getGlobal(gg->name)->type;
            if (gg->type != newType) {
              gg->type = newType;
              needsRefinalize = true;
            }
          } else if (auto it = upgradedGlobalTypes.find(gg->name);
                     it != upgradedGlobalTypes.end()) {
            if (gg->type != it->second) {
              gg->type = it->second;
              needsRefinalize = true;
            }
          }
        }
        if (needsRefinalize) {
          funcsToRefinalize.insert(func.get());
        }
      }
      for (auto* gg : globalInitGlobalGets) {
        if (auto it = globalRenameMap.find(gg->name);
            it != globalRenameMap.end()) {
          gg->name = it->second;
          gg->type = module->getGlobal(gg->name)->type;
        } else if (auto it = upgradedGlobalTypes.find(gg->name);
                   it != upgradedGlobalTypes.end()) {
          gg->type = it->second;
        }
      }
      if (!globalsToRemove.empty()) {
        module->removeGlobals(
          [&](Global* g) { return globalsToRemove.count(g->name) > 0; });
      }
      modifiedGlobals = true;
    }

    if (uses.empty() && refFuncUses.empty()) {
      for (auto* func : funcsToRefinalize) {
        ReFinalize().walkFunctionInModule(func, module);
      }
      return;
    }

    // A global.get is an opcode plus a LEB global index. The new globals are
    // appended (or ordered by reorder-globals), so estimate the index size
    // from the current number of globals.
    Index globalGetSize = 1 + getLEBSize(U32LEB(module->globals.size()));

    Builder builder(*module);
    // First pass over constants: hoist wide constants (encodedSize >
    // globalGetSize), which are profitable even outside the top-128 1-byte
    // global slots.
    for (auto& [value, ptrs] : uses) {
      Index encodedSize = getEncodedSize(value);
      if (encodedSize <= globalGetSize) {
        continue;
      }
      Index numUses = ptrs.size();
      Index saved = numUses * (encodedSize - globalGetSize);
      Name globalName;
      auto iter = existingGlobals.find(value);
      if (iter != existingGlobals.end()) {
        globalName = iter->second;
        existingGlobalUses[globalName] += numUses;
      } else {
        // A new global costs its type, mutability, the initializer (the
        // constant and an end opcode), and a little for the global count and
        // the ordering of other globals.
        Index globalCost = encodedSize + 4;
        if (saved <= globalCost) {
          continue;
        }
        globalName = Names::getValidGlobalName(*module, getGlobalName(value));
        module->addGlobal(builder.makeGlobal(
          globalName, value.type, builder.makeConst(value), Builder::Immutable));
        existingGlobals.insert({value, globalName});
        existingGlobalUses[globalName] = numUses;
      }
      for (auto** ptr : ptrs) {
        *ptr = builder.makeGlobalGet(globalName, value.type);
      }
      modifiedGlobals = true;
    }

    // In top128 mode, reorder-globals places the hottest globals into the 128
    // 1-byte U32LEB slots (where global.get is 2 bytes). Build a min-heap of
    // the top-128 existing global use counts so we know the exact marginal
    // displacement cost of adding or promoting a global into the top 128.
    std::priority_queue<int, std::vector<int>, std::greater<int>> top128Heap;
    if (top128) {
      for (auto& g : module->globals) {
        int u = int(existingGlobalUses[g->name]);
        if (top128Heap.size() < 128) {
          top128Heap.push(u);
        } else if (u > top128Heap.top()) {
          top128Heap.pop();
          top128Heap.push(u);
        }
      }
    }
    auto getMarginal128Uses = [&]() -> int {
      return top128Heap.size() >= 128 ? top128Heap.top() : 0;
    };
    auto recordTop128Global = [&](int totalUses, int oldUses) {
      if (top128Heap.size() < 128) {
        if (oldUses == 0) {
          top128Heap.push(totalUses);
        }
      } else if (oldUses <= top128Heap.top() && totalUses > top128Heap.top()) {
        top128Heap.pop();
        top128Heap.push(totalUses);
      }
    };

    // Second pass in top128 mode: first reuse any existing 3-byte Const or
    // RefFunc globals whose total uses place them in the top 128, then
    // evaluate new 3-byte Const and RefFunc global candidates in descending
    // order of net savings before displacement.
    struct Top128Candidate {
      bool isRefFunc;
      Literal constVal;
      Name funcName;
      ExprPtrs* ptrs;
      int numUses;
      int globalCost;
      Index orderIdx;
      bool accepted = false;
    };
    std::vector<Top128Candidate> candidates;
    Index nextOrderIdx = 0;

    if (top128) {
      for (auto& [value, ptrs] : uses) {
        Index encodedSize = getEncodedSize(value);
        if (encodedSize > globalGetSize || encodedSize != 3) {
          continue;
        }
        Index numUses = ptrs.size();
        auto iter = existingGlobals.find(value);
        if (iter != existingGlobals.end()) {
          Name globalName = iter->second;
          int oldUses = int(existingGlobalUses[globalName]);
          int totalUses = oldUses + int(numUses);
          if (totalUses <= getMarginal128Uses()) {
            continue;
          }
          existingGlobalUses[globalName] = totalUses;
          recordTop128Global(totalUses, oldUses);
          for (auto** ptr : ptrs) {
            *ptr = builder.makeGlobalGet(globalName, value.type);
          }
          modifiedGlobals = true;
        } else {
          // A new 3-byte Const global costs 6 bytes (1B type + 1B mut + 3B
          // init + 1B end) and saves 1 byte per use in slots 0..127.
          candidates.push_back(
            {false, value, Name(), &ptrs, int(numUses), 6, nextOrderIdx++});
        }
      }

      if (collectRefFuncs) {
        for (auto& [funcName, ptrs] : refFuncUses) {
          auto* f = module->getFunctionOrNull(funcName);
          if (!f || f->imported()) {
            continue;
          }
          Index numUses = ptrs.size();
          auto iter = existingRefFuncGlobals.find(funcName);
          if (iter != existingRefFuncGlobals.end()) {
            Name globalName = iter->second;
            int oldUses = int(existingGlobalUses[globalName]);
            int totalUses = oldUses + int(numUses);
            if (totalUses <= getMarginal128Uses()) {
              continue;
            }
            existingGlobalUses[globalName] = totalUses;
            recordTop128Global(totalUses, oldUses);
            Type globalType = module->getGlobal(globalName)->type;
            for (auto** ptr : ptrs) {
              *ptr = builder.makeGlobalGet(globalName, globalType);
            }
            if (auto uIter = refFuncUsers.find(funcName);
                uIter != refFuncUsers.end()) {
              for (auto* userFunc : uIter->second) {
                funcsToRefinalize.insert(userFunc);
              }
            }
            modifiedGlobals = true;
          } else {
            // A new RefFunc global costs ~7 bytes (2-3B ref type + 1B mut + 3B
            // ref.func + 1B end) and saves 1 byte per use in a top-128 slot.
            candidates.push_back({true,
                                  Literal(),
                                  funcName,
                                  &ptrs,
                                  int(numUses),
                                  7,
                                  nextOrderIdx++});
          }
        }
      }

      std::vector<Index> sortedCandIdxs(candidates.size());
      for (Index i = 0; i < candidates.size(); i++) {
        sortedCandIdxs[i] = i;
      }
      std::sort(sortedCandIdxs.begin(),
                sortedCandIdxs.end(),
                [&](Index a, Index b) {
                  int netA = candidates[a].numUses - candidates[a].globalCost;
                  int netB = candidates[b].numUses - candidates[b].globalCost;
                  if (netA != netB) {
                    return netA > netB;
                  }
                  return candidates[a].orderIdx < candidates[b].orderIdx;
                });

      for (Index idx : sortedCandIdxs) {
        auto& cand = candidates[idx];
        int marginal128Uses = getMarginal128Uses();
        if (cand.numUses - cand.globalCost > marginal128Uses) {
          cand.accepted = true;
          recordTop128Global(cand.numUses, 0);
        }
      }

      for (auto& cand : candidates) {
        if (!cand.accepted) {
          continue;
        }
        if (!cand.isRefFunc) {
          Name globalName =
            Names::getValidGlobalName(*module, getGlobalName(cand.constVal));
          module->addGlobal(builder.makeGlobal(globalName,
                                               cand.constVal.type,
                                               builder.makeConst(cand.constVal),
                                               Builder::Immutable));
          existingGlobals.insert({cand.constVal, globalName});
          for (auto** ptr : *cand.ptrs) {
            *ptr = builder.makeGlobalGet(globalName, cand.constVal.type);
          }
          modifiedGlobals = true;
        } else {
          auto* f = module->getFunction(cand.funcName);
          Type globalType = f->type;
          Name globalName = Names::getValidGlobalName(
            *module, std::string("ref_func_") + cand.funcName.toString());
          module->addGlobal(
            builder.makeGlobal(globalName,
                               globalType,
                               builder.makeRefFunc(cand.funcName),
                               Builder::Immutable));
          existingRefFuncGlobals.insert({cand.funcName, globalName});
          for (auto** ptr : *cand.ptrs) {
            *ptr = builder.makeGlobalGet(globalName, globalType);
          }
          if (auto uIter = refFuncUsers.find(cand.funcName);
              uIter != refFuncUsers.end()) {
            for (auto* userFunc : uIter->second) {
              funcsToRefinalize.insert(userFunc);
            }
          }
          modifiedGlobals = true;
        }
      }
    }

    for (auto* func : funcsToRefinalize) {
      ReFinalize().walkFunctionInModule(func, module);
    }

    // If composite globals were walked and globals were modified, ensure all
    // plain immutable globals precede all composite globals so that any
    // global.get added to a composite global's initializer references a
    // preceding global.
    if (hasCompositeGlobals && modifiedGlobals) {
      std::vector<std::unique_ptr<Global>> imported;
      std::vector<std::unique_ptr<Global>> plainImmutable;
      std::vector<std::unique_ptr<Global>> rest;
      for (auto& g : module->globals) {
        if (g->imported()) {
          imported.push_back(std::move(g));
        } else if (!g->mutable_ && isPlainGlobalInit(g->init)) {
          plainImmutable.push_back(std::move(g));
        } else {
          rest.push_back(std::move(g));
        }
      }
      module->globals.clear();
      for (auto& g : imported) {
        module->globals.push_back(std::move(g));
      }
      for (auto& g : plainImmutable) {
        module->globals.push_back(std::move(g));
      }
      for (auto& g : rest) {
        module->globals.push_back(std::move(g));
      }
      module->updateMaps();
    }
  }

private:
  // Returns the size in bytes of the binary encoding of a constant (including
  // the opcode), or 0 if the type is not handled.
  Index getEncodedSize(const Literal& value) {
    if (!value.type.isBasic()) {
      return 0;
    }
    switch (value.type.getBasic()) {
      case Type::i32:
        return 1 + getLEBSize(S32LEB(value.geti32()));
      case Type::i64:
        return 1 + getLEBSize(S64LEB(value.geti64()));
      case Type::f32:
        return 1 + 4;
      case Type::f64:
        return 1 + 8;
      case Type::v128:
      case Type::none:
      case Type::unreachable:
        return 0;
    }
    WASM_UNREACHABLE("unexpected type");
  }

  template<typename T> Index getLEBSize(const T& leb) {
    BufferWithRandomAccess buffer;
    buffer << leb;
    return buffer.size();
  }

  std::string getGlobalName(const Literal& value) {
    std::stringstream ss;
    ss << "const_" << value.type << '_';
    switch (value.type.getBasic()) {
      case Type::i32:
        ss << value.geti32();
        break;
      case Type::i64:
        ss << value.geti64();
        break;
      case Type::f32:
        Literal::printFloat(ss, value.getf32());
        break;
      case Type::f64:
        Literal::printDouble(ss, value.getf64());
        break;
      default:
        WASM_UNREACHABLE("unexpected type");
    }
    return ss.str();
  }
};

} // anonymous namespace

Pass* createHoistConstantsPass() { return new HoistConstants(); }

} // namespace wasm
