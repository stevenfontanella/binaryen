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

    // Collect all Const (and optionally RefFunc) expressions in function
    // bodies, in parallel.
    struct SiteWalker : public PostWalker<SiteWalker> {
      SiteList& sites;
      bool collectRefFuncs;
      SiteWalker(SiteList& sites, bool collectRefFuncs)
        : sites(sites), collectRefFuncs(collectRefFuncs) {}
      void visitConst(Const* curr) { sites.consts.push_back(getCurrentPointer()); }
      void visitRefFunc(RefFunc* curr) {
        if (collectRefFuncs) {
          sites.refFuncs.push_back(getCurrentPointer());
        }
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
    }

    bool hasCompositeGlobals = false;
    if (walkGlobals) {
      for (auto& global : module->globals) {
        if (global->imported() || isPlainGlobalInit(global->init) ||
            global->type.isExact()) {
          continue;
        }
        hasCompositeGlobals = true;
        SiteList sites;
        SiteWalker(sites, collectRefFuncs).walk(global->init);
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
    }

    if (uses.empty() && refFuncUses.empty()) {
      return;
    }

    // Existing immutable globals that are initialized to a plain constant or
    // ref.func can be reused for free.
    std::unordered_map<Literal, Name> existingGlobals;
    std::unordered_map<Name, Name> existingRefFuncGlobals;
    for (auto& global : module->globals) {
      if (global->imported() || global->mutable_) {
        continue;
      }
      if (auto* c = global->init->dynCast<Const>()) {
        if (global->type == c->type && getEncodedSize(c->value) > 0) {
          existingGlobals.insert({c->value, global->name});
        }
      } else if (collectRefFuncs) {
        if (auto* rf = global->init->dynCast<RefFunc>()) {
          if (auto* f = module->getFunctionOrNull(rf->func)) {
            if (global->type == f->type) {
              existingRefFuncGlobals.insert({rf->func, global->name});
            }
          }
        }
      }
    }

    // A global.get is an opcode plus a LEB global index. The new globals are
    // appended (or ordered by reorder-globals), so estimate the index size
    // from the current number of globals.
    Index globalGetSize = 1 + getLEBSize(U32LEB(module->globals.size()));
    // In top128 mode, reorder-globals places the hottest globals into the 128
    // 1-byte U32LEB slots (where global.get is 2 bytes). When the module has
    // >= 128 globals, a new top-128 global displaces the marginal top-128
    // global (~26 uses/slot); when < 128 globals, top-128 slots are free.
    int top128Cutoff = module->globals.size() >= 128 ? 26 : 1;

    Builder builder(*module);
    bool modifiedGlobals = false;
    for (auto& [value, ptrs] : uses) {
      Index encodedSize = getEncodedSize(value);
      Index numUses = ptrs.size();
      Name globalName;
      if (encodedSize > globalGetSize) {
        Index saved = numUses * (encodedSize - globalGetSize);
        auto iter = existingGlobals.find(value);
        if (iter != existingGlobals.end()) {
          globalName = iter->second;
        } else {
          // A new global costs its type, mutability, the initializer (the
          // constant and an end opcode), and a little for the global count and
          // the ordering of other globals.
          Index globalCost = encodedSize + 4;
          if (saved <= globalCost) {
            continue;
          }
          globalName = Names::getValidGlobalName(*module, getGlobalName(value));
          module->addGlobal(builder.makeGlobal(globalName,
                                               value.type,
                                               builder.makeConst(value),
                                               Builder::Immutable));
          existingGlobals.insert({value, globalName});
        }
      } else if (top128 && encodedSize == 3) {
        auto iter = existingGlobals.find(value);
        if (iter != existingGlobals.end()) {
          if (int(numUses) < top128Cutoff) {
            continue;
          }
          globalName = iter->second;
        } else {
          // A new 3-byte Const global costs 6 bytes (1B type + 1B mut + 3B
          // init + 1B end) and saves 1 byte per use when placed in slots
          // 0..127 (2B global.get vs 3B inline const).
          if (int(numUses) - 6 < top128Cutoff) {
            continue;
          }
          globalName = Names::getValidGlobalName(*module, getGlobalName(value));
          module->addGlobal(builder.makeGlobal(globalName,
                                               value.type,
                                               builder.makeConst(value),
                                               Builder::Immutable));
          existingGlobals.insert({value, globalName});
        }
      } else {
        continue;
      }
      for (auto** ptr : ptrs) {
        *ptr = builder.makeGlobalGet(globalName, value.type);
      }
      modifiedGlobals = true;
    }

    std::unordered_set<Function*> funcsToRefinalize;
    if (collectRefFuncs) {
      for (auto& [funcName, ptrs] : refFuncUses) {
        auto* f = module->getFunctionOrNull(funcName);
        if (!f || f->imported()) {
          continue;
        }
        Index numUses = ptrs.size();
        Type globalType = f->type;
        Name globalName;
        auto iter = existingRefFuncGlobals.find(funcName);
        if (iter != existingRefFuncGlobals.end()) {
          if (int(numUses) < top128Cutoff) {
            continue;
          }
          globalName = iter->second;
        } else {
          // A new RefFunc global costs ~7 bytes (2-3B ref type + 1B mut + 3B
          // ref.func + 1B end) and saves 1 byte per use in a top-128 slot.
          if (int(numUses) - 7 < top128Cutoff) {
            continue;
          }
          globalName = Names::getValidGlobalName(
            *module, std::string("ref_func_") + funcName.toString());
          module->addGlobal(builder.makeGlobal(globalName,
                                               globalType,
                                               builder.makeRefFunc(funcName),
                                               Builder::Immutable));
          existingRefFuncGlobals.insert({funcName, globalName});
        }
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
