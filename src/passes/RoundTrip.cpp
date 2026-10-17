/*
 * Copyright 2019 WebAssembly Community Group participants
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
// Write the module to binary, and load it from there. This is useful in
// testing to check for the effects of roundtripping in a single wasm-opt
// parameter.
//

#include "ir/module-utils.h"
#include "pass.h"
#include "wasm-binary.h"
#include "wasm-traversal.h"
#include "wasm.h"

namespace wasm {

struct RoundTrip : public Pass {
  // Reloading the wasm may alter function names etc., which means our global
  // function effect tracking can get confused, and effects may seem to appear.
  // To avoid that, mark this pass as adding effects, which will clear all
  // cached effects and such.
  bool addsEffects() override { return true; }

  void run(Module* module) override {
    BufferWithRandomAccess buffer;
    // Save features, which would not otherwise make it through a round trip if
    // the target features section has been stripped. We also need them in order
    // to tell the builder which features to build with.
    auto features = module->features;

    // We need to know whether we should preserve the type order when we read
    // the module back in.
    bool preserveTypeOrder = !module->typeIndices.empty();

    // Preserve OutlineExpressions multi-write scratch global names (oe$out_*)
    // and DeferGlobalAllocations PAL helper function names (defer$pal$*)
    // across binary roundtripping so subsequent runs reuse them.
    Index numImportedGlobals = 0;
    ModuleUtils::iterImportedGlobals(
      *module, [&](Global* g) { numImportedGlobals += g->type.size(); });
    std::vector<std::pair<Index, Name>> savedScratchGlobals;
    Index definedGlobalIdx = 0;
    ModuleUtils::iterDefinedGlobals(*module, [&](Global* g) {
      if (g->name.startsWith("oe$out_") && g->type.size() == 1) {
        savedScratchGlobals.emplace_back(numImportedGlobals + definedGlobalIdx,
                                         g->name);
      }
      definedGlobalIdx += g->type.size();
    });

    Index numImportedFuncs = 0;
    ModuleUtils::iterImportedFunctions(*module,
                                       [&](Function*) { numImportedFuncs++; });
    std::vector<std::pair<Index, Name>> savedPalFuncs;
    Index definedFuncIdx = 0;
    ModuleUtils::iterDefinedFunctions(*module, [&](Function* f) {
      if (f->name.startsWith("defer$pal$")) {
        savedPalFuncs.emplace_back(numImportedFuncs + definedFuncIdx, f->name);
      }
      definedFuncIdx++;
    });

    // Write, clear, and read the module
    WasmBinaryWriter(module, buffer, getPassOptions()).write();
    ModuleUtils::clearModule(*module);
    auto input = buffer.getAsChars();
    WasmBinaryReader parser(*module, features, input);
    parser.setDWARF(getPassOptions().debugInfo);
    try {
      parser.read();
    } catch (ParseException& p) {
      p.dump(std::cerr);
      std::cerr << '\n';
      Fatal() << "error in parsing wasm binary";
    }

    if (!preserveTypeOrder) {
      module->typeIndices.clear();
    }

    if (!savedScratchGlobals.empty()) {
      std::unordered_map<Name, Name> globalRenameMap;
      for (auto& [idx, oldName] : savedScratchGlobals) {
        if (idx < module->globals.size() && !module->getGlobalOrNull(oldName)) {
          Name currName = module->globals[idx]->name;
          if (currName != oldName) {
            globalRenameMap[currName] = oldName;
            module->globals[idx]->name = oldName;
          }
        }
      }
      if (!globalRenameMap.empty()) {
        module->updateMaps();
        struct GlobalRenamer : public PostWalker<GlobalRenamer> {
          const std::unordered_map<Name, Name>& renameMap;
          GlobalRenamer(const std::unordered_map<Name, Name>& renameMap)
            : renameMap(renameMap) {}
          void visitGlobalGet(GlobalGet* curr) {
            if (auto it = renameMap.find(curr->name); it != renameMap.end()) {
              curr->name = it->second;
            }
          }
          void visitGlobalSet(GlobalSet* curr) {
            if (auto it = renameMap.find(curr->name); it != renameMap.end()) {
              curr->name = it->second;
            }
          }
        } renamer(globalRenameMap);
        renamer.walkModule(module);
        for (auto& ex : module->exports) {
          if (ex->kind == ExternalKind::Global) {
            if (auto it = globalRenameMap.find(*ex->getInternalName());
                it != globalRenameMap.end()) {
              *ex->getInternalName() = it->second;
            }
          }
        }
      }
    }

    if (!savedPalFuncs.empty()) {
      std::map<Name, Name> funcRenameMap;
      for (auto& [idx, oldName] : savedPalFuncs) {
        if (idx < module->functions.size() &&
            !module->getFunctionOrNull(oldName)) {
          Name currName = module->functions[idx]->name;
          if (currName != oldName) {
            funcRenameMap[currName] = oldName;
          }
        }
      }
      if (!funcRenameMap.empty()) {
        ModuleUtils::renameFunctions(*module, funcRenameMap);
      }
    }
  }
};

Pass* createRoundTripPass() { return new RoundTrip(); }

} // namespace wasm
