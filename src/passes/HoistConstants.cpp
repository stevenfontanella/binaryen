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
#include "pass.h"
#include "support/insert_ordered.h"
#include "wasm-binary.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

struct HoistConstants : public Pass {
  using ConstPtrs = std::vector<Expression**>;

  void run(Module* module) override {
    // Collect all Const expressions in function bodies, in parallel.
    struct ConstWalker : public PostWalker<ConstWalker> {
      ConstPtrs& ptrs;
      ConstWalker(ConstPtrs& ptrs) : ptrs(ptrs) {}
      void visitConst(Const* curr) { ptrs.push_back(getCurrentPointer()); }
    };

    ModuleUtils::ParallelFunctionAnalysis<ConstPtrs> analysis(
      *module, [&](Function* func, ConstPtrs& ptrs) {
        if (!func->imported()) {
          ConstWalker(ptrs).walk(func->body);
        }
      });

    // Group the uses by value, in a deterministic order (function order, then
    // use order within each function).
    InsertOrderedMap<Literal, ConstPtrs> uses;
    for (auto& func : module->functions) {
      for (auto** ptr : analysis.map[func.get()]) {
        auto& value = (*ptr)->cast<Const>()->value;
        if (getEncodedSize(value) > 0) {
          uses[value].push_back(ptr);
        }
      }
    }
    if (uses.empty()) {
      return;
    }

    // Existing immutable globals that are initialized to a plain constant can
    // be reused for free.
    std::unordered_map<Literal, Name> existingGlobals;
    for (auto& global : module->globals) {
      if (global->imported() || global->mutable_) {
        continue;
      }
      if (auto* c = global->init->dynCast<Const>()) {
        if (global->type == c->type && getEncodedSize(c->value) > 0) {
          existingGlobals.insert({c->value, global->name});
        }
      }
    }

    // A global.get is an opcode plus a LEB global index. The new globals are
    // appended, so estimate the index size from the current number of globals.
    Index globalGetSize = 1 + getLEBSize(U32LEB(module->globals.size()));

    Builder builder(*module);
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
      for (auto** ptr : ptrs) {
        *ptr = builder.makeGlobalGet(globalName, value.type);
      }
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
