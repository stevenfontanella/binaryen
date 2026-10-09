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
// SparseFieldToLookup: Replace reads of a sparsely-populated immutable
// reference field with a call to a per-field lookup function, so that the
// field itself can be removed from the type (by a following --gto).
//
// Consider a vtable-like type hierarchy whose instances are all created in
// global initializers, with an immutable nullable field that is null in almost
// every instance:
//
//  (global $vt1 (struct.new $VT (ref.null none) ...))
//  (global $vt2 (struct.new $VT (ref.func $f) ...))
//  (global $vt3 (struct.new $VT (ref.null none) ...))
//  ... thousands more, nearly all null ...
//
// Each null operand costs bytes in the global section, and the field costs
// bytes in every type declaration of the hierarchy. If the field is read only
// in a few places, it is smaller to replace the reads by a lookup keyed on the
// identity of the receiver:
//
//  (func $lookup (param (ref null $VT)) (result funcref)
//    (if (ref.eq (local.get 0) (global.get $vt2)) (return (ref.func $f)))
//    (ref.null nofunc))
//
// and to replace each (struct.get $VT k (X)) with (call $lookup (X)). After
// that the field has no reads at all, and GlobalTypeOptimization (--gto)
// removes it from all the types and drops the operands from all struct.news.
//
// Conditions (closed world only):
//  * The field is immutable and of nullable reference type, and the type that
//    declares it and all of its subtypes are private.
//  * Every struct.new of a type in the hierarchy that is in function code (or
//    in a table/elem initializer) passes null for the field. Non-null values
//    only appear in global initializers.
//  * The number of non-null instances is at most MaxEntries (pass argument
//    sparse-field-to-lookup-max-entries, default 32), as the lookup is a
//    linear chain of comparisons.
//  * A byte-size model says it is worthwhile.
//
// Instances that are not the direct initializer of an immutable global are
// hoisted into new immutable globals so that they can serve as keys. Field
// values that are not trivially copyable (anything but ref.func and a get of an
// immutable global) are hoisted into new immutable globals too.
//
// This pass is meant to run once, late in a pipeline, followed by --gto.
//

#include "ir/find_all.h"
#include "ir/module-utils.h"
#include "ir/names.h"
#include "ir/subtypes.h"
#include "pass.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

struct FieldKey {
  HeapType type;
  Index index;
  bool operator==(const FieldKey& other) const {
    return type == other.type && index == other.index;
  }
};

struct FieldKeyHash {
  size_t operator()(const FieldKey& key) const {
    return std::hash<HeapType>{}(key.type) ^ (size_t(key.index) * 0x9e3779b9);
  }
};

struct SparseFieldToLookup : public Pass {
  // Only adds functions and globals and replaces struct.gets with calls.
  bool requiresNonNullableLocalFixups() override { return false; }

  // Approximate byte costs.
  // A null operand in a struct.new (ref.null + heap type).
  static constexpr Index NullOperandBytes = 2;
  // A field in a type declaration (storage type + mutability).
  static constexpr Index FieldDeclBytes = 2;
  // One if/ref.eq/return entry in the lookup function, including the overhead
  // of hoisting values and nested instances to globals (amortized).
  static constexpr Index EntryBytes = 18;
  // The lookup function itself (function entry, type, body wrapper).
  static constexpr Index FunctionBytes = 6;

  Index maxEntries = 32;

  // Information about struct.gets, collected per function.
  struct FunctionInfo {
    // Reads of each field.
    std::unordered_map<FieldKey, Index, FieldKeyHash> reads;
    // Fields that are read atomically (which we leave alone).
    std::unordered_set<FieldKey, FieldKeyHash> atomicReads;
    // Fields that receive a non-null value in a struct.new in code.
    std::unordered_set<FieldKey, FieldKeyHash> nonNullNews;
    // Null operands passed in struct.news in code, per field.
    std::unordered_map<FieldKey, Index, FieldKeyHash> nullNews;
  };

  // A struct.new in a global initializer.
  struct GlobalSite {
    // The global that contains the struct.new.
    Name global;
    // The location of the struct.new in that global's init.
    Expression** location;
    StructNew* structNew;
    // Whether the struct.new is the direct init of an immutable global, in
    // which case that global can be used as the key.
    bool isImmutableGlobalInit;
  };

  // What to do with the reads of a field.
  struct Lookup {
    Name func;
    Type resultType;
  };

  Module* module;

  void run(Module* module_) override {
    module = module_;
    if (!module->features.hasGC()) {
      return;
    }
    if (getPassOptions().worldMode != WorldMode::Closed) {
      Fatal() << "SparseFieldToLookup requires --closed-world";
    }

    auto arg = getArgumentOrDefault(
      "sparse-field-to-lookup",
      getArgumentOrDefault("sparse-field-to-lookup-max-entries", ""));
    if (!arg.empty()) {
      maxEntries = std::stoul(arg);
    }

    // Collect information from function code, in parallel.
    ModuleUtils::ParallelFunctionAnalysis<FunctionInfo> analysis(
      *module, [&](Function* func, FunctionInfo& info) {
        if (func->imported()) {
          return;
        }
        struct Scanner : public PostWalker<Scanner> {
          FunctionInfo& info;
          Scanner(FunctionInfo& info) : info(info) {}
          void visitStructGet(StructGet* curr) {
            auto type = curr->ref->type;
            if (!type.isRef() || !type.getHeapType().isStruct()) {
              return;
            }
            FieldKey key{type.getHeapType(), curr->index};
            info.reads[key]++;
            if (curr->order != MemoryOrder::Unordered) {
              info.atomicReads.insert(key);
            }
          }
          void visitStructNew(StructNew* curr) {
            if (!curr->type.isRef()) {
              return;
            }
            auto heapType = curr->type.getHeapType();
            auto& fields = heapType.getStruct().fields;
            for (Index i = 0; i < fields.size(); i++) {
              FieldKey key{heapType, i};
              if (isNullOperand(curr, i)) {
                info.nullNews[key]++;
              } else {
                info.nonNullNews.insert(key);
              }
            }
          }
        };
        Scanner(info).walk(func->body);
      });

    // Merge the per-function information.
    FunctionInfo codeInfo;
    for (auto& func : module->functions) {
      auto& info = analysis.map[func.get()];
      for (auto& [key, count] : info.reads) {
        codeInfo.reads[key] += count;
      }
      for (auto& key : info.atomicReads) {
        codeInfo.atomicReads.insert(key);
      }
      for (auto& key : info.nonNullNews) {
        codeInfo.nonNullNews.insert(key);
      }
      for (auto& [key, count] : info.nullNews) {
        codeInfo.nullNews[key] += count;
      }
    }

    // Types created in table or elem initializers cannot be optimized at all
    // (such objects are not identifiable by a global).
    std::unordered_set<HeapType> unoptimizableTypes;
    auto markUnoptimizable = [&](Expression* init) {
      for (auto* structNew : FindAll<StructNew>(init).list) {
        if (structNew->type.isRef()) {
          unoptimizableTypes.insert(structNew->type.getHeapType());
        }
      }
    };
    for (auto& table : module->tables) {
      if (table->init) {
        markUnoptimizable(table->init);
      }
    }
    for (auto& segment : module->elementSegments) {
      for (auto* item : segment->data) {
        markUnoptimizable(item);
      }
    }

    // Collect the struct.news in global initializers, in module order. Within
    // a global the order is post-order (children before parents), which is
    // what we want when hoisting nested instances.
    std::unordered_map<HeapType, std::vector<GlobalSite>> globalSites;
    for (auto& global : module->globals) {
      if (global->imported()) {
        continue;
      }
      for (auto** location : FindAllPointers<StructNew>(global->init).list) {
        auto* structNew = (*location)->cast<StructNew>();
        if (!structNew->type.isRef()) {
          continue;
        }
        bool isImmutableGlobalInit =
          structNew == global->init && !global->mutable_;
        globalSites[structNew->type.getHeapType()].push_back(
          GlobalSite{global->name, location, structNew, isImmutableGlobalInit});
      }
    }

    // Find the candidate fields.
    auto privateTypesVec =
      ModuleUtils::getPrivateHeapTypes(*module, getPassOptions().worldMode);
    std::unordered_set<HeapType> privateTypes(privateTypesVec.begin(),
                                              privateTypesVec.end());
    SubTypes subTypes(*module);

    std::unordered_map<FieldKey, Lookup, FieldKeyHash> lookups;
    std::vector<std::unique_ptr<Function>> newFunctions;
    bool addedGlobals = false;

    for (auto type : privateTypesVec) {
      if (!type.isStruct() || type.isShared()) {
        continue;
      }
      auto& fields = type.getStruct().fields;
      // We only consider fields that this type declares (and not ones it
      // inherits), as the field's hierarchy starts here.
      Index firstDeclared = 0;
      if (auto super = type.getDeclaredSuperType()) {
        firstDeclared = super->getStruct().fields.size();
      }
      auto cone = subTypes.getSubTypes(type);
      bool coneOk = true;
      for (auto sub : cone) {
        if (!privateTypes.contains(sub) || unoptimizableTypes.contains(sub)) {
          coneOk = false;
          break;
        }
      }
      if (!coneOk) {
        continue;
      }

      for (Index k = firstDeclared; k < fields.size(); k++) {
        auto& field = fields[k];
        if (field.mutable_ == Mutable || !field.type.isRef() ||
            !field.type.isNullable()) {
          continue;
        }

        // Gather the data for this field over the whole cone.
        Index reads = 0;
        Index nullCount = 0;
        bool ok = true;
        std::vector<GlobalSite*> entries;
        for (auto sub : cone) {
          FieldKey key{sub, k};
          if (codeInfo.nonNullNews.contains(key) ||
              codeInfo.atomicReads.contains(key)) {
            ok = false;
            break;
          }
          if (auto iter = codeInfo.reads.find(key);
              iter != codeInfo.reads.end()) {
            reads += iter->second;
          }
          if (auto iter = codeInfo.nullNews.find(key);
              iter != codeInfo.nullNews.end()) {
            nullCount += iter->second;
          }
          if (auto iter = globalSites.find(sub); iter != globalSites.end()) {
            for (auto& site : iter->second) {
              if (isNullOperand(site.structNew, k)) {
                nullCount++;
              } else {
                entries.push_back(&site);
              }
            }
          }
          if (entries.size() > maxEntries) {
            ok = false;
            break;
          }
        }
        if (!ok || reads == 0 || entries.empty()) {
          // With no reads, GTO can remove the field by itself; with no
          // non-null entries, CFP handles it.
          continue;
        }

        // The size model.
        Index benefit =
          nullCount * NullOperandBytes + cone.size() * FieldDeclBytes;
        Index cost = entries.size() * EntryBytes + FunctionBytes;
        if (benefit <= cost) {
          continue;
        }

        // Optimize! Build the lookup function.
        Builder builder(*module);
        auto paramType = Type(type, Nullable);
        auto resultType = field.type;
        std::string typeName;
        if (auto iter = module->typeNames.find(type);
            iter != module->typeNames.end()) {
          typeName = iter->second.name.toString();
        } else {
          typeName = std::to_string(newFunctions.size());
        }
        auto funcName = Names::getValidFunctionName(
          *module, "lookup$" + typeName + "$" + std::to_string(k));
        std::vector<Expression*> body;
        for (auto* site : entries) {
          auto keyGlobal = getKeyGlobal(*site, addedGlobals);
          auto* value = getValue(*site, k, addedGlobals);
          auto* keyGet = builder.makeGlobalGet(
            keyGlobal, module->getGlobal(keyGlobal)->type);
          body.push_back(builder.makeIf(
            builder.makeRefEq(builder.makeLocalGet(0, paramType), keyGet),
            builder.makeReturn(value)));
        }
        body.push_back(
          builder.makeRefNull(resultType.getHeapType().getBottom()));
        auto func =
          builder.makeFunction(funcName,
                               HeapType(Signature(paramType, resultType)),
                               {},
                               builder.makeBlock(body, resultType));
        newFunctions.push_back(std::move(func));

        for (auto sub : cone) {
          lookups[FieldKey{sub, k}] = Lookup{funcName, resultType};
        }
      }
    }

    if (lookups.empty()) {
      return;
    }

    for (auto& func : newFunctions) {
      module->addFunction(std::move(func));
    }

    // Replace the reads, in parallel.
    struct Replacer : public WalkerPass<PostWalker<Replacer>> {
      bool isFunctionParallel() override { return true; }
      std::unique_ptr<Pass> create() override {
        return std::make_unique<Replacer>(lookups);
      }
      const std::unordered_map<FieldKey, Lookup, FieldKeyHash>& lookups;
      Replacer(
        const std::unordered_map<FieldKey, Lookup, FieldKeyHash>& lookups)
        : lookups(lookups) {}

      void visitStructGet(StructGet* curr) {
        auto type = curr->ref->type;
        if (!type.isRef() || !type.getHeapType().isStruct()) {
          return;
        }
        auto iter = lookups.find(FieldKey{type.getHeapType(), curr->index});
        if (iter == lookups.end()) {
          return;
        }
        auto& lookup = iter->second;
        Builder builder(*getModule());
        auto* ref = curr->ref;
        if (ref->type.isNullable() && !getPassOptions().trapsNeverHappen) {
          // The struct.get trapped on null; preserve that.
          ref = builder.makeRefAs(RefAsNonNull, ref);
        }
        Expression* replacement =
          builder.makeCall(lookup.func, {ref}, lookup.resultType);
        if (curr->type != Type::unreachable &&
            replacement->type != curr->type) {
          // The field is refined in this subtype; the lookup returns the
          // declaring type's field type, so cast back (this never fails).
          replacement = builder.makeRefCast(replacement, curr->type);
        }
        replaceCurrent(replacement);
      }
    };
    Replacer(lookups).run(getPassRunner(), module);

    if (addedGlobals) {
      // Sort the globals so that added ones appear before their uses.
      PassRunner runner(module);
      runner.add("reorder-globals-always");
      runner.setIsNested(true);
      runner.run();
    }
  }

  static bool isNullOperand(StructNew* structNew, Index k) {
    if (structNew->isWithDefault()) {
      return true;
    }
    return structNew->operands[k]->is<RefNull>();
  }

  // Returns the name of an immutable global whose value is the object created
  // by the struct.new at the given site, hoisting it into a new global if
  // needed.
  Name getKeyGlobal(GlobalSite& site, bool& addedGlobals) {
    if (site.isImmutableGlobalInit) {
      return site.global;
    }
    if (auto* get = (*site.location)->dynCast<GlobalGet>()) {
      // Already hoisted (for another field).
      return get->name;
    }
    assert(*site.location == site.structNew);
    Builder builder(*module);
    auto name =
      Names::getValidGlobalName(*module, site.global.toString() + ".hoisted");
    auto type = Type(site.structNew->type.getHeapType(), NonNullable);
    module->addGlobal(
      builder.makeGlobal(name, type, site.structNew, Builder::Immutable));
    *site.location = builder.makeGlobalGet(name, type);
    addedGlobals = true;
    return name;
  }

  // Returns an expression for the value of field k at the given site, which
  // is safe to use in the lookup function. Values that are not trivially
  // copyable are hoisted into a new immutable global, and the struct.new then
  // refers to that global (until GTO removes the operand entirely).
  Expression* getValue(GlobalSite& site, Index k, bool& addedGlobals) {
    Builder builder(*module);
    auto* operand = site.structNew->operands[k];
    if (auto* get = operand->dynCast<GlobalGet>()) {
      auto* global = module->getGlobal(get->name);
      if (!global->mutable_) {
        return builder.makeGlobalGet(get->name, global->type);
      }
    } else if (auto* func = operand->dynCast<RefFunc>()) {
      return builder.makeRefFunc(func->func, func->type);
    }
    auto name = Names::getValidGlobalName(
      *module, site.global.toString() + ".field" + std::to_string(k));
    auto type = operand->type;
    module->addGlobal(
      builder.makeGlobal(name, type, operand, Builder::Immutable));
    site.structNew->operands[k] = builder.makeGlobalGet(name, type);
    addedGlobals = true;
    return builder.makeGlobalGet(name, type);
  }
};

} // anonymous namespace

Pass* createSparseFieldToLookupPass() { return new SparseFieldToLookup(); }

} // namespace wasm
