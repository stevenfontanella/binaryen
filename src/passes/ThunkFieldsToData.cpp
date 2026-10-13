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
// ThunkFieldsToData: Replace vtable-like fields whose values are all
// references to nullary "thunk" functions (functions that just return a
// constant or an immutable global) with the data that the thunks return.
//
// Consider a hierarchy of vtable types, created in global initializers, with
// a getClass slot:
//
//  (type $vt (sub (struct (field (ref $get_class)))))
//  (type $get_class (func (result (ref null $Class))))
//  (global $vt1 (struct.new $vt (ref.func $class1_thunk)))
//  (func $class1_thunk (result (ref null $Class)) (global.get $class1))
//  (global $class1 (struct.new $Class ...))
//  ...
//  (call_ref $get_class (struct.get $vt 0 (X)))
//
// Each thunk costs a function, and the call_ref costs more than a struct.get.
// This pass handles fields (over the whole cone of the declaring type) whose
// every struct.new operand is a ref.func of a thunk, and whose every read is
// the target of a call_ref, in one of two ways:
//
//  * Eager: the field type becomes the thunk result type, the operands become
//    the thunk bodies, and (call_ref (struct.get)) becomes (struct.get).
//
//  * Lazy: when the thunks return immutable globals that hold identical
//    constant objects (like the Class objects above, which are all created
//    with the same operands and are used nowhere else), the field becomes a
//    mutable nullable slot initialized to null, and a helper function creates
//    the object on first read and caches it in the slot, keeping the
//    one-object-per-vtable identity:
//
//    (func $helper (param $vt (ref null $vt)) (result (ref null $Class))
//      (block $out (result (ref null $Class))
//        (br_on_non_null $out (struct.get $vt 0 (local.get $vt)))
//        ;; sites whose thunk is not a constant keep calling the thunk, keyed
//        ;; on the identity of the vtable
//        (if (ref.eq (local.get $vt) (global.get $vtX)) (return (call
//        $thunkX))) (struct.set $vt 0 (local.get $vt)
//          (local.tee $obj (struct.new $Class ...)))
//        (local.get $obj)))
//
//    and (call_ref (struct.get $vt 0 (X))) becomes (call $helper (X)).
//
//  * Indexed: when every thunk is a call to the same function $F whose
//    operands are the thunk's parameters (in order, each used at most once)
//    plus exactly one constant,
//
//    (func $thunk5 (param $this (ref $vt)) (result (ref $Str))
//      (call $F (i32.const 5)))
//
//    the field becomes the constant's type (the constant is the data that
//    differs per vtable) and the call becomes a direct call of $F:
//
//    (call_ref $sig (struct.get $vt 0 (X)) (args...)) =>
//    (call $F (args... with (struct.get $vt 0 (X)) in the constant's place))
//
//    (the receiver and arguments are spilled to locals first if they have
//    side effects, to keep the evaluation order). This handles factory thunks
//    like those DeferGlobalAllocations creates.
//
// The thunks (and the globals they returned, in lazy mode) become
// unreferenced; a following --remove-unused-module-elements removes them.
//
// Conditions (closed world only): the cone of the declaring type is private,
// all its instances are created in global initializers, the field is
// immutable, and its type is a signature (nullary, except in indexed mode).
// A byte-size model decides per field; --pass-arg=thunk-fields-to-data-always@1
// ignores the model.
//

#include <algorithm>

#include "ir/effects.h"
#include "ir/find_all.h"
#include "ir/module-utils.h"
#include "ir/names.h"
#include "ir/subtypes.h"
#include "ir/type-updating.h"
#include "ir/utils.h"
#include "pass.h"
#include "wasm-binary.h"
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

struct ThunkFieldsToData : public Pass {
  // The identity chain in a lazy helper is linear, so keep it short.
  static constexpr Index MaxExceptions = 16;

  // Approximate byte costs.
  // A function entry (function section, size, locals, end) beyond its body.
  static constexpr Index FunctionOverheadBytes = 5;
  // The lazy helper function.
  static constexpr Index HelperBytes = 40;
  // One identity check in the helper (if, ref.eq, local.get, global.get,
  // return, call).
  static constexpr Index ExceptionBytes = 14;
  // Hoisting a nested struct.new into a global.
  static constexpr Index HoistBytes = 6;
  // What DeferGlobalAllocations would save on a global that we make eager.
  static constexpr Index DeferredGlobalBytes = 15;

  struct FunctionInfo {
    std::unordered_map<FieldKey, Index, FieldKeyHash> reads;
    // Reads that are directly the target of a call_ref without operands.
    std::unordered_map<FieldKey, Index, FieldKeyHash> callReads;
    std::unordered_set<HeapType> news;
    std::unordered_map<Name, Index> funcUses;
    std::unordered_map<Name, Index> globalUses;
  };

  // A struct.new in a global initializer.
  struct GlobalSite {
    Name global;
    Expression** location;
    StructNew* structNew;
    bool isImmutableGlobalInit;
  };

  enum class SiteMode { Eager, Lazy, Exception };

  struct Site {
    GlobalSite* globalSite;
    Function* thunk;
    SiteMode mode;
    // Indexed mode: the constant the thunk passes to the callee.
    Const* constant = nullptr;
  };

  struct Decision {
    HeapType type;
    Index index;
    bool lazy = false;
    // The thunk result type.
    Type resultType;
    // The new field type.
    Type fieldType;
    std::vector<Site> sites;
    // Lazy mode: the template object and the helper.
    StructNew* lazyTemplate = nullptr;
    Name helper;
    // Indexed mode: the callee and its operand shape (ConstOperand for the
    // constant, otherwise the index of the thunk parameter).
    bool indexed = false;
    Name callee;
    std::vector<Index> shape;
  };

  static constexpr Index ConstOperand = Index(-1);

  // The classification of a thunk in indexed mode.
  struct IndexedThunk {
    Name callee;
    std::vector<Index> shape;
    Const* constant;
  };

  Module* module;
  bool always = false;
  std::unordered_map<Name, Index> funcUses;
  std::unordered_map<Name, Index> globalUses;
  std::unordered_map<Name, Index> globalCodeUses;
  std::unordered_map<Name, Index> globalIndexes;
  std::unordered_map<Name, Index> functionIndexes;

  void run(Module* module_) override {
    module = module_;
    if (!module->features.hasGC()) {
      return;
    }
    if (getPassOptions().worldMode != WorldMode::Closed) {
      Fatal() << "ThunkFieldsToData requires --closed-world";
    }
    always = !getArgumentOrDefault("thunk-fields-to-data-always", "").empty();

    computeIndexes();

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
            info.reads[FieldKey{type.getHeapType(), curr->index}]++;
          }
          // Reads that are directly the target of a call_ref.
          void visitCallRef(CallRef* curr) {
            auto* get = curr->target->dynCast<StructGet>();
            if (!get || get->type == Type::unreachable ||
                get->order != MemoryOrder::Unordered) {
              return;
            }
            auto type = get->ref->type;
            if (!type.isRef() || !type.getHeapType().isStruct()) {
              return;
            }
            info.callReads[FieldKey{type.getHeapType(), get->index}]++;
          }
          void visitStructNew(StructNew* curr) {
            if (curr->type.isRef()) {
              info.news.insert(curr->type.getHeapType());
            }
          }
          void visitRefFunc(RefFunc* curr) { info.funcUses[curr->func]++; }
          void visitCall(Call* curr) { info.funcUses[curr->target]++; }
          void visitGlobalGet(GlobalGet* curr) {
            info.globalUses[curr->name]++;
          }
        };
        Scanner(info).walk(func->body);
      });

    FunctionInfo codeInfo;
    for (auto& func : module->functions) {
      auto& info = analysis.map[func.get()];
      for (auto& [key, count] : info.reads) {
        codeInfo.reads[key] += count;
      }
      for (auto& [key, count] : info.callReads) {
        codeInfo.callReads[key] += count;
      }
      for (auto type : info.news) {
        codeInfo.news.insert(type);
      }
      for (auto& [name, count] : info.funcUses) {
        funcUses[name] += count;
      }
      for (auto& [name, count] : info.globalUses) {
        globalUses[name] += count;
        globalCodeUses[name] += count;
      }
    }

    // Uses outside of function code.
    struct NonCodeScanner : public PostWalker<NonCodeScanner> {
      ThunkFieldsToData& parent;
      NonCodeScanner(ThunkFieldsToData& parent) : parent(parent) {}
      void visitRefFunc(RefFunc* curr) { parent.funcUses[curr->func]++; }
      void visitGlobalGet(GlobalGet* curr) { parent.globalUses[curr->name]++; }
    } nonCodeScanner(*this);
    std::unordered_set<HeapType> unoptimizableTypes;
    auto markUnoptimizable = [&](Expression* init) {
      for (auto* structNew : FindAll<StructNew>(init).list) {
        if (structNew->type.isRef()) {
          unoptimizableTypes.insert(structNew->type.getHeapType());
        }
      }
    };
    for (auto& global : module->globals) {
      if (!global->imported()) {
        nonCodeScanner.walk(global->init);
      }
    }
    for (auto& segment : module->elementSegments) {
      if (segment->offset) {
        nonCodeScanner.walk(segment->offset);
        markUnoptimizable(segment->offset);
      }
      for (auto*& item : segment->data) {
        nonCodeScanner.walk(item);
        markUnoptimizable(item);
      }
    }
    for (auto& segment : module->dataSegments) {
      if (segment->offset) {
        nonCodeScanner.walk(segment->offset);
      }
    }
    for (auto& ex : module->exports) {
      if (ex->kind == ExternalKind::Function) {
        funcUses[*ex->getInternalName()]++;
      } else if (ex->kind == ExternalKind::Global) {
        globalUses[*ex->getInternalName()]++;
      }
    }
    if (module->start.is()) {
      funcUses[module->start]++;
    }
    for (auto type : codeInfo.news) {
      unoptimizableTypes.insert(type);
    }

    // Collect the struct.news in global initializers.
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

    std::vector<Decision> decisions;
    for (auto type : privateTypesVec) {
      if (!type.isStruct() || type.isShared()) {
        continue;
      }
      auto& fields = type.getStruct().fields;
      Index firstDeclared = 0;
      if (auto super = type.getDeclaredSuperType()) {
        firstDeclared = super->getStruct().fields.size();
      }
      if (firstDeclared >= fields.size()) {
        continue;
      }
      auto cone = subTypes.getSubTypes(type);
      bool coneOk = true;
      for (auto sub : cone) {
        if (!privateTypes.count(sub) || unoptimizableTypes.count(sub)) {
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
            !field.type.getHeapType().isSignature()) {
          continue;
        }
        auto sig = field.type.getHeapType().getSignature();
        if (!sig.results.isConcrete() || sig.results.isTuple()) {
          continue;
        }
        auto decision = analyzeField(type, k, cone, codeInfo, globalSites);
        if (decision) {
          decisions.push_back(std::move(*decision));
        }
      }
    }
    if (decisions.empty()) {
      return;
    }

    apply(decisions);
  }

private:
  void computeIndexes() {
    Index importedGlobals = 0;
    for (auto& global : module->globals) {
      if (global->imported()) {
        importedGlobals++;
      }
    }
    Index nextImported = 0, nextDefined = importedGlobals;
    for (auto& global : module->globals) {
      globalIndexes[global->name] =
        global->imported() ? nextImported++ : nextDefined++;
    }
    Index importedFuncs = 0;
    for (auto& func : module->functions) {
      if (func->imported()) {
        importedFuncs++;
      }
    }
    nextImported = 0;
    nextDefined = importedFuncs;
    for (auto& func : module->functions) {
      functionIndexes[func->name] =
        func->imported() ? nextImported++ : nextDefined++;
    }
  }

  template<typename T> Index getLEBSize(const T& leb) {
    BufferWithRandomAccess buffer;
    buffer << leb;
    return buffer.size();
  }

  // The estimated number of bytes for a constant expression.
  Index getOperandBytes(Expression* curr) {
    if (auto* c = curr->dynCast<Const>()) {
      auto type = c->type;
      if (type == Type::i32) {
        return 1 + getLEBSize(S32LEB(c->value.geti32()));
      } else if (type == Type::i64) {
        return 1 + getLEBSize(S64LEB(c->value.geti64()));
      } else if (type == Type::f32) {
        return 5;
      } else if (type == Type::f64) {
        return 9;
      }
      return 17;
    }
    if (auto* null = curr->dynCast<RefNull>()) {
      return null->type.getHeapType().isBasic() ? 2 : 3;
    }
    if (auto* func = curr->dynCast<RefFunc>()) {
      return 1 + getLEBSize(U32LEB(functionIndexes[func->func]));
    }
    if (auto* get = curr->dynCast<GlobalGet>()) {
      return 1 + getLEBSize(U32LEB(globalIndexes[get->name]));
    }
    if (auto* i31 = curr->dynCast<RefI31>()) {
      return 2 + getOperandBytes(i31->value);
    }
    // struct.new_default: opcode and a type index.
    return 4;
  }

  Index getGlobalEntryBytes(StructNew* init) {
    Index bytes = 3 + 1 + 4 + 1;
    for (auto* operand : init->operands) {
      bytes += getOperandBytes(operand);
    }
    return bytes;
  }

  // Whether an expression is a constant that can be copied into a global
  // initializer and into the lazy helper.
  bool isValidConstant(Expression* curr) {
    if (curr->is<Const>() || curr->is<RefNull>() || curr->is<RefFunc>()) {
      return true;
    }
    if (auto* get = curr->dynCast<GlobalGet>()) {
      return !module->getGlobal(get->name)->mutable_;
    }
    if (auto* i31 = curr->dynCast<RefI31>()) {
      return i31->value->is<Const>();
    }
    return false;
  }

  // Whether a global init is a struct.new of constants that could be created
  // lazily.
  bool isLazyInit(Expression* init) {
    auto* structNew = init->dynCast<StructNew>();
    if (!structNew || structNew->desc || !structNew->type.isRef() ||
        structNew->type.getHeapType().isShared()) {
      return false;
    }
    for (auto* operand : structNew->operands) {
      if (isValidConstant(operand)) {
        continue;
      }
      if (auto* nested = operand->dynCast<StructNew>()) {
        // struct.new_default
        if (nested->operands.empty() && !nested->desc) {
          continue;
        }
      }
      return false;
    }
    return true;
  }

  // Whether DeferGlobalAllocations would defer this global if we left it
  // alone (so that making it eager has a cost).
  bool isDeferrable(Global* global) {
    if (global->mutable_ || global->imported() ||
        globalUses[global->name] != globalCodeUses[global->name] ||
        !isLazyInit(global->init)) {
      return false;
    }
    auto* init = global->init->cast<StructNew>();
    return global->type.isRef() && !global->type.isNullable() &&
           global->type.getHeapType() == init->type.getHeapType();
  }

  // The body of a thunk if it is a constant we can copy, or null.
  Expression* getThunkConstant(Function* func) {
    if (func->imported()) {
      return nullptr;
    }
    auto* body = func->body;
    if (body->is<Const>() || body->is<RefNull>() || body->is<RefFunc>()) {
      return body;
    }
    if (auto* get = body->dynCast<GlobalGet>()) {
      if (!module->getGlobal(get->name)->mutable_) {
        return body;
      }
    }
    return nullptr;
  }

  // Classifies a thunk whose body is a call of another function with the
  // thunk's parameters (in order, each at most once) and exactly one constant
  // as operands.
  std::optional<IndexedThunk> getIndexedThunk(Function* func) {
    if (func->imported()) {
      return {};
    }
    auto* call = func->body->dynCast<Call>();
    if (!call || call->target == func->name) {
      return {};
    }
    IndexedThunk info;
    info.callee = call->target;
    info.constant = nullptr;
    Index nextParam = 0;
    Index numParams = func->getNumParams();
    for (auto* operand : call->operands) {
      if (auto* c = operand->dynCast<Const>()) {
        if (info.constant || !c->type.isNumber()) {
          return {};
        }
        info.constant = c;
        info.shape.push_back(ConstOperand);
      } else if (auto* get = operand->dynCast<LocalGet>()) {
        if (get->index < nextParam || get->index >= numParams) {
          return {};
        }
        nextParam = get->index + 1;
        info.shape.push_back(get->index);
      } else {
        return {};
      }
    }
    if (!info.constant) {
      return {};
    }
    return info;
  }

  std::optional<Decision>
  analyzeField(HeapType type,
               Index k,
               const std::vector<HeapType>& cone,
               FunctionInfo& codeInfo,
               std::unordered_map<HeapType, std::vector<GlobalSite>>& sites) {
    auto& field = type.getStruct().fields[k];
    auto resultType = field.type.getHeapType().getSignature().results;

    // All reads must be call_ref targets.
    Index reads = 0, callReads = 0;
    for (auto sub : cone) {
      FieldKey key{sub, k};
      if (auto iter = codeInfo.reads.find(key); iter != codeInfo.reads.end()) {
        reads += iter->second;
      }
      if (auto iter = codeInfo.callReads.find(key);
          iter != codeInfo.callReads.end()) {
        callReads += iter->second;
      }
    }
    if (reads == 0 || reads != callReads) {
      return {};
    }

    // All operands must be ref.funcs of functions.
    Decision decision;
    decision.type = type;
    decision.index = k;
    decision.resultType = resultType;
    std::unordered_map<Name, Index> siteUses;
    for (auto sub : cone) {
      auto iter = sites.find(sub);
      if (iter == sites.end()) {
        continue;
      }
      for (auto& site : iter->second) {
        if (site.structNew->isWithDefault()) {
          return {};
        }
        auto* refFunc = site.structNew->operands[k]->dynCast<RefFunc>();
        if (!refFunc) {
          return {};
        }
        auto* func = module->getFunction(refFunc->func);
        siteUses[func->name]++;
        decision.sites.push_back(Site{&site, func, SiteMode::Eager});
      }
    }
    if (decision.sites.empty()) {
      return {};
    }

    // Indexed mode: all thunks call the same function with the same shape.
    if (auto indexed = analyzeIndexed(decision, reads)) {
      return indexed;
    }

    // Eager and lazy modes need nullary thunks.
    for (auto& site : decision.sites) {
      if (site.thunk->getParams() != Type::none) {
        return {};
      }
    }

    // Classify the sites. Lazy candidates are thunks used only here that
    // return a global used only by the thunk, whose init is a constant object
    // of the result type.
    std::vector<std::pair<StructNew*, std::vector<Index>>> templates;
    for (Index i = 0; i < decision.sites.size(); i++) {
      auto& site = decision.sites[i];
      auto* constant = getThunkConstant(site.thunk);
      if (!constant) {
        site.mode = SiteMode::Exception;
        continue;
      }
      auto* get = constant->dynCast<GlobalGet>();
      if (!get || !resultType.isRef() || siteUses[site.thunk->name] != 1 ||
          funcUses[site.thunk->name] != 1 || globalUses[get->name] != 1) {
        continue;
      }
      auto* global = module->getGlobal(get->name);
      if (global->imported() || !isLazyInit(global->init) ||
          !Type::isSubType(global->init->type, resultType)) {
        continue;
      }
      auto* init = global->init->cast<StructNew>();
      bool found = false;
      for (auto& [templ, indexes] : templates) {
        if (ExpressionAnalyzer::equal(templ, init)) {
          indexes.push_back(i);
          found = true;
          break;
        }
      }
      if (!found) {
        templates.push_back({init, {i}});
      }
    }
    std::vector<Index>* lazyIndexes = nullptr;
    for (auto& [templ, indexes] : templates) {
      if (!lazyIndexes || indexes.size() > lazyIndexes->size()) {
        lazyIndexes = &indexes;
        decision.lazyTemplate = templ;
      }
    }
    // Lazy creation only pays off with at least a few sites; a single site is
    // better served by DeferGlobalAllocations.
    decision.lazy = lazyIndexes && lazyIndexes->size() >= 2;
    if (decision.lazy) {
      for (auto i : *lazyIndexes) {
        decision.sites[i].mode = SiteMode::Lazy;
      }
      // Eager nulls would be confused with the empty slot.
      for (auto& site : decision.sites) {
        if (site.mode == SiteMode::Eager &&
            getThunkConstant(site.thunk)->is<RefNull>()) {
          site.mode = SiteMode::Exception;
        }
      }
    } else {
      decision.lazyTemplate = nullptr;
    }

    // Exceptions need a lazy helper and an identity key.
    Index exceptions = 0, hoists = 0;
    for (auto& site : decision.sites) {
      if (site.mode == SiteMode::Exception) {
        exceptions++;
        if (!site.globalSite->isImmutableGlobalInit) {
          hoists++;
        }
      }
    }
    if (exceptions > 0 && (!decision.lazy || exceptions > MaxExceptions)) {
      return {};
    }

    // The size model.
    int64_t benefit = 0, cost = 0;
    std::unordered_map<Name, Index> deadThunkUses;
    std::unordered_set<Name> eagerGlobals;
    for (auto& site : decision.sites) {
      Index refFuncBytes =
        getOperandBytes(site.globalSite->structNew->operands[k]);
      switch (site.mode) {
        case SiteMode::Lazy: {
          auto* get = site.thunk->body->cast<GlobalGet>();
          auto* global = module->getGlobal(get->name);
          benefit += getGlobalEntryBytes(global->init->cast<StructNew>());
          benefit += refFuncBytes;
          cost += 2;
          deadThunkUses[site.thunk->name]++;
          break;
        }
        case SiteMode::Eager: {
          auto* constant = getThunkConstant(site.thunk);
          benefit += refFuncBytes;
          cost += getOperandBytes(constant);
          deadThunkUses[site.thunk->name]++;
          if (auto* get = constant->dynCast<GlobalGet>()) {
            eagerGlobals.insert(get->name);
          }
          break;
        }
        case SiteMode::Exception:
          benefit += refFuncBytes;
          cost += 2 + ExceptionBytes;
          if (!site.globalSite->isImmutableGlobalInit) {
            cost += HoistBytes;
          }
          break;
      }
    }
    for (auto& [name, uses] : deadThunkUses) {
      if (funcUses[name] == uses) {
        auto* func = module->getFunction(name);
        benefit += FunctionOverheadBytes + getOperandBytes(func->body);
      }
    }
    for (auto name : eagerGlobals) {
      if (isDeferrable(module->getGlobal(name))) {
        cost += DeferredGlobalBytes;
      }
    }
    if (decision.lazy) {
      // call_ref + struct.get becomes a call.
      benefit += reads * 4;
      cost += HelperBytes;
    } else {
      // The call_ref goes away.
      benefit += reads * 3;
    }
    if (!always && benefit <= cost) {
      return {};
    }

    if (decision.lazy) {
      decision.fieldType = Type(resultType.getHeapType(), Nullable);
    } else {
      decision.fieldType = resultType;
    }
    return decision;
  }

  // Indexed mode: every thunk calls the same function with the same operand
  // shape. The field then holds the constant.
  std::optional<Decision> analyzeIndexed(Decision decision, Index reads) {
    auto k = decision.index;
    std::unordered_map<Name, Index> deadThunkUses;
    int64_t benefit = 0, cost = 0;
    bool first = true;
    for (auto& site : decision.sites) {
      auto info = getIndexedThunk(site.thunk);
      if (!info) {
        return {};
      }
      if (first) {
        decision.callee = info->callee;
        decision.shape = info->shape;
        decision.fieldType = info->constant->type;
        first = false;
      } else if (info->callee != decision.callee ||
                 info->shape != decision.shape ||
                 info->constant->type != decision.fieldType) {
        return {};
      }
      site.constant = info->constant;
      deadThunkUses[site.thunk->name]++;
      benefit += getOperandBytes(site.globalSite->structNew->operands[k]);
      cost += getOperandBytes(site.constant);
    }
    decision.indexed = true;
    for (auto& [name, uses] : deadThunkUses) {
      if (funcUses[name] == uses) {
        auto* func = module->getFunction(name);
        auto* call = func->body->cast<Call>();
        benefit += FunctionOverheadBytes + 1 +
                   getLEBSize(U32LEB(functionIndexes[call->target]));
        for (auto* operand : call->operands) {
          benefit += operand->is<Const>() ? getOperandBytes(operand) : 2;
        }
      }
    }
    // call_ref with a type index becomes a call with a function index.
    cost += reads;
    if (!always && benefit <= cost) {
      return {};
    }
    return decision;
  }

  void apply(std::vector<Decision>& decisions) {
    Builder builder(*module);
    bool addedGlobals = false;
    bool needsReorder = false;

    // Update the operands, and build the helpers.
    std::unordered_map<FieldKey, Decision*, FieldKeyHash> decisionMap;
    SubTypes subTypes(*module);
    for (auto& decision : decisions) {
      for (auto sub : subTypes.getSubTypes(decision.type)) {
        decisionMap[FieldKey{sub, decision.index}] = &decision;
      }
      auto k = decision.index;
      std::vector<std::pair<Name, Name>> exceptions;
      for (auto& site : decision.sites) {
        auto& operand = site.globalSite->structNew->operands[k];
        if (decision.indexed) {
          operand = ExpressionManipulator::copy(site.constant, *module);
          continue;
        }
        switch (site.mode) {
          case SiteMode::Lazy:
          case SiteMode::Exception:
            operand = builder.makeRefNull(
              decision.resultType.getHeapType().getBottom());
            break;
          case SiteMode::Eager: {
            auto* constant = getThunkConstant(site.thunk);
            operand = ExpressionManipulator::copy(constant, *module);
            if (constant->is<GlobalGet>()) {
              // The global may be defined after the one we put it in.
              needsReorder = true;
            }
            break;
          }
        }
        if (site.mode == SiteMode::Exception) {
          exceptions.push_back(
            {getKeyGlobal(*site.globalSite, addedGlobals), site.thunk->name});
        }
      }
      if (decision.lazy) {
        decision.helper = makeHelper(decision, exceptions);
      }
    }

    // Replace the reads, in parallel.
    struct Replacer : public WalkerPass<PostWalker<Replacer>> {
      bool isFunctionParallel() override { return true; }
      std::unique_ptr<Pass> create() override {
        return std::make_unique<Replacer>(decisionMap);
      }
      const std::unordered_map<FieldKey, Decision*, FieldKeyHash>& decisionMap;
      Replacer(const std::unordered_map<FieldKey, Decision*, FieldKeyHash>&
                 decisionMap)
        : decisionMap(decisionMap) {}

      void visitCallRef(CallRef* curr) {
        auto* get = curr->target->dynCast<StructGet>();
        if (!get || get->type == Type::unreachable) {
          return;
        }
        auto type = get->ref->type;
        auto iter = decisionMap.find(FieldKey{type.getHeapType(), get->index});
        if (iter == decisionMap.end()) {
          return;
        }
        auto& decision = *iter->second;
        Builder builder(*getModule());
        Expression* replacement;
        if (decision.indexed) {
          replacement = makeIndexedCall(curr, get, decision);
        } else if (decision.lazy) {
          replacement = builder.makeCall(
            decision.helper, {get->ref}, decision.resultType, curr->isReturn);
        } else {
          get->type = decision.resultType;
          replacement = get;
          if (curr->isReturn) {
            replacement = builder.makeReturn(get);
          }
        }
        replaceCurrent(replacement);
      }

      // (call_ref (struct.get k (X)) (args)) => (call F (args with the field
      // read in the constant's place)). The receiver is evaluated first in the
      // original; if moving things around could change the behavior, spill
      // the receiver and the arguments to locals in the original order.
      Expression*
      makeIndexedCall(CallRef* curr, StructGet* get, Decision& decision) {
        Builder builder(*getModule());
        auto& options = getPassOptions();
        auto& wasm = *getModule();
        get->type = decision.fieldType;
        std::vector<Expression*> prelude;
        auto spill = [&](Expression*& value) {
          if (value->type == Type::unreachable) {
            // Keep it where it is; what follows is unreachable anyway.
            prelude.push_back(value);
            value = builder.makeUnreachable();
            return;
          }
          auto index = Builder::addVar(getFunction(), value->type);
          prelude.push_back(builder.makeLocalSet(index, value));
          value = builder.makeLocalGet(index, value->type);
        };
        // The arguments before the constant in the shape are now evaluated
        // before the field read, and the arguments the thunk ignores vanish.
        bool needsSpill = false;
        std::vector<bool> used(curr->operands.size(), false);
        bool beforeConst = true;
        for (auto index : decision.shape) {
          if (index == ConstOperand) {
            beforeConst = false;
            continue;
          }
          assert(index < curr->operands.size());
          used[index] = true;
          if (beforeConst && !EffectAnalyzer::canReorder(
                               options, wasm, get, curr->operands[index])) {
            needsSpill = true;
          }
        }
        for (Index i = 0; i < curr->operands.size(); i++) {
          if (!used[i] && EffectAnalyzer(options, wasm, curr->operands[i])
                            .hasSideEffects()) {
            needsSpill = true;
          }
        }
        if (needsSpill) {
          spill(get->ref);
          for (auto& operand : curr->operands) {
            spill(operand);
          }
        }
        // Arguments the thunk ignores vanish: they are side effect free, or
        // have been spilled.
        std::vector<Expression*> operands;
        for (auto index : decision.shape) {
          if (index == ConstOperand) {
            operands.push_back(get);
          } else {
            assert(index < curr->operands.size());
            operands.push_back(curr->operands[index]);
          }
        }
        auto* callee = wasm.getFunction(decision.callee);
        Expression* call = builder.makeCall(
          decision.callee, operands, callee->getResults(), curr->isReturn);
        if (prelude.empty()) {
          return call;
        }
        prelude.push_back(call);
        return builder.makeBlock(prelude, call->type);
      }
    };
    Replacer(decisionMap).run(getPassRunner(), module);

    // Update the types.
    class TypeRewriter : public GlobalTypeRewriter {
      const std::unordered_map<FieldKey, Decision*, FieldKeyHash>& decisionMap;

    public:
      TypeRewriter(Module& wasm,
                   WorldMode worldMode,
                   const std::unordered_map<FieldKey, Decision*, FieldKeyHash>&
                     decisionMap)
        : GlobalTypeRewriter(wasm, worldMode), decisionMap(decisionMap) {}

      void modifyStruct(HeapType oldType, Struct& struct_) override {
        for (Index i = 0; i < struct_.fields.size(); i++) {
          auto iter = decisionMap.find(FieldKey{oldType, i});
          if (iter == decisionMap.end()) {
            continue;
          }
          auto& decision = *iter->second;
          struct_.fields[i] = Field(getTempType(decision.fieldType),
                                    decision.lazy ? Mutable : Immutable);
        }
      }
    };
    TypeRewriter(*module, getPassOptions().worldMode, decisionMap).update();
    ReFinalize().run(getPassRunner(), module);

    if (addedGlobals || needsReorder) {
      PassRunner runner(module);
      runner.add("reorder-globals-always");
      runner.setIsNested(true);
      runner.run();
    }
  }

  // Returns the name of an immutable global whose value is the object created
  // by the struct.new at the given site, hoisting it into a new global if
  // needed.
  Name getKeyGlobal(GlobalSite& site, bool& addedGlobals) {
    if (site.isImmutableGlobalInit) {
      return site.global;
    }
    if (auto* get = (*site.location)->dynCast<GlobalGet>()) {
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

  Name makeHelper(Decision& decision,
                  std::vector<std::pair<Name, Name>>& exceptions) {
    Builder builder(*module);
    auto type = decision.type;
    auto k = decision.index;
    auto resultType = decision.resultType;
    auto paramType = Type(type, Nullable);
    std::string typeName;
    if (auto iter = module->typeNames.find(type);
        iter != module->typeNames.end()) {
      typeName = iter->second.name.toString();
    } else {
      typeName = std::to_string(functionIndexes.size());
    }
    auto name = Names::getValidFunctionName(
      *module, "lazy$" + typeName + "$" + std::to_string(k));

    Index vtLocal = 0;
    Index objLocal = 1;
    Name label("out");
    std::vector<Expression*> list;
    // The types here reflect the old field type; ReFinalize fixes them up
    // after the types are rewritten.
    list.push_back(builder.makeBrOn(
      BrOnNonNull,
      label,
      builder.makeStructGet(k,
                            builder.makeLocalGet(vtLocal, paramType),
                            MemoryOrder::Unordered,
                            decision.fieldType)));
    for (auto& [keyGlobal, thunk] : exceptions) {
      auto* keyGet =
        builder.makeGlobalGet(keyGlobal, module->getGlobal(keyGlobal)->type);
      auto* call =
        builder.makeCall(thunk, {}, module->getFunction(thunk)->getResults());
      list.push_back(builder.makeIf(
        builder.makeRefEq(builder.makeLocalGet(vtLocal, paramType), keyGet),
        builder.makeReturn(call)));
    }
    auto* object = ExpressionManipulator::copy(decision.lazyTemplate, *module);
    auto objType = object->type;
    list.push_back(
      builder.makeStructSet(k,
                            builder.makeLocalGet(vtLocal, paramType),
                            builder.makeLocalTee(objLocal, object, objType),
                            MemoryOrder::Unordered));
    list.push_back(builder.makeLocalGet(objLocal, objType));
    auto* body = builder.makeBlock(label, list, resultType);
    module->addFunction(builder.makeFunction(
      name, Signature(paramType, resultType), {objType}, body));
    return name;
  }
};

} // anonymous namespace

Pass* createThunkFieldsToDataPass() { return new ThunkFieldsToData(); }

} // namespace wasm
