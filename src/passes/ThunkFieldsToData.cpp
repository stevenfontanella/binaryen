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
  // Maximum distinct call templates in a multi-callee indexed field.
  static constexpr Index MaxIndexedTemplates = 4;

  static constexpr Index ConstOperand = Index(-1);
  static constexpr Index FixedOperand = Index(-2);

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
    // Reads that are directly the target of a call_ref (possibly through
    // if/select arms).
    std::unordered_map<FieldKey, Index, FieldKeyHash> callReads;
    // Groups of fields read together in the arms of an if/select call_ref
    // target.
    std::vector<std::vector<FieldKey>> coupledCallReads;
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
    // Indexed mode: the constant value the field holds for this site.
    Literal constant = {};
  };

  struct IndexedTemplate {
    Name callee;
    std::vector<Index> shape;
    std::vector<Expression*> fixedOperands;
    Type constType;

    bool equals(const IndexedTemplate& other) const {
      if (callee != other.callee || constType != other.constType ||
          shape != other.shape) {
        return false;
      }
      for (Index i = 0; i < shape.size(); i++) {
        if (shape[i] == FixedOperand &&
            !ExpressionAnalyzer::equal(fixedOperands[i],
                                       other.fixedOperands[i])) {
          return false;
        }
      }
      return true;
    }
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
    // varying constant, FixedOperand for a fixed constant, otherwise the index
    // of the thunk parameter), or multiple templates dispatched by a helper.
    bool indexed = false;
    bool useComplement = false;
    Name callee;
    std::vector<Index> shape;
    std::vector<Expression*> fixedOperands;
    std::vector<IndexedTemplate> templates;
  };

  static bool collectCallRefTargetGets(Expression* target,
                                       std::vector<StructGet*>& gets) {
    if (auto* get = target->dynCast<StructGet>()) {
      if (get->type == Type::unreachable ||
          get->order != MemoryOrder::Unordered) {
        return false;
      }
      auto type = get->ref->type;
      if (!type.isRef() || !type.getHeapType().isStruct()) {
        return false;
      }
      gets.push_back(get);
      return true;
    }
    if (auto* iff = target->dynCast<If>()) {
      if (!iff->ifFalse || iff->type == Type::unreachable) {
        return false;
      }
      return collectCallRefTargetGets(iff->ifTrue, gets) &&
             collectCallRefTargetGets(iff->ifFalse, gets);
    }
    if (auto* sel = target->dynCast<Select>()) {
      if (sel->type == Type::unreachable) {
        return false;
      }
      return collectCallRefTargetGets(sel->ifTrue, gets) &&
             collectCallRefTargetGets(sel->ifFalse, gets);
    }
    return false;
  }

  static void updateTargetTypes(Expression* target, Type fieldType) {
    target->type = fieldType;
    if (auto* iff = target->dynCast<If>()) {
      updateTargetTypes(iff->ifTrue, fieldType);
      updateTargetTypes(iff->ifFalse, fieldType);
    } else if (auto* sel = target->dynCast<Select>()) {
      updateTargetTypes(sel->ifTrue, fieldType);
      updateTargetTypes(sel->ifFalse, fieldType);
    }
  }

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
          // Reads that are directly the target of a call_ref (possibly through
          // if/select arms).
          void visitCallRef(CallRef* curr) {
            std::vector<StructGet*> gets;
            if (!collectCallRefTargetGets(curr->target, gets)) {
              return;
            }
            for (auto* get : gets) {
              auto type = get->ref->type;
              info.callReads[FieldKey{type.getHeapType(), get->index}]++;
            }
            if (gets.size() > 1) {
              std::vector<FieldKey> group;
              group.reserve(gets.size());
              for (auto* get : gets) {
                group.push_back(
                  FieldKey{get->ref->type.getHeapType(), get->index});
              }
              info.coupledCallReads.push_back(std::move(group));
            }
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
      for (auto& group : info.coupledCallReads) {
        codeInfo.coupledCallReads.push_back(group);
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
    if (!codeInfo.coupledCallReads.empty() && !decisions.empty()) {
      bool changed = true;
      while (changed) {
        changed = false;
        std::unordered_map<FieldKey, Index, FieldKeyHash> keyToDecision;
        for (Index i = 0; i < decisions.size(); i++) {
          for (auto sub : subTypes.getSubTypes(decisions[i].type)) {
            keyToDecision[FieldKey{sub, decisions[i].index}] = i;
          }
        }
        std::vector<bool> drop(decisions.size(), false);
        for (auto& group : codeInfo.coupledCallReads) {
          std::optional<Index> firstIdx;
          bool valid = true;
          for (auto& key : group) {
            auto iter = keyToDecision.find(key);
            if (iter == keyToDecision.end()) {
              valid = false;
              break;
            }
            if (!firstIdx) {
              firstIdx = iter->second;
              if (!areCoupledDecisionsCompatible(decisions[*firstIdx],
                                                 decisions[*firstIdx])) {
                valid = false;
                break;
              }
            } else if (!areCoupledDecisionsCompatible(decisions[*firstIdx],
                                                      decisions[iter->second])) {
              valid = false;
              break;
            }
          }
          if (!valid) {
            for (auto& key : group) {
              auto iter = keyToDecision.find(key);
              if (iter != keyToDecision.end() && !drop[iter->second]) {
                drop[iter->second] = true;
                changed = true;
              }
            }
          }
        }
        if (changed) {
          std::vector<Decision> kept;
          for (Index i = 0; i < decisions.size(); i++) {
            if (!drop[i]) {
              kept.push_back(std::move(decisions[i]));
            }
          }
          decisions = std::move(kept);
        }
      }
    }
    if (decisions.empty()) {
      return;
    }

    apply(decisions);
  }

private:
  bool areCoupledDecisionsCompatible(const Decision& d1, const Decision& d2) {
    if (d1.lazy || d2.lazy) {
      return false;
    }
    if (d1.indexed != d2.indexed || d1.fieldType != d2.fieldType ||
        d1.resultType != d2.resultType) {
      return false;
    }
    if (!d1.indexed) {
      return true;
    }
    if (d1.useComplement != d2.useComplement ||
        d1.templates.size() != d2.templates.size()) {
      return false;
    }
    for (Index i = 0; i < d1.templates.size(); i++) {
      if (!d1.templates[i].equals(d2.templates[i])) {
        return false;
      }
    }
    return true;
  }
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

  Index getConstBytes(Literal value) {
    auto type = value.type;
    if (type == Type::i32) {
      return 1 + getLEBSize(S32LEB(value.geti32()));
    } else if (type == Type::i64) {
      return 1 + getLEBSize(S64LEB(value.geti64()));
    } else if (type == Type::f32) {
      return 5;
    } else if (type == Type::f64) {
      return 9;
    }
    return 17;
  }

  // The estimated number of bytes for a constant expression.
  Index getOperandBytes(Expression* curr) {
    if (auto* c = curr->dynCast<Const>()) {
      return getConstBytes(c->value);
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

  // Indexed mode: every thunk calls a function with the thunk's parameters
  // (each used at most once, in any order), optional fixed constants, and
  // exactly one varying numeric constant. Up to MaxIndexedTemplates distinct
  // templates are allowed per field.
  std::optional<Decision> analyzeIndexed(Decision decision, Index reads) {
    struct LayoutKey {
      Name callee;
      std::vector<Index> localPositions;
      bool operator==(const LayoutKey& other) const {
        return callee == other.callee && localPositions == other.localPositions;
      }
    };
    struct LayoutKeyHash {
      size_t operator()(const LayoutKey& key) const {
        size_t h = std::hash<Name>{}(key.callee);
        for (auto pos : key.localPositions) {
          h ^= (size_t(pos) + 0x9e3779b9 + (h << 6) + (h >> 2));
        }
        return h;
      }
    };

    std::unordered_map<LayoutKey, std::vector<Index>, LayoutKeyHash>
      layoutGroups;
    for (Index s = 0; s < decision.sites.size(); s++) {
      auto* func = decision.sites[s].thunk;
      if (func->imported()) {
        return {};
      }
      auto* call = func->body->dynCast<Call>();
      if (!call || call->target == func->name) {
        return {};
      }
      Index numParams = func->getNumParams();
      std::vector<bool> usedParam(numParams, false);
      LayoutKey key;
      key.callee = call->target;
      key.localPositions.reserve(call->operands.size());
      bool hasNumericConst = false;
      for (auto* operand : call->operands) {
        if (auto* get = operand->dynCast<LocalGet>()) {
          if (get->index >= numParams || usedParam[get->index]) {
            return {};
          }
          usedParam[get->index] = true;
          key.localPositions.push_back(get->index);
        } else if (isValidConstant(operand)) {
          key.localPositions.push_back(Index(-1));
          if (auto* c = operand->dynCast<Const>()) {
            if (c->type.isNumber()) {
              hasNumericConst = true;
            }
          }
        } else {
          return {};
        }
      }
      if (!hasNumericConst) {
        return {};
      }
      layoutGroups[std::move(key)].push_back(s);
    }

    std::vector<IndexedTemplate> siteTemplates(decision.sites.size());
    for (auto& [layout, siteIdxs] : layoutGroups) {
      Index numOps = layout.localPositions.size();
      std::vector<Index> candidateConstPositions;
      for (Index p = 0; p < numOps; p++) {
        if (layout.localPositions[p] != Index(-1)) {
          continue;
        }
        Type commonType = Type::none;
        bool allNumericConst = true;
        for (auto s : siteIdxs) {
          auto* call = decision.sites[s].thunk->body->cast<Call>();
          auto* c = call->operands[p]->dynCast<Const>();
          if (!c || !c->type.isNumber()) {
            allNumericConst = false;
            break;
          }
          if (commonType == Type::none) {
            commonType = c->type;
          } else if (c->type != commonType) {
            allNumericConst = false;
            break;
          }
        }
        if (allNumericConst) {
          candidateConstPositions.push_back(p);
        }
      }
      if (candidateConstPositions.empty()) {
        return {};
      }

      Index bestConstPos = candidateConstPositions[0];
      if (candidateConstPositions.size() > 1) {
        Index bestTemplateCount = Index(-1);
        for (auto candPos : candidateConstPositions) {
          std::vector<Index> uniqueReps;
          for (auto s : siteIdxs) {
            auto* callS = decision.sites[s].thunk->body->cast<Call>();
            bool found = false;
            for (auto rep : uniqueReps) {
              auto* callRep = decision.sites[rep].thunk->body->cast<Call>();
              bool sameFixed = true;
              for (Index q = 0; q < numOps; q++) {
                if (q == candPos || layout.localPositions[q] != Index(-1)) {
                  continue;
                }
                if (!ExpressionAnalyzer::equal(callS->operands[q],
                                               callRep->operands[q])) {
                  sameFixed = false;
                  break;
                }
              }
              if (sameFixed) {
                found = true;
                break;
              }
            }
            if (!found) {
              uniqueReps.push_back(s);
              if (uniqueReps.size() >= bestTemplateCount) {
                break;
              }
            }
          }
          if (uniqueReps.size() < bestTemplateCount) {
            bestTemplateCount = uniqueReps.size();
            bestConstPos = candPos;
          }
        }
      }

      for (auto s : siteIdxs) {
        auto* call = decision.sites[s].thunk->body->cast<Call>();
        auto* c = call->operands[bestConstPos]->cast<Const>();
        decision.sites[s].constant = c->value;
        IndexedTemplate templ;
        templ.callee = layout.callee;
        templ.constType = c->type;
        templ.shape.resize(numOps);
        templ.fixedOperands.assign(numOps, nullptr);
        for (Index q = 0; q < numOps; q++) {
          if (q == bestConstPos) {
            templ.shape[q] = ConstOperand;
          } else if (layout.localPositions[q] != Index(-1)) {
            templ.shape[q] = layout.localPositions[q];
          } else {
            templ.shape[q] = FixedOperand;
            templ.fixedOperands[q] = call->operands[q];
          }
        }
        siteTemplates[s] = std::move(templ);
      }
    }

    std::vector<IndexedTemplate> templates;
    for (auto& templ : siteTemplates) {
      bool found = false;
      for (auto& existing : templates) {
        if (existing.equals(templ)) {
          found = true;
          break;
        }
      }
      if (!found) {
        templates.push_back(templ);
        if (templates.size() > MaxIndexedTemplates) {
          return {};
        }
      }
    }
    if (templates.empty()) {
      return {};
    }

    if (templates.size() > 1) {
      for (auto& templ : templates) {
        if (templ.constType != Type::i32) {
          return {};
        }
      }
      std::sort(templates.begin(),
                templates.end(),
                [&](const IndexedTemplate& a, const IndexedTemplate& b) {
                  if (a.callee != b.callee) {
                    return functionIndexes[a.callee] <
                           functionIndexes[b.callee];
                  }
                  return a.shape < b.shape;
                });
      bool allNonNegative = true;
      for (auto& site : decision.sites) {
        if (site.constant.geti32() < 0) {
          allNonNegative = false;
          break;
        }
      }
      decision.useComplement = (templates.size() == 2 && allNonNegative);
      Index shift = templates.size() == 2 ? 1 : 2;
      for (Index s = 0; s < decision.sites.size(); s++) {
        Index tag = 0;
        for (Index t = 0; t < templates.size(); t++) {
          if (templates[t].equals(siteTemplates[s])) {
            tag = t;
            break;
          }
        }
        int32_t kVal = decision.sites[s].constant.geti32();
        if (decision.useComplement) {
          decision.sites[s].constant = Literal(tag == 0 ? kVal : ~kVal);
        } else {
          int32_t shifted = int32_t(uint32_t(kVal) << shift);
          if ((shifted >> shift) != kVal) {
            return {};
          }
          decision.sites[s].constant =
            Literal(int32_t(uint32_t(shifted) | uint32_t(tag)));
        }
      }
    }

    decision.indexed = true;
    decision.fieldType = templates[0].constType;
    decision.templates = std::move(templates);
    if (decision.templates.size() == 1) {
      decision.callee = decision.templates[0].callee;
      decision.shape = decision.templates[0].shape;
      decision.fixedOperands = decision.templates[0].fixedOperands;
    }

    auto k = decision.index;
    std::unordered_map<Name, Index> deadThunkUses;
    int64_t benefit = 0, cost = 0;
    for (auto& site : decision.sites) {
      deadThunkUses[site.thunk->name]++;
      benefit += getOperandBytes(site.globalSite->structNew->operands[k]);
      cost += getConstBytes(site.constant);
    }
    for (auto& [name, uses] : deadThunkUses) {
      if (funcUses[name] == uses) {
        auto* func = module->getFunction(name);
        auto* call = func->body->cast<Call>();
        benefit += FunctionOverheadBytes + 1 +
                   getLEBSize(U32LEB(functionIndexes[call->target]));
        for (auto* operand : call->operands) {
          benefit += operand->is<LocalGet>() ? 2 : getOperandBytes(operand);
        }
      }
    }
    // call_ref with a type index becomes a call with a function index.
    cost += reads;
    if (decision.templates.size() == 1) {
      for (Index i = 0; i < decision.shape.size(); i++) {
        if (decision.shape[i] == FixedOperand) {
          cost += reads * getOperandBytes(decision.fixedOperands[i]);
        }
      }
    } else {
      cost += FunctionOverheadBytes;
      if (decision.templates.size() > 2) {
        cost += 14;
      }
      for (Index t = 0; t < decision.templates.size(); t++) {
        auto& templ = decision.templates[t];
        cost += 1 + getLEBSize(U32LEB(functionIndexes[templ.callee]));
        for (Index i = 0; i < templ.shape.size(); i++) {
          if (templ.shape[i] == FixedOperand) {
            cost += getOperandBytes(templ.fixedOperands[i]);
          } else {
            cost += 2;
          }
        }
        if (t > 0) {
          cost += 8;
        }
      }
    }
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
    struct SharedIndexedHelper {
      Signature sig;
      bool useComplement;
      std::vector<IndexedTemplate> templates;
      Name helper;
    };
    std::vector<SharedIndexedHelper> sharedIndexedHelpers;

    for (auto& decision : decisions) {
      for (auto sub : subTypes.getSubTypes(decision.type)) {
        decisionMap[FieldKey{sub, decision.index}] = &decision;
      }
      auto k = decision.index;
      std::vector<std::pair<Name, Name>> exceptions;
      for (auto& site : decision.sites) {
        auto& operand = site.globalSite->structNew->operands[k];
        if (decision.indexed) {
          operand = builder.makeConst(site.constant);
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
      if (decision.indexed && decision.templates.size() > 1) {
        auto sig =
          decision.type.getStruct().fields[k].type.getHeapType().getSignature();
        Name helper;
        for (auto& shared : sharedIndexedHelpers) {
          if (shared.sig == sig &&
              shared.useComplement == decision.useComplement &&
              shared.templates.size() == decision.templates.size()) {
            bool same = true;
            for (Index i = 0; i < shared.templates.size(); i++) {
              if (!shared.templates[i].equals(decision.templates[i])) {
                same = false;
                break;
              }
            }
            if (same) {
              helper = shared.helper;
              break;
            }
          }
        }
        if (!helper.is()) {
          helper = makeIndexedHelper(decision, sig);
          sharedIndexedHelpers.push_back(
            {sig, decision.useComplement, decision.templates, helper});
        }
        decision.callee = helper;
        decision.shape.clear();
        decision.shape.push_back(ConstOperand);
        Index numParams = sig.params.size();
        for (Index p = 0; p < numParams; p++) {
          decision.shape.push_back(p);
        }
        decision.fixedOperands.assign(decision.shape.size(), nullptr);
      } else if (decision.lazy) {
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
        std::vector<StructGet*> gets;
        if (!collectCallRefTargetGets(curr->target, gets)) {
          return;
        }
        auto type = gets[0]->ref->type;
        auto iter =
          decisionMap.find(FieldKey{type.getHeapType(), gets[0]->index});
        if (iter == decisionMap.end()) {
          return;
        }
        auto& decision = *iter->second;
        Builder builder(*getModule());
        Expression* replacement;
        if (decision.indexed) {
          replacement = makeIndexedCall(curr, decision);
        } else if (decision.lazy) {
          auto* get = curr->target->cast<StructGet>();
          replacement = builder.makeCall(
            decision.helper, {get->ref}, decision.resultType, curr->isReturn);
        } else {
          updateTargetTypes(curr->target, decision.resultType);
          replacement = curr->target;
          if (curr->isReturn) {
            replacement = builder.makeReturn(replacement);
          }
        }
        replaceCurrent(replacement);
      }

      // (call_ref (struct.get k (X)) (args)) => (call F (args with the field
      // read in the constant's place)). The receiver is evaluated first in the
      // original; if moving things around could change the behavior, spill
      // the receiver and the arguments to locals in the original order.
      Expression* makeIndexedCall(CallRef* curr, Decision& decision) {
        Builder builder(*getModule());
        auto& options = getPassOptions();
        auto& wasm = *getModule();
        updateTargetTypes(curr->target, decision.fieldType);
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
        std::vector<Index> usedOrder;
        for (auto index : decision.shape) {
          if (index == ConstOperand) {
            beforeConst = false;
            continue;
          }
          if (index == FixedOperand) {
            continue;
          }
          assert(index < curr->operands.size());
          used[index] = true;
          if (beforeConst &&
              !EffectAnalyzer::canReorder(
                options, wasm, curr->target, curr->operands[index])) {
            needsSpill = true;
          }
          for (auto prevIndex : usedOrder) {
            if (prevIndex > index &&
                !EffectAnalyzer::canReorder(
                  options, wasm, curr->operands[index], curr->operands[prevIndex])) {
              needsSpill = true;
            }
          }
          usedOrder.push_back(index);
        }
        for (Index i = 0; i < curr->operands.size(); i++) {
          if (!used[i] && EffectAnalyzer(options, wasm, curr->operands[i])
                            .hasSideEffects()) {
            needsSpill = true;
          }
        }
        if (needsSpill) {
          if (auto* get = curr->target->dynCast<StructGet>()) {
            spill(get->ref);
          } else {
            spill(curr->target);
          }
          for (auto& operand : curr->operands) {
            spill(operand);
          }
        }
        // Arguments the thunk ignores vanish: they are side effect free, or
        // have been spilled.
        std::vector<Expression*> operands;
        for (Index i = 0; i < decision.shape.size(); i++) {
          auto index = decision.shape[i];
          if (index == ConstOperand) {
            operands.push_back(curr->target);
          } else if (index == FixedOperand) {
            operands.push_back(
              ExpressionManipulator::copy(decision.fixedOperands[i], wasm));
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

  Name makeIndexedHelper(Decision& decision, Signature thunkSig) {
    Builder builder(*module);
    auto type = decision.type;
    auto k = decision.index;
    auto resultType = decision.resultType;
    std::string typeName;
    if (auto iter = module->typeNames.find(type);
        iter != module->typeNames.end()) {
      typeName = iter->second.name.toString();
    } else {
      typeName = std::to_string(functionIndexes.size());
    }
    auto name = Names::getValidFunctionName(
      *module, "indexed$" + typeName + "$" + std::to_string(k));

    std::vector<Type> paramTypes;
    paramTypes.push_back(Type::i32);
    for (auto paramType : thunkSig.params) {
      paramTypes.push_back(paramType);
    }

    Index numTemplates = decision.templates.size();
    auto makeKeyExpr = [&](Index t) -> Expression* {
      if (decision.useComplement) {
        if (t == 0) {
          return builder.makeLocalGet(0, Type::i32);
        }
        return builder.makeBinary(XorInt32,
                                  builder.makeLocalGet(0, Type::i32),
                                  builder.makeConst(Literal(int32_t(-1))));
      }
      if (numTemplates == 2) {
        return builder.makeBinary(ShrSInt32,
                                  builder.makeLocalGet(0, Type::i32),
                                  builder.makeConst(Literal(int32_t(1))));
      }
      return builder.makeLocalGet(0, Type::i32);
    };

    auto makeTemplateCall = [&](Index t) -> Expression* {
      auto& templ = decision.templates[t];
      std::vector<Expression*> operands;
      operands.reserve(templ.shape.size());
      for (Index i = 0; i < templ.shape.size(); i++) {
        auto idx = templ.shape[i];
        if (idx == ConstOperand) {
          operands.push_back(makeKeyExpr(t));
        } else if (idx == FixedOperand) {
          operands.push_back(
            ExpressionManipulator::copy(templ.fixedOperands[i], *module));
        } else {
          operands.push_back(builder.makeLocalGet(1 + idx, paramTypes[1 + idx]));
        }
      }
      auto* callee = module->getFunction(templ.callee);
      return builder.makeCall(templ.callee, operands, callee->getResults());
    };

    std::vector<Type> vars;
    Expression* body = nullptr;
    if (numTemplates == 2) {
      Expression* cond;
      if (decision.useComplement) {
        cond = builder.makeBinary(LtSInt32,
                                  builder.makeLocalGet(0, Type::i32),
                                  builder.makeConst(Literal(int32_t(0))));
      } else {
        cond = builder.makeBinary(AndInt32,
                                  builder.makeLocalGet(0, Type::i32),
                                  builder.makeConst(Literal(int32_t(1))));
      }
      body = builder.makeIf(
        cond, makeTemplateCall(1), makeTemplateCall(0), resultType);
    } else {
      Index tagLocal = paramTypes.size();
      vars.push_back(Type::i32);
      std::vector<Expression*> list;
      list.push_back(builder.makeLocalSet(
        tagLocal,
        builder.makeBinary(AndInt32,
                           builder.makeLocalGet(0, Type::i32),
                           builder.makeConst(Literal(int32_t(3))))));
      list.push_back(builder.makeLocalSet(
        0,
        builder.makeBinary(ShrSInt32,
                           builder.makeLocalGet(0, Type::i32),
                           builder.makeConst(Literal(int32_t(2))))));
      Expression* dispatch = makeTemplateCall(0);
      for (Index t = 1; t < numTemplates; t++) {
        Expression* cond = builder.makeLocalGet(tagLocal, Type::i32);
        if (t > 1) {
          cond = builder.makeBinary(
            EqInt32, cond, builder.makeConst(Literal(int32_t(t))));
        }
        dispatch =
          builder.makeIf(cond, makeTemplateCall(t), dispatch, resultType);
      }
      list.push_back(dispatch);
      body = builder.makeBlock(list, resultType);
    }

    module->addFunction(builder.makeFunction(
      name, Signature(Type(paramTypes), resultType), std::move(vars), body));
    return name;
  }
};

} // anonymous namespace

Pass* createThunkFieldsToDataPass() { return new ThunkFieldsToData(); }

} // namespace wasm
