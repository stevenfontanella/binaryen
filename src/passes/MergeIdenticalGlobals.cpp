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
// Merges private immutable globals whose initializers allocate structurally
// identical constant objects, and hoists repeated identical allocations that
// are nested inside global initializers into shared globals.
//
// Compilers for class-based languages emit many constant objects in global
// initializers (vtables, itables, enum values, empty collections, ...), and
// large numbers of them are byte-identical: e.g. the vtables of all classes
// that only inherit Object's methods, or the empty itable nested inside every
// vtable. Sharing one object between them removes the duplicate initializers
// and the global entries.
//
// Sharing is only valid when nothing can observe that there are fewer
// objects, so (in a closed world) we require that:
//
//  * no field of any struct allocated in the initializer, and no array
//    allocated in it, is ever written (through the type itself or any
//    supertype); and
//  * the global is not used directly as a ref.eq operand (identity chains).
//
// One common exception is handled: a mutable field that is only ever written
// under a null check of that same field of the same object (a lazily
// initialized slot, such as the Class object slot of a J2CL vtable) can never
// be written on an object whose initializer sets it to a non-null value, so
// such fields do not prevent merging. With
//   --pass-arg=merge-identical-globals-merge-lazy-null@1
// such slots may also be initialized to null, in which case the lazily
// created object becomes shared between the merged objects (for J2CL: the
// classes behind the merged vtables share one Class object).
//
// Nested allocations are hoisted innermost first, into new immutable globals
// (or an existing identical global), when the estimated size gain is
// positive. This pass is meant for the end of a closed-world pipeline; later
// passes (remove-unused-module-elements, duplicate-function-elimination) fold
// the users of the merged globals.
//

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

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

// How a struct field is written anywhere in the module.
enum class WriteState { None, Guarded, Unguarded };

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

// Per-function write information.
struct Writes {
  std::unordered_set<FieldKey, FieldKeyHash> guarded, unguarded;
  std::unordered_set<HeapType> arrays;
  // Globals used directly as ref.eq operands.
  std::unordered_set<Name> identity;
};

struct MergeIdenticalGlobals : public Pass {
  Module* module;
  bool mergeLazyNull = false;

  Writes writes;
  std::unordered_map<HeapType, std::vector<WriteState>> fieldStateCache;
  std::unordered_map<HeapType, bool> arrayWrittenCache;
  std::unordered_set<Name> exportedGlobals;
  std::unordered_map<Name, Index> globalIndexes;
  std::unordered_map<Name, Index> functionIndexes;

  void run(Module* module_) override {
    module = module_;
    if (!module->features.hasGC()) {
      return;
    }
    mergeLazyNull =
      !getArgumentOrDefault("merge-identical-globals-merge-lazy-null", "")
         .empty();

    collectWrites();
    for (auto& ex : module->exports) {
      if (ex->kind == ExternalKind::Global) {
        exportedGlobals.insert(*ex->getInternalName());
      }
    }
    computeIndexes();

    // Merging globals can make the initializers that read them identical, so
    // iterate (bounded).
    constexpr Index MaxMergeRounds = 8;
    for (Index round = 0; round < MaxMergeRounds; round++) {
      roots.clear();
      if (!mergeRoots()) {
        break;
      }
    }
    if (hoistNested()) {
      // The new globals were appended; sort so that they appear before their
      // uses (and after the globals they use).
      PassRunner runner(module);
      runner.add("reorder-globals-always");
      runner.setIsNested(true);
      runner.run();
    }
  }

  // --- Write analysis -------------------------------------------------------

  void collectWrites() {
    struct Scanner : public ExpressionStackWalker<Scanner> {
      Writes& writes;
      // Candidate guarded sets: the field key, the local the ref reads, and
      // whether the set is guarded by a null check of that field.
      struct SetInfo {
        FieldKey key;
        Index local;
        bool guarded;
      };
      std::vector<SetInfo> sets;
      std::vector<Index> localSets;

      Scanner(Writes& writes) : writes(writes) {}

      static bool getStructRefType(Expression* ref, HeapType& type) {
        if (!ref->type.isRef()) {
          return false;
        }
        type = ref->type.getHeapType();
        return type.isStruct();
      }

      static bool getArrayRefType(Expression* ref, HeapType& type) {
        if (!ref->type.isRef()) {
          return false;
        }
        type = ref->type.getHeapType();
        return type.isArray();
      }

      void visitLocalSet(LocalSet* curr) {
        if (curr->index >= localSets.size()) {
          localSets.resize(curr->index + 1);
        }
        localSets[curr->index]++;
      }

      void visitStructSet(StructSet* curr) {
        HeapType type;
        if (!getStructRefType(curr->ref, type)) {
          return;
        }
        FieldKey key{type, curr->index};
        auto* get = curr->ref->dynCast<LocalGet>();
        if (!get) {
          writes.unguarded.insert(key);
          return;
        }
        sets.push_back({key, get->index, isGuarded(curr->index, get->index)});
      }

      void visitStructRMW(StructRMW* curr) {
        HeapType type;
        if (getStructRefType(curr->ref, type)) {
          writes.unguarded.insert({type, curr->index});
        }
      }

      void visitStructCmpxchg(StructCmpxchg* curr) {
        HeapType type;
        if (getStructRefType(curr->ref, type)) {
          writes.unguarded.insert({type, curr->index});
        }
      }

      void noteArrayWrite(Expression* ref) {
        HeapType type;
        if (getArrayRefType(ref, type)) {
          writes.arrays.insert(type);
        }
      }

      void visitArraySet(ArraySet* curr) { noteArrayWrite(curr->ref); }
      void visitArrayCopy(ArrayCopy* curr) { noteArrayWrite(curr->destRef); }
      void visitArrayFill(ArrayFill* curr) { noteArrayWrite(curr->ref); }
      void visitArrayInitData(ArrayInitData* curr) {
        noteArrayWrite(curr->ref);
      }
      void visitArrayInitElem(ArrayInitElem* curr) {
        noteArrayWrite(curr->ref);
      }
      void visitArrayRMW(ArrayRMW* curr) { noteArrayWrite(curr->ref); }
      void visitArrayCmpxchg(ArrayCmpxchg* curr) { noteArrayWrite(curr->ref); }

      void visitRefEq(RefEq* curr) {
        for (auto* operand : {curr->left, curr->right}) {
          if (auto* get = operand->dynCast<GlobalGet>()) {
            writes.identity.insert(get->name);
          }
        }
      }

      // Whether |expr| is (struct.get k (local.get local)), the read of the
      // field being guarded.
      static bool isFieldRead(Expression* expr, Index k, Index local) {
        auto* get = expr->dynCast<StructGet>();
        if (!get || get->index != k) {
          return false;
        }
        auto* localGet = get->ref->dynCast<LocalGet>();
        return localGet && localGet->index == local;
      }

      static bool isNullCheck(Expression* expr, Index k, Index local) {
        auto* isNull = expr->dynCast<RefIsNull>();
        return isNull && isFieldRead(isNull->value, k, local);
      }

      static bool isNotNullCheck(Expression* expr, Index k, Index local) {
        auto* unary = expr->dynCast<Unary>();
        return unary && unary->op == EqZInt32 &&
               isNullCheck(unary->value, k, local);
      }

      // Whether the current struct.set of field k of (local.get local) can
      // only execute if that field is null: it is inside the true arm of an
      // if on (ref.is_null (struct.get k (local.get local))) (or the false
      // arm of the negation), or after a (br_on_non_null $b (struct.get ..))
      // in the same block $b.
      bool isGuarded(Index k, Index local) {
        assert(!expressionStack.empty());
        for (Index i = expressionStack.size() - 1; i > 0; i--) {
          auto* parent = expressionStack[i - 1];
          auto* child = expressionStack[i];
          if (auto* iff = parent->dynCast<If>()) {
            if (child == iff->ifTrue && isNullCheck(iff->condition, k, local)) {
              return true;
            }
            if (child == iff->ifFalse &&
                isNotNullCheck(iff->condition, k, local)) {
              return true;
            }
          } else if (auto* block = parent->dynCast<Block>()) {
            if (!block->name.is()) {
              continue;
            }
            for (auto* item : block->list) {
              if (item == child) {
                break;
              }
              auto* brOn = item->dynCast<BrOn>();
              if (brOn && brOn->op == BrOnNonNull &&
                  brOn->name == block->name &&
                  isFieldRead(brOn->ref, k, local)) {
                return true;
              }
            }
          }
        }
        return false;
      }

      void finish() {
        for (auto& set : sets) {
          bool localStable =
            set.local >= localSets.size() || localSets[set.local] == 0;
          if (set.guarded && localStable) {
            writes.guarded.insert(set.key);
          } else {
            writes.unguarded.insert(set.key);
          }
        }
      }
    };

    ModuleUtils::ParallelFunctionAnalysis<Writes> analysis(
      *module, [&](Function* func, Writes& funcWrites) {
        if (func->imported()) {
          return;
        }
        Scanner scanner(funcWrites);
        scanner.walkFunction(func);
        scanner.finish();
      });
    for (auto& [_, funcWrites] : analysis.map) {
      writes.guarded.insert(funcWrites.guarded.begin(),
                            funcWrites.guarded.end());
      writes.unguarded.insert(funcWrites.unguarded.begin(),
                              funcWrites.unguarded.end());
      writes.arrays.insert(funcWrites.arrays.begin(), funcWrites.arrays.end());
      writes.identity.insert(funcWrites.identity.begin(),
                             funcWrites.identity.end());
    }
  }

  // The write state of each field of a struct type, considering writes through
  // all of its supertypes.
  const std::vector<WriteState>& getFieldStates(HeapType type) {
    auto iter = fieldStateCache.find(type);
    if (iter != fieldStateCache.end()) {
      return iter->second;
    }
    auto& states = fieldStateCache[type];
    states.resize(type.getStruct().fields.size(), WriteState::None);
    for (std::optional<HeapType> curr = type; curr;
         curr = curr->getDeclaredSuperType()) {
      Index numFields = curr->getStruct().fields.size();
      for (Index k = 0; k < numFields; k++) {
        FieldKey key{*curr, k};
        if (writes.unguarded.count(key)) {
          states[k] = WriteState::Unguarded;
        } else if (writes.guarded.count(key) && states[k] == WriteState::None) {
          states[k] = WriteState::Guarded;
        }
      }
    }
    return states;
  }

  bool isArrayWritten(HeapType type) {
    auto iter = arrayWrittenCache.find(type);
    if (iter != arrayWrittenCache.end()) {
      return iter->second;
    }
    bool written = false;
    for (std::optional<HeapType> curr = type; curr;
         curr = curr->getDeclaredSuperType()) {
      if (writes.arrays.count(*curr)) {
        written = true;
        break;
      }
    }
    return arrayWrittenCache[type] = written;
  }

  // --- Mergeability ---------------------------------------------------------

  static bool isLeaf(Expression* curr) {
    if (curr->is<Const>() || curr->is<RefNull>() || curr->is<RefFunc>() ||
        curr->is<GlobalGet>() || curr->is<StringConst>()) {
      return true;
    }
    if (auto* i31 = curr->dynCast<RefI31>()) {
      return i31->value->is<Const>();
    }
    return false;
  }

  static bool isAllocation(Expression* curr) {
    return curr->is<StructNew>() || curr->is<ArrayNewFixed>() ||
           curr->is<ArrayNew>();
  }

  // Whether |curr| allocates an object (a tree of allocations over constant
  // leaves) that may be shared with other identical allocations.
  bool isShareable(Expression* curr) {
    if (!curr->type.isRef() || curr->type.getHeapType().isShared()) {
      return false;
    }
    auto type = curr->type.getHeapType();
    if (auto* structNew = curr->dynCast<StructNew>()) {
      if (structNew->desc) {
        return false;
      }
      auto& states = getFieldStates(type);
      for (Index k = 0; k < states.size(); k++) {
        if (states[k] == WriteState::Unguarded) {
          return false;
        }
        if (states[k] == WriteState::Guarded && !mergeLazyNull) {
          // The lazy slot must already be initialized, so that the guarded
          // writes can never happen on this object.
          if (structNew->isWithDefault() ||
              structNew->operands[k]->is<RefNull>()) {
            return false;
          }
        }
      }
      for (auto* operand : structNew->operands) {
        if (!isShareableOperand(operand)) {
          return false;
        }
      }
      return true;
    }
    if (auto* arrayNew = curr->dynCast<ArrayNewFixed>()) {
      if (isArrayWritten(type)) {
        return false;
      }
      for (auto* value : arrayNew->values) {
        if (!isShareableOperand(value)) {
          return false;
        }
      }
      return true;
    }
    if (auto* arrayNew = curr->dynCast<ArrayNew>()) {
      if (isArrayWritten(type) || !arrayNew->size->is<Const>()) {
        return false;
      }
      return arrayNew->isWithDefault() || isLeaf(arrayNew->init);
    }
    return false;
  }

  bool isShareableOperand(Expression* curr) {
    return isLeaf(curr) || isShareable(curr);
  }

  // --- Phase 1: merge identical root globals ---------------------------------

  // Groups of structurally identical immutable globals, keyed by the hash of
  // the initializer. Each entry is the representative (first) global.
  std::unordered_map<size_t, std::vector<Global*>> roots;

  Global* findRoot(size_t hash, Type type, Expression* init) {
    auto iter = roots.find(hash);
    if (iter == roots.end()) {
      return nullptr;
    }
    for (auto* global : iter->second) {
      if (global->type == type &&
          ExpressionAnalyzer::equal(global->init, init)) {
        return global;
      }
    }
    return nullptr;
  }

  // Finds an existing identical global whose type is acceptable at every
  // occurrence.
  template<typename T>
  Global* findRoot(size_t hash, Expression* init, const T& occurrences) {
    auto iter = roots.find(hash);
    if (iter == roots.end()) {
      return nullptr;
    }
    for (auto* global : iter->second) {
      if (!ExpressionAnalyzer::equal(global->init, init)) {
        continue;
      }
      bool fits = true;
      for (auto& occurrence : occurrences) {
        if (!Type::isSubType(global->type, occurrence.required)) {
          fits = false;
          break;
        }
      }
      if (fits) {
        return global;
      }
    }
    return nullptr;
  }

  bool isMergeableRoot(Global* global) {
    return !global->imported() && !global->mutable_ &&
           !exportedGlobals.count(global->name) &&
           !writes.identity.count(global->name) && isAllocation(global->init) &&
           isShareable(global->init);
  }

  // Returns true if any globals were merged.
  bool mergeRoots() {
    std::unordered_map<Name, Name> renames;
    for (auto& global : module->globals) {
      if (!isMergeableRoot(global.get())) {
        continue;
      }
      auto hash = ExpressionAnalyzer::hash(global->init);
      if (auto* first = findRoot(hash, global->type, global->init)) {
        renames[global->name] = first->name;
      } else {
        roots[hash].push_back(global.get());
      }
    }
    if (renames.empty()) {
      return false;
    }

    struct Renamer : public WalkerPass<PostWalker<Renamer>> {
      bool isFunctionParallel() override { return true; }
      std::unordered_map<Name, Name>& renames;
      Renamer(std::unordered_map<Name, Name>& renames) : renames(renames) {}
      std::unique_ptr<Pass> create() override {
        return std::make_unique<Renamer>(renames);
      }
      void visitGlobalGet(GlobalGet* curr) {
        auto iter = renames.find(curr->name);
        if (iter != renames.end()) {
          curr->name = iter->second;
        }
      }
    };
    Renamer renamer(renames);
    renamer.run(getPassRunner(), module);
    renamer.runOnModuleCode(getPassRunner(), module);

    module->removeGlobals(
      [&](Global* global) { return renames.count(global->name) > 0; });
    return true;
  }

  // --- Phase 0: hoist nested allocations ------------------------------------

  // Hoists innermost repeated allocations, round by round (after a round the
  // parents of hoisted allocations have become innermost themselves).
  bool hoistNested() {
    bool addedGlobals = false;
    constexpr Index MaxRounds = 16;
    for (Index round = 0; round < MaxRounds; round++) {
      if (!hoistNestedRound()) {
        break;
      }
      addedGlobals = true;
    }
    return addedGlobals;
  }

  struct Occurrence {
    Expression** location;
    Expression* expr;
    // The type the parent requires at this position.
    Type required;
  };

  // The type a parent allocation requires for the given child.
  static Type getRequiredType(Expression* parent, Expression* child) {
    if (auto* structNew = parent->dynCast<StructNew>()) {
      auto& fields = structNew->type.getHeapType().getStruct().fields;
      for (Index i = 0; i < structNew->operands.size(); i++) {
        if (structNew->operands[i] == child) {
          return fields[i].type;
        }
      }
    } else if (parent->is<ArrayNewFixed>() || parent->is<ArrayNew>()) {
      return parent->type.getHeapType().getArray().element.type;
    }
    return child->type;
  }

  bool hoistNestedRound() {
    // Collect the innermost nested allocations (all children are leaves) in
    // all global initializers.
    struct Collector
      : public ExpressionStackWalker<Collector,
                                     UnifiedExpressionVisitor<Collector>> {
      MergeIdenticalGlobals& parent;
      std::vector<Occurrence>& occurrences;
      Collector(MergeIdenticalGlobals& parent,
                std::vector<Occurrence>& occurrences)
        : parent(parent), occurrences(occurrences) {}
      void visitExpression(Expression* curr) {
        auto* parentExpr = getParent();
        if (!parentExpr || !isAllocation(curr)) {
          return;
        }
        for (auto* child : ChildIterator(curr)) {
          if (!isLeaf(child)) {
            return;
          }
        }
        if (parent.isShareable(curr)) {
          occurrences.push_back(
            {getCurrentPointer(), curr, getRequiredType(parentExpr, curr)});
        }
      }
    };
    std::vector<Occurrence> occurrences;
    Collector collector(*this, occurrences);
    for (auto& global : module->globals) {
      if (!global->imported()) {
        collector.walk(global->init);
      }
    }
    if (occurrences.empty()) {
      return false;
    }

    // Group by structure.
    struct Group {
      std::vector<Occurrence> members;
    };
    InsertOrderedMap<size_t, std::vector<Group>> groups;
    for (auto& occurrence : occurrences) {
      auto hash = ExpressionAnalyzer::hash(occurrence.expr);
      auto& list = groups[hash];
      bool found = false;
      for (auto& group : list) {
        if (ExpressionAnalyzer::equal(group.members[0].expr, occurrence.expr)) {
          group.members.push_back(occurrence);
          found = true;
          break;
        }
      }
      if (!found) {
        list.push_back({{occurrence}});
      }
    }

    bool changed = false;
    Builder builder(*module);
    for (auto& [hash, list] : groups) {
      for (auto& group : list) {
        auto* expr = group.members[0].expr;
        Index count = group.members.size();
        Index bytes = getBytes(expr);
        Name name;
        Type type;
        if (auto* existing = findRoot(hash, expr, group.members)) {
          // An identical global exists already; reuse it when that is not
          // larger (a global.get of it is at most 3 bytes).
          if (bytes <= 3) {
            continue;
          }
          name = existing->name;
          type = existing->type;
        } else {
          if (count < 2) {
            continue;
          }
          // A new global costs its type and mutability, the initializer and
          // an end opcode, plus a little for the global count; each hoisted
          // occurrence becomes a global.get, which gets a small index when
          // it is used often.
          Index getBytes = count >= 64 ? 2 : 3;
          if (bytes <= getBytes ||
              (count - 1) * (bytes - getBytes) <= bytes + 5) {
            continue;
          }
          name = Names::getValidGlobalName(*module, "hoisted");
          type = expr->type;
          auto* global = module->addGlobal(
            builder.makeGlobal(name, type, expr, Builder::Immutable));
          roots[hash].push_back(global);
        }
        for (auto& member : group.members) {
          *member.location = builder.makeGlobalGet(name, type);
        }
        changed = true;
      }
    }
    return changed;
  }

  // --- Size estimates -------------------------------------------------------

  void computeIndexes() {
    Index i = 0;
    for (auto& global : module->globals) {
      globalIndexes[global->name] = i++;
    }
    i = 0;
    for (auto& func : module->functions) {
      functionIndexes[func->name] = i++;
    }
  }

  template<typename T> static Index getLEBSize(const T& leb) {
    BufferWithRandomAccess buffer;
    buffer << leb;
    return buffer.size();
  }

  // The estimated number of bytes for the binary encoding of a constant
  // expression.
  Index getBytes(Expression* curr) {
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
      auto iter = globalIndexes.find(get->name);
      Index index = iter != globalIndexes.end() ? iter->second : 128;
      return 1 + getLEBSize(U32LEB(index));
    }
    if (auto* i31 = curr->dynCast<RefI31>()) {
      return 2 + getBytes(i31->value);
    }
    if (curr->is<StringConst>()) {
      return 4;
    }
    // An allocation: opcode prefix and opcode, a type index, and for
    // array.new_fixed a count.
    Index bytes = curr->is<ArrayNewFixed>() ? 5 : 4;
    for (auto* child : ChildIterator(curr)) {
      bytes += getBytes(child);
    }
    return bytes;
  }
};

} // anonymous namespace

Pass* createMergeIdenticalGlobalsPass() { return new MergeIdenticalGlobals(); }

} // namespace wasm
