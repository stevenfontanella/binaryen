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
// Defers the allocation of constant global objects to their first use.
//
// Many immutable globals are initialized with a struct.new of constant
// operands (string literal wrappers, enum constants, class metadata) and are
// only read by global.get in function code. Each such global costs a full
// global entry (type, mutability, struct.new, operands, end) even if it is
// used only once.
//
// This pass groups such globals by struct type. For each group the operands
// that are identical in all members stay in a shared factory function, while
// the operands that vary become columns of immutable array "tables" (one
// array.new_fixed per column). A mutable array caches the created objects so
// that each original global still corresponds to exactly one object (identity
// and mutable fields are preserved). The factory is
//
//   (func $factory (param $i i32) (result (ref $T))
//     (block $cached (result (ref $T))
//       (br_on_non_null $cached (array.get $cache (local.get $i)))
//       (array.set $cache (local.get $i)
//         (local.tee $obj (struct.new $T fixed... (array.get $column i)...)))
//       (local.get $obj)))
//
// and every (global.get $g_k) becomes (call $factory (i32.const k)). A global
// is deferred only when the model says it saves bytes, i.e. when its global
// entry is larger than its table bytes plus the extra bytes at its use sites,
// so globals with many uses stay eager. Groups are chunked so that every
// array.new_fixed stays below engine limits.
//
// Uses from other global initializers, segments, and exports keep a global
// eager. Groups whose members are mostly used as struct.new operands (e.g.
// vtables) are skipped by default, as a call on every allocation is a runtime
// cost; --pass-arg=defer-global-allocations-allocation-operands@1 includes
// them.
//
// This moves the allocation from instantiation to first use, and adds a call,
// an array read and a null check to every use. It is meant for the very end
// of a size-focused pipeline (before reorder-globals/reorder-functions).
//

#include <algorithm>

#include "ir/module-utils.h"
#include "ir/names.h"
#include "ir/utils.h"
#include "pass.h"
#include "support/hash.h"
#include "support/insert_ordered.h"
#include "wasm-binary.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

// Keep array.new_fixed well below the V8 limit of 10000 operands, and keep
// the indexes at most 2 bytes as an SLEB.
constexpr Index MaxChunkSize = 8192;

struct DeferGlobalAllocations : public Pass {
  // Per-function information about global.get uses.
  struct FunctionInfo {
    // All global.gets, in traversal order.
    std::vector<Expression**> gets;
    // Globals used directly as operands of an allocation.
    std::unordered_map<Name, Index> allocOperandUses;
  };

  struct Candidate {
    Global* global;
    StructNew* init;
    // Estimated size of the global entry in the binary.
    Index initBytes = 0;
    // Code uses.
    Index uses = 0;
    Index allocOperandUses = 0;
    // Order of first use in the code.
    Index firstUse = Index(-1);
  };

  struct Group {
    HeapType type;
    Type globalType;
    std::vector<Candidate> members;
  };

  struct Deferral {
    Name factory;
    Index index;
  };

  Module* module;
  std::unordered_map<Name, Index> globalIndexes;
  std::unordered_map<Name, Index> functionIndexes;
  bool includeAllocationOperands = false;
  // Ignore the size model and defer every candidate (for testing, and for
  // startup-focused builds).
  bool always = false;
  Index numFactories = 0;

  void run(Module* module_) override {
    module = module_;
    if (!module->features.hasGC()) {
      return;
    }
    includeAllocationOperands =
      !getArgumentOrDefault("defer-global-allocations-allocation-operands", "")
         .empty();
    always =
      !getArgumentOrDefault("defer-global-allocations-always", "").empty();

    computeIndexes();

    // Globals that are used outside of function code must stay eager.
    std::unordered_set<Name> nonCodeUses;
    {
      struct Scanner : public PostWalker<Scanner> {
        std::unordered_set<Name>& names;
        Scanner(std::unordered_set<Name>& names) : names(names) {}
        void visitGlobalGet(GlobalGet* curr) { names.insert(curr->name); }
      } scanner(nonCodeUses);
      for (auto& global : module->globals) {
        if (!global->imported()) {
          scanner.walk(global->init);
        }
      }
      for (auto& segment : module->elementSegments) {
        if (segment->offset) {
          scanner.walk(segment->offset);
        }
        for (auto*& item : segment->data) {
          scanner.walk(item);
        }
      }
      for (auto& segment : module->dataSegments) {
        if (segment->offset) {
          scanner.walk(segment->offset);
        }
      }
      for (auto& ex : module->exports) {
        if (ex->kind == ExternalKind::Global) {
          nonCodeUses.insert(*ex->getInternalName());
        }
      }
    }

    // Find the candidate globals and group them by type.
    InsertOrderedMap<std::pair<HeapType, Type>, Group> groups;
    std::unordered_map<Name, std::pair<Group*, Index>> candidateMap;
    for (auto& global : module->globals) {
      if (global->imported() || global->mutable_ ||
          nonCodeUses.count(global->name)) {
        continue;
      }
      auto* init = global->init->dynCast<StructNew>();
      if (!init || init->desc || !init->type.isRef()) {
        continue;
      }
      auto type = init->type.getHeapType();
      // The global's type must be the struct type itself (possibly exact), so
      // that a call to the factory can have the same type as the global.get.
      if (type.isShared() || !global->type.isRef() ||
          global->type.isNullable() || global->type.getHeapType() != type ||
          init->operands.size() != type.getStruct().fields.size() ||
          !isValidOperands(init)) {
        continue;
      }
      Candidate candidate;
      candidate.global = global.get();
      candidate.init = init;
      candidate.initBytes = getGlobalEntryBytes(global.get(), init);
      auto& group = groups[{type, global->type}];
      group.type = type;
      group.globalType = global->type;
      group.members.push_back(candidate);
      candidateMap[global->name] = {&group, Index(group.members.size() - 1)};
    }
    if (groups.empty()) {
      return;
    }

    // Count the uses in function code, in parallel.
    ModuleUtils::ParallelFunctionAnalysis<FunctionInfo> analysis(
      *module, [&](Function* func, FunctionInfo& info) {
        if (func->imported()) {
          return;
        }
        struct Scanner : public PostWalker<Scanner> {
          FunctionInfo& info;
          const std::unordered_map<Name, std::pair<Group*, Index>>& candidates;
          Scanner(FunctionInfo& info,
                  const std::unordered_map<Name, std::pair<Group*, Index>>&
                    candidates)
            : info(info), candidates(candidates) {}
          void visitGlobalGet(GlobalGet* curr) {
            if (candidates.count(curr->name)) {
              info.gets.push_back(getCurrentPointer());
            }
          }
          void noteOperands(ExpressionList& operands) {
            for (auto* operand : operands) {
              if (auto* get = operand->dynCast<GlobalGet>()) {
                if (candidates.count(get->name)) {
                  info.allocOperandUses[get->name]++;
                }
              }
            }
          }
          void visitStructNew(StructNew* curr) { noteOperands(curr->operands); }
          void visitArrayNewFixed(ArrayNewFixed* curr) {
            noteOperands(curr->values);
          }
          void visitArrayNew(ArrayNew* curr) {
            if (auto* get =
                  curr->init ? curr->init->dynCast<GlobalGet>() : nullptr) {
              if (candidates.count(get->name)) {
                info.allocOperandUses[get->name]++;
              }
            }
          }
        };
        Scanner(info, candidateMap).walk(func->body);
      });

    // Merge the per-function information in a deterministic order.
    std::vector<Expression**> allGets;
    for (auto& func : module->functions) {
      auto& info = analysis.map[func.get()];
      for (auto** ptr : info.gets) {
        auto& [group, i] = candidateMap[(*ptr)->cast<GlobalGet>()->name];
        auto& candidate = group->members[i];
        if (candidate.uses == 0) {
          candidate.firstUse = allGets.size();
        }
        candidate.uses++;
        allGets.push_back(ptr);
      }
      for (auto& [name, count] : info.allocOperandUses) {
        auto& [group, i] = candidateMap[name];
        group->members[i].allocOperandUses += count;
      }
    }

    // Decide what to defer, and build the factories.
    std::unordered_map<Name, Deferral> deferrals;
    for (auto& [key, group] : groups) {
      processGroup(group, deferrals);
    }
    if (deferrals.empty()) {
      return;
    }

    // Replace the uses and remove the globals.
    Builder builder(*module);
    for (auto** ptr : allGets) {
      auto* get = (*ptr)->cast<GlobalGet>();
      auto iter = deferrals.find(get->name);
      if (iter == deferrals.end()) {
        continue;
      }
      auto& deferral = iter->second;
      *ptr = builder.makeCall(deferral.factory,
                              {builder.makeConst(int32_t(deferral.index))},
                              get->type);
    }
    for (auto& [name, _] : deferrals) {
      module->removeGlobal(name);
    }
  }

private:
  void computeIndexes() {
    Index importedGlobals = 0, definedGlobals = 0;
    for (auto& global : module->globals) {
      if (global->imported()) {
        importedGlobals++;
      } else {
        definedGlobals++;
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

  // Whether an operand is a constant we can move into the factory or a table.
  bool isValidOperand(Expression* curr) {
    if (curr->is<Const>() || curr->is<RefNull>() || curr->is<RefFunc>()) {
      return true;
    }
    if (auto* get = curr->dynCast<GlobalGet>()) {
      return !module->getGlobal(get->name)->mutable_;
    }
    if (auto* i31 = curr->dynCast<RefI31>()) {
      return i31->value->is<Const>();
    }
    if (auto* structNew = curr->dynCast<StructNew>()) {
      // struct.new_default
      return structNew->operands.empty() && !structNew->desc;
    }
    return false;
  }

  bool isValidOperands(StructNew* init) {
    for (auto* operand : init->operands) {
      if (!isValidOperand(operand)) {
        return false;
      }
    }
    return true;
  }

  template<typename T> Index getLEBSize(const T& leb) {
    BufferWithRandomAccess buffer;
    buffer << leb;
    return buffer.size();
  }

  // The estimated number of bytes for the binary encoding of a (constant)
  // operand expression.
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

  Index getGlobalEntryBytes(Global* global, StructNew* init) {
    // The type (ref $T) and mutability, struct.new with a type index, the
    // operands, and the end.
    Index bytes = 3 + 1 + 4 + 1;
    for (auto* operand : init->operands) {
      bytes += getOperandBytes(operand);
    }
    return bytes;
  }

  Index getSLEBSize(Index value) { return getLEBSize(S32LEB(int32_t(value))); }

  void processGroup(Group& group,
                    std::unordered_map<Name, Deferral>& deferrals) {
    auto& members = group.members;
    // Only globals that are used in code matter (unused ones are left to
    // other passes).
    members.erase(
      std::remove_if(members.begin(),
                     members.end(),
                     [](const Candidate& c) { return c.uses == 0; }),
      members.end());
    if (members.empty()) {
      return;
    }
    if (!includeAllocationOperands) {
      Index uses = 0, allocOperandUses = 0;
      for (auto& member : members) {
        uses += member.uses;
        allocOperandUses += member.allocOperandUses;
      }
      if (allocOperandUses * 2 > uses) {
        return;
      }
    }

    auto& fields = group.type.getStruct().fields;
    Index numFields = fields.size();
    auto varying = getVaryingColumns(members, numFields);

    // A global.get is replaced by a call (opcode + function index; the factory
    // is used often, so assume a 2-byte index) and an i32.const.
    const Index callBytes = 3;

    // Greedily select members, cheapest first, so that the index estimate is
    // meaningful.
    std::vector<Candidate*> sorted;
    for (auto& member : members) {
      sorted.push_back(&member);
    }
    std::sort(sorted.begin(), sorted.end(), [](Candidate* a, Candidate* b) {
      if (a->uses != b->uses) {
        return a->uses < b->uses;
      }
      return a->firstUse < b->firstUse;
    });
    std::vector<Candidate*> chosen;
    int64_t saving = 0;
    for (auto* member : sorted) {
      Index tableBytes = 0;
      for (auto i : varying) {
        tableBytes += getOperandBytes(member->init->operands[i]);
      }
      Index index = chosen.size() % MaxChunkSize;
      Index getBytes =
        1 + getLEBSize(U32LEB(globalIndexes[member->global->name]));
      int64_t siteDelta =
        int64_t(callBytes + 1 + getSLEBSize(index)) - int64_t(getBytes);
      int64_t memberSaving = int64_t(member->initBytes) - int64_t(tableBytes) -
                             int64_t(member->uses) * siteDelta;
      if (always || memberSaving > 0) {
        chosen.push_back(member);
        saving += memberSaving;
      }
    }
    if (chosen.empty()) {
      return;
    }
    // Recompute the columns over the chosen members only.
    std::vector<Candidate> chosenMembers;
    for (auto* member : chosen) {
      chosenMembers.push_back(*member);
    }
    varying = getVaryingColumns(chosenMembers, numFields);

    // Fixed costs per chunk: the factory function (code, function section
    // entry, type), the cache global and its type, and per table column the
    // global, its type, and the read in the factory.
    Index numChunks = (chosen.size() + MaxChunkSize - 1) / MaxChunkSize;
    Index overhead = numChunks * (70 + 20 * varying.size());
    if (!always && saving <= int64_t(overhead)) {
      return;
    }

    // Assign the indexes in first-use order, which keeps the constants in
    // long initialization functions in ascending order.
    std::sort(chosen.begin(), chosen.end(), [](Candidate* a, Candidate* b) {
      return a->firstUse < b->firstUse;
    });
    for (Index chunk = 0; chunk < numChunks; chunk++) {
      Index begin = chunk * MaxChunkSize;
      Index end = std::min(Index(chosen.size()), begin + MaxChunkSize);
      std::vector<Candidate*> chunkMembers(chosen.begin() + begin,
                                           chosen.begin() + end);
      auto factory = makeFactory(group, chunkMembers, varying);
      for (Index i = 0; i < chunkMembers.size(); i++) {
        deferrals[chunkMembers[i]->global->name] = {factory, i};
      }
    }
  }

  std::vector<Index> getVaryingColumns(std::vector<Candidate>& members,
                                       Index numFields) {
    std::vector<Index> varying;
    for (Index i = 0; i < numFields; i++) {
      auto* first = members[0].init->operands[i];
      for (auto& member : members) {
        if (!ExpressionAnalyzer::equal(first, member.init->operands[i])) {
          varying.push_back(i);
          break;
        }
      }
    }
    return varying;
  }

  Name makeFactory(Group& group,
                   std::vector<Candidate*>& members,
                   std::vector<Index>& varying) {
    Builder builder(*module);
    auto type = group.type;
    auto resultType = group.globalType;
    auto& fields = type.getStruct().fields;
    std::string typeName;
    if (auto iter = module->typeNames.find(type);
        iter != module->typeNames.end()) {
      typeName = iter->second.name.toString();
    } else {
      typeName = std::to_string(numFactories);
    }
    numFactories++;
    std::string baseName = "defer$" + typeName;

    // The cache.
    // The cache holds nullable versions of the result type (which may be
    // exact).
    auto cacheElementType = resultType.with(Nullable);
    HeapType cacheType(Array(Field(cacheElementType, Mutable)));
    auto cacheName = Names::getValidGlobalName(*module, baseName + "$cache");
    auto cacheGlobalType = Type(cacheType, NonNullable);
    module->addGlobal(builder.makeGlobal(
      cacheName,
      cacheGlobalType,
      builder.makeArrayNew(cacheType,
                           builder.makeConst(int32_t(members.size()))),
      Builder::Immutable));

    // The tables.
    struct Column {
      Index field;
      Name global;
      Type globalType;
    };
    std::vector<Column> columns;
    for (auto i : varying) {
      HeapType tableType(Array(Field(fields[i].type, Immutable)));
      if (fields[i].isPacked()) {
        tableType = HeapType(Array(Field(fields[i].packedType, Immutable)));
      }
      std::vector<Expression*> values;
      for (auto* member : members) {
        values.push_back(
          ExpressionManipulator::copy(member->init->operands[i], *module));
      }
      auto tableName = Names::getValidGlobalName(
        *module, baseName + "$table$" + std::to_string(i));
      auto tableGlobalType = Type(tableType, NonNullable);
      module->addGlobal(
        builder.makeGlobal(tableName,
                           tableGlobalType,
                           builder.makeArrayNewFixed(tableType, values),
                           Builder::Immutable));
      columns.push_back({i, tableName, tableGlobalType});
    }

    // The factory.
    Index indexLocal = 0;
    Index objectLocal = 1;
    Name label("cached");
    auto* cacheGet =
      builder.makeArrayGet(builder.makeGlobalGet(cacheName, cacheGlobalType),
                           builder.makeLocalGet(indexLocal, Type::i32),
                           MemoryOrder::Unordered,
                           cacheElementType);
    auto* brOn = builder.makeBrOn(BrOnNonNull, label, cacheGet);
    std::vector<Expression*> operands;
    Index column = 0;
    for (Index i = 0; i < fields.size(); i++) {
      if (column < columns.size() && columns[column].field == i) {
        auto& col = columns[column++];
        operands.push_back(builder.makeArrayGet(
          builder.makeGlobalGet(col.global, col.globalType),
          builder.makeLocalGet(indexLocal, Type::i32),
          MemoryOrder::Unordered,
          fields[i].type,
          false));
      } else {
        operands.push_back(
          ExpressionManipulator::copy(members[0]->init->operands[i], *module));
      }
    }
    auto* structNew = builder.makeStructNew(type, operands);
    auto* tee = builder.makeLocalTee(objectLocal, structNew, resultType);
    auto* cacheSet =
      builder.makeArraySet(builder.makeGlobalGet(cacheName, cacheGlobalType),
                           builder.makeLocalGet(indexLocal, Type::i32),
                           tee,
                           MemoryOrder::Unordered);
    auto* body = builder.makeBlock(
      label,
      {brOn, cacheSet, builder.makeLocalGet(objectLocal, resultType)},
      resultType);
    auto factoryName = Names::getValidFunctionName(*module, baseName);
    module->addFunction(builder.makeFunction(
      factoryName, Signature(Type::i32, resultType), {resultType}, body));
    return factoryName;
  }
};

} // anonymous namespace

Pass* createDeferGlobalAllocationsPass() {
  return new DeferGlobalAllocations();
}

} // namespace wasm
