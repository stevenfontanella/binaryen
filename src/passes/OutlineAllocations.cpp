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
// Outlines duplicated allocations into shared helper functions.
//
// Aggressive inlining (and source-level duplication) can leave many copies of
// allocation expressions like
//
//   (struct.new $T (global.get $vtable) (i32.const 0) (ref.null none) ..)
//
// whose operands are mostly constants. Each copy costs a few bytes per
// operand, while a call to a helper that performs the allocation costs about
// three bytes. This pass finds struct.new / array.new_fixed expressions in
// function bodies that have the same type and the same constant operands
// (numeric constants, ref.null, ref.func, string.const, global.get of immutable
// globals, and nested allocations of such constants), and replaces them with
// calls to a helper function containing a single copy of the allocation, when
// that is estimated to shrink the binary. Non-constant operands ("holes") are
// passed to the helper as parameters, in order, so the evaluation order of
// anything with side effects is unchanged, and each call still performs a fresh
// allocation.
//
// Allocations that are assigned to a global and are valid constant expressions
// are left alone, as other passes may be able to hoist them into the global's
// initializer, which is better.
//

#include <deque>
#include <sstream>

#include "ir/find_all.h"
#include "ir/manipulation.h"
#include "ir/module-utils.h"
#include "ir/names.h"
#include "ir/properties.h"
#include "pass.h"
#include "support/insert_ordered.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

// Rough estimates of binary sizes, in bytes.

// A call opcode plus a typical two-byte function index.
const Index CallBytes = 3;
// A local.get in the helper, for each hole.
const Index LocalGetBytes = 2;
// Function section entry, code size, local declarations, end opcode, and a
// share of a type section entry.
const Index HelperOverheadBytes = 6;

// The number of bytes used to encode a value as a signed LEB128.
int getSLEBSize(int64_t value) {
  int size = 0;
  while (true) {
    uint8_t byte = value & 0x7f;
    value >>= 7;
    size++;
    if ((value == 0 && !(byte & 0x40)) || (value == -1 && (byte & 0x40))) {
      return size;
    }
  }
}

// The number of bytes used to encode a value as an unsigned LEB128.
int getULEBSize(uint64_t value) {
  int size = 1;
  while (value >>= 7) {
    size++;
  }
  return size;
}

ExpressionList& getOperands(Expression* curr) {
  if (auto* structNew = curr->dynCast<StructNew>()) {
    return structNew->operands;
  }
  return curr->cast<ArrayNewFixed>()->values;
}

// The declared type of operand |i| of an allocation, which is the type we give
// a hole's parameter in the helper (so that it can be shared by sites that have
// different, refined, types there).
Type getOperandType(Expression* curr, Index i) {
  auto heapType = curr->type.getHeapType();
  if (curr->is<StructNew>()) {
    return heapType.getStruct().fields[i].type;
  }
  return heapType.getArray().element.type;
}

// An allocation in a function body that we may outline.
struct Site {
  // The struct.new or array.new_fixed.
  Expression* expr;
  // Describes the shape of the allocation: its type, its constant operands, and
  // the positions of the holes. Sites with equal keys can share a helper. Empty
  // if this site cannot be outlined or be part of another site's key.
  std::string key;
  // The indexes of the operands that are not constant.
  std::vector<Index> holes;
  // The estimated binary size of the allocation, not including the holes.
  Index bytes = 0;
  // The enclosing allocation that contains this one as a constant operand (so
  // that this site is part of its key), if any.
  Site* parent = nullptr;
  // Whether we must not outline this site. Such a site may still be part of a
  // parent's key.
  bool skip = false;
  // Whether this site is the entire body of a function that we can use as the
  // helper for its key (either one we created earlier, or by coincidence).
  bool helperBody = false;

  bool isConstant() { return !key.empty() && holes.empty(); }
};

struct FunctionInfo {
  // A deque so that pointers to sites remain valid as we add more.
  std::deque<Site> sites;
  // The number of references to each function, by call or ref.func. We use
  // this to decide where to place the helpers in the function index space.
  std::unordered_map<Name, Index> funcRefs;
};

struct Scanner : public PostWalker<Scanner> {
  Module& wasm;
  FunctionInfo& info;

  std::unordered_map<Expression*, Site*> siteMap;

  Scanner(Module& wasm, FunctionInfo& info) : wasm(wasm), info(info) {}

  void visitStructNew(StructNew* curr) { noteSite(curr); }

  void visitArrayNewFixed(ArrayNewFixed* curr) { noteSite(curr); }

  void visitCall(Call* curr) { info.funcRefs[curr->target]++; }

  void visitRefFunc(RefFunc* curr) { info.funcRefs[curr->func]++; }

  void visitGlobalSet(GlobalSet* curr) {
    if (!Properties::isValidConstantExpression(wasm, curr->value)) {
      return;
    }
    // A constant expression assigned to a global may be hoisted into the
    // global's initializer by other passes, which is better than outlining, so
    // leave all allocations in it alone.
    for (auto* child : FindAll<StructNew>(curr->value).list) {
      markSkip(child);
    }
    for (auto* child : FindAll<ArrayNewFixed>(curr->value).list) {
      markSkip(child);
    }
  }

  void markSkip(Expression* curr) {
    auto iter = siteMap.find(curr);
    if (iter != siteMap.end()) {
      iter->second->skip = true;
    }
  }

  void noteSite(Expression* curr) {
    info.sites.emplace_back();
    auto& site = info.sites.back();
    site.expr = curr;
    siteMap[curr] = &site;
    if (curr->type == Type::unreachable) {
      return;
    }
    auto& operands = getOperands(curr);
    if (auto* structNew = curr->dynCast<StructNew>()) {
      if (structNew->desc) {
        // TODO: handle descriptors.
        return;
      }
      // struct.new_default has nothing to outline, but it can be part of a
      // parent's key.
      if (structNew->isWithDefault()) {
        site.skip = true;
      }
    } else if (operands.empty()) {
      site.skip = true;
    }
    std::stringstream key;
    if (curr->is<StructNew>()) {
      key << 'S';
      // Prefix opcode, opcode, and a typical two-byte type index.
      site.bytes = 4;
    } else {
      key << 'A';
      // As above, plus the number of elements.
      site.bytes = 5;
    }
    key << curr->type.getHeapType().getID();
    for (Index i = 0; i < operands.size(); i++) {
      key << ' ';
      if (noteConstant(operands[i], key, site.bytes)) {
        if (auto iter = siteMap.find(operands[i]); iter != siteMap.end()) {
          iter->second->parent = &site;
        }
      } else {
        key << '_';
        site.holes.push_back(i);
      }
    }
    site.key = key.str();
  }

  // If |curr| is a constant that can be part of a helper, appends a description
  // of it to |key|, adds its estimated size to |bytes|, and returns true.
  bool noteConstant(Expression* curr, std::stringstream& key, Index& bytes) {
    if (auto* c = curr->dynCast<Const>()) {
      key << 'c' << c->type << ':';
      switch (c->type.getBasic()) {
        case Type::i32:
          key << c->value.geti32();
          bytes += 1 + getSLEBSize(c->value.geti32());
          return true;
        case Type::i64:
          key << c->value.geti64();
          bytes += 1 + getSLEBSize(c->value.geti64());
          return true;
        case Type::f32:
          key << c->value.reinterpreti32();
          bytes += 5;
          return true;
        case Type::f64:
          key << c->value.reinterpreti64();
          bytes += 9;
          return true;
        case Type::v128:
          for (auto byte : c->value.getv128()) {
            key << int(byte) << ',';
          }
          bytes += 18;
          return true;
        default:
          WASM_UNREACHABLE("unexpected const type");
      }
    }
    if (auto* null = curr->dynCast<RefNull>()) {
      key << 'n' << null->type.getID();
      bytes += 2;
      return true;
    }
    if (auto* func = curr->dynCast<RefFunc>()) {
      key << 'f' << func->func.str;
      bytes += 3;
      return true;
    }
    if (auto* string = curr->dynCast<StringConst>()) {
      key << 's' << string->string.str;
      bytes += 3;
      return true;
    }
    if (auto* get = curr->dynCast<GlobalGet>()) {
      if (wasm.getGlobal(get->name)->mutable_) {
        return false;
      }
      key << 'g' << get->name.str;
      bytes += 3;
      return true;
    }
    if (curr->is<StructNew>() || curr->is<ArrayNewFixed>()) {
      // We visit children before parents, so this site is already known.
      auto& child = *siteMap.at(curr);
      if (!child.isConstant()) {
        return false;
      }
      key << '{' << child.key << '}';
      bytes += child.bytes;
      return true;
    }
    return false;
  }
};

// Information about a replacement of a site with a call to a helper.
struct Replacement {
  Name helper;
  const Site* site;
};

using Replacements = std::unordered_map<Expression*, Replacement>;

struct Replacer : public WalkerPass<PostWalker<Replacer>> {
  bool isFunctionParallel() override { return true; }

  bool requiresNonNullableLocalFixups() override { return false; }

  std::unique_ptr<Pass> create() override {
    return std::make_unique<Replacer>(replacements);
  }

  Replacer(const Replacements& replacements) : replacements(replacements) {}

  void visitStructNew(StructNew* curr) { maybeReplace(curr); }

  void visitArrayNewFixed(ArrayNewFixed* curr) { maybeReplace(curr); }

  void maybeReplace(Expression* curr) {
    auto iter = replacements.find(curr);
    if (iter == replacements.end()) {
      return;
    }
    auto& replacement = iter->second;
    auto& operands = getOperands(curr);
    std::vector<Expression*> args;
    for (auto i : replacement.site->holes) {
      args.push_back(operands[i]);
    }
    replaceCurrent(
      Builder(*getModule()).makeCall(replacement.helper, args, curr->type));
  }

private:
  const Replacements& replacements;
};

struct OutlineAllocations : public Pass {
  // We only add calls to functions that have no effects besides allocation, so
  // the effects of existing functions do not change.
  bool addsEffects() override { return false; }

  bool requiresNonNullableLocalFixups() override { return false; }

  struct KeyInfo {
    std::vector<Site*> sites;
    // The function that performs this allocation, if we found or created one.
    Function* helper = nullptr;
    // Whether we decided to outline the sites with this key.
    bool outline = false;
  };

  Module* wasm;

  // Insertion-ordered so that the helpers are created in a deterministic order.
  InsertOrderedMap<std::string, KeyInfo> keys;

  void run(Module* module) override {
    wasm = module;
    if (!wasm->features.hasGC()) {
      return;
    }

    ModuleUtils::ParallelFunctionAnalysis<FunctionInfo> analysis(
      *wasm, [&](Function* func, FunctionInfo& info) {
        if (!func->imported()) {
          Scanner(*wasm, info).walk(func->body);
        }
      });

    // Group the sites by key, and find existing helpers.
    for (auto& func : wasm->functions) {
      auto& info = analysis.map[func.get()];
      for (auto& site : info.sites) {
        if (site.key.empty() || site.skip) {
          continue;
        }
        keys[site.key].sites.push_back(&site);
      }
      if (!info.sites.empty() && info.sites.back().expr == func->body) {
        noteExistingHelper(func.get(), info.sites.back());
      }
    }

    // Decide what to outline. Nested allocations complicate this, as a site
    // that is a constant operand of an outlined parent vanishes along with the
    // parent, and so must not be counted for its own key, which may change the
    // decision for that key, and so forth. Iterate to a fixed point (or just
    // stop after a while, as any decision is safe, just not optimal).
    for (Index i = 0; i < 10; i++) {
      bool changed = false;
      for (auto& [key, info] : keys) {
        if (info.sites.empty()) {
          continue;
        }
        Index count = 0;
        for (auto* site : info.sites) {
          if (isCandidate(site)) {
            count++;
          }
        }
        bool outline;
        if (info.helper) {
          // The helper already exists, so any use of it is a win.
          outline = count > 0;
        } else {
          auto* site = info.sites[0];
          int64_t saved = int64_t(count) * int64_t(site->bytes - CallBytes);
          int64_t cost = int64_t(site->bytes) + HelperOverheadBytes +
                         LocalGetBytes * int64_t(site->holes.size());
          outline = saved > cost;
        }
        if (outline != info.outline) {
          info.outline = outline;
          changed = true;
        }
      }
      if (!changed) {
        break;
      }
    }

    // Create the helpers and gather the replacements.
    Replacements replacements;
    std::vector<Function*> newHelpers;
    for (auto& [key, info] : keys) {
      if (!info.outline) {
        continue;
      }
      for (auto* site : info.sites) {
        if (!isReplaced(site)) {
          continue;
        }
        if (!info.helper) {
          info.helper = createHelper(site);
          newHelpers.push_back(info.helper);
        } else if (!canCall(info.helper, site)) {
          // An existing helper's parameter types may have been refined so that
          // this site cannot use it.
          continue;
        }
        replacements[site->expr] = Replacement{info.helper->name, site};
      }
    }
    if (replacements.empty()) {
      return;
    }

    placeHelpers(newHelpers, analysis.map, replacements);

    Replacer(replacements).run(getPassRunner(), wasm);
  }

  // Whether a site is eligible to be outlined, given the current decisions.
  bool isEligible(Site* site) {
    if (site->skip || site->helperBody) {
      return false;
    }
    auto iter = keys.find(site->key);
    return iter != keys.end() && iter->second.outline;
  }

  // Whether a site could be replaced, given the current decisions for the
  // other keys: it is eligible, and no enclosing allocation that contains it as
  // a constant will be replaced instead (which would make it vanish).
  bool isCandidate(Site* site) {
    if (site->skip || site->helperBody) {
      return false;
    }
    for (auto* parent = site->parent; parent; parent = parent->parent) {
      if (isEligible(parent)) {
        return false;
      }
    }
    return true;
  }

  // Whether a site will be replaced, given the current decisions.
  bool isReplaced(Site* site) {
    return isEligible(site) && isCandidate(site);
  }

  // If the function's body is just an allocation whose holes are its params, in
  // order, note it as the helper for the allocation's key.
  void noteExistingHelper(Function* func, Site& site) {
    if (site.key.empty() || site.skip ||
        site.holes.size() != func->getNumParams() ||
        func->getResults() != site.expr->type) {
      return;
    }
    auto& operands = getOperands(site.expr);
    for (Index i = 0; i < site.holes.size(); i++) {
      auto* get = operands[site.holes[i]]->dynCast<LocalGet>();
      if (!get || get->index != i) {
        return;
      }
    }
    site.helperBody = true;
    auto& info = keys[site.key];
    if (!info.helper) {
      info.helper = func;
    }
  }

  bool canCall(Function* helper, Site* site) {
    auto params = helper->getParams();
    auto& operands = getOperands(site->expr);
    for (Index i = 0; i < site->holes.size(); i++) {
      if (!Type::isSubType(operands[site->holes[i]]->type, params[i])) {
        return false;
      }
    }
    return true;
  }

  Function* createHelper(Site* site) {
    Builder builder(*wasm);
    auto* expr = site->expr;
    auto heapType = expr->type.getHeapType();
    auto& operands = getOperands(expr);
    std::vector<Type> params;
    std::vector<Expression*> newOperands;
    Index hole = 0;
    for (Index i = 0; i < operands.size(); i++) {
      if (hole < site->holes.size() && site->holes[hole] == i) {
        auto type = getOperandType(expr, i);
        newOperands.push_back(builder.makeLocalGet(hole, type));
        params.push_back(type);
        hole++;
      } else {
        newOperands.push_back(
          ExpressionManipulator::copy(operands[i], *wasm));
      }
    }
    Expression* body;
    if (expr->is<StructNew>()) {
      body = builder.makeStructNew(heapType, newOperands);
    } else {
      body = builder.makeArrayNewFixed(heapType, newOperands);
    }
    std::string root = "alloc";
    if (auto iter = wasm->typeNames.find(heapType);
        iter != wasm->typeNames.end() && iter->second.name.is()) {
      root += '$' + iter->second.name.toString();
    }
    auto name = Names::getValidFunctionName(*wasm, root);
    return wasm->addFunction(Builder::makeFunction(
      name, Signature(Type(params), expr->type), {}, body));
  }

  // The helpers are called many times, so it is worth giving them small
  // function indexes, if that does not cost more in the references to the
  // functions they push to larger indexes.
  void placeHelpers(
    const std::vector<Function*>& newHelpers,
    ModuleUtils::ParallelFunctionAnalysis<FunctionInfo>::Map& infos,
    const Replacements& replacements) {
    auto numNew = newHelpers.size();
    if (numNew == 0) {
      return;
    }
    std::unordered_map<Name, Index> funcRefs;
    for (auto& [_, info] : infos) {
      for (auto& [name, count] : info.funcRefs) {
        funcRefs[name] += count;
      }
    }
    for (auto& [_, replacement] : replacements) {
      funcRefs[replacement.helper]++;
    }
    auto noteRefs = [&](Expression* curr) {
      for (auto* ref : FindAll<RefFunc>(curr).list) {
        funcRefs[ref->func]++;
      }
    };
    for (auto& global : wasm->globals) {
      if (!global->imported()) {
        noteRefs(global->init);
      }
    }
    for (auto& segment : wasm->elementSegments) {
      for (auto* item : segment->data) {
        noteRefs(item);
      }
    }
    // The new helpers are currently at the end. Count the indexes as they are
    // now (imports come first), and see what moving the helpers to the front
    // of the defined functions would gain and cost.
    Index numImports = 0;
    std::vector<Function*> defined;
    for (auto& func : wasm->functions) {
      if (func->imported()) {
        numImports++;
      } else {
        defined.push_back(func.get());
      }
    }
    Index numOld = defined.size() - numNew;
    int64_t gain = 0;
    int64_t cost = 0;
    for (Index i = 0; i < defined.size(); i++) {
      int64_t refs = funcRefs[defined[i]->name];
      int64_t index = numImports + i;
      if (i >= numOld) {
        // A new helper, which would move to the front.
        int64_t newIndex = numImports + (i - numOld);
        gain += refs * (getULEBSize(index) - getULEBSize(newIndex));
      } else {
        cost += refs * (getULEBSize(index + numNew) - getULEBSize(index));
      }
    }
    if (gain <= cost) {
      return;
    }
    auto& functions = wasm->functions;
    std::rotate(functions.begin(), functions.end() - numNew, functions.end());
    // The map from names to functions is unaffected by the reordering.
  }
};

} // anonymous namespace

Pass* createOutlineAllocationsPass() { return new OutlineAllocations(); }

} // namespace wasm
