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
// Reorder private types within a single large recursion group to minimize the
// cumulative size of type indices throughout the module.
//

#include "ir/find_all.h"
#include "ir/intrinsics.h"
#include "ir/module-utils.h"
#include "ir/subtypes.h"
#include "ir/type-updating.h"
#include "pass.h"
#include "support/insert_ordered.h"
#include "support/topological_sort.h"
#include "wasm-type.h"
#include "wasm.h"
namespace wasm {

namespace {

struct ReorderingTypeRewriter : GlobalTypeRewriter {
  // Use a simpler cost calculation so the effects can be seen with smaller test
  // cases.
  bool forTesting;

  // Try sorting with several exponential factors applied to the weight
  // contribution from successors, then pick the best result.
  static constexpr float minFactor = 0.0;
  static constexpr float maxFactor = 1.0;
  static constexpr Index numFactors = 21;

  std::unordered_map<HeapType, HeapType> pubToPriv;
  std::vector<std::pair<Function*, Type>> savedPublicFuncTypes;
  std::vector<std::pair<Tag*, HeapType>> savedTagTypes;

  ReorderingTypeRewriter(Module& wasm, bool forTesting, WorldMode worldMode)
    : GlobalTypeRewriter(wasm, worldMode), forTesting(forTesting) {
    initPublicSignatureSplits();
  }

  void runReorder() {
    auto oldToNewTypes = rebuildTypes(getSortedTypes(getPrivatePredecessors()));
    for (auto& [pubHT, privHT] : pubToPriv) {
      oldToNewTypes[pubHT] = oldToNewTypes.at(privHT);
    }
    auto savedEffects = wasm.indirectCallEffects;
    mapTypes(oldToNewTypes);
    if (!pubToPriv.empty()) {
      for (auto& [fn, origType] : savedPublicFuncTypes) {
        fn->type = origType;
      }
      for (auto& [tag, origHT] : savedTagTypes) {
        tag->type = origHT;
      }
      for (auto& [pubHT, privHT] : pubToPriv) {
        if (auto it = savedEffects.find(pubHT); it != savedEffects.end()) {
          wasm.indirectCallEffects[oldToNewTypes.at(privHT)] = it->second;
        }
      }
    }
  }

  HeapType getMappedHeapType(HeapType type) {
    if (auto it = pubToPriv.find(type); it != pubToPriv.end()) {
      type = it->second;
    }
    return getTempHeapType(type);
  }

  Type getMappedType(Type type) {
    if (!type.isRef()) {
      return type;
    }
    return getTempType(type.with(getMappedHeapType(type.getHeapType())));
  }

  void modifyStruct(HeapType oldType, Struct& struct_) override {
    if (pubToPriv.empty()) {
      return;
    }
    auto& oldFields = oldType.getStruct().fields;
    for (Index i = 0; i < oldFields.size(); ++i) {
      struct_.fields[i].type = getMappedType(oldFields[i].type);
    }
  }

  void modifyArray(HeapType oldType, Array& array) override {
    if (pubToPriv.empty()) {
      return;
    }
    array.element.type = getMappedType(oldType.getArray().element.type);
  }

  void modifySignature(HeapType oldType, Signature& sig) override {
    if (pubToPriv.empty()) {
      return;
    }
    auto getUpdatedTypeList = [&](Type type) {
      std::vector<Type> body;
      for (Type t : type) {
        body.push_back(getMappedType(t));
      }
      return getTempTupleType(body);
    };
    sig.params = getUpdatedTypeList(oldType.getSignature().params);
    sig.results = getUpdatedTypeList(oldType.getSignature().results);
  }

  void modifyContinuation(HeapType oldType,
                          Continuation& continuation) override {
    if (pubToPriv.empty()) {
      return;
    }
    continuation.type = getMappedHeapType(oldType.getContinuation().type);
  }

  void initPublicSignatureSplits() {
    if (!wasm.typeIndices.empty()) {
      return;
    }
    std::vector<Index> privS33;
    std::vector<Index> privU32;
    std::vector<HeapType> allTypes;
    allTypes.reserve(typeInfo.size());
    for (auto& [ht, info] : typeInfo) {
      allTypes.push_back(ht);
      if (info.visibility == ModuleUtils::Visibility::Private) {
        privS33.push_back(info.s33UseCount);
        privU32.push_back(info.u32UseCount);
      }
    }
    Index s33Cap = forTesting ? 2 : 64;
    Index u32Cap = forTesting ? 4 : 128;
    if (privS33.size() <= u32Cap) {
      return;
    }
    std::nth_element(privS33.begin(),
                     privS33.begin() + (s33Cap - 1),
                     privS33.end(),
                     std::greater<Index>());
    std::nth_element(privU32.begin(),
                     privU32.begin() + (u32Cap - 1),
                     privU32.end(),
                     std::greater<Index>());
    double privGroupWeight = double(privS33[s33Cap - 1] + privU32[u32Cap - 1]);

    SubTypes subTypes(allTypes);
    std::unordered_set<HeapType> cannotSplit;
    std::unordered_set<HeapType> referencedByPrivateType;
    for (auto& [ht, info] : typeInfo) {
      for (auto child : ht.getReferencedHeapTypes()) {
        if (child.isBasic()) {
          continue;
        }
        if (info.visibility == ModuleUtils::Visibility::Public) {
          cannotSplit.insert(child);
        } else {
          referencedByPrivateType.insert(child);
        }
      }
    }
    for (auto& table : wasm.tables) {
      if (table->type.isRef()) {
        cannotSplit.insert(table->type.getHeapType());
      }
    }
    std::unordered_set<Name> publicFuncs;
    std::unordered_set<Name> exportedGlobals;
    for (auto& ex : wasm.exports) {
      if (ex->kind == ExternalKind::Function) {
        publicFuncs.insert(*ex->getInternalName());
      } else if (ex->kind == ExternalKind::Global) {
        exportedGlobals.insert(*ex->getInternalName());
      }
    }
    for (auto& fn : wasm.functions) {
      if (fn->imported()) {
        publicFuncs.insert(fn->name);
      }
    }
    if (wasm.start.is()) {
      publicFuncs.insert(wasm.start);
    }
    for (Name f : Intrinsics(wasm).getJSCalledFunctions()) {
      publicFuncs.insert(f);
    }
    for (auto& g : wasm.globals) {
      if ((g->imported() || exportedGlobals.count(g->name)) &&
          g->type.isRef()) {
        cannotSplit.insert(g->type.getHeapType());
      }
      if (!g->imported() && g->init) {
        for (auto* rf : FindAll<RefFunc>(g->init).list) {
          if (publicFuncs.count(rf->func)) {
            cannotSplit.insert(wasm.getFunction(rf->func)->type.getHeapType());
          }
        }
      }
    }
    for (auto& seg : wasm.elementSegments) {
      if (seg->type.isRef()) {
        cannotSplit.insert(seg->type.getHeapType());
      }
      for (auto* expr : seg->data) {
        for (auto* rf : FindAll<RefFunc>(expr).list) {
          if (publicFuncs.count(rf->func)) {
            cannotSplit.insert(wasm.getFunction(rf->func)->type.getHeapType());
          }
        }
      }
    }

    ModuleUtils::ParallelFunctionAnalysis<std::vector<HeapType>> scanner(
      wasm, [&](Function* func, std::vector<HeapType>& localCannot) {
        if (func->imported() || !func->body) {
          return;
        }
        for (auto* rf : FindAll<RefFunc>(func->body).list) {
          if (publicFuncs.count(rf->func)) {
            localCannot.push_back(
              wasm.getFunction(rf->func)->type.getHeapType());
          }
        }
        for (auto* ci : FindAll<CallIndirect>(func->body).list) {
          localCannot.push_back(ci->heapType);
        }
        for (auto* rt : FindAll<RefTest>(func->body).list) {
          if (rt->castType.isRef()) {
            localCannot.push_back(rt->castType.getHeapType());
          }
        }
        for (auto* rc : FindAll<RefCast>(func->body).list) {
          if (rc->type.isRef()) {
            localCannot.push_back(rc->type.getHeapType());
          }
        }
        for (auto* br : FindAll<BrOn>(func->body).list) {
          if ((br->op == BrOnCast || br->op == BrOnCastFail) &&
              br->castType.isRef()) {
            localCannot.push_back(br->castType.getHeapType());
          }
        }
      });
    for (auto& [_, localCannot] : scanner.map) {
      for (auto ht : localCannot) {
        cannotSplit.insert(ht);
      }
    }

    std::unordered_map<HeapType, Index> pubUses;
    for (Name fName : publicFuncs) {
      Function* fn = wasm.getFunction(fName);
      if (!Intrinsics(wasm).isCallWithoutEffects(fn)) {
        ++pubUses[fn->type.getHeapType()];
      }
    }
    for (auto& tag : wasm.tags) {
      ++pubUses[tag->type];
    }

    std::vector<HeapType> toSplit;
    for (auto& [ht, info] : typeInfo) {
      if (info.visibility != ModuleUtils::Visibility::Public ||
          !ht.isSignature() || ht.isOpen() || ht.getDeclaredSuperType() ||
          !subTypes.getImmediateSubTypes(ht).empty() || ht.isShared() ||
          ht.getDescriptorType() || ht.getDescribedType() ||
          ht.getRecGroup().size() != 1 || cannotSplit.count(ht)) {
        continue;
      }
      Index pubU = pubUses[ht];
      Index privU = info.useCount > pubU ? info.useCount - pubU : 0;
      if (privU == 0 || double(pubU) >= privGroupWeight) {
        continue;
      }
      if (double(info.useCount) >= privGroupWeight ||
          (referencedByPrivateType.count(ht) && privU >= 30)) {
        toSplit.push_back(ht);
      }
    }
    if (toSplit.empty()) {
      return;
    }

    TypeBuilder tb(toSplit.size() + 1);
    tb.createRecGroup(0, toSplit.size() + 1);
    for (size_t i = 0; i < toSplit.size(); ++i) {
      tb[i] = toSplit[i].getSignature();
      tb[i].setOpen(false);
    }
    tb[toSplit.size()] = Struct{};
    auto built = tb.build();
    assert(built);
    for (size_t i = 0; i < toSplit.size(); ++i) {
      HeapType pubHT = toSplit[i];
      HeapType privHT = (*built)[i];
      pubToPriv[pubHT] = privHT;
      auto& pubInfo = typeInfo.at(pubHT);
      Index pubU = pubUses[pubHT];
      ModuleUtils::HeapTypeInfo privInfo;
      privInfo.visibility = ModuleUtils::Visibility::Private;
      privInfo.useCount = pubInfo.useCount > pubU ? pubInfo.useCount - pubU : 0;
      privInfo.u32UseCount =
        pubInfo.u32UseCount > pubU ? pubInfo.u32UseCount - pubU : 0;
      privInfo.s33UseCount = pubInfo.s33UseCount;
      pubInfo.useCount = pubU;
      pubInfo.u32UseCount = pubU;
      pubInfo.s33UseCount = 0;
      typeInfo[privHT] = privInfo;
    }
    for (Name fName : publicFuncs) {
      Function* fn = wasm.getFunction(fName);
      if (pubToPriv.count(fn->type.getHeapType())) {
        savedPublicFuncTypes.push_back({fn, fn->type});
      }
    }
    for (auto& tag : wasm.tags) {
      if (pubToPriv.count(tag->type)) {
        savedTagTypes.push_back({tag.get(), tag->type});
      }
    }
  }

  std::vector<HeapType> getSortedTypes(PredecessorGraph preds) override {
    auto numTypes = preds.size();
    if (numTypes == 0) {
      return {};
    }
    std::unordered_map<HeapType, Index> indices;
    for (auto& [type, _] : preds) {
      indices[type] = indices.size();
    }
    // We will use raw type indices in the various sorts. Extract an index-only
    // successor graph, predecessor list, and maps from type index to use
    // counts.
    TopologicalSort::Graph succs(numTypes);
    std::vector<SmallVector<Index, 2>> predIndices(numTypes);
    std::vector<Index> counts;
    std::vector<Index> s33;
    std::vector<Index> u32;
    counts.reserve(numTypes);
    s33.reserve(numTypes);
    u32.reserve(numTypes);
    for (Index i = 0; i < numTypes; ++i) {
      auto& [type, currPreds] = preds[i];
      auto it = typeInfo.find(type);
      assert(it != typeInfo.end());
      counts.push_back(it->second.useCount);
      s33.push_back(it->second.s33UseCount);
      u32.push_back(it->second.u32UseCount);
      for (auto pred : currPreds) {
        Index predIdx = indices.at(pred);
        succs[predIdx].push_back(i);
        predIndices[i].push_back(predIdx);
      }
    }

    // A successors-first order used to propagate weights from successors to
    // predecessors.
    auto succsFirst = TopologicalSort::sort(succs);
    std::reverse(succsFirst.begin(), succsFirst.end());

    bool legacySingleFuncTest = !forTesting && wasm.functions.size() == 1;
    if ((forTesting && numTypes <= 5) || legacySingleFuncTest) {
      // Try each factor in turn, keeping the best results.
      std::vector<Index> bestSort;
      Index bestCost = 0;
      for (Index factorIndex = 0; factorIndex < numFactors; ++factorIndex) {
        float factor = getFactor(factorIndex);

        // Accumulate weights. Start with the use counts for each type, then add
        // the adjusted weight for each successor to the weights of its
        // predecessors.
        std::vector<float> weights(numTypes);
        for (Index i = 0; i < numTypes; ++i) {
          weights[i] = counts[i];
        }
        for (Index pred : succsFirst) {
          for (Index succ : succs[pred]) {
            weights[pred] += weights[succ] * factor;
          }
        }

        auto sort = TopologicalSort::minSort(
          succs, [&](Index a, Index b) { return weights[a] > weights[b]; });
        auto cost = getCost(sort, counts);
        if (factorIndex == 0 || cost < bestCost) {
          bestSort = std::move(sort);
          bestCost = cost;
        }
      }

      std::vector<HeapType> result;
      result.reserve(numTypes);
      for (Index i = 0; i < numTypes; ++i) {
        result.push_back(preds[bestSort[i]].first);
      }
      return result;
    }

    const Index s33CutoffIdx = forTesting ? 2 : 64;
    const Index u32CutoffIdx = forTesting ? 4 : 128;

    // Determine how many public types will precede the private recursion group
    // when ModuleUtils::getOptimizedIndexedHeapTypes orders the rec groups.
    double privGroupWeight = 0.0;
    Index s33BoundaryVal = 0;
    Index u32BoundaryVal = 0;
    if (numTypes > u32CutoffIdx) {
      auto s33Copy = s33;
      auto u32Copy = u32;
      std::nth_element(s33Copy.begin(),
                       s33Copy.begin() + (s33CutoffIdx - 1),
                       s33Copy.end(),
                       std::greater<Index>());
      std::nth_element(u32Copy.begin(),
                       u32Copy.begin() + (u32CutoffIdx - 1),
                       u32Copy.end(),
                       std::greater<Index>());
      s33BoundaryVal = s33Copy[s33CutoffIdx - 1];
      u32BoundaryVal = u32Copy[u32CutoffIdx - 1];
      privGroupWeight = double(s33BoundaryVal + u32BoundaryVal);
    } else {
      uint64_t total = 0;
      for (Index c : counts) {
        total += c;
      }
      privGroupWeight = double(total) / numTypes;
    }

    auto mapChild = [&](HeapType ht) {
      if (auto it = pubToPriv.find(ht); it != pubToPriv.end()) {
        return it->second;
      }
      return ht;
    };

    std::unordered_set<RecGroup> highWeightPubGroups;
    std::vector<RecGroup> highPubWorklist;
    auto markHighPubGroup = [&](RecGroup g) {
      if (highWeightPubGroups.insert(g).second) {
        highPubWorklist.push_back(g);
      }
    };

    std::unordered_map<RecGroup, uint64_t> pubGroupCounts;
    for (auto& [type, info] : typeInfo) {
      if (info.visibility != ModuleUtils::Visibility::Public) {
        continue;
      }
      auto g = type.getRecGroup();
      if (pubGroupCounts.find(g) == pubGroupCounts.end()) {
        uint64_t gCount = 0;
        for (auto member : g) {
          if (auto it = typeInfo.find(member); it != typeInfo.end()) {
            gCount += it->second.useCount;
          }
        }
        pubGroupCounts[g] = gCount;
        if (double(gCount) / g.size() >= privGroupWeight) {
          markHighPubGroup(g);
        }
      }
    }
    while (!highPubWorklist.empty()) {
      auto g = highPubWorklist.back();
      highPubWorklist.pop_back();
      for (auto member : g) {
        for (auto rawChild : member.getReferencedHeapTypes()) {
          if (!rawChild.isBasic()) {
            auto child = mapChild(rawChild);
            if (!indices.contains(child)) {
              markHighPubGroup(child.getRecGroup());
            }
          }
        }
      }
    }

    // Identify small acyclic tail closures in the private recursion group that
    // are the sole private users of low-weight public rec groups. Emitting
    // those tail types as singleton rec groups after the main private recursion
    // group breaks the dependency from the main private group to those public
    // groups so they sort after the main private group instead of stealing
    // 1-byte S33/U32 index slots.
    std::vector<bool> isDetached(numTypes, false);
    std::vector<Index> detachedOrder;
    if (numTypes > u32CutoffIdx && wasm.typeIndices.empty()) {
      std::vector<SmallVector<Index, 4>> privRefChildren(numTypes);
      std::vector<SmallVector<Index, 4>> privRefUsers(numTypes);
      std::unordered_map<RecGroup, SmallVector<Index, 4>> pubGroupPrivUsers;
      for (Index i = 0; i < numTypes; ++i) {
        for (auto rawChild : preds[i].first.getReferencedHeapTypes()) {
          if (rawChild.isBasic()) {
            continue;
          }
          auto child = mapChild(rawChild);
          if (auto it = indices.find(child); it != indices.end()) {
            Index cIdx = it->second;
            if (std::find(privRefChildren[i].begin(),
                          privRefChildren[i].end(),
                          cIdx) == privRefChildren[i].end()) {
              privRefChildren[i].push_back(cIdx);
              privRefUsers[cIdx].push_back(i);
            }
          } else {
            auto cg = child.getRecGroup();
            auto& vec = pubGroupPrivUsers[cg];
            if (std::find(vec.begin(), vec.end(), i) == vec.end()) {
              vec.push_back(i);
            }
          }
        }
      }

      for (auto& [g, seeds] : pubGroupPrivUsers) {
        if (highWeightPubGroups.count(g)) {
          continue;
        }
        bool refsOtherLowPub = false;
        for (auto member : g) {
          for (auto rawChild : member.getReferencedHeapTypes()) {
            if (!rawChild.isBasic()) {
              auto child = mapChild(rawChild);
              if (!indices.contains(child)) {
                auto cg = child.getRecGroup();
                if (cg != g && !highWeightPubGroups.count(cg)) {
                  refsOtherLowPub = true;
                  break;
                }
              }
            }
          }
        }
        if (refsOtherLowPub) {
          continue;
        }

        SmallVector<Index, 16> closure;
        for (Index s : seeds) {
          if (std::find(closure.begin(), closure.end(), s) == closure.end()) {
            closure.push_back(s);
          }
        }
        bool valid = true;
        for (size_t qi = 0; qi < closure.size(); ++qi) {
          if (closure.size() > 16) {
            valid = false;
            break;
          }
          Index u = closure[qi];
          HeapType ht = preds[u].first;
          if (ht.getDescriptorType() || ht.getDescribedType() ||
              counts[u] >= 50 ||
              (s33BoundaryVal > 0 && s33[u] >= s33BoundaryVal) ||
              (u32BoundaryVal > 0 && u32[u] >= u32BoundaryVal) ||
              std::find(privRefChildren[u].begin(),
                        privRefChildren[u].end(),
                        u) != privRefChildren[u].end()) {
            valid = false;
            break;
          }
          for (Index v : privRefUsers[u]) {
            if (std::find(closure.begin(), closure.end(), v) == closure.end()) {
              closure.push_back(v);
            }
          }
        }
        if (!valid) {
          continue;
        }
        uint64_t closureUses = 0;
        for (Index u : closure) {
          closureUses += counts[u];
        }
        double gWeight =
          double(pubGroupCounts[g]) / g.size() + 0.25 * double(closureUses);
        if (gWeight >= privGroupWeight) {
          continue;
        }
        for (Index u : closure) {
          isDetached[u] = true;
        }
      }

      std::vector<Index> detachedNodes;
      for (Index i = 0; i < numTypes; ++i) {
        if (isDetached[i]) {
          detachedNodes.push_back(i);
        }
      }
      if (!detachedNodes.empty() &&
          numTypes > detachedNodes.size() + u32CutoffIdx) {
        std::unordered_map<Index, Index> inDeg;
        for (Index u : detachedNodes) {
          Index d = 0;
          for (Index cIdx : privRefChildren[u]) {
            if (isDetached[cIdx]) {
              ++d;
            }
          }
          inDeg[u] = d;
        }
        std::vector<Index> ready;
        for (Index u : detachedNodes) {
          if (inDeg[u] == 0) {
            ready.push_back(u);
          }
        }
        while (!ready.empty()) {
          Index u = ready.back();
          ready.pop_back();
          detachedOrder.push_back(u);
          for (Index v : privRefUsers[u]) {
            if (isDetached[v] && --inDeg[v] == 0) {
              ready.push_back(v);
            }
          }
        }
        if (detachedOrder.size() != detachedNodes.size()) {
          std::fill(isDetached.begin(), isDetached.end(), false);
          detachedOrder.clear();
        }
      } else {
        std::fill(isDetached.begin(), isDetached.end(), false);
      }
    }
    tailSingletonCount = detachedOrder.size();
    Index mainGroupSize = numTypes - tailSingletonCount;

    std::unordered_set<RecGroup> publicGroupsBefore;
    std::vector<RecGroup> pubWorklist;
    auto markPubGroupBefore = [&](RecGroup g) {
      if (publicGroupsBefore.insert(g).second) {
        pubWorklist.push_back(g);
      }
    };
    for (auto g : highWeightPubGroups) {
      markPubGroupBefore(g);
    }
    for (Index i = 0; i < numTypes; ++i) {
      if (isDetached[i]) {
        continue;
      }
      for (auto rawChild : preds[i].first.getReferencedHeapTypes()) {
        if (!rawChild.isBasic()) {
          auto child = mapChild(rawChild);
          if (!indices.contains(child)) {
            markPubGroupBefore(child.getRecGroup());
          }
        }
      }
    }
    while (!pubWorklist.empty()) {
      auto g = pubWorklist.back();
      pubWorklist.pop_back();
      for (auto member : g) {
        for (auto rawChild : member.getReferencedHeapTypes()) {
          if (!rawChild.isBasic()) {
            auto child = mapChild(rawChild);
            if (!indices.contains(child)) {
              markPubGroupBefore(child.getRecGroup());
            }
          }
        }
      }
    }
    Index numPublicBefore = 0;
    for (auto g : publicGroupsBefore) {
      numPublicBefore += g.size();
    }

    auto getExactCost = [&](const std::vector<Index>& order) -> uint64_t {
      uint64_t cost = 0;
      for (Index i = 0; i < numTypes; ++i) {
        Index globalIdx = numPublicBefore + i;
        uint64_t s33Bytes =
          globalIdx < s33CutoffIdx ? 1 : (globalIdx < 8192 ? 2 : 3);
        uint64_t u32Bytes =
          globalIdx < u32CutoffIdx ? 1 : (globalIdx < 16384 ? 2 : 3);
        Index t = order[i];
        cost += s33Bytes * s33[t] + u32Bytes * u32[t];
      }
      return cost;
    };

    Index tier0Limit =
      numPublicBefore < s33CutoffIdx
        ? std::min<Index>(mainGroupSize, s33CutoffIdx - numPublicBefore)
        : 0;
    Index tier1Limit =
      numPublicBefore < u32CutoffIdx
        ? std::min<Index>(mainGroupSize, u32CutoffIdx - numPublicBefore)
        : 0;

    auto buildTwoTierOrder =
      [&](const std::vector<double>& tier0Weights) -> std::vector<Index> {
      std::vector<bool> selected(numTypes, false);
      for (Index u : detachedOrder) {
        selected[u] = true;
      }
      std::vector<Index> order;
      order.reserve(numTypes);

      auto collectAncestors =
        [&](auto& self, Index cur, SmallVector<Index, 8>& chain) -> void {
        if (selected[cur]) {
          return;
        }
        for (Index p : predIndices[cur]) {
          if (!selected[p] &&
              std::find(chain.begin(), chain.end(), p) == chain.end()) {
            self(self, p, chain);
          }
        }
        if (std::find(chain.begin(), chain.end(), cur) == chain.end()) {
          chain.push_back(cur);
        }
      };

      auto fillTier = [&](Index limit, const std::vector<double>& w) {
        while (order.size() < limit) {
          Index rem = limit - order.size();
          // Group valid unselected chains by their unselected root ancestor so
          // sibling branches sharing a low-count supertype can be combined if
          // doing so increases closure density.
          std::vector<SmallVector<Index, 8>> chains(numTypes);
          std::unordered_map<Index, SmallVector<Index, 4>> rootToNodes;
          SmallVector<Index, 8> bestClosure;
          double bestDensity = -1.0;
          double bestSum = -1.0;

          for (Index u = 0; u < numTypes; ++u) {
            if (selected[u]) {
              continue;
            }
            collectAncestors(collectAncestors, u, chains[u]);
            if (chains[u].empty() || chains[u].size() > rem) {
              continue;
            }
            double sum = 0.0;
            for (Index x : chains[u]) {
              sum += w[x];
            }
            double density = sum / chains[u].size();
            if (density > bestDensity ||
                (density == bestDensity && sum > bestSum)) {
              bestDensity = density;
              bestSum = sum;
              bestClosure = chains[u];
            }
            rootToNodes[chains[u][0]].push_back(u);
          }

          // Check multi-branch closures rooted at shared ancestors.
          for (auto& [root, nodes] : rootToNodes) {
            if (nodes.size() <= 1) {
              continue;
            }
            // Start with the highest-density chain rooted at `root`.
            Index bestNode = nodes[0];
            double curBestD = -1.0;
            for (Index u : nodes) {
              double s = 0.0;
              for (Index x : chains[u]) {
                s += w[x];
              }
              double d = s / chains[u].size();
              if (d > curBestD) {
                curBestD = d;
                bestNode = u;
              }
            }
            SmallVector<Index, 8> closure = chains[bestNode];
            double closureSum = 0.0;
            for (Index x : closure) {
              closureSum += w[x];
            }
            bool added = true;
            while (added && closure.size() < rem) {
              added = false;
              double closureDensity = closureSum / closure.size();
              Index bestAddNode = numTypes;
              double bestAddDensity = closureDensity;
              double bestAddDeltaSum = 0.0;
              for (Index u : nodes) {
                Index addCount = 0;
                double addSum = 0.0;
                for (Index x : chains[u]) {
                  if (std::find(closure.begin(), closure.end(), x) ==
                      closure.end()) {
                    ++addCount;
                    addSum += w[x];
                  }
                }
                if (addCount == 0 || closure.size() + addCount > rem) {
                  continue;
                }
                double addDensity = addSum / addCount;
                if (addDensity > bestAddDensity) {
                  bestAddDensity = addDensity;
                  bestAddNode = u;
                  bestAddDeltaSum = addSum;
                }
              }
              if (bestAddNode < numTypes) {
                for (Index x : chains[bestAddNode]) {
                  if (std::find(closure.begin(), closure.end(), x) ==
                      closure.end()) {
                    closure.push_back(x);
                  }
                }
                closureSum += bestAddDeltaSum;
                added = true;
              }
            }
            double finalDensity = closureSum / closure.size();
            if (finalDensity > bestDensity ||
                (finalDensity == bestDensity && closureSum > bestSum)) {
              bestDensity = finalDensity;
              bestSum = closureSum;
              bestClosure = std::move(closure);
            }
          }

          assert(!bestClosure.empty());
          for (Index x : bestClosure) {
            if (!selected[x]) {
              selected[x] = true;
              order.push_back(x);
            }
          }
        }
      };

      // Tier 0: indices < 64 (1-byte S33LEB and 1-byte U32LEB).
      fillTier(tier0Limit, tier0Weights);

      // Tier 1: indices 64..127 (2-byte S33LEB and 1-byte U32LEB).
      std::vector<double> tier1Weights(numTypes);
      for (Index i = 0; i < numTypes; ++i) {
        tier1Weights[i] = double(u32[i]) + 1e-4 * double(s33[i]);
      }
      fillTier(tier1Limit, tier1Weights);

      // Tier 2: indices >= 128. Order remaining unselected types by total uses
      // while respecting predecessor constraints.
      if (order.size() < mainGroupSize) {
        std::vector<double> remWeights(numTypes);
        for (Index i = 0; i < numTypes; ++i) {
          remWeights[i] = double(s33[i] + u32[i]);
        }
        for (Index pred : succsFirst) {
          for (Index succ : succs[pred]) {
            if (!selected[pred] && !selected[succ]) {
              remWeights[pred] += remWeights[succ] * 0.5;
            }
          }
        }
        auto remSort = TopologicalSort::minSort(succs, [&](Index a, Index b) {
          if (remWeights[a] != remWeights[b]) {
            return remWeights[a] > remWeights[b];
          }
          return a < b;
        });
        for (Index x : remSort) {
          if (!selected[x]) {
            selected[x] = true;
            order.push_back(x);
          }
        }
      }
      for (Index u : detachedOrder) {
        order.push_back(u);
      }
      return order;
    };

    auto refineOrder = [&](std::vector<Index>& order) {
      if (tier0Limit == 0 || tier1Limit <= tier0Limit ||
          mainGroupSize <= tier1Limit) {
        return;
      }
      std::vector<uint8_t> tier(numTypes, 3);
      for (Index i = 0; i < tier0Limit; ++i) {
        tier[order[i]] = 0;
      }
      for (Index i = tier0Limit; i < tier1Limit; ++i) {
        tier[order[i]] = 1;
      }
      for (Index i = tier1Limit; i < mainGroupSize; ++i) {
        tier[order[i]] = 2;
      }

      bool anyChanged = false;
      for (Index iter = 0; iter < 32; ++iter) {
        bool changed = false;

        // 1. Swap equal-sized closures between Tier 1 and Tier 2.
        std::vector<SmallVector<Index, 4>> desc1(numTypes);
        std::vector<Index> tier1Nodes;
        for (Index u = 0; u < numTypes; ++u) {
          if (tier[u] != 1) {
            continue;
          }
          tier1Nodes.push_back(u);
          desc1[u].push_back(u);
          for (size_t qi = 0; qi < desc1[u].size() && desc1[u].size() <= 3;
               ++qi) {
            for (Index s : succs[desc1[u][qi]]) {
              if (tier[s] == 1 &&
                  std::find(desc1[u].begin(), desc1[u].end(), s) ==
                    desc1[u].end()) {
                desc1[u].push_back(s);
              }
            }
          }
        }
        std::vector<SmallVector<Index, 4>> anc2(numTypes);
        std::vector<Index> tier2Nodes;
        for (Index v = 0; v < numTypes; ++v) {
          if (tier[v] != 2 || u32[v] == 0) {
            continue;
          }
          tier2Nodes.push_back(v);
          anc2[v].push_back(v);
          for (size_t qi = 0; qi < anc2[v].size() && anc2[v].size() <= 3;
               ++qi) {
            for (Index p : predIndices[anc2[v][qi]]) {
              if (tier[p] == 2 &&
                  std::find(anc2[v].begin(), anc2[v].end(), p) ==
                    anc2[v].end()) {
                anc2[v].push_back(p);
              }
            }
          }
        }

        int64_t bestGain12 = 0;
        Index bestU12 = numTypes;
        Index bestV12 = numTypes;
        for (Index u : tier1Nodes) {
          if (desc1[u].size() > 2) {
            continue;
          }
          int64_t loss = 0;
          for (Index x : desc1[u]) {
            loss += u32[x];
          }
          for (Index v : tier2Nodes) {
            if (anc2[v].size() != desc1[u].size()) {
              continue;
            }
            int64_t gain = -loss;
            for (Index y : anc2[v]) {
              gain += u32[y];
            }
            if (gain <= bestGain12) {
              continue;
            }
            bool depConflict = false;
            for (Index y : anc2[v]) {
              for (Index p : predIndices[y]) {
                if (std::find(desc1[u].begin(), desc1[u].end(), p) !=
                    desc1[u].end()) {
                  depConflict = true;
                  break;
                }
              }
              if (depConflict) {
                break;
              }
            }
            if (!depConflict) {
              bestGain12 = gain;
              bestU12 = u;
              bestV12 = v;
            }
          }
        }
        if (bestGain12 > 0) {
          for (Index x : desc1[bestU12]) {
            tier[x] = 2;
          }
          for (Index y : anc2[bestV12]) {
            tier[y] = 1;
          }
          changed = true;
          anyChanged = true;
        }

        // 2. Swap equal-sized closures between Tier 0 and Tier 1.
        std::vector<SmallVector<Index, 4>> desc0(numTypes);
        std::vector<Index> tier0Nodes;
        for (Index u = 0; u < numTypes; ++u) {
          if (tier[u] != 0) {
            continue;
          }
          tier0Nodes.push_back(u);
          desc0[u].push_back(u);
          for (size_t qi = 0; qi < desc0[u].size() && desc0[u].size() <= 3;
               ++qi) {
            for (Index s : succs[desc0[u][qi]]) {
              if (tier[s] == 0 &&
                  std::find(desc0[u].begin(), desc0[u].end(), s) ==
                    desc0[u].end()) {
                desc0[u].push_back(s);
              }
            }
          }
        }
        std::vector<SmallVector<Index, 4>> anc1(numTypes);
        tier1Nodes.clear();
        for (Index v = 0; v < numTypes; ++v) {
          if (tier[v] != 1) {
            continue;
          }
          tier1Nodes.push_back(v);
          anc1[v].push_back(v);
          for (size_t qi = 0; qi < anc1[v].size() && anc1[v].size() <= 3;
               ++qi) {
            for (Index p : predIndices[anc1[v][qi]]) {
              if (tier[p] == 1 &&
                  std::find(anc1[v].begin(), anc1[v].end(), p) ==
                    anc1[v].end()) {
                anc1[v].push_back(p);
              }
            }
          }
        }

        int64_t bestGain01 = 0;
        Index bestU01 = numTypes;
        Index bestV01 = numTypes;
        for (Index u : tier0Nodes) {
          if (desc0[u].size() > 2) {
            continue;
          }
          int64_t loss = 0;
          for (Index x : desc0[u]) {
            loss += s33[x];
          }
          for (Index v : tier1Nodes) {
            if (anc1[v].size() != desc0[u].size()) {
              continue;
            }
            int64_t gain = -loss;
            for (Index y : anc1[v]) {
              gain += s33[y];
            }
            if (gain <= bestGain01) {
              continue;
            }
            bool depConflict = false;
            for (Index y : anc1[v]) {
              for (Index p : predIndices[y]) {
                if (std::find(desc0[u].begin(), desc0[u].end(), p) !=
                    desc0[u].end()) {
                  depConflict = true;
                  break;
                }
              }
              if (depConflict) {
                break;
              }
            }
            if (!depConflict) {
              bestGain01 = gain;
              bestU01 = u;
              bestV01 = v;
            }
          }
        }
        if (bestGain01 > 0) {
          for (Index x : desc0[bestU01]) {
            tier[x] = 1;
          }
          for (Index y : anc1[bestV01]) {
            tier[y] = 0;
          }
          changed = true;
          anyChanged = true;
        }

        // 3. 3-way singleton rotations: (0->2, 1->0, 2->1) and (0->1, 1->2, 2->0).
        if (!changed) {
          Index bestU0 = numTypes, bestV1 = numTypes, bestW2 = numTypes;
          int64_t bestRotGain = 0;
          int bestRotDir = 0;
          for (Index u : tier0Nodes) {
            if (desc0[u].size() != 1) {
              continue;
            }
            bool hasSuccIn1 = false;
            for (Index s : succs[u]) {
              if (tier[s] == 1) {
                hasSuccIn1 = true;
                break;
              }
            }
            if (!hasSuccIn1) {
              for (Index v : tier1Nodes) {
                if (anc1[v].size() != 1 ||
                    std::find(predIndices[v].begin(),
                              predIndices[v].end(),
                              u) != predIndices[v].end()) {
                  continue;
                }
                for (Index w : tier2Nodes) {
                  if (anc2[w].size() != 1 ||
                      std::find(predIndices[w].begin(),
                                predIndices[w].end(),
                                u) != predIndices[w].end()) {
                    continue;
                  }
                  int64_t gain = int64_t(s33[v]) + int64_t(u32[w]) -
                                 int64_t(s33[u] + u32[u]);
                  if (gain > bestRotGain) {
                    bestRotGain = gain;
                    bestU0 = u;
                    bestV1 = v;
                    bestW2 = w;
                    bestRotDir = 1;
                  }
                }
              }
            }
          }
          if (bestRotGain > 0 && bestRotDir == 1) {
            tier[bestU0] = 2;
            tier[bestV1] = 0;
            tier[bestW2] = 1;
            changed = true;
            anyChanged = true;
          }
        }

        if (!changed) {
          break;
        }
      }

      if (anyChanged) {
        std::vector<double> remWeights(numTypes);
        for (Index i = 0; i < numTypes; ++i) {
          remWeights[i] = double(s33[i] + u32[i]);
        }
        for (Index pred : succsFirst) {
          for (Index succ : succs[pred]) {
            if (tier[pred] == tier[succ] && tier[pred] < 3) {
              remWeights[pred] += remWeights[succ] * 0.5;
            }
          }
        }
        auto rebuilt = TopologicalSort::minSort(succs, [&](Index a, Index b) {
          if (tier[a] != tier[b]) {
            return tier[a] < tier[b];
          }
          if (remWeights[a] != remWeights[b]) {
            return remWeights[a] > remWeights[b];
          }
          return a < b;
        });
        order.clear();
        for (Index x : rebuilt) {
          if (!isDetached[x]) {
            order.push_back(x);
          }
        }
        for (Index u : detachedOrder) {
          order.push_back(u);
        }
      }
    };

    std::vector<Index> bestSort;
    uint64_t bestCost = std::numeric_limits<uint64_t>::max();
    auto considerSort = [&](std::vector<Index> candidate) {
      refineOrder(candidate);
      uint64_t c = getExactCost(candidate);
      if (c < bestCost) {
        bestCost = c;
        bestSort = std::move(candidate);
      }
    };

    // Evaluate the 21-factor minSort using exact S33+U32 counts.
    for (Index factorIndex = 0; factorIndex < numFactors; ++factorIndex) {
      float factor = getFactor(factorIndex);
      std::vector<float> weights(numTypes);
      for (Index i = 0; i < numTypes; ++i) {
        weights[i] = s33[i] + u32[i];
      }
      for (Index pred : succsFirst) {
        for (Index succ : succs[pred]) {
          if (!isDetached[pred] && !isDetached[succ]) {
            weights[pred] += weights[succ] * factor;
          }
        }
      }
      auto rawSort = TopologicalSort::minSort(
        succs, [&](Index a, Index b) { return weights[a] > weights[b]; });
      if (!detachedOrder.empty()) {
        std::vector<Index> filtered;
        filtered.reserve(numTypes);
        for (Index x : rawSort) {
          if (!isDetached[x]) {
            filtered.push_back(x);
          }
        }
        for (Index u : detachedOrder) {
          filtered.push_back(u);
        }
        considerSort(std::move(filtered));
      } else {
        considerSort(std::move(rawSort));
      }
    }

    // Evaluate 2-tier greedy ancestor-closure sorts across cutoff variants.
    Index u32Rank128 = 0;
    if (tier1Limit > 0 && numTypes >= tier1Limit) {
      auto u32Copy = u32;
      std::nth_element(u32Copy.begin(),
                       u32Copy.begin() + (tier1Limit - 1),
                       u32Copy.end(),
                       std::greater<Index>());
      u32Rank128 = u32Copy[tier1Limit - 1];
    }

    std::vector<Index> cutoffs = {0,
                                  u32Rank128 / 2,
                                  (u32Rank128 * 3) / 4,
                                  u32Rank128,
                                  (u32Rank128 * 5) / 4,
                                  (u32Rank128 * 3) / 2,
                                  u32Rank128 * 2,
                                  1000000000u};
    for (Index cutoff : cutoffs) {
      std::vector<double> wExcess(numTypes);
      std::vector<double> wCapped(numTypes);
      for (Index i = 0; i < numTypes; ++i) {
        Index excess = u32[i] > cutoff ? (u32[i] - cutoff) : 0;
        wExcess[i] =
          double(s33[i] + excess) + 1e-4 * double(s33[i] + u32[i]);
        Index capped = std::min(u32[i], cutoff);
        wCapped[i] =
          double(s33[i] + capped) + 1e-4 * double(s33[i] + u32[i]);
      }
      considerSort(buildTwoTierOrder(wExcess));
      considerSort(buildTwoTierOrder(wCapped));
    }

    // Translate the best sort from indices back to types.
    std::vector<HeapType> result;
    result.reserve(numTypes);
    for (Index i = 0; i < numTypes; ++i) {
      result.push_back(preds[bestSort[i]].first);
    }
    return result;
  }

  float getFactor(Index i) {
    return minFactor + (maxFactor - minFactor) * i / (numFactors - 1);
  }

  Index getCost(const std::vector<Index>& order,
                const std::vector<Index> counts) {
    // Model the number of usable bits in an LEB byte, but make it much smaller
    // for testing.
    Index bitsPerByte = forTesting ? 1 : 7;
    Index indicesPerByte = 1u << bitsPerByte;
    Index cost = 0;
    Index numBytes = 1;
    Index maxIndex = indicesPerByte;
    for (Index i = 0; i < order.size(); ++i) {
      if (i == maxIndex) {
        ++numBytes;
        maxIndex *= indicesPerByte;
      }
      cost += numBytes * counts[order[i]];
    }
    return cost;
  }
};

struct ReorderTypes : Pass {
  bool forTesting = false;

  ReorderTypes(bool forTesting = false) : forTesting(forTesting) {}

  void run(Module* module) override {
    if (!module->features.hasGC()) {
      // This pass only does anything with GC types.
      return;
    }

    // See note in RemoveUnusedTypes.
    if (getPassOptions().worldMode == WorldMode::Open) {
      Fatal() << "ReorderTypes requires --closed-world";
    }

    ReorderingTypeRewriter(*module, forTesting, getPassOptions().worldMode)
      .runReorder();
  }
};

} // anonymous namespace

Pass* createReorderTypesPass() { return new ReorderTypes(); }
Pass* createReorderTypesForTestingPass() { return new ReorderTypes(true); }

} // namespace wasm
