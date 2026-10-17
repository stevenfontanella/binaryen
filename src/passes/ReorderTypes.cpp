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

#include "ir/module-utils.h"
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

  ReorderingTypeRewriter(Module& wasm, bool forTesting, WorldMode worldMode)
    : GlobalTypeRewriter(wasm, worldMode), forTesting(forTesting) {}

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
      privGroupWeight =
        double(s33Copy[s33CutoffIdx - 1] + u32Copy[u32CutoffIdx - 1]);
    } else {
      uint64_t total = 0;
      for (Index c : counts) {
        total += c;
      }
      privGroupWeight = double(total) / numTypes;
    }

    std::unordered_set<RecGroup> publicGroupsBefore;
    std::vector<RecGroup> pubWorklist;
    auto markPubGroupBefore = [&](RecGroup g) {
      if (publicGroupsBefore.insert(g).second) {
        pubWorklist.push_back(g);
      }
    };

    std::unordered_set<RecGroup> seenPubGroups;
    for (auto& [type, info] : typeInfo) {
      if (info.visibility != ModuleUtils::Visibility::Public) {
        continue;
      }
      auto g = type.getRecGroup();
      if (seenPubGroups.insert(g).second) {
        uint64_t gCount = 0;
        for (auto member : g) {
          if (auto it = typeInfo.find(member); it != typeInfo.end()) {
            gCount += it->second.useCount;
          }
        }
        if (double(gCount) / g.size() >= privGroupWeight) {
          markPubGroupBefore(g);
        }
      }
    }
    for (auto& [type, _] : preds) {
      for (auto child : type.getReferencedHeapTypes()) {
        if (!child.isBasic() && !indices.contains(child)) {
          markPubGroupBefore(child.getRecGroup());
        }
      }
    }
    while (!pubWorklist.empty()) {
      auto g = pubWorklist.back();
      pubWorklist.pop_back();
      for (auto member : g) {
        for (auto child : member.getReferencedHeapTypes()) {
          if (!child.isBasic() && !indices.contains(child)) {
            markPubGroupBefore(child.getRecGroup());
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
        ? std::min<Index>(numTypes, s33CutoffIdx - numPublicBefore)
        : 0;
    Index tier1Limit =
      numPublicBefore < u32CutoffIdx
        ? std::min<Index>(numTypes, u32CutoffIdx - numPublicBefore)
        : 0;

    auto buildTwoTierOrder =
      [&](const std::vector<double>& tier0Weights) -> std::vector<Index> {
      std::vector<bool> selected(numTypes, false);
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
      if (order.size() < numTypes) {
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
      return order;
    };

    std::vector<Index> bestSort;
    uint64_t bestCost = std::numeric_limits<uint64_t>::max();
    auto considerSort = [&](std::vector<Index> candidate) {
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
          weights[pred] += weights[succ] * factor;
        }
      }
      considerSort(TopologicalSort::minSort(
        succs, [&](Index a, Index b) { return weights[a] > weights[b]; }));
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

    std::vector<Index> cutoffs = {
      0, u32Rank128 / 2, u32Rank128, (u32Rank128 * 3) / 2, 1000000000u};
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
      .update();
  }
};

} // anonymous namespace

Pass* createReorderTypesPass() { return new ReorderTypes(); }
Pass* createReorderTypesForTestingPass() { return new ReorderTypes(true); }

} // namespace wasm
