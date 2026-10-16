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
// A small LZ77 + canonical Huffman coder for data stored in passive segments,
// shared by StringConstsToData (streams of UTF-16 code units) and
// CompressDataSegments (streams of bytes), with builders for the wasm
// decoders.
//
// The input is a stream of 16-bit units (bytes are units < 256). It is
// LZ77-compressed into a byte stream: a token byte holds a kind in the low 2
// bits and a length in the high 6 bits (63 means that one more byte follows
// and is added), and the kinds are
//
//   0: a run of narrow literals, 1 byte per unit (all units < 256)
//   1: a run of wide literals, 2 bytes per unit, little-endian
//   2: a match of length + 3 units, 1 byte distance - 1 follows
//   3: a match of length + 3 units, 2 bytes distance - 1 follow
//
// The bytes of the token stream are then entropy-coded: each byte belongs to
// one of eight classes (token, extra length, narrow literal, wide literal low
// and high byte, short distance, long distance low and high byte), and each
// class gets its own canonical Huffman code with code lengths of at most 15
// bits. The segment then holds, per used class, the number of codes of each
// length (15 bytes) followed by the symbols in canonical order, and after the
// tables one bit stream of the codes in token order (packed LSB-first, each
// code written from its most significant bit). The decoder reads a code bit
// by bit, as in zlib's puff.c, directly from the tables in the segment, so no
// decoding tables are built at runtime. When the bit stream plus the tables
// would not be smaller than the plain bytes, the plain bytes are kept.
//
// The LZ77 parse is optimal for a cost model: the first parse uses byte
// costs, and when the result is Huffman-coded the parse is repeated with the
// code lengths as bit costs (a few times, keeping the smallest result).
//

#ifndef wasm_passes_lz_huffman_h
#define wasm_passes_lz_huffman_h

#include <algorithm>
#include <array>
#include <optional>
#include <queue>
#include <vector>

#include "ir/names.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm::LZHuffman {

// The classes of bytes in the token stream, each with its own Huffman code.
enum Table : uint8_t {
  TokenTable,
  ExtraTable,
  NarrowTable,
  WideLoTable,
  WideHiTable,
  ShortDistTable,
  DistLoTable,
  DistHiTable,
  NumTables
};

// A byte of the token stream and its class.
using Symbol = std::pair<Table, uint8_t>;

static constexpr Index MaxCodeLength = 15;
static constexpr Index MinMatch = 3;

struct Segment {
  std::vector<char> data;
  // Whether the bytes are Huffman-coded.
  bool huffman = false;
  // The offset of the table of each class in the data (unused: 0).
  std::array<Index, NumTables> tableOffsets{};
  // The bit offset of the first code.
  Index firstBit = 0;
  // The code length of each symbol of each class (0: unused), when coded.
  std::array<std::vector<uint8_t>, NumTables> lengths;
};

struct Encoder {
  // The units to encode.
  std::vector<uint16_t> units;
  // Whether to find matches (otherwise only literal runs are emitted).
  bool compress = true;
  // 0: never Huffman-code, 1: always, 2: when smaller (by more than the
  // decoder's size).
  int huffman = 2;
  // The size of the symbol decoder, which the plain layout saves.
  Index decoderSize = 100;

  // The largest length a token can hold: 63 + one extra byte.
  static constexpr Index MaxTokenLength = 63 + 255;
  static constexpr Index MaxMatch = MaxTokenLength + MinMatch;
  static constexpr Index ShortDistance = 256;
  static constexpr Index MaxDistance = 65536;
  static constexpr Index HashBits = 17;
  static constexpr Index MaxChain = 1024;

  struct Match {
    Index length = 0;
    Index distance = 0;
  };

  // The match candidates kept per position: matches found walking the hash
  // chain from the nearest to the farthest position, each longer than the
  // previous one (so they are ordered by increasing distance and length).
  static constexpr Index MaxCandidates = 8;
  using Candidates = std::array<Match, MaxCandidates>;

  // Finds, for every position, the candidate matches.
  void findMatches(std::vector<Candidates>& matches) {
    Index n = units.size();
    matches.assign(n, Candidates{});
    if (n < MinMatch) {
      return;
    }
    std::vector<Index> head(Index(1) << HashBits, Index(-1));
    std::vector<Index> prev(n, Index(-1));
    auto hash = [&](Index i) {
      uint32_t h = units[i] * 0x9E3779B1u ^ units[i + 1] * 0x85EBCA77u ^
                   units[i + 2] * 0xC2B2AE3Du;
      return h >> (32 - HashBits);
    };
    for (Index i = 0; i + MinMatch <= n; i++) {
      auto h = hash(i);
      Index maxLength = std::min(MaxMatch, n - i);
      Index j = head[h];
      Index chain = 0;
      auto& candidates = matches[i];
      Index numCandidates = 0;
      Index bestLength = MinMatch - 1;
      while (j != Index(-1) && chain++ < MaxChain) {
        Index distance = i - j;
        if (distance > MaxDistance) {
          break;
        }
        Index length = 0;
        while (length < maxLength && units[j + length] == units[i + length]) {
          length++;
        }
        if (length > bestLength) {
          bestLength = length;
          // When the list is full, the longest match replaces the last one.
          if (numCandidates == MaxCandidates) {
            numCandidates--;
          }
          candidates[numCandidates++] = {length, distance};
          if (length == maxLength) {
            break;
          }
        }
        j = prev[j];
      }
      prev[i] = head[h];
      head[h] = i;
    }
  }

  // The cost of each symbol in the parse, in bits. The first parse uses 8 bits
  // per byte; later parses use the code lengths of the previous Huffman code.
  struct CostModel {
    std::array<std::array<Index, 256>, NumTables> cost;
    // The cost of the token that starts a run of literals.
    Index literalRun;

    CostModel() {
      for (auto& table : cost) {
        table.fill(8);
      }
      literalRun = 8;
    }

    Index matchCost(Index length, Index distance) const {
      Index tokenLength = length - MinMatch;
      Index bits = 0;
      if (tokenLength < 63) {
        bits += cost[TokenTable]
                    [(distance <= ShortDistance ? 2 : 3) | (tokenLength << 2)];
      } else {
        bits +=
          cost[TokenTable][(distance <= ShortDistance ? 2 : 3) | (63 << 2)] +
          cost[ExtraTable][tokenLength - 63];
      }
      distance--;
      if (distance < ShortDistance) {
        bits += cost[ShortDistTable][distance];
      } else {
        bits +=
          cost[DistLoTable][distance & 0xFF] + cost[DistHiTable][distance >> 8];
      }
      return bits;
    }

    Index literalCost(uint16_t unit) const {
      if (unit < 256) {
        return cost[NarrowTable][unit];
      }
      return cost[WideLoTable][unit & 0xFF] + cost[WideHiTable][unit >> 8];
    }
  };

  // Optimal parsing, backwards, of the units into literals and matches under
  // the given cost model. Two states: after a literal (a literal run is open)
  // and after a match (a literal needs a new run token).
  std::vector<Symbol> parse(const std::vector<Candidates>& matches,
                            const CostModel& model) {
    Index n = units.size();
    std::vector<Index> costAfterLiteral(n + 1, 0), costAfterMatch(n + 1, 0);
    // The chosen match at each position for each state; length 0 is a
    // literal.
    std::vector<Match> choiceAfterLiteral(n), choiceAfterMatch(n);
    for (Index i = n; i > 0;) {
      i--;
      Match bestMatch;
      Index bestCost = Index(-1);
      // The candidates are ordered by increasing distance and length, so for
      // each length the first candidate that is long enough has the smallest
      // distance.
      auto& candidates = matches[i];
      Index k = 0;
      for (Index length = MinMatch; k < MaxCandidates; length++) {
        while (k < MaxCandidates && candidates[k].length < length) {
          k++;
        }
        if (k == MaxCandidates) {
          break;
        }
        Index cost = model.matchCost(length, candidates[k].distance) +
                     costAfterMatch[i + length];
        if (cost < bestCost) {
          bestCost = cost;
          bestMatch = {length, candidates[k].distance};
        }
      }
      Index literal = model.literalCost(units[i]) + costAfterLiteral[i + 1];
      if (literal <= bestCost) {
        costAfterLiteral[i] = literal;
        choiceAfterLiteral[i] = Match();
      } else {
        costAfterLiteral[i] = bestCost;
        choiceAfterLiteral[i] = bestMatch;
      }
      if (literal + model.literalRun <= bestCost) {
        costAfterMatch[i] = literal + model.literalRun;
        choiceAfterMatch[i] = Match();
      } else {
        costAfterMatch[i] = bestCost;
        choiceAfterMatch[i] = bestMatch;
      }
    }

    // Emit the tokens.
    std::vector<Symbol> out;
    auto emitToken = [&](Index kind, Index length) {
      assert(length <= MaxTokenLength);
      if (length < 63) {
        out.emplace_back(TokenTable, uint8_t(kind | (length << 2)));
      } else {
        out.emplace_back(TokenTable, uint8_t(kind | (63 << 2)));
        out.emplace_back(ExtraTable, uint8_t(length - 63));
      }
    };
    std::vector<uint16_t> pending;
    auto flushLiterals = [&]() {
      // Split the pending literals into narrow and wide runs. A single narrow
      // unit between wide units is emitted as wide, since a run token costs
      // as much as the byte it would save.
      Index count = pending.size();
      std::vector<bool> wide(count);
      for (Index i = 0; i < count; i++) {
        wide[i] = pending[i] >= 256;
      }
      for (Index i = 0; i < count; i++) {
        if (!wide[i] && (i == 0 || wide[i - 1]) &&
            (i + 1 == count || wide[i + 1]) && count > 1) {
          wide[i] = true;
        }
      }
      Index i = 0;
      while (i < count) {
        Index j = i;
        while (j < count && wide[j] == wide[i] && j - i < MaxTokenLength) {
          j++;
        }
        emitToken(wide[i] ? 1 : 0, j - i);
        for (Index k = i; k < j; k++) {
          if (wide[i]) {
            out.emplace_back(WideLoTable, uint8_t(pending[k] & 0xFF));
            out.emplace_back(WideHiTable, uint8_t(pending[k] >> 8));
          } else {
            out.emplace_back(NarrowTable, uint8_t(pending[k]));
          }
        }
        i = j;
      }
      pending.clear();
    };
    bool afterLiteral = false;
    for (Index i = 0; i < n;) {
      auto& choice = afterLiteral ? choiceAfterLiteral[i] : choiceAfterMatch[i];
      if (choice.length == 0) {
        pending.push_back(units[i]);
        afterLiteral = true;
        i++;
        continue;
      }
      flushLiterals();
      Index distance = choice.distance - 1;
      emitToken(distance < ShortDistance ? 2 : 3, choice.length - MinMatch);
      if (distance < ShortDistance) {
        out.emplace_back(ShortDistTable, uint8_t(distance));
      } else {
        out.emplace_back(DistLoTable, uint8_t(distance & 0xFF));
        out.emplace_back(DistHiTable, uint8_t(distance >> 8));
      }
      afterLiteral = false;
      i += choice.length;
    }
    flushLiterals();
    return out;
  }

  // The number of parses: the first with byte costs, the others with the bit
  // costs of the previous parse's Huffman code.
  static constexpr Index MaxParses = 5;

  Segment encode() {
    std::vector<Candidates> matches;
    if (compress) {
      findMatches(matches);
    } else {
      matches.assign(units.size(), Candidates{});
    }
    CostModel model;
    Segment best;
    for (Index pass = 0; pass < MaxParses; pass++) {
      auto symbols = parse(matches, model);
      auto segment = packSymbols(symbols);
      bool better = pass == 0 || segment.data.size() < best.data.size();
      if (!segment.huffman || !compress) {
        if (better) {
          best = std::move(segment);
        }
        break;
      }
      auto& lengthsOf = segment.lengths;
      if (better) {
        best = segment;
      }
      // Derive the bit costs from the code lengths. Unused symbols cost a bit
      // more than the longest used code of their class.
      for (Index table = 0; table < NumTables; table++) {
        auto& lengths = lengthsOf[table];
        Index unused = 8;
        if (!lengths.empty()) {
          unused = *std::max_element(lengths.begin(), lengths.end()) + 2;
        }
        for (Index symbol = 0; symbol < 256; symbol++) {
          Index length = lengths.empty() ? 0 : lengths[symbol];
          model.cost[table][symbol] = length ? length : unused;
        }
      }
      // A literal run costs the average literal run token.
      Index literalRunBits = 0, literalRuns = 0;
      for (auto& symbol : symbols) {
        if (symbol.first == TokenTable && (symbol.second & 3) < 2) {
          literalRunBits += model.cost[TokenTable][symbol.second];
          literalRuns++;
        }
      }
      if (literalRuns) {
        model.literalRun = (literalRunBits + literalRuns / 2) / literalRuns;
      }
    }
    return best;
  }

  // Huffman coding of the token stream bytes.

  // Computes code lengths of at most MaxCodeLength bits for the given symbol
  // counts (0 for unused symbols). Halves the counts until the longest code
  // fits, which loses very little on real data.
  static std::vector<uint8_t>
  computeCodeLengths(const std::vector<uint64_t>& counts) {
    std::vector<uint8_t> lengths(counts.size(), 0);
    std::vector<Index> used;
    for (Index i = 0; i < counts.size(); i++) {
      if (counts[i]) {
        used.push_back(i);
      }
    }
    if (used.empty()) {
      return lengths;
    }
    Index m = used.size();
    if (m == 1) {
      lengths[used[0]] = 1;
      return lengths;
    }
    std::vector<uint64_t> freqs;
    for (auto i : used) {
      freqs.push_back(counts[i]);
    }
    while (true) {
      // Leaves are nodes 0..m-1, internal nodes follow.
      std::vector<Index> parent(2 * m - 1, Index(-1));
      using Node = std::pair<uint64_t, Index>;
      std::priority_queue<Node, std::vector<Node>, std::greater<Node>> queue;
      for (Index i = 0; i < m; i++) {
        queue.push({freqs[i], i});
      }
      Index next = m;
      while (queue.size() > 1) {
        auto a = queue.top();
        queue.pop();
        auto b = queue.top();
        queue.pop();
        parent[a.second] = parent[b.second] = next;
        queue.push({a.first + b.first, next});
        next++;
      }
      Index maxLength = 0;
      for (Index i = 0; i < m; i++) {
        Index depth = 0;
        for (Index j = i; parent[j] != Index(-1); j = parent[j]) {
          depth++;
        }
        lengths[used[i]] = depth;
        maxLength = std::max(maxLength, depth);
      }
      if (maxLength <= MaxCodeLength) {
        break;
      }
      for (auto& freq : freqs) {
        freq = std::max<uint64_t>(1, freq >> 1);
      }
    }
    // The count of codes of a length must fit in a byte, which only fails
    // when all 256 symbols have 8-bit codes; move to 7 + 253 x 8 + 2 x 9.
    if (m == 256 && std::all_of(lengths.begin(),
                                lengths.end(),
                                [](uint8_t length) { return length == 8; })) {
      lengths[0] = 7;
      lengths[1] = lengths[2] = 9;
    }
    return lengths;
  }

  // Lays out the symbols as plain bytes or, when smaller, as Huffman tables
  // plus a bit stream.
  Segment packSymbols(const std::vector<Symbol>& symbols) {
    Segment plain;
    for (auto& symbol : symbols) {
      plain.data.push_back(char(symbol.second));
    }
    if (huffman == 0) {
      return plain;
    }

    Segment coded;
    coded.huffman = true;
    std::array<std::vector<uint64_t>, NumTables> counts;
    for (auto& count : counts) {
      count.assign(256, 0);
    }
    for (auto& symbol : symbols) {
      counts[symbol.first][symbol.second]++;
    }
    std::array<std::vector<uint8_t>, NumTables> lengths;
    std::array<std::array<uint32_t, 256>, NumTables> codes{};
    for (Index table = 0; table < NumTables; table++) {
      lengths[table] = computeCodeLengths(counts[table]);
      std::array<Index, MaxCodeLength + 1> lengthCounts{};
      for (auto length : lengths[table]) {
        lengthCounts[length]++;
      }
      if (lengthCounts[0] == 256) {
        // Unused class.
        continue;
      }
      coded.tableOffsets[table] = coded.data.size();
      for (Index length = 1; length <= MaxCodeLength; length++) {
        assert(lengthCounts[length] < 256);
        coded.data.push_back(char(lengthCounts[length]));
      }
      // Canonical codes: for each length in order, the symbols in increasing
      // order get consecutive codes.
      uint32_t code = 0;
      for (Index length = 1; length <= MaxCodeLength; length++) {
        for (Index symbol = 0; symbol < 256; symbol++) {
          if (lengths[table][symbol] == length) {
            coded.data.push_back(char(symbol));
            codes[table][symbol] = code++;
          }
        }
        code <<= 1;
      }
    }
    coded.firstBit = coded.data.size() * 8;
    coded.lengths = lengths;
    Index bits = 0;
    for (auto& symbol : symbols) {
      auto length = lengths[symbol.first][symbol.second];
      auto code = codes[symbol.first][symbol.second];
      for (Index i = length; i > 0; i--) {
        if (bits % 8 == 0) {
          coded.data.push_back(0);
        }
        coded.data.back() |= char(((code >> (i - 1)) & 1) << (bits % 8));
        bits++;
      }
    }
    if (huffman == 1 || coded.data.size() + decoderSize < plain.data.size()) {
      return coded;
    }
    return plain;
  }
};

// The runtime pieces of the Huffman decoding: the symbol decoder function and
// the global holding the current bit position. Both can be shared by several
// decoders in a module.
struct SymbolDecoder {
  Name func;
  Name bitsGlobal;
};

// The names under which the symbol decoder is created, so that a later pass
// can find and reuse it.
inline const char* SymbolFuncName = "lz$symbol";
inline const char* BitsGlobalName = "lz$bits";

// Finds an existing symbol decoder in the module, if any. Matches structurally
// as well as by name so that a decoder created by an earlier pass can still be
// shared after passes that strip names (--roundtrip) or relax parameter
// nullability (--relax-nullability).
inline std::optional<SymbolDecoder> findSymbolDecoder(Module& module,
                                                      HeapType i8ArrayType) {
  auto check = [&](Function* func) -> std::optional<SymbolDecoder> {
    if (!func || func->imported() || func->vars.size() != 5) {
      return std::nullopt;
    }
    auto params = func->getParams();
    if (params.size() != 2 || !params[0].isRef() ||
        params[0].getHeapType() != i8ArrayType || params[1] != Type::i32 ||
        func->getResults() != Type::i32) {
      return std::nullopt;
    }
    for (auto var : func->vars) {
      if (var != Type::i32) {
        return std::nullopt;
      }
    }
    struct Scanner : public PostWalker<Scanner> {
      HeapType i8ArrayType;
      Index globalGets = 0;
      Index globalSets = 0;
      Index loops = 0;
      Index arrayGets = 0;
      Name getGlobal;
      Name setGlobal;
      bool hasMaxLen = false;
      bool bad = false;
      Scanner(HeapType i8ArrayType) : i8ArrayType(i8ArrayType) {}
      void visitGlobalGet(GlobalGet* curr) {
        globalGets++;
        getGlobal = curr->name;
      }
      void visitGlobalSet(GlobalSet* curr) {
        globalSets++;
        setGlobal = curr->name;
      }
      void visitLoop(Loop* curr) { loops++; }
      void visitArrayGet(ArrayGet* curr) {
        if (curr->signed_ || curr->ref->type.isNull() ||
            curr->ref->type.getHeapType() != i8ArrayType) {
          bad = true;
        }
        arrayGets++;
      }
      void visitConst(Const* curr) {
        if (curr->type == Type::i32 &&
            curr->value.geti32() == int32_t(MaxCodeLength)) {
          hasMaxLen = true;
        }
      }
      void visitCall(Call* curr) { bad = true; }
      void visitCallRef(CallRef* curr) { bad = true; }
      void visitCallIndirect(CallIndirect* curr) { bad = true; }
      void visitArraySet(ArraySet* curr) { bad = true; }
      void visitStructGet(StructGet* curr) { bad = true; }
      void visitStructSet(StructSet* curr) { bad = true; }
    };
    Scanner scanner(i8ArrayType);
    scanner.walk(func->body);
    if (scanner.bad || scanner.globalGets != 1 || scanner.globalSets != 1 ||
        scanner.getGlobal != scanner.setGlobal || scanner.loops != 1 ||
        scanner.arrayGets != 3 || !scanner.hasMaxLen) {
      return std::nullopt;
    }
    auto* global = module.getGlobalOrNull(scanner.getGlobal);
    if (!global || global->imported() || !global->mutable_ ||
        global->type != Type::i32) {
      return std::nullopt;
    }
    return SymbolDecoder{func->name, global->name};
  };
  if (auto* func = module.getFunctionOrNull(SymbolFuncName)) {
    if (auto found = check(func)) {
      return found;
    }
  }
  for (auto& func : module.functions) {
    if (auto found = check(func.get())) {
      return found;
    }
  }
  return std::nullopt;
}

// The canonical Huffman symbol decoder (see puff.c in zlib), which reads
// the code bit by bit from the position in the $bits global and looks the
// symbol up in the table at offset $t of the segment (15 counts of the codes
// of each length, then the symbols in code order):
//
//   (func $symbol (param $src (ref $i8)) (param $t i32) (result i32)
//     (local $pos i32) (local $code i32) (local $first i32)
//     (local $index i32) (local $count i32)
//     (local.set $pos (global.get $bits))
//     (local.set $index (i32.add (local.get $t) (i32.const 15)))
//     (loop $next
//       (local.set $code
//         (i32.or (local.get $code)
//           (i32.and
//             (i32.shr_u
//               (array.get_u $i8 (local.get $src)
//                 (i32.shr_u (local.get $pos) (i32.const 3)))
//               (i32.and (local.get $pos) (i32.const 7)))
//             (i32.const 1))))
//       (local.set $pos (i32.add (local.get $pos) (i32.const 1)))
//       (local.set $count (array.get_u $i8 (local.get $src) (local.get $t)))
//       (local.set $t (i32.add (local.get $t) (i32.const 1)))
//       (if (i32.lt_u (local.get $code)
//                     (i32.add (local.get $first) (local.get $count)))
//         (then
//           (global.set $bits (local.get $pos))
//           (return
//             (array.get_u $i8 (local.get $src)
//               (i32.add (local.get $index)
//                        (i32.sub (local.get $code) (local.get $first)))))))
//       (local.set $index (i32.add (local.get $index) (local.get $count)))
//       (local.set $first
//         (i32.shl (i32.add (local.get $first) (local.get $count))
//                  (i32.const 1)))
//       (local.set $code (i32.shl (local.get $code) (i32.const 1)))
//       (br $next)))
//
// The bits global is initialized to firstBit.
inline SymbolDecoder
makeSymbolDecoder(Module& module, HeapType i8ArrayType, Index firstBit) {
  Builder builder(module);
  auto i32 = Type::i32;
  SymbolDecoder names;
  names.bitsGlobal = Names::getValidGlobalName(module, BitsGlobalName);
  module.addGlobal(builder.makeGlobal(names.bitsGlobal,
                                      i32,
                                      builder.makeConst(int32_t(firstBit)),
                                      Builder::Mutable));
  names.func = Names::getValidFunctionName(module, SymbolFuncName);
  Index src = 0, t = 1, pos = 2, code = 3, first = 4, index = 5, count = 6;
  Type i8Ref(i8ArrayType, NonNullable);
  auto get = [&](Index index) { return builder.makeLocalGet(index, i32); };
  auto add = [&](Expression* a, Expression* b) {
    return builder.makeBinary(AddInt32, a, b);
  };
  auto byteAt = [&](Expression* index) {
    return builder.makeArrayGet(builder.makeLocalGet(src, i8Ref),
                                index,
                                MemoryOrder::Unordered,
                                i32,
                                false);
  };
  auto* bit = builder.makeBinary(
    AndInt32,
    builder.makeBinary(
      ShrUInt32,
      byteAt(builder.makeBinary(ShrUInt32, get(pos), builder.makeConst(3))),
      builder.makeBinary(AndInt32, get(pos), builder.makeConst(7))),
    builder.makeConst(1));
  auto* found = builder.makeIf(
    builder.makeBinary(LtUInt32, get(code), add(get(first), get(count))),
    builder.makeBlock(
      {builder.makeGlobalSet(names.bitsGlobal, get(pos)),
       builder.makeReturn(byteAt(add(
         get(index), builder.makeBinary(SubInt32, get(code), get(first)))))}));
  Name label("next");
  auto* loop = builder.makeLoop(
    label,
    builder.makeBlock(
      {builder.makeLocalSet(code, builder.makeBinary(OrInt32, get(code), bit)),
       builder.makeLocalSet(pos, add(get(pos), builder.makeConst(1))),
       builder.makeLocalSet(count, byteAt(get(t))),
       builder.makeLocalSet(t, add(get(t), builder.makeConst(1))),
       found,
       builder.makeLocalSet(index, add(get(index), get(count))),
       builder.makeLocalSet(first,
                            builder.makeBinary(ShlInt32,
                                               add(get(first), get(count)),
                                               builder.makeConst(1))),
       builder.makeLocalSet(
         code, builder.makeBinary(ShlInt32, get(code), builder.makeConst(1))),
       builder.makeBreak(label)}));
  auto* body = builder.makeBlock(
    {builder.makeLocalSet(pos, builder.makeGlobalGet(names.bitsGlobal, i32)),
     builder.makeLocalSet(
       index, add(get(t), builder.makeConst(int32_t(MaxCodeLength)))),
     loop});
  module.addFunction(builder.makeFunction(
    names.func, Signature({i8Ref, i32}, i32), {i32, i32, i32, i32, i32}, body));
  return names;
}

struct DecoderOptions {
  // The name of the decoder function to create.
  Name decodeFunc;
  // The data segment holding the coded bytes.
  Name segment;
  // The mutable global that receives the decoded array.
  Name dataGlobal;
  // The array type of the segment bytes.
  HeapType i8ArrayType;
  // The array type of the decoded units: an i16 array, or an i8 array when
  // all units are bytes (then no wide literal runs occur).
  HeapType dstArrayType;
  // The number of units to decode.
  Index numUnits = 0;
  // The symbol decoder, when the segment is Huffman-coded.
  SymbolDecoder symbols;
  // Whether the bits global is shared with other decoders and must be set to
  // the segment's first bit around the decoding (and restored afterwards).
  bool setBits = false;
};

// The decoder, which expands the segment into the unit array once:
//
//   (func $decode
//     (local $src (ref $i8)) (local $dst (ref $i16))
//     (local $p i32) (local $q i32) (local $tok i32) (local $len i32)
//     (local $d i32)
//     (local.set $src (array.new_data $i8 $seg (i32.const 0) (i32.const S)))
//     (local.set $dst (array.new_default $i16 (i32.const N)))
//     (loop $next
//       (local.set $tok (..byte..))
//       (local.set $len (i32.shr_u (local.get $tok) (i32.const 2)))
//       (if (i32.eq (local.get $len) (i32.const 63))
//         (then
//           (local.set $len (i32.add (i32.const 63) (..byte..)))))
//       (local.set $tok (i32.and (local.get $tok) (i32.const 3)))
//       (if (i32.lt_u (local.get $tok) (i32.const 2))
//         (then
//           (loop $lit
//             (if (local.get $tok)
//               (then
//                 (array.set $i16 (local.get $dst) (local.get $q)
//                   (i32.or (..byte..) (i32.shl (..byte..) (i32.const 8)))))
//               (else
//                 (array.set $i16 (local.get $dst) (local.get $q)
//                   (..byte..))))
//             (local.set $q (i32.add (local.get $q) (i32.const 1)))
//             (br_if $lit (local.tee $len (i32.sub (local.get $len) 1)))))
//         (else
//           (if (i32.eq (local.get $tok) (i32.const 3))
//             (then
//               (local.set $d
//                 (i32.or (..byte..) (i32.shl (..byte..) (i32.const 8)))))
//             (else
//               (local.set $d (..byte..))))
//           (local.set $d (i32.add (local.get $d) (i32.const 1)))
//           (local.set $len (i32.add (local.get $len) (i32.const 3)))
//           (loop $copy
//             (array.set $i16 (local.get $dst) (local.get $q)
//               (array.get_u $i16 (local.get $dst)
//                 (i32.sub (local.get $q) (local.get $d))))
//             (local.set $q (i32.add (local.get $q) (i32.const 1)))
//             (br_if $copy (local.tee $len (i32.sub (local.get $len) 1))))))
//       (br_if $next (i32.lt_u (local.get $q) (i32.const N))))
//     (data.drop $seg)
//     (global.set $data (local.get $dst)))
//
// where (..byte..) reads the next byte of its class: with plain bytes
//
//   (array.get_u $i8 (local.get $src)
//     (i32.sub (local.tee $p (i32.add (local.get $p) (i32.const 1))) 1))
//
// and with Huffman coding
//
//   (call $symbol (local.get $src) (i32.const tableOffset))
//
// With a byte destination the literal loop just reads narrow literals.
inline void
makeDecoder(Module& module, const Segment& segment, const DecoderOptions& o) {
  Builder builder(module);
  Index src = 0, dst = 1, p = 2, q = 3, tok = 4, len = 5, d = 6, oldBits = 7;
  Type i8Ref(o.i8ArrayType, NonNullable);
  Type dstRef(o.dstArrayType, NonNullable);
  bool bytes = o.dstArrayType.getArray().element.packedType == Field::i8;
  auto i32 = Type::i32;
  auto get = [&](Index index) { return builder.makeLocalGet(index, i32); };
  auto add = [&](Expression* a, Expression* b) {
    return builder.makeBinary(AddInt32, a, b);
  };
  auto addConst = [&](Index index, int32_t value) {
    return builder.makeLocalSet(index,
                                add(get(index), builder.makeConst(value)));
  };
  auto readByte = [&](Table table) -> Expression* {
    if (segment.huffman) {
      return builder.makeCall(
        o.symbols.func,
        {builder.makeLocalGet(src, i8Ref),
         builder.makeConst(int32_t(segment.tableOffsets[table]))},
        i32);
    }
    return builder.makeArrayGet(
      builder.makeLocalGet(src, i8Ref),
      builder.makeBinary(
        SubInt32,
        builder.makeLocalTee(p, add(get(p), builder.makeConst(1)), i32),
        builder.makeConst(1)),
      MemoryOrder::Unordered,
      i32,
      false);
  };
  auto readPair = [&](Table lo, Table hi) {
    return builder.makeBinary(
      OrInt32,
      readByte(lo),
      builder.makeBinary(ShlInt32, readByte(hi), builder.makeConst(8)));
  };
  auto setDst = [&](Expression* value) {
    return builder.makeArraySet(
      builder.makeLocalGet(dst, dstRef), get(q), value, MemoryOrder::Unordered);
  };
  auto countDown = [&](Name label) {
    return builder.makeBreak(
      label,
      nullptr,
      builder.makeLocalTee(
        len,
        builder.makeBinary(SubInt32, get(len), builder.makeConst(1)),
        i32));
  };

  auto* setSrc = builder.makeLocalSet(
    src,
    builder.makeArrayNewData(o.i8ArrayType,
                             o.segment,
                             builder.makeConst(0),
                             builder.makeConst(int32_t(segment.data.size()))));
  auto* setDstArray = builder.makeLocalSet(
    dst,
    builder.makeArrayNew(o.dstArrayType,
                         builder.makeConst(int32_t(o.numUnits))));

  auto* readTok = builder.makeLocalSet(tok, readByte(TokenTable));
  auto* setLen = builder.makeLocalSet(
    len, builder.makeBinary(ShrUInt32, get(tok), builder.makeConst(2)));
  auto* extendLen =
    builder.makeIf(builder.makeBinary(EqInt32, get(len), builder.makeConst(63)),
                   builder.makeLocalSet(
                     len, add(builder.makeConst(63), readByte(ExtraTable))));
  auto* maskTok = builder.makeLocalSet(
    tok, builder.makeBinary(AndInt32, get(tok), builder.makeConst(3)));

  Name litLabel("lit");
  Expression* setLiteral = setDst(readByte(NarrowTable));
  if (!bytes) {
    setLiteral = builder.makeIf(
      get(tok), setDst(readPair(WideLoTable, WideHiTable)), setLiteral);
  }
  auto* literal = builder.makeLoop(
    litLabel,
    builder.makeBlock({setLiteral, addConst(q, 1), countDown(litLabel)}));

  Name copyLabel("copy");
  auto* setD =
    builder.makeIf(builder.makeBinary(EqInt32, get(tok), builder.makeConst(3)),
                   builder.makeLocalSet(d, readPair(DistLoTable, DistHiTable)),
                   builder.makeLocalSet(d, readByte(ShortDistTable)));
  auto* copy = builder.makeLoop(
    copyLabel,
    builder.makeBlock(
      {setDst(builder.makeArrayGet(builder.makeLocalGet(dst, dstRef),
                                   builder.makeBinary(SubInt32, get(q), get(d)),
                                   MemoryOrder::Unordered,
                                   i32,
                                   false)),
       addConst(q, 1),
       countDown(copyLabel)}));
  auto* match =
    builder.makeBlock({setD, addConst(d, 1), addConst(len, MinMatch), copy});

  auto* dispatch =
    builder.makeIf(builder.makeBinary(LtUInt32, get(tok), builder.makeConst(2)),
                   literal,
                   match);
  Name nextLabel("next");
  auto* next = builder.makeBreak(
    nextLabel,
    nullptr,
    builder.makeBinary(
      LtUInt32, get(q), builder.makeConst(int32_t(o.numUnits))));
  auto* loop = builder.makeLoop(
    nextLabel,
    builder.makeBlock({readTok, setLen, extendLen, maskTok, dispatch, next}));
  std::vector<Expression*> body;
  if (segment.huffman && o.setBits) {
    body.push_back(builder.makeLocalSet(
      oldBits, builder.makeGlobalGet(o.symbols.bitsGlobal, i32)));
    body.push_back(builder.makeGlobalSet(
      o.symbols.bitsGlobal, builder.makeConst(int32_t(segment.firstBit))));
  }
  body.push_back(setSrc);
  body.push_back(setDstArray);
  body.push_back(loop);
  body.push_back(builder.makeDataDrop(o.segment));
  body.push_back(
    builder.makeGlobalSet(o.dataGlobal, builder.makeLocalGet(dst, dstRef)));
  if (segment.huffman && o.setBits) {
    body.push_back(builder.makeGlobalSet(o.symbols.bitsGlobal, get(oldBits)));
  }
  std::vector<Type> vars{i8Ref, dstRef, i32, i32, i32, i32, i32};
  if (segment.huffman && o.setBits) {
    vars.push_back(i32);
  }
  module.addFunction(builder.makeFunction(o.decodeFunc,
                                          Signature(Type::none, Type::none),
                                          std::move(vars),
                                          builder.makeBlock(body)));
}

} // namespace wasm::LZHuffman

#endif // wasm_passes_lz_huffman_h
