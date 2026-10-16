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
// Compresses passive data segments that are only read with array.new_data.
//
// The bytes of all such segments (that are compressible on their own) are
// concatenated and LZ77 + Huffman coded (see lz-huffman.h) into one new
// passive segment. A decoder expands that segment on first use into a mutable
// i8 array kept in a global, and each
//
//   (array.new_data $T $seg (off) (n))
//
// becomes a call to a small helper for $T that allocates the array and fills
// it from the decoded bytes: with array.copy for i8 arrays, and by assembling
// each element from its little-endian bytes for i16, i32, i64, f32 and f64
// arrays. data.drop of a compressed segment becomes a nop.
//
// A size model skips the transformation when it would not help;
// --pass-arg=compress-data-segments-always@1 ignores the model, and
// --pass-arg=compress-data-segments-min-size@N (default 16) sets the smallest
// segment considered, and --pass-arg=compress-data-segments-huffman@0|1 never
// or always Huffman-codes the LZ77 bytes (the default decides by size).
//

#include <algorithm>
#include <cstring>
#include <optional>
#include <unordered_set>

#include "ir/module-utils.h"
#include "ir/names.h"
#include "pass.h"
#include "passes/lz-huffman.h"
#include "support/insert_ordered.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

struct CompressDataSegments : public Pass {
  struct NewSite {
    ArrayNewData* curr;
    Function* func = nullptr;
    bool offsetParamClean = false;
    bool sizeParamClean = false;
  };

  // The uses of a segment.
  struct Uses {
    std::vector<NewSite> news;
    Index drops = 0;
    // Any use other than array.new_data and data.drop.
    bool other = false;
  };

  struct Slice {
    Index off;
    Index n;
    Index w;
    bool zigzag;
    bool operator==(const Slice& o) const {
      return off == o.off && n == o.n && w == o.w && zigzag == o.zigzag;
    }
    bool operator<(const Slice& o) const {
      if (off != o.off) {
        return off < o.off;
      }
      if (n != o.n) {
        return n < o.n;
      }
      if (w != o.w) {
        return w < o.w;
      }
      return zigzag < o.zigzag;
    }
  };

  struct Candidate {
    DataSegment* segment;
    std::vector<uint8_t> bytes;
  };

  using FunctionUses = std::unordered_map<Name, Uses>;

  // A compressed segment: its base offset in the decoded bytes.
  struct Compressed {
    Index base;
  };

  Module* module;
  bool always = false;
  Index minSize = 16;
  // 0: never Huffman-code, 1: always, 2: by size.
  int huffman = 2;

  HeapType i8ArrayType;
  Name dataGlobalName;
  Name decodeFuncName;
  Name getFuncName;
  Name segmentName;
  InsertOrderedMap<Name, Compressed> compressed;
  // The slice helper for each array type.
  std::unordered_map<HeapType, Name> sliceFuncs;
  std::unordered_set<Name> paramFuncs;
  std::unordered_set<Name> escapedParamFuncs;
  std::unordered_map<Name, std::vector<Call*>> paramFuncCalls;

  void run(Module* module_) override {
    module = module_;
    if (!module->features.hasGC() || !module->features.hasBulkMemory()) {
      return;
    }
    always = !getArgumentOrDefault("compress-data-segments-always", "").empty();
    minSize =
      std::stoul(getArgumentOrDefault("compress-data-segments-min-size", "16"));
    auto huffmanArg =
      getArgumentOrDefault("compress-data-segments-huffman", "");
    if (huffmanArg == "0") {
      huffman = 0;
    } else if (huffmanArg == "1") {
      huffman = 1;
    }

    // Find the uses of each segment.
    ModuleUtils::ParallelFunctionAnalysis<FunctionUses> analysis(
      *module, [&](Function* func, FunctionUses& uses) {
        if (!func->imported()) {
          scan(func, func->body, uses);
        }
      });
    FunctionUses uses;
    for (auto& [_, funcUses] : analysis.map) {
      merge(funcUses, uses);
    }
    // Any mention of a segment in module-level code pins it.
    FunctionUses moduleUses;
    for (auto& segment : module->dataSegments) {
      if (segment->offset) {
        scan(nullptr, segment->offset, moduleUses);
      }
    }
    for (auto& global : module->globals) {
      if (!global->imported()) {
        scan(nullptr, global->init, moduleUses);
      }
    }
    for (auto& [name, _] : moduleUses) {
      uses[name].other = true;
    }

    // Collect functions that forward parameter offsets/sizes into
    // array.new_data so we can resolve their constant call-site arguments.
    for (auto& segment : module->dataSegments) {
      if (!segment->isPassive() || segment->data.size() < minSize) {
        continue;
      }
      auto it = uses.find(segment->name);
      if (it == uses.end() || it->second.other || it->second.news.empty()) {
        continue;
      }
      for (auto& site : it->second.news) {
        if (site.func &&
            (!site.curr->offset->is<Const>() ||
             !site.curr->size->is<Const>()) &&
            (site.curr->offset->is<Const>() || site.offsetParamClean) &&
            (site.curr->size->is<Const>() || site.sizeParamClean)) {
          paramFuncs.insert(site.func->name);
        }
      }
    }
    if (!paramFuncs.empty()) {
      collectParamFuncCallers();
    }

    // Select the candidate segments: passive, large enough, only used by
    // array.new_data of arrays we can fill, with resolvable multi-byte slices,
    // and compressible on their own after delta + byte-plane transformation.
    std::vector<Candidate> candidates;
    for (auto& segment : module->dataSegments) {
      if (!segment->isPassive() || segment->data.size() < minSize) {
        continue;
      }
      auto it = uses.find(segment->name);
      if (it == uses.end() || it->second.other || it->second.news.empty()) {
        continue;
      }
      bool ok = true;
      for (auto& site : it->second.news) {
        if (!canFill(site.curr->type.getHeapType())) {
          ok = false;
          break;
        }
      }
      if (!ok) {
        continue;
      }
      auto transformed = resolveAndTransform(*segment, it->second);
      if (!transformed) {
        continue;
      }
      if (!always && !isCompressible(*transformed)) {
        continue;
      }
      candidates.push_back({segment.get(), std::move(*transformed)});
    }
    if (candidates.empty()) {
      return;
    }

    // Encode the concatenation.
    LZHuffman::Encoder encoder;
    encoder.huffman = huffman;
    encoder.splitNarrow = false;
    Index plainBytes = 0;
    Index numSites = 0;
    for (auto& candidate : candidates) {
      auto* segment = candidate.segment;
      compressed[segment->name] = {Index(encoder.units.size())};
      for (auto byte : candidate.bytes) {
        encoder.units.push_back(byte);
      }
      plainBytes += segment->data.size() + 3;
      numSites += uses[segment->name].news.size();
    }
    auto coded = encoder.encode();

    // The decoded bytes use the canonical i8 array type, which is also the
    // type StringConstsToData's symbol decoder reads from, so that it can be
    // shared.
    i8ArrayType = HeapTypes::getMutI8Array();
    InsertOrderedSet<HeapType> types;
    for (auto& candidate : candidates) {
      for (auto& site : uses[candidate.segment->name].news) {
        types.insert(site.curr->type.getHeapType());
      }
    }
    auto symbols = LZHuffman::findSymbolDecoder(*module, i8ArrayType);

    // The size model: the decoder (and the symbol decoder, unless shared), the
    // getter, the global and the slice helpers, plus the larger offsets at
    // the sites.
    Index helperBytes = 180 + 30 + 10 + numSites * 2;
    if (coded.huffman && !symbols) {
      helperBytes += 120;
    }
    for (auto type : types) {
      helperBytes += type.getArray().element.packedType == Field::i8 ? 30 : 70;
    }
    if (!always && coded.data.size() + helperBytes >= plainBytes) {
      return;
    }

    // Build the runtime pieces.
    Builder builder(*module);
    segmentName = Names::getValidDataSegmentName(*module, "segments");
    module->addDataSegment(builder.makeDataSegment(
      segmentName, Name(), nullptr, coded.data.data(), coded.data.size()));
    dataGlobalName = Names::getValidGlobalName(*module, "segments$data");
    module->addGlobal(builder.makeGlobal(dataGlobalName,
                                         Type(i8ArrayType, Nullable),
                                         builder.makeRefNull(i8ArrayType),
                                         Builder::Mutable));
    decodeFuncName = Names::getValidFunctionName(*module, "segments$decode");
    LZHuffman::DecoderOptions options;
    options.decodeFunc = decodeFuncName;
    options.segment = segmentName;
    options.dataGlobal = dataGlobalName;
    options.i8ArrayType = i8ArrayType;
    options.dstArrayType = i8ArrayType;
    options.numUnits = encoder.units.size();
    if (coded.huffman) {
      if (symbols) {
        options.symbols = *symbols;
        options.setBits = true;
      } else {
        options.symbols =
          LZHuffman::makeSymbolDecoder(*module, i8ArrayType, coded.firstBit);
      }
    }
    LZHuffman::makeDecoder(*module, coded, options);
    makeGetter();
    for (auto type : types) {
      makeSlice(type);
    }

    // Rewrite the uses and remove the segments.
    struct Rewriter : public WalkerPass<PostWalker<Rewriter>> {
      bool isFunctionParallel() override { return true; }
      std::unique_ptr<Pass> create() override {
        return std::make_unique<Rewriter>(parent);
      }
      CompressDataSegments& parent;
      Rewriter(CompressDataSegments& parent) : parent(parent) {}
      void visitArrayNewData(ArrayNewData* curr) {
        auto it = parent.compressed.find(curr->segment);
        if (it == parent.compressed.end()) {
          return;
        }
        Builder builder(*getModule());
        Expression* offset = curr->offset;
        Index base = it->second.base;
        if (base) {
          if (auto* c = offset->dynCast<Const>()) {
            c->value = Literal(int32_t(c->value.geti32() + base));
          } else {
            offset = builder.makeBinary(
              AddInt32, offset, builder.makeConst(int32_t(base)));
          }
        }
        replaceCurrent(
          builder.makeCall(parent.sliceFuncs.at(curr->type.getHeapType()),
                           {offset, curr->size},
                           curr->type));
      }
      void visitDataDrop(DataDrop* curr) {
        if (parent.compressed.count(curr->segment)) {
          replaceCurrent(Builder(*getModule()).makeNop());
        }
      }
    };
    Rewriter(*this).run(getPassRunner(), module);
    for (auto& candidate : candidates) {
      module->removeDataSegment(candidate.segment->name);
    }
  }

  void scan(Function* func, Expression* root, FunctionUses& uses) {
    struct Scanner : public PostWalker<Scanner> {
      Function* func;
      FunctionUses& uses;
      Index loopDepth = 0;
      std::unordered_set<Index> writtenParams;
      Scanner(Function* func, FunctionUses& uses) : func(func), uses(uses) {}
      static void doEnterLoop(Scanner* self, Expression** currp) {
        self->loopDepth++;
      }
      static void doLeaveLoop(Scanner* self, Expression** currp) {
        self->loopDepth--;
      }
      static void scan(Scanner* self, Expression** currp) {
        if ((*currp)->is<Loop>()) {
          self->pushTask(doLeaveLoop, currp);
          PostWalker<Scanner>::scan(self, currp);
          self->pushTask(doEnterLoop, currp);
        } else {
          PostWalker<Scanner>::scan(self, currp);
        }
      }
      void visitLocalSet(LocalSet* curr) {
        if (func && curr->index < func->getNumParams()) {
          writtenParams.insert(curr->index);
        }
      }
      void visitArrayNewData(ArrayNewData* curr) {
        bool offsetClean = false;
        bool sizeClean = false;
        if (func && loopDepth == 0) {
          if (auto* get = curr->offset->dynCast<LocalGet>()) {
            offsetClean = get->index < func->getNumParams() &&
                          !writtenParams.count(get->index);
          }
          if (auto* get = curr->size->dynCast<LocalGet>()) {
            sizeClean = get->index < func->getNumParams() &&
                        !writtenParams.count(get->index);
          }
        }
        uses[curr->segment].news.push_back(
          {curr, func, offsetClean, sizeClean});
      }
      void visitDataDrop(DataDrop* curr) { uses[curr->segment].drops++; }
      void visitArrayInitData(ArrayInitData* curr) {
        uses[curr->segment].other = true;
      }
      void visitMemoryInit(MemoryInit* curr) {
        uses[curr->segment].other = true;
      }
    };
    Scanner(func, uses).walk(root);
  }

  void collectParamFuncCallers() {
    for (auto& exp : module->exports) {
      if (exp->kind == ExternalKind::Function) {
        escapedParamFuncs.insert(*exp->getInternalName());
      }
    }
    struct RefScanner : public PostWalker<RefScanner> {
      CompressDataSegments& parent;
      std::unordered_set<Name>& escaped;
      std::unordered_map<Name, std::vector<Call*>>* calls;
      RefScanner(CompressDataSegments& parent,
                 std::unordered_set<Name>& escaped,
                 std::unordered_map<Name, std::vector<Call*>>* calls = nullptr)
        : parent(parent), escaped(escaped), calls(calls) {}
      void visitRefFunc(RefFunc* curr) {
        if (parent.paramFuncs.count(curr->func)) {
          escaped.insert(curr->func);
        }
      }
      void visitCall(Call* curr) {
        if (calls && parent.paramFuncs.count(curr->target)) {
          (*calls)[curr->target].push_back(curr);
        }
      }
    };
    for (auto& global : module->globals) {
      if (!global->imported()) {
        RefScanner(*this, escapedParamFuncs).walk(global->init);
      }
    }
    for (auto& seg : module->elementSegments) {
      if (seg->offset) {
        RefScanner(*this, escapedParamFuncs).walk(seg->offset);
      }
      for (auto* init : seg->data) {
        RefScanner(*this, escapedParamFuncs).walk(init);
      }
    }
    struct FuncCallInfo {
      std::unordered_set<Name> escaped;
      std::unordered_map<Name, std::vector<Call*>> calls;
    };
    ModuleUtils::ParallelFunctionAnalysis<FuncCallInfo> callAnalysis(
      *module, [&](Function* func, FuncCallInfo& info) {
        if (!func->imported()) {
          RefScanner(*this, info.escaped, &info.calls).walk(func->body);
        }
      });
    for (auto& [_, info] : callAnalysis.map) {
      escapedParamFuncs.insert(info.escaped.begin(), info.escaped.end());
      for (auto& [target, vec] : info.calls) {
        auto& dst = paramFuncCalls[target];
        dst.insert(dst.end(), vec.begin(), vec.end());
      }
    }
  }

  static void transformSlice(
    std::vector<uint8_t>& buf, Index off, Index n, Index w, bool zigzag) {
    if (n == 0 || w <= 1) {
      return;
    }
    std::vector<uint8_t> tmp(size_t(n) * w);
    uint64_t prev = 0;
    for (Index i = 0; i < n; i++) {
      uint64_t v = 0;
      for (Index b = 0; b < w; b++) {
        v |= uint64_t(buf[off + i * w + b]) << (8 * b);
      }
      uint64_t d = v - prev;
      if (zigzag) {
        int64_t sdiff = (w == 2)   ? int16_t(d)
                        : (w == 4) ? int32_t(d)
                                   : int64_t(d);
        d = (uint64_t(sdiff) << 1) ^ uint64_t(sdiff >> 63);
      }
      prev = v;
      for (Index b = 0; b < w; b++) {
        tmp[size_t(b) * n + i] = uint8_t((d >> (8 * b)) & 0xFF);
      }
    }
    std::memcpy(buf.data() + off, tmp.data(), tmp.size());
  }

  std::optional<std::vector<uint8_t>>
  resolveAndTransform(const DataSegment& segment, const Uses& segUses) {
    Index singleTableCount = 0;
    if (segment.data.size() >= 4) {
      uint16_t u0 =
        uint8_t(segment.data[0]) | (uint16_t(uint8_t(segment.data[1])) << 8);
      uint16_t u1 =
        uint8_t(segment.data[2]) | (uint16_t(uint8_t(segment.data[3])) << 8);
      if (u0 == 0 && u1 > 0) {
        for (auto& site : segUses.news) {
          if (site.curr->type.getHeapType().getArray().element.getByteSize() ==
                2 &&
              site.curr->offset->is<Const>() && site.curr->size->is<Const>() &&
              site.curr->offset->cast<Const>()->value.geti32() == 0 &&
              site.curr->size->cast<Const>()->value.geti32() == 2) {
            singleTableCount = u1;
            break;
          }
        }
      }
    }

    std::vector<Slice> slices;
    auto addResolved = [&](const NewSite& site, Index w, bool zigzag) -> bool {
      if (site.curr->offset->is<Const>() && site.curr->size->is<Const>()) {
        uint32_t off = site.curr->offset->cast<Const>()->value.geti32();
        uint32_t n = site.curr->size->cast<Const>()->value.geti32();
        if (n > 0) {
          slices.push_back({off, n, w, zigzag});
        }
        return true;
      }
      if (site.func && paramFuncs.count(site.func->name) &&
          !escapedParamFuncs.count(site.func->name) &&
          (site.curr->offset->is<Const>() || site.offsetParamClean) &&
          (site.curr->size->is<Const>() || site.sizeParamClean)) {
        auto it = paramFuncCalls.find(site.func->name);
        if (it == paramFuncCalls.end() || it->second.empty()) {
          return false;
        }
        for (auto* call : it->second) {
          Expression* offExpr =
            site.curr->offset->is<Const>()
              ? site.curr->offset
              : call->operands[site.curr->offset->cast<LocalGet>()->index];
          Expression* nExpr =
            site.curr->size->is<Const>()
              ? site.curr->size
              : call->operands[site.curr->size->cast<LocalGet>()->index];
          if (!offExpr->is<Const>() || !nExpr->is<Const>()) {
            return false;
          }
          uint32_t off = offExpr->cast<Const>()->value.geti32();
          uint32_t n = nExpr->cast<Const>()->value.geti32();
          if (n > 0) {
            slices.push_back({off, n, w, zigzag});
          }
        }
        return true;
      }
      if (w > 1 && singleTableCount > 0) {
        if (auto* add = site.curr->offset->dynCast<Binary>()) {
          if (add->op == AddInt32 &&
              (add->right->is<Const>() || add->left->is<Const>())) {
            uint32_t colBase = add->right->is<Const>()
                                 ? add->right->cast<Const>()->value.geti32()
                                 : add->left->cast<Const>()->value.geti32();
            if (colBase >= 4) {
              slices.push_back({colBase, singleTableCount, w, zigzag});
              return true;
            }
          }
        }
      }
      return false;
    };

    for (auto& site : segUses.news) {
      auto element = site.curr->type.getHeapType().getArray().element;
      Index w = element.getByteSize();
      bool zigzag = !element.type.isFloat();
      if (!addResolved(site, w, zigzag) && w > 1) {
        return std::nullopt;
      }
    }

    for (auto& s : slices) {
      if (uint64_t(s.off) + uint64_t(s.n) * s.w > segment.data.size()) {
        return std::nullopt;
      }
    }
    std::sort(slices.begin(), slices.end());
    slices.erase(std::unique(slices.begin(), slices.end()), slices.end());
    for (size_t i = 0; i + 1 < slices.size(); i++) {
      if (uint64_t(slices[i].off) + uint64_t(slices[i].n) * slices[i].w >
          slices[i + 1].off) {
        return std::nullopt;
      }
    }

    std::vector<uint8_t> bytes(segment.data.begin(), segment.data.end());
    for (auto& s : slices) {
      if (s.w > 1) {
        transformSlice(bytes, s.off, s.n, s.w, s.zigzag);
      }
    }
    return bytes;
  }

  static void merge(FunctionUses& from, FunctionUses& into) {
    for (auto& [name, uses] : from) {
      auto& target = into[name];
      target.news.insert(target.news.end(), uses.news.begin(), uses.news.end());
      target.drops += uses.drops;
      target.other = target.other || uses.other;
    }
  }

  // Whether we can fill arrays of the type from the decoded bytes.
  static bool canFill(HeapType type) {
    if (!type.isArray()) {
      return false;
    }
    auto element = type.getArray().element;
    if (element.mutable_ != Mutable) {
      return false;
    }
    return element.isPacked() || element.type == Type::i32 ||
           element.type == Type::i64 || element.type == Type::f32 ||
           element.type == Type::f64;
  }

  // Whether the segment's bytes compress on their own (ignoring the Huffman
  // tables, which the concatenation shares), so that it contributes to the
  // concatenation instead of just inflating it and the decoding time.
  static bool isCompressible(const std::vector<uint8_t>& data) {
    LZHuffman::Encoder encoder;
    encoder.huffman = 1;
    encoder.splitNarrow = false;
    for (auto byte : data) {
      encoder.units.push_back(byte);
    }
    auto coded = encoder.encode();
    Index streamBytes = coded.data.size() - coded.firstBit / 8;
    // Require a tenth of the bytes to be saved, so that incompressible data
    // (which would only cost decoding time) is left alone.
    return streamBytes * 10 < data.size() * 9;
  }

  // The getter, which decodes on the first call:
  //
  //   (func $get (result (ref $i8))
  //     (if (ref.is_null (global.get $data)) (then (call $decode)))
  //     (ref.as_non_null (global.get $data)))
  //
  void makeGetter() {
    Builder builder(*module);
    getFuncName = Names::getValidFunctionName(*module, "segments$get");
    Type nullable(i8ArrayType, Nullable);
    auto* body = builder.makeBlock(
      {builder.makeIf(
         builder.makeRefIsNull(builder.makeGlobalGet(dataGlobalName, nullable)),
         builder.makeCall(decodeFuncName, {}, Type::none)),
       builder.makeRefAs(RefAsNonNull,
                         builder.makeGlobalGet(dataGlobalName, nullable))});
    module->addFunction(builder.makeFunction(
      getFuncName,
      Signature(Type::none, Type(i8ArrayType, NonNullable)),
      {},
      body));
  }

  // The slice helper of an array type, which builds an array of n elements
  // from the decoded bytes at the offset. For i8 arrays:
  //
  //   (func $slice (param $off i32) (param $n i32) (result (ref $T))
  //     (local $dst (ref $T))
  //     (local.set $dst (array.new_default $T (local.get $n)))
  //     (array.copy $T $i8 (local.get $dst) (i32.const 0) (call $get)
  //       (local.get $off) (local.get $n))
  //     (local.get $dst))
  //
  // and for the others, with w bytes per element stored delta-coded and
  // byte-plane transposed (plane b at off + b * n):
  //
  //   (func $slice (param $off i32) (param $n i32) (result (ref $T))
  //     (local $dst (ref $T)) (local $data (ref $i8)) (local $i i32)
  //     (local $k i32) (local $v i32/i64) (local $acc i32/i64)
  //     (local.set $dst (array.new_default $T (local.get $n)))
  //     (local.set $data (call $get))
  //     (loop $next
  //       (if (i32.lt_u (local.get $i) (local.get $n))
  //         (then
  //           (..assemble $v from the w byte planes at $off + $k * $n..)
  //           (array.set $T (local.get $dst) (local.get $i)
  //             (local.tee $acc (i32.add (local.get $acc) (local.get $v))))
  //           (local.set $off (i32.add (local.get $off) (i32.const 1)))
  //           (local.set $i (i32.add (local.get $i) (i32.const 1)))
  //           (br $next))))
  //     (local.get $dst))
  //
  void makeSlice(HeapType type) {
    Builder builder(*module);
    auto name = Names::getValidFunctionName(*module, "segments$slice");
    sliceFuncs[type] = name;
    // The result is exact, like array.new_data's.
    Type ref(type, NonNullable, Exact);
    Type i8Ref(i8ArrayType, NonNullable);
    auto i32 = Type::i32;
    Index off = 0, n = 1, dst = 2, data = 3, i = 4, k = 5, v = 6, acc = 7;
    auto get = [&](Index index, Type type = Type::i32) {
      return builder.makeLocalGet(index, type);
    };
    auto* alloc = builder.makeLocalSet(dst, builder.makeArrayNew(type, get(n)));
    auto element = type.getArray().element;
    if (element.packedType == Field::i8) {
      auto* body = builder.makeBlock(
        {alloc,
         builder.makeArrayCopy(get(dst, ref),
                               builder.makeConst(0),
                               builder.makeCall(getFuncName, {}, i8Ref),
                               get(off),
                               get(n)),
         get(dst, ref)});
      module->addFunction(
        builder.makeFunction(name, Signature({i32, i32}, ref), {ref}, body));
      return;
    }
    Index width = element.getByteSize();
    bool wide = width == 8;
    Type valueType = wide ? Type::i64 : Type::i32;
    // Assemble the delta from its byte planes, from the most significant plane
    // (w - 1) down to the least significant plane (0):
    //
    //   (local.set $k (i32.const w))
    //   (local.set $v (i64.const 0))
    //   (loop $byte
    //     (local.set $v
    //       (i64.or (i64.shl (local.get $v) (i64.const 8))
    //         (i64.extend_i32_u
    //           (array.get_u $i8 (local.get $data)
    //             (i32.add (local.get $off)
    //               (i32.mul (local.get $n)
    //                 (local.tee $k (i32.sub (local.get $k) (i32.const 1)))))))))
    //     (br_if $byte (local.get $k)))
    //
    Expression* byte = builder.makeArrayGet(
      get(data, i8Ref),
      builder.makeBinary(
        AddInt32,
        get(off),
        builder.makeBinary(
          MulInt32,
          get(n),
          builder.makeLocalTee(
            k,
            builder.makeBinary(SubInt32, get(k), builder.makeConst(1)),
            i32))),
      MemoryOrder::Unordered,
      i32,
      false);
    if (wide) {
      byte = builder.makeUnary(ExtendUInt32, byte);
    }
    Name byteLabel("byte");
    auto* bytes = builder.makeBlock(
      {builder.makeLocalSet(k, builder.makeConst(int32_t(width))),
       builder.makeLocalSet(v,
                            wide ? builder.makeConst(int64_t(0))
                                 : builder.makeConst(int32_t(0))),
       builder.makeLoop(
         byteLabel,
         builder.makeBlock(
           {builder.makeLocalSet(
              v,
              builder.makeBinary(
                wide ? OrInt64 : OrInt32,
                builder.makeBinary(wide ? ShlInt64 : ShlInt32,
                                   get(v, valueType),
                                   wide ? builder.makeConst(int64_t(8))
                                        : builder.makeConst(int32_t(8))),
                byte)),
            builder.makeBreak(byteLabel, nullptr, get(k))}))});
    Expression* delta = get(v, valueType);
    if (!element.type.isFloat()) {
      delta = builder.makeBinary(
        wide ? XorInt64 : XorInt32,
        builder.makeBinary(wide ? ShrUInt64 : ShrUInt32,
                           get(v, valueType),
                           wide ? builder.makeConst(int64_t(1))
                                : builder.makeConst(int32_t(1))),
        builder.makeBinary(
          wide ? SubInt64 : SubInt32,
          wide ? builder.makeConst(int64_t(0)) : builder.makeConst(int32_t(0)),
          builder.makeBinary(wide ? AndInt64 : AndInt32,
                             get(v, valueType),
                             wide ? builder.makeConst(int64_t(1))
                                  : builder.makeConst(int32_t(1)))));
    }
    Expression* value = builder.makeLocalTee(
      acc,
      builder.makeBinary(wide ? AddInt64 : AddInt32, get(acc, valueType), delta),
      valueType);
    if (element.type == Type::f32) {
      value = builder.makeUnary(ReinterpretInt32, value);
    } else if (element.type == Type::f64) {
      value = builder.makeUnary(ReinterpretInt64, value);
    }
    Name label("next");
    auto* loop = builder.makeLoop(
      label,
      builder.makeIf(
        builder.makeBinary(LtUInt32, get(i), get(n)),
        builder.makeBlock(
          {bytes,
           builder.makeArraySet(
             get(dst, ref), get(i), value, MemoryOrder::Unordered),
           builder.makeLocalSet(
             off, builder.makeBinary(AddInt32, get(off), builder.makeConst(1))),
           builder.makeLocalSet(
             i, builder.makeBinary(AddInt32, get(i), builder.makeConst(1))),
           builder.makeBreak(label)})));
    auto* body = builder.makeBlock(
      {alloc,
       builder.makeLocalSet(data, builder.makeCall(getFuncName, {}, i8Ref)),
       loop,
       get(dst, ref)});
    module->addFunction(
      builder.makeFunction(name,
                           Signature({i32, i32}, ref),
                           {ref, i8Ref, i32, i32, valueType, valueType},
                           body));
  }
};

} // anonymous namespace

Pass* createCompressDataSegmentsPass() { return new CompressDataSegments(); }

} // namespace wasm
