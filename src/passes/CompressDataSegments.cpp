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

#include "ir/effects.h"
#include "ir/iteration.h"
#include "ir/manipulation.h"
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
  InsertOrderedSet<Name> paramFuncs;
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
    for (auto& func : module->functions) {
      merge(analysis.map[func.get()], uses);
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
    std::unordered_set<Name> runtimeFuncs = {decodeFuncName, getFuncName};
    if (options.symbols.func.is()) {
      runtimeFuncs.insert(options.symbols.func);
    }
    eliminateReadOnlyI8Slices(runtimeFuncs);
    InsertOrderedSet<HeapType> remainingTypes;
    for (auto& fn : module->functions) {
      if (fn->imported() || runtimeFuncs.count(fn->name)) {
        continue;
      }
      struct TypeFinder : public PostWalker<TypeFinder> {
        CompressDataSegments& parent;
        InsertOrderedSet<HeapType>& remainingTypes;
        TypeFinder(CompressDataSegments& parent,
                   InsertOrderedSet<HeapType>& remainingTypes)
          : parent(parent), remainingTypes(remainingTypes) {}
        void visitArrayNewData(ArrayNewData* curr) {
          if (parent.compressed.count(curr->segment)) {
            remainingTypes.insert(curr->type.getHeapType());
          }
        }
      } finder(*this, remainingTypes);
      finder.walk(fn->body);
    }
    for (auto type : types) {
      if (remainingTypes.count(type)) {
        makeSlice(type);
      }
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
    for (auto& func : module->functions) {
      auto& info = callAnalysis.map[func.get()];
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

  static Expression*
  addOffset(Expression* idx, Expression* offsetExpr, Builder& builder) {
    if (auto* cOff = offsetExpr->dynCast<Const>()) {
      int32_t delta = cOff->value.geti32();
      if (delta == 0) {
        return idx;
      }
      if (auto* cIdx = idx->dynCast<Const>()) {
        cIdx->value = Literal(int32_t(cIdx->value.geti32() + delta));
        return idx;
      }
      if (auto* bin = idx->dynCast<Binary>()) {
        if (bin->op == AddInt32) {
          if (auto* cRight = bin->right->dynCast<Const>()) {
            cRight->value = Literal(int32_t(cRight->value.geti32() + delta));
            return idx;
          }
          if (auto* cLeft = bin->left->dynCast<Const>()) {
            cLeft->value = Literal(int32_t(cLeft->value.geti32() + delta));
            return idx;
          }
        }
      }
      return builder.makeBinary(AddInt32, idx, builder.makeConst(delta));
    }
    if (auto* cIdx = idx->dynCast<Const>()) {
      if (cIdx->value.geti32() == 0) {
        return offsetExpr;
      }
    }
    return builder.makeBinary(AddInt32, idx, offsetExpr);
  }

  void
  eliminateReadOnlyI8Slices(const std::unordered_set<Name>& runtimeFuncs) {
    Builder builder(*module);
    Type i8Ref(i8ArrayType, NonNullable);

    // Step 1: Un-outline 4-param multi-slice paramFuncs returning an i8
    // ArrayNewData on a compressed segment (such as RollRepeatedCalls /
    // RollConstantTables column pair initializers).
    for (Name pfName : paramFuncs) {
      if (escapedParamFuncs.count(pfName) || runtimeFuncs.count(pfName)) {
        continue;
      }
      Function* pf = module->getFunctionOrNull(pfName);
      if (!pf || pf->imported() || pf->getNumParams() != 4 ||
          !pf->vars.empty() || !pf->body) {
        continue;
      }
      auto* b = pf->body->dynCast<Block>();
      if (!b || b->list.size() != 2) {
        continue;
      }
      auto* gs = b->list[0]->dynCast<GlobalSet>();
      auto* ad1 = b->list[1]->dynCast<ArrayNewData>();
      auto* ad2 = gs ? gs->value->dynCast<ArrayNewData>() : nullptr;
      if (!gs || !ad1 || !ad2 || ad1->type.getHeapType() != i8ArrayType ||
          !compressed.count(ad1->segment)) {
        continue;
      }
      auto* off1 = ad1->offset->dynCast<LocalGet>();
      auto* len1 = ad1->size->dynCast<LocalGet>();
      auto* off2 = ad2->offset->dynCast<LocalGet>();
      auto* len2 = ad2->size->dynCast<LocalGet>();
      if (!off1 || !len1 || !off2 || !len2) {
        continue;
      }
      std::unordered_set<Index> pSet = {
        off1->index, len1->index, off2->index, len2->index};
      if (pSet.size() != 4) {
        continue;
      }
      auto itCalls = paramFuncCalls.find(pfName);
      if (itCalls == paramFuncCalls.end() || itCalls->second.empty()) {
        continue;
      }
      bool allPureArgs = true;
      for (auto* c : itCalls->second) {
        if (c->operands.size() != 4) {
          allPureArgs = false;
          break;
        }
        for (auto* op : c->operands) {
          if (!op->is<Const>() && !op->is<GlobalGet>()) {
            allPureArgs = false;
            break;
          }
        }
        if (!allPureArgs) {
          break;
        }
      }
      if (!allPureArgs) {
        continue;
      }

      Index replacedCalls = 0;
      for (auto& fnPtr : module->functions) {
        Function* fn = fnPtr.get();
        if (fn->imported() || fn == pf || runtimeFuncs.count(fn->name) ||
            !fn->body) {
          continue;
        }
        struct CallerCounter : public PostWalker<CallerCounter> {
          Name pfName;
          Name gsName;
          Index pfCalls = 0;
          Index gsGets = 0;
          CallerCounter(Name pfName, Name gsName)
            : pfName(pfName), gsName(gsName) {}
          void visitCall(Call* curr) {
            if (curr->target == pfName) {
              pfCalls++;
            }
          }
          void visitGlobalGet(GlobalGet* curr) {
            if (curr->name == gsName) {
              gsGets++;
            }
          }
        } counter(pfName, gs->name);
        counter.walk(fn->body);
        if (counter.pfCalls == 0) {
          continue;
        }
        struct BlockInliner : public PostWalker<BlockInliner> {
          CompressDataSegments& parent;
          Builder& builder;
          Name pfName;
          Name gsName;
          ArrayNewData* ad1;
          ArrayNewData* ad2;
          Index off1, len1, off2, len2;
          bool canEliminateGlobal;
          Index& replacedCalls;
          BlockInliner(CompressDataSegments& parent,
                       Builder& builder,
                       Name pfName,
                       Name gsName,
                       ArrayNewData* ad1,
                       ArrayNewData* ad2,
                       Index off1,
                       Index len1,
                       Index off2,
                       Index len2,
                       bool canEliminateGlobal,
                       Index& replacedCalls)
            : parent(parent), builder(builder), pfName(pfName), gsName(gsName),
              ad1(ad1), ad2(ad2), off1(off1), len1(len1), off2(off2),
              len2(len2), canEliminateGlobal(canEliminateGlobal),
              replacedCalls(replacedCalls) {}
          void visitBlock(Block* curr) {
            for (Index i = 0; i < curr->list.size(); i++) {
              auto* ls = curr->list[i]->dynCast<LocalSet>();
              if (!ls || ls->isTee()) {
                continue;
              }
              auto* c = ls->value->dynCast<Call>();
              if (!c || c->target != pfName) {
                continue;
              }
              auto* newAd1 = builder.makeArrayNewData(
                ad1->type.getHeapType(),
                ad1->segment,
                ExpressionManipulator::copy(c->operands[off1], *parent.module),
                ExpressionManipulator::copy(c->operands[len1], *parent.module));
              newAd1->type = ad1->type;
              auto* newAd2 = builder.makeArrayNewData(
                ad2->type.getHeapType(),
                ad2->segment,
                ExpressionManipulator::copy(c->operands[off2], *parent.module),
                ExpressionManipulator::copy(c->operands[len2], *parent.module));
              newAd2->type = ad2->type;
              replacedCalls++;
              if (canEliminateGlobal && i + 1 < curr->list.size()) {
                if (auto* ls2 = curr->list[i + 1]->dynCast<LocalSet>()) {
                  if (!ls2->isTee()) {
                    if (auto* gg = ls2->value->dynCast<GlobalGet>()) {
                      if (gg->name == gsName) {
                        ls->value = newAd1;
                        ls2->value = newAd2;
                        continue;
                      }
                    }
                  }
                }
              }
              ls->value = newAd1;
              curr->list[i] =
                builder.makeBlock({builder.makeGlobalSet(gsName, newAd2), ls});
            }
          }
        } inliner(*this,
                  builder,
                  pfName,
                  gs->name,
                  ad1,
                  ad2,
                  off1->index,
                  len1->index,
                  off2->index,
                  len2->index,
                  counter.pfCalls == 1 && counter.gsGets == 1,
                  replacedCalls);
        inliner.walk(fn->body);
      }
      if (replacedCalls == itCalls->second.size()) {
        pf->body = builder.makeUnreachable();
      }
    }

    // Step 2: Build AST parent/slot maps and symbol tables.
    struct ParentInfo {
      Expression* parent = nullptr;
      Expression** slot = nullptr;
      bool inLoop = false;
    };
    struct FuncAnalysis {
      std::unordered_map<Expression*, ParentInfo> parents;
      std::unordered_map<Expression*, Index> postOrder;
      Index counter = 0;
      std::unordered_map<Index, std::vector<LocalGet*>> localGets;
      std::unordered_map<Index, std::vector<LocalSet*>> localSets;
      std::vector<ArrayNewData*> i8News;
    };
    std::unordered_map<Function*, FuncAnalysis> funcData;
    std::unordered_map<Name, std::vector<std::pair<Function*, GlobalGet*>>>
      globalGets;
    std::unordered_map<Name, std::vector<std::pair<Function*, GlobalSet*>>>
      globalSets;
    std::unordered_map<Name, std::vector<std::pair<Function*, Call*>>> allCalls;
    std::unordered_set<Name> escapedFuncs;
    std::unordered_set<Name> exportedGlobals;

    for (auto& exp : module->exports) {
      if (exp->kind == ExternalKind::Function) {
        escapedFuncs.insert(*exp->getInternalName());
      } else if (exp->kind == ExternalKind::Global) {
        exportedGlobals.insert(*exp->getInternalName());
      }
    }

    struct GlobalRefScanner : public PostWalker<GlobalRefScanner> {
      std::unordered_set<Name>& escapedFuncs;
      std::unordered_map<Name, std::vector<std::pair<Function*, GlobalGet*>>>&
        globalGets;
      GlobalRefScanner(
        std::unordered_set<Name>& escapedFuncs,
        std::unordered_map<Name, std::vector<std::pair<Function*, GlobalGet*>>>&
          globalGets)
        : escapedFuncs(escapedFuncs), globalGets(globalGets) {}
      void visitRefFunc(RefFunc* curr) { escapedFuncs.insert(curr->func); }
      void visitGlobalGet(GlobalGet* curr) {
        globalGets[curr->name].push_back({nullptr, curr});
      }
    } globalRefScanner(escapedFuncs, globalGets);

    for (auto& g : module->globals) {
      if (!g->imported() && g->init) {
        globalRefScanner.walk(g->init);
      }
    }
    for (auto& seg : module->elementSegments) {
      if (seg->offset) {
        globalRefScanner.walk(seg->offset);
      }
      for (auto* init : seg->data) {
        globalRefScanner.walk(init);
      }
    }

    std::function<void(
      Expression**, Expression*, bool, Function*, FuncAnalysis&)>
      walkAST = [&](Expression** slot,
                    Expression* parent,
                    bool inLoop,
                    Function* fn,
                    FuncAnalysis& fa) {
        Expression* curr = *slot;
        fa.parents[curr] = {parent, slot, inLoop};
        bool childInLoop = inLoop || curr->is<Loop>();
        for (Expression*& child : ChildIterator(curr)) {
          walkAST(&child, curr, childInLoop, fn, fa);
        }
        fa.postOrder[curr] = fa.counter++;
        if (auto* lg = curr->dynCast<LocalGet>()) {
          fa.localGets[lg->index].push_back(lg);
        } else if (auto* ls = curr->dynCast<LocalSet>()) {
          fa.localSets[ls->index].push_back(ls);
        } else if (auto* gg = curr->dynCast<GlobalGet>()) {
          globalGets[gg->name].push_back({fn, gg});
        } else if (auto* gs = curr->dynCast<GlobalSet>()) {
          globalSets[gs->name].push_back({fn, gs});
        } else if (auto* call = curr->dynCast<Call>()) {
          allCalls[call->target].push_back({fn, call});
        } else if (auto* rf = curr->dynCast<RefFunc>()) {
          escapedFuncs.insert(rf->func);
        } else if (auto* ad = curr->dynCast<ArrayNewData>()) {
          if (ad->type.getHeapType() == i8ArrayType &&
              compressed.count(ad->segment)) {
            fa.i8News.push_back(ad);
          }
        }
      };

    for (auto& fnPtr : module->functions) {
      Function* fn = fnPtr.get();
      if (fn->imported() || !fn->body) {
        continue;
      }
      walkAST(&fn->body, nullptr, false, fn, funcData[fn]);
    }

    auto unwrapUpwardsI8Ref =
      [&](Expression* start, const FuncAnalysis& fa) -> Expression* {
      Expression* cur = start;
      while (true) {
        auto it = fa.parents.find(cur);
        if (it == fa.parents.end() || !it->second.parent) {
          break;
        }
        Expression* p = it->second.parent;
        if (auto* ra = p->dynCast<RefAs>()) {
          if (ra->op == RefAsNonNull && ra->value == cur) {
            cur = ra;
            continue;
          }
        }
        if (auto* rc = p->dynCast<RefCast>()) {
          if (rc->ref == cur && rc->type.isRef() &&
              rc->type.getHeapType() == i8ArrayType) {
            cur = rc;
            continue;
          }
        }
        break;
      }
      return cur;
    };

    auto unwrapDownwardsI8Ref = [&](Expression* e) -> Expression* {
      while (true) {
        if (auto* ra = e->dynCast<RefAs>()) {
          if (ra->op == RefAsNonNull) {
            e = ra->value;
            continue;
          }
        }
        if (auto* rc = e->dynCast<RefCast>()) {
          if (rc->type.isRef() && rc->type.getHeapType() == i8ArrayType) {
            e = rc->ref;
            continue;
          }
        }
        break;
      }
      return e;
    };

    // Step 3: Discover Kind-1 and Kind-2 i8 reader helpers.
    struct Kind1Helper {
      Function* func = nullptr;
      Index pArr = 0;
      Index pIdx = 0;
      Expression** topRefSlot = nullptr;
    };
    struct Kind2Helper {
      Function* func = nullptr;
      Index pArr = 0;
      Expression** topRefSlot = nullptr;
      Expression** innerIndexPtr = nullptr;
    };
    InsertOrderedMap<Name, Kind1Helper> kind1Helpers;
    InsertOrderedMap<Name, Kind2Helper> kind2Helpers;

    for (int pass = 0; pass < 2; pass++) {
      for (auto& fnPtr : module->functions) {
        Function* fn = fnPtr.get();
        if (fn->imported() || !fn->body || escapedFuncs.count(fn->name) ||
            runtimeFuncs.count(fn->name) || kind1Helpers.count(fn->name) ||
            kind2Helpers.count(fn->name) || allCalls[fn->name].empty()) {
          continue;
        }
        const auto& fa = funcData.at(fn);
        for (Index pArr = 0; pArr < fn->getNumParams(); pArr++) {
          if (!fn->getLocalType(pArr).isRef()) {
            continue;
          }
          auto itGets = fa.localGets.find(pArr);
          if (itGets == fa.localGets.end() || itGets->second.size() != 1 ||
              fa.localSets.count(pArr)) {
            continue;
          }
          LocalGet* arrGet = itGets->second[0];
          Expression* topRef = unwrapUpwardsI8Ref(arrGet, fa);
          const auto& topP = fa.parents.at(topRef);
          if (!topP.parent) {
            continue;
          }
          Expression* readOp = nullptr;
          Expression** innerIndexPtr = nullptr;
          if (auto* ag = topP.parent->dynCast<ArrayGet>()) {
            if (ag->ref == topRef && ag->ref->type.isRef() &&
                ag->ref->type.getHeapType() == i8ArrayType) {
              readOp = ag;
              innerIndexPtr = &ag->index;
            }
          } else if (auto* call = topP.parent->dynCast<Call>()) {
            auto itH = kind1Helpers.find(call->target);
            if (itH != kind1Helpers.end() &&
                itH->second.pArr < call->operands.size() &&
                call->operands[itH->second.pArr] == topRef) {
              readOp = call;
              innerIndexPtr = &call->operands[itH->second.pIdx];
            }
          }
          if (!readOp || !innerIndexPtr) {
            continue;
          }
          Expression* idxExpr = *innerIndexPtr;
          std::vector<LocalGet*> candGets;
          if (auto* lg = idxExpr->dynCast<LocalGet>()) {
            candGets.push_back(lg);
          } else if (auto* add = idxExpr->dynCast<Binary>()) {
            if (add->op == AddInt32) {
              if (auto* lgR = add->right->dynCast<LocalGet>()) {
                candGets.push_back(lgR);
              }
              if (auto* lgL = add->left->dynCast<LocalGet>()) {
                candGets.push_back(lgL);
              }
            }
          }
          bool foundKind1 = false;
          for (LocalGet* candGet : candGets) {
            Index cand = candGet->index;
            if (cand >= fn->getNumParams() || cand == pArr ||
                fn->getLocalType(cand) != Type::i32) {
              continue;
            }
            const auto& cGets = fa.localGets.at(cand);
            auto itCSets = fa.localSets.find(cand);
            bool noSets =
              (itCSets == fa.localSets.end() || itCSets->second.empty());
            if (cGets.size() == 1 && noSets) {
              kind1Helpers[fn->name] = {fn, pArr, cand, topP.slot};
              foundKind1 = true;
              break;
            }
            if (idxExpr == candGet && !fa.parents.at(readOp).inLoop &&
                cGets[0] == candGet && !noSets &&
                itCSets->second[0]->value == readOp &&
                fa.parents.at(readOp).parent == itCSets->second[0]) {
              kind1Helpers[fn->name] = {fn, pArr, cand, topP.slot};
              foundKind1 = true;
              break;
            }
          }
          if (foundKind1) {
            break;
          }
          if (pass == 1 && !fa.parents.at(readOp).inLoop) {
            kind2Helpers[fn->name] = {fn, pArr, topP.slot, innerIndexPtr};
            break;
          }
        }
      }
    }

    struct ReadOnlyUse {
      Expression** refPtr = nullptr;
      Expression** indexPtr = nullptr;
      bool isKind2 = false;
    };

    InsertOrderedMap<Name, Kind2Helper> activeKind2Helpers;

    auto isReadOnlyI8Use = [&](Expression* expr,
                               const FuncAnalysis& fa,
                               ReadOnlyUse& out,
                               bool allowAllKind2 = false) -> bool {
      Expression* topRef = unwrapUpwardsI8Ref(expr, fa);
      auto it = fa.parents.find(topRef);
      if (it == fa.parents.end() || !it->second.parent) {
        return false;
      }
      Expression* parent = it->second.parent;
      if (auto* ag = parent->dynCast<ArrayGet>()) {
        if (ag->ref == topRef && ag->ref->type.isRef() &&
            ag->ref->type.getHeapType() == i8ArrayType) {
          out = {it->second.slot, &ag->index, false};
          return true;
        }
      } else if (auto* call = parent->dynCast<Call>()) {
        auto it1 = kind1Helpers.find(call->target);
        if (it1 != kind1Helpers.end() &&
            it1->second.pArr < call->operands.size() &&
            call->operands[it1->second.pArr] == topRef) {
          out = {&call->operands[it1->second.pArr],
                 &call->operands[it1->second.pIdx],
                 false};
          return true;
        }
        const auto& k2Map = allowAllKind2 ? kind2Helpers : activeKind2Helpers;
        auto it2 = k2Map.find(call->target);
        if (it2 != k2Map.end() && it2->second.pArr < call->operands.size() &&
            call->operands[it2->second.pArr] == topRef) {
          out = {&call->operands[it2->second.pArr], nullptr, true};
          return true;
        }
      }
      return false;
    };

    // Verify which Kind-2 helpers have all callers passing a read-only i8
    // slice from a compressed segment.
    for (auto& [hName, h] : kind2Helpers) {
      const auto& calls = allCalls[hName];
      if (calls.empty()) {
        continue;
      }
      bool allValid = true;
      for (auto& [callerFn, call] : calls) {
        if (!callerFn || h.pArr >= call->operands.size()) {
          allValid = false;
          break;
        }
        const auto& callerFa = funcData.at(callerFn);
        Expression* arg = unwrapDownwardsI8Ref(call->operands[h.pArr]);
        if (auto* ad = arg->dynCast<ArrayNewData>()) {
          if (ad->type.getHeapType() != i8ArrayType ||
              !compressed.count(ad->segment) || !ad->offset->is<Const>() ||
              EffectAnalyzer(getPassOptions(), *module, ad->size)
                .hasSideEffects()) {
            allValid = false;
            break;
          }
        } else if (auto* lg = arg->dynCast<LocalGet>()) {
          if (callerFn->isParam(lg->index)) {
            allValid = false;
            break;
          }
          auto itS = callerFa.localSets.find(lg->index);
          if (itS == callerFa.localSets.end() || itS->second.size() != 1 ||
              itS->second[0]->isTee()) {
            allValid = false;
            break;
          }
          auto* adL = itS->second[0]->value->dynCast<ArrayNewData>();
          if (!adL || adL->type.getHeapType() != i8ArrayType ||
              !compressed.count(adL->segment)) {
            allValid = false;
            break;
          }
          for (LocalGet* otherGet : callerFa.localGets.at(lg->index)) {
            ReadOnlyUse dummy;
            if (!isReadOnlyI8Use(otherGet, callerFa, dummy, true)) {
              allValid = false;
              break;
            }
          }
          if (!allValid) {
            break;
          }
        } else {
          allValid = false;
          break;
        }
      }
      if (allValid) {
        activeKind2Helpers[hName] = h;
      }
    }

    // Step 4: Eliminate lazy-init i8 table globals (Pattern a).
    struct LazyGlobalInfo {
      Name name;
      Function* initFunc = nullptr;
      If* iff = nullptr;
      GlobalSet* gs = nullptr;
      int32_t totalOffset = 0;
      GlobalGet* condGet = nullptr;
      std::vector<ReadOnlyUse> uses;
    };
    std::vector<LazyGlobalInfo> lazyGlobals;
    std::unordered_set<Name> lazyGlobalNames;

    for (auto& gPtr : module->globals) {
      Global* g = gPtr.get();
      if (g->imported() || !g->mutable_ || exportedGlobals.count(g->name) ||
          g->type != Type(i8ArrayType, Nullable) || !g->init ||
          !g->init->is<RefNull>()) {
        continue;
      }
      auto itSets = globalSets.find(g->name);
      auto itGets = globalGets.find(g->name);
      if (itSets == globalSets.end() || itSets->second.size() != 1 ||
          itGets == globalGets.end() || itGets->second.size() < 2) {
        continue;
      }
      auto [initFn, gs] = itSets->second[0];
      if (!initFn) {
        continue;
      }
      auto* ad = gs->value->dynCast<ArrayNewData>();
      if (!ad || ad->type.getHeapType() != i8ArrayType ||
          !compressed.count(ad->segment) || !ad->offset->is<Const>() ||
          EffectAnalyzer(getPassOptions(), *module, ad->size)
            .hasSideEffects()) {
        continue;
      }
      const auto& initFa = funcData.at(initFn);
      Expression* gsParent = initFa.parents.at(gs).parent;
      If* iff = gsParent ? gsParent->dynCast<If>() : nullptr;
      if (!iff && gsParent && gsParent->is<Block>()) {
        Expression* blkParent = initFa.parents.at(gsParent).parent;
        iff = blkParent ? blkParent->dynCast<If>() : nullptr;
      }
      if (!iff || iff->ifFalse) {
        continue;
      }
      auto* rn = iff->condition->dynCast<RefIsNull>();
      auto* condGet = rn ? rn->value->dynCast<GlobalGet>() : nullptr;
      if (!condGet || condGet->name != g->name) {
        continue;
      }
      std::vector<ReadOnlyUse> roUses;
      bool allReadsOk = true;
      for (auto [useFn, gg] : itGets->second) {
        if (!useFn) {
          allReadsOk = false;
          break;
        }
        if (gg == condGet) {
          continue;
        }
        ReadOnlyUse u;
        if (!isReadOnlyI8Use(gg, funcData.at(useFn), u)) {
          allReadsOk = false;
          break;
        }
        roUses.push_back(u);
      }
      if (!allReadsOk || roUses.empty()) {
        continue;
      }
      int32_t totalOffset = ad->offset->cast<Const>()->value.geti32() +
                            int32_t(compressed.at(ad->segment).base);
      lazyGlobals.push_back(
        {g->name, initFn, iff, gs, totalOffset, condGet, std::move(roUses)});
      lazyGlobalNames.insert(g->name);
    }

    for (auto& lg : lazyGlobals) {
      const auto& initFa = funcData.at(lg.initFunc);
      for (auto& u : lg.uses) {
        if (u.isKind2) {
          *u.refPtr = builder.makeConst(lg.totalOffset);
        } else {
          *u.refPtr = builder.makeCall(getFuncName, {}, i8Ref);
          *u.indexPtr = addOffset(
            *u.indexPtr, builder.makeConst(lg.totalOffset), builder);
        }
      }
      GlobalSet* nextGs = nullptr;
      if (auto* tb = lg.iff->ifTrue->dynCast<Block>()) {
        for (Expression* stmt : tb->list) {
          if (auto* otherGs = stmt->dynCast<GlobalSet>()) {
            if (otherGs != lg.gs && !lazyGlobalNames.count(otherGs->name)) {
              Global* otherG = module->getGlobalOrNull(otherGs->name);
              if (otherG && !otherG->imported() && otherG->mutable_ &&
                  otherG->type.isRef() && otherG->type.isNullable() &&
                  otherG->init && otherG->init->is<RefNull>()) {
                nextGs = otherGs;
                break;
              }
            }
          }
        }
      }
      if (nextGs) {
        *initFa.parents.at(lg.gs).slot = builder.makeNop();
        lg.condGet->name = nextGs->name;
        lg.condGet->type = module->getGlobal(nextGs->name)->type;
      } else {
        *initFa.parents.at(lg.iff).slot = builder.makeNop();
      }
      module->removeGlobal(lg.name);
    }

    // Step 5: Eliminate local and inline read-only i8 slices (Pattern b).
    for (auto& fnPtr : module->functions) {
      Function* fn = fnPtr.get();
      if (fn->imported() || !fn->body || runtimeFuncs.count(fn->name)) {
        continue;
      }
      auto& fa = funcData.at(fn);
      for (ArrayNewData* ad : fa.i8News) {
        auto itP = fa.parents.find(ad);
        if (itP == fa.parents.end() || !itP->second.parent) {
          continue;
        }
        Expression* p = itP->second.parent;
        int32_t base = int32_t(compressed.at(ad->segment).base);
        bool sizeHasEffects =
          EffectAnalyzer(getPassOptions(), *module, ad->size).hasSideEffects();

        // Case 5a: Assigned to a local via (local.set $col (array.new_data ..))
        if (auto* ls = p->dynCast<LocalSet>()) {
          if (ls->isTee() || ls->value != ad) {
            continue;
          }
          Index localIdx = ls->index;
          std::vector<LocalGet*> targetGets;
          const auto& allSets = fa.localSets.at(localIdx);
          auto itAllGets = fa.localGets.find(localIdx);
          if (itAllGets == fa.localGets.end() || itAllGets->second.empty()) {
            continue;
          }
          if (!fn->isParam(localIdx) && allSets.size() == 1) {
            targetGets = itAllGets->second;
          } else if (auto* topBlk = fn->body->dynCast<Block>()) {
            if (fa.parents.at(ls).parent == topBlk && allSets.back() == ls) {
              Index lsPO = fa.postOrder.at(ls);
              for (LocalGet* g : itAllGets->second) {
                if (fa.postOrder.at(g) > lsPO) {
                  targetGets.push_back(g);
                }
              }
            }
          }
          if (targetGets.empty()) {
            continue;
          }
          std::vector<ReadOnlyUse> roUses;
          bool allOk = true;
          for (LocalGet* g : targetGets) {
            ReadOnlyUse u;
            if (!isReadOnlyI8Use(g, fa, u)) {
              allOk = false;
              break;
            }
            roUses.push_back(u);
          }
          if (!allOk) {
            continue;
          }
          Expression** lsSlot = fa.parents.at(ls).slot;
          if (auto* cOff = ad->offset->dynCast<Const>();
              cOff && !sizeHasEffects) {
            int32_t totalOffset = cOff->value.geti32() + base;
            *lsSlot = builder.makeNop();
            for (auto& u : roUses) {
              if (u.isKind2) {
                *u.refPtr = builder.makeConst(totalOffset);
              } else {
                *u.refPtr = builder.makeCall(getFuncName, {}, i8Ref);
                *u.indexPtr = addOffset(
                  *u.indexPtr, builder.makeConst(totalOffset), builder);
              }
            }
          } else {
            Index offLocal = Builder::addVar(fn, Type::i32);
            Expression* totalOffExpr = ad->offset;
            if (base != 0) {
              totalOffExpr = builder.makeBinary(
                AddInt32, totalOffExpr, builder.makeConst(base));
            }
            Expression* setOff = builder.makeLocalSet(offLocal, totalOffExpr);
            if (sizeHasEffects) {
              *lsSlot = builder.makeBlock({setOff, builder.makeDrop(ad->size)});
            } else {
              *lsSlot = setOff;
            }
            for (auto& u : roUses) {
              if (u.isKind2) {
                *u.refPtr = builder.makeLocalGet(offLocal, Type::i32);
              } else {
                *u.refPtr = builder.makeCall(getFuncName, {}, i8Ref);
                *u.indexPtr =
                  addOffset(*u.indexPtr,
                            builder.makeLocalGet(offLocal, Type::i32),
                            builder);
              }
            }
          }
          continue;
        }

        // Case 5b: Inline read-only i8 ArrayNewData.
        if (p->is<GlobalSet>()) {
          continue;
        }
        auto* cOff = ad->offset->dynCast<Const>();
        if (!cOff || sizeHasEffects) {
          continue;
        }
        ReadOnlyUse u;
        if (!isReadOnlyI8Use(ad, fa, u)) {
          continue;
        }
        int32_t totalOffset = cOff->value.geti32() + base;
        if (u.isKind2) {
          *u.refPtr = builder.makeConst(totalOffset);
        } else {
          *u.refPtr = builder.makeCall(getFuncName, {}, i8Ref);
          *u.indexPtr =
            addOffset(*u.indexPtr, builder.makeConst(totalOffset), builder);
        }
      }
    }

    // Finalize active Kind-2 helpers whose callers now pass i32 byte offsets.
    for (auto& [hName, h] : activeKind2Helpers) {
      bool allCallersI32 = true;
      for (auto& [_, call] : allCalls[hName]) {
        if (h.pArr >= call->operands.size() ||
            call->operands[h.pArr]->type != Type::i32) {
          allCallersI32 = false;
          break;
        }
      }
      if (!allCallersI32) {
        continue;
      }
      auto paramsType = h.func->getParams();
      std::vector<Type> params(paramsType.begin(), paramsType.end());
      params[h.pArr] = Type::i32;
      h.func->setParams(Type(params));
      *h.topRefSlot = builder.makeCall(getFuncName, {}, i8Ref);
      *h.innerIndexPtr = addOffset(
        *h.innerIndexPtr, builder.makeLocalGet(h.pArr, Type::i32), builder);
    }

    // Step 6: Sink (call $segments$get) into Kind-1 helpers when 100% of their
    // callers pass (call $segments$get) as pArr, and remove parameter pArr.
    std::unordered_set<Name> sunkKind1;
    bool changed = true;
    bool allowInliningIntoKind1 = false;
    while (changed) {
      changed = false;
      for (auto& [hName, h] : kind1Helpers) {
        if (sunkKind1.count(hName)) {
          continue;
        }
        auto& calls = allCalls[hName];
        if (calls.empty()) {
          continue;
        }
        Index getterCalls = 0;
        std::vector<size_t> nonGetterIdxs;
        for (size_t ci = 0; ci < calls.size(); ci++) {
          auto* call = calls[ci].second;
          if (h.pArr < call->operands.size()) {
            if (auto* argCall = call->operands[h.pArr]->dynCast<Call>()) {
              if (argCall->target == getFuncName) {
                getterCalls++;
                continue;
              }
            }
          }
          nonGetterIdxs.push_back(ci);
        }
        if (!nonGetterIdxs.empty() && getterCalls >= 2 &&
            nonGetterIdxs.size() <= 2 && h.pArr == 0 &&
            h.func->getNumParams() == 3) {
          if (auto* ag = h.func->body->dynCast<ArrayGet>()) {
            if (auto* add = ag->index->dynCast<Binary>()) {
              auto* lg1 = add->left->dynCast<LocalGet>();
              auto* lg2 = add->right->dynCast<LocalGet>();
              if (add->op == AddInt32 && lg1 && lg2 && lg1->index == 1 &&
                  lg2->index == 2) {
                bool canInlineAll = true;
                for (size_t ci : nonGetterIdxs) {
                  Function* cFn = calls[ci].first;
                  Call* c = calls[ci].second;
                  if (!cFn || !funcData.count(cFn) ||
                      !funcData.at(cFn).parents.count(c)) {
                    canInlineAll = false;
                    break;
                  }
                  if (!allowInliningIntoKind1 && kind1Helpers.count(cFn->name) &&
                      !sunkKind1.count(cFn->name)) {
                    canInlineAll = false;
                    break;
                  }
                }
                if (canInlineAll) {
                  for (auto it = nonGetterIdxs.rbegin();
                       it != nonGetterIdxs.rend();
                       ++it) {
                    Function* cFn = calls[*it].first;
                    Call* c = calls[*it].second;
                    Expression* arrArg = c->operands[0];
                    RefCast* rc = nullptr;
                    if (!arrArg->type.isRef() ||
                        arrArg->type.getHeapType() != i8ArrayType) {
                      rc = builder.makeRefCast(
                        arrArg, Type(i8ArrayType, Nullable));
                      arrArg = rc;
                    }
                    auto* inlined = builder.makeArrayGet(
                      arrArg,
                      builder.makeBinary(
                        AddInt32, c->operands[1], c->operands[2]),
                      ag->order,
                      ag->type,
                      ag->signed_);
                    *funcData.at(cFn).parents.at(c).slot = inlined;
                    auto itCallerH1 = kind1Helpers.find(cFn->name);
                    if (itCallerH1 != kind1Helpers.end() &&
                        itCallerH1->second.topRefSlot == &c->operands[0]) {
                      itCallerH1->second.topRefSlot =
                        rc ? &rc->ref : &inlined->ref;
                    }
                    calls.erase(calls.begin() + *it);
                  }
                  nonGetterIdxs.clear();
                }
              }
            }
          }
        }
        if (!nonGetterIdxs.empty() || calls.empty()) {
          continue;
        }
        *h.topRefSlot = builder.makeCall(getFuncName, {}, i8Ref);
        struct LocalShift : public PostWalker<LocalShift> {
          Index removed;
          LocalShift(Index removed) : removed(removed) {}
          void visitLocalGet(LocalGet* curr) {
            if (curr->index > removed) {
              curr->index--;
            }
          }
          void visitLocalSet(LocalSet* curr) {
            if (curr->index > removed) {
              curr->index--;
            }
          }
        } shifter(h.pArr);
        shifter.walk(h.func->body);
        auto paramsType = h.func->getParams();
        std::vector<Type> params(paramsType.begin(), paramsType.end());
        params.erase(params.begin() + h.pArr);
        h.func->setParams(Type(params));
        Builder::clearLocalNames(h.func);
        for (auto& [_, call] : calls) {
          call->operands.erase(call->operands.begin() + h.pArr);
        }
        sunkKind1.insert(hName);
        changed = true;
      }
      if (!changed && !allowInliningIntoKind1) {
        allowInliningIntoKind1 = true;
        changed = true;
      }
    }
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
