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
// Moves lowered string constants from imports into a passive data segment.
//
// After --string-lowering every string constant is an imported global from
// module "string.const" whose contents live in the "string.consts" JSON custom
// section. Each such string costs an import entry (module and field names,
// type, mutability) plus its JSON entry. This pass replaces the imports whose
// uses can be rewritten with a data segment holding the raw code units, and
// creates the JS string at the first use with the wasm:js-string
// "fromCharCodeArray" builtin.
//
// Two kinds of uses are rewritten:
//
//  * Columns: immutable globals initialized by an array.new_fixed of an
//    externref array whose operands are all string imports, and which are read
//    only by array.get in function code (the tables produced by
//    --defer-global-allocations). The column global is removed and
//    (array.get $col (global.get $col) (idx)) becomes (call $helper (idx)).
//    The caller is expected to cache the result (as the deferred-allocation
//    factories do).
//
//  * Direct uses: (global.get $string) in function code becomes
//    (call $helper (i32.const k)), where the helper caches the created string
//    in a mutable externref array.
//
// Strings used anywhere else (other global initializers, segments, exports)
// stay imports. The JSON custom section is rebuilt from the remaining
// "string.const" imports, which are renumbered consecutively, so a loader that
// provides imports["string.const"][i] = json[i] keeps working.
//
// Data layout: all strings share one passive data segment. Each group of
// strings (a column, or a chunk of direct uses) stores its strings
// back-to-back, followed by one byte per string encoding
// (length << 1) | isI16 (0xFF escapes to a 4-byte little-endian value).
// Strings whose code units all fit in a byte are stored as one byte per unit
// and expanded into an i16 array at runtime; other strings are stored as
// little-endian i16. On the first use of a group, a helper reads the length
// bytes and builds an i32 array of (offset, packed length) pairs, so no
// per-string constants appear in the code or in global initializers.
//
// A size model skips the transformation when it would not help;
// --pass-arg=string-consts-to-data-always@1 ignores the model.
//

#include <algorithm>

#include "ir/module-utils.h"
#include "ir/names.h"
#include "pass.h"
#include "passes/string-utils.h"
#include "support/insert_ordered.h"
#include "support/json.h"
#include "support/string.h"
#include "wasm-builder.h"
#include "wasm.h"

namespace wasm {

namespace {

// Maximum number of strings per group of direct uses, which keeps the indexes
// at most 2 bytes as an SLEB.
constexpr Index MaxChunkSize = 8192;

const char* StringConstsModule = "string.const";
const char* StringConstsSection = "string.consts";

struct StringConstsToData : public Pass {
  struct StringInfo {
    // The contents, as little-endian WTF-16 code units.
    std::string wtf16;
    // Uses in function code.
    Index codeUses = 0;
    // Order of the first code use.
    Index firstUse = Index(-1);
    // Whether the string is used in a convertible column.
    bool inColumn = false;
    // Whether the string has a use that cannot be rewritten.
    bool pinned = false;
  };

  struct ColumnInfo {
    Global* global;
    ArrayNewFixed* init;
    // Reads of the column, all of the form (array.get (global.get $col) ..).
    std::vector<Expression**> reads;
    // All global.gets of the column in function code.
    Index gets = 0;
    bool pinned = false;
  };

  struct FunctionInfo {
    std::vector<Expression**> stringGets;
    std::vector<Expression**> columnReads;
    std::unordered_map<Name, Index> columnGets;
  };

  Module* module;
  bool always = false;
  InsertOrderedMap<Name, StringInfo> strings;
  InsertOrderedMap<Name, ColumnInfo> columns;

  Name segmentName;
  Name makeFuncName;
  Name fillFuncName;
  HeapType i8ArrayType = HeapTypes::getMutI8Array();
  HeapType i16ArrayType = HeapTypes::getMutI16Array();
  HeapType tableType = HeapType(Array(Field(Type::i32, Mutable)));
  Type refExtern = Type(HeapType::ext, NonNullable);
  std::vector<char> segmentData;
  Index numHelpers = 0;

  void run(Module* module_) override {
    module = module_;
    if (!module->features.hasGC() || !module->features.hasBulkMemory()) {
      return;
    }
    always = !getArgumentOrDefault("string-consts-to-data-always", "").empty();

    if (!readStrings()) {
      return;
    }

    findColumnsAndPinnedUses();

    // Scan the function code, in parallel.
    ModuleUtils::ParallelFunctionAnalysis<FunctionInfo> analysis(
      *module, [&](Function* func, FunctionInfo& info) {
        if (func->imported()) {
          return;
        }
        struct Scanner : public PostWalker<Scanner> {
          StringConstsToData& parent;
          FunctionInfo& info;
          Scanner(StringConstsToData& parent, FunctionInfo& info)
            : parent(parent), info(info) {}
          void visitGlobalGet(GlobalGet* curr) {
            if (parent.strings.count(curr->name)) {
              info.stringGets.push_back(getCurrentPointer());
            } else if (parent.columns.count(curr->name)) {
              info.columnGets[curr->name]++;
            }
          }
          void visitArrayGet(ArrayGet* curr) {
            if (auto* get = curr->ref->dynCast<GlobalGet>()) {
              if (parent.columns.count(get->name)) {
                info.columnReads.push_back(getCurrentPointer());
              }
            }
          }
        };
        Scanner(*this, info).walk(func->body);
      });

    // Merge the per-function information in a deterministic order.
    std::vector<Expression**> stringGets;
    for (auto& func : module->functions) {
      auto& info = analysis.map[func.get()];
      for (auto** ptr : info.stringGets) {
        auto& string = strings[(*ptr)->cast<GlobalGet>()->name];
        if (string.codeUses == 0) {
          string.firstUse = stringGets.size();
        }
        string.codeUses++;
        stringGets.push_back(ptr);
      }
      for (auto** ptr : info.columnReads) {
        auto& column =
          columns[(*ptr)->cast<ArrayGet>()->ref->cast<GlobalGet>()->name];
        column.reads.push_back(ptr);
      }
      for (auto& [name, count] : info.columnGets) {
        columns[name].gets += count;
      }
    }

    // A column is convertible only if every global.get of it is a read.
    auto pinColumn = [&](ColumnInfo& column) {
      column.pinned = true;
      for (auto* value : column.init->values) {
        strings[value->cast<GlobalGet>()->name].pinned = true;
      }
    };
    for (auto& [name, column] : columns) {
      if (column.pinned || column.gets != column.reads.size() ||
          column.reads.empty()) {
        pinColumn(column);
      }
    }
    std::vector<ColumnInfo*> convertibleColumns;
    int64_t modelSaving = 0;
    for (auto& [name, column] : columns) {
      if (column.pinned) {
        continue;
      }
      // The helper and the table global.
      int64_t saving = -60;
      for (auto* value : column.init->values) {
        auto& string = strings[value->cast<GlobalGet>()->name];
        // The global.get in the column.
        saving += 3;
        saving -= getEntryBytes(string.wtf16);
        if (!string.pinned && !string.inColumn) {
          // The import can be removed (count it once per string).
          saving += getImportBytes(string.wtf16);
        }
        string.inColumn = true;
      }
      if (always || saving > 0) {
        convertibleColumns.push_back(&column);
        modelSaving += saving;
      } else {
        pinColumn(column);
      }
    }

    // Strings with direct code uses that can be removed.
    std::vector<std::pair<Name, StringInfo*>> rawStrings;
    for (auto& [name, string] : strings) {
      if (string.pinned || string.codeUses == 0) {
        continue;
      }
      // The import is removed; each use becomes a call with a constant, which
      // is 3-4 bytes larger than the global.get.
      int64_t saving = int64_t(getImportBytes(string.wtf16)) -
                       int64_t(getEntryBytes(string.wtf16)) -
                       int64_t(string.codeUses) * 4;
      if (always || saving > 0) {
        rawStrings.emplace_back(name, &string);
        modelSaving += saving;
      } else {
        string.pinned = true;
      }
    }
    if (convertibleColumns.empty() && rawStrings.empty()) {
      return;
    }
    // The decoder and fill functions, the segment, and the raw helpers.
    if (!always && modelSaving <= 200 + (rawStrings.empty() ? 0 : 80)) {
      return;
    }

    // Build everything.
    Builder builder(*module);
    makeDecoder();
    makeFiller();

    for (auto* column : convertibleColumns) {
      std::vector<std::string*> contents;
      for (auto* value : column->init->values) {
        contents.push_back(&strings[value->cast<GlobalGet>()->name].wtf16);
      }
      auto helper = makeHelper(contents, false);
      for (auto** ptr : column->reads) {
        auto* read = (*ptr)->cast<ArrayGet>();
        *ptr = builder.makeCall(helper, {read->index}, refExtern);
      }
      module->removeGlobal(column->global->name);
    }

    if (!rawStrings.empty()) {
      std::sort(rawStrings.begin(),
                rawStrings.end(),
                [](const std::pair<Name, StringInfo*>& a,
                   const std::pair<Name, StringInfo*>& b) {
                  return a.second->firstUse < b.second->firstUse;
                });
      std::unordered_map<Name, std::pair<Name, Index>> rawIndexes;
      Index numChunks = (rawStrings.size() + MaxChunkSize - 1) / MaxChunkSize;
      for (Index chunk = 0; chunk < numChunks; chunk++) {
        Index begin = chunk * MaxChunkSize;
        Index end = std::min(Index(rawStrings.size()), begin + MaxChunkSize);
        std::vector<std::string*> contents;
        for (Index i = begin; i < end; i++) {
          contents.push_back(&rawStrings[i].second->wtf16);
        }
        auto helper = makeHelper(contents, true);
        for (Index i = begin; i < end; i++) {
          rawIndexes[rawStrings[i].first] = {helper, i - begin};
        }
      }
      for (auto** ptr : stringGets) {
        auto* get = (*ptr)->cast<GlobalGet>();
        auto iter = rawIndexes.find(get->name);
        if (iter == rawIndexes.end()) {
          continue;
        }
        auto& [helper, index] = iter->second;
        *ptr = builder.makeCall(
          helper, {builder.makeConst(int32_t(index))}, refExtern);
      }
    }

    // Add the segment.
    module->addDataSegment(Builder::makeDataSegment(
      segmentName, Name(), nullptr, segmentData.data(), segmentData.size()));

    // Remove the imports whose uses are all gone, and rebuild the JSON.
    for (auto& [name, string] : strings) {
      if (!string.pinned && (string.codeUses > 0 || string.inColumn)) {
        module->removeGlobal(name);
      }
    }
    writeStrings();
  }

private:
  // Reads the string.consts section and the imports that refer to it. Returns
  // false if there is nothing to do or the section is not in the expected
  // form.
  bool readStrings() {
    auto iter = std::find_if(module->customSections.begin(),
                             module->customSections.end(),
                             [&](CustomSection& section) {
                               return section.name == StringConstsSection;
                             });
    if (iter == module->customSections.end()) {
      return false;
    }
    std::string copy(iter->data.begin(), iter->data.end());
    json::Value array;
    array.parse(copy.data(), json::Value::WTF16);
    if (!array.isArray()) {
      return false;
    }
    for (auto& global : module->globals) {
      if (!global->imported() || global->module != StringConstsModule) {
        continue;
      }
      if (global->mutable_ || global->type != refExtern) {
        return false;
      }
      auto base = global->base.toString();
      if (base.empty() || !std::all_of(base.begin(), base.end(), [](char c) {
            return c >= '0' && c <= '9';
          })) {
        return false;
      }
      auto index = std::stoull(base);
      if (index >= array.size() || !array[index]->isString()) {
        return false;
      }
      strings[global->name].wtf16 = array[index]->getIString().toString();
    }
    return !strings.empty();
  }

  void findColumnsAndPinnedUses() {
    struct Scanner : public PostWalker<Scanner> {
      StringConstsToData& parent;
      Scanner(StringConstsToData& parent) : parent(parent) {}
      void visitGlobalGet(GlobalGet* curr) {
        if (auto iter = parent.strings.find(curr->name);
            iter != parent.strings.end()) {
          iter->second.pinned = true;
        } else if (auto iter = parent.columns.find(curr->name);
                   iter != parent.columns.end()) {
          iter->second.pinned = true;
        }
      }
    } scanner(*this);

    // Find the column candidates first, so that uses of them in other globals
    // can be noted.
    for (auto& global : module->globals) {
      if (global->imported() || global->mutable_) {
        continue;
      }
      auto* init = global->init->dynCast<ArrayNewFixed>();
      if (!init || init->values.empty()) {
        continue;
      }
      auto element = init->type.getHeapType().getArray().element;
      if (!element.type.isRef() ||
          element.type.getHeapType() != HeapType::ext) {
        continue;
      }
      bool allStrings = true;
      for (auto* value : init->values) {
        auto* get = value->dynCast<GlobalGet>();
        if (!get || !strings.count(get->name)) {
          allStrings = false;
          break;
        }
      }
      if (allStrings) {
        columns[global->name] = {global.get(), init, {}, 0, false};
      }
    }

    for (auto& global : module->globals) {
      if (global->imported() || columns.count(global->name)) {
        continue;
      }
      scanner.walk(global->init);
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
        auto name = *ex->getInternalName();
        if (auto iter = strings.find(name); iter != strings.end()) {
          iter->second.pinned = true;
        } else if (auto iter = columns.find(name); iter != columns.end()) {
          iter->second.pinned = true;
        }
      }
    }
  }

  // Size model helpers.

  bool isLatin1(const std::string& wtf16) {
    for (size_t i = 1; i < wtf16.size(); i += 2) {
      if (wtf16[i] != 0) {
        return false;
      }
    }
    return true;
  }

  // The bytes of an import entry and the JSON entry for a string.
  Index getImportBytes(const std::string& wtf16) {
    std::stringstream json;
    String::printEscapedJSON(json, wtf16, /*rawUTF8=*/true);
    // Module name (length + 12 chars), field name (length + up to 5 digits),
    // kind, type (2) and mutability; the JSON entry and its comma.
    return 13 + 5 + 4 + json.str().size() + 1;
  }

  // The bytes of the data and the length byte(s) for a string.
  Index getEntryBytes(const std::string& wtf16) {
    Index units = wtf16.size() / 2;
    Index bytes = isLatin1(wtf16) ? units : 2 * units;
    return bytes + (units < 128 ? 1 : 5);
  }

  // Emits the strings of a group into the segment, followed by their length
  // bytes. Returns the offsets of the strings and of the length bytes.
  struct GroupLayout {
    Index stringsOffset;
    Index lengthsOffset;
    Index lengthsBytes;
  };

  GroupLayout addGroup(std::vector<std::string*>& contents) {
    GroupLayout layout;
    layout.stringsOffset = segmentData.size();
    std::vector<uint32_t> packedLengths;
    for (auto* content : contents) {
      auto& wtf16 = *content;
      Index units = wtf16.size() / 2;
      // The empty string is treated as i16 so that no expansion loop runs.
      if (units > 0 && isLatin1(wtf16)) {
        packedLengths.push_back(units << 1);
        for (Index i = 0; i < units; i++) {
          segmentData.push_back(wtf16[2 * i]);
        }
      } else {
        packedLengths.push_back((units << 1) | 1);
        segmentData.insert(segmentData.end(), wtf16.begin(), wtf16.end());
      }
    }
    layout.lengthsOffset = segmentData.size();
    for (auto packed : packedLengths) {
      if (packed < 0xFF) {
        segmentData.push_back(char(packed));
      } else {
        segmentData.push_back(char(0xFF));
        for (Index i = 0; i < 4; i++) {
          segmentData.push_back(char((packed >> (8 * i)) & 0xFF));
        }
      }
    }
    layout.lengthsBytes = segmentData.size() - layout.lengthsOffset;
    return layout;
  }

  Name getFromCharCodeArray() {
    for (auto& func : module->functions) {
      if (func->imported() && func->module == WasmStringsModule &&
          func->base == "fromCharCodeArray") {
        auto params = func->getParams();
        if (params.size() == 3 && params[0].isRef() &&
            params[0].getHeapType().isArray()) {
          i16ArrayType = params[0].getHeapType();
        }
        return func->name;
      }
    }
    // Add the import.
    auto name = Names::getValidFunctionName(*module, "fromCharCodeArray");
    Builder builder(*module);
    auto* func = module->addFunction(builder.makeFunction(
      name,
      Signature({Type(i16ArrayType, Nullable), Type::i32, Type::i32},
                refExtern),
      {}));
    func->module = WasmStringsModule;
    func->base = "fromCharCodeArray";
    return name;
  }

  // The shared decoder:
  //
  //   (func $make (param $off i32) (param $packed i32) (result (ref extern))
  //     (local $len i32) (local $a8 (ref $i8)) (local $a16 (ref $i16))
  //     (local.set $len (i32.shr_u (local.get $packed) (i32.const 1)))
  //     (if (i32.and (local.get $packed) (i32.const 1))
  //       (return (call $fromCharCodeArray
  //         (array.new_data $i16 $seg (local.get $off) (local.get $len))
  //         (i32.const 0) (local.get $len))))
  //     (local.set $a8
  //       (array.new_data $i8 $seg (local.get $off) (local.get $len)))
  //     (local.set $a16 (array.new_default $i16 (local.get $len)))
  //     (loop $copy
  //       (local.set $len (i32.sub (local.get $len) (i32.const 1)))
  //       (array.set $i16 (local.get $a16) (local.get $len)
  //         (array.get_u $i8 (local.get $a8) (local.get $len)))
  //       (br_if $copy (local.get $len)))
  //     (call $fromCharCodeArray (local.get $a16) (i32.const 0)
  //       (array.len (local.get $a16))))
  //
  void makeDecoder() {
    auto fromCharCodeArray = getFromCharCodeArray();
    segmentName = Names::getValidDataSegmentName(*module, "string.consts");
    Builder builder(*module);
    Index off = 0, packed = 1, len = 2, a8 = 3, a16 = 4;
    Type i8Ref(i8ArrayType, NonNullable);
    Type i16Ref(i16ArrayType, NonNullable);
    auto i32 = Type::i32;
    auto getLen = [&]() { return builder.makeLocalGet(len, i32); };
    auto* setLen = builder.makeLocalSet(
      len,
      builder.makeBinary(
        ShrUInt32, builder.makeLocalGet(packed, i32), builder.makeConst(1)));
    auto* i16Path = builder.makeIf(
      builder.makeBinary(
        AndInt32, builder.makeLocalGet(packed, i32), builder.makeConst(1)),
      builder.makeReturn(builder.makeCall(
        fromCharCodeArray,
        {builder.makeArrayNewData(
           i16ArrayType, segmentName, builder.makeLocalGet(off, i32), getLen()),
         builder.makeConst(0),
         getLen()},
        refExtern)));
    auto* setA8 = builder.makeLocalSet(
      a8,
      builder.makeArrayNewData(
        i8ArrayType, segmentName, builder.makeLocalGet(off, i32), getLen()));
    auto* setA16 =
      builder.makeLocalSet(a16, builder.makeArrayNew(i16ArrayType, getLen()));
    Name copyLabel("copy");
    auto* loop = builder.makeLoop(
      copyLabel,
      builder.makeBlock(
        {builder.makeLocalSet(
           len, builder.makeBinary(SubInt32, getLen(), builder.makeConst(1))),
         builder.makeArraySet(
           builder.makeLocalGet(a16, i16Ref),
           getLen(),
           builder.makeArrayGet(builder.makeLocalGet(a8, i8Ref),
                                getLen(),
                                MemoryOrder::Unordered,
                                i32,
                                false),
           MemoryOrder::Unordered),
         builder.makeBreak(copyLabel, nullptr, getLen())}));
    auto* result = builder.makeCall(
      fromCharCodeArray,
      {builder.makeLocalGet(a16, i16Ref),
       builder.makeConst(0),
       builder.makeArrayLen(builder.makeLocalGet(a16, i16Ref))},
      refExtern);
    auto* body =
      builder.makeBlock({setLen, i16Path, setA8, setA16, loop, result});
    makeFuncName = Names::getValidFunctionName(*module, "string.consts$make");
    module->addFunction(builder.makeFunction(makeFuncName,
                                             Signature({i32, i32}, refExtern),
                                             {i32, i8Ref, i16Ref},
                                             body));
  }

  // The shared table filler, which turns the length bytes of a group into a
  // table of (offset, packed length) pairs:
  //
  //   (func $fill (param $table (ref $table)) (param $lengths (ref $i8))
  //               (param $off i32)
  //     (local $i i32) (local $p i32) (local $v i32)
  //     (loop $next
  //       (array.set $table (local.get $table) (local.get $i) (local.get $off))
  //       (local.set $v (array.get_u $i8 (local.get $lengths) (local.get $p)))
  //       (local.set $p (i32.add (local.get $p) (i32.const 1)))
  //       (if (i32.eq (local.get $v) (i32.const 255))
  //         (then
  //           (local.set $v (i32.or ..4 bytes at $p, little-endian..))
  //           (local.set $p (i32.add (local.get $p) (i32.const 4)))))
  //       (array.set $table (local.get $table)
  //         (i32.or (local.get $i) (i32.const 1)) (local.get $v))
  //       (local.set $off (i32.add (local.get $off)
  //         (i32.shl (i32.shr_u (local.get $v) (i32.const 1))
  //                  (i32.and (local.get $v) (i32.const 1)))))
  //       (br_if $next (i32.lt_u (local.tee $i (i32.add (local.get $i) 2))
  //                              (array.len (local.get $table))))))
  //
  void makeFiller() {
    Builder builder(*module);
    Index table = 0, lengths = 1, off = 2, i = 3, p = 4, v = 5;
    Type tableRef(tableType, NonNullable);
    Type i8Ref(i8ArrayType, NonNullable);
    auto i32 = Type::i32;
    auto get = [&](Index index) { return builder.makeLocalGet(index, i32); };
    auto readByte = [&](Expression* index) {
      return builder.makeArrayGet(builder.makeLocalGet(lengths, i8Ref),
                                  index,
                                  MemoryOrder::Unordered,
                                  i32,
                                  false);
    };
    auto* setOffset =
      builder.makeArraySet(builder.makeLocalGet(table, tableRef),
                           get(i),
                           get(off),
                           MemoryOrder::Unordered);
    auto* readV = builder.makeLocalSet(v, readByte(get(p)));
    auto* incP = builder.makeLocalSet(
      p, builder.makeBinary(AddInt32, get(p), builder.makeConst(1)));
    Expression* longV = nullptr;
    for (Index byte = 0; byte < 4; byte++) {
      Expression* index = get(p);
      if (byte > 0) {
        index =
          builder.makeBinary(AddInt32, index, builder.makeConst(int32_t(byte)));
      }
      Expression* value = readByte(index);
      if (byte > 0) {
        value = builder.makeBinary(
          ShlInt32, value, builder.makeConst(int32_t(8 * byte)));
      }
      longV = longV ? builder.makeBinary(OrInt32, longV, value) : value;
    }
    auto* escape = builder.makeIf(
      builder.makeBinary(EqInt32, get(v), builder.makeConst(255)),
      builder.makeBlock(
        {builder.makeLocalSet(v, longV),
         builder.makeLocalSet(
           p, builder.makeBinary(AddInt32, get(p), builder.makeConst(4)))}));
    auto* setLength = builder.makeArraySet(
      builder.makeLocalGet(table, tableRef),
      builder.makeBinary(OrInt32, get(i), builder.makeConst(1)),
      get(v),
      MemoryOrder::Unordered);
    auto* advance = builder.makeLocalSet(
      off,
      builder.makeBinary(
        AddInt32,
        get(off),
        builder.makeBinary(
          ShlInt32,
          builder.makeBinary(ShrUInt32, get(v), builder.makeConst(1)),
          builder.makeBinary(AndInt32, get(v), builder.makeConst(1)))));
    Name label("next");
    auto* next = builder.makeBreak(
      label,
      nullptr,
      builder.makeBinary(
        LtUInt32,
        builder.makeLocalTee(
          i, builder.makeBinary(AddInt32, get(i), builder.makeConst(2)), i32),
        builder.makeArrayLen(builder.makeLocalGet(table, tableRef))));
    auto* body = builder.makeLoop(
      label,
      builder.makeBlock(
        {setOffset, readV, incP, escape, setLength, advance, next}));
    fillFuncName = Names::getValidFunctionName(*module, "string.consts$fill");
    module->addFunction(
      builder.makeFunction(fillFuncName,
                           Signature({tableRef, i8Ref, i32}, Type::none),
                           {i32, i32, i32},
                           body));
  }

  // Creates the table global for a group of strings and a helper that creates
  // string k:
  //
  //   (func $helper (param $k i32) (result (ref extern))
  //     (if (ref.is_null (global.get $table))
  //       (then
  //         (global.set $table
  //           (local.tee $t (array.new_default $table (i32.const 2 * N))))
  //         (call $fill (local.get $t)
  //           (array.new_data $i8 $seg (i32.const lengthsOffset)
  //                                    (i32.const lengthsBytes))
  //           (i32.const stringsOffset))))
  //     (call $make
  //       (array.get $table (global.get $table) (i32.shl (local.get $k) 1))
  //       (array.get $table (global.get $table)
  //         (i32.or (i32.shl (local.get $k) 1) 1))))
  //
  // With a cache, the result is stored in a mutable externref array and
  // returned from there when present.
  Name makeHelper(std::vector<std::string*>& contents, bool cached) {
    Builder builder(*module);
    auto i32 = Type::i32;
    std::string baseName = std::string("string.consts$") +
                           (cached ? "raw$" : "col$") +
                           std::to_string(numHelpers++);
    auto layout = addGroup(contents);

    auto tableGlobalType = Type(tableType, Nullable);
    auto tableRef = Type(tableType, NonNullable);
    auto tableName = Names::getValidGlobalName(*module, baseName + "$table");
    module->addGlobal(builder.makeGlobal(tableName,
                                         tableGlobalType,
                                         builder.makeRefNull(tableType),
                                         Builder::Mutable));

    Index k = 0, t = 1, s = 2;
    auto* init = builder.makeIf(
      builder.makeRefIsNull(builder.makeGlobalGet(tableName, tableGlobalType)),
      builder.makeBlock(
        {builder.makeGlobalSet(
           tableName,
           builder.makeLocalTee(
             t,
             builder.makeArrayNew(
               tableType, builder.makeConst(int32_t(2 * contents.size()))),
             tableRef)),
         builder.makeCall(fillFuncName,
                          {builder.makeLocalGet(t, tableRef),
                           builder.makeArrayNewData(
                             i8ArrayType,
                             segmentName,
                             builder.makeConst(int32_t(layout.lengthsOffset)),
                             builder.makeConst(int32_t(layout.lengthsBytes))),
                           builder.makeConst(int32_t(layout.stringsOffset))},
                          Type::none)}));
    auto readTable = [&](bool length) {
      Expression* index = builder.makeBinary(
        ShlInt32, builder.makeLocalGet(k, i32), builder.makeConst(1));
      if (length) {
        index = builder.makeBinary(OrInt32, index, builder.makeConst(1));
      }
      return builder.makeArrayGet(
        builder.makeGlobalGet(tableName, tableGlobalType),
        index,
        MemoryOrder::Unordered,
        i32,
        false);
    };
    auto* make = builder.makeCall(
      makeFuncName, {readTable(false), readTable(true)}, refExtern);
    std::vector<Type> vars = {tableRef};
    Expression* body;
    if (cached) {
      auto cacheElementType = Type(HeapType::ext, Nullable);
      HeapType cacheType(Array(Field(cacheElementType, Mutable)));
      auto cacheGlobalType = Type(cacheType, NonNullable);
      auto cacheName = Names::getValidGlobalName(*module, baseName + "$cache");
      module->addGlobal(builder.makeGlobal(
        cacheName,
        cacheGlobalType,
        builder.makeArrayNew(cacheType,
                             builder.makeConst(int32_t(contents.size()))),
        Builder::Immutable));
      vars.push_back(refExtern);
      Name label("cached");
      auto* cacheGet =
        builder.makeArrayGet(builder.makeGlobalGet(cacheName, cacheGlobalType),
                             builder.makeLocalGet(k, i32),
                             MemoryOrder::Unordered,
                             cacheElementType);
      auto* brOn = builder.makeBrOn(BrOnNonNull, label, cacheGet);
      auto* cacheSet =
        builder.makeArraySet(builder.makeGlobalGet(cacheName, cacheGlobalType),
                             builder.makeLocalGet(k, i32),
                             builder.makeLocalTee(s, make, refExtern),
                             MemoryOrder::Unordered);
      body = builder.makeBlock(
        label,
        {brOn, init, cacheSet, builder.makeLocalGet(s, refExtern)},
        refExtern);
    } else {
      body = builder.makeBlock({init, make}, refExtern);
    }
    auto helperName = Names::getValidFunctionName(*module, baseName);
    module->addFunction(builder.makeFunction(
      helperName, Signature(i32, refExtern), std::move(vars), body));
    return helperName;
  }

  // Rebuilds the JSON section from the remaining imports, renumbering them.
  void writeStrings() {
    std::stringstream json;
    Index index = 0;
    for (auto& global : module->globals) {
      if (!global->imported() || global->module != StringConstsModule) {
        continue;
      }
      if (index > 0) {
        json << ',';
      }
      String::printEscapedJSON(
        json, strings[global->name].wtf16, /*rawUTF8=*/true);
      global->base = std::to_string(index++);
    }
    auto iter = std::find_if(module->customSections.begin(),
                             module->customSections.end(),
                             [&](CustomSection& section) {
                               return section.name == StringConstsSection;
                             });
    assert(iter != module->customSections.end());
    if (index == 0) {
      module->customSections.erase(iter);
      return;
    }
    auto str = '[' + json.str() + ']';
    iter->data = std::vector<char>(str.begin(), str.end());
  }
};

} // anonymous namespace

Pass* createStringConstsToDataPass() { return new StringConstsToData(); }

} // namespace wasm
