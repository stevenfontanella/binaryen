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
// Data layout: all strings are laid out as one stream of UTF-16 code units.
// Each group of strings (a column, or a chunk of direct uses) stores its
// strings back-to-back, followed by one unit per string holding its length
// (0xFFFF escapes to two more units, little-endian). The unit stream is
// LZ77-compressed into a byte stream that is stored in one passive data
// segment: a token byte holds a kind in the low 2 bits and a length in the
// high 6 bits (63 means that one more byte follows and is added), and the
// kinds are
//
//   0: a run of narrow literals, 1 byte per unit (all units < 256)
//   1: a run of wide literals, 2 bytes per unit, little-endian
//   2: a match of length + 3 units, 1 byte distance - 1 follows
//   3: a match of length + 3 units, 2 bytes distance - 1 follow
//
// On the first use of any string the stream is decoded once into a mutable
// i16 array kept in a global, and the segment is dropped. On the first use of
// a group, a helper reads the group's lengths from the array and builds an
// i32 array of (start, end) pairs, so no per-string constants appear in the
// code or in global initializers. A string is created with
// fromCharCodeArray(array, start, end).
//
// A size model skips the transformation when it would not help;
// --pass-arg=string-consts-to-data-always@1 ignores the model.
// --pass-arg=string-consts-to-data-compress@0 emits only literal runs.
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
  bool compress = true;
  InsertOrderedMap<Name, StringInfo> strings;
  InsertOrderedMap<Name, ColumnInfo> columns;

  Name segmentName;
  Name dataGlobalName;
  Name decodeFuncName;
  Name fillFuncName;
  Name fromCharCodeArray;
  HeapType i8ArrayType = HeapTypes::getMutI8Array();
  HeapType i16ArrayType = HeapTypes::getMutI16Array();
  HeapType tableType = HeapType(Array(Field(Type::i32, Mutable)));
  Type refExtern = Type(HeapType::ext, NonNullable);
  // Whether fromCharCodeArray takes a non-nullable array.
  bool arrayParamNonNullable = false;
  // The decoded unit stream.
  std::vector<uint16_t> units;
  Index numHelpers = 0;

  void run(Module* module_) override {
    module = module_;
    if (!module->features.hasGC() || !module->features.hasBulkMemory()) {
      return;
    }
    always = !getArgumentOrDefault("string-consts-to-data-always", "").empty();
    compress =
      getArgumentOrDefault("string-consts-to-data-compress", "1") != "0";

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
    if (!always && modelSaving <= 400 + (rawStrings.empty() ? 0 : 80)) {
      return;
    }

    // Build everything.
    Builder builder(*module);
    prepareNames();
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

    // Compress the unit stream, add the segment and the decoder.
    auto segmentData = compressUnits();
    module->addDataSegment(Builder::makeDataSegment(
      segmentName, Name(), nullptr, segmentData.data(), segmentData.size()));
    makeDecoder(segmentData.size());

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

  // The bytes of an import entry and the JSON entry for a string.
  Index getImportBytes(const std::string& wtf16) {
    std::stringstream json;
    String::printEscapedJSON(json, wtf16, /*rawUTF8=*/true);
    // Module name (length + 12 chars), field name (length + up to 5 digits),
    // kind, type (2) and mutability; the JSON entry and its comma.
    return 13 + 5 + 4 + json.str().size() + 1;
  }

  // The (uncompressed) bytes of the data and the length unit for a string.
  Index getEntryBytes(const std::string& wtf16) {
    Index bytes = 2;
    for (size_t i = 0; i < wtf16.size(); i += 2) {
      bytes += wtf16[i + 1] ? 2 : 1;
    }
    return bytes;
  }

  // Emits the strings of a group into the unit stream, followed by their
  // lengths. Returns the offsets (in units) of the strings and of the lengths.
  struct GroupLayout {
    Index stringsOffset;
    Index lengthsOffset;
  };

  GroupLayout addGroup(std::vector<std::string*>& contents) {
    GroupLayout layout;
    layout.stringsOffset = units.size();
    std::vector<uint32_t> lengths;
    for (auto* content : contents) {
      auto& wtf16 = *content;
      Index numUnits = wtf16.size() / 2;
      lengths.push_back(numUnits);
      for (Index i = 0; i < numUnits; i++) {
        units.push_back(uint8_t(wtf16[2 * i]) |
                        (uint16_t(uint8_t(wtf16[2 * i + 1])) << 8));
      }
    }
    layout.lengthsOffset = units.size();
    for (auto length : lengths) {
      if (length < 0xFFFF) {
        units.push_back(length);
      } else {
        units.push_back(0xFFFF);
        units.push_back(length & 0xFFFF);
        units.push_back(length >> 16);
      }
    }
    return layout;
  }

  // LZ77 compression of the unit stream into the token format described at
  // the top of the file.

  // The largest length a token can hold: 63 + one extra byte.
  static constexpr Index MaxTokenLength = 63 + 255;
  static constexpr Index MinMatch = 3;
  static constexpr Index MaxMatch = MaxTokenLength + MinMatch;
  static constexpr Index ShortDistance = 256;
  static constexpr Index MaxDistance = 65536;
  static constexpr Index HashBits = 17;
  static constexpr Index MaxChain = 1024;

  struct Match {
    Index length = 0;
    Index distance = 0;
  };

  static Index matchCost(Index length, Index distance) {
    return 1 + (distance <= ShortDistance ? 1 : 2) +
           (length - MinMatch >= 63 ? 1 : 0);
  }

  static Index literalCost(uint16_t unit) { return unit < 256 ? 1 : 2; }

  // Finds, for every position, the longest match and the longest match with a
  // short distance.
  void findMatches(std::vector<Match>& longMatches,
                   std::vector<Match>& shortMatches) {
    Index n = units.size();
    longMatches.resize(n);
    shortMatches.resize(n);
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
      Match& best = longMatches[i];
      Match& bestShort = shortMatches[i];
      while (j != Index(-1) && chain++ < MaxChain) {
        Index distance = i - j;
        if (distance > MaxDistance) {
          break;
        }
        Index length = 0;
        while (length < maxLength && units[j + length] == units[i + length]) {
          length++;
        }
        if (length >= MinMatch) {
          if (length > best.length) {
            best = {length, distance};
          }
          if (distance <= ShortDistance && length > bestShort.length) {
            bestShort = {length, distance};
          }
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

  std::vector<char> compressUnits() {
    Index n = units.size();
    std::vector<Match> longMatches, shortMatches;
    if (compress) {
      findMatches(longMatches, shortMatches);
    } else {
      longMatches.resize(n);
      shortMatches.resize(n);
    }

    // Optimal parsing, backwards. Two states: after a literal (a literal run
    // is open) and after a match (a literal needs a new run token).
    std::vector<Index> costAfterLiteral(n + 1, 0), costAfterMatch(n + 1, 0);
    // The chosen match at each position for each state; length 0 is a
    // literal.
    std::vector<Match> choiceAfterLiteral(n), choiceAfterMatch(n);
    for (Index i = n; i > 0;) {
      i--;
      Match bestMatch;
      Index bestCost = Index(-1);
      auto consider = [&](Match& match) {
        for (Index length = match.length; length >= MinMatch; length--) {
          Index cost =
            matchCost(length, match.distance) + costAfterMatch[i + length];
          if (cost < bestCost) {
            bestCost = cost;
            bestMatch = {length, match.distance};
          }
        }
      };
      consider(longMatches[i]);
      consider(shortMatches[i]);
      Index literal = literalCost(units[i]) + costAfterLiteral[i + 1];
      if (literal <= bestCost) {
        costAfterLiteral[i] = literal;
        choiceAfterLiteral[i] = Match();
      } else {
        costAfterLiteral[i] = bestCost;
        choiceAfterLiteral[i] = bestMatch;
      }
      if (literal + 1 <= bestCost) {
        costAfterMatch[i] = literal + 1;
        choiceAfterMatch[i] = Match();
      } else {
        costAfterMatch[i] = bestCost;
        choiceAfterMatch[i] = bestMatch;
      }
    }

    // Emit the tokens.
    std::vector<char> out;
    auto emitToken = [&](Index kind, Index length) {
      assert(length <= MaxTokenLength);
      if (length < 63) {
        out.push_back(char(kind | (length << 2)));
      } else {
        out.push_back(char(kind | (63 << 2)));
        out.push_back(char(length - 63));
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
          out.push_back(char(pending[k] & 0xFF));
          if (wide[i]) {
            out.push_back(char(pending[k] >> 8));
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
      out.push_back(char(distance & 0xFF));
      if (distance >= ShortDistance) {
        out.push_back(char(distance >> 8));
      }
      afterLiteral = false;
      i += choice.length;
    }
    flushLiterals();
    return out;
  }

  void prepareNames() {
    fromCharCodeArray = getFromCharCodeArray();
    segmentName = Names::getValidDataSegmentName(*module, "string.consts");
    Builder builder(*module);
    dataGlobalName = Names::getValidGlobalName(*module, "string.consts$data");
    module->addGlobal(builder.makeGlobal(dataGlobalName,
                                         Type(i16ArrayType, Nullable),
                                         builder.makeRefNull(i16ArrayType),
                                         Builder::Mutable));
    decodeFuncName =
      Names::getValidFunctionName(*module, "string.consts$decode");
    fillFuncName = Names::getValidFunctionName(*module, "string.consts$fill");
  }

  Name getFromCharCodeArray() {
    for (auto& func : module->functions) {
      if (func->imported() && func->module == WasmStringsModule &&
          func->base == "fromCharCodeArray") {
        auto params = func->getParams();
        if (params.size() == 3 && params[0].isRef() &&
            params[0].getHeapType().isArray()) {
          i16ArrayType = params[0].getHeapType();
          arrayParamNonNullable = params[0].isNonNullable();
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

  // The decoder, which expands the segment into the unit array once:
  //
  //   (func $decode
  //     (local $src (ref $i8)) (local $dst (ref $i16))
  //     (local $p i32) (local $q i32) (local $tok i32) (local $len i32)
  //     (local $d i32)
  //     (local.set $src (array.new_data $i8 $seg (i32.const 0) (i32.const S)))
  //     (local.set $dst (array.new_default $i16 (i32.const N)))
  //     (loop $next
  //       (local.set $tok (array.get_u $i8 (local.get $src) (local.get $p)))
  //       (local.set $p (i32.add (local.get $p) (i32.const 1)))
  //       (local.set $len (i32.shr_u (local.get $tok) (i32.const 2)))
  //       (if (i32.eq (local.get $len) (i32.const 63))
  //         (then
  //           (local.set $len (i32.add (i32.const 63) (..byte at $p..)))
  //           (local.set $p (i32.add (local.get $p) (i32.const 1)))))
  //       (local.set $tok (i32.and (local.get $tok) (i32.const 3)))
  //       (if (i32.lt_u (local.get $tok) (i32.const 2))
  //         (then
  //           (loop $lit
  //             (if (local.get $tok)
  //               (then
  //                 (array.set $i16 (local.get $dst) (local.get $q)
  //                   (i32.or (..byte at $p..)
  //                           (i32.shl (..byte at $p + 1..) (i32.const 8))))
  //                 (local.set $p (i32.add (local.get $p) (i32.const 2))))
  //               (else
  //                 (array.set $i16 (local.get $dst) (local.get $q)
  //                   (..byte at $p..))
  //                 (local.set $p (i32.add (local.get $p) (i32.const 1)))))
  //             (local.set $q (i32.add (local.get $q) (i32.const 1)))
  //             (br_if $lit (local.tee $len (i32.sub (local.get $len) 1)))))
  //         (else
  //           (local.set $d (i32.add (..byte at $p..) (i32.const 1)))
  //           (local.set $p (i32.add (local.get $p) (i32.const 1)))
  //           (if (i32.eq (local.get $tok) (i32.const 3))
  //             (then
  //               (local.set $d (i32.add (local.get $d)
  //                 (i32.shl (..byte at $p..) (i32.const 8))))
  //               (local.set $p (i32.add (local.get $p) (i32.const 1)))))
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
  void makeDecoder(Index segmentSize) {
    Builder builder(*module);
    Index src = 0, dst = 1, p = 2, q = 3, tok = 4, len = 5, d = 6;
    Type i8Ref(i8ArrayType, NonNullable);
    Type i16Ref(i16ArrayType, NonNullable);
    auto i32 = Type::i32;
    auto get = [&](Index index) { return builder.makeLocalGet(index, i32); };
    auto add = [&](Expression* a, Expression* b) {
      return builder.makeBinary(AddInt32, a, b);
    };
    auto addConst = [&](Index index, int32_t value) {
      return builder.makeLocalSet(index,
                                  add(get(index), builder.makeConst(value)));
    };
    auto byteAt = [&](Expression* index) {
      return builder.makeArrayGet(builder.makeLocalGet(src, i8Ref),
                                  index,
                                  MemoryOrder::Unordered,
                                  i32,
                                  false);
    };
    auto setDst = [&](Expression* value) {
      return builder.makeArraySet(builder.makeLocalGet(dst, i16Ref),
                                  get(q),
                                  value,
                                  MemoryOrder::Unordered);
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
      builder.makeArrayNewData(i8ArrayType,
                               segmentName,
                               builder.makeConst(0),
                               builder.makeConst(int32_t(segmentSize))));
    auto* setDstArray = builder.makeLocalSet(
      dst,
      builder.makeArrayNew(i16ArrayType,
                           builder.makeConst(int32_t(units.size()))));

    auto* readTok = builder.makeLocalSet(tok, byteAt(get(p)));
    auto* setLen = builder.makeLocalSet(
      len, builder.makeBinary(ShrUInt32, get(tok), builder.makeConst(2)));
    auto* extendLen = builder.makeIf(
      builder.makeBinary(EqInt32, get(len), builder.makeConst(63)),
      builder.makeBlock(
        {builder.makeLocalSet(len, add(builder.makeConst(63), byteAt(get(p)))),
         addConst(p, 1)}));
    auto* maskTok = builder.makeLocalSet(
      tok, builder.makeBinary(AndInt32, get(tok), builder.makeConst(3)));

    Name litLabel("lit");
    auto* wideUnit = builder.makeBinary(
      OrInt32,
      byteAt(get(p)),
      builder.makeBinary(ShlInt32,
                         byteAt(add(get(p), builder.makeConst(1))),
                         builder.makeConst(8)));
    auto* literal = builder.makeLoop(
      litLabel,
      builder.makeBlock(
        {builder.makeIf(
           get(tok),
           builder.makeBlock({setDst(wideUnit), addConst(p, 2)}),
           builder.makeBlock({setDst(byteAt(get(p))), addConst(p, 1)})),
         addConst(q, 1),
         countDown(litLabel)}));

    Name copyLabel("copy");
    auto* setD =
      builder.makeLocalSet(d, add(byteAt(get(p)), builder.makeConst(1)));
    auto* extendD = builder.makeIf(
      builder.makeBinary(EqInt32, get(tok), builder.makeConst(3)),
      builder.makeBlock(
        {builder.makeLocalSet(d,
                              add(get(d),
                                  builder.makeBinary(ShlInt32,
                                                     byteAt(get(p)),
                                                     builder.makeConst(8)))),
         addConst(p, 1)}));
    auto* copy = builder.makeLoop(
      copyLabel,
      builder.makeBlock({setDst(builder.makeArrayGet(
                           builder.makeLocalGet(dst, i16Ref),
                           builder.makeBinary(SubInt32, get(q), get(d)),
                           MemoryOrder::Unordered,
                           i32,
                           false)),
                         addConst(q, 1),
                         countDown(copyLabel)}));
    auto* match = builder.makeBlock(
      {setD, addConst(p, 1), extendD, addConst(len, MinMatch), copy});

    auto* dispatch = builder.makeIf(
      builder.makeBinary(LtUInt32, get(tok), builder.makeConst(2)),
      literal,
      match);
    Name nextLabel("next");
    auto* next = builder.makeBreak(
      nextLabel,
      nullptr,
      builder.makeBinary(
        LtUInt32, get(q), builder.makeConst(int32_t(units.size()))));
    auto* loop = builder.makeLoop(
      nextLabel,
      builder.makeBlock(
        {readTok, addConst(p, 1), setLen, extendLen, maskTok, dispatch, next}));
    auto* body =
      builder.makeBlock({setSrc,
                         setDstArray,
                         loop,
                         builder.makeDataDrop(segmentName),
                         builder.makeGlobalSet(
                           dataGlobalName, builder.makeLocalGet(dst, i16Ref))});
    module->addFunction(
      builder.makeFunction(decodeFuncName,
                           Signature(Type::none, Type::none),
                           {i8Ref, i16Ref, i32, i32, i32, i32, i32},
                           body));
  }

  // Reads the unit array global, which has been decoded at this point.
  Expression* getData(bool nonNull) {
    Builder builder(*module);
    Expression* get =
      builder.makeGlobalGet(dataGlobalName, Type(i16ArrayType, Nullable));
    if (!nonNull) {
      return get;
    }
    return builder.makeRefAs(RefAsNonNull, get);
  }

  // The shared table filler, which turns the length units of a group into a
  // table of (start, end) pairs:
  //
  //   (func $fill (param $table (ref $table)) (param $p i32) (param $off i32)
  //     (local $i i32) (local $v i32) (local $data (ref $i16))
  //     (if (ref.is_null (global.get $data)) (then (call $decode)))
  //     (local.set $data (ref.as_non_null (global.get $data)))
  //     (loop $next
  //       (array.set $table (local.get $table) (local.get $i) (local.get $off))
  //       (local.set $v (array.get_u $i16 (local.get $data) (local.get $p)))
  //       (local.set $p (i32.add (local.get $p) (i32.const 1)))
  //       (if (i32.eq (local.get $v) (i32.const 65535))
  //         (then
  //           (local.set $v (i32.or ..2 units at $p, little-endian..))
  //           (local.set $p (i32.add (local.get $p) (i32.const 2)))))
  //       (local.set $off (i32.add (local.get $off) (local.get $v)))
  //       (array.set $table (local.get $table)
  //         (i32.or (local.get $i) (i32.const 1)) (local.get $off))
  //       (br_if $next (i32.lt_u (local.tee $i (i32.add (local.get $i) 2))
  //                              (array.len (local.get $table))))))
  //
  void makeFiller() {
    Builder builder(*module);
    Index table = 0, p = 1, off = 2, i = 3, v = 4, data = 5;
    Type tableRef(tableType, NonNullable);
    Type i16Ref(i16ArrayType, NonNullable);
    auto i32 = Type::i32;
    auto get = [&](Index index) { return builder.makeLocalGet(index, i32); };
    auto readUnit = [&](Expression* index) {
      return builder.makeArrayGet(builder.makeLocalGet(data, i16Ref),
                                  index,
                                  MemoryOrder::Unordered,
                                  i32,
                                  false);
    };
    auto* decode =
      builder.makeIf(builder.makeRefIsNull(builder.makeGlobalGet(
                       dataGlobalName, Type(i16ArrayType, Nullable))),
                     builder.makeCall(decodeFuncName, {}, Type::none));
    auto* setData = builder.makeLocalSet(data, getData(true));
    auto* setStart = builder.makeArraySet(builder.makeLocalGet(table, tableRef),
                                          get(i),
                                          get(off),
                                          MemoryOrder::Unordered);
    auto* readV = builder.makeLocalSet(v, readUnit(get(p)));
    auto* incP = builder.makeLocalSet(
      p, builder.makeBinary(AddInt32, get(p), builder.makeConst(1)));
    auto* longV = builder.makeBinary(
      OrInt32,
      readUnit(get(p)),
      builder.makeBinary(
        ShlInt32,
        readUnit(builder.makeBinary(AddInt32, get(p), builder.makeConst(1))),
        builder.makeConst(16)));
    auto* escape = builder.makeIf(
      builder.makeBinary(EqInt32, get(v), builder.makeConst(0xFFFF)),
      builder.makeBlock(
        {builder.makeLocalSet(v, longV),
         builder.makeLocalSet(
           p, builder.makeBinary(AddInt32, get(p), builder.makeConst(2)))}));
    auto* advance =
      builder.makeLocalSet(off, builder.makeBinary(AddInt32, get(off), get(v)));
    auto* setEnd = builder.makeArraySet(
      builder.makeLocalGet(table, tableRef),
      builder.makeBinary(OrInt32, get(i), builder.makeConst(1)),
      get(off),
      MemoryOrder::Unordered);
    Name label("next");
    auto* next = builder.makeBreak(
      label,
      nullptr,
      builder.makeBinary(
        LtUInt32,
        builder.makeLocalTee(
          i, builder.makeBinary(AddInt32, get(i), builder.makeConst(2)), i32),
        builder.makeArrayLen(builder.makeLocalGet(table, tableRef))));
    auto* loop = builder.makeLoop(
      label,
      builder.makeBlock(
        {setStart, readV, incP, escape, advance, setEnd, next}));
    module->addFunction(
      builder.makeFunction(fillFuncName,
                           Signature({tableRef, i32, i32}, Type::none),
                           {i32, i32, i16Ref},
                           builder.makeBlock({decode, setData, loop})));
  }

  // Creates the table global for a group of strings and a helper that creates
  // string k:
  //
  //   (func $helper (param $k i32) (result (ref extern))
  //     (if (ref.is_null (global.get $table))
  //       (then
  //         (global.set $table
  //           (local.tee $t (array.new_default $table (i32.const 2 * N))))
  //         (call $fill (local.get $t) (i32.const lengthsOffset)
  //                                    (i32.const stringsOffset))))
  //     (call $fromCharCodeArray
  //       (ref.as_non_null (global.get $data))
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
                           builder.makeConst(int32_t(layout.lengthsOffset)),
                           builder.makeConst(int32_t(layout.stringsOffset))},
                          Type::none)}));
    auto readTable = [&](bool end) {
      Expression* index = builder.makeBinary(
        ShlInt32, builder.makeLocalGet(k, i32), builder.makeConst(1));
      if (end) {
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
      fromCharCodeArray,
      {getData(arrayParamNonNullable), readTable(false), readTable(true)},
      refExtern);
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
