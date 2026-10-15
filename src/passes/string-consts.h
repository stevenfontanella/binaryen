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

#ifndef wasm_passes_string_consts_h
#define wasm_passes_string_consts_h

//
// Reading the string constants that --string-lowering leaves in a module: the
// imported externref globals from module "string.const" whose contents live in
// the "string.consts" JSON custom section, indexed by the import base name.
//

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "support/json.h"
#include "wasm.h"

namespace wasm::StringConsts {

inline const char* ImportModule = "string.const";
inline const char* SectionName = "string.consts";

// The contents of every string constant import, as little-endian WTF-16 code
// units, by global name and in module order. Returns nothing if the module
// has no string constants or the section is not in the expected form.
inline std::optional<std::vector<std::pair<Name, std::string>>>
read(Module& module) {
  auto iter = std::find_if(
    module.customSections.begin(),
    module.customSections.end(),
    [&](CustomSection& section) { return section.name == SectionName; });
  if (iter == module.customSections.end()) {
    return std::nullopt;
  }
  std::string copy(iter->data.begin(), iter->data.end());
  json::Value array;
  array.parse(copy.data(), json::Value::WTF16);
  if (!array.isArray()) {
    return std::nullopt;
  }
  std::vector<std::pair<Name, std::string>> strings;
  for (auto& global : module.globals) {
    if (!global->imported() || global->module != ImportModule) {
      continue;
    }
    if (global->mutable_ || global->type != Type(HeapType::ext, NonNullable)) {
      return std::nullopt;
    }
    auto base = global->base.toString();
    if (base.empty() || !std::all_of(base.begin(), base.end(), [](char c) {
          return c >= '0' && c <= '9';
        })) {
      return std::nullopt;
    }
    auto index = std::stoull(base);
    if (index >= array.size() || !array[index]->isString()) {
      return std::nullopt;
    }
    strings.emplace_back(global->name, array[index]->getIString().toString());
  }
  if (strings.empty()) {
    return std::nullopt;
  }
  return strings;
}

// Lexicographic order of two strings by WTF-16 code unit.
inline bool less(const std::string& a, const std::string& b) {
  size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i + 1 < n; i += 2) {
    uint16_t x = uint8_t(a[i]) | (uint16_t(uint8_t(a[i + 1])) << 8);
    uint16_t y = uint8_t(b[i]) | (uint16_t(uint8_t(b[i + 1])) << 8);
    if (x != y) {
      return x < y;
    }
  }
  return a.size() < b.size();
}

} // namespace wasm::StringConsts

#endif // wasm_passes_string_consts_h
