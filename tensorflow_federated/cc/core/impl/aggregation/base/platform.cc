/*
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_federated/cc/core/impl/aggregation/base/platform.h"

#include <stdlib.h>
#include <sys/stat.h>

#include <fstream>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <direct.h>
#endif

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"

namespace tensorflow_federated {

namespace {
#ifdef _WIN32
constexpr char kPathSeparator[] = "\\";
#else
constexpr char kPathSeparator[] = "/";
#endif
}  // namespace

std::string ConcatPath(absl::string_view path1, absl::string_view path2) {
  if (path1.empty()) {
    return std::string(path2);
  }
  return absl::StrCat(path1, kPathSeparator, path2);
}

absl::string_view StripTrailingPathSeparator(absl::string_view path) {
  return absl::StripSuffix(path, kPathSeparator);
}

namespace internal {

template <typename T>
absl::StatusOr<T> ReadFile(absl::string_view file_name) {
  auto file_name_str = std::string(file_name);
  std::ifstream is(file_name_str);
  if (!is) {
    return absl::InternalError(
        absl::StrCat("cannot read file ", file_name_str));
  }
  std::ostringstream buffer;
  buffer << is.rdbuf();
  if (!is) {
    return absl::InternalError(
        absl::StrCat("error reading file ", file_name_str));
  }
  return static_cast<T>(buffer.str());
}

}  // namespace internal

absl::StatusOr<std::string> ReadFileToString(absl::string_view file_name) {
  return internal::ReadFile<std::string>(file_name);
}

absl::StatusOr<absl::Cord> ReadFileToCord(absl::string_view file_name) {
  return internal::ReadFile<absl::Cord>(file_name);
}

absl::Status WriteStringToFile(absl::string_view file_name,
                               absl::string_view content) {
  auto file_name_str = std::string(file_name);
  std::ofstream os(file_name_str);
  if (!os) {
    return absl::InternalError(
        absl::StrCat("cannot create file ", file_name_str));
  }
  os << content;
  if (!os) {
    return absl::InternalError(
        absl::StrCat("error writing to file ", file_name_str));
  }
  return absl::OkStatus();
}

absl::Status WriteCordToFile(absl::string_view file_name,
                             const absl::Cord& content) {
  auto file_name_str = std::string(file_name);
  std::ofstream os(file_name_str);
  if (!os) {
    return absl::InternalError(
        absl::StrCat("cannot create file ", file_name_str));
  }
  for (absl::string_view chunk : content.Chunks()) {
    os << chunk;
    if (!os) {
      return absl::InternalError(
          absl::StrCat("error writing to file ", file_name_str));
    }
  }
  return absl::OkStatus();
}

bool FileExists(absl::string_view file_name) {
  struct stat info;
  return stat(std::string(file_name).c_str(), &info) == 0;
}

std::string GetDataPath(absl::string_view relative_path) {
  return std::string(relative_path);
}

}  // namespace tensorflow_federated
