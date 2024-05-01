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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_PLATFORM_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_PLATFORM_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"

// This file defines platform dependent utilities.

namespace tensorflow_federated {

/**
 * Concatenates two file path components using platform specific separator.
 */
std::string ConcatPath(absl::string_view path1, absl::string_view path2);

/**
 * Strips a single platform specific path separator from the end of a path if
 * it is present, returns the original path otherwise.
 */
absl::string_view StripTrailingPathSeparator(absl::string_view path);

/**
 * Reads file content into string.
 */
absl::StatusOr<std::string> ReadFileToString(absl::string_view file_name);

/**
 * Reads file content into absl::Cord.
 */
absl::StatusOr<absl::Cord> ReadFileToCord(absl::string_view file_name);

/**
 * Writes string content into file.
 */
absl::Status WriteStringToFile(absl::string_view file_name,
                               absl::string_view content);

/**
 * Writes cord content into file.
 */
absl::Status WriteCordToFile(absl::string_view file_name,
                             const absl::Cord& content);

/**
 * Returns true if the file exists.
 */
bool FileExists(absl::string_view file_name);

/**
 *  Get absolute path given `relative_path`
 */
std::string GetDataPath(absl::string_view relative_path);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_PLATFORM_H_
