/*
 * Copyright 2022 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/base/base_name.h"

#include <cstring>
#include <string>

namespace tensorflow_federated {

#ifdef _WIN32
constexpr char kPathSeparator = '\\';
#else
constexpr char kPathSeparator = '/';
#endif

std::string BaseName(const std::string& path) {
  // Note: the code below needs to be compatible with baremetal build with
  // nanolibc. Therefore it is implemented via the standard "C" library strrchr.
  const char* separator_ptr = strrchr(path.c_str(), kPathSeparator);
  if (separator_ptr == nullptr) return path;

  return path.substr((separator_ptr - path.c_str()) + 1);
}

}  // namespace tensorflow_federated
