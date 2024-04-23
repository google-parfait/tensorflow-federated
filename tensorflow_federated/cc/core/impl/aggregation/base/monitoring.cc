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

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"

#include <sstream>
#include <string>

#include "tensorflow_federated/cc/core/impl/aggregation/base/base_name.h"

namespace tensorflow_federated {

namespace internal {

StatusBuilder::StatusBuilder(StatusCode code, const char* file, int line)
    : file_(file), line_(line), code_(code), message_() {}

StatusBuilder::StatusBuilder(StatusBuilder const& other)
    : file_(other.file_),
      line_(other.line_),
      code_(other.code_),
      message_(other.message_.str()) {}

StatusBuilder::operator Status() {
  auto message_str = message_.str();
  if (code_ != OK) {
    std::ostringstream status_message;
    status_message << "(at " << BaseName(file_) << ":" << line_ << message_str;
    message_str = status_message.str();
  }
  return Status(code_, message_str);
}

}  // namespace internal

}  // namespace tensorflow_federated
