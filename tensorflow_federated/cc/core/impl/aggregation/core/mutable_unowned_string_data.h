/*
 * Copyright 2024 Google LLC
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

#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_MUTABLE_UNOWNED_STRING_DATA_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_MUTABLE_UNOWNED_STRING_DATA_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace tensorflow_federated {
namespace aggregation {

// MutableUnownedStringData implements TensorData by wrapping std::vector and
// using it as backing storage for string_view objects. MutableUnownedStringData
// can be mutated using std::vector methods. The MutableUnownedStringData object
// does not own the string values. Use MutableStringData instead if you want
// a TensorData object that owns the strings.
class MutableUnownedStringData : public std::vector<absl::string_view>,
                                 public TensorData {
 public:
  // Derive constructors from the base vector class.
  using std::vector<absl::string_view>::vector;

  ~MutableUnownedStringData() override = default;

  // Implementation of the base class methods.
  size_t byte_size() const override {
    return this->size() * sizeof(absl::string_view);
  }
  const void* data() const override {
    return this->std::vector<absl::string_view>::data();
  }

  // Copy the MutableUnownedStringData into a string.
  std::string EncodeContent() {
    return std::string(reinterpret_cast<const char*>(this->data()),
                       this->byte_size());
  }

  // Create and return a new MutableUnownedStringData populated with the data
  // from content.
  static std::unique_ptr<MutableUnownedStringData> CreateFromEncodedContent(
      const std::string& content) {
    const absl::string_view* data =
        reinterpret_cast<const absl::string_view*>(content.data());
    return std::make_unique<MutableUnownedStringData>(
        data, data + content.size() / sizeof(absl::string_view));
  }
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_MUTABLE_UNOWNED_STRING_DATA_H_
