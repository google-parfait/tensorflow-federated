/* Copyright 2022, The TensorFlow Federated Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STRUCT_TRAVERSAL_ORDER_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STRUCT_TRAVERSAL_ORDER_H_

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"

namespace tensorflow_federated {

inline std::vector<std::string> NamesFromStructType(
    const v0::StructType& struct_type) {
  std::vector<std::string> names;
  for (const auto& struct_el : struct_type.element()) {
    if (struct_el.name().length()) {
      names.emplace_back(struct_el.name());
    }
  }
  return names;
}

using NameAndIndex = std::pair<std::string, uint32_t>;

inline absl::StatusOr<std::vector<uint32_t>> TFNestTraversalOrderFromStruct(
    const v0::StructType& struct_type) {
  auto struct_names = NamesFromStructType(struct_type);
  std::vector<uint32_t> traversal_order(struct_type.element_size(), 0);
  // Initialize traversal order as iteration order over the structure.
  std::iota(traversal_order.begin(), traversal_order.end(), 0);
  if (struct_names.empty()) {
    // Unnamed structures are always traversed by tf.nest in iteration order.
    return traversal_order;
  } else {
    if (struct_names.size() != struct_type.element_size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot compute traversal order from partially-named "
                       "structure. Found a structure with ",
                       struct_names.size(), "names, but structure has ",
                       struct_type.element_size(), "elements"));
    }
    // tf.data uses tf.nest to flatten structures, and tf.nest flattens in
    // *sorted* order for ODicts. Value serialization ensures that ODict is
    // the only Python representation of structures here (e.g., not attr.s,
    // which tf.nest treats differently). So we construct traversal order
    // here by sorting the indices according to the names present in the struct.
    std::sort(traversal_order.begin(), traversal_order.end(),
              [&](uint32_t i1, uint32_t i2) {
                return struct_names[i1] < struct_names[i2];
              });
  }
  return traversal_order;
}

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STRUCT_TRAVERSAL_ORDER_H_
