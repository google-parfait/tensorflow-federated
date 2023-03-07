/* Copyright 2023, The TensorFlow Federated Authors.

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

#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"

#include <string>

namespace tensorflow_federated {

std::string GetNodeName(absl::string_view tensor_name) {
  absl::string_view::size_type pos = tensor_name.find(':');
  if (pos == absl::string_view::npos) {
    return std::string(tensor_name);
  } else {
    return std::string(tensor_name.substr(0, pos));
  }
}

}  // namespace tensorflow_federated
