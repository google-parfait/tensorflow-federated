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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_EXAMPLES_CUSTOM_DATA_BACKEND_DATA_BACKEND_EXAMPLE_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_EXAMPLES_CUSTOM_DATA_BACKEND_DATA_BACKEND_EXAMPLE_H_

#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/executors/data_backend.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated_examples {

// An example implementation of `DataBackend` used to show Python interop.
class DataBackendExample : public tensorflow_federated::DataBackend {
 public:
  absl::Status ResolveToValue(
      const tensorflow_federated::v0::Data& data_reference,
      tensorflow_federated::v0::Value& value_out) final;
};

}  // namespace tensorflow_federated_examples

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_EXAMPLES_CUSTOM_DATA_BACKEND_DATA_BACKEND_EXAMPLE_H_
