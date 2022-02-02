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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATA_BACKEND_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATA_BACKEND_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

// A dynamically-dispatched interface for resolving references to data within
// the `DataExecutor`.
class DataBackend {
 public:
  // Resolves a `tensorflow_federated::v0::Data` object to a concrete
  // `tensorflow_federated::v0::Value` proto, writing the result to `value_out`.
  //
  // This function must be safe to call concurrently from multiple threads.
  virtual absl::Status ResolveToValue(const v0::Data& data_reference,
                                      const v0::Type& data_type,
                                      v0::Value& value_out) = 0;

  // Resolves a `tensorflow_federated::v0::Data` object to a concrete
  // `tensorflow_federated::v0::Value` proto, returning the result as a new
  // proto object.
  absl::StatusOr<v0::Value> ResolveToValue(const v0::Data& data_reference,
                                           const v0::Type& data_type) {
    v0::Value out;
    TFF_TRY(ResolveToValue(data_reference, data_type, out));
    return out;
  }

  virtual ~DataBackend() {}
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATA_BACKEND_H_
