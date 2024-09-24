/* Copyright 2021, The TensorFlow Federated Authors.

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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATASET_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATASET_UTILS_H_

#include <memory>

#include "absl/status/statusor.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

absl::StatusOr<tensorflow::Tensor> GraphDefTensorFromSequence(
    const v0::Value::Sequence& sequence_pb);

absl::StatusOr<std::unique_ptr<tensorflow::data::standalone::Dataset>>
DatasetFromGraphDefTensor(const tensorflow::Tensor& tensor);

absl::StatusOr<std::unique_ptr<tensorflow::data::standalone::Dataset>>
DatasetFromSequence(const v0::Value::Sequence& sequence_pb);
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATASET_UTILS_H_
