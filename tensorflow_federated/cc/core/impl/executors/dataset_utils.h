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

#include <vector>

#include "absl/status/statusor.h"
#include "federated_language/proto/computation.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

absl::StatusOr<tensorflow::Tensor> GraphDefTensorFromSequence(
    const v0::Value::Sequence& sequence_pb);

// Extracts output_types and output_shapes from a serialized GraphDef tensor
// by parsing the GraphDef and finding dataset nodes with these attributes.
absl::StatusOr<std::pair<tensorflow::DataTypeVector,
                         std::vector<tensorflow::PartialTensorShape>>>
ExtractOutputTypesAndShapesFromGraphDef(
    const tensorflow::Tensor& graph_def_tensor);

// Iterates a dataset represented as a serialized GraphDef string tensor
// and returns all elements as a vector of tensor vectors.
//
// Each inner vector represents one element of the dataset. The outer vector
// contains all elements. `output_types` and `output_shapes` describe the
// per-element tensor types and shapes.
absl::StatusOr<std::vector<std::vector<tensorflow::Tensor>>>
IterateDatasetFromGraphDef(
    const tensorflow::Tensor& graph_def_tensor,
    const tensorflow::DataTypeVector& output_types,
    const std::vector<tensorflow::PartialTensorShape>& output_shapes);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATASET_UTILS_H_
