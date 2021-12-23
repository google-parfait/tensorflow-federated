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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATASET_FROM_TENSOR_STRUCTURES_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATASET_FROM_TENSOR_STRUCTURES_H_

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow_federated {

// Creates a TensorFlow dataset from a list of tensor structures and serializes
// it into a single string tensor.
//
// Requirements: `tensor_structures` must be a list of lists of tensors. Each
// sub-list represents a structure that will be yielded from the dataset, hence
// all structures must have the same shape. Corresponding elements
// across structures (e.g. the third element of every tensor_structures[i])
// must have the same shape and dtype.
absl::StatusOr<tensorflow::Tensor> DatasetFromTensorStructures(
    absl::Span<const std::vector<tensorflow::Tensor>> tensor_structures);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATASET_FROM_TENSOR_STRUCTURES_H_
