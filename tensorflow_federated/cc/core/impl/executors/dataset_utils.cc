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

#include "tensorflow_federated/cc/core/impl/executors/dataset_utils.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "federated_language/proto/computation.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow_federated/cc/core/impl/executors/dataset_from_tensor_structures.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"

namespace tensorflow_federated {

absl::StatusOr<tensorflow::Tensor> GraphDefTensorFromSequence(
    const v0::Value::Sequence& sequence_pb) {
  std::vector<std::vector<tensorflow::Tensor>> tensor_structures;
  for (const v0::Value::Sequence::Element& element_pb : sequence_pb.element()) {
    std::vector<tensorflow::Tensor> tensors;
    for (const federated_language::Array& array_pb : element_pb.flat_value()) {
      // Repeated fields are used for strings and scalars to maintain
      // compatibility with TensorFlow.
      if (array_pb.has_content()) {
        tensors.push_back(TFF_TRY(TensorFromArrayContent(array_pb)));
      } else {
        tensors.push_back(TFF_TRY(TensorFromArray(array_pb)));
      }
    }
    tensor_structures.push_back(tensors);
  }
  return DatasetFromTensorStructures(tensor_structures);
}

absl::StatusOr<std::unique_ptr<tensorflow::data::standalone::Dataset>>
DatasetFromGraphDefTensor(const tensorflow::Tensor& tensor) {
  tensorflow::GraphDef dataset_graph;
  if (!dataset_graph.ParseFromString(tensor.flat<tensorflow::tstring>()(0))) {
    return absl::InternalError(
        "Error parsing Dataset GraphDef from TFF Value serialized GraphDef "
        "string.");
  }
  std::unique_ptr<tensorflow::data::standalone::Dataset> dataset;
  tensorflow::data::standalone::Dataset::Params params;
  absl::Status status = tensorflow::data::standalone::Dataset::FromGraph(
      params, dataset_graph, &dataset);
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Dataset FromGraph creation failed while converting "
                     "TFF Value to tf data Dataset. Error: ",
                     status.message()));
  }

  return std::move(dataset);
}
}  // namespace tensorflow_federated
