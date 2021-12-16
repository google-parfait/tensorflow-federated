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
#include "tensorflow_federated/cc/core/impl/executors/dataset_conversions.h"

#include "absl/status/status.h"

namespace tensorflow_federated {

absl::StatusOr<std::unique_ptr<tensorflow::data::standalone::Dataset>>
SequenceValueToDataset(const v0::Value::Sequence& sequence_pb) {
  if (!sequence_pb.has_serialized_graph_def()) {
    return absl::UnimplementedError(
        "Conversion from TFF Value to TF Dataset requires graphdef-based "
        "serialization.");
  }
  tensorflow::GraphDef dataset_graph;
  if (!dataset_graph.ParseFromString(sequence_pb.serialized_graph_def())) {
    return absl::InternalError(
        "Error parsing Dataset GraphDef from TFF Value serialized GraphDef "
        "string.");
  }
  std::unique_ptr<tensorflow::data::standalone::Dataset> dataset;
  tensorflow::data::standalone::Dataset::Params params;
  tensorflow::Status status = tensorflow::data::standalone::Dataset::FromGraph(
      params, dataset_graph, &dataset);
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Dataset FromGraph creation failed while converting "
                     "TFF Value to tf data Dataset. Error: ",
                     status.error_message()));
  }

  return std::move(dataset);
}

}  // namespace tensorflow_federated
