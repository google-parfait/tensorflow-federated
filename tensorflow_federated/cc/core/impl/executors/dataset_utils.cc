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

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "federated_language/proto/computation.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow_federated/cc/core/impl/executors/dataset_from_tensor_structures.h"
#include "tensorflow_federated/cc/core/impl/executors/session_provider.h"
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

absl::StatusOr<std::pair<tensorflow::DataTypeVector,
                         std::vector<tensorflow::PartialTensorShape>>>
ExtractOutputTypesAndShapesFromGraphDef(
    const tensorflow::Tensor& graph_def_tensor) {
  if (graph_def_tensor.dtype() != tensorflow::DT_STRING ||
      graph_def_tensor.dims() != 0) {
    return absl::InvalidArgumentError(
        "Expected a scalar string tensor for GraphDef");
  }
  tensorflow::GraphDef graph_def;
  if (!graph_def.ParseFromString(
          graph_def_tensor.scalar<tensorflow::tstring>()())) {
    return absl::InvalidArgumentError("Failed to parse GraphDef from tensor");
  }
  tensorflow::DataTypeVector output_types;
  std::vector<tensorflow::PartialTensorShape> output_shapes;
  bool found = false;
  for (const auto& node : graph_def.node()) {
    auto it_types = node.attr().find("output_types");
    if (it_types == node.attr().end()) {
      it_types = node.attr().find("Toutput_types");
    }
    auto it_shapes = node.attr().find("output_shapes");
    if (it_types != node.attr().end() && it_shapes != node.attr().end()) {
      output_types.clear();
      for (auto dtype : it_types->second.list().type()) {
        output_types.push_back(static_cast<tensorflow::DataType>(dtype));
      }
      output_shapes.clear();
      for (const auto& shape_proto : it_shapes->second.list().shape()) {
        tensorflow::PartialTensorShape shape;
        auto status = tensorflow::PartialTensorShape::BuildPartialTensorShape(
            shape_proto, &shape);
        if (!status.ok()) {
          return absl::InternalError(absl::StrCat(
              "Failed to parse shape from GraphDef: ", status.message()));
        }
        output_shapes.push_back(shape);
      }
      found = true;
      // Don't break: keep iterating to find the last matching node (the root
      // dataset node).
    }
  }
  if (!found) {
    return absl::InvalidArgumentError(
        "No dataset node with output_types/output_shapes found in GraphDef");
  }
  return std::make_pair(output_types, output_shapes);
}

absl::StatusOr<std::vector<std::vector<tensorflow::Tensor>>>
IterateDatasetFromGraphDef(
    const tensorflow::Tensor& graph_def_tensor,
    const tensorflow::DataTypeVector& output_types,
    const std::vector<tensorflow::PartialTensorShape>& output_shapes) {
  // Build a graph:
  //   Placeholder(DT_STRING) -> DatasetFromGraph -> {IteratorV2 + MakeIterator}
  //   -> IteratorGetNext
  namespace tf = ::tensorflow;
  tf::Scope scope = tf::Scope::NewRootScope();

  // Placeholder for the serialized GraphDef.
  tf::Node* placeholder;
  scope.UpdateStatus(tf::NodeBuilder("graph_def_input", "Placeholder")
                         .Attr("dtype", tf::DT_STRING)
                         .Device("/device:CPU:0")
                         .Finalize(scope.graph(), &placeholder));

  // DatasetFromGraph: converts the serialized GraphDef back into a dataset
  // variant.
  tf::Node* dataset;
  scope.UpdateStatus(tf::NodeBuilder("dataset", "DatasetFromGraph")
                         .Input(placeholder, 0)
                         .Device("/device:CPU:0")
                         .Finalize(scope.graph(), &dataset));

  // IteratorV2: creates a named iterator resource that persists across
  // session.Run calls.
  tf::Node* iterator;
  scope.UpdateStatus(tf::NodeBuilder("iterator", "IteratorV2")
                         .Attr("shared_name", "tff_iterate")
                         .Attr("container", "")
                         .Attr("output_types", output_types)
                         .Attr("output_shapes", output_shapes)
                         .Device("/device:CPU:0")
                         .Finalize(scope.graph(), &iterator));

  // MakeIterator: binds the dataset to the iterator.
  tf::Node* make_iterator;
  scope.UpdateStatus(tf::NodeBuilder("make_iterator", "MakeIterator")
                         .Input(dataset, 0)
                         .Input(iterator, 0)
                         .Device("/device:CPU:0")
                         .Finalize(scope.graph(), &make_iterator));

  // IteratorGetNext: retrieves the next element from the iterator.
  tf::Node* get_next;
  scope.UpdateStatus(tf::NodeBuilder("get_next", "IteratorGetNext")
                         .Input(iterator, 0)
                         .Attr("output_types", output_types)
                         .Attr("output_shapes", output_shapes)
                         .Device("/device:CPU:0")
                         .Finalize(scope.graph(), &get_next));

  tf::GraphDef graph_def;
  absl::Status scope_status = scope.ToGraphDef(&graph_def);
  if (!scope_status.ok()) {
    return absl::InternalError(absl::StrCat("Failed to create iterator graph: ",
                                            scope_status.message()));
  }

  // Create a session and run.
  SessionProvider session_provider(std::move(graph_def));
  auto session = TFF_TRY(session_provider.BorrowSession());

  // Feed the GraphDef tensor and initialize the iterator.
  std::vector<std::pair<std::string, tf::Tensor>> inputs = {
      {"graph_def_input:0", graph_def_tensor}};
  absl::Status status =
      session->Run(inputs, /*output_tensor_names=*/{},
                   /*target_tensor_names=*/{"make_iterator"}, nullptr);
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to make iterator: ", status.message()));
  }

  // Build the output tensor names for IteratorGetNext.
  std::vector<std::string> output_names;
  output_names.reserve(output_types.size());
  for (std::size_t i = 0; i < output_types.size(); i++) {
    output_names.push_back(absl::StrCat("get_next:", i));
  }

  // Iterate until OutOfRange.
  std::vector<std::vector<tf::Tensor>> result;
  while (true) {
    std::vector<tf::Tensor> outputs;
    status = session->Run({}, output_names, {}, &outputs);
    if (absl::IsOutOfRange(status)) {
      break;
    }
    if (!status.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to get next element: ", status.message()));
    }
    result.push_back(std::move(outputs));
  }

  return result;
}

}  // namespace tensorflow_federated
