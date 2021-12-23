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

#include "tensorflow_federated/cc/core/impl/executors/dataset_from_tensor_structures.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/dataset_metadata.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow_federated/cc/core/impl/executors/session_provider.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"

namespace tensorflow_federated {

namespace {

namespace tf = ::tensorflow;
using TensorStructuresSpan =
    ::absl::Span<const std::vector<::tensorflow::Tensor>>;

std::string InputTensorName(size_t i, size_t element_index) {
  return absl::StrCat("structure_", i, "_element_", element_index);
}

template <typename T>
std::string MismatchedElementsMessage(absl::string_view property,
                                      size_t element_index,
                                      const T& first_value,
                                      const T& second_value,
                                      size_t second_position) {
  return absl::StrCat(
      "Expected each structure to have elements whose ", property,
      "s match the corresponding elements in all other structures, but found ",
      property, " ", first_value, " in element ", element_index,
      " of structure 0, but ", property, " ", second_value, " in element ",
      element_index, " of structure ", second_position, ".");
}

struct DTypeAndShape {
  tf::DataType dtype;
  tf::TensorShape shape;
};

absl::StatusOr<DTypeAndShape> GetDtypeAndShapeForStructureElement(
    TensorStructuresSpan tensor_structures, size_t element_index) {
  const tf::Tensor& first = tensor_structures[0][element_index];
  tf::DataType dtype = first.dtype();
  tf::TensorShape shape = first.shape();
  for (size_t i = 0; i < tensor_structures.size(); i++) {
    const tf::Tensor& tensor = tensor_structures[i][element_index];
    if (tensor.dtype() != first.dtype()) {
      return absl::InvalidArgumentError(MismatchedElementsMessage(
          "dtype", element_index, first.dtype(), tensor.dtype(), i));
    }
    if (tensor.dims() != first.dims()) {
      return absl::InvalidArgumentError(MismatchedElementsMessage(
          "rank", element_index, first.dims(), tensor.dims(), i));
    }
    // Tensors of different runtime lengths will fail the call to `tf.stack`.
    // In the future, it'd be good to support this: the resulting dataset
    // can simply yielding values of different lengths using unknown
    // (runtime-determined) dimensions, but the current impl using
    // DatasetFromTensorSlices does not allow this since the input slices
    // would need to be ragged.
    for (int dim_i = 0; dim_i < first.dims(); dim_i++) {
      if (tensor.dim_size(dim_i) != shape.dim_size(dim_i)) {
        return absl::UnimplementedError(absl::StrCat(
            "Attempted to create dataset with tensor elements with different "
            "dimensions. This is not yet supported. Found tensor with dtype ",
            tensor.dtype(), "and rank ", tensor.dims(),
            ". In the first structure, this tensor has dimension ",
            shape.dim_size(dim_i), " at position ", dim_i,
            ", but in structure ", i, " this tensor has dimension ",
            tensor.dim_size(dim_i)));
      }
    }
  }
  // Includes a trivial copy of `dtype` and `shape`.
  return DTypeAndShape{dtype, shape};
}

struct GraphWithOutput {
  tf::GraphDef graph;
  std::string output_tensor_name;
};

// Creates a `tf::GraphDef` that transforms an input list of structures of
// tensors into a `tf.data.Dataset`.
//
// Returns the graph and the name of the string tensor containing the serialized
// dataset.
absl::StatusOr<GraphWithOutput> DatasetFromTensorStructuresGraph(
    TensorStructuresSpan tensor_structures) {
  const size_t num_structures = tensor_structures.size();
  if (num_structures == 0) {
    return absl::InvalidArgumentError(
        "Cannot create dataset from structure list of length zero.");
  }
  size_t elements_per_structure = tensor_structures[0].size();
  for (const auto& structure : tensor_structures) {
    if (structure.size() != elements_per_structure) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot create a dataset from tensor structures of different sizes ",
          elements_per_structure, " and ", structure.size(), "."));
    }
  }

  // The following code generates a graph as follows:
  //
  // # For each input tensor:
  // input_$i_$j = tf.compat.v1.placeholder(
  //   name='structure_$i_element_$j', dtype=..., shape=...)
  //
  // # For each element in the structure:
  // stack_$m = tf.stack([input_1_$m, $input_2_$m, ...])
  //
  // # Finally the conversion to serialized dataset:
  // dataset = tf.data.Dataset.from_tensor_slices(
  //   (stack_1, stack_2, ...), name='dataset')
  // result = tf.raw_ops.DatasetToGraphV2(
  //   input_dataset=tf.data.experimental.to_variant(dataset),
  //   name='serialized')
  tf::Scope scope = tf::Scope::NewRootScope();
  std::vector<tf::DataType> element_dtypes;
  element_dtypes.reserve(elements_per_structure);
  std::vector<tf::TensorShape> element_shapes;
  element_shapes.reserve(elements_per_structure);
  std::vector<tf::NodeBuilder::NodeOut> ds_from_slice_inputs;
  ds_from_slice_inputs.reserve(elements_per_structure);
  for (size_t element_index = 0; element_index < elements_per_structure;
       element_index++) {
    DTypeAndShape dtype_and_shape = TFF_TRY(
        GetDtypeAndShapeForStructureElement(tensor_structures, element_index));
    const tf::DataType& dtype = dtype_and_shape.dtype;
    const tf::TensorShape& shape = dtype_and_shape.shape;
    element_dtypes.push_back(dtype);
    element_shapes.push_back(shape);
    std::vector<tf::Input> stack_inputs;
    stack_inputs.reserve(num_structures);
    for (size_t i = 0; i < num_structures; i++) {
      tf::ops::Placeholder placeholder(scope, dtype,
                                       tf::ops::Placeholder::Shape(shape));
      placeholder.node()->set_name(InputTensorName(i, element_index));
      stack_inputs.push_back(placeholder);
    }
    tf::ops::Stack stack(scope, ::tensorflow::InputList(stack_inputs));
    stack.node()->set_name(absl::StrCat("stack_", element_index));
    ds_from_slice_inputs.push_back(tf::NodeBuilder::NodeOut(stack.node()));
  }
  tf::NodeBuilder ds_from_slice_builder("dataset", "TensorSliceDataset");
  tf::data::Metadata metadata;
  metadata.set_name("dataset");
  ds_from_slice_builder.Attr("Toutput_types", element_dtypes)
      .Attr("is_files", false)
      .Attr("metadata", metadata.SerializeAsString())
      .Attr("output_shapes", element_shapes)
      .Input(ds_from_slice_inputs);
  tf::Node* ds_from_slice;
  scope.UpdateStatus(
      ds_from_slice_builder.Finalize(scope.graph(), &ds_from_slice));
  static constexpr absl::string_view output_tensor_name = "serialized_dataset";
  tf::NodeBuilder ds_to_graph_builder(output_tensor_name, "DatasetToGraphV2");
  ds_to_graph_builder.Input(ds_from_slice, 0)
      .Attr("external_state_policy", 0)
      .Attr("strip_device_assignment", true)
      .Device("/device:CPU:0");
  scope.UpdateStatus(ds_to_graph_builder.Finalize(scope.graph(), nullptr));
  tf::GraphDef graph_def;
  tf::Status status = scope.ToGraphDef(&graph_def);
  if (!status.ok()) {
    return absl::InternalError(absl::StrCat("Failure to create dataset graph: ",
                                            status.error_message()));
  }
  return GraphWithOutput{std::move(graph_def), std::string(output_tensor_name)};
}

}  // namespace

absl::StatusOr<tf::Tensor> DatasetFromTensorStructures(
    TensorStructuresSpan tensor_structures) {
  GraphWithOutput graph_and_output_tensor_name =
      TFF_TRY(DatasetFromTensorStructuresGraph(tensor_structures));
  tf::GraphDef& graph_def = graph_and_output_tensor_name.graph;
  std::string& output_tensor_name =
      graph_and_output_tensor_name.output_tensor_name;
  std::vector<std::pair<std::string, tf::Tensor>> inputs;
  for (size_t i = 0; i < tensor_structures.size(); i++) {
    absl::Span<const tf::Tensor> structure = tensor_structures[i];
    for (size_t element_index = 0; element_index < structure.size();
         element_index++) {
      inputs.push_back(std::make_pair(InputTensorName(i, element_index),
                                      structure[element_index]));
    }
  }
  SessionProvider session_provider(std::move(graph_def), absl::nullopt);
  auto session = TFF_TRY(session_provider.BorrowSession());
  std::vector<tf::Tensor> outputs;
  tensorflow::Status status =
      session->Run(inputs, {std::move(output_tensor_name)},
                   /*target_tensor_names=*/{}, &outputs);
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to run DatasetFromTensorStructures computation: ",
                     status.error_message()));
  }
  if (outputs.size() != 1) {
    return absl::InternalError(
        absl::StrCat("Expected DatasetFromTensorStructures to return exactly "
                     "one tensor, but found ",
                     outputs.size(), " tensors."));
  }
  return std::move(outputs.back());
}

}  // namespace tensorflow_federated
