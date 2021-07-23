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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_VALUE_TEST_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_VALUE_TEST_UTILS_H_

#include <functional>
#include <string_view>
#include <utility>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/dataset_ops_internal.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.proto.h"
#include "tensorflow/core/framework/types.proto.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/proto/v0/computation.proto.h"

namespace tensorflow_federated {
namespace testing {

inline v0::Value IntrinsicV(std::string uri) {
  v0::Value value_proto;
  *value_proto.mutable_computation()->mutable_intrinsic()->mutable_uri() =
      std::move(uri);
  return value_proto;
}

// NOTE: Returns a value whose federated type `.member` field is unset.
inline v0::Value ServerV(v0::Value server_val) {
  v0::Value value_proto;
  v0::FederatedType* type_proto =
      value_proto.mutable_federated()->mutable_type();
  type_proto->set_all_equal(true);
  *type_proto->mutable_placement()->mutable_value()->mutable_uri() = "server";
  *value_proto.mutable_federated()->add_value() = server_val;
  return value_proto;
}

// NOTE: Returns a value whose federated type `.member` field is unset.
inline v0::Value ClientsV(const absl::Span<const v0::Value> client_values,
                          bool all_equal = false) {
  v0::Value value_proto;
  v0::FederatedType* type_proto =
      value_proto.mutable_federated()->mutable_type();
  type_proto->set_all_equal(all_equal);
  *type_proto->mutable_placement()->mutable_value()->mutable_uri() = "clients";
  auto values_pb = value_proto.mutable_federated()->mutable_value();
  values_pb->Reserve(client_values.size());
  for (const auto& client_value : client_values) {
    *values_pb->Add() = client_value;
  }
  return value_proto;
}

template <typename... Ts>
v0::Value TensorV(Ts... tensor_constructor_args) {
  tensorflow::Tensor tensor(tensor_constructor_args...);
  tensorflow::TensorProto tensor_proto;
  if (tensor.dtype() == tensorflow::DT_STRING) {
    tensor.AsProtoField(&tensor_proto);
  } else {
    tensor.AsProtoTensorContent(&tensor_proto);
  }
  v0::Value value_proto;
  value_proto.mutable_tensor()->PackFrom(tensor_proto);
  return value_proto;
}

inline v0::Value StructV(const absl::Span<const v0::Value> elements) {
  v0::Value value_proto;
  auto struct_proto = value_proto.mutable_struct_();
  for (const auto& element : elements) {
    *struct_proto->add_element()->mutable_value() = element;
  }
  return value_proto;
}

inline tensorflow::tstring CreateSerializedRangeDatasetGraphDef(
    int64_t stop, tensorflow::DataType dtype) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::internal::RangeDataset dataset(
      root, /*start=*/tensorflow::ops::Const(root, 0LL),
      /*stop=*/tensorflow::ops::Const(root, stop),
      /*step=*/tensorflow::ops::Const(root, 1LL),
      /*output_types=*/{dtype},
      /*output_shapes=*/{tensorflow::TensorShape({1})});
  tensorflow::ops::internal::DatasetToGraphV2 graph_def_tensor(root, dataset);
  tensorflow::ClientSession session(root);
  std::vector<tensorflow::Tensor> outputs;
  auto status = session.Run(/*fetch_outputs=*/{graph_def_tensor}, &outputs);
  tensorflow::tstring graph_def = outputs[0].flat<tensorflow::tstring>()(0);
  return graph_def;
}

inline v0::Value SequenceV(int64_t stop, tensorflow::DataType dtype) {
  tensorflow::tstring sequence_graph =
      CreateSerializedRangeDatasetGraphDef(stop, dtype);
  v0::Value value_proto;
  v0::Value::Sequence* sequence_pb = value_proto.mutable_sequence();
  *sequence_pb->mutable_serialized_graph_def() =
      std::string(sequence_graph.data(), sequence_graph.size());
  return value_proto;
}

namespace intrinsic {

#define INTRINSIC_FUNC(name, uri) \
  inline v0::Value name() { return IntrinsicV(#uri); }

INTRINSIC_FUNC(FederatedAggregateV, federated_aggregate);
INTRINSIC_FUNC(FederatedBroadcastV, federated_broadcast);
INTRINSIC_FUNC(FederatedMapV, federated_map);
INTRINSIC_FUNC(FederatedEvalAtClientsV, federated_eval_at_clients);
INTRINSIC_FUNC(FederatedEvalAtServerV, federated_eval_at_server);
INTRINSIC_FUNC(FederatedValueAtClientsV, federated_value_at_clients);
INTRINSIC_FUNC(FederatedValueAtServerV, federated_value_at_server);
INTRINSIC_FUNC(FederatedZipAtClientsV, federated_zip_at_clients);
INTRINSIC_FUNC(FederatedZipAtServerV, federated_zip_at_server);

#undef INTRINSIC_FUNC

}  // namespace intrinsic

inline v0::Value ComputationV(v0::Computation computation_pb) {
  v0::Value value_pb;
  *value_pb.mutable_computation() = computation_pb;
  return value_pb;
}

inline v0::Computation SelectionComputation(v0::Computation source_pb,
                                            int32_t index) {
  v0::Computation computation_pb;
  v0::Selection* selection_pb = computation_pb.mutable_selection();
  *selection_pb->mutable_source() = source_pb;
  selection_pb->set_index(index);
  return computation_pb;
}

inline v0::Computation StructComputation(
    std::vector<v0::Computation> elements) {
  v0::Computation computation_pb;
  v0::Struct* struct_pb = computation_pb.mutable_struct_();
  for (const auto& element : elements) {
    v0::Struct::Element* element_pb = struct_pb->add_element();
    *element_pb->mutable_value() = element;
  }
  return computation_pb;
}

inline v0::Computation LambdaComputation(
    std::optional<absl::string_view> parameter_name,
    v0::Computation result_computation_value) {
  v0::Computation computation_pb;
  v0::Lambda* lambda_pb = computation_pb.mutable_lambda();
  if (parameter_name != std::nullopt) {
    lambda_pb->set_parameter_name(parameter_name.value());
  }
  *lambda_pb->mutable_result() = result_computation_value;
  return computation_pb;
}

inline v0::Computation BlockComputation(
    std::vector<std::tuple<std::string, v0::Computation>> locals,
    v0::Computation result) {
  v0::Computation computation_pb;
  v0::Block* block_pb = computation_pb.mutable_block();
  for (const auto& local : locals) {
    v0::Block::Local* new_local_pb = block_pb->add_local();
    new_local_pb->set_name(std::get<0>(local));
    *new_local_pb->mutable_value() = std::get<1>(local);
  }
  *block_pb->mutable_result() = result;
  return computation_pb;
}

inline v0::Computation ReferenceComputation(absl::string_view reference_name) {
  v0::Computation computation_pb;
  computation_pb.mutable_reference()->set_name(reference_name);
  return computation_pb;
}

inline v0::Computation IntrinsicComputation(absl::string_view uri) {
  v0::Computation computation_pb;
  computation_pb.mutable_intrinsic()->set_uri(uri);
  return computation_pb;
}

inline v0::Computation DataComputation(absl::string_view uri) {
  v0::Computation computation_pb;
  computation_pb.mutable_data()->set_uri(uri);
  return computation_pb;
}

inline v0::Computation PlacementComputation(absl::string_view uri) {
  v0::Computation computation_pb;
  computation_pb.mutable_placement()->set_uri(uri);
  return computation_pb;
}

}  // namespace testing
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_VALUE_TEST_UTILS_H_
