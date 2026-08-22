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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TENSORFLOW_TEST_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TENSORFLOW_TEST_UTILS_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "googlemock/include/gmock/gmock.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "federated_language/proto/array.pb.h"
#include "federated_language/proto/computation.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/dataset_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"
#include "tensorflow_federated/cc/testing/protobuf_matchers.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {
namespace testing {

template <typename... Ts>
v0::Value TensorV(Ts... tensor_constructor_args) {
  tensorflow::Tensor tensor(tensor_constructor_args...);
  v0::Value value_proto;
  if (tensorflow::DataTypeCanUseMemcpy(tensor.dtype())) {
    absl::StatusOr<federated_language::Array> array_pb =
        ArrayContentFromTensor(tensor);
    CHECK_OK(array_pb.status());
    *value_proto.mutable_array() = *std::move(array_pb);
  } else {
    absl::StatusOr<federated_language::Array> array_pb =
        ArrayFromTensor(tensor);
    CHECK_OK(array_pb.status());
    *value_proto.mutable_array() = *std::move(array_pb);
  }
  return value_proto;
}

inline v0::Value TensorVFromIntList(absl::Span<const int32_t> elements) {
  size_t num_elements = elements.size();
  tensorflow::TensorShape shape({static_cast<int64_t>(num_elements)});
  tensorflow::Tensor tensor(tensorflow::DT_INT32, shape);
  auto flat = tensor.flat<int32_t>();
  for (size_t i = 0; i < num_elements; i++) {
    flat(i) = elements[i];
  }
  return TensorV(tensor);
}

inline v0::Value TensorSequenceV(int64_t start, int64_t stop, int64_t step) {
  v0::Value value_pb;
  v0::Value::Sequence* sequence_pb = value_pb.mutable_sequence();

  for (int64_t i = start; i < stop; i += step) {
    v0::Value::Sequence::Element* element_pb = sequence_pb->add_element();
    tensorflow::Tensor tensor(i);
    federated_language::Array* array_pb = element_pb->add_flat_value();
    if (tensorflow::DataTypeCanUseMemcpy(tensor.dtype())) {
      absl::StatusOr<federated_language::Array> content_array_pb =
          ArrayContentFromTensor(tensor);
      CHECK_OK(content_array_pb.status());
      *array_pb = *std::move(content_array_pb);
    } else {
      absl::StatusOr<federated_language::Array> list_array_pb =
          ArrayFromTensor(tensor);
      CHECK_OK(list_array_pb.status());
      *array_pb = *std::move(list_array_pb);
    }
  }

  federated_language::TensorType* tensor_type_pb =
      sequence_pb->mutable_element_type()->mutable_tensor();
  tensor_type_pb->set_dtype(federated_language::DataType::DT_INT64);
  tensor_type_pb->add_dims(1);

  return value_pb;
}

// Iterates a dataset from a GraphDef string tensor and returns all elements.
inline absl::StatusOr<std::vector<std::vector<tensorflow::Tensor>>>
SequenceValueToList(const tensorflow::Tensor& graph_def_tensor) {
  auto types_and_shapes =
      TFF_TRY(ExtractOutputTypesAndShapesFromGraphDef(graph_def_tensor));
  return IterateDatasetFromGraphDef(graph_def_tensor, types_and_shapes.first,
                                    types_and_shapes.second);
}

MATCHER(TensorsProtoEqual,
        absl::StrCat(negation ? "aren't" : "are",
                     " tensors equal under proto comparison")) {
  const tensorflow::Tensor& first = std::get<0>(arg);
  const tensorflow::Tensor& second = std::get<1>(arg);
  tensorflow::TensorProto first_proto;
  first.AsProtoTensorContent(&first_proto);
  tensorflow::TensorProto second_proto;
  second.AsProtoTensorContent(&second_proto);
  return testing::EqualsProto(second_proto)
      .impl()
      .MatchAndExplain(first_proto, result_listener);
}

}  // namespace testing
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TENSORFLOW_TEST_UTILS_H_
