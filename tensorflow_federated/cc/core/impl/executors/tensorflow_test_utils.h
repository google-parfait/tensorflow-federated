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
#include <memory>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "googlemock/include/gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "federated_language/proto/array.pb.h"
#include "federated_language/proto/computation.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "third_party/py/federated_language_executor/executor.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/dataset_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"
#include "tensorflow_federated/cc/testing/protobuf_matchers.h"

namespace tensorflow_federated {
namespace testing {

template <typename... Ts>
federated_language_executor::Value TensorV(Ts... tensor_constructor_args) {
  tensorflow::Tensor tensor(tensor_constructor_args...);
  federated_language_executor::Value value_proto;
  *value_proto.mutable_array() = ArrayFromTensor(tensor).value();
  return value_proto;
}

inline federated_language_executor::Value TensorVFromIntList(
    absl::Span<const int32_t> elements) {
  size_t num_elements = elements.size();
  tensorflow::TensorShape shape({static_cast<int64_t>(num_elements)});
  tensorflow::Tensor tensor(tensorflow::DT_INT32, shape);
  auto flat = tensor.flat<int32_t>();
  for (size_t i = 0; i < num_elements; i++) {
    flat(i) = elements[i];
  }
  return TensorV(tensor);
}

inline absl::StatusOr<std::vector<std::vector<tensorflow::Tensor>>>
SequenceValueToList(const tensorflow::Tensor& graph_def_tensor) {
  std::unique_ptr<tensorflow::data::standalone::Dataset> dataset =
      TFF_TRY(DatasetFromGraphDefTensor(graph_def_tensor));
  std::unique_ptr<tensorflow::data::standalone::Iterator> iterator;
  absl::Status status = dataset->MakeIterator(&iterator);
  if (!status.ok()) {
    return absl::InternalError(absl::StrCat(
        "Unable to make iterator from sequence dataset: ", status.message()));
  }
  std::vector<std::vector<tensorflow::Tensor>> outputs;
  while (true) {
    bool end_of_input;
    std::vector<tensorflow::Tensor> output;
    status = iterator->GetNext(&output, &end_of_input);
    if (!status.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to get the ", outputs.size(),
                       "th element of the sequence: ", status.message()));
    }
    if (end_of_input) {
      break;
    }
    outputs.push_back(std::move(output));
  }
  return outputs;
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
