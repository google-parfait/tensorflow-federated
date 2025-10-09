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

#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "federated_language/proto/array.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "federated_language_executor/executor.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"

namespace tensorflow_federated {

absl::Status SerializeTensorValue(const tensorflow::Tensor tensor,
                                  v0::Value* value_pb) {
  // Repeated fields are used for strings and constants to maintain
  // compatibility with TensorFlow.
  federated_language::Array array_pb;
  if (!tensorflow::TensorShapeUtils::IsScalar(tensor.shape()) &&
      tensor.dtype() != tensorflow::DataType::DT_STRING) {
    array_pb = TFF_TRY(ArrayContentFromTensor(tensor));
  } else {
    array_pb = TFF_TRY(ArrayFromTensor(tensor));
  }
  value_pb->mutable_array()->Swap(&array_pb);
  return absl::OkStatus();
}

absl::StatusOr<tensorflow::Tensor> DeserializeTensorValue(
    const v0::Value& value_pb) {
  if (!value_pb.has_array()) {
    LOG(ERROR) << "Attempted to deserialize non-Array value to a Tensor";
    LOG(ERROR) << "Value proto: " << value_pb.ShortDebugString();
    return absl::InvalidArgumentError(
        "value_pb must have an `array` oneof field to be deserializable to a "
        "Tensor");
  }

  // Repeated fields are used for strings and constants to maintain
  // compatibility with TensorFlow.
  if (value_pb.array().has_content()) {
    return TensorFromArrayContent(value_pb.array());
  } else {
    return TensorFromArray(value_pb.array());
  }
}

}  // namespace tensorflow_federated
