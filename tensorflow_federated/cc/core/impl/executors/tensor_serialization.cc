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

#include "absl/status/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.proto.h"
#include "tensorflow/core/framework/types.proto.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/proto/v0/executor.proto.h"

namespace tensorflow_federated {

namespace tf = ::tensorflow;

absl::Status SerializeTensorValue(const tf::Tensor tensor,
                                  v0::Value* value_pb) {
  tf::TensorProto tensor_proto;
  if (tensor.dtype() == tf::DT_STRING) {
    // For some reason, strings don't work with AsProtoTensorContent()?
    // >>> ValueError: cannot create an OBJECT array from memory buffer
    tensor.AsProtoField(&tensor_proto);
  } else {
    tensor.AsProtoTensorContent(&tensor_proto);
  }
  value_pb->mutable_tensor()->PackFrom(tensor_proto);
  return absl::OkStatus();
}

absl::StatusOr<tf::Tensor> DeserializeTensorValue(const v0::Value& value_pb) {
  if (!value_pb.has_tensor()) {
    LOG(ERROR) << "Attempted to deserialize non-Tensor value to a Tensor";
    LOG(ERROR) << "Value proto: " << value_pb.ShortDebugString();
    return absl::InvalidArgumentError(
        "value_pb must have a `tensor` oneof field to be deserializable to a "
        "Tensor");
  }
  tensorflow::TensorProto tensor_proto;
  value_pb.tensor().UnpackTo(&tensor_proto);
  tensorflow::Tensor tensor;
  if (!tensor.FromProto(tensor_proto)) {
    LOG(ERROR) << "Failed to deserialize tensor value contents to tensor";
    LOG(ERROR) << "Value proto: " << value_pb.ShortDebugString();
    return absl::InvalidArgumentError(
        "Seriailzed tensor Value proto could not be parsed into Tensor "
        "object.");
  }
  return tensor;
}

}  // namespace tensorflow_federated
