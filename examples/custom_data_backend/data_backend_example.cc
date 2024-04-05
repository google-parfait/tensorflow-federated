/* Copyright 2022, The TensorFlow Federated Authors.

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

#include "tensorflow_federated/examples/custom_data_backend/data_backend_example.h"

#include <string>

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow_federated_examples {

namespace {

using ::tensorflow_federated::v0::Data;
using ::tensorflow_federated::v0::Type;
using ::tensorflow_federated::v0::Value;

// Constant URIs and values resolved by `DataBackendExample`.
// These definitions are repeated in data_backend_example_bindings_test.py.
static constexpr absl::string_view STRING_URI = "string";
static constexpr absl::string_view STRING_VALUE = "fooey";
static constexpr absl::string_view INT_STRUCT_URI = "int_struct";
static constexpr int32_t INT_VALUE = 55;

// Packs `tensor` into `value_out`.
void PackTensorInto(const tensorflow::Tensor& tensor, Value& value_out) {
  tensorflow::TensorProto tensor_proto;
  tensor.AsProtoField(&tensor_proto);
  value_out.mutable_tensor()->PackFrom(tensor_proto);
}

}  // namespace

absl::Status DataBackendExample::ResolveToValue(const Data& data_reference,
                                                const Type& data_type,
                                                Value& value_out) {
  if (!data_reference.has_uri()) {
    return absl::UnimplementedError(
        "`DataBackendExample` does not support resolving non-URI data "
        "blocks.");
  }
  const std::string& uri = data_reference.uri();
  if (uri == STRING_URI) {
    tensorflow::Tensor tensor((std::string(STRING_VALUE)));
    PackTensorInto(tensor, value_out);
    return absl::OkStatus();
  }
  if (uri == INT_STRUCT_URI) {
    tensorflow::Tensor int_tensor(INT_VALUE);
    tensorflow_federated::v0::Value_Struct* struct_out =
        value_out.mutable_struct_();
    PackTensorInto(int_tensor, *struct_out->add_element()->mutable_value());
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown URI: ", uri));
}

}  // namespace tensorflow_federated_examples
