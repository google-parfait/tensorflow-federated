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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TYPE_TEST_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TYPE_TEST_UTILS_H_

#include "tensorflow_federated/proto/v0/computation.pb.h"

namespace tensorflow_federated {
namespace testing {

// Construct a v0::Type of shape <T, <T, T>> for parameter T.
inline v0::Type NestedStructT(v0::TensorType::DataType dtype) {
  v0::Type float_tensor_type;
  float_tensor_type.mutable_tensor()->set_dtype(dtype);
  v0::Type nested_struct_type;
  for (int i = 0; i < 2; i++) {
    *nested_struct_type.mutable_struct_()->add_element()->mutable_value() =
        float_tensor_type;
  }
  v0::Type return_type;
  *return_type.mutable_struct_()->add_element()->mutable_value() =
      float_tensor_type;
  *return_type.mutable_struct_()->add_element()->mutable_value() =
      nested_struct_type;
  return return_type;
}

// Construct a tensor type with the provided datatype and shape specification.
inline v0::Type TensorT(v0::TensorType::DataType dtype,
                        absl::Span<const int64_t> shape = {}) {
  v0::Type tensor_type;
  tensor_type.mutable_tensor()->set_dtype(dtype);
  for (const int64_t& dim : shape) {
    tensor_type.mutable_tensor()->add_dims(dim);
  }
  return tensor_type;
}

// Construct an unnamed struct type with the provided elements
inline v0::Type StructT(absl::Span<const v0::Type> elements) {
  v0::Type struct_type;
  for (const v0::Type& el_type : elements) {
    *struct_type.mutable_struct_()->add_element()->mutable_value() = el_type;
  }
  return struct_type;
}

// Construct a functional v0::Type with no argument, and provided return type.
inline v0::Type NoArgFunctionT(v0::Type return_type) {
  v0::Type function_type;
  *function_type.mutable_function()->mutable_result() = return_type;
  return function_type;
}

// Construct a functional v0::Type with accepting and returning the same type.
inline v0::Type IdentityFunctionT(v0::Type arg_type) {
  v0::Type function_type;
  *function_type.mutable_function()->mutable_parameter() = arg_type;
  *function_type.mutable_function()->mutable_result() = arg_type;
  return function_type;
}

// Construct a functional v0::Type with provided argument and return types.
inline v0::Type FunctionT(v0::Type parameter_type, v0::Type return_type) {
  v0::Type function_type;
  *function_type.mutable_function()->mutable_parameter() = parameter_type;
  *function_type.mutable_function()->mutable_result() = return_type;
  return function_type;
}
// Construct a v0::Type of shape <T,...> for parameter T, with num_reps
// elements.
inline v0::Type FlatStructT(v0::TensorType::DataType dtype, int num_reps) {
  v0::Type float_tensor_type;
  float_tensor_type.mutable_tensor()->set_dtype(dtype);
  v0::Type struct_type;
  for (int i = 0; i < num_reps; i++) {
    *struct_type.mutable_struct_()->add_element()->mutable_value() =
        float_tensor_type;
  }
  return struct_type;
}
}  // namespace testing
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TYPE_TEST_UTILS_H_
