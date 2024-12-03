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

#include <cstdint>

#include "absl/types/span.h"
#include "federated_language/proto/computation.pb.h"
#include "federated_language/proto/data_type.pb.h"

namespace tensorflow_federated {
namespace testing {

// Construct a federated_language::Type of shape <T, <T, T>> for parameter T.
inline federated_language::Type NestedStructT(
    federated_language::DataType dtype) {
  federated_language::Type float_tensor_type;
  float_tensor_type.mutable_tensor()->set_dtype(dtype);
  federated_language::Type nested_struct_type;
  for (int i = 0; i < 2; i++) {
    *nested_struct_type.mutable_struct_()->add_element()->mutable_value() =
        float_tensor_type;
  }
  federated_language::Type return_type;
  *return_type.mutable_struct_()->add_element()->mutable_value() =
      float_tensor_type;
  *return_type.mutable_struct_()->add_element()->mutable_value() =
      nested_struct_type;
  return return_type;
}

// Construct a tensor type with the provided datatype and shape specification.
inline federated_language::Type TensorT(federated_language::DataType dtype,
                                        absl::Span<const int64_t> shape = {}) {
  federated_language::Type tensor_type;
  tensor_type.mutable_tensor()->set_dtype(dtype);
  for (const int64_t& dim : shape) {
    tensor_type.mutable_tensor()->add_dims(dim);
  }
  return tensor_type;
}

// Construct an unnamed struct type with the provided elements
inline federated_language::Type StructT(
    absl::Span<const federated_language::Type> elements) {
  federated_language::Type struct_type;
  for (const federated_language::Type& el_type : elements) {
    *struct_type.mutable_struct_()->add_element()->mutable_value() = el_type;
  }
  return struct_type;
}

// Construct a functional federated_language::Type with no argument, and
// provided return type.
inline federated_language::Type NoArgFunctionT(
    federated_language::Type return_type) {
  federated_language::Type function_type;
  *function_type.mutable_function()->mutable_result() = return_type;
  return function_type;
}

// Construct a functional federated_language::Type with accepting and returning
// the same type.
inline federated_language::Type IdentityFunctionT(
    federated_language::Type arg_type) {
  federated_language::Type function_type;
  *function_type.mutable_function()->mutable_parameter() = arg_type;
  *function_type.mutable_function()->mutable_result() = arg_type;
  return function_type;
}

// Construct a functional federated_language::Type with provided argument and
// return types.
inline federated_language::Type FunctionT(
    federated_language::Type parameter_type,
    federated_language::Type return_type) {
  federated_language::Type function_type;
  *function_type.mutable_function()->mutable_parameter() = parameter_type;
  *function_type.mutable_function()->mutable_result() = return_type;
  return function_type;
}
// Construct a federated_language::Type of shape <T,...> for parameter T, with
// num_reps elements.
inline federated_language::Type FlatStructT(federated_language::DataType dtype,
                                            int num_reps) {
  federated_language::Type float_tensor_type;
  float_tensor_type.mutable_tensor()->set_dtype(dtype);
  federated_language::Type struct_type;
  for (int i = 0; i < num_reps; i++) {
    *struct_type.mutable_struct_()->add_element()->mutable_value() =
        float_tensor_type;
  }
  return struct_type;
}
}  // namespace testing
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TYPE_TEST_UTILS_H_
