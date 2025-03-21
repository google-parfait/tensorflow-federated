/* Copyright 2024, The TensorFlow Federated Authors.

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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_XLA_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_XLA_UTILS_H_

#include "absl/status/statusor.h"
#include "federated_language/proto/array.pb.h"
#include "federated_language/proto/computation.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape.h"

namespace tensorflow_federated {

// Creates a ::federated_language::DataType from a xla::PrimitiveType.
absl::StatusOr<federated_language::DataType> DataTypeFromPrimitiveType(
    xla::PrimitiveType primative_type);

// Creates a xla::PrimitiveType from a ::federated_language::DataType.
absl::StatusOr<xla::PrimitiveType> PrimitiveTypeFromDataType(
    federated_language::DataType data_type);

// Creates a federated_language::ArrayShape from a xla::Shape.
federated_language::ArrayShape ArrayShapeFromShape(const xla::Shape& shape);

// Creates a xla::Shape from a federated_language::TensorType.
absl::StatusOr<xla::Shape> ShapeFromTensorType(
    const federated_language::TensorType& tensor_type_pb);

// Creates a xla::Shape from a federated_language::ArrayShape.
absl::StatusOr<xla::Shape> ShapeFromArrayShape(
    federated_language::DataType data_type,
    const federated_language::ArrayShape& shape_pb);

// Creates a federated_language::Array from a xla::Literal.
absl::StatusOr<federated_language::Array> ArrayFromLiteral(
    const xla::Literal& literal);

// Creates a xla::Literal from a federated_language::Array.
absl::StatusOr<xla::Literal> LiteralFromArray(
    const federated_language::Array& array_pb);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_XLA_UTILS_H_
