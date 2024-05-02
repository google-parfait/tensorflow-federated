/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TENSORFLOW_CONVERTERS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TENSORFLOW_CONVERTERS_H_

#include <memory>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated::aggregation::tensorflow {

// Converts Tensorflow DataType to Aggregation DataType.
// Returns an error status if the input data type isn't supported by
// the Aggregation Core.
StatusOr<DataType> ToAggDataType(::tensorflow::DataType dtype);

// Converts a PartialTensorShape, which may have unknown dimensions, to
// Aggregation TensorShape.
// Note that the Tensorflow partial shape is expected to be valid.
TensorShape ToAggShape(const ::tensorflow::PartialTensorShape& shape);

// Converts Tensorflow TensorShape to Aggregation TensorShape.
// Note that the Tensorflow shape is expected to be valid (it seems impossible
// to create an invalid shape).
TensorShape ToAggShape(const ::tensorflow::TensorShape& shape);

// Converts Tensorflow TensorSpecProto to Aggregation TensorSpec.
// Returns an error status if supplied TensorSpecProto data type or shape isn't
// supported by the Aggregation Core.
StatusOr<TensorSpec> ToAggTensorSpec(const ::tensorflow::TensorSpecProto& spec);

// Converts Tensorflow TensorProto to Aggregation Tensor.
StatusOr<Tensor> ToAggTensor(const ::tensorflow::TensorProto& tensor_proto);

// Converts Tensorflow Tensor to Aggregation Tensor.
// Returns an error status if supplied Tensor data type or shape isn't
// supported by the Aggregation Core.
// Note that this function consumes the Tensorflow tensor.
StatusOr<Tensor> ToAggTensor(std::unique_ptr<::tensorflow::Tensor> tensor);

// Converts Aggregation DataType to TensorFlow DataType.
// Returns an error status if the input data type isn't supported by
// the Aggregation Core.
StatusOr<::tensorflow::DataType> ToTfDataType(DataType dtype);

// Converts an Aggregation TensorShape to a TensorFlow TensorShape.
StatusOr<::tensorflow::TensorShape> ToTfShape(const TensorShape& shape);

// Converts an Aggregation Tensor to a TensorFlow Tensor.
// Resulting numeric tensors may not be properly aligned, as alignment depends
// on the size of the buffer of the input tensor and TensorFlow has particular
// requirements for alignment. The only way to guarantee proper alignment would
// be to allocate a new buffer and copy data over, which we are not doing for
// efficiency reasons. Use methods like tensorflow::Tensor->unaligned_flat to
// retrieve data.
StatusOr<::tensorflow::Tensor> ToTfTensor(Tensor tensor);

}  // namespace tensorflow_federated::aggregation::tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TENSORFLOW_CONVERTERS_H_
