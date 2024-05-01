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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_SPEC_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_SPEC_H_

#include <string>
#include <utility>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace tensorflow_federated {
namespace aggregation {

// A tuple representing tensor name, data type, and shape.
class TensorSpec final {
 public:
  TensorSpec(std::string name, DataType dtype, TensorShape shape)
      : name_(std::move(name)), dtype_(dtype), shape_(std::move(shape)) {}

  // A tensor spec created with the default constructor is not valid and thus
  // should not actually be used.
  TensorSpec() : name_(""), dtype_{DT_INVALID}, shape_{} {}

  bool operator==(const TensorSpec& other) const {
    return name_ == other.name_ && dtype_ == other.dtype_ &&
           shape_ == other.shape_;
  }

  bool operator!=(const TensorSpec& other) const { return !(*this == other); }

  // Creates a TensorSpec instance from a TensorSpecProto.
  static StatusOr<TensorSpec> FromProto(
      const TensorSpecProto& tensor_spec_proto);

  // Converts TensorSpec to TensorSpecProto.
  TensorSpecProto ToProto() const;

  const std::string& name() const { return name_; }
  DataType dtype() const { return dtype_; }
  const TensorShape& shape() const { return shape_; }

 private:
  std::string name_;
  DataType dtype_;
  TensorShape shape_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_SPEC_H_
