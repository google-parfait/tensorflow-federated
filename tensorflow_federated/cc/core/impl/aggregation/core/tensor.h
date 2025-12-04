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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_H_

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace tensorflow_federated {
namespace aggregation {

// Tensor class is a container that packages the tensor data with the tensor
// metadata such as the value type and the shape.
//
// For the most part, the aggregation code won't be consuming tensors directly.
// Instead the aggregation code will be working with AggVector instances that
// represent the tensor data in a flattened way.
class Tensor final {
 public:
  // Tensor class isn't copyable.
  Tensor(const Tensor&) = delete;

  // Move constructor.
  Tensor(Tensor&& other)
      : dtype_(other.dtype_),
        shape_(std::move(other.shape_)),
        data_(std::move(other.data_)),
        name_(std::move(other.name_)) {
    other.dtype_ = DT_INVALID;
  }

  // Move assignment.
  Tensor& operator=(Tensor&& other) {
    dtype_ = other.dtype_;
    shape_ = std::move(other.shape_);
    data_ = std::move(other.data_);
    name_ = std::move(other.name_);
    other.dtype_ = DT_INVALID;
    return *this;
  }

  // Define a default constructor to allow for initalization of array
  // to enable creation of a vector of Tensors.
  // A tensor created with the default constructor is not valid and thus should
  // not actually be used.
  Tensor() : dtype_(DT_INVALID), shape_{}, data_(nullptr), name_("") {}

  // Validates parameters and creates a Tensor instance.
  static StatusOr<Tensor> Create(DataType dtype, TensorShape shape,
                                 std::unique_ptr<TensorData> data,
                                 std::string name = "");

  // Creates a Tensor instance from a TensorProto.
  static StatusOr<Tensor> FromProto(const TensorProto& tensor_proto);

  // Creates a Tensor instance from a TensorProto, consuming the proto.
  static StatusOr<Tensor> FromProto(TensorProto&& tensor_proto);

  // Converts Tensor to TensorProto
  TensorProto ToProto() const;

  // Validates the tensor.
  Status CheckValid() const;

  // Gets the tensor value type.
  DataType dtype() const { return dtype_; }

  // Gets the tensor shape.
  const TensorShape& shape() const { return shape_; }

  // Gets the number of elements in the tensor.
  size_t num_elements() const { return shape_.NumElements().value(); }

  // Readonly access to the tensor data.
  const TensorData& data() const { return *data_; }

  // Readonly access to the tensor name.
  const std::string& name() const { return name_; }

  // Returns true is the current tensor data is dense.
  // TODO: b/266974165 - Implement sparse tensors.
  bool is_dense() const { return true; }

  // Returns true if the tensor is a scalar.
  bool is_scalar() const { return num_elements() == 1; }

  // Provides access to the tensor data via a strongly typed AggVector.
  template <typename T>
  AggVector<T> AsAggVector() const {
    TFF_CHECK(internal::TypeTraits<T>::kDataType == dtype_)
        << "Incompatible tensor dtype()";
    return AggVector<T>(data_.get());
  }

  // Returns the elements of the tensor as a vector of strings. This can be
  // called on tensors of any type. The elements of the tensor are formatted as
  // strings using absl::StrCat.
  std::vector<std::string> ToStringVector() const {
    DTYPE_CASES(dtype_, T, return TensorValuesToStringVector<T>());
  }

  // Provides access to the (numerical) tensor data as an integral scalar.
  // Values are automatically casted and rounded.
  template <typename T, typename std::enable_if<
                            std::is_integral<T>::value>::type* = nullptr>
  T CastToScalar() const {
    TFF_CHECK(is_scalar())
        << "CastToScalar should only be used on scalar tensors";
    T scalar_val;
    NUMERICAL_ONLY_DTYPE_CASES(
        dtype_, K, scalar_val = static_cast<T>(std::round(*GetData<K>())));
    return scalar_val;
  }

  // Provides access to the (numerical) tensor data as a non-integral scalar.
  // Values are automatically casted to the requested type.
  template <typename T, typename std::enable_if<
                            std::is_floating_point<T>::value>::type* = nullptr>
  T CastToScalar() const {
    TFF_CHECK(is_scalar())
        << "CastToScalar should only be used on scalar tensors";
    T scalar_val;
    NUMERICAL_ONLY_DTYPE_CASES(dtype_, K,
                               scalar_val = static_cast<T>(*GetData<K>()));
    return scalar_val;
  }

  // Provides access to the (string) tensor data as a scalar.
  template <typename T, typename std::enable_if<std::is_same<
                            string_view, T>::value>::type* = nullptr>
  T CastToScalar() const {
    TFF_CHECK(is_scalar())
        << "CastToScalar should only be used on scalar tensors";
    return *GetData<T>();
  }

  // Provides access to the tensor data as a scalar.
  template <typename T>
  T AsScalar() const {
    TFF_CHECK(is_scalar()) << "AsScalar should only be used on scalar tensors";
    return *GetData<T>();
  }

  // Provides access to the tensor data as a span.
  template <typename T>
  absl::Span<const T> AsSpan() const {
    TFF_CHECK(internal::TypeTraits<T>::kDataType == dtype_)
        << "Incompatible tensor dtype()";
    return absl::Span<const T>(GetData<T>(), num_elements());
  }

  // Updates the tensor name.
  Status set_name(absl::string_view name);

  // TODO: b/222605809 - Add serialization functions.

 private:
  Tensor(DataType dtype, TensorShape shape, size_t num_elements,
         std::unique_ptr<TensorData> data, std::string name = "")
      : dtype_(dtype),
        shape_(std::move(shape)),
        data_(std::move(data)),
        name_(std::move(name)) {}

  // Returns a pointer to the tensor data.
  template <typename T>
  const T* GetData() const {
    TFF_CHECK(internal::TypeTraits<T>::kDataType == dtype_)
        << "Incompatible tensor dtype()";
    return reinterpret_cast<const T*>(data_->data());
  }

  template <typename T>
  std::vector<std::string> TensorValuesToStringVector() const {
    std::vector<std::string> vec(num_elements());
    if (num_elements() > 0) {
      AggVector<T> agg_vector = AsAggVector<T>();
      for (auto [i, v] : agg_vector) {
        vec[i] = absl::StrCat(v);
      }
    }
    return vec;
  }

  // Tensor data type.
  DataType dtype_;
  // Tensor shape.
  TensorShape shape_;
  // The underlying tensor data.
  std::unique_ptr<TensorData> data_;
  // Name field to identify what the tensor represents.
  std::string name_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_H_
