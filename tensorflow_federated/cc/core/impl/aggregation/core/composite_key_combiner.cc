/*
 * Copyright 2023 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace tensorflow_federated {
namespace aggregation {

namespace {

// Advances the pointer by sizeof(T) bytes.
template <typename T>
void AdvancePtr(const void*& ptr) {
  ptr = static_cast<const void*>(static_cast<const uint8_t*>(ptr) + sizeof(T));
}

TensorShape GetTensorShapeForSize(size_t size) {
  TFF_CHECK(size <= LONG_MAX)
      << "TensorShape: Dimension size too large to be represented as a "
         "signed long.";
  return TensorShape({static_cast<int64_t>(size)});
}

// Given a vector of void* pointers, where the data pointed to can be safely
// interpreted as type T, returns a Tensor of underlying data type
// corresponding to T and the same length as the input vector.
// Each void* pointer in the input vector is incremented by size(T) bytes.
template <typename T>
absl::StatusOr<Tensor> GetTensorForType(
    std::vector<const void*>& key_iters,
    const std::shared_ptr<InternPool>& intern_pool) {
  auto output_tensor_data = std::make_unique<MutableVectorData<T>>();
  output_tensor_data->reserve(key_iters.size());
  for (const void*& key_it : key_iters) {
    T value;
    std::memcpy(&value, key_it, sizeof(T));
    output_tensor_data->push_back(value);
    AdvancePtr<T>(key_it);
  }
  return Tensor::Create(internal::TypeTraits<T>::kDataType,
                        GetTensorShapeForSize(key_iters.size()),
                        std::move(output_tensor_data));
}

// Implementation of TensorData for DT_STRING data type that shares
// ownership of the intern pool with the CompositeKeyCombiner.
// This helps to avoid making multiple copies of the strings that are originally
// interned in the CompositeKeyCombiner.
class InternedStringData : public TensorData {
 public:
  explicit InternedStringData(std::vector<string_view>&& strings_views,
                              const std::shared_ptr<InternPool>& intern_pool)
      : string_views_(std::move(strings_views)), intern_pool_(intern_pool) {}
  ~InternedStringData() override = default;

  // Implementation of TensorData methods.
  size_t byte_size() const override {
    return string_views_.size() * sizeof(string_view);
  }
  const void* data() const override { return string_views_.data(); }

 private:
  std::vector<string_view> string_views_;
  // The intern pool contains the strings that the above string_views point to.
  std::shared_ptr<InternPool> intern_pool_;
};

// Specialization of GetTensorForType for DT_STRING data type.
// Given a vector of void* pointers, where the data pointed to can be safely
// interpreted as a pointer to a string, returns a tensor of type DT_STRING
// and the same length as the input vector containing these strings.
// Each void* pointer in the input vector is incremented by size(intptr_tß)
// bytes. The returned tensor will own all strings it refers to and is thus safe
// to use after this class is destroyed.
template <>
absl::StatusOr<Tensor> GetTensorForType<string_view>(
    std::vector<const void*>& key_iters,
    const std::shared_ptr<InternPool>& intern_pool) {
  std::vector<string_view> string_views_for_output;
  string_views_for_output.reserve(key_iters.size());
  for (const void*& key_it : key_iters) {
    intptr_t ptr_int;
    std::memcpy(&ptr_int, key_it, sizeof(intptr_t));
    // The integer stored to represent a string is the address of the string
    // stored in the intern_pool_. Thus this integer can be safely cast to a
    // pointer and dereferenced to obtain the string.
    const std::string* ptr = reinterpret_cast<const std::string*>(ptr_int);
    string_views_for_output.push_back(*ptr);
    AdvancePtr<intptr_t>(key_it);
  }
  return Tensor::Create(DT_STRING, GetTensorShapeForSize(key_iters.size()),
                        std::make_unique<InternedStringData>(
                            std::move(string_views_for_output), intern_pool));
}

// Size needed to store a key in the composite key.
template <typename T>
size_t GetKeySize() {
  return sizeof(T);
}

// Specialization of GetKeySize for DT_STRING data type.
// The size of a key of type DT_STRING is the size of a pointer to the interned
// string.
template <>
size_t GetKeySize<string_view>() {
  return sizeof(intptr_t);
}

}  // namespace

size_t CompositeKeyCombiner::CalculateCompositeKeySize(
    absl::Span<const DataType> dtypes) {
  size_t composite_key_size = 0;
  for (DataType dtype : dtypes) {
    DTYPE_CASES(dtype, T, composite_key_size += GetKeySize<T>());
  }
  return composite_key_size;
}

CompositeKeyCombiner::CompositeKeyCombiner(std::vector<DataType> dtypes)
    : dtypes_(std::move(dtypes)),
      storage_(std::make_shared<CompositeKeyStorage>(
          CalculateCompositeKeySize(dtypes_))),
      composite_keys_(storage_->key_size()) {}

CompositeKeyCombiner::CompositeKeyCombiner(
    std::vector<DataType> dtypes, std::shared_ptr<CompositeKeyStorage> storage)
    : dtypes_(std::move(dtypes)),
      storage_(std::move(storage)),
      composite_keys_(storage_->key_size()) {
  TFF_CHECK(storage_ != nullptr);
  TFF_CHECK(storage_->key_size() == CalculateCompositeKeySize(dtypes_));
}

// Returns a single tensor containing the ordinals of the composite keys
// formed from the InputTensorList.
absl::StatusOr<Tensor> CompositeKeyCombiner::Accumulate(
    const InputTensorList& tensors) {
  TFF_ASSIGN_OR_RETURN(TensorShape shape, CheckValidAndGetShape(tensors));
  TFF_ASSIGN_OR_RETURN(size_t num_elements, shape.NumElements());

  return Tensor::Create(internal::TypeTraits<int64_t>::kDataType, shape,
                        CreateOrdinals(tensors, num_elements, composite_keys_));
}

// Creates ordinals for composite keys spread across input tensors: in a nested
// for loop, transfer the bytes into a CompositeKey, then insert each.
std::unique_ptr<MutableVectorData<int64_t>>
CompositeKeyCombiner::CreateOrdinals(const InputTensorList& tensors,
                                     size_t num_elements,
                                     CompositeKeyMap& composite_key_map) {
  // Initialize the ordinals vector
  auto ordinals = std::make_unique<MutableVectorData<int64_t>>(num_elements);

  // To set up the creation of composite keys, make a vector of pointers to the
  // data held in the tensors.
  std::vector<const void*> iterators;
  iterators.reserve(tensors.size());
  for (const Tensor* t : tensors) {
    iterators.push_back(t->data().data());
  }

  for (int64_t& ordinal : *ordinals) {
    CompositeKey composite_key = NewCompositeKey();
    // Construct a composite key by iterating through tensors and copying the
    // representation of data elements.
    void* key_ptr = composite_key.data();
    auto data_type_iter = dtypes_.begin();
    for (auto& itr : iterators) {
      // Copy the 64-bit representation of the element into the position in the
      // composite key data corresponding to this tensor.
      DTYPE_CASES(*(data_type_iter++), T, CopyToDest<T>(itr, key_ptr));
    }

    // Get the ordinal associated with the composite key
    // (or make new mapping if none exists) and insert the ordinal representing
    // the composite key into the correct position in the output tensor.
    auto [assigned_ordinal, inserted] =
        composite_key_map.InsertAndGetOrdinal(composite_key);
    ordinal = assigned_ordinal;
    if (inserted) {
      AdvanceCompositeKey(composite_key.data());
    }
  }
  return ordinals;
}

OutputTensorList CompositeKeyCombiner::GetOutputKeys() const {
  OutputTensorList output_keys;
  // Reserve space for a tensor for each data type. Even if no data has been
  // accumulated, there will always be one tensor output for each data type that
  // this CompositeKeyCombiner was configured to accept.
  output_keys.reserve(dtypes_.size());
  // key_iters vector is initialized to point to the first byte of each
  // composite key.
  std::vector<const void*> key_iters(composite_keys_.size());
  for (const auto& [key, ordinal] : composite_keys_) {
    TFF_CHECK(ordinal < key_iters.size());
    TFF_CHECK(key_iters[ordinal] == nullptr);
    key_iters[ordinal] = key.data();
  }

  for (DataType dtype : dtypes_) {
    absl::StatusOr<Tensor> t;
    DTYPE_CASES(dtype, T,
                t = GetTensorForType<T>(key_iters, storage_->intern_pool()));
    TFF_CHECK(t.status().ok()) << t.status().message();
    output_keys.push_back(std::move(t.value()));
  }
  return output_keys;
}

std::optional<int64_t> CompositeKeyCombiner::FindOrdinal(
    CompositeKey key) const {
  auto it = composite_keys_.find(key);
  return it == composite_keys_.end() ? std::nullopt
                                     : std::optional<int64_t>(it->second);
}

absl::StatusOr<TensorShape> CompositeKeyCombiner::CheckValidAndGetShape(
    const InputTensorList& tensors) const {
  if (tensors.size() == 0) {
    return absl::InvalidArgumentError(
        "InputTensorList must contain at least one tensor.");
  } else if (tensors.size() != dtypes_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "InputTensorList size ", tensors.size(),
        "is not the same as the length of expected dtypes ", dtypes_.size()));
  }
  // All the tensors in the input list should have the same shape and have
  // a dense encoding.
  const TensorShape* shape = nullptr;
  for (int i = 0; i < tensors.size(); ++i) {
    const Tensor* t = tensors[i];
    if (shape == nullptr) {
      shape = &t->shape();
    } else {
      if (*shape != t->shape()) {
        return absl::InvalidArgumentError(
            "All tensors in the InputTensorList must have the expected "
            "shape.");
      }
    }
    if (!t->is_dense())
      return absl::InvalidArgumentError(
          "All tensors in the InputTensorList must be dense.");
    // Ensure the data types of the input tensors match those provided to the
    // constructor of this CompositeKeyCombiner.
    DataType expected_dtype = dtypes_[i];
    if (expected_dtype != t->dtype()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Tensor at position ", i, " did not have expected dtype ",
          expected_dtype, " and instead had dtype ", t->dtype()));
    }
  }
  TFF_CHECK(shape != nullptr)
      << "All tensors in the InputTensorList must have shape.";
  return *shape;
}

}  // namespace aggregation
}  // namespace tensorflow_federated
