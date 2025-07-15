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
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/vector_string_data.h"

namespace tensorflow_federated {
namespace aggregation {

namespace {

// Advances the pointer by sizeof(T) bytes.
template <typename T>
void AdvancePtr(void*& ptr) {
  ptr = static_cast<void*>(static_cast<uint8_t*>(ptr) + sizeof(T));
}

template <typename T>
void AdvancePtr(const void*& ptr) {
  ptr = static_cast<const void*>(static_cast<const uint8_t*>(ptr) + sizeof(T));
}

// Copies the bytes pointed to by source_ptr to the destination pointed to by
// dest_ptr and advances source_ptr and dest_ptr to the next T.
//
// The number of bytes copied will be the size of the type T.
//
// It is the responsibility of the caller to ensure that source_ptr is only used
// in subsequent code if it still points to a valid T after being incremented.
template <typename T>
inline void CopyToDest(const void*& source_ptr, void*& dest_ptr,
                       absl::node_hash_set<std::string>& intern_pool) {
  // Copy the bytes pointed to by source_ptr to the destination pointed to by
  // dest_ptr.
  std::memcpy(dest_ptr, source_ptr, sizeof(T));
  // Advance source_ptr and dest_ptr to the next T.
  AdvancePtr<T>(source_ptr);
  AdvancePtr<T>(dest_ptr);
}

// Specialization of CopyToDest for DT_STRING data type that interns the
// string_view pointed to by value_ptr. The address of the string in the
// intern pool is then converted to a 64 bit integer and copied to the
// destination pointed to by dest_ptr. Finally source_ptr is incremented to the
// next string_view.
//
// It is the responsibility of the caller to ensure that source_ptr is only used
// in subsequent code if it still points to a valid string_view after being
// incremented.
template <>
inline void CopyToDest<string_view>(
    const void*& source_ptr, void*& dest_ptr,
    absl::node_hash_set<std::string>& intern_pool) {
  auto string_view_ptr = static_cast<const string_view*>(source_ptr);
  // Insert the string into the intern pool if it does not already exist. This
  // makes a copy of the string so that the intern pool owns the storage.
  auto it = intern_pool.emplace(*string_view_ptr).first;
  // The iterator of a node_hash_set may be invalidated by inserting more
  // elements, but the pointer to the underlying element is guaranteed to be
  // stable.
  // Thus, get the address of the string after dereferencing the iterator.
  const std::string* interned_string_ptr = &*it;
  // The stable address of the string can be interpreted as a 64-bit integer.
  intptr_t ptr_int = reinterpret_cast<intptr_t>(interned_string_ptr);
  // Set the destination storage to the integer representation of the string
  // address.
  std::memcpy(dest_ptr, &ptr_int, sizeof(intptr_t));
  // Set the source_ptr to point to the next string_view assuming that it points
  // to an array of string_view.
  source_ptr = static_cast<const void*>(++string_view_ptr);
  // Advance the dest_ptr.
  AdvancePtr<intptr_t>(dest_ptr);
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
StatusOr<Tensor> GetTensorForType(std::vector<const void*>& key_iters) {
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

// Specialization of GetTensorForType for DT_STRING data type.
// Given a vector of void* pointers, where the data pointed to can be safely
// interpreted as a pointer to a string, returns a tensor of type DT_STRING
// and the same length as the input vector containing these strings.
// Each void* pointer in the input vector is incremented by size(intptr_tß)
// bytes. The returned tensor will own all strings it refers to and is thus safe
// to use after this class is destroyed.
template <>
StatusOr<Tensor> GetTensorForType<string_view>(
    std::vector<const void*>& key_iters) {
  std::vector<std::string> strings_for_output;
  for (const void*& key_it : key_iters) {
    intptr_t ptr_int;
    std::memcpy(&ptr_int, key_it, sizeof(intptr_t));
    // The integer stored to represent a string is the address of the string
    // stored in the intern_pool_. Thus this integer can be safely cast to a
    // pointer and dereferenced to obtain the string.
    const std::string* ptr = reinterpret_cast<const std::string*>(ptr_int);
    strings_for_output.push_back(*ptr);
    AdvancePtr<intptr_t>(key_it);
  }
  return Tensor::Create(
      DT_STRING, GetTensorShapeForSize(key_iters.size()),
      std::make_unique<VectorStringData>(std::move(strings_for_output)));
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

CompositeKeyCombiner::CompositeKeyCombiner(std::vector<DataType> dtypes)
    : dtypes_(dtypes), composite_key_size_(0) {
  // Calculate the size of a composite key for the given data types.
  for (DataType dtype : dtypes) {
    DTYPE_CASES(dtype, T, composite_key_size_ += GetKeySize<T>());
  }
}

// Returns a single tensor containing the ordinals of the composite keys
// formed from the InputTensorList.
StatusOr<Tensor> CompositeKeyCombiner::Accumulate(
    const InputTensorList& tensors) {
  TFF_ASSIGN_OR_RETURN(TensorShape shape, CheckValidAndGetShape(tensors));
  TFF_ASSIGN_OR_RETURN(size_t num_elements, shape.NumElements());

  return Tensor::Create(internal::TypeTraits<int64_t>::kDataType, shape,
                        CreateOrdinals(tensors, num_elements, composite_keys_,
                                       composite_key_next_));
}

// Creates ordinals for composite keys spread across input tensors: in a nested
// for loop, transfer the bytes into a CompositeKey, then
// call SaveCompositeKeyAndGetOrdinal on each.
std::unique_ptr<MutableVectorData<int64_t>>
CompositeKeyCombiner::CreateOrdinals(
    const InputTensorList& tensors, size_t num_elements,
    absl::flat_hash_map<CompositeKey, int64_t>& composite_key_map,
    int64_t& current_ordinal) {
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
      DTYPE_CASES(*(data_type_iter++), T,
                  CopyToDest<T>(itr, key_ptr, intern_pool_));
    }

    // Get the ordinal associated with the composite key
    // (or make new mapping if none exists) and insert the ordinal representing
    // the composite key into the correct position in the output tensor.
    ordinal = SaveCompositeKeyAndGetOrdinal(std::move(composite_key),
                                            composite_key_map, current_ordinal);
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
    StatusOr<Tensor> t;
    DTYPE_CASES(dtype, T, t = GetTensorForType<T>(key_iters));
    TFF_CHECK(t.status().ok()) << t.status().message();
    output_keys.push_back(std::move(t.value()));
  }
  return output_keys;
}

StatusOr<TensorShape> CompositeKeyCombiner::CheckValidAndGetShape(
    const InputTensorList& tensors) {
  if (tensors.size() == 0) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "InputTensorList must contain at least one tensor.";
  } else if (tensors.size() != dtypes_.size()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "InputTensorList size " << tensors.size()
           << "is not the same as the length of expected dtypes "
           << dtypes_.size();
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
        return TFF_STATUS(INVALID_ARGUMENT)
               << "All tensors in the InputTensorList must have the expected "
                  "shape.";
      }
    }
    if (!t->is_dense())
      return TFF_STATUS(INVALID_ARGUMENT)
             << "All tensors in the InputTensorList must be dense.";
    // Ensure the data types of the input tensors match those provided to the
    // constructor of this CompositeKeyCombiner.
    DataType expected_dtype = dtypes_[i];
    if (expected_dtype != t->dtype()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "Tensor at position " << i << " did not have expected dtype "
             << expected_dtype << " and instead had dtype " << t->dtype();
    }
  }
  TFF_CHECK(shape != nullptr)
      << "All tensors in the InputTensorList must have shape.";
  return *shape;
}

}  // namespace aggregation
}  // namespace tensorflow_federated
