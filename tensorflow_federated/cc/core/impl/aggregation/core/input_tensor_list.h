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
#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_INPUT_TENSOR_LIST_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_INPUT_TENSOR_LIST_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

// Maximum size of InputTensorList for which inlined storage will be used.
// Any InputTensorList with more elements than kInlinedSize will use allocated
// storage.
// TODO: b/277984442 - Determine optimal size for this constant based on
// microbenchmarks.
constexpr int32_t kInlinedSize = 5;

// InputTensorList holds pointers to some number of unowned tensors to be used
// as input to a function.
//
// For efficiency, if there are fewer than kInlinedSize tensors, the memory to
// hold the pointers is inlined rather than allocated.
class InputTensorList final {
 public:
  typedef const Tensor* const* const_iterator;

  // Creates an InputTensorList with the provided elements.
  InputTensorList(std::initializer_list<const Tensor*> list);

  // Creates an InputTensorList with a single input tensor.
  InputTensorList(const Tensor& tensor) : InputTensorList({&tensor}) {}

  // Creates an InputTensorList of a specific size. All elements will initially
  // be set to nullptr.
  explicit InputTensorList(size_t size);

  // InputTensorList class isn't copyable.
  InputTensorList(const InputTensorList&) = delete;

  // Move constructor.
  InputTensorList(InputTensorList&& other);

  // Move assignment.
  InputTensorList& operator=(InputTensorList&& other);

  ~InputTensorList();

  inline const_iterator begin() const { return data_ptr_; }

  inline const_iterator end() const { return data_ptr_ + size_; }

  inline size_t size() const { return size_; }

  inline const Tensor* const& operator[](size_t i) const {
    return data_ptr_[i];
  }

  inline const Tensor*& operator[](size_t i) { return data_ptr_[i]; }

 private:
  union DataStorage {
    constexpr DataStorage() : inlined{} {};
    ~DataStorage() {}
    const Tensor* inlined[kInlinedSize];
    std::vector<const Tensor*> allocated;
  };

  void MoveData(InputTensorList&& other);

  size_t size_;
  bool is_allocated_;
  DataStorage data_storage_;
  const Tensor** data_ptr_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_INPUT_TENSOR_LIST_H_
