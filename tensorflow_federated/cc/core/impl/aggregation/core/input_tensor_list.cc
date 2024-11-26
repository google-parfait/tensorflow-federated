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

#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"

#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

InputTensorList::InputTensorList(std::initializer_list<const Tensor*> list)
    : InputTensorList(list.size()) {
  size_t i = 0;
  for (const Tensor* t : list) {
    data_ptr_[i++] = t;
  }
}

InputTensorList::InputTensorList(const std::vector<const Tensor*>& list)
    : InputTensorList(list.size()) {
  size_t i = 0;
  for (const Tensor* t : list) {
    data_ptr_[i++] = t;
  }
}

InputTensorList::InputTensorList(size_t size)
    : size_(size), is_allocated_(size > kInlinedSize) {
  if (is_allocated_) {
    // Since the `allocated` union member is a class with user-defined
    // constructors and destructors, to switch the active member, explicit
    // placement new is needed. See
    // https://en.cppreference.com/w/cpp/language/union.
    new (&data_storage_.allocated) std::vector<const Tensor*>(size);
    data_ptr_ = data_storage_.allocated.data();
  } else {
    // Use the new syntax to initialize elements to nullptr.
    new (&data_storage_.inlined) int[size]();
    data_ptr_ = data_storage_.inlined;
  }
}

InputTensorList::InputTensorList(InputTensorList&& other)
    : size_(other.size_), is_allocated_(other.is_allocated_) {
  MoveData(std::move(other));
}

InputTensorList& InputTensorList::operator=(InputTensorList&& other) {
  // Destroy any existing allocated storage.
  if (is_allocated_) {
    data_storage_.allocated.~vector();
  }
  size_ = other.size_;
  is_allocated_ = other.is_allocated_;
  MoveData(std::move(other));
  return *this;
}

void InputTensorList::MoveData(InputTensorList&& other) {
  if (is_allocated_) {
    new (&data_storage_.allocated) std::vector<const Tensor*>;
    data_storage_.allocated = std::move(other.data_storage_.allocated);
    data_ptr_ = data_storage_.allocated.data();
    other.data_storage_.allocated.~vector();
  } else {
    // If the storage is inlined copy the data; this is cheap since
    // size_ < kInlinedSize.
    for (size_t i = 0; i < size_; ++i) {
      data_storage_.inlined[i] = other.data_storage_.inlined[i];
    }
    data_ptr_ = data_storage_.inlined;
  }
  new (&other.data_storage_.inlined) int[0]();
  other.size_ = 0;
  other.is_allocated_ = false;
}

InputTensorList::~InputTensorList() {
  // Since the `allocated` union member is a class with user-defined
  // constructors and destructors, explicit destruction is needed. See
  // https://en.cppreference.com/w/cpp/language/union.
  if (is_allocated_) {
    data_storage_.allocated.~vector();
  }
}

}  // namespace aggregation
}  // namespace tensorflow_federated
