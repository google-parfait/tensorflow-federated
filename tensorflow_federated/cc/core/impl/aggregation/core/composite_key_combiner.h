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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_COMPOSITE_KEY_COMBINER_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_COMPOSITE_KEY_COMBINER_H_

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace tensorflow_federated {
namespace aggregation {

// CompositeKey holds a representation of items of potentially different data
// types, which when combined form a key that should be used for grouping.
// All composite keys have the same size in bytes, which depends on the number
// of keys and their data types.
//
// Composite keys are stored as packed bytes - 4 or 8 bytes per key depending
// on the data type of each key. For string keys, the strings are interned into
// a shared pool, then a pointer to the interned string is stored in the
// composite key to uniquely represent the string key.
//
// CompositeKey class can be thought of as string_view into the composite key
// data, except that the size of the key isn't stored because it is the same
// across all keys. CompositeKeys are freely copyable and all copies of a key
// share the same interned data which is owned by the CompositeKeyStore and
// sub-allocated from larger blocks of memory.
class CompositeKey {
 public:
  explicit CompositeKey(char* data) : data_(data) {}
  char* data() const { return data_; }

  // Equality for CompositeKey - this is done as if the key was a string_view
  // of the fixed size.
  class Equal {
   public:
    explicit Equal(size_t key_size) : key_size_(key_size) {}
    bool operator()(const CompositeKey& l, const CompositeKey& r) const {
      return absl::string_view(l.data(), key_size_) ==
             absl::string_view(r.data(), key_size_);
    }

   private:
    size_t key_size_;
  };

  // Hash for CompositeKey - this is done as if the key was a string_view of
  // the fixed size.
  class Hash {
   public:
    explicit Hash(size_t key_size) : key_size_(key_size) {}
    size_t operator()(const CompositeKey& key) const {
      return absl::HashOf(absl::string_view(key.data(), key_size_));
    }

   private:
    size_t key_size_;
  };

 private:
  char* data_;
};

// CompositeKeyMap is a hash map that maps composite keys to ordinals.
// The hash and equality operators for the map interpret the keys as if
// they were string_views of a fixed size. The map stores CompositeKey
// instances, but the actual key data is externally stored (e.g. inside
// CompositeKeyStore) and must be stable in memory.
class CompositeKeyMap final
    : public absl::flat_hash_map<CompositeKey, int64_t, CompositeKey::Hash,
                                 CompositeKey::Equal> {
 public:
  explicit CompositeKeyMap(size_t key_size)
      : absl::flat_hash_map<CompositeKey, int64_t, CompositeKey::Hash,
                            CompositeKey::Equal>(
            {}, CompositeKey::Hash(key_size), CompositeKey::Equal(key_size)) {}

  // Inserts the given composite key in the map if it doesn't already exist, or
  // retrieves the ordinal associated with the composite key.
  //
  // Returns the ordinal associated with the composite key, which is incremented
  // with each new unique composite key, and a boolean indicating whether the
  // composite key was newly inserted.
  inline std::pair<int64_t, bool> InsertAndGetOrdinal(
      CompositeKey composite_key) {
    auto [it, inserted] = insert({composite_key, size()});
    // return the ordinal associated with the composite key
    return {it->second, inserted};
  }
};

// CompositeKeyStore owns the memory for the composite keys.
class CompositeKeyStore final {
 public:
  virtual ~CompositeKeyStore() = default;

  explicit CompositeKeyStore(size_t key_size, size_t block_size_in_keys = 1024)
      : key_size_(key_size),
        block_size_(block_size_in_keys * key_size),
        key_storage_(1, NextBlock()),
        current_block_(&key_storage_.back()) {}

  // Returns a key that points to the first available slot in the current
  // block. Please note that multiple calls to this method will return keys
  // that point to the same address unless the AdvanceKey() method is called,
  // which moves the pointer to the next available slot.
  inline CompositeKey CurrentKey() const {
    return CompositeKey{
        reinterpret_cast<char*>(current_block_->data() + next_key_offset_)};
  }

  // Advances the key to the next available slot, allocating a new block if
  // necessary.
  inline void AdvanceKey() {
    next_key_offset_ += key_size_;
    // Check if there is enough space in the current block for another key.
    if (next_key_offset_ + key_size_ > current_block_->size()) {
      // Allocate a new block to get ready for the next key.
      key_storage_.push_back(NextBlock());
      current_block_ = &key_storage_.back();
      next_key_offset_ = 0;
    }
  }

 private:
  // Key blocks are allocated as FixedArray<uint8_t> of size block_size_ to
  // ensure pointer stability of existing keys pointing into the block.
  using KeyBlock = absl::FixedArray<uint8_t>;

  inline KeyBlock NextBlock() const {
    return absl::FixedArray<uint8_t>(block_size_);
  }

  // The size of the composite key in bytes.
  size_t key_size_;
  // The size of a block of keys in bytes.
  size_t block_size_;
  // Key storage, pre-allocated to hold one block of keys. Using a list to
  // provide pointer stability for the key pointers.
  std::list<KeyBlock> key_storage_;
  KeyBlock* current_block_;
  // Next key offset in the current block of keys.
  size_t next_key_offset_ = 0;
};

using InternPool = absl::node_hash_set<std::string>;

// Class operating on sets of tensors of the same shape to combine indices for
// which the same combination of elements occurs, or in other words, indices
// containing the same composite key.
//
// This class contains two methods: Accumulate and GetOutputKeys, which can
// each be called multiple times.
//
// Accumulate takes in an InputTensorList of tensors of the same shape, and
// returns a Tensor of the same shape containing ordinals to represent the
// composite key that exists at each index. Composite keys are stored
// across calls to Accumulate, so if the same composite key is ever
// encountered in two different indices, whether in the same or a different
// call to Accumulate, the same ordinal will be returned in both these
// indices.
//
// GetOutputKeys returns the composite keys that have been seen in all
// previous calls to Accumulate, represented by a vector of Tensors. If the
// ordinal returned by Accumulate for that composite key was i, the composite
// key will be found at position i in the output vector.
//
// This class is not thread safe.
class CompositeKeyCombiner {
 public:
  virtual ~CompositeKeyCombiner() = default;

  // CompositeKeyCombiner is not copyable or moveable.
  CompositeKeyCombiner(const CompositeKeyCombiner&) = delete;
  CompositeKeyCombiner& operator=(const CompositeKeyCombiner&) = delete;
  CompositeKeyCombiner(CompositeKeyCombiner&&) = delete;
  CompositeKeyCombiner& operator=(CompositeKeyCombiner&&) = delete;

  // Creates a CompositeKeyCombiner if inputs are valid or crashes otherwise.
  explicit CompositeKeyCombiner(std::vector<DataType> dtypes);

  // Returns a single tensor containing the ordinals of the composite keys
  // formed from the tensors in the InputTensorList.
  //
  // The shape of each of the input tensors must match the shape provided to
  // the constructor, and the dtypes of the input tensors must match the
  // dtypes provided to the constructor.
  //
  // For each index in the input tensors, the combination of elements from
  // each tensor at that index forms a "composite key." Across calls to
  // Accumulate, each unique composite key will be represented by a unique
  // ordinal.
  //
  // The returned tensor is of data type DT_INT64 and the same shape that was
  // provided to the constructor.
  virtual StatusOr<Tensor> Accumulate(const InputTensorList& tensors);

  // Obtains the vector of output keys ordered by their representative
  // ordinal.
  //
  // The datatypes of the tensors in the output vector will match the data
  // types provided to the constructor.
  //
  // For each unique combination of elements that was seen across all calls to
  // Accumulate on this class so far, the vector of output tensors will
  // include that combination of elements. The ordering of the elements within
  // the output tensors will correspond to the ordinals returned by
  // Accumulate. For example, if Accumulate returned the integer 5 in the
  // output tensor at position 8 when it encountered this combination of
  // elements in the input tensor list at position 8, then the elements in the
  // composite key will appear at position 5 in the output tensors returned by
  // this method.
  OutputTensorList GetOutputKeys() const;

  // Gets a reference to the expected types for this CompositeKeyCombiner.
  const std::vector<DataType>& dtypes() const { return dtypes_; }

  // Checks that the provided InputTensorList can be accumulated into this
  // CompositeKeyCombiner.
  StatusOr<TensorShape> CheckValidAndGetShape(
      const InputTensorList& tensors) const;

 protected:
  // Creates ordinals for composite keys spread across input tensors.
  // Specifically, the i-th entry of the output is the ordinal for the
  // composite key made by combining the i-th entry of the first tensor, the
  // i-th entry of the second tensor, ...
  //
  // The implementation transfers the bytes of the tensors into a composite key
  // then inserts each composite key into the CompositeKeyMap and obtains an
  // new ordinal for each unique composite key.
  //
  // The CompositeKeyMap is explicitly passed in to allow for temporary data
  // structures to be used in DPCompositeKeyCombiner::AccumulateWithBound.
  std::unique_ptr<MutableVectorData<int64_t>> CreateOrdinals(
      const InputTensorList& tensors, size_t num_elements,
      CompositeKeyMap& composite_key_map);

  // Functions to grant access to members
  inline size_t GetCompositeKeySize() { return composite_key_size_; }
  inline CompositeKeyMap& GetCompositeKeys() { return composite_keys_; }
  inline InternPool& GetInternPool() { return *intern_pool_; }
  // Creates a new composite key.
  inline CompositeKey NewCompositeKey() {
    return composite_key_store_.CurrentKey();
  }

 private:
  // The data types of the tensors in valid inputs to Accumulate, in this
  // exact order.
  std::vector<DataType> dtypes_;
  // The size of the composite key in bytes.
  size_t composite_key_size_;
  // Set of unique strings encountered in tensors of type DT_STRING on calls
  // to Accumulate. Used as an optimization to avoid storing the same string
  // multiple times even if it appears in many composite keys.
  std::shared_ptr<InternPool> intern_pool_;
  // Composite key store.
  CompositeKeyStore composite_key_store_;
  // Mapping of composite keys to their ordinal.
  CompositeKeyMap composite_keys_;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_COMPOSITE_KEY_COMBINER_H_
