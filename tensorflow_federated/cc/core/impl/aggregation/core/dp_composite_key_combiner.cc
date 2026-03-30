/*
 * Copyright 2024 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/random/random.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/domain_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {
// Advances the pointer by T bytes.
template <typename T>
void AdvancePtr(void*& ptr) {
  ptr = static_cast<void*>(static_cast<uint8_t*>(ptr) + sizeof(T));
}

// In order to MakeCompositeKeyFromDomainTensors, we need a variant of
// CopyToDest that takes an index. That is, we need to be able to retrieve the
// datum at a given index of the array that a source_ptr points to, then copy it
// to a dest_ptr. We do not advance source_ptr because we have indexing.
template <typename T>
void IndexedCopyToDest(const void* source_ptr, size_t index, void*& dest_ptr,
                       absl::node_hash_set<std::string>& intern_pool) {
  const T& source_data = static_cast<const T*>(source_ptr)[index];
  // Copy the bytes pointed to by source_ptr to the destination pointed to by
  // dest_ptr.
  std::memcpy(dest_ptr, &source_data, sizeof(T));
  // Advance dest_ptr to the next T.
  AdvancePtr<T>(dest_ptr);
}

// Specialization of IndexedCopyToDest for DT_STRING data type that interns the
// string_view.
template <>
void IndexedCopyToDest<string_view>(
    const void* source_ptr, size_t index, void*& dest_ptr,
    absl::node_hash_set<std::string>& intern_pool) {
  const string_view& source_data =
      static_cast<const string_view*>(source_ptr)[index];
  // Insert the string into the intern pool if it does not already exist. This
  // makes a copy of the string so that the intern pool owns the storage.
  auto it = intern_pool.emplace(source_data).first;
  // The iterator of a node_hash_set set may be invalidated by inserting more
  // elements, but the pointer to the underlying element is guaranteed to be
  // stable.
  // Thus, get the address of the string after dereferencing the iterator.
  const std::string* interned_string_ptr = &*it;
  // The stable address of the string can be interpreted as a 64-bit integer.
  intptr_t ptr_int = reinterpret_cast<intptr_t>(interned_string_ptr);
  // Set the destination storage to the integer representation of the string
  // address.
  std::memcpy(dest_ptr, &ptr_int, sizeof(intptr_t));
  // Advance the dest_ptr.
  AdvancePtr<intptr_t>(dest_ptr);
}
}  // namespace

// Class that supports DPCompositeKeyCombiner::AccumulateWithBound. Each call to
// AccumulateWithBound creates a temporary mapping from composite keys to
// integers; we call them "local ordinals" to disambiguate from the ordinals
// maintained by CompositeKeyCombiner.
// LocalToGlobalInserter helps create a vector such that the integer at index i
// is the ordinal for the composite key whose local ordinal is i.
class LocalToGlobalInserter
    : public std::insert_iterator<std::vector<int64_t>> {
 public:
  // Constructor that takes the same input as insert_iterator's, along with a
  // pointer to a DPCompositeKeyCombiner. The pointer will be used to update its
  // member data structures.
  LocalToGlobalInserter(std::vector<int64_t>& c,
                        std::vector<int64_t>::iterator i,
                        DPCompositeKeyCombiner* key_combiner)
      : std::insert_iterator<std::vector<int64_t>>(c, i),
        key_combiner_(key_combiner) {}

  // These operators are no-ops, just like in insert_iterator (see
  // https://en.cppreference.com/w/cpp/iterator/insert_iterator), but are still
  // needed for compilation to succeed.
  constexpr LocalToGlobalInserter& operator*() { return *this; }
  constexpr LocalToGlobalInserter& operator++() { return *this; }
  constexpr LocalToGlobalInserter& operator++(int v) { return *this; }

  // An r-value assignment operator suffices for our purposes: the input will
  // provided by std::sample, which iterates through an
  // absl::node_hash_map<CompositeKey, int64_t>. The
  // FixedArray<uint64_t> is a composite key and the int is its local ordinal
  LocalToGlobalInserter& operator=(std::pair<CompositeKey, int64_t>&& p) {
    // Map the local ordinal to the ordinal produced by
    // InsertAndGetOrdinal called on the CompositeKeyMap from key_combiner_ so
    // that novel composite keys will be assigned ordinals that have not yet
    // been assigned.
    (*(this->container))[p.second] =
        key_combiner_->GetCompositeKeys()
            .InsertAndGetOrdinal(std::move(p.first))
            .first;
    return *this;
  }

 private:
  DPCompositeKeyCombiner* key_combiner_;
};

DPCompositeKeyCombiner::DPCompositeKeyCombiner(
    const std::vector<DataType>& dtypes, int64_t l0_bound,
    std::optional<DomainSpec> domain_spec)
    : CompositeKeyCombiner(dtypes),
      l0_bound_(l0_bound),
      domain_spec_(std::move(domain_spec)),
      bitgen_(absl::BitGen()) {}

StatusOr<Tensor> DPCompositeKeyCombiner::Accumulate(
    const InputTensorList& tensors) {
  TFF_ASSIGN_OR_RETURN(TensorShape shape, CheckValidAndGetShape(tensors));

  TFF_ASSIGN_OR_RETURN(size_t num_elements, shape.NumElements());

  if (l0_bound_ > 0 && num_elements > l0_bound_) {
    return AccumulateWithBound(tensors, shape, num_elements);
  }

  if (domain_spec_.has_value()) {
    return Tensor::Create(
        internal::TypeTraits<int64_t>::kDataType, shape,
        CreateInDomainOrdinals(tensors, num_elements, GetCompositeKeys()));
  }

  return CompositeKeyCombiner::Accumulate(tensors);
}

StatusOr<Tensor> DPCompositeKeyCombiner::AccumulateWithBound(
    const InputTensorList& tensors, TensorShape& shape, size_t num_elements) {
  // The following maps a view of a composite key to its local ordinal.
  CompositeKeyMap composite_keys_to_local_ordinal(GetCompositeKeySize());
  composite_keys_to_local_ordinal.reserve(num_elements);

  // The i-th element of the following is the local ordinal associated with the
  // i-th composite key. Created the same way CompositeKeyCombiner::Accumulate
  // creates ordinals but datastructures for lookup & storage are local to this
  // function call, instead of being class members.
  std::unique_ptr<MutableVectorData<int64_t>> local_ordinals =
      CreateOrdinals(tensors, num_elements, composite_keys_to_local_ordinal);

  // Create a mapping from local ordinals to global ordinals.
  // Default to kNoOrdinal.
  std::vector<int64_t> local_to_global(num_elements, kNoOrdinal);
  local_to_global.reserve(l0_bound_);

  // Sample l0_bound_ of the composite keys. For each composite_key that is
  // sampled, update the local_to_global map.
  // The process involves std::move of
  // strings so composite_keys_to_local_ordinal will be invalidated.
  LocalToGlobalInserter inserter(local_to_global, local_to_global.begin(),
                                 this);
  std::sample(composite_keys_to_local_ordinal.begin(),
              composite_keys_to_local_ordinal.end(), inserter, l0_bound_,
              bitgen_);

  // Finally, transform the local ordinals into global ordinals
  for (int64_t& ordinal : *local_ordinals) {
    // The ordinal is kNoOrdinal if the provided key was not in the domain spec.
    // If that is the case, we do not update the ordinal.
    if (ordinal != kNoOrdinal) {
      ordinal = local_to_global[ordinal];
    }
  }
  return Tensor::Create(internal::TypeTraits<int64_t>::kDataType, shape,
                        std::move(local_ordinals));
}

// Retrieve the ordinal associated with the composite key formed by the data in
// domain_tensors at the given indices (or kNoOrdinal if not found).
int64_t DPCompositeKeyCombiner::GetOrdinal(
    TensorSpan domain_tensors, const absl::FixedArray<int64_t>& indices) {
  for (auto index : indices) {
    TFF_CHECK(index >= 0) << "DPCompositeKeyCombiner::GetOrdinal: Indices must "
                             "be non-negative.";
  }

  size_t expected_num_keys = dtypes().size();
  TFF_CHECK(domain_tensors.size() == expected_num_keys)
      << "DPCompositeKeyCombiner::GetOrdinal: The number of tensors in the "
         "output tensor list must match the number of keys expected by the "
         "CompositeKeyCombiner.";
  TFF_CHECK(indices.size() == expected_num_keys)
      << "DPCompositeKeyCombiner::GetOrdinal: The number of indices must match "
         "the number of keys expected by the CompositeKeyCombiner.";

  CompositeKey composite_key =
      MakeCompositeKeyFromDomainTensors(domain_tensors, indices);
  const auto& it = GetCompositeKeys().find(composite_key);
  return it == GetCompositeKeys().end() ? kNoOrdinal : it->second;
}

// Populates composite_key with data drawn from domain_tensors. Indices specify
// which domain elements to copy.
CompositeKey DPCompositeKeyCombiner::MakeCompositeKeyFromDomainTensors(
    TensorSpan domain_tensors, const absl::FixedArray<int64_t>& indices) {
  auto data_type_iter = dtypes().begin();
  for (auto& domain_tensor : domain_tensors) {
    // Check that the data types of the input tensors match those provided to
    // the constructor.
    TFF_CHECK(*data_type_iter == domain_tensor.dtype())
        << "DPCompositeKeyCombiner::GetOrdinal: Data types must match. Got "
        << domain_tensor.dtype() << " but expected " << *data_type_iter;
    data_type_iter++;
  }

  auto index_iter = indices.begin();
  data_type_iter = dtypes().begin();
  CompositeKey composite_key = NewCompositeKey();
  void* dest_ptr = composite_key.data();
  for (auto& domain_tensor : domain_tensors) {
    // Copy over data from the current tensor at the given index.
    const void* source_ptr = domain_tensor.data().data();
    DTYPE_CASES(*data_type_iter, T,
                IndexedCopyToDest<T>(source_ptr, *index_iter, dest_ptr,
                                     GetInternPool()));

    // Advance the data-type and tensor iterators.
    data_type_iter++;
    index_iter++;
  }
  return composite_key;
}

std::unique_ptr<MutableVectorData<int64_t>>
DPCompositeKeyCombiner::CreateOrdinals(const InputTensorList& tensors,
                                       size_t num_elements,
                                       CompositeKeyMap& composite_key_map) {
  if (domain_spec_.has_value()) {
    return CreateInDomainOrdinals(tensors, num_elements, composite_key_map);
  }
  return CompositeKeyCombiner::CreateOrdinals(tensors, num_elements,
                                              composite_key_map);
}

std::unique_ptr<MutableVectorData<int64_t>>
DPCompositeKeyCombiner::CreateInDomainOrdinals(
    const InputTensorList& tensors, size_t num_elements,
    CompositeKeyMap& composite_key_map) {
  // Unlike CompositeKeyCombiner::CreateOrdinals, we initialize all ordinals to
  // kNoOrdinal. This is because we will skip composite keys that are not in the
  // domain specified by domain_spec_, and we want to ensure that those keys
  // are assigned kNoOrdinal.
  auto ordinals =
      std::make_unique<MutableVectorData<int64_t>>(num_elements, kNoOrdinal);

  std::vector<const void*> iterators;
  iterators.reserve(tensors.size());
  for (const Tensor* t : tensors) {
    iterators.push_back(t->data().data());
  }

  for (size_t i = 0; i < num_elements; ++i) {
    CompositeKey composite_key = NewCompositeKey();
    void* key_ptr = composite_key.data();
    char* original_key_ptr = static_cast<char*>(key_ptr);
    auto data_type_iter = dtypes().begin();
    // Before assigning an ordinal to a composite key, we check if it is in the
    // domain specified by domain_spec_.
    bool in_domain = true;
    for (int j = 0; j < iterators.size(); ++j) {
      const void* itr = iterators[j];
      DTYPE_CASES(
          *(data_type_iter), T,
          in_domain =
              in_domain &&
              domain_spec_->IsMember<T>(static_cast<const T*>(itr)[i], j)
                  .value_or(false));
      DTYPE_CASES(*(data_type_iter++), T,
                  IndexedCopyToDest<T>(itr, i, key_ptr, GetInternPool()));
    }

    // If a composite key is not in domain, we do not insert it into the
    // composite_key_map, and its corresponding ordinal remains kNoOrdinal.
    if (!in_domain) {
      continue;
    }

    auto [assigned_ordinal, inserted] =
        composite_key_map.InsertAndGetOrdinal(composite_key);
    (*ordinals)[i] = assigned_ordinal;
    if (inserted) {
      AdvanceCompositeKey(original_key_ptr);
    }
  }
  return ordinals;
}

}  // namespace aggregation
}  // namespace tensorflow_federated
