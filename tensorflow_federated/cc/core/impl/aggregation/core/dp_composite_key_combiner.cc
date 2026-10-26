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
    std::optional<DomainSpec> domain_spec,
    std::shared_ptr<CompositeKeyStorage> storage)
    : CompositeKeyCombiner(
          dtypes, storage != nullptr ? std::move(storage)
                                     : std::make_shared<CompositeKeyStorage>(
                                           CalculateCompositeKeySize(dtypes))),
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
      const void*& itr = iterators[j];
      DTYPE_CASES(
          *(data_type_iter), T,
          in_domain = in_domain &&
                      domain_spec_->IsMember<T>(*static_cast<const T*>(itr), j)
                          .value_or(false));
      DTYPE_CASES(*(data_type_iter++), T, CopyToDest<T>(itr, key_ptr));
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
