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
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/random/random.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
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
    // SaveCompositeKeyAndGetOrdinal. Give that function access to the members
    // of key_combiner_ so that novel composite keys will be assigned ordinals
    // that have not yet been assigned.
    (*(this->container))[p.second] = SaveCompositeKeyAndGetOrdinal(
        std::move(p.first), key_combiner_->GetCompositeKeys(),
        key_combiner_->GetCompositeKeyNext(), key_combiner_->GetKeyVec());
    return *this;
  }

 private:
  DPCompositeKeyCombiner* key_combiner_;
};

DPCompositeKeyCombiner::DPCompositeKeyCombiner(
    const std::vector<DataType>& dtypes, int64_t l0_bound)
    : CompositeKeyCombiner(dtypes),
      l0_bound_(l0_bound),
      bitgen_(absl::BitGen()) {
  TFF_CHECK(l0_bound_ > 0) << "l0_bound must be positive";
}

StatusOr<Tensor> DPCompositeKeyCombiner::Accumulate(
    const InputTensorList& tensors) {
  TFF_ASSIGN_OR_RETURN(TensorShape shape, CheckValidAndGetShape(tensors));

  TFF_ASSIGN_OR_RETURN(size_t num_elements, shape.NumElements());

  // We do not need to perform contribution bounding if the contribution is
  // already bounded.
  return (num_elements <= l0_bound_)
             ? CompositeKeyCombiner::Accumulate(tensors)
             : AccumulateWithBound(tensors, shape, num_elements);
}

StatusOr<Tensor> DPCompositeKeyCombiner::AccumulateWithBound(
    const InputTensorList& tensors, TensorShape& shape, size_t num_elements) {
  // The following contains unique composite keys in the order they were first
  // created. We call the index of a composite key in this vector its "local
  // ordinal," as it is only meaningful within one Accumulate call.
  std::vector<const uint64_t*> local_key_vec;
  local_key_vec.reserve(num_elements);

  // The following maps a view of a composite key to its local ordinal.
  absl::node_hash_map<CompositeKey, int64_t> composite_keys_to_local_ordinal;
  composite_keys_to_local_ordinal.reserve(num_elements);

  int64_t local_ordinal = 0;

  // The i-th element of the following is the local ordinal associated with the
  // i-th composite key. Created the same way CompositeKeyCombiner::Accumulate
  // creates ordinals but datastructures for lookup & storage are local to this
  // function call, instead of being class members.
  std::unique_ptr<MutableVectorData<int64_t>> local_ordinals =
      CreateOrdinals(tensors, num_elements, composite_keys_to_local_ordinal,
                     local_ordinal, local_key_vec);

  // Create a mapping from local ordinals to global ordinals. Default to -1.
  std::vector<int64_t> local_to_global(num_elements, -1);
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
    ordinal = local_to_global[ordinal];
  }
  return Tensor::Create(internal::TypeTraits<int64_t>::kDataType, shape,
                        std::move(local_ordinals));
}

}  // namespace aggregation
}  // namespace tensorflow_federated
