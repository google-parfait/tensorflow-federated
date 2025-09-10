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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_closed_domain_histogram.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "algorithms/numerical-mechanisms.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_noise_mechanisms.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/group_by_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {
using differential_privacy::NumericalMechanism;

namespace {
// Given a tensor containing a column of aggregates and an ordinal, push the
// aggregate associated with that ordinal to the back of a MutableVectorData
// container. If the ordinal is kNoOrdinal, push 0 instead.
// Adds noise if a mechanism is provided.
template <typename T>
void CopyAggregateFromColumn(const Tensor& column_of_aggregates,
                             int64_t ordinal, MutableVectorData<T>& container,
                             NumericalMechanism* mechanism) {
  T zero = static_cast<T>(0);
  T noise_to_add =
      (mechanism == nullptr) ? zero : mechanism->AddNoise(/*result=*/zero);

  T value_to_push_back =
      (ordinal == kNoOrdinal)
          ? noise_to_add
          : noise_to_add + column_of_aggregates.AsSpan<T>()[ordinal];

  // Add noisy value to the container.
  container.push_back(value_to_push_back);
}

}  // namespace

DPClosedDomainHistogram::DPClosedDomainHistogram(
    const std::vector<TensorSpec>& input_key_specs,
    const std::vector<TensorSpec>* output_key_specs,
    const std::vector<Intrinsic>* intrinsics,
    std::unique_ptr<CompositeKeyCombiner> key_combiner,
    std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
    double epsilon_per_agg, double delta_per_agg, int64_t l0_bound,
    TensorSpan domain_tensors, int num_inputs)
    : GroupByAggregator(input_key_specs, output_key_specs, intrinsics,
                        std::move(key_combiner), std::move(aggregators),
                        num_inputs),
      epsilon_per_agg_(epsilon_per_agg),
      delta_per_agg_(delta_per_agg),
      l0_bound_(l0_bound),
      domain_tensors_(domain_tensors) {}

StatusOr<Tensor> DPClosedDomainHistogram::CreateOrdinalsByGroupingKeysForMerge(
    const InputTensorList& inputs) {
  if (num_keys_per_input() > 0) {
    InputTensorList keys(num_keys_per_input());
    for (int i = 0; i < num_keys_per_input(); ++i) {
      keys[i] = inputs[i];
    }
    // Call the version of Accumulate that has no L0 bounding
    return key_combiner()->CompositeKeyCombiner::Accumulate(std::move(keys));
  }
  // If there are no keys, we should aggregate all elements in a column into one
  // element, as if there were an imaginary key column with identical values for
  // all rows.
  auto ordinals =
      std::make_unique<MutableVectorData<int64_t>>(inputs[0]->num_elements());
  return Tensor::Create(internal::TypeTraits<int64_t>::kDataType,
                        inputs[0]->shape(), std::move(ordinals));
}

// Advance the indices that specify a composite key. Boolean output indicates if
// we can continue advancing the indices.
bool DPClosedDomainHistogram::IncrementDomainIndices(
    absl::FixedArray<int64_t>& domain_indices, int64_t which_key) {
  ++domain_indices[which_key];
  // If we've reached the end of a key's domain...
  if (domain_indices[which_key] == domain_tensors_[which_key].num_elements()) {
    // ...reset to 0
    domain_indices[which_key] = 0;

    // If we've completely iterated through the composite key domain, indicate
    // that we cannot progress any further.
    if (which_key + 1 == domain_tensors_.size()) {
      return false;
    }

    // Otherwise, recurse (perform a carry)
    return IncrementDomainIndices(domain_indices, which_key + 1);
  }
  // It is still possible to increment the domain indices.
  return true;
}

StatusOr<OutputTensorList> DPClosedDomainHistogram::Report() && {
  TFF_RETURN_IF_ERROR(CheckValid());
  if (!CanReport()) {
    return TFF_STATUS(FAILED_PRECONDITION)
           << "DPOpenDomainHistogram::Report: the report goal isn't met";
  }
  // Compute the noiseless aggregates.
  OutputTensorList noiseless_aggregates = std::move(*this).TakeOutputs();

  OutputTensorList noisy_aggregates;

  // Calculate size of domain of composite keys.
  int64_t domain_size = 1;
  for (const auto& domain_tensor : domain_tensors_) {
    domain_size *= domain_tensor.num_elements();
  }

  // Make a noise mechanism for each aggregation.
  std::vector<std::unique_ptr<NumericalMechanism>> mechanisms;
  for (int i = 0; i < intrinsics().size(); ++i) {
    const Intrinsic& intrinsic = intrinsics()[i];
    // Do not bother making mechanism if epsilon is too large.
    if (epsilon_per_agg_ >= kEpsilonThreshold) {
      mechanisms.push_back(nullptr);
      continue;
    }

    // Get norm bounds for the ith aggregation.
    double linfinity_bound =
        intrinsic.parameters[kLinfinityIndex].CastToScalar<double>();
    double l1_bound = intrinsic.parameters[kL1Index].CastToScalar<double>();
    double l2_bound = intrinsic.parameters[kL2Index].CastToScalar<double>();

    // Create a noise mechanism out of those norm bounds and privacy params.
    TFF_ASSIGN_OR_RETURN(
        DPHistogramBundle noise_mechanism,
        CreateDPHistogramBundle(epsilon_per_agg_, delta_per_agg_, l0_bound_,
                                linfinity_bound, l1_bound, l2_bound,
                                /*open_domain=*/false));
    mechanisms.push_back(std::move(noise_mechanism.mechanism));
  }

  // Create MutableVectorData containers, one for each output tensor, that are
  // each big enough to hold domain_size elements.
  // If all output tensors had the same type like int64_t we could create an
  // std::vector<MutableVectorData<int64_t>> container to hold them.
  // But the types being contained could vary, so we instead make a vector of
  // TensorData pointers that each point to a MutableVectorData object.
  std::vector<std::unique_ptr<TensorData>> noisy_aggregate_data;
  for (const Tensor& tensor : noiseless_aggregates) {
    DTYPE_CASES(tensor.dtype(), T,
                auto container = std::make_unique<MutableVectorData<T>>();
                container->reserve(domain_size);
                noisy_aggregate_data.push_back(std::move(container)););
  }

  // Iterate through the domain of composite keys.
  absl::FixedArray<int64_t> domain_indices(domain_tensors_.size(), 0);
  do {
    // Each composite key is associated with a row of the output. i-th entry of
    // that row will be written to i-th entry of noisy_aggregate_data.

    // Maintain the index of the next key to output.
    int64_t key_to_output = 0;
    // Maintain the index of the next mechanism to use.
    int64_t mech_to_use = 0;

    // Loop to populate the row of the output for the current composite key.
    for (int64_t i = 0; i < noisy_aggregate_data.size(); i++) {
      // Get the TensorData container we will be writing to (a column).
      TensorData& container = *(noisy_aggregate_data[i]);

      // Search for the next key that is specified as part of the output.
      while (key_to_output < num_keys_per_input() &&
             output_key_specs()[key_to_output].name().empty()) {
        key_to_output++;
      }

      // The first batch of entries in this row of the output are the grouping
      // keys that make up the composite key and were specified to be output.
      if (key_to_output < num_keys_per_input()) {
        // Get the tensor that specifies the domain for this key
        const Tensor& domain_tensor = domain_tensors_[key_to_output];

        // Get the index of the value in domain_tensor that we will be copying.
        int64_t index_in_tensor = domain_indices[key_to_output];

        // Copy over to the container after changing to MutableVectorData.
        DTYPE_CASES(domain_tensor.dtype(), T,
                    dynamic_cast<MutableVectorData<T>&>(container).push_back(
                        domain_tensor.AsSpan<T>()[index_in_tensor]));

        // Move on to the next key.
        key_to_output++;
      } else {
        // The second batch of entries are the actual aggregates.

        // Get the ordinal for the composite key indexed by domain_indices
        auto& dp_key_combiner =
            dynamic_cast<DPCompositeKeyCombiner&>(*(key_combiner()));
        int64_t ordinal =
            dp_key_combiner.GetOrdinal(domain_tensors_, domain_indices);

        // Get the current column of aggregates
        const Tensor& column_of_aggregates = noiseless_aggregates[i];

        // Copy the number associated with the ordinal to the container.
        NUMERICAL_ONLY_DTYPE_CASES(
            column_of_aggregates.dtype(), T,
            CopyAggregateFromColumn<T>(
                column_of_aggregates, ordinal,
                dynamic_cast<MutableVectorData<T>&>(container),
                mechanisms[mech_to_use].get()));

        // Move on to the next mechanism.
        mech_to_use++;
      }
    }
  } while (IncrementDomainIndices(domain_indices));

  // Turn the TensorData objects into Tensors.
  for (int64_t i = 0; i < noisy_aggregate_data.size(); ++i) {
    DataType dtype = noiseless_aggregates[i].dtype();
    TensorShape shape({domain_size});
    TFF_ASSIGN_OR_RETURN(
        auto new_tensor,
        Tensor::Create(dtype, shape, std::move(noisy_aggregate_data[i])));
    noisy_aggregates.push_back(std::move(new_tensor));
  }

  return noisy_aggregates;
}

}  // namespace aggregation
}  // namespace tensorflow_federated
