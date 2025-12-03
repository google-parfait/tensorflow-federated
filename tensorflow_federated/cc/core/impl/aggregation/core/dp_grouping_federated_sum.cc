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

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

// Below is an implementation of a sum grouping aggregator for numeric types,
// with clipping of Linfinity, L1, and L2 norms as determined by the
// parameters linfinity_bound_, l1_bound_, and l2_bound_. They can take on non-
// positive values, in which case the aggregator does not make any adjustments
// to data.
template <typename InputT, typename OutputT>
class DPGroupingFederatedSum final
    : public OneDimGroupingAggregator<InputT, OutputT> {
 public:
  using OneDimGroupingAggregator<InputT, OutputT>::OneDimGroupingAggregator;
  using OneDimGroupingAggregator<InputT, OutputT>::data;

  DPGroupingFederatedSum(InputT linfinity_bound, double l1_bound,
                         double l2_bound)
      : OneDimGroupingAggregator<InputT, OutputT>(),
        linfinity_bound_(linfinity_bound),
        l1_bound_(l1_bound),
        l2_bound_(l2_bound) {}

  DPGroupingFederatedSum(InputT linfinity_bound, double l1_bound,
                         double l2_bound,
                         std::unique_ptr<MutableVectorData<OutputT>> data,
                         int num_inputs)
      : OneDimGroupingAggregator<InputT, OutputT>(std::move(data), num_inputs),
        linfinity_bound_(linfinity_bound),
        l1_bound_(l1_bound),
        l2_bound_(l2_bound) {}

 private:
  // The following method clamps the input value to the linfinity bound if given
  // TODO: b/354733266 - When the intrinsic is updated to have min- and max-
  // contributions, clamp to that range instead of [0, linfinity_bound_].
  inline InputT Clamp(const InputT& input_value) {
    return (linfinity_bound_ <= 0)
               ? input_value
               : std::min(std::max(input_value, static_cast<InputT>(0)),
                          linfinity_bound_);
  }

  // The following method returns a scalar such that, when it is applied to
  // the clamped version of local histogram, the l1 and l2 norms are at most
  // l1_bound_ and l2_bound_.
  inline double ComputeRescalingFactor(
      const absl::flat_hash_map<int64_t, InputT>& local_histogram) {
    // no re-scaling if norm bounds were not provided
    if (l1_bound_ <= 0 && l2_bound_ <= 0) {
      return 1.0;
    }

    // Compute norms after clamping magnitudes.
    double l1 = 0;
    double squared_l2 = 0;
    for (const auto& [unused, raw_value] : local_histogram) {
      // To do: optimize the number of Clamp calls. Currently called once in
      // this function and again in the final loop of AggregateVectorByOrdinals.
      InputT value = Clamp(raw_value);
      l1 += (value < 0) ? -value : value;
      squared_l2 += static_cast<double>(value) * static_cast<double>(value);
    }
    double l2 = sqrt(squared_l2);

    // Compute rescaling factor based on the norms.
    double rescaling_factor = 1.0;
    if (l1_bound_ > 0 && l1 > 0 && l1_bound_ / l1 < 1.0) {
      rescaling_factor = l1_bound_ / l1;
    }
    if (l2_bound_ > 0 && l2 > 0 && l2_bound_ / l2 < rescaling_factor) {
      rescaling_factor = l2_bound_ / l2;
    }
    return rescaling_factor;
  }

  // The following method is very much the same as GroupingFederatedSum's
  // except it clamps and rescales value_vector.
  void AggregateVectorByOrdinals(
      const AggVector<int64_t>& ordinals_vector,
      const AggVector<InputT>& value_vector) override {
    auto value_it = value_vector.begin();

    // Create a local histogram from ordinals & values, aggregating when there
    // are multiple values for the same ordinal.
    absl::flat_hash_map<int64_t, InputT> local_histogram;
    local_histogram.reserve(ordinals_vector.size());
    for (const auto& [index, ordinal] : ordinals_vector) {
      TFF_CHECK(value_it.index() == index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";

      // Only aggregate values of valid ordinals.
      if (ordinal >= 0) {
        local_histogram[ordinal] += value_it.value();
      }

      value_it++;
    }

    double rescaling_factor = ComputeRescalingFactor(local_histogram);

    // Propagate to the actual state
    for (const auto& [ordinal, value] : local_histogram) {
      // Compute the scaled value to satisfy the L1 and L2 constraints.
      double scaled_value = Clamp(value) * rescaling_factor;
      DCHECK(ordinal < data().size())
          << "Ordinal too big: " << ordinal << " vs. " << data().size();
      AggregateValue(ordinal, static_cast<OutputT>(scaled_value));
    }
  }

  // Norm bounds should not be applied when merging, since this input data
  // represents the pre-accumulated (and already per-client bounded) data from
  // multiple clients.
  void MergeVectorByOrdinals(const AggVector<int64_t>& ordinals_vector,
                             const AggVector<OutputT>& value_vector) override {
    auto value_it = value_vector.begin();
    for (auto o : ordinals_vector) {
      int64_t output_index = o.value;
      TFF_CHECK(value_it.index() == o.index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";
      AggregateValue(output_index, value_it++.value());
    }
  }

  inline void AggregateValue(int64_t i, OutputT value) { data()[i] += value; }
  OutputT GetDefaultValue() override { return OutputT{0}; }

  const InputT linfinity_bound_;
  const double l1_bound_;
  const double l2_bound_;
};

// Make a DPGFS object out of norm bounds and aggregator state, if provided.
template <typename InputT, typename OutputT>
StatusOr<std::unique_ptr<TensorAggregator>> CreateDPGroupingFederatedSum(
    InputT linfinity_bound, double l1_bound, double l2_bound,
    const OneDimGroupingAggregatorState* aggregator_state) {
  return aggregator_state == nullptr
             ? std::make_unique<DPGroupingFederatedSum<InputT, OutputT>>(
                   linfinity_bound, l1_bound, l2_bound)
             : std::make_unique<DPGroupingFederatedSum<InputT, OutputT>>(
                   linfinity_bound, l1_bound, l2_bound,
                   MutableVectorData<OutputT>::CreateFromEncodedContent(
                       aggregator_state->vector_data()),
                   aggregator_state->num_inputs());
}

// Same as above except input and output types are identical.
template <typename T>
StatusOr<std::unique_ptr<TensorAggregator>> CreateDPGroupingFederatedSum(
    T linfinity_bound, double l1_bound, double l2_bound,
    const OneDimGroupingAggregatorState* aggregator_state) {
  return CreateDPGroupingFederatedSum<T, T>(linfinity_bound, l1_bound, l2_bound,
                                            aggregator_state);
}
template <>
StatusOr<std::unique_ptr<TensorAggregator>> CreateDPGroupingFederatedSum(
    string_view linfinity_bound, double l1_bound, double l2_bound,
    const OneDimGroupingAggregatorState* aggregator_state) {
  return TFF_STATUS(INVALID_ARGUMENT)
         << "DPGroupingFederatedSumFactory: DPGroupingFederatedSum only"
            " supports numeric datatypes.";
}

// A factory class for the GroupingFederatedSum.
// Permits parameters in the DPGroupingFederatedSum intrinsic,
// unlike GroupingFederatedSumFactory.
class DPGroupingFederatedSumFactory final
    : public OneDimBaseGroupingAggregatorFactory {
 public:
  DPGroupingFederatedSumFactory() = default;

  // DPGroupingFederatedSumFactory is not copyable or moveable.
  DPGroupingFederatedSumFactory(const DPGroupingFederatedSumFactory&) = delete;
  DPGroupingFederatedSumFactory& operator=(
      const DPGroupingFederatedSumFactory&) = delete;

 private:
  StatusOr<std::unique_ptr<TensorAggregator>> CreateInternal(
      const Intrinsic& intrinsic,
      const OneDimGroupingAggregatorState* aggregator_state) const override {
    TFF_CHECK(kDPSumUri == intrinsic.uri)
        << "DPGroupingFederatedSumFactory: Expected intrinsic URI " << kDPSumUri
        << " but got uri " << intrinsic.uri;
    // Check that the configuration is valid for grouping_federated_sum.
    if (intrinsic.inputs.size() != 1) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupingFederatedSumFactory: Exactly one input "
                "is expected but got "
             << intrinsic.inputs.size();
    }

    if (intrinsic.outputs.size() != 1) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupingFederatedSumFactory: Exactly one output tensor is "
                "expected but got "
             << intrinsic.outputs.size();
    }

    if (!intrinsic.nested_intrinsics.empty()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupingFederatedSumFactory: Not expected to have inner "
                "aggregations.";
    }

    const TensorSpec& input_spec = intrinsic.inputs[0];
    const TensorSpec& output_spec = intrinsic.outputs[0];
    if (input_spec.shape() != output_spec.shape()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupingFederatedSumFactory: Input and output tensors have "
                "mismatched shapes.";
    }

    const auto& linfinity_tensor = intrinsic.parameters[kLinfinityIndex];
    const double l1 = intrinsic.parameters[kL1Index].CastToScalar<double>();
    const double l2 = intrinsic.parameters[kL2Index].CastToScalar<double>();

    const DataType input_type = input_spec.dtype();
    const DataType output_type = output_spec.dtype();

    if (internal::GetTypeKind(input_type) != internal::TypeKind::kNumeric) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPGroupingFederatedSumFactory: DPGroupingFederatedSum only"
                " supports numeric datatypes.";
    }

    if (input_type != output_type) {
      // In the GoogleSQL spec, summing floats produces doubles and summing
      // int32 produces int64. Allow the input and output type to differ in this
      // case.
      if (input_type == DataType::DT_INT32 &&
          output_type == DataType::DT_INT64) {
        int32_t linfinity_bound = linfinity_tensor.CastToScalar<int32_t>();
        return CreateDPGroupingFederatedSum<int32_t, int64_t>(
            linfinity_bound, l1, l2, aggregator_state);
      } else if (input_type == DataType::DT_FLOAT &&
                 output_type == DataType::DT_DOUBLE) {
        float linfinity_bound = linfinity_tensor.CastToScalar<float>();
        return CreateDPGroupingFederatedSum<float, double>(
            linfinity_bound, l1, l2, aggregator_state);
      } else {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "DPGroupingFederatedSumFactory: Input and output tensors "
                  "have mismatched dtypes: input tensor has dtype "
               << DataType_Name(input_type) << " and output tensor has dtype "
               << DataType_Name(output_type);
      }
    }

    // Edge case: linfinity tensor is negative (not provided) but the input type
    // is unsigned. Reset to 0 so the no-clamping behavior is preserved.
    if (linfinity_tensor.CastToScalar<double>() < 0 &&
        input_type == DataType::DT_UINT64) {
      return CreateDPGroupingFederatedSum<uint64_t, uint64_t>(
          /*linfinity_bound=*/0, l1, l2, aggregator_state);
    }

    StatusOr<std::unique_ptr<TensorAggregator>> aggregator;
    DTYPE_CASES(
        input_type, T,
        aggregator = CreateDPGroupingFederatedSum<T>(
            linfinity_tensor.CastToScalar<T>(), l1, l2, aggregator_state));
    return aggregator;
  }
};

REGISTER_AGGREGATOR_FACTORY(kDPSumUri, DPGroupingFederatedSumFactory);

}  // namespace aggregation
}  // namespace tensorflow_federated
