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

#include <cstdint>
#include <memory>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

constexpr char kGoogleSqlSumUri[] = "GoogleSQL:sum";

// Implementation of a generic sum grouping aggregator for numeric types.
template <typename InputT, typename OutputT>
class GroupingFederatedSum final
    : public OneDimGroupingAggregator<InputT, OutputT> {
 public:
  using OneDimGroupingAggregator<InputT, OutputT>::OneDimGroupingAggregator;
  using OneDimGroupingAggregator<InputT, OutputT>::data;

 private:
  void AggregateVectorByOrdinals(
      const AggVector<int64_t>& ordinals_vector,
      const AggVector<InputT>& value_vector) override {
    auto value_it = value_vector.begin();
    for (auto o : ordinals_vector) {
      int64_t output_index = o.value;
      // If this function returned a failed Status at this point, the
      // data_vector_ may have already been partially modified, leaving the
      // GroupingAggregator in a bad state. Thus, check that the indices of the
      // ordinals tensor and the data tensor match with TFF_CHECK instead.
      //
      // TODO: b/266974165 - Revisit the constraint that the indices of the
      // values must match the indices of the ordinals when sparse tensors are
      // implemented. It may be possible for the value to be omitted for a given
      // ordinal in which case the default value should be used.
      TFF_CHECK(value_it.index() == o.index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";
      // Delegate the actual aggregation to the specific aggregation
      // intrinsic implementation.
      AggregateValue(output_index, OutputT{value_it++.value()});
    }
  }

  // The Merge implementation for GroupingFederatedSum is the same as the
  // Accumulate implementation above, except that the type of input values in
  // the Merge case matches OutputT rather than InputT.
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
};

template <typename OutputT>
StatusOr<std::unique_ptr<TensorAggregator>> CreateGroupingFederatedSum(
    const OneDimGroupingAggregatorState* aggregator_state) {
  if (internal::TypeTraits<OutputT>::type_kind !=
      internal::TypeKind::kNumeric) {
    // Ensure the type is numeric in case new non-numeric types are added.
    return TFF_STATUS(INVALID_ARGUMENT)
           << "GroupingFederatedSum is only supported for numeric datatypes.";
  }
  return aggregator_state == nullptr
             ? std::make_unique<GroupingFederatedSum<OutputT, OutputT>>()
             : std::make_unique<GroupingFederatedSum<OutputT, OutputT>>(
                   MutableVectorData<OutputT>::CreateFromEncodedContent(
                       aggregator_state->vector_data()),
                   aggregator_state->num_inputs());
}

template <>
StatusOr<std::unique_ptr<TensorAggregator>>
CreateGroupingFederatedSum<string_view>(
    const OneDimGroupingAggregatorState* aggregator_state) {
  return TFF_STATUS(INVALID_ARGUMENT)
         << "GroupingFederatedSum isn't supported for DT_STRING datatype.";
}

// Factory class for the GroupingFederatedSum.
class GroupingFederatedSumFactory final
    : public OneDimBaseGroupingAggregatorFactory {
 public:
  GroupingFederatedSumFactory() = default;

  // GroupingFederatedSumFactory isn't copyable or moveable.
  GroupingFederatedSumFactory(const GroupingFederatedSumFactory&) = delete;
  GroupingFederatedSumFactory& operator=(const GroupingFederatedSumFactory&) =
      delete;

 private:
  StatusOr<std::unique_ptr<TensorAggregator>> CreateInternal(
      const Intrinsic& intrinsic,
      const OneDimGroupingAggregatorState* aggregator_state) const override {
    if (kGoogleSqlSumUri != intrinsic.uri) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "GroupingFederatedSumFactory: Expected intrinsic URI "
             << kGoogleSqlSumUri << " but got uri " << intrinsic.uri;
    }
    // Check that the configuration is valid for grouping_federated_sum.
    if (intrinsic.inputs.size() != 1) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "GroupingFederatedSumFactory: Exactly one input "
                "is expected but got "
             << intrinsic.inputs.size();
    }

    if (intrinsic.outputs.size() != 1) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "GroupingFederatedSumFactory: Exactly one output tensor is "
                "expected but got "
             << intrinsic.outputs.size();
    }

    if (!intrinsic.parameters.empty()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "GroupingFederatedSumFactory: No "
                "input parameters expected but got "
             << intrinsic.parameters.size();
    }

    if (!intrinsic.nested_intrinsics.empty()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "GroupingFederatedSumFactory: Not expected to have inner "
                "aggregations.";
    }

    const TensorSpec& input_spec = intrinsic.inputs[0];
    const TensorSpec& output_spec = intrinsic.outputs[0];
    if (input_spec.shape() != output_spec.shape()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "GroupingFederatedSumFactory: Input and output tensors have "
                "mismatched shapes.";
    }

    if (input_spec.dtype() != output_spec.dtype()) {
      // In the GoogleSQL spec, summing floats produces doubles and summing
      // int32 produces int64. Allow the input and output type to differ in this
      // case.
      if (input_spec.dtype() == DataType::DT_INT32 &&
          output_spec.dtype() == DataType::DT_INT64) {
        return aggregator_state == nullptr
                   ? std::make_unique<GroupingFederatedSum<int32_t, int64_t>>()
                   : std::make_unique<GroupingFederatedSum<int32_t, int64_t>>(
                         MutableVectorData<int64_t>::CreateFromEncodedContent(
                             aggregator_state->vector_data()),
                         aggregator_state->num_inputs());
      } else if (input_spec.dtype() == DataType::DT_FLOAT &&
                 output_spec.dtype() == DataType::DT_DOUBLE) {
        return aggregator_state == nullptr
                   ? std::make_unique<GroupingFederatedSum<float, double>>()
                   : std::make_unique<GroupingFederatedSum<float, double>>(
                         MutableVectorData<double>::CreateFromEncodedContent(
                             aggregator_state->vector_data()),
                         aggregator_state->num_inputs());
      } else {
        return TFF_STATUS(INVALID_ARGUMENT)
               << "GroupingFederatedSumFactory: Input and output tensors have "
                  "mismatched dtypes: input tensor has dtype "
               << DataType_Name(input_spec.dtype())
               << " and output tensor has dtype "
               << DataType_Name(output_spec.dtype());
      }
    }

    StatusOr<std::unique_ptr<TensorAggregator>> aggregator;
    DTYPE_CASES(
        output_spec.dtype(), OutputT,
        aggregator = CreateGroupingFederatedSum<OutputT>(aggregator_state));
    return aggregator;
  }
};

REGISTER_AGGREGATOR_FACTORY(kGoogleSqlSumUri, GroupingFederatedSumFactory);

}  // namespace aggregation
}  // namespace tensorflow_federated
