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

#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"

#include <cstddef>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"

namespace tensorflow_federated {
namespace aggregation {

Status OneDimBaseGroupingAggregator::MergeWith(TensorAggregator&& other) {
  // When merging OneDimBaseGroupingAggregators, the ordinals for the input
  // being merged need to be recomputed in order to correspond to the ordinal
  // mapping used by the aggregator into which the input is being merged.
  // Thus, the inner state of the aggregator being merged is not useful by
  // itself. We instead use the MergeTensors method below to merge
  // OneDimGroupingAggregators, which provides the new ordinals to use while
  // merging. The outer GroupByAggregator is responsible for providing these
  // new ordinals.
  return TFF_STATUS(UNIMPLEMENTED)
         << "OneDimGroupingAggregator::MergeWith is not supported. Use "
            "MergeTensors instead.";
}

Status OneDimBaseGroupingAggregator::ValidateTensorInputs(
    const InputTensorList& tensors) {
  TFF_CHECK(tensors.size() == 2)
      << "OneDimGroupingAggregator should operate on 2 input tensors";

  const Tensor* ordinals = tensors[0];
  if (ordinals->dtype() != DT_INT64) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "OneDimGroupingAggregator::AggregateTensors: dtype mismatch "
              "for tensor 0. Expected DT_INT64.";
  }
  const Tensor* tensor = tensors[1];
  if (ordinals->shape() != tensor->shape()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "OneDimGroupingAggregator::AggregateTensors: tensor shape "
              "mismatch. Shape of both tensors must be the same.";
  }
  size_t num_dimensions = tensor->shape().dim_sizes().size();
  if (num_dimensions > (size_t)1) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "OneDimGroupingAggregator::AggregateTensors: Only 1 "
              "dimensional tensors supported. Input tensor has "
           << num_dimensions << " dimensions.";
  }
  if (!ordinals->is_dense() || !tensor->is_dense()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "OneDimGroupingAggregator::AggregateTensors: Only dense "
              "tensors are supported.";
  }
  return TFF_STATUS(OK);
}

}  // namespace aggregation
}  // namespace tensorflow_federated
