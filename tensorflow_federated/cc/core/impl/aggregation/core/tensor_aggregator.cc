/*
 * Copyright 2022 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"

namespace tensorflow_federated {
namespace aggregation {

Status TensorAggregator::Accumulate(InputTensorList tensors) {
  TFF_RETURN_IF_ERROR(CheckValid());

  // Delegate aggregation to the derived class.
  return AggregateTensors(std::move(tensors));
}

bool TensorAggregator::CanReport() const { return CheckValid().ok(); }

StatusOr<OutputTensorList> TensorAggregator::Report() && {
  TFF_RETURN_IF_ERROR(CheckValid());
  if (!CanReport()) {
    return TFF_STATUS(FAILED_PRECONDITION)
           << "TensorAggregator::Report: the report goal isn't met";
  }
  return std::move(*this).TakeOutputs();
}

StatusOr<std::vector<std::string>> TensorAggregator::Partition(
    int num_partitions) && {
  return TFF_STATUS(UNIMPLEMENTED)
         << "TensorAggregator::Partition is not supported";
}

}  // namespace aggregation
}  // namespace tensorflow_federated
