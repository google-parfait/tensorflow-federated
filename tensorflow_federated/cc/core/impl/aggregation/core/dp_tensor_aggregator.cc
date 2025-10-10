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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_tensor_aggregator.h"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

// Produces a comma-separated string consisting of the shape's dimensions.
std::string TensorShapeToString(const TensorShape& shape) {
  std::stringstream output;
  std::copy(shape.dim_sizes().begin(), shape.dim_sizes().end(),
            std::ostream_iterator<int>(output, ","));
  std::string output_str = output.str();
  // Remove the trailing comma if there is one.
  return output_str.empty() ? output_str
                            : output_str.erase(output_str.size() - 1);
}

Status DPTensorAggregator::ValidateInputs(const InputTensorList& input) const {
  if (input.size() != input_specs_.size()) {
    return TFF_STATUS(INVALID_ARGUMENT)
           << "DPTensorAggregator::ValidateInputs: Expected exactly "
           << input_specs_.size() << " tensors, but got " << input.size();
  }

  for (int i = 0; i < input.size(); ++i) {
    const Tensor* input_tensor = input[i];
    const TensorSpec& input_spec = input_specs_[i];
    // Data type of input must match the spec.
    if (input_tensor->dtype() != input_spec.dtype()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPTensorAggregator::ValidateInputs: Expected an input of "
                "type"
             << " " << input_spec.dtype() << ", but got "
             << input_tensor->dtype() << " for input[" << i << "]";
    }
    // If the spec's shape is not {-1}, then the input shape must match.
    // (-1 indicates unknown dimensionality)
    if (input_spec.shape() != TensorShape{-1} &&
        input_tensor->shape() != input_spec.shape()) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPTensorAggregator::ValidateInputs: Expected input with "
                "shape"
             << " {" << TensorShapeToString(input_spec.shape()) << "},"
             << " but got"
             << " {" << TensorShapeToString(input_tensor->shape()) << "}"
             << "for input[" << i << "]";
    }
  }

  return TFF_STATUS(OK);
}

}  // namespace aggregation
}  // namespace tensorflow_federated
