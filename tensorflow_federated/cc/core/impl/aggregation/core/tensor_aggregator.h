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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_AGGREGATOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_AGGREGATOR_H_

#include <optional>
#include <string>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

using OutputTensorList = std::vector<Tensor>;

// TensorAggregator is a base class for implementing Aggregation intrinsics
// with Tensor being an input and output type for the aggregation.

// Abstract base class for aggregators that compute an aggregate of input
// Tensors into a final aggregate of Tensors using a multi-stage process in
// which items are first partially aggregated at an intermediate layer, then the
// partial aggregates are further combined, and finally projected into the
// result. This multi-stage process consists of the following:
// a) The aggregator is created with a zero value of an arbitrary intermediate
//    type U. Please note that the type U is never surfaced and considered an
//    implementation detail, so it doesn't need to be explicitly parameterized.
// b) The method Accumulate is used to accumulate Tensors into the U-typed
//    partial aggregate. It can optionally take in serialized metadata that
//    can be used to pass additional information about how the input should be
//    aggregated.
// c) The method Merge is used to merge the intermediate U-typed aggregates of
//    the two aggregator instances producing a merged U-typed aggregate.
// d) The method Report is used to project the top-level U-typed aggregate into
//    the final result of Tensors.
class TensorAggregator {
 public:
  virtual ~TensorAggregator() = default;

  // Accumulates an input into the intermediate aggregate. Serialized metadata
  // can be optionally provided to pass additional information about how the
  // input should be aggregated. The method may fail if the input isn't
  // compatible with the current TensorAggregator or if the TensorAggregator
  // instance has already been 'consumed'.
  Status Accumulate(
      InputTensorList tensors,
      std::optional<google::protobuf::Any> metadata = std::nullopt);

  // Merges intermediate aggregates from the other TensorAggregator instance
  // into the current TensorAggregator instance. Doing so 'consumes' the other
  // TensorAggregator instance. The method may fail if the two TensorAggregator
  // instances aren't compatible.
  virtual Status MergeWith(TensorAggregator&& other) = 0;

  // Returns true if the current TensorAggregator instance can produce a report,
  // for example if a sufficient number of inputs has been accumulated.
  virtual bool CanReport() const;

  // Produces the final report, 'consuming' the current TensorAggregator
  // instance. Once the current instance is consumed it can no longer perform
  // any operations. This method fails when CanReport method returns false.
  virtual StatusOr<OutputTensorList> Report() &&;

  // Returns the number of aggregated inputs.
  virtual int GetNumInputs() const = 0;

  // Serialize the internal state of the TensorAggregator as a string.
  virtual StatusOr<std::string> Serialize() && = 0;

 protected:
  // Construct TensorAggregator
  explicit TensorAggregator() = default;

  // The actual implementation of the tensor aggregation to be provided by
  // a derived class.
  virtual Status AggregateTensors(
      InputTensorList tensors,
      std::optional<google::protobuf::Any> metadata) = 0;

  // Checks if the current TensorAggregator is valid e.g. the resulting output
  // hasn't been consumed.
  virtual Status CheckValid() const = 0;

  // Consumes the output of this TensorAggregator.
  virtual OutputTensorList TakeOutputs() && = 0;

 private:
  // Extracts the aggregated tensor and makes the current aggregator "consumed".
  OutputTensorList TakeTensors() &&;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_AGGREGATOR_H_
