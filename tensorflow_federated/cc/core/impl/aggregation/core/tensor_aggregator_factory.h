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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_AGGREGATOR_FACTORY_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_AGGREGATOR_FACTORY_H_

#include <memory>
#include <string>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"

namespace tensorflow_federated {
namespace aggregation {

// This class is the interface for the abstract factory that creates an instance
// of a TensorAggregator derived class.
class TensorAggregatorFactory {
 public:
  virtual ~TensorAggregatorFactory() = default;

  // Creates an instance of a specific aggregator for the specified type of the
  // aggregation intrinsic and the tensor specifications.
  // The lifetime of the provided Intrinsic must outlast that of the returned
  // TensorAggregator as it is valid for the TensorAggregator implementation to
  // hold pointers referring to the Intrinsic.
  virtual StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const = 0;

  // Creates an instance of a specific aggregator for the specified type of the
  // aggregation intrinsic and serialized aggregator state.
  // The lifetime of the provided Intrinsic must outlast that of the returned
  // TensorAggregator as it is valid for the TensorAggregator implementation to
  // hold pointers referring to the Intrinsic.
  virtual StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const = 0;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_AGGREGATOR_FACTORY_H_
