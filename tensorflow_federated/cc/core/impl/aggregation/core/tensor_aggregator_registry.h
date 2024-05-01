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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_AGGREGATOR_REGISTRY_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_AGGREGATOR_REGISTRY_H_

#include <memory>
#include <string>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"

namespace tensorflow_federated {
namespace aggregation {

// Registers a factory instance for the given intrinsic type.
void RegisterAggregatorFactory(const std::string& intrinsic_uri,
                               const TensorAggregatorFactory* factory);

// Looks up a factory instance for the given intrinsic type.
StatusOr<const TensorAggregatorFactory*> GetAggregatorFactory(
    const std::string& intrinsic_uri);

// Creates a TensorAggregator via the factory registered for the given
// intrinsic.
StatusOr<std::unique_ptr<TensorAggregator>> CreateTensorAggregator(
    const Intrinsic& intrinsic);

// Creates a TensorAggregator with the given internal state via the factory
// registered for the given intrinsic
StatusOr<std::unique_ptr<TensorAggregator>> DeserializeTensorAggregator(
    const Intrinsic& intrinsic, std::string serialized_state);

namespace internal {

template <typename FactoryType>
struct Registrar {
  explicit Registrar(const std::string& intrinsic_uri) {
    RegisterAggregatorFactory(intrinsic_uri, new FactoryType());
  }
};

}  // namespace internal

// This macro is used to register a factory type with the intrinsic uri.
#define REGISTER_AGGREGATOR_FACTORY(intrinsic_uri, FactoryType)              \
  static auto unused =                                                       \
      ::tensorflow_federated::aggregation::internal::Registrar<FactoryType>( \
          intrinsic_uri);

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_TENSOR_AGGREGATOR_REGISTRY_H_
