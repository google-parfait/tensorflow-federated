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

#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"

namespace tensorflow_federated {
namespace aggregation {

namespace internal {

class Registry final {
 public:
  void RegisterAggregatorFactory(const std::string& intrinsic_uri,
                                 const TensorAggregatorFactory* factory) {
    TFF_CHECK(factory != nullptr);

    absl::MutexLock lock(&mutex_);
    TFF_CHECK(map_.find(intrinsic_uri) == map_.end())
        << "A factory for intrinsic_uri '" << intrinsic_uri
        << "' is already registered.";
    map_[intrinsic_uri] = factory;
    TFF_LOG(INFO) << "TensorAggregatorFactory for intrinsic_uri '"
                  << intrinsic_uri << "' is registered.";
  }

  StatusOr<const TensorAggregatorFactory*> GetAggregatorFactory(
      const std::string& intrinsic_uri) {
    absl::MutexLock lock(&mutex_);
    auto it = map_.find(intrinsic_uri);
    if (it == map_.end()) {
      return TFF_STATUS(NOT_FOUND)
             << "Unknown factory for intrinsic_uri '" << intrinsic_uri << "'.";
    }
    return it->second;
  }

 private:
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, const TensorAggregatorFactory*> map_
      ABSL_GUARDED_BY(mutex_);
};

Registry* GetRegistry() {
  static Registry* global_registry = new Registry();
  return global_registry;
}

}  // namespace internal

// Registers a factory instance for the given intrinsic type.
void RegisterAggregatorFactory(const std::string& intrinsic_uri,
                               const TensorAggregatorFactory* factory) {
  internal::GetRegistry()->RegisterAggregatorFactory(intrinsic_uri, factory);
}

// Looks up a factory instance for the given intrinsic type.
StatusOr<const TensorAggregatorFactory*> GetAggregatorFactory(
    const std::string& intrinsic_uri) {
  return internal::GetRegistry()->GetAggregatorFactory(intrinsic_uri);
}

StatusOr<std::unique_ptr<TensorAggregator>> CreateTensorAggregator(
    const Intrinsic& intrinsic) {
  return (*GetAggregatorFactory(intrinsic.uri))->Create(intrinsic);
}

StatusOr<std::unique_ptr<TensorAggregator>> DeserializeTensorAggregator(
    const Intrinsic& intrinsic, std::string serialized_state) {
  return (*GetAggregatorFactory(intrinsic.uri))
      ->Deserialize(intrinsic, serialized_state);
}

}  // namespace aggregation
}  // namespace tensorflow_federated
