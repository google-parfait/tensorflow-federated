/* Copyright 2021, The TensorFlow Federated Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_COMPOSING_EXECUTOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_COMPOSING_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"

namespace tensorflow_federated {

// An executor to be used as an intermediate aggregator for some subset of a
// `ComposingExecutor`'s clients.
class ComposingChild {
 public:
  static absl::StatusOr<ComposingChild> Make(
      std::shared_ptr<Executor> executor, const CardinalityMap& cardinalities) {
    uint32_t num_clients = TFF_TRY(NumClientsFromCardinalities(cardinalities));
    return ComposingChild(std::move(executor), num_clients);
  }

  const std::shared_ptr<Executor>& Executor() const { return executor_; }

  uint32_t NumClients() const { return num_clients_; }

 private:
  std::shared_ptr<::tensorflow_federated::Executor> executor_;
  uint32_t num_clients_;

  ComposingChild(std::shared_ptr<::tensorflow_federated::Executor> executor,
                 uint32_t num_clients)
      : executor_(std::move(executor)), num_clients_(num_clients) {}
};

// Returns an executor that splits handling of federated values and intrinsics
// across multiple child executors.
//
// The `server` executor will be used for executing unplaced and server-placed
// computations and does not need to understand federated values or intrinsics.
//
// The `children` executors will be used for executing shards of federated
// computations and must be able to resolve federated values and intrinsics.
std::shared_ptr<Executor> CreateComposingExecutor(
    std::shared_ptr<Executor> server, std::vector<ComposingChild> children);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_COMPOSING_EXECUTOR_H_
