
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
#include "tensorflow_federated/cc/core/impl/executor_stacks/local_stacks.h"
#include <functional>
#include "absl/status/statusor.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/sequence_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"

namespace tensorflow_federated {

absl::StatusOr<std::shared_ptr<Executor>> CreateLocalExecutor(
    const CardinalityMap& cardinalities,
    std::function<absl::StatusOr<std::shared_ptr<Executor>>(int32_t)>
        leaf_executor_fn,
    std::function<absl::StatusOr<std::shared_ptr<Executor>>(int32_t)>
        client_leaf_executor_fn) {
  std::shared_ptr<Executor> server = CreateReferenceResolvingExecutor(
      CreateSequenceExecutor(TFF_TRY(leaf_executor_fn(-1))));
  std::shared_ptr<Executor> client = server;
  if (client_leaf_executor_fn != nullptr) {
    client = CreateReferenceResolvingExecutor(
        CreateSequenceExecutor(TFF_TRY(client_leaf_executor_fn(-1))));
  }
  return CreateReferenceResolvingExecutor(TFF_TRY(CreateFederatingExecutor(
      /*server_child=*/server, /*client_child=*/client, cardinalities)));
}
}  // namespace tensorflow_federated
