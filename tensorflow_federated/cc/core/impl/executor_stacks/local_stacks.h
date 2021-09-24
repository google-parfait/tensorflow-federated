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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTOR_STACKS_LOCAL_STACKS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTOR_STACKS_LOCAL_STACKS_H_

#include <functional>
#include <memory>

#include "absl/status/statusor.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"

namespace tensorflow_federated {

// Constructs an executor stack which executes computations entirely on a local
// machine. This stack can be configured with a leaf executor which will be used
// to execute non-federated computations embedded in TFF's computation protos,
// e.g. TensorFlow graphs or Jax computations.
//
// Returns an absl::Status if construction fails, and a shared_ptr to an
// instance of Executor if construction succeeds.
absl::StatusOr<std::shared_ptr<Executor>> CreateLocalExecutor(
    const CardinalityMap& cardinalities,
    std::function<absl::StatusOr<std::shared_ptr<Executor>>()>
        leaf_executor_fn = CreateTensorFlowExecutor);
}  // namespace tensorflow_federated
#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTOR_STACKS_LOCAL_STACKS_H_
