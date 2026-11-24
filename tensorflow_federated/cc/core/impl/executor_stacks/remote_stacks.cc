
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

#include "tensorflow_federated/cc/core/impl/executor_stacks/remote_stacks.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "include/grpcpp/grpcpp.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/composing_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/remote_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/remote_executor_stack.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/streaming_remote_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"

namespace tensorflow_federated {

absl::StatusOr<std::shared_ptr<Executor>> CreateRemoteExecutorStack(
    const std::vector<std::shared_ptr<grpc::ChannelInterface>>& channels,
    const CardinalityMap& cardinalities,
    int32_t max_concurrent_computation_calls) {
  auto rre_tf_leaf_executor = [max_concurrent_computation_calls]() {
    return CreateReferenceResolvingExecutor(
        CreateTensorFlowExecutor(max_concurrent_computation_calls));
  };
  ComposingChildFn composing_child_factory =
      [](std::shared_ptr<grpc::ChannelInterface> channel,
         const CardinalityMap& cardinalities)
      -> absl::StatusOr<ComposingChild> {
    return TFF_TRY(ComposingChild::Make(
        CreateRemoteExecutor(std::move(channel), cardinalities),
        cardinalities));
  };

  return CreateRemoteExecutorStack(
      channels, cardinalities, rre_tf_leaf_executor, composing_child_factory);
}

absl::StatusOr<std::shared_ptr<Executor>> CreateStreamingRemoteExecutorStack(
    const std::vector<std::shared_ptr<grpc::ChannelInterface>>& channels,
    const CardinalityMap& cardinalities) {
  auto rre_tf_leaf_executor = []() {
    return CreateReferenceResolvingExecutor(CreateTensorFlowExecutor());
  };
  ComposingChildFn composing_child_factory =
      [](std::shared_ptr<grpc::ChannelInterface> channel,
         const CardinalityMap& cardinalities)
      -> absl::StatusOr<ComposingChild> {
    return TFF_TRY(ComposingChild::Make(
        CreateStreamingRemoteExecutor(std::move(channel), cardinalities),
        cardinalities));
  };

  return CreateRemoteExecutorStack(
      channels, cardinalities, rre_tf_leaf_executor, composing_child_factory);
}

}  // namespace tensorflow_federated
