/* Copyright 2026, The TensorFlow Federated Authors.

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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_REMOTE_EXECUTOR_STACK_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_REMOTE_EXECUTOR_STACK_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "include/grpcpp/grpcpp.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/composing_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace tensorflow_federated {

using ExecutorFn = std::function<absl::StatusOr<std::shared_ptr<Executor>>()>;
using ComposingChildFn = std::function<absl::StatusOr<ComposingChild>(
    std::shared_ptr<grpc::ChannelInterface>, const CardinalityMap&)>;
using ComposingExecutorFn = std::function<std::shared_ptr<Executor>(
    std::shared_ptr<Executor>, std::vector<ComposingChild>)>;

// Queries the state of `channels` and returns those that are READY or IDLE,
// waiting up to `wait_connected_duration_millis` milliseconds for each channel.
std::vector<std::shared_ptr<grpc::ChannelInterface>> FilterToLiveChannels(
    const std::vector<std::shared_ptr<grpc::ChannelInterface>>& channels,
    int wait_connected_duration_millis = 1000);

// Creates an executor stack that proxies for a group of remote workers using
// caller-supplied leaf and composing factories.
//
// Upon object construction, the channels that represent connections to these
// workers are queried for their state, and only workers whose channels are
// READY or IDLE are addressed with work for the lifetime of this object.
absl::StatusOr<std::shared_ptr<Executor>> CreateRemoteExecutorStack(
    const std::vector<std::shared_ptr<grpc::ChannelInterface>>& channels,
    const CardinalityMap& cardinalities, ExecutorFn leaf_executor_fn,
    ComposingChildFn composing_child_fn,
    ComposingExecutorFn composing_executor_fn = CreateComposingExecutor);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_REMOTE_EXECUTOR_STACK_H_
