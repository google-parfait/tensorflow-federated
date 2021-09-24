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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTOR_STACKS_REMOTE_STACKS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTOR_STACKS_REMOTE_STACKS_H_

#include <functional>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "grpcpp/grpcpp.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/composing_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace tensorflow_federated {

using ExecutorFn = std::function<absl::StatusOr<std::shared_ptr<Executor>>()>;
using ComposingChildFn = std::function<absl::StatusOr<ComposingChild>(
    std::shared_ptr<grpc::ChannelInterface>, const CardinalityMap&)>;

// Creates an executor stack which proxies for a group of remote workers.
//
// Upon object construction, the channels which represent connections to these
// workers will be queried for their state, and only those workers whose
// channels are healthy will be addressed with work for the lifetime of this
// object. Here we define "healthy" as having a channel state which is either
// READY or IDLE; see
// https://grpc.github.io/grpc/core/md_doc_connectivity-semantics-and-api.html
// for the semantic meaning of these states.
//
// It is important to not the consequence of this behavior: if one of
// the workers which was initially up on construction of the remote executor
// stack goes down, the subsequent calls to the returned executor may fail.
// Relatedly, if any initially unavailable workers come up, they will remain
// unused until this function is reinvoked.
//
// This method may block on an RPC call to each channel in order to verify that
// it is healthy.
absl::StatusOr<std::shared_ptr<Executor>> CreateRemoteExecutorStack(
    absl::Span<const std::shared_ptr<grpc::ChannelInterface>> channels,
    const CardinalityMap& cardinalities);

// Creates an executor stack which proxies for a group of remote workers.
//
// This function is an overload for the above, intended to be used for testing.
// See the documentation above for details.
absl::StatusOr<std::shared_ptr<Executor>> CreateRemoteExecutorStack(
    absl::Span<const std::shared_ptr<grpc::ChannelInterface>> channels,
    const CardinalityMap& cardinalities, ExecutorFn leaf_executor_fn,
    ComposingChildFn composing_child_fn);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTOR_STACKS_REMOTE_STACKS_H_
