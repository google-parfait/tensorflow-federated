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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EXECUTOR_SERVICE_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EXECUTOR_SERVICE_H_

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "grpc/include/grpcpp/grpcpp.h"
#include "grpc/include/grpcpp/server_context.h"
#include "grpc/include/grpcpp/support/status.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/proto/v0/computation.proto.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"
#include "tensorflow_federated/proto/v0/executor.proto.h"

namespace tensorflow_federated {

using ExecutorFactory = std::function<absl::StatusOr<std::shared_ptr<Executor>>(
    const CardinalityMap&)>;

// Service hosting a TFF executor stack via gRPC as defined in executor.proto.
// Each of the `Create...` methods below should be conceptualized as returning
// immediately; this depends crucially on implementations of
// `tensorflow_federated::Executor` respecting the same performance contract.
//
// `Compute` is the single method below which can block at the application
// layer. It serves as a signal from the client that values need to be
// materialized on the other side of the gRPC channel.
//
// The methods `SetCardinalities` and `ClearExecutor` can be thought of as
// executor-stack state management. `SetCardinalities` must be called before the
// service can serve `Create` or `Compute` requests, as the results of some
// computations can depend on the cardinalities hosted by this executor stack.
// `ClearExecutor` can be used to reset the stack, IE, reset the internal
// executor to null.
//
// Finally, `Dispose` serves as an explicit resource-management request;
// `Dispose` serves to inform the service that it can free any resources
// associated to the values which the `Dispose` request specified.
class ExecutorService : public v0::Executor::Service {
  using RemoteValueId = std::string;

 public:
  // Constructor takes a function which will return a
  // tensorflow_federated::Executor when invoked with a mapping from TFF
  // placements to integers. After the service is constructed, it must be
  // configured with a `SetCardinalities` request (which instantiates an
  // underlying concrete tensorflow_federated::Executor) before it can start
  // executing other arbitrary requests.
  ExecutorService(ExecutorFactory executor_factory) {
    executor_factory_ = executor_factory;
  }
  ExecutorService(ExecutorService&& other) {
    executor_factory_ = other.executor_factory_;
  }
  ~ExecutorService() {}

  // Embed a value in the underlying executor stack.
  grpc::Status CreateValue(grpc::ServerContext* context,
                           const v0::CreateValueRequest* request,
                           v0::CreateValueResponse* response) override;

  // Invoke an embedded function on an embedded argument.
  grpc::Status CreateCall(grpc::ServerContext* context,
                          const v0::CreateCallRequest* request,
                          v0::CreateCallResponse* response) override;

  // Package several embedded values together as a single value.
  grpc::Status CreateStruct(grpc::ServerContext* context,
                            const v0::CreateStructRequest* request,
                            v0::CreateStructResponse* response) override;

  // Select a single value from an embedded value of TFF type Struct.
  grpc::Status CreateSelection(grpc::ServerContext* context,
                               const v0::CreateSelectionRequest* request,
                               v0::CreateSelectionResponse* response) override;

  // Materialize a value on the client. Blocking. The value requested to be
  // materialized must be non-functional.
  grpc::Status Compute(grpc::ServerContext* context,
                       const v0::ComputeRequest* request,
                       v0::ComputeResponse* response) override;

  // Configure the underlying executor stack to host a particular set of
  // cardinalities. This method invalidates any previously returned ValueRefs.
  grpc::Status SetCardinalities(
      grpc::ServerContext* context, const v0::SetCardinalitiesRequest* request,
      v0::SetCardinalitiesResponse* response) override;

  // Reset the hosted executor to null. After invoking this method, the service
  // will be unable to execute any Create or Compute requests until
  // `SetCardinalities` is called once again the configure the underlying
  // executor.
  grpc::Status ClearExecutor(grpc::ServerContext* context,
                             const v0::ClearExecutorRequest* request,
                             v0::ClearExecutorResponse* response) override;

  // Free the resources associated to the embedded values specified.
  grpc::Status Dispose(grpc::ServerContext* context,
                       const v0::DisposeRequest* request,
                       v0::DisposeResponse* response) override;

 private:
  ExecutorFactory executor_factory_;
  absl::Mutex executor_mutex_;
  // We store executor and generation together as they should be read together.
  // We start generation at -1 because the service is not ready to receive
  // requests until `SetCardinalities` is called; the first live executor will
  // have generation 0.
  std::pair<std::shared_ptr<Executor>, int> executor_and_generation_
      ABSL_GUARDED_BY(executor_mutex_) = std::make_pair(nullptr, -1);
  absl::StatusOr<std::pair<std::shared_ptr<Executor>, int>> RequireExecutor_(
      std::string method_name);
  absl::Status EnsureGeneration_(int, int);
  // We StatusOr here so that we can check whether the incoming RemoteValueIds
  // correspond to the currently live executor. RemoteValueId is a string of the
  // format "a-b", where a is a uint64 and b is another int. a represents the
  // ValueId of the underlying value in executor_, and b represents the
  // generation of executor_.
  absl::StatusOr<ValueId> ResolveRemoteValue_(const v0::ValueRef&, int);
  RemoteValueId CreateRemoteValue_(ValueId, int);
};
}  // namespace tensorflow_federated
#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EXECUTOR_SERVICE_H_
