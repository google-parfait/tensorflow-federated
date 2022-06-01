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

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/grpcpp.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_conversion.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

using ExecutorFactory = std::function<absl::StatusOr<std::shared_ptr<Executor>>(
    const CardinalityMap&)>;

// Service hosting TFF executor stacks via gRPC as defined in executor.proto.
//
// The `GetExecutor` method provides access to an `ExecutorId` which is used to
// query the executor using the `Create...` and `Materialize` methods. Once
// finished, `DisposeExecutor` should be used to release the resources
// associated with the underlying executor.
//
// Each of the `Create...` methods below should be conceptualized as returning
// immediately; this depends crucially on implementations of
// `tensorflow_federated::Executor` respecting the same performance contract.
//
// `Compute` is the single method below which can block at the application
// layer. It serves as a signal from the client that values need to be
// materialized on the other side of the gRPC channel.
//
// Finally, `Dispose` serves as an explicit resource-management request;
// `Dispose` tells the service that it can free any resources
// associated with the specified `ValueId`s.
class ExecutorService : public v0::ExecutorGroup::Service {
  using RemoteValueId = std::string;

 public:
  // Constructor takes a function which will return a
  // tensorflow_federated::Executor when invoked with a mapping from TFF
  // placements to integers. After the service is constructed, it must be
  // configured with a `GetExecutor` request (which instantiates an
  // underlying concrete tensorflow_federated::Executor) before it can start
  // executing other requests.
  explicit ExecutorService(const ExecutorFactory& executor_factory) {
    executor_factory_ = executor_factory;
  }
  ExecutorService(ExecutorService&& other) {
    executor_factory_ = other.executor_factory_;
  }
  ~ExecutorService() override {}

  // Configure the underlying executor stack to host a particular executor
  // configuration and return an identifier used to access the resulting
  // executor.
  grpc::Status GetExecutor(grpc::ServerContext* context,
                           const v0::GetExecutorRequest* request,
                           v0::GetExecutorResponse* response) override;

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

  // Free the resources associated to the embedded values specified.
  grpc::Status Dispose(grpc::ServerContext* context,
                       const v0::DisposeRequest* request,
                       v0::DisposeResponse* response) override;

  // Free the resources associated with a particular executor.
  grpc::Status DisposeExecutor(grpc::ServerContext* context,
                               const v0::DisposeExecutorRequest* request,
                               v0::DisposeExecutorResponse* response) override;

 private:
  struct ExecutorEntry {
    std::shared_ptr<Executor> executor;
    // The number of current external connections to this executor.
    // This is `count(times returned by GetExecutor) -
    // count(times passed to DisposeExecutor)`.
    uint32_t remote_refcount;
  };

  // Writes the `std::shared_ptr<Executor>` corresponding to `executor_id` into
  // `executor_out`. `method name` is the name of the caller that requested
  // access to this executor, and is used for debug purposes only.
  grpc::Status RequireExecutor(absl::string_view method_name,
                               const v0::ExecutorId& executor_id,
                               std::shared_ptr<Executor>& executor_out);

  // Destroys executor associated to `executor_id`, resetting refcount to 0.
  // Invoked by error-handling logic in internal methods, e.g. in the case that
  // the associated executor indicates it needs to be reconfigured. This
  // function should only be called when all potentially outstanding references
  // to this executor are necessarily invalid, e.g. in the case that the
  // executor must be configured with cardinalities before it can handle any new
  // requests.
  void DestroyExecutor(const v0::ExecutorId& executor_id);

  // Function which contains switching logic, determining e.g. whether to
  // destroy an underlying executor.
  grpc::Status HandleNotOK(const absl::Status& status,
                           const v0::ExecutorId& executor_id);

  ExecutorFactory executor_factory_;
  absl::Mutex executors_mutex_;
  absl::flat_hash_map<std::string, ExecutorEntry> executors_
      ABSL_GUARDED_BY(executors_mutex_);
};
}  // namespace tensorflow_federated
#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EXECUTOR_SERVICE_H_
