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
#include "absl/random/random.h"
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
  explicit ExecutorService(const ExecutorFactory& executor_factory)
      : executor_resolver_(executor_factory) {}

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
  // A cheaply-copyable struct used to track executors and pass handles to them
  // between the executor resolver and the service.
  struct ExecutorEntry {
    std::shared_ptr<Executor> executor;
    // The number of current external connections to this executor.
    // This is `count(times returned by GetExecutor) -
    // count(times passed to DisposeExecutor)`.
    uint32_t remote_refcount;
    // The unique identifier of this executor returned to clients.
    const std::string executor_id;
  };

  // Writes the `std::shared_ptr<Executor>` corresponding to `executor_id` into
  // `executor_out`. `method name` is the name of the caller that requested
  // access to this executor, and is used for debug purposes only.
  grpc::Status RequireExecutor(std::string_view method_name,
                               const v0::ExecutorId& executor_id,
                               std::shared_ptr<Executor>& executor_out);

  // Function which contains switching logic, determining e.g. whether to
  // destroy an underlying executor.
  grpc::Status HandleNotOK(const absl::Status& status,
                           const v0::ExecutorId& executor_id);

  using ExecutorId = std::string;

  struct ExecutorRequirements {
    CardinalityMap cardinalities;
  };

  // Internal class providing logic for managing Executors in the service.
  // This class has three main responsibilities:
  // * Return an executor when the service requires one, via a GetExecutor
  // request, managing the relationship between Executor requirements (IE, the
  // data carrying semantic meaning by which clients wish to request an
  // executor), and the identifiers used to refer to a consistent executor
  // instance (in order to use the stateful Executor interface).
  // * Resolve the identifiers handed back to clients to concrete Executor
  // instances, on which calls to the Executor interface may be made by the
  // service.
  // * Manage validity of these executors. In particular, this requires managing
  // refcounts to these executors, and providing the service a way to invalidate
  // all outstanding references to an executor which the service determines to
  // be in an invalid state.
  class ExecutorResolver {
   public:
    explicit ExecutorResolver(ExecutorFactory factory) : ex_factory_(factory) {}

    // Indicates to the ExecutorResolver that the executor associated to the
    // ExecutorId argument is no longer valid or no longer necessary, and should
    // not be returned to future callers. This function may be called
    // repeatedly, and will not err; it makes the guarantee that after
    // invocation, this object no longer holds a handle to an executor entry
    // with this ID. Future calls to ExecutorForId with this ID will return
    // FAILED_PRECONDITION.
    void DestroyExecutor(const ExecutorId&);

    // Indicates that a client has released a handle to the executor identified
    // by the argument. May trigger a DisposeExecutor, if the ExecutorResolver
    // can determine that there are no clients with outstanding handles to the
    // executor associated to this ID.
    absl::Status DisposeExecutor(const ExecutorId&);

    // Returns an executor for the given ID, or an absl Status if no executor
    // can be found. Returns FailedPrecondition if the executor does not exist
    // (since existence of the executor is a precondition for returning it), and
    // InternalError if an implementation error has caused the maps held
    // internally to be out of sync. This error indicates a bug in the
    // implementation.
    absl::StatusOr<ExecutorEntry> ExecutorForId(const ExecutorId&);
    // Returns an executor ID with the specified requirements. May construct a
    // new executor; only returns a no-OK status if this construction fails or
    // another internal error occurs.
    absl::StatusOr<ExecutorId> ExecutorIDForRequirements(
        const ExecutorRequirements&);

   private:
    absl::Mutex executors_mutex_;
    void DestroyExecutorImpl(const ExecutorId&)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(executors_mutex_);
    ExecutorFactory ex_factory_;

    // We key this map by the cardinalities of the associated executors.
    absl::flat_hash_map<std::string, ExecutorEntry> executors_
        ABSL_GUARDED_BY(executors_mutex_);
    // This map is keyed by the executor ids returned to clients, and used to
    // resolve executor IDs to concrete executor instances via the executors_
    // map. Every entry in this map must correspond to a cardinalities key for
    // an executor in the executors_ map.
    absl::flat_hash_map<std::string, std::string> keys_to_cardinalities_
        ABSL_GUARDED_BY(executors_mutex_);

    // Provide a unique identifier to each service, as well as each executor
    // generated within a given service instance. These are used to disambiguate
    // calls to Dispose which are simply 'late' (e.g., occur after a service has
    // already been rebooted) from those which are true errors in operation
    // (explicitly deleting a single value twice, for example).
    static constexpr int kServiceIndexRange = 100000;
    absl::BitGen gen_;
    int service_id_ = absl::Uniform(gen_, 0, kServiceIndexRange);
    int executor_index_ ABSL_GUARDED_BY(executors_mutex_) = 0;
  };

  ExecutorResolver executor_resolver_;
};
}  // namespace tensorflow_federated
#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EXECUTOR_SERVICE_H_
