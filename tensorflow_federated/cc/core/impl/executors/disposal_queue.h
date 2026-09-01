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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DISPOSAL_QUEUE_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DISPOSAL_QUEUE_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

// Thread-safe queue for dispatching `ValueRef` disposals to a remote
// executor service.
//
// When `max_batch_size > 0`, `DisposalQueue` buffers `ValueRef`s and flushes
// them in batches when the size threshold is reached or when a timer delay
// expires, reducing RPC churn and thread contention.
//
// When `max_batch_size == 0`, buffering is disabled: each `Dispose()` call
// immediately dispatches an individual asynchronous `Dispose` RPC without
// queueing or timer delay.
class DisposalQueue : public std::enable_shared_from_this<DisposalQueue> {
 public:
  static constexpr size_t kDefaultMaxBatchSize = 256;
  static constexpr absl::Duration kDefaultFlushDelay = absl::Milliseconds(10);

  DisposalQueue(v0::ExecutorId executor_pb,
                std::shared_ptr<v0::ExecutorGroup::StubInterface> stub,
                size_t max_batch_size = kDefaultMaxBatchSize,
                absl::Duration flush_delay = kDefaultFlushDelay);

  ~DisposalQueue();

  // Pushes a `ValueRef` to the queue for batch disposal (or immediate dispatch
  // if `max_batch_size == 0`).
  // This operation is non-blocking and thread-safe.
  void Dispose(v0::ValueRef value_ref);

  // Synchronously extracts and asynchronously dispatches all currently pending
  // value references.
  void Flush();

  // Closes the queue, clearing any pending references and preventing any
  // future requests from being dispatched. Intended for use during executor
  // teardown (e.g. `DisposeExecutor`).
  void Close();

 private:
  void ScheduleTimerFlush();
  void DispatchBatch(std::vector<v0::ValueRef> batch);

  const v0::ExecutorId executor_pb_;
  const std::shared_ptr<v0::ExecutorGroup::StubInterface> stub_;
  const size_t max_batch_size_;
  const absl::Duration flush_delay_;

  absl::Mutex mutex_;
  bool closed_ ABSL_GUARDED_BY(mutex_) = false;
  bool timer_active_ ABSL_GUARDED_BY(mutex_) = false;
  std::vector<v0::ValueRef> pending_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DISPOSAL_QUEUE_H_
