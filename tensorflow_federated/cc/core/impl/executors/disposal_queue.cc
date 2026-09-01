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

#include "tensorflow_federated/cc/core/impl/executors/disposal_queue.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "include/grpcpp/client_context.h"
#include "include/grpcpp/support/status.h"
#include "tensorflow_federated/cc/core/impl/executors/status_conversion.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

DisposalQueue::DisposalQueue(
    v0::ExecutorId executor_pb,
    std::shared_ptr<v0::ExecutorGroup::StubInterface> stub,
    size_t max_batch_size, absl::Duration flush_delay)
    : executor_pb_(std::move(executor_pb)),
      stub_(std::move(stub)),
      max_batch_size_(max_batch_size),
      flush_delay_(flush_delay) {}

DisposalQueue::~DisposalQueue() { Close(); }

void DisposalQueue::Dispose(v0::ValueRef value_ref) {
  {
    absl::MutexLock lock(mutex_);
    if (closed_) {
      return;
    }
  }

  // When max_batch_size_ == 0, buffering is disabled: dispatch immediately
  // without queueing into pending_ or scheduling timers.
  if (max_batch_size_ == 0) {
    std::vector<v0::ValueRef> batch;
    batch.push_back(std::move(value_ref));
    DispatchBatch(std::move(batch));
    return;
  }

  std::vector<v0::ValueRef> to_flush;
  bool schedule_timer = false;
  {
    absl::MutexLock lock(mutex_);
    if (closed_) {
      return;
    }
    pending_.push_back(std::move(value_ref));
    if (pending_.size() >= max_batch_size_) {
      // Double-buffering: swap the buffer out under the lock so pending_ is
      // cleared immediately, and the caller can release the mutex before
      // triggering network dispatch or thread scheduling.
      to_flush.swap(pending_);
    } else if (!timer_active_) {
      timer_active_ = true;
      schedule_timer = true;
    }
  }

  if (!to_flush.empty()) {
    DispatchBatch(std::move(to_flush));
  } else if (schedule_timer) {
    ScheduleTimerFlush();
  }
}

void DisposalQueue::Flush() {
  std::vector<v0::ValueRef> to_flush;
  {
    absl::MutexLock lock(mutex_);
    if (closed_ || pending_.empty()) {
      return;
    }
    // Swap out the pending references under the lock so subsequent callers are
    // not blocked while DispatchBatch constructs and sends the gRPC request.
    to_flush.swap(pending_);
  }
  if (!to_flush.empty()) {
    DispatchBatch(std::move(to_flush));
  }
}

void DisposalQueue::Close() {
  absl::MutexLock lock(mutex_);
  closed_ = true;
  pending_.clear();
}

void DisposalQueue::ScheduleTimerFlush() {
  ThreadRun([self = shared_from_this(), delay = flush_delay_]() {
    absl::SleepFor(delay);
    std::vector<v0::ValueRef> to_flush;
    {
      absl::MutexLock lock(self->mutex_);
      self->timer_active_ = false;
      if (self->closed_ || self->pending_.empty()) {
        return;
      }
      // Extract the pending batch via an O(1) vector swap under the lock,
      // releasing the mutex before issuing the remote Dispose RPC.
      to_flush.swap(self->pending_);
    }
    if (!to_flush.empty()) {
      self->DispatchBatch(std::move(to_flush));
    }
  });
}

void DisposalQueue::DispatchBatch(std::vector<v0::ValueRef> batch) {
  ThreadRun(
      [batch = std::move(batch), executor_pb = executor_pb_, stub = stub_]() {
        v0::DisposeRequest request;
        v0::DisposeResponse response;
        grpc::ClientContext context;
        *request.mutable_executor() = std::move(executor_pb);
        request.mutable_value_ref()->Reserve(batch.size());
        for (const v0::ValueRef& ref : batch) {
          *request.add_value_ref() = ref;
        }
        grpc::Status status = stub->Dispose(&context, request, &response);
        if (!status.ok()) {
          LOG(ERROR) << "Error disposing of " << batch.size()
                     << " values for executor [" << request.executor().id()
                     << "]: " << grpc_to_absl(status);
        }
      });
}

}  // namespace tensorflow_federated
