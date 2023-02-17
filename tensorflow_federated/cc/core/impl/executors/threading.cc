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

#include "tensorflow_federated/cc/core/impl/executors/threading.h"

#include <functional>
#include <future>  // NOLINT
#include <string_view>
#include <thread>  // NOLINT
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

namespace tensorflow_federated {

ThreadPool::ThreadPool(int32_t num_threads, std::string_view name)
    : pool_name_(name) {
  if (num_threads < 1) {
    LOG(QFATAL) << "num_threads must be positive";
  }
  absl::MutexLock lock(&pool_mutex_);
  for (int32_t i = 0; i < num_threads; ++i) {
    threads_.emplace_back([this]() {
      while (true) {
        pool_mutex_.LockWhen(
            absl::Condition(this, &ThreadPool::has_work_or_closed));
        if (work_queue_.empty() && closed_) {
          pool_mutex_.Unlock();
          return;
        }
        const std::function<void()> f = std::move(work_queue_.front());
        work_queue_.pop_front();
        pool_mutex_.Unlock();
        f();
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  Close();
  for (std::thread& t : threads_) {
    t.join();
  }
}

absl::Status ThreadPool::Schedule(std::function<void()> task) {
  absl::MutexLock lock(&pool_mutex_);
  if (closed_) {
    return absl::FailedPreconditionError(
        "Called Schedule() on a ThreadPool that is closed.");
  }
  work_queue_.push_back(std::move(task));
  return absl::OkStatus();
}

bool ThreadPool::has_work_or_closed() ABSL_SHARED_LOCKS_REQUIRED(pool_mutex_) {
  return !work_queue_.empty() || closed_;
}

void ThreadPool::Close() {
  absl::MutexLock lock(&pool_mutex_);
  closed_ = true;
}

bool ParallelTasksInner_::AllDone_() ABSL_SHARED_LOCKS_REQUIRED(mutex_) {
  return remaining_tasks_ == 0;
}

absl::Status ParallelTasks::add_task(std::function<absl::Status()> task) {
  {
    absl::WriterMutexLock lock(&shared_inner_->mutex_);
    shared_inner_->remaining_tasks_ += 1;
  }
  auto void_task = [inner = shared_inner_, task = std::move(task)]() {
    absl::Status result = task();
    absl::WriterMutexLock lock(&inner->mutex_);
    inner->status_.Update(std::move(result));
    inner->remaining_tasks_ -= 1;
  };
  if (thread_pool_ != nullptr) {
    return thread_pool_->Schedule(std::move(void_task));
  } else {
    std::thread th(std::move(void_task));
    th.detach();
    return absl::OkStatus();
  }
}

absl::Status ParallelTasks::WaitAll() {
  // NOTE: we must not short-circuit on errors, as the threaded tasks must not
  // be allowed to outlive any temporary variables they reference from the
  // scope that called `WaitAll`.
  shared_inner_->mutex_.ReaderLockWhen(
      absl::Condition(&*shared_inner_, &ParallelTasksInner_::AllDone_));
  absl::Status status = shared_inner_->status_;
  shared_inner_->mutex_.ReaderUnlock();
  return status;
}

}  // namespace tensorflow_federated
