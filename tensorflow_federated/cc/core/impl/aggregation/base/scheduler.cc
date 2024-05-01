/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_federated/cc/core/impl/aggregation/base/scheduler.h"

#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <queue>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/move_to_lambda.h"

namespace tensorflow_federated {

namespace {

// A helper class to track information about lifetime of an object.
// Uses a shared pointer (SharedMarker) to a boolean memory fragment
// which remembers if the object has been destroyed. Capturing the
// marker in a lambda gives us a clean way to CHECK fail if the
// object is accessed post destruction.
class LifetimeTracker {
 public:
  using SharedMarker = std::shared_ptr<bool>;
  LifetimeTracker() : marker_(std::make_shared<bool>(true)) {}
  virtual ~LifetimeTracker() { *marker_ = false; }
  SharedMarker& marker() { return marker_; }

 private:
  SharedMarker marker_;
};

// Implementation of workers.
class WorkerImpl : public Worker, public LifetimeTracker {
 public:
  explicit WorkerImpl(Scheduler* scheduler) : scheduler_(scheduler) {}

  ~WorkerImpl() override = default;

  void Schedule(std::function<void()> task) override {
    absl::MutexLock lock(&busy_);
    steps_.emplace_back(std::move(task));
    MaybeRunNext();
  }

 private:
  void MaybeRunNext() ABSL_EXCLUSIVE_LOCKS_REQUIRED(busy_) {
    if (running_ || steps_.empty()) {
      // Already running, and next task will be executed when finished, or
      // nothing to run.
      return;
    }
    auto task = std::move(steps_.front());
    steps_.pop_front();
    running_ = true;
    auto wrapped_task = MoveToLambda(std::move(task));
    auto marker = this->marker();
    scheduler_->Schedule([this, marker, wrapped_task] {
      // Call the Task which is stored in wrapped_task.value.
      (*wrapped_task)();

      // Run the next task.
      TFF_CHECK(*marker) << "Worker destroyed before all tasks finished";
      {
        // Try run next task if any.
        absl::MutexLock lock(&this->busy_);
        this->running_ = false;
        this->MaybeRunNext();
      }
    });
  }

  Scheduler* scheduler_;
  absl::Mutex busy_;
  bool running_ ABSL_GUARDED_BY(busy_) = false;
  std::deque<std::function<void()>> steps_ ABSL_GUARDED_BY(busy_);
};

// Implementation of thread pools.
class ThreadPoolScheduler : public Scheduler {
 public:
  explicit ThreadPoolScheduler(std::size_t thread_count)
      : idle_condition_(absl::Condition(IdleCondition, this)),
        active_count_(thread_count) {
    TFF_CHECK(thread_count > 0) << "invalid thread_count";

    // Create threads.
    for (int i = 0; i < thread_count; ++i) {
      threads_.emplace_back(std::thread([this] { this->PerThreadActivity(); }));
    }
  }

  ~ThreadPoolScheduler() override {
    {
      absl::MutexLock lock(&busy_);
      TFF_CHECK(IdleCondition(this))
          << "Thread pool must be idle at destruction time";

      threads_should_join_ = true;
      work_available_cond_var_.SignalAll();
    }

    for (auto& thread : threads_) {
      TFF_CHECK(thread.joinable()) << "Attempted to destroy a threadpool from "
                                      "one of its running threads";
      thread.join();
    }
  }

  void Schedule(std::function<void()> task) override {
    absl::MutexLock lock(&busy_);
    todo_.push(std::move(task));
    // Wake up a *single* thread to handle this task.
    work_available_cond_var_.Signal();
  }

  void WaitUntilIdle() override {
    busy_.LockWhen(idle_condition_);
    busy_.Unlock();
  }

  static bool IdleCondition(ThreadPoolScheduler* pool)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pool->busy_) {
    return pool->todo_.empty() && pool->active_count_ == 0;
  }

  void PerThreadActivity() {
    for (;;) {
      std::function<void()> task;
      {
        absl::MutexLock lock(&busy_);
        --active_count_;
        while (todo_.empty()) {
          if (threads_should_join_) {
            return;
          }

          work_available_cond_var_.Wait(&busy_);
        }

        // Destructor invariant
        TFF_CHECK(!threads_should_join_);
        task = std::move(todo_.front());
        todo_.pop();
        ++active_count_;
      }

      task();
    }
  }

  // A vector of threads allocated for execution.
  std::vector<std::thread> threads_;

  // A CondVar used to signal availability of tasks.
  //
  // We would prefer to use the more declarative absl::Condition instead,
  // however, this one only allows to wake up all threads if a new task is
  // available -- but we want to wake up only one in this case.
  absl::CondVar work_available_cond_var_;

  // See IdleCondition
  absl::Condition idle_condition_;

  // A mutex protecting mutable state in this class.
  absl::Mutex busy_;

  // Set when worker threads should join instead of waiting for work.
  bool threads_should_join_ ABSL_GUARDED_BY(busy_) = false;

  // Queue of tasks with work to do.
  std::queue<std::function<void()>> todo_ ABSL_GUARDED_BY(busy_);

  // The number of threads currently doing work in this pool.
  std::size_t active_count_ ABSL_GUARDED_BY(busy_);
};

}  // namespace

std::unique_ptr<Worker> Scheduler::CreateWorker() {
  return std::make_unique<WorkerImpl>(this);
}

std::unique_ptr<Scheduler> CreateThreadPoolScheduler(std::size_t thread_count) {
  return std::make_unique<ThreadPoolScheduler>(thread_count);
}

}  // namespace tensorflow_federated
