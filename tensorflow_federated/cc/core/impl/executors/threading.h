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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_THREADING_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_THREADING_H_

#include <future>  // NOLINT
#include <memory>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"

namespace tensorflow_federated {

// Runs the provided provided no-arg function on
// another thread, returning a future to the result.
template <typename Func,
          typename ReturnValue = typename std::invoke_result<Func>::type>
std::shared_future<ReturnValue> ThreadRun(Func lambda) {
  std::packaged_task<ReturnValue()> task(lambda);
  auto future_ptr = std::shared_future<ReturnValue>(task.get_future());
  std::thread th(std::move(task));
  th.detach();
  return future_ptr;
}

// Awaits the result of a ValueFuture, usually a future returning a
// StatusOr<ExecutorValue>. Returns the resulting status or value wrapped again
// as a StatusOr.
template <typename ValueFuture>
auto Wait(const ValueFuture& future) {
  future.wait();
  const auto& result = future.get();
  using StatusOrValue = typename std::remove_reference<decltype(result)>::type;
  if (!result.ok()) {
    return StatusOrValue(result.status());
  }
  return StatusOrValue(result.value());
}

// Extracts the `ExecutorValue`s from `successfully_completed_futures`.
template <typename ExecutorValue>
auto GetAll(
    const absl::Span<const std::shared_future<absl::StatusOr<ExecutorValue>>>
        successfully_completed_futures) {
  std::vector<ExecutorValue> out;
  out.reserve(successfully_completed_futures.size());
  for (auto& future : successfully_completed_futures) {
    out.emplace_back(future.get().value());
  }
  return out;
}

// Convenience function converting vector to absl::Span for GetAll.
template <typename ValueFuture>
auto GetAll(const std::vector<ValueFuture>& futures) {
  return GetAll(absl::Span<const ValueFuture>(futures));
}

// Waits for all of the futures in `futures` to complete, returning an error if
// any of them fail.
template <typename ExecutorValue>
absl::StatusOr<std::vector<ExecutorValue>> WaitAll(
    const absl::Span<const std::shared_future<absl::StatusOr<ExecutorValue>>>
        futures) {
  for (const auto& future : futures) {
    future.wait();
    if (!future.get().ok()) {
      return future.get().status();
    }
  }
  return GetAll(futures);
}

// Convenience function converting vector to absl::Span for WaitAll.
template <typename ValueFuture>
auto WaitAll(const std::vector<ValueFuture>& futures) {
  return WaitAll(absl::Span<const ValueFuture>(futures));
}

// Converts an already-ready result into a future.
template <typename ExecutorValue>
std::shared_future<absl::StatusOr<ExecutorValue>> ReadyFuture(
    ExecutorValue&& value) {
  std::promise<absl::StatusOr<ExecutorValue>> promise;
  promise.set_value(absl::StatusOr<ExecutorValue>(
      absl::in_place_t(), std::forward<ExecutorValue>(value)));
  return promise.get_future();
}

// Returns whether or not all of the futures in `futures` have completed, or
// an error if any of them has completed with an error.
template <typename ValueFuture>
absl::StatusOr<bool> AllReady(const absl::Span<const ValueFuture> futures) {
  bool all_ready = true;
  for (const ValueFuture& future : futures) {
    if (future.wait_for(std::chrono::duration<uint8_t>::zero()) ==
        std::future_status::ready) {
      if (!future.get().ok()) {
        return future.get().status();
      }
    } else {
      all_ready = false;
    }
  }
  return all_ready;
}

// Convenience function converting vector to absl::Span for AllReady.
template <typename ValueFuture>
auto AllReady(const std::vector<ValueFuture>& futures) {
  return AllReady(absl::Span<const ValueFuture>(futures));
}

// Runs `lambda` on the successful results of `futures` and returns a future
// for the result of `lambda`.
//
// If all of `futures` have completed already, `lambda` will be run on the
// current thread. If any `futures` have already failed, this function will
// immediately return the result of their failure.
//
// If not all `futures` are completed, a new thread will be spawned to await
// their results, and `lambda` will be run on that thread if and when `futures`
// all complete successfully.
template <typename Func, typename ValueFuture>
absl::StatusOr<ValueFuture> Map(std::vector<ValueFuture>&& futures,
                                Func lambda) {
  bool all_ready = TFF_TRY(AllReady(futures));
  if (all_ready) {
    return ReadyFuture(TFF_TRY(lambda(GetAll(futures))));
  }
  return ThreadRun(
      [futures = std::move(futures), lambda = std::move(lambda)]()
          -> absl::StatusOr<std::remove_const_t<
              std::remove_reference_t<decltype(futures[0].get().value())>>> {
        return lambda(TFF_TRY(WaitAll(futures)));
      });
}

class ParallelTasksInner_ {
 private:
  bool AllDone_() ABSL_SHARED_LOCKS_REQUIRED(mutex_);
  absl::Mutex mutex_;
  absl::Status status_ ABSL_GUARDED_BY(mutex_) = absl::OkStatus();
  uint32_t remaining_tasks_ ABSL_GUARDED_BY(mutex_) = 0;
  friend class ParallelTasks;
};

// A group of `absl::Status`-returning functions to be run in parallel.
class ParallelTasks {
 public:
  // Default constructor.
  ParallelTasks() : shared_inner_(std::make_shared<ParallelTasksInner_>()) {}

  // Move constructor.
  ParallelTasks(ParallelTasks&& other)
      : shared_inner_(std::move(other.shared_inner_)) {}

  // Move assignment not provided.
  // Note: this would need to wait for the previous tasks (if any) to complete
  // before performing the assignment. It isn't possible to join two tasks
  // groups, but we mustn't allow the old tasks to continue unbounded, lest they
  // outlive the scope of variables that they reference.

  // Destructor which calls `WaitAll` to ensure that no tasks outlive the scope
  // of the `ParallelTasks` object.
  ~ParallelTasks() {
    if (shared_inner_ != nullptr) {
      WaitAll().IgnoreError();
    }
  }

  // Spawns a thread to run a function and adds it to the parallel task group.
  void add_task(std::function<absl::Status()> task);

  // Waits until all tasks passed to `add_task` have successfully completed.
  //
  // Returns an `absl::Status` containing the first non-`ok` result of a task,
  // or `ok` if all tasks completed successfully.
  //
  // Note: This method does *not* short-circuit on errors, allowing tasks to
  // reference local variables without fear of the task outliving the scope from
  // which `WaitAll` was invoked.
  absl::Status WaitAll();

 private:
  std::shared_ptr<ParallelTasksInner_> shared_inner_;
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_THREADING_H_
