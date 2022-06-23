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
#include <thread>  // NOLINT
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

ABSL_FLAG(bool, tff_debug_force_sequence_tasks, false,
          "If true, prevents ParallelTasks objects from detaching new "
          "threads and instead makes each ParallelTasks::add_task call "
          "blocking.");

namespace tensorflow_federated {

ParallelTasks::ParallelTasks()
    : debug_mode_(absl::GetFlag(FLAGS_tff_debug_force_sequence_tasks)) {}

bool ParallelTasksInner_::AllDone_() ABSL_SHARED_LOCKS_REQUIRED(mutex_) {
  return remaining_tasks_ == 0;
}

void ParallelTasks::add_task(std::function<absl::Status()> task) {
  if (absl::GetFlag(FLAGS_tff_debug_force_sequence_tasks)) {
    // Debugging path, don't launch new threads, run the task immediately and
    // block.
    absl::MutexLock lock(&shared_inner_->mutex_);
    ++shared_inner_->remaining_tasks_;
    shared_inner_->status_.Update(task());
    --shared_inner_->remaining_tasks_;
    return;
  }
  {
    absl::MutexLock lock(&shared_inner_->mutex_);
    ++shared_inner_->remaining_tasks_;
  }
  std::thread task_thread([inner = shared_inner_, task = std::move(task)]() {
    absl::Status result = task();
    absl::WriterMutexLock lock(&inner->mutex_);
    inner->status_.Update(std::move(result));
    --inner->remaining_tasks_;
  });
  task_thread.detach();
}

absl::Status ParallelTasks::WaitAll() {
  // NOTE: we must not short-circuit on errors, as the threaded tasks must not
  // be allowed to outlive any temporary variables they reference from the
  // scope that called `WaitAll`.
  shared_inner_->mutex_.LockWhen(
      absl::Condition(&*shared_inner_, &ParallelTasksInner_::AllDone_));
  absl::Status status = shared_inner_->status_;
  shared_inner_->mutex_.Unlock();
  return status;
}

}  // namespace tensorflow_federated
