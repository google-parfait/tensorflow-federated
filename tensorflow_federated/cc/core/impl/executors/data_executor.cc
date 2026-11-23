/* Copyright 2022, The TensorFlow Federated Authors.

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

#include "tensorflow_federated/cc/core/impl/executors/data_executor.h"

#include <cstdint>
#include <future>  // NOLINT
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow_federated/cc/core/impl/executors/data_backend.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"

namespace tensorflow_federated {

namespace {

using SharedId = std::shared_ptr<const OwnedValueId>;
using ValueFuture = std::shared_future<absl::StatusOr<SharedId>>;

class DataExecutor : public ExecutorBase<ValueFuture> {
 public:
  DataExecutor(std::shared_ptr<Executor> child,
               std::shared_ptr<DataBackend> data_backend)
      : child_(std::move(child)), data_backend_(std::move(data_backend)) {}

  // Wait for active detached tasks and clear tracked futures before destroying
  // `child_`.
  ~DataExecutor() override {
    WaitForTasks();
    ClearTracked();
  }

 private:
  // Tracks active asynchronous closures so the destructor drains them before
  // destroying `child_` and `data_backend_`.
  class ActiveTaskGuard {
   public:
    explicit ActiveTaskGuard(DataExecutor* executor) : executor_(executor) {
      executor_->StartTask();
    }
    ActiveTaskGuard(const ActiveTaskGuard&) = delete;
    ActiveTaskGuard& operator=(const ActiveTaskGuard&) = delete;
    ActiveTaskGuard(ActiveTaskGuard&& other) noexcept
        : executor_(std::exchange(other.executor_, nullptr)) {}
    ActiveTaskGuard& operator=(ActiveTaskGuard&& other) noexcept {
      if (this != &other) {
        if (executor_ != nullptr) {
          executor_->FinishTask();
        }
        executor_ = std::exchange(other.executor_, nullptr);
      }
      return *this;
    }
    ~ActiveTaskGuard() {
      if (executor_ != nullptr) {
        executor_->FinishTask();
      }
    }

   private:
    DataExecutor* executor_;
  };

  void StartTask() {
    absl::MutexLock lock(tasks_mutex_);
    ++active_tasks_;
  }

  void FinishTask() {
    absl::MutexLock lock(tasks_mutex_);
    --active_tasks_;
  }

  bool HasNoActiveTasks() const ABSL_SHARED_LOCKS_REQUIRED(tasks_mutex_) {
    return active_tasks_ == 0;
  }

  void WaitForTasks() {
    absl::MutexLock lock(tasks_mutex_);
    tasks_mutex_.Await(absl::Condition(this, &DataExecutor::HasNoActiveTasks));
  }

 protected:
  absl::string_view ExecutorName() final {
    static constexpr absl::string_view kExecutorName = "DataExecutor";
    return kExecutorName;
  }

  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final {
    if (value_pb.has_computation() && value_pb.computation().has_data()) {
      // Note: `value_pb` is copied here in order to ensure that it remains
      // available for the lifetime of the resolving thread. However, it should
      // be relatively small and inexpensive (currently just a URI).
      federated_language::Data data = value_pb.computation().data();
      federated_language::Type data_type = value_pb.computation().type();
      return ThreadRun([this, data = std::move(data),
                        data_type = std::move(data_type),
                        task_guard = ActiveTaskGuard(
                            this)]() -> absl::StatusOr<SharedId> {
        (void)task_guard;
        Trace("DataExecutor::DataBackend::ResolveToValue");
        v0::Value resolved_value;
        TFF_TRY(data_backend_->ResolveToValue(data, data_type, resolved_value));
        OwnedValueId child_value = TFF_TRY(child_->CreateValue(resolved_value));
        return std::make_shared<OwnedValueId>(std::move(child_value));
      });
    } else {
      OwnedValueId child_value = TFF_TRY(child_->CreateValue(value_pb));
      return ReadyFuture(
          std::make_shared<const OwnedValueId>(std::move(child_value)));
    }
  }

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture function_future,
      std::optional<ValueFuture> argument_future) final {
    std::vector<ValueFuture> futures = {function_future};
    if (argument_future.has_value()) {
      futures.push_back(std::move(*argument_future));
    }
    return Map(
        std::move(futures),
        [this, task_guard = ActiveTaskGuard(this)](
            std::vector<SharedId> values) -> absl::StatusOr<SharedId> {
          (void)task_guard;
          // `values` holds the resolved `futures`, either
          // `{function}` or `{function, argument}`.
          ValueId function_id = values[0]->ref();
          std::optional<ValueId> argument_id = std::nullopt;
          if (values.size() == 2) {
            argument_id = values[1]->ref();
          }
          OwnedValueId child_value =
              TFF_TRY(child_->CreateCall(function_id, argument_id));
          return std::make_shared<const OwnedValueId>(std::move(child_value));
        });
  }

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> member_futures) final {
    return Map(
        std::move(member_futures),
        [this, task_guard = ActiveTaskGuard(this)](
            std::vector<SharedId> members) -> absl::StatusOr<SharedId> {
          (void)task_guard;
          std::vector<ValueId> ids;
          ids.reserve(members.size());
          for (const auto& member : members) {
            ids.push_back(member->ref());
          }
          OwnedValueId child_value = TFF_TRY(child_->CreateStruct(ids));
          return std::make_shared<const OwnedValueId>(std::move(child_value));
        });
  }

  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture source_future,
                                              const uint32_t index) final {
    return Map(
        std::vector<ValueFuture>({std::move(source_future)}),
        [index, this, task_guard = ActiveTaskGuard(this)](
            std::vector<SharedId> source_in_vec) -> absl::StatusOr<SharedId> {
          (void)task_guard;
          OwnedValueId child_value =
              TFF_TRY(child_->CreateSelection(source_in_vec[0]->ref(), index));
          return std::make_shared<const OwnedValueId>(std::move(child_value));
        });
  }

  absl::Status Materialize(ValueFuture value_fut, v0::Value* value_pb) final {
    SharedId value = TFF_TRY(Wait(std::move(value_fut)));
    return child_->Materialize(value->ref(), value_pb);
  }

 private:
  std::shared_ptr<Executor> child_;
  std::shared_ptr<DataBackend> data_backend_;
  absl::Mutex tasks_mutex_;
  int32_t active_tasks_ ABSL_GUARDED_BY(tasks_mutex_) = 0;
};

}  // namespace

std::shared_ptr<Executor> CreateDataExecutor(
    std::shared_ptr<Executor> child,
    std::shared_ptr<DataBackend> data_backend) {
  return std::make_shared<DataExecutor>(std::move(child),
                                        std::move(data_backend));
}

}  // namespace tensorflow_federated
