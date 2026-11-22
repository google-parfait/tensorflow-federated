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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/executors/data_backend.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/task.h"

namespace tensorflow_federated {

namespace {

using SharedId = std::shared_ptr<const OwnedValueId>;
using ValueTask = SharedTask<absl::StatusOr<SharedId>>;

class DataExecutor : public ExecutorBase<ValueTask> {
 public:
  DataExecutor(std::shared_ptr<Executor> child,
               std::shared_ptr<DataBackend> data_backend)
      : child_(std::move(child)), data_backend_(std::move(data_backend)) {}

 protected:
  absl::string_view ExecutorName() final {
    static constexpr absl::string_view kExecutorName = "DataExecutor";
    return kExecutorName;
  }

  absl::StatusOr<ValueTask> CreateExecutorValue(
      const v0::Value& value_pb) final {
    if (value_pb.has_computation() && value_pb.computation().has_data()) {
      // Note: `value_pb` is copied here in order to ensure that it remains
      // available for the lifetime of the resolving thread. However, it should
      // be relatively small and inexpensive (currently just a URI).
      federated_language::Data data = value_pb.computation().data();
      federated_language::Type data_type = value_pb.computation().type();
      return ScheduleTask(
          /*pool=*/nullptr,
          [](std::shared_ptr<Executor> child,
             std::shared_ptr<DataBackend> data_backend,
             federated_language::Data d,
             federated_language::Type dt) -> Task<absl::StatusOr<SharedId>> {
            v0::Value resolved_value;
            absl::Status resolve_status =
                data_backend->ResolveToValue(d, dt, resolved_value);
            if (!resolve_status.ok()) {
              co_return resolve_status;
            }
            absl::StatusOr<OwnedValueId> child_value =
                child->CreateValue(resolved_value);
            if (!child_value.ok()) {
              co_return child_value.status();
            }
            co_return std::make_shared<const OwnedValueId>(
                std::move(child_value).value());
          }(child_, data_backend_, std::move(data), std::move(data_type)));
    } else {
      OwnedValueId child_value = TFF_TRY(child_->CreateValue(value_pb));
      return MakeReadySharedTask<absl::StatusOr<SharedId>>(
          std::make_shared<const OwnedValueId>(std::move(child_value)));
    }
  }

  absl::StatusOr<ValueTask> CreateCall(
      ValueTask function_task, std::optional<ValueTask> argument_task) final {
    if (function_task.is_ready() &&
        (!argument_task.has_value() || argument_task->is_ready())) {
      const absl::StatusOr<SharedId>& fn_res = SyncWait(function_task);
      if (!fn_res.ok()) {
        return MakeReadySharedTask<absl::StatusOr<SharedId>>(fn_res.status());
      }
      if (argument_task.has_value()) {
        const absl::StatusOr<SharedId>& arg_res = SyncWait(*argument_task);
        if (!arg_res.ok()) {
          return MakeReadySharedTask<absl::StatusOr<SharedId>>(
              arg_res.status());
        }
      }
    }
    return ScheduleTask(
        /*pool=*/nullptr,
        [](std::shared_ptr<Executor> child, ValueTask fn,
           std::optional<ValueTask> arg) -> Task<absl::StatusOr<SharedId>> {
          absl::StatusOr<SharedId> fn_val = co_await fn;
          if (!fn_val.ok()) {
            co_return fn_val.status();
          }
          ValueId function_id = (*fn_val)->ref();
          std::optional<ValueId> argument_id = std::nullopt;
          if (arg.has_value()) {
            absl::StatusOr<SharedId> arg_val = co_await *arg;
            if (!arg_val.ok()) {
              co_return arg_val.status();
            }
            argument_id = (*arg_val)->ref();
          }
          absl::StatusOr<OwnedValueId> child_value =
              child->CreateCall(function_id, argument_id);
          if (!child_value.ok()) {
            co_return child_value.status();
          }
          co_return std::make_shared<const OwnedValueId>(
              std::move(child_value).value());
        }(child_, std::move(function_task), std::move(argument_task)));
  }

  absl::StatusOr<ValueTask> CreateStruct(
      std::vector<ValueTask> member_tasks) final {
    for (const ValueTask& task : member_tasks) {
      if (task.is_ready()) {
        const absl::StatusOr<SharedId>& res = SyncWait(task);
        if (!res.ok()) {
          return MakeReadySharedTask<absl::StatusOr<SharedId>>(res.status());
        }
      }
    }
    return ScheduleTask(
        /*pool=*/nullptr,
        [](std::shared_ptr<Executor> child,
           std::vector<ValueTask> members) -> Task<absl::StatusOr<SharedId>> {
          std::vector<ValueId> ids;
          ids.reserve(members.size());
          for (ValueTask& member_task : members) {
            absl::StatusOr<SharedId> member = co_await member_task;
            if (!member.ok()) {
              co_return member.status();
            }
            ids.push_back((*member)->ref());
          }
          absl::StatusOr<OwnedValueId> child_value = child->CreateStruct(ids);
          if (!child_value.ok()) {
            co_return child_value.status();
          }
          co_return std::make_shared<const OwnedValueId>(
              std::move(child_value).value());
        }(child_, std::move(member_tasks)));
  }

  absl::StatusOr<ValueTask> CreateSelection(ValueTask source_task,
                                            const uint32_t index) final {
    if (source_task.is_ready()) {
      const absl::StatusOr<SharedId>& res = SyncWait(source_task);
      if (!res.ok()) {
        return MakeReadySharedTask<absl::StatusOr<SharedId>>(res.status());
      }
    }
    return ScheduleTask(
        /*pool=*/nullptr,
        [](std::shared_ptr<Executor> child, ValueTask src,
           uint32_t idx) -> Task<absl::StatusOr<SharedId>> {
          absl::StatusOr<SharedId> source = co_await src;
          if (!source.ok()) {
            co_return source.status();
          }
          absl::StatusOr<OwnedValueId> child_value =
              child->CreateSelection((*source)->ref(), idx);
          if (!child_value.ok()) {
            co_return child_value.status();
          }
          co_return std::make_shared<const OwnedValueId>(
              std::move(child_value).value());
        }(child_, std::move(source_task), index));
  }

  absl::Status Materialize(ValueTask value_task, v0::Value* value_pb) final {
    const absl::StatusOr<SharedId>& value_res = SyncWait(value_task);
    if (!value_res.ok()) {
      return value_res.status();
    }
    return child_->Materialize((*value_res)->ref(), value_pb);
  }

 private:
  std::shared_ptr<Executor> child_;
  std::shared_ptr<DataBackend> data_backend_;
};

}  // namespace

std::shared_ptr<Executor> CreateDataExecutor(
    std::shared_ptr<Executor> child,
    std::shared_ptr<DataBackend> data_backend) {
  return std::make_shared<DataExecutor>(std::move(child),
                                        std::move(data_backend));
}

}  // namespace tensorflow_federated
