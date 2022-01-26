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

#include <future>  // NOLINT
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "tensorflow_federated/cc/core/impl/executors/data_backend.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"

namespace tensorflow_federated {

namespace {

using SharedId = std::shared_ptr<OwnedValueId>;
using ValueFuture = std::shared_future<absl::StatusOr<SharedId>>;

class DataExecutor : public ExecutorBase<ValueFuture> {
 public:
  DataExecutor(std::shared_ptr<Executor> child,
               std::shared_ptr<DataBackend> data_backend)
      : child_(std::move(child)), data_backend_(std::move(data_backend)) {}

 protected:
  const char* ExecutorName() final { return "DataExecutor"; }

  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final {
    if (value_pb.has_computation() && value_pb.computation().has_data()) {
      // Note: `value_pb` is copied here in order to ensure that it remains
      // available for the lifetime of the resolving thread. However, it should
      // be relatively small and inexpensive (currently just a URI).
      v0::Data data = value_pb.computation().data();
      return ThreadRun([this, data = std::move(data),
                        this_keepalive = shared_from_this()]()
                           -> absl::StatusOr<std::shared_ptr<OwnedValueId>> {
        v0::Value resolved_value;
        TFF_TRY(data_backend_->ResolveToValue(data, resolved_value));
        OwnedValueId child_value = TFF_TRY(child_->CreateValue(resolved_value));
        return std::make_shared<OwnedValueId>(std::move(child_value));
      });
    } else {
      OwnedValueId child_value = TFF_TRY(child_->CreateValue(value_pb));
      return ReadyFuture(
          std::make_shared<OwnedValueId>(std::move(child_value)));
    }
  }

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture function_future,
      absl::optional<ValueFuture> argument_future) final {
    std::vector<ValueFuture> futures({function_future});
    if (argument_future.has_value()) {
      futures.push_back(std::move(*argument_future));
    }
    return Map(std::move(futures),
               [this, this_keepalive = shared_from_this()](
                   std::vector<SharedId>&& values) -> absl::StatusOr<SharedId> {
                 ValueId function_id = values[0]->ref();
                 absl::optional<ValueId> argument_id = absl::nullopt;
                 if (values.size() == 2) {
                   argument_id = values[1]->ref();
                 }
                 OwnedValueId child_value =
                     TFF_TRY(child_->CreateCall(function_id, argument_id));
                 return std::make_shared<OwnedValueId>(std::move(child_value));
               });
  }

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> member_futures) final {
    return Map(
        std::move(member_futures),
        [this, this_keepalive = shared_from_this()](
            std::vector<SharedId>&& members) -> absl::StatusOr<SharedId> {
          std::vector<ValueId> ids;
          ids.reserve(members.size());
          for (const auto& member : members) {
            ids.push_back(member->ref());
          }
          OwnedValueId child_value = TFF_TRY(child_->CreateStruct(ids));
          return std::make_shared<OwnedValueId>(std::move(child_value));
        });
  }

  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture source_future,
                                              const uint32_t index) final {
    return Map(
        std::vector<ValueFuture>({std::move(source_future)}),
        [index, this, this_keepalive = shared_from_this()](
            std::vector<SharedId>&& source_in_vec) -> absl::StatusOr<SharedId> {
          OwnedValueId child_value =
              TFF_TRY(child_->CreateSelection(source_in_vec[0]->ref(), index));
          return std::make_shared<OwnedValueId>(std::move(child_value));
        });
  }

  absl::Status Materialize(ValueFuture value_fut, v0::Value* value_pb) final {
    SharedId value = TFF_TRY(Wait(std::move(value_fut)));
    return child_->Materialize(value->ref(), value_pb);
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
