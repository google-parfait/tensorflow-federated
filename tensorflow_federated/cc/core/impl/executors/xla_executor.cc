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

#include "tensorflow_federated/cc/core/impl/executors/xla_executor.h"

#include <future>  // NOLINT
#include <memory>
#include <vector>

#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace tensorflow_federated {

// Representation for concrete values embedded in the XLA executor.
// This class promises to be cheaply copyable.
class XLAExecutorValue {};

using ValueFuture = std::shared_future<XLAExecutorValue>;

class XLAExecutor : public ExecutorBase<ValueFuture> {
 public:
  absl::string_view ExecutorName() final { return "XLAExecutor"; }
  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final {
    return absl::UnimplementedError("Not implemented yet");
  }

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture fn, absl::optional<ValueFuture> arg) final {
    return absl::UnimplementedError("Not implemented yet");
  }

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> members) final {
    return absl::UnimplementedError("Not implemented yet");
  }

  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture value,
                                              const uint32_t index) final {
    return absl::UnimplementedError("Not implemented yet");
  }

  absl::Status Materialize(ValueFuture value, v0::Value* value_pb) final {
    return absl::UnimplementedError("Not implemented yet.");
  }
};
std::shared_ptr<Executor> CreateXLAExecutor() {
  return std::make_shared<XLAExecutor>();
}
}  // namespace tensorflow_federated
