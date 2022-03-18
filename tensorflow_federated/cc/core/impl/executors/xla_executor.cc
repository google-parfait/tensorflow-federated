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
#include <utility>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace tensorflow_federated {

namespace {

class ExecutorValue;

using ValueFuture = std::shared_future<absl::StatusOr<ExecutorValue>>;

class XLAExecutor : public ExecutorBase<ValueFuture> {
 public:
  explicit XLAExecutor() {}
  ~XLAExecutor() override {}

  absl::string_view ExecutorName() final { return "XLAExecutor"; }
};

// Representation for values inside the XLA Executor.
class ExecutorValue {
 public:
  // Whether a given `ExecutorValue` is a structure or a single tensor.
  enum class ValueType { TENSOR, STRUCT };

  // Constructs an `ExecutorValue` from a `tensorflow::Tensor`.
  // NOTE: `tensorflow::Tensor` is internally refcounted, so copies of it are
  // inexpensive.
  explicit ExecutorValue(const tensorflow::Tensor t) : value_(std::move(t)) {}

  // Constructs a structural `ExecutorValue` from a list of elements.
  explicit ExecutorValue(std::shared_ptr<std::vector<ExecutorValue>> elements)
      : value_(elements) {}

  // Copy constructor.
  //
  // Copies are shallow: we only have to bump the reference count for either
  // the elements list or the `tensorflow::Tensor`.
  explicit ExecutorValue(const ExecutorValue& other) : value_(other.value_) {}

  // Move constructor.
  ExecutorValue(ExecutorValue&& other) : value_(std::move(other.value_)) {}
  // Move assignment.
  ExecutorValue& operator=(ExecutorValue&& other) {
    this->value_ = std::move(other.value_);
    return *this;
  }

  // Returns whether this value is a structure or a single tensor.
  ValueType type() const {
    if (absl::holds_alternative<tensorflow::Tensor>(value_)) {
      return ValueType::TENSOR;
    } else {
      return ValueType::STRUCT;
    }
  }

  // Returns a reference to the inner tensor.
  // Requires that `type()` is `ValueType::TENSOR`.
  const tensorflow::Tensor& tensor() const {
    return absl::get<tensorflow::Tensor>(value_);
  }

  // Returns a reference to the inner elements list.
  // Requires that `type()` is `ValueType::STRUCT`.
  absl::Span<const ExecutorValue> elements() const {
    return *absl::get<std::shared_ptr<std::vector<ExecutorValue>>>(value_);
  }

 private:
  ExecutorValue() = delete;

  absl::variant<tensorflow::Tensor, std::shared_ptr<std::vector<ExecutorValue>>>
      value_;
};

}  // namespace

}  // namespace tensorflow_federated
