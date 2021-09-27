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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EXECUTOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EXECUTOR_H_

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

class OwnedValueId;

using ValueId = uint64_t;

// A dynamically-dispatched executor interface.
//
// This interface allows users to execute TensorFlow Federated computations.
// Input values and computation graphs are inserted via `CreateValue`.
// Values may be combined or split apart using `CreateStruct` and
// `CreateSelection`, allowing the user to construct inputs and deconstruct
// outputs of functions (called using `CreateCall`). Finally, users may request
// concrete results by `Materialize`ing a value back into a proto
// representation. For example:
//
// ```
// // Imagine we have a `v0::Value` holding a TensorFlow function and some
// // tensors: (my_tf_fn, t1, t2).
// v0::Value graph_and_val = ...;
// // and some executor on which to run our computations.
// std::unique_ptr<Executor> exec = ...;
//
// // Say we want to run the computation `my_tf_fn(t1, t2)`.
// // First, we import our `v0::Value`.
// OwnedValueId in = TFF_TRY(exec->CreateValue(graph_and_val));
//
// // We construct values for the function and the argument struct `<t1, t2>`.
// OwnedValueId my_tf_function = TFF_TRY(exec->CreateSelection(in, 0));
// OwnedValueId t1 = TFF_TRY(exec->CreateSelection(in, 1));
// OwnedValueId t2 = TFF_TRY(exec->CreateSelection(in, 2));
// OwnedValueId arg = TFF_TRY(exec->CreateStruct([t1, t2]));
//
// // Then `CreateCall` on the function with the appropriate argument.
// OwnedValueId result = TFF_TRY(exec->CreateCall(my_tf_function, arg));
//
// // Finally, we can materialize our result back into a proto.
// v0::Value result_val;
// exec->Materialize(result, &result_val);
// ... do more with `result_val ...
// ```
//
// It is common for `Executor` implementations to internally delegate to other
// executors so that each may handle only a portion of the total computation
// interface.
//
// Note that, between ingestion (`CreateValue`) and output (`Materialize`),
// `Executor` users interact with values purely based on ID. This allows for
// the `Executor` virtual interface to stay the same despite
// implementations using different value representations internally. Resources
// associated with the underlying value will be retained until the corresponding
// `OwnedValueId` is destructed, resulting in a `Dispose` call on the underlying
// executor.
class Executor {
 public:
  // Embeds a `Value` into the executor.
  //
  // This method is expected return quickly. It should not block on complex
  // computation or IO-bound work, instead kicking off that work to run
  // asynchronously.
  //
  // Note: structure field names present in `CreateValue` are not persisted,
  // and will not be present in the returned `Materialize`d structure.
  virtual absl::StatusOr<OwnedValueId> CreateValue(
      const v0::Value& value_pb) = 0;

  // Calls `function` with optional `argument`.
  //
  // This method is expected return quickly. It should not block on complex
  // computation or IO-bound work, instead kicking off that work to run
  // asynchronously.
  virtual absl::StatusOr<OwnedValueId> CreateCall(
      const ValueId function, const absl::optional<const ValueId> argument) = 0;

  // Creates a structure with ordered `members`.
  //
  // `CreateStruct` is most commonly used to pack argument values prior to a
  // `CreateCall` invocation, or to pack together the final results of a
  // computation prior to `Materialize`ing them.
  //
  // The value that results from `CreateStruct` can also be unpacked using
  // `CreateSelection`.
  //
  // This method is expected return quickly. It should not block on complex
  // computation or IO-bound work, instead kicking off that work to run
  // asynchronously.
  virtual absl::StatusOr<OwnedValueId> CreateStruct(
      const absl::Span<const ValueId> members) = 0;

  // Selects the value at `index` from structure-typed `source` value.
  //
  // `CreateSelection` is most commonly used to unpack results after a
  // `CreateCall` invocation, or to unpack the initial aggregate value provided
  // in `CreateValue`.
  //
  // `CreateSelection` can also be used to unpack values created using
  // `CreateStruct`.
  //
  // This method is expected to return quickly. It should not block on complex
  // computation or IO-bound work, instead kicking off that work to run
  // asynchronously.
  virtual absl::StatusOr<OwnedValueId> CreateSelection(
      const ValueId source, const uint32_t index) = 0;

  // Materialize the value as a concrete structure.
  //
  // This method is blocking: it may synchronously wait for the result of
  // complex computations or IO-bound work.
  virtual absl::Status Materialize(const ValueId value,
                                   v0::Value* value_pb) = 0;

  // Convenience method for calling `Materialize` without a pre-existing proto.
  inline absl::StatusOr<v0::Value> Materialize(const ValueId value) {
    v0::Value value_pb;
    TFF_TRY(Materialize(value, &value_pb));
    return value_pb;
  }

  // Dispose of a value, releasing any associated resources.
  //
  // Users of this class should not typically access this function directly.
  // The `OwnedValueId`s returned will `Dispose` of themselves on destruction.
  virtual absl::Status Dispose(const ValueId value) = 0;

  virtual ~Executor() {}
};

// A single-owner `ValueId` which will `Dispose` itself upon destruction.
class OwnedValueId {
 public:
  explicit OwnedValueId(std::weak_ptr<Executor> exec, ValueId id)
      : exec_(std::move(exec)), id_(id) {}
  // Move constructor.
  OwnedValueId(OwnedValueId&& other)
      : exec_(std::move(other.exec_)), id_(other.id_) {
    other.forget();
  }
  // Move assignment.
  OwnedValueId& operator=(OwnedValueId&& other) {
    // Dispose of any old value.
    this->release();
    this->exec_ = std::move(other.exec_);
    this->id_ = other.id_;
    other.forget();
    return *this;
  }
  // Returns an unowned `ValueId` pointing to the same value.
  // It is invalid to use the resulting ID after destructing this
  // `OwnedValueId`.
  ValueId ref() const {
    CHECK(id_ != INVALID_ID) << id_ << " != " << INVALID_ID;
    return id_;
  }
  // Implicit `ref` conversion for convenience. See `ref` documentation.
  operator ValueId() const { return ref(); }
  // Discards ownership. The caller is responsible for cleanup.
  void forget() { id_ = INVALID_ID; }
  // Releases the value in the underlying executor, invalidating this object.
  void release() {
    if (id_ != INVALID_ID) {
      if (auto exec_strong = exec_.lock()) {
        exec_strong->Dispose(id_).IgnoreError();
      }
      id_ = INVALID_ID;
    }
  }
  ~OwnedValueId() { release(); }

 private:
  std::weak_ptr<Executor> exec_;
  ValueId id_;
  // Use ValueId max to indicate "no value."
  static const ValueId INVALID_ID = std::numeric_limits<ValueId>::max();
};

// A base class to allow for easy implementation of `Executor`.
// `Executor` implementations should typically inherit from
// `ExecutorBase<executor-specific-value-implementation>`.
//
// `ExecutorBase` must always be placed inside of a `shared_ptr`. Constructing
// an `ExecutorBase` outside of a `shared_ptr` will result in undefined behavior
// due to the use of `enable_shared_from_this`.
//
// Note: `ExecutorValue`s must be copy-constructible. Typically, this template
// parameter will be a `std::shared_ptr` or similar wrapper around a concrete
// value class.
template <class ExecutorValue>
class ExecutorBase : public Executor,
                     public std::enable_shared_from_this<Executor> {
 private:
  absl::Mutex mutex_;
  ValueId next_value_id_ ABSL_GUARDED_BY(mutex_) = 0;
  absl::flat_hash_map<ValueId, ExecutorValue> tracked_values_
      ABSL_GUARDED_BY(mutex_);

  // Tracks the provided value and returns the ID which refers to it.
  absl::StatusOr<OwnedValueId> TrackValue(ExecutorValue value) {
    absl::WriterMutexLock lock(&mutex_);
    ValueId id = next_value_id_++;
    tracked_values_.emplace(id, std::move(value));
    return absl::StatusOr<OwnedValueId>(absl::in_place_t(), weak_from_this(),
                                        id);
  }

  // Returns a copy of the value previously stored with `TrackValue`.
  absl::StatusOr<ExecutorValue> GetTracked(ValueId value_id) {
    absl::ReaderMutexLock lock(&mutex_);
    auto value_iter = tracked_values_.find(value_id);
    if (value_iter == tracked_values_.end()) {
      return absl::NotFoundError(
          absl::StrCat(ExecutorName(), " value not found: ", value_id));
    }
    return value_iter->second;
  }

 protected:
  // Logs the current method and records its trace to the TensorFlow profiler.
  absl::optional<tensorflow::profiler::TraceMe> Trace(const char* method_name) {
    bool enabled = VLOG_IS_ON(1) || tensorflow::profiler::TraceMe::Active();
    if (!enabled) {
      return absl::nullopt;
    }
    std::string path = absl::StrCat(ExecutorName(), "::", method_name);
    absl::string_view path_view(path);
    VLOG(1) << path_view;
    // Safe to pass in a view here: `TraceMe` internally copies to an owned
    // `std::string`.
    return absl::make_optional<tensorflow::profiler::TraceMe>(path_view);
  }

  // Clears all currently tracked values from the executor.
  // This method is intended to be used by child class destructors to ensure
  // that the `ExecutorValue` references held by `tracked_values_` have been
  // destroyed.
  void ClearTracked() {
    absl::WriterMutexLock lock(&mutex_);
    tracked_values_.clear();
  }

  // Returns the string name of the current executor.
  virtual const char* ExecutorName() = 0;
  virtual absl::StatusOr<ExecutorValue> CreateExecutorValue(
      const v0::Value& value_pb) = 0;
  virtual absl::StatusOr<ExecutorValue> CreateCall(
      ExecutorValue function, absl::optional<ExecutorValue> argument) = 0;
  virtual absl::StatusOr<ExecutorValue> CreateStruct(
      std::vector<ExecutorValue> members) = 0;
  virtual absl::StatusOr<ExecutorValue> CreateSelection(
      ExecutorValue value, const uint32_t index) = 0;
  virtual absl::Status Materialize(ExecutorValue value,
                                   v0::Value* value_pb) = 0;
  virtual ~ExecutorBase() {}

 public:
  absl::StatusOr<OwnedValueId> CreateValue(const v0::Value& value_pb) final {
    auto trace = Trace("CreateValue");
    return TrackValue(TFF_TRY(CreateExecutorValue(value_pb)));
  }

  absl::StatusOr<OwnedValueId> CreateCall(
      const ValueId function,
      const absl::optional<const ValueId> argument) final {
    auto trace = Trace("CreateCall");
    ExecutorValue function_val = TFF_TRY(GetTracked(function));
    absl::optional<ExecutorValue> argument_val;
    if (argument.has_value()) {
      argument_val = TFF_TRY(GetTracked(argument.value()));
    }
    return TrackValue(
        TFF_TRY(CreateCall(std::move(function_val), std::move(argument_val))));
  }

  absl::StatusOr<OwnedValueId> CreateStruct(
      const absl::Span<const ValueId> members) final {
    auto trace = Trace("CreateStruct");
    std::vector<ExecutorValue> member_values;
    for (const ValueId member_id : members) {
      member_values.emplace_back(TFF_TRY(GetTracked(member_id)));
    }
    return TrackValue(TFF_TRY(CreateStruct(std::move(member_values))));
  }

  absl::StatusOr<OwnedValueId> CreateSelection(const ValueId source,
                                               const uint32_t index) final {
    auto trace = Trace("CreateSelection");
    return TrackValue(
        TFF_TRY(CreateSelection(TFF_TRY(GetTracked(source)), index)));
  }

  absl::Status Materialize(const ValueId value_id, v0::Value* value_pb) final {
    auto trace = Trace("Materialize");
    return Materialize(TFF_TRY(GetTracked(value_id)), value_pb);
  }

  absl::Status Dispose(const ValueId value) final {
    auto trace = Trace("Dispose");
    absl::WriterMutexLock lock(&mutex_);
    auto value_iter = tracked_values_.find(value);
    if (value_iter == tracked_values_.end()) {
      return absl::NotFoundError(absl::StrCat(
          ExecutorName(), " value not found: ", value, ", cannot dispose."));
    }
    tracked_values_.erase(value_iter);
    return absl::OkStatus();
  }
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EXECUTOR_H_
