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

#include "tensorflow_federated/cc/core/impl/executors/sequence_executor.h"

#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_federated/cc/core/impl/executors/dataset_conversions.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/sequence_intrinsics.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

namespace {

using Embedded = std::shared_ptr<OwnedValueId>;

Embedded ShareValueId(OwnedValueId id) {
  return std::make_shared<OwnedValueId>(std::move(id));
}

class SequenceIterator {
 public:
  // Returns next element of a data stream, or absl::nullopt if no such element
  // exists. Returns a non-OK absl::Status if an unexpected issue occurs pulling
  // the next element from the stream.
  virtual absl::StatusOr<absl::optional<Embedded>> GetNextEmbedded() = 0;
  virtual ~SequenceIterator() {}
};

absl::StatusOr<Embedded> EmbedTensorsAsType(
    std::vector<tensorflow::Tensor> tensors,
    std::shared_ptr<Executor> target_executor, v0::Type type) {
  // TODO(b/217763470): implement handling of structures, structures of
  // structures, etc.
  switch (type.type_case()) {
    case v0::Type::kTensor: {
      v0::Value tensor_value;
      TFF_TRY(SerializeTensorValue(tensors.at(0), &tensor_value));
      return ShareValueId(TFF_TRY(target_executor->CreateValue(tensor_value)));
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Embedding tensors as structure not yet "
                       "supported. Attempted to serialize as type: ",
                       type.Utf8DebugString()));
  }
}

class MappedIterator : public SequenceIterator {
 public:
  explicit MappedIterator(std::shared_ptr<Executor> target_executor,
                          std::shared_ptr<SequenceIterator> existing_iterator,
                          Embedded mapping_fn)
      : target_executor_(target_executor),
        existing_iterator_(std::move(existing_iterator)),
        mapping_fn_(mapping_fn) {}

  ~MappedIterator() final {}

  absl::StatusOr<absl::optional<Embedded>> GetNextEmbedded() final {
    absl::optional<Embedded> unmapped_value =
        TFF_TRY(existing_iterator_->GetNextEmbedded());
    if (!unmapped_value.has_value()) {
      return absl::nullopt;
    }
    return ShareValueId(TFF_TRY(target_executor_->CreateCall(
        mapping_fn_->ref(), unmapped_value.value()->ref())));
  }

 private:
  MappedIterator() = delete;
  std::shared_ptr<Executor> target_executor_;
  std::shared_ptr<SequenceIterator> existing_iterator_;
  Embedded mapping_fn_;
};

class DatasetIterator : public SequenceIterator {
 public:
  explicit DatasetIterator(tensorflow::data::standalone::Dataset* ds,
                           v0::Type element_type,
                           std::shared_ptr<Executor> target_executor)
      : ds_(std::move(ds)),
        element_type_(element_type),
        target_executor_(target_executor) {}

  ~DatasetIterator() final {}

  absl::StatusOr<absl::optional<Embedded>> GetNextEmbedded() final {
    TFF_TRY(EnsureIteratorInitialized());
    v0::Value next_value;
    std::vector<tensorflow::Tensor> output_tensors;
    bool end_of_data = false;
    auto status = ds_iterator_->GetNext(&output_tensors, &end_of_data);
    if (!status.ok()) {
      return absl::InternalError(absl::StrCat(
          "error pulling elements from dataset: ", status.error_message()));
    }
    if (end_of_data) {
      return absl::nullopt;
    }
    return TFF_TRY(
        EmbedTensorsAsType(output_tensors, target_executor_, element_type_));
  }

 private:
  absl::Status EnsureIteratorInitialized() {
    if (ds_iterator_ != nullptr) {
      // Already initialized; return.
      return absl::OkStatus();
    }
    if (!(ds_->MakeIterator(&ds_iterator_)).ok()) {
      return absl::InternalError("DO NOT SUBMIT: some good error message,");
    }
    return absl::OkStatus();
  }
  DatasetIterator() = delete;
  tensorflow::data::standalone::Dataset* ds_;
  v0::Type element_type_;
  std::unique_ptr<tensorflow::data::standalone::Iterator> ds_iterator_ =
      nullptr;
  std::shared_ptr<Executor> target_executor_;
};

class SequenceIterator;

using IteratorFactory =
    std::function<absl::StatusOr<std::shared_ptr<SequenceIterator>>()>;

using SequenceVariant =
    absl::variant<std::shared_ptr<v0::Value>, IteratorFactory>;

// Internal interface representing an iterable value embedded
// in the sequence executor.
class Sequence {
 public:
  enum class SequenceValueType { ITERATOR_FACTORY, VALUE };
  explicit Sequence(SequenceVariant&& value, std::shared_ptr<Executor> executor)
      : value_(std::move(value)), executor_(executor) {}

  inline SequenceValueType type() const {
    if (absl::holds_alternative<std::shared_ptr<v0::Value>>(value_)) {
      return SequenceValueType::VALUE;
    } else {
      return SequenceValueType::ITERATOR_FACTORY;
    }
  }

  inline std::shared_ptr<v0::Value> proto() {
    return absl::get<std::shared_ptr<v0::Value>>(value_);
  }

  absl::StatusOr<std::shared_ptr<SequenceIterator>> CreateIterator() {
    if (type() == SequenceValueType::VALUE) {
      if (ds_ == nullptr) {
        ds_ = TFF_TRY(SequenceValueToDataset(proto()->sequence()));
      }
      return std::make_shared<DatasetIterator>(
          ds_.get(), proto()->sequence().element_type(), executor_);
    } else {
      return iterator_factory()();
    }
  }

 private:
  inline IteratorFactory iterator_factory() {
    return absl::get<IteratorFactory>(value_);
  }
  SequenceVariant value_;
  std::unique_ptr<tensorflow::data::standalone::Dataset> ds_;
  std::shared_ptr<Executor> executor_;
};

class SequenceExecutorValue;

using ValueVariant =
    absl::variant<Embedded, SequenceIntrinsic, std::shared_ptr<Sequence>,
                  std::shared_ptr<std::vector<SequenceExecutorValue>>>;

class SequenceExecutorValue {
 public:
  enum class ValueType { EMBEDDED, INTRINSIC, SEQUENCE, STRUCT };
  inline static SequenceExecutorValue CreateEmbedded(Embedded id) {
    return SequenceExecutorValue(id);
  }
  inline static SequenceExecutorValue CreateIntrinsic(
      SequenceIntrinsic intrinsic) {
    return SequenceExecutorValue(std::move(intrinsic));
  }
  inline static SequenceExecutorValue CreateSequence(
      std::shared_ptr<Sequence> sequence) {
    return SequenceExecutorValue(sequence);
  }
  inline static SequenceExecutorValue CreateStruct(
      std::vector<SequenceExecutorValue>&& structure) {
    return SequenceExecutorValue(
        std::make_shared<std::vector<SequenceExecutorValue>>(structure));
  }

  inline ValueType type() const {
    if (absl::holds_alternative<Embedded>(value_)) {
      return ValueType::EMBEDDED;
    } else if (absl::holds_alternative<SequenceIntrinsic>(value_)) {
      return ValueType::INTRINSIC;
    } else if (absl::holds_alternative<std::shared_ptr<Sequence>>(value_)) {
      return ValueType::SEQUENCE;
    } else {
      return ValueType::STRUCT;
    }
  }
  absl::Status CheckTypeForArgument(ValueType expected_type,
                                    absl::string_view fn_name,
                                    absl::optional<int> position_in_struct) {
    if (type() != expected_type) {
      if (position_in_struct.has_value()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Expected value in argument struct position ",
                         position_in_struct.value(), " for function ", fn_name,
                         " to have value type ", expected_type,
                         "; found instead type ", type()));
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected argument for function ", fn_name, " to have value type ",
            expected_type, "; found instead type ", type()));
      }
    }
    return absl::OkStatus();
  }

  inline const Embedded embedded() const { return absl::get<Embedded>(value_); }
  inline const std::shared_ptr<Sequence> sequence_value() const {
    return absl::get<std::shared_ptr<Sequence>>(value_);
  }
  inline const std::shared_ptr<std::vector<SequenceExecutorValue>>
  struct_value() const {
    return absl::get<std::shared_ptr<std::vector<SequenceExecutorValue>>>(
        value_);
  }
  inline SequenceIntrinsic intrinsic() {
    return absl::get<SequenceIntrinsic>(value_);
  }
  explicit SequenceExecutorValue(ValueVariant value)
      : value_(std::move(value)) {}

 private:
  SequenceExecutorValue() = delete;
  ValueVariant value_;
};

absl::Status CheckLenForUseAsArgument(const SequenceExecutorValue& value,
                                      absl::string_view function_name,
                                      size_t len) {
  if (value.type() != SequenceExecutorValue::ValueType::STRUCT) {
    return absl::InvalidArgumentError(
        absl::StrCat(function_name, " expects a struct argument; received a ",
                     value.type(), " instead."));
  }

  if (value.struct_value()->size() != len) {
    return absl::InvalidArgumentError(
        absl::StrCat(function_name, " expected ", len, " arguments, found ",
                     value.struct_value()->size()));
  }
  return absl::OkStatus();
}

// We return futures since pulling elements from a sequence may be slow, and
// otherwise would block.
using ValueFuture = std::shared_future<absl::StatusOr<SequenceExecutorValue>>;

class SequenceExecutor : public ExecutorBase<ValueFuture> {
 public:
  explicit SequenceExecutor(std::shared_ptr<Executor> target_executor)
      : target_executor_(target_executor) {}
  ~SequenceExecutor() override {}

  absl::string_view ExecutorName() final { return "SequenceExecutor"; }

  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final {
    switch (value_pb.value_case()) {
      case v0::Value::kSequence: {
        // We store the sequence value as a proto in the SequenceExecutor for
        // lazy embedding in the lower-level executor if needed, e.g. in
        // response to a CreateCall, or construction of an iterable from this
        // sequence in the sequence executor itself.
        return ReadyFuture(
            SequenceExecutorValue::CreateSequence(std::make_shared<Sequence>(
                std::make_shared<v0::Value>(value_pb), target_executor_)));
      }
      case v0::Value::kComputation: {
        if (value_pb.computation().has_intrinsic()) {
          absl::string_view intrinsic_uri =
              value_pb.computation().intrinsic().uri();
          absl::StatusOr<SequenceIntrinsic> intrinsic_or_status =
              SequenceIntrinsicFromUri(intrinsic_uri);
          if (intrinsic_or_status.ok()) {
            return ReadyFuture(SequenceExecutorValue::CreateIntrinsic(
                SequenceIntrinsic(intrinsic_or_status.value())));
          }
        }
      }
        // We fall-through here to let intrinsics possibly meant for
        // lower-level executors pass through.
        ABSL_FALLTHROUGH_INTENDED;
      default:
        return ReadyFuture(SequenceExecutorValue::CreateEmbedded(
            ShareValueId(TFF_TRY(target_executor_->CreateValue(value_pb)))));
    }
  }

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture function, absl::optional<ValueFuture> argument) final {
    std::function<absl::StatusOr<SequenceExecutorValue>()> thread_fn =
        [function, argument, this]() -> absl::StatusOr<SequenceExecutorValue> {
      auto fn = TFF_TRY(Wait(function));
      if (fn.type() != SequenceExecutorValue::ValueType::INTRINSIC) {
        absl::optional<ValueId> embedded_arg = absl::nullopt;
        if (argument.has_value()) {
          embedded_arg = TFF_TRY(Embed(TFF_TRY(Wait(argument.value()))))->ref();
        }
        return SequenceExecutorValue::CreateEmbedded(ShareValueId(TFF_TRY(
            target_executor_->CreateCall(fn.embedded()->ref(), embedded_arg))));
      }
      // We know we are executing a sequence intrinsic; check the argument
      // has a value.
      if (!argument.has_value()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Must supply an argument when calling a sequence intrinsic; "
            "called intrinsic ",
            SequenceIntrinsicToUri(fn.intrinsic()), " without an argument."));
      }
      auto arg = TFF_TRY(Wait(argument.value()));
      SequenceIntrinsic intrinsic = fn.intrinsic();
      switch (intrinsic) {
        case SequenceIntrinsic::REDUCE: {
          return SequenceExecutorValue::CreateEmbedded(
              TFF_TRY(ReduceSequence(arg)));
        }
        case SequenceIntrinsic::MAP: {
          return SequenceExecutorValue::CreateSequence(
              TFF_TRY(MapSequence(arg)));
        }
      }
    };
    return ThreadRun(thread_fn);
  }

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> members) final {
    std::function<absl::StatusOr<SequenceExecutorValue>(
        std::vector<SequenceExecutorValue> &&)>
        mapping_fn = [executor = this->target_executor_](
                         std::vector<SequenceExecutorValue>&& member_elems) {
          return SequenceExecutorValue::CreateStruct(std::move(member_elems));
        };
    return Map(std::move(members), mapping_fn);
  }

  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture value,
                                              const uint32_t index) final {
    std::function<absl::StatusOr<SequenceExecutorValue>(
        std::vector<SequenceExecutorValue> &&)>
        mapping_fn = [executor = this->target_executor_,
                      index](std::vector<SequenceExecutorValue>&& source_vector)
        -> absl::StatusOr<SequenceExecutorValue> {
      // We know there is exactly one element here since we will construct
      // the future-vector which supplies the argument below.
      SequenceExecutorValue source = source_vector.at(0);

      switch (source.type()) {
        case SequenceExecutorValue::ValueType::EMBEDDED: {
          return SequenceExecutorValue::CreateEmbedded(ShareValueId(TFF_TRY(
              executor->CreateSelection(source.embedded()->ref(), index))));
        }
        case SequenceExecutorValue::ValueType::STRUCT: {
          auto struct_val = source.struct_value();

          if (index < 0 || index >= struct_val->size()) {
            return absl::InvalidArgumentError(
                absl::StrCat("Attempted to select an element out-of-bounds of "
                             "the underlying "
                             "structure; structure is of length ",
                             struct_val->size(),
                             ", but attempted to select element ", index));
          }
          return struct_val->at(index);
        }
        default:
          return absl::InvalidArgumentError(
              "Can only select from embedded or structure values.");
      }
    };
    return Map(std::vector<ValueFuture>({value}), mapping_fn);
  }

  absl::Status Materialize(ValueFuture value, v0::Value* value_pb) final {
    SequenceExecutorValue exec_value = TFF_TRY(Wait(value));
    switch (exec_value.type()) {
      case SequenceExecutorValue::ValueType::EMBEDDED:
      case SequenceExecutorValue::ValueType::SEQUENCE:
      case SequenceExecutorValue::ValueType::STRUCT: {
        Embedded embedded_value = TFF_TRY(Embed(exec_value));
        TFF_TRY(target_executor_->Materialize(embedded_value->ref(), value_pb));
        return absl::OkStatus();
      }
      case SequenceExecutorValue::ValueType::INTRINSIC: {
        return absl::UnimplementedError(
            "Materialization of sequence intrinsics is not supported.");
      }
    }
    return absl::UnimplementedError("Not implemented yet!");
  }

 private:
  absl::StatusOr<std::shared_ptr<OwnedValueId>> Embed(
      SequenceExecutorValue val) {
    switch (val.type()) {
      case SequenceExecutorValue::ValueType::EMBEDDED:
        return val.embedded();
      case SequenceExecutorValue::ValueType::SEQUENCE:
        if (val.sequence_value()->type() ==
            Sequence::SequenceValueType::VALUE) {
          return ShareValueId(TFF_TRY(
              target_executor_->CreateValue(*val.sequence_value()->proto())));
        } else {
          return absl::InvalidArgumentError(
              "Can't embed a sequence which has been processed with a sequence "
              "map.");
        }
      case SequenceExecutorValue::ValueType::STRUCT: {
        auto struct_val = val.struct_value();
        std::vector<Embedded> owned_ids;
        std::vector<ValueId> refs;
        owned_ids.reserve(struct_val->size());
        refs.reserve(struct_val->size());
        for (const SequenceExecutorValue& elem : *struct_val) {
          Embedded embedded_val = TFF_TRY(Embed(elem));
          owned_ids.emplace_back(embedded_val);
          refs.emplace_back(embedded_val->ref());
        }
        return ShareValueId(TFF_TRY(target_executor_->CreateStruct(refs)));
      }
      default:
        return absl::UnimplementedError(
            "Embed only implemented for Embedded, Sequence, and Struct "
            "values.");
    }
  }

  absl::StatusOr<std::shared_ptr<OwnedValueId>> ReduceSequence(
      SequenceExecutorValue arg) {
    TFF_TRY(CheckLenForUseAsArgument(arg, kSequenceReduceUri, 3));
    auto arg_struct = arg.struct_value();
    SequenceExecutorValue sequence_value = arg_struct->at(0);
    SequenceExecutorValue zero_value = arg_struct->at(1);
    SequenceExecutorValue fn_value = arg_struct->at(2);

    TFF_TRY(sequence_value.CheckTypeForArgument(
        SequenceExecutorValue::ValueType::SEQUENCE, kSequenceReduceUri, 0));
    TFF_TRY(zero_value.CheckTypeForArgument(
        SequenceExecutorValue::ValueType::EMBEDDED, kSequenceReduceUri, 1));
    TFF_TRY(fn_value.CheckTypeForArgument(
        SequenceExecutorValue::ValueType::EMBEDDED, kSequenceReduceUri, 2));

    std::shared_ptr<Sequence> sequence = sequence_value.sequence_value();
    Embedded initial_value = zero_value.embedded();
    Embedded reduce_fn = fn_value.embedded();

    std::shared_ptr<SequenceIterator> iterator =
        TFF_TRY(sequence->CreateIterator());
    v0::Value value;
    std::shared_ptr<OwnedValueId> accumulator = initial_value;
    absl::optional<Embedded> embedded_value =
        TFF_TRY(iterator->GetNextEmbedded());
    while (embedded_value.has_value()) {
      OwnedValueId arg_struct = TFF_TRY(target_executor_->CreateStruct(
          {accumulator->ref(), embedded_value.value()->ref()}));
      accumulator = ShareValueId(
          TFF_TRY(target_executor_->CreateCall(reduce_fn->ref(), arg_struct)));
      embedded_value = TFF_TRY(iterator->GetNextEmbedded());
    }
    return accumulator;
  }

  absl::StatusOr<std::shared_ptr<Sequence>> MapSequence(
      SequenceExecutorValue arg) {
    TFF_TRY(CheckLenForUseAsArgument(arg, kSequenceMapUri, 2));
    auto arg_struct = arg.struct_value();
    SequenceExecutorValue fn_value = arg_struct->at(0);
    SequenceExecutorValue sequence_value = arg_struct->at(1);

    TFF_TRY(fn_value.CheckTypeForArgument(
        SequenceExecutorValue::ValueType::EMBEDDED, kSequenceMapUri, 0));
    TFF_TRY(sequence_value.CheckTypeForArgument(
        SequenceExecutorValue::ValueType::SEQUENCE, kSequenceMapUri, 1));

    IteratorFactory iterator_fn = [sequence = sequence_value.sequence_value(),
                                   embedded_fn = fn_value.embedded(),
                                   executor = target_executor_]()
        -> absl::StatusOr<std::shared_ptr<SequenceIterator>> {
      std::shared_ptr<SequenceIterator> iter =
          TFF_TRY(sequence->CreateIterator());
      return std::make_shared<MappedIterator>(executor, std::move(iter),
                                              embedded_fn);
    };

    return std::make_shared<Sequence>(std::move(iterator_fn), target_executor_);
  }
  std::shared_ptr<Executor> target_executor_;
};

}  // namespace

std::shared_ptr<Executor> CreateSequenceExecutor(
    std::shared_ptr<Executor> target_executor) {
  return std::make_unique<SequenceExecutor>(target_executor);
}

}  // namespace tensorflow_federated
