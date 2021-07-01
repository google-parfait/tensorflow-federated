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

#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"

#include <future>  // NOLINT
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/dataset_ops_internal.h"
#include "tensorflow/cc/ops/experimental_dataset_ops_internal.h"
#include "tensorflow/core/framework/graph.proto.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.proto.h"
#include "tensorflow/core/framework/types.proto.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/config.proto.h"
#include "tensorflow/core/protobuf/rewriter_config.proto.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/proto/v0/computation.proto.h"
#include "tensorflow_federated/proto/v0/executor.proto.h"

namespace tensorflow_federated {

namespace {

absl::string_view ErrLogImpl(absl::string_view msg, absl::SourceLocation loc) {
  auto streamer = absl::LogErrorStreamer(std::move(loc));
  streamer.stream() << msg;
  return msg;
}

// Logs an error message to LOG(ERROR), returning the message.
#define ERR_LOG(msg) ErrLogImpl(msg, ABSL_LOC)

const tensorflow::SessionOptions& get_session_options() {
  // Creates default SessionOptions on first call, and re-uses them for all
  // future calls.
  static tensorflow::SessionOptions* session_options = []() {
    tensorflow::SessionOptions* options_pb = new tensorflow::SessionOptions();
    tensorflow::GraphOptions* graph_options_pb =
        options_pb->config.mutable_graph_options();
    // Disable JIT/runtime Grappler. TFF typically has short lived sessions,
    // meaning that the benefit from running grappler is generally not realized
    // and can be very expensive (anecdotally ~30% of CPU time).
    graph_options_pb->mutable_rewrite_options()->set_disable_meta_optimizer(
        true);
    graph_options_pb->mutable_optimizer_options()->set_opt_level(
        tensorflow::OptimizerOptions::L0);
    graph_options_pb->mutable_optimizer_options()->set_global_jit_level(
        tensorflow::OptimizerOptions::OFF);
    // Don't eagerly allocate all GPU memory in each session.
    options_pb->config.mutable_gpu_options()->set_allow_growth(true);
    // Let the session know the graph will not change.
    auto experimental_pb = options_pb->config.mutable_experimental();
    experimental_pb->set_optimize_for_static_graph(true);
    experimental_pb->set_disable_output_partition_graphs(true);
    options_pb->config.set_allow_soft_placement(true);
    return options_pb;
  }();
  return *session_options;
}

// This class acts as a function from graph -> session, caching previously-
// created sessions for later use.
//
// It is intended to limit the number of threads simultaneously calling
// `NewSession` and `Session->create` to the number of hardware CPUs. This
// allows other incoming threads the opportunity to wait for an already-created
// session to finish rather than adding extra total work.
class SessionProvider {
 public:
  SessionProvider(tensorflow::GraphDef&& graph) : graph_(graph) {
    maybe_open_cpus_ = std::thread::hardware_concurrency();
  }

  absl::StatusOr<std::unique_ptr<tensorflow::Session>> TakeSession() {
    lock_.LockWhen(
        absl::Condition(this, &SessionProvider::SessionOrCpuAvailable));
    if (!sessions_.empty()) {
      std::unique_ptr<tensorflow::Session> session(std::move(sessions_.back()));
      sessions_.pop_back();
      lock_.Unlock();
      return session;
    }
    maybe_open_cpus_--;
    lock_.Unlock();
    auto session = CreateSession();
    lock_.Lock();
    maybe_open_cpus_++;
    lock_.Unlock();
    return session;
  }

  void ReturnSession(std::unique_ptr<tensorflow::Session>&& session) {
    lock_.Lock();
    sessions_.emplace_back(std::move(session));
    lock_.Unlock();
  }

  // An RAII container which returns the session to the provider on destruction.
  class SessionRental {
   public:
    SessionRental(std::unique_ptr<tensorflow::Session>&& session,
                  SessionProvider& provider)
        : session_(std::move(session)), provider_(provider) {}

    SessionRental(SessionRental&& other)
        : session_(std::move(other.session_)), provider_(other.provider_) {}

    void ReturnRental() {
      if (session_ != nullptr) {
        provider_.ReturnSession(std::move(session_));
      }
    }

    ~SessionRental() { ReturnRental(); }

    tensorflow::Session* operator->() { return &*session_; }

   private:
    std::unique_ptr<tensorflow::Session> session_;
    SessionProvider& provider_;
  };

  absl::StatusOr<SessionRental> BorrowSession() {
    return SessionRental(TFF_TRY(TakeSession()), *this);
  }

 private:
  bool SessionOrCpuAvailable() {
    return !sessions_.empty() || maybe_open_cpus_ > 0;
  }

  absl::StatusOr<std::unique_ptr<tensorflow::Session>> CreateSession() {
    // TODO(b/192457299) how expensive is this work, really? is it worth the
    // extra work to cache it and limit it to num_cpus? If not, we could get rid
    // of the whole `SessionProvider` class.
    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(get_session_options()));
    if (session == nullptr) {
      return absl::InternalError("Failed to create TensorFlow session.");
    }
    auto status = session->Create(graph_);
    if (!status.ok()) {
      return absl::InternalError(absl::StrCat(
          "Failed to create graph in session: ", status.error_message()));
    }
    return session;
  }

  // Move-only.
  SessionProvider(SessionProvider&& other) = default;
  SessionProvider& operator=(SessionProvider&& other) = default;
  SessionProvider(const SessionProvider&) = delete;
  SessionProvider& operator=(const SessionProvider&) = delete;

  absl::Mutex lock_;
  std::vector<std::unique_ptr<tensorflow::Session>> sessions_;
  uint16_t maybe_open_cpus_;
  tensorflow::GraphDef graph_;
};

class ExecutorValue;

constexpr char kPlaceholderOp[] = "Placeholder";
constexpr char kDatasetToGraphOp[] = "DatasetToGraphV2";
constexpr char kDatasetFromGraphOp[] = "DatasetFromGraph";

absl::string_view GetNodeName(absl::string_view tensor_name) {
  absl::string_view::size_type pos = tensor_name.find(':');
  if (pos == absl::string_view::npos) {
    return tensor_name;
  } else {
    return tensor_name.substr(0, pos);
  }
}

struct NamesForBindingRewrite {
  // The variant dataset node name without tensor output identifier (no ":N"
  // suffix).
  std::string variant_node_name;
  // The graph serialziation/deserialization node name without tensor output
  // identifier (no ":N" suffix).
  std::string graph_def_node_name;
  // The graph tensor name with a tensor output identifier (a ":N" suffix).
  std::string graph_def_tensor_name;
};

// Computes the names for the nodes and tensors we will add to the graph when
// wrapping sequence bindings in datset serialization ops.
NamesForBindingRewrite GetVariantTensorNodeNameAndReplacement(
    absl::string_view variant_tensor_name,
    absl::string_view replace_node_suffix) {
  NamesForBindingRewrite names;
  names.variant_node_name = GetNodeName(variant_tensor_name);
  names.graph_def_node_name =
      absl::StrCat(names.variant_node_name, "/", replace_node_suffix);
  names.graph_def_tensor_name = absl::StrCat(names.graph_def_node_name, ":0");
  return names;
}

// Given a GraphDef and a tensor binding, replace sequence bindings that use the
// variant_tensor_name binding with a new binding that uses `DatasetFromGraph`
// ops to deserialize a serialized GraphDef proto into the Dataset's variant
// tensor. This is necessary to avoid isues with stateful datasets used across
// sessions.
//
// Example:
//
//   ┌────────────────────────────┐
//   │placeholder (variant tensor)│
//   └┬───────────────────────────┘
//   ┌▽────────┐
//   │dependent│
//   └─────────┘
//
// Becomes:
//
//   ┌───────────────────────────┐
//   │placeholder (string tensor)│
//   └┬──────────────────────────┘
//   ┌▽───────────────┐
//   │DatasetFromGraph│
//   └┬───────────────┘
//   ┌▽────────┐
//   │dependent│
//   └─────────┘
//
// This is used on parameter bindings of `v0::TensorFlow` computations. This is
// the reseverse of `AddSerializationOpsForResults`, which is used on the result
// bindings of the function.
absl::Status AddDeserializationOpsForParameters(
    tensorflow::GraphDef& graphdef_pb, v0::TensorFlow::Binding& binding) {
  switch (binding.binding_case()) {
    case v0::TensorFlow::Binding::kSequence: {
      // Get the name of the placeholder we're operating on and create a name
      // for the new deserialization op.
      const std::string& variant_tensor_name =
          binding.sequence().variant_tensor_name();
      auto graph_names = GetVariantTensorNodeNameAndReplacement(
          variant_tensor_name, kDatasetFromGraphOp);
      for (tensorflow::NodeDef& node_pb : *graphdef_pb.mutable_node()) {
        // Change the placeholder op from variant to string, this will now
        // be a placeholder for a serialized graphdef bytes.
        if (node_pb.name() == graph_names.variant_node_name) {
          if (node_pb.op() != kPlaceholderOp) {
            return absl::InvalidArgumentError(absl::StrCat(
                "Computation used a variant tensor binding for a sequence that "
                "was not a Placeholder. Need to wrap this binding, but unknown "
                "how to proceed with non-Placeholder nodes. Got op [",
                node_pb.op(), "] from node [", node_pb.name(), "]"));
          }
          (*node_pb.mutable_attr())["dtype"].set_type(tensorflow::DT_STRING);
          continue;
        }
        // Update any op that depended on the placeholder to depend on the new
        // DatasetFromGraph op that we will add at the end.
        std::transform(
            node_pb.mutable_input()->begin(), node_pb.mutable_input()->end(),
            node_pb.mutable_input()->begin(),
            [&variant_tensor_name,
             &graph_names](const std::string& input_name) -> std::string {
              if (input_name == variant_tensor_name) {
                return graph_names.graph_def_tensor_name;
              } else if (input_name == graph_names.variant_node_name) {
                return graph_names.graph_def_node_name;
              } else {
                return input_name;
              }
            });
      }
      tensorflow::NodeDef* graph_from_dataset_node = graphdef_pb.add_node();
      graph_from_dataset_node->set_name(graph_names.graph_def_node_name);
      graph_from_dataset_node->set_op(kDatasetFromGraphOp);
      graph_from_dataset_node->add_input()->assign(variant_tensor_name);
      // Update the binding to inform the new format, but continue to use
      // the placeholder node that was originally created.
      binding.mutable_sequence()->Clear();
      binding.mutable_sequence()->set_graph_def_tensor_name(
          variant_tensor_name);
      return absl::OkStatus();
    }
    case v0::TensorFlow::Binding::kStruct: {
      for (auto& member : *binding.mutable_struct_()->mutable_element()) {
        TFF_TRY(AddDeserializationOpsForParameters(graphdef_pb, member));
      }
      return absl::OkStatus();
    }
    default: {
      // Do nothing for non-Sequence values. This typically should be
      // a Tensor value.
      return absl::OkStatus();
    }
  }
}

// Given a GraphDef and a tensor binding, replace sequences that use the
// variant_tensor_name binding with a new binding that uses `DatasetToGraphV2`
// ops to serialize the Dataset's variant tensors. This is necessary to avoid
// isues with stateful datasets used across sessions.
//
// Example:
//
//   ┌────────┐
//   │Identity│
//   └┬───────┘
//   ┌▽─────────────────────┐
//   │dataset variant tensor│
//   └──────────────────────┘
//
// Becomes:
//
//   ┌────────┐
//   │Identity│
//   └┬───────┘
//   ┌▽───────────────┐
//   │DatasetToGraphV2│
//   └┬───────────────┘
//   ┌▽─────────────────────┐
//   │dataset variant tensor│
//   └──────────────────────┘
//
// This is used on result bindings of `v0::TensorFlow` computations. This is
// the reseverse of `AddDeserializationOpsForParameters`, which is used on the
// parameter bindings of the function.
absl::Status AddSerializationOpsForResults(tensorflow::GraphDef& graphdef_pb,
                                           v0::TensorFlow::Binding& binding) {
  switch (binding.binding_case()) {
    case v0::TensorFlow::Binding::kSequence: {
      if (binding.sequence().binding_case() ==
          v0::TensorFlow::SequenceBinding::kGraphDefTensorName) {
        // Already using the correct binding, simply return.
        return absl::OkStatus();
      }
      // Otherwise we need to wrap this binding with one that first calls the
      // DatasetToGraphV2 op on the variant tensor coming out of the graph.
      const std::string& variant_tensor_name =
          binding.sequence().variant_tensor_name();
      auto graph_names = GetVariantTensorNodeNameAndReplacement(
          variant_tensor_name, kDatasetToGraphOp);
      // We only need to add a new node to the graph and update the binding,
      // since we'll depend on what is already in the graph.
      tensorflow::NodeDef* graph_from_dataset_node = graphdef_pb.add_node();
      graph_from_dataset_node->set_name(graph_names.graph_def_node_name);
      graph_from_dataset_node->set_op(kDatasetToGraphOp);
      graph_from_dataset_node->add_input()->assign(variant_tensor_name);
      // Set the default ATTRS.
      // external_state_policy == 0 warns when state will be lost. We expect
      // the state (shuffle buffers, etc) to be lost, but it's nice to continue
      // logging when such an event occurs.
      (*graph_from_dataset_node->mutable_attr())["external_state_policy"].set_i(
          0);
      // We strip the device placement, we want the session we're about to enter
      // to provide the device placement.
      (*graph_from_dataset_node->mutable_attr())["strip_device_assignment"]
          .set_b(true);
      // Update the binding to inform the new format.
      binding.mutable_sequence()->Clear();
      binding.mutable_sequence()->set_graph_def_tensor_name(
          graph_names.graph_def_tensor_name);
      return absl::OkStatus();
    }
    case v0::TensorFlow::Binding::kStruct: {
      for (auto& member : *binding.mutable_struct_()->mutable_element()) {
        TFF_TRY(AddSerializationOpsForResults(graphdef_pb, member));
      }
      return absl::OkStatus();
    }
    default: {
      // Do nothing for non-Sequence values. This typically should be
      // a Tensor value.
      return absl::OkStatus();
    }
  }
}

absl::Status AddDatastSerializationToSequenceBindings(
    tensorflow::GraphDef& graphdef_pb,
    std::optional<v0::TensorFlow::Binding>& parameter_binding,
    v0::TensorFlow::Binding& result_binding) {
  if (parameter_binding != std::nullopt) {
    TFF_TRY(AddDeserializationOpsForParameters(graphdef_pb,
                                               parameter_binding.value()));
  }
  TFF_TRY(AddSerializationOpsForResults(graphdef_pb, result_binding));
  return absl::OkStatus();
}

// A `Computation` is a TensorFlow function consisting of a graph to execute
// as well as a set of labeled tensor inputs and outputs.
class Computation {
 public:
  static absl::StatusOr<std::shared_ptr<Computation>> FromProto(
      const v0::TensorFlow& comp_pb) {
    tensorflow::GraphDef graphdef_pb;
    if (!comp_pb.graph_def().UnpackTo(&graphdef_pb)) {
      return absl::InternalError(ERR_LOG("Could not unpack graphdef proto"));
    }
    std::optional<v0::TensorFlow::Binding> parameter_shape;
    if (comp_pb.has_parameter()) {
      parameter_shape = comp_pb.parameter();
    }
    v0::TensorFlow::Binding result_shape = comp_pb.result();
    TFF_TRY(AddDatastSerializationToSequenceBindings(
        graphdef_pb, parameter_shape, result_shape));
    std::vector<std::string> output_tensor_names;
    TFF_TRY(TensorNamesFromBinding(result_shape, &output_tensor_names));
    return std::make_shared<Computation>(
        std::move(graphdef_pb), comp_pb.initialize_op(),
        std::move(parameter_shape), comp_pb.result(),
        std::move(output_tensor_names));
  }

  absl::StatusOr<ExecutorValue> Call(std::optional<ExecutorValue> arg);

  Computation(tensorflow::GraphDef graph, std::string init_op,
              std::optional<v0::TensorFlow::Binding> parameter_shape,
              v0::TensorFlow::Binding output_shape,
              std::vector<std::string> output_tensor_names)
      : session_provider_(std::move(graph)),
        init_op_(std::move(init_op)),
        parameter_shape_(std::move(parameter_shape)),
        output_shape_(std::move(output_shape)),
        output_tensor_names_(std::move(output_tensor_names)) {}

 private:
  static absl::Status TensorNamesFromBinding(
      const v0::TensorFlow::Binding& binding,
      std::vector<std::string>* tensor_names) {
    switch (binding.binding_case()) {
      case v0::TensorFlow::Binding::kTensor: {
        tensor_names->push_back(binding.tensor().tensor_name());
        return absl::OkStatus();
      }
      case v0::TensorFlow::Binding::kStruct: {
        for (const auto& member : binding.struct_().element()) {
          TFF_TRY(TensorNamesFromBinding(member, tensor_names));
        }
        return absl::OkStatus();
      }
      case v0::TensorFlow::Binding::kSequence: {
        tensor_names->push_back(binding.sequence().graph_def_tensor_name());
        return absl::OkStatus();
      }
      default: {
        return absl::UnimplementedError(
            absl::StrCat("Cannot parse binding type ", binding.binding_case()));
      }
    }
  }

  // Move-only.
  Computation(Computation&& other) = default;
  Computation& operator=(Computation&& other) = default;
  Computation(const Computation&) = delete;
  Computation& operator=(const Computation&) = delete;

  SessionProvider session_provider_;
  std::string init_op_;
  std::optional<v0::TensorFlow::Binding> parameter_shape_;
  v0::TensorFlow::Binding output_shape_;
  std::vector<std::string> output_tensor_names_;
};

// A tensor that holds sequence data.
class SequenceTensor {
 public:
  explicit SequenceTensor(tensorflow::Tensor&& tensor)
      : tensor_(std::move(tensor)) {}
  const tensorflow::Tensor& as_tensor() const { return tensor_; }

 private:
  tensorflow::Tensor tensor_;
};

// Representation for values inside the TensorFlow Executor.
class ExecutorValue {
 public:
  // Whether a given `ExecutorValue` is a structure or a single tensor.
  // TODO(b/192457495): add more cases here as necessary (for computations,
  // sequences, and more...).
  enum class ValueType { TENSOR, STRUCT, COMPUTATION, SEQUENCE };

  // Constructs an `ExecutorValue` from a `tensorflow::Tensor`.
  // NOTE: `tensorflow::Tensor` is internally refcounted, so copies of it are
  // inexpensive.
  explicit ExecutorValue(const tensorflow::Tensor t) : value_(std::move(t)) {}

  // Constructs an `ExecutorValue` from a `SequenceTensor`.
  explicit ExecutorValue(const SequenceTensor st) : value_(std::move(st)) {}

  // Constructs a structural `ExecutorValue from a list of elements.
  explicit ExecutorValue(std::shared_ptr<std::vector<ExecutorValue>> elements)
      : value_(elements) {}

  // Constructs a functional `ExecutorValue` from a TensorFlow computation.
  explicit ExecutorValue(std::shared_ptr<Computation> computation)
      : value_(computation) {}

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
    if (std::holds_alternative<tensorflow::Tensor>(value_)) {
      return ValueType::TENSOR;
    } else if (std::holds_alternative<std::shared_ptr<Computation>>(value_)) {
      return ValueType::COMPUTATION;
    } else if (std::holds_alternative<SequenceTensor>(value_)) {
      return ValueType::SEQUENCE;
    } else {
      return ValueType::STRUCT;
    }
  }

  // Returns a reference to the inner tensor.
  // Requires that `type()` is `ValueType::TENSOR`.
  const tensorflow::Tensor& tensor() const {
    return std::get<tensorflow::Tensor>(value_);
  }

  // Returns a reference to the inner elements list.
  // Requires that `type()` is `ValueType::STRUCT`.
  absl::Span<const ExecutorValue> elements() const {
    return *std::get<std::shared_ptr<std::vector<ExecutorValue>>>(value_);
  }

  const std::shared_ptr<Computation>& computation() const {
    return std::get<std::shared_ptr<Computation>>(value_);
  }

  const tensorflow::Tensor& sequence() const {
    return std::get<SequenceTensor>(value_).as_tensor();
  }

  const absl::Status Bind(
      const v0::TensorFlow::Binding& shape,
      std::vector<std::pair<std::string, tensorflow::Tensor>>* bindings) const {
    switch (type()) {
      case ValueType::TENSOR: {
        if (!shape.has_tensor()) {
          return BindKindMismatch("tensor", shape);
        }
        bindings->emplace_back(shape.tensor().tensor_name(), tensor());
        return absl::OkStatus();
      }
      case ValueType::STRUCT: {
        if (!shape.has_struct_()) {
          return BindKindMismatch("struct", shape);
        }
        if (shape.struct_().element_size() != elements().size()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Attempted to bind struct with ", elements().size(),
                           " fields to an argument struct with ",
                           shape.struct_().element_size(), " fields."));
        }
        for (int i = 0; i < elements().size(); i++) {
          TFF_TRY(elements()[i].Bind(shape.struct_().element(i), bindings));
        }
        return absl::OkStatus();
      }
      case ValueType::COMPUTATION: {
        return absl::InvalidArgumentError(
            "Attempted to bind computation value as argument to a TensorFlow "
            "computation. This is not supported.");
      }
      case ValueType::SEQUENCE: {
        if (!shape.has_sequence()) {
          return BindKindMismatch("sequence", shape);
        }
        bindings->emplace_back(shape.sequence().graph_def_tensor_name(),
                               sequence());
        return absl::OkStatus();
      }
    }
  }

  static absl::StatusOr<ExecutorValue> FromTensorsAndBindingStructure(
      const v0::TensorFlow::Binding& binding_structure,
      absl::Span<tensorflow::Tensor>* tensors) {
    bool is_sequence = false;
    switch (binding_structure.binding_case()) {
      case v0::TensorFlow::Binding::kSequence: {
        is_sequence = true;
      }
        TF_FALLTHROUGH_INTENDED;
      case v0::TensorFlow::Binding::kTensor: {
        if (tensors->empty()) {
          return absl::InternalError(
              "TensorFlow computation had fewer output tensors than expected.");
        }
        tensorflow::Tensor& tensor = tensors->front();
        tensors->remove_prefix(1);
        if (is_sequence) {
          return ExecutorValue(SequenceTensor(std::move(tensor)));
        } else {
          return ExecutorValue(std::move(tensor));
        }
      }
      case v0::TensorFlow::Binding::kStruct: {
        auto elements = std::make_shared<std::vector<ExecutorValue>>();
        elements->reserve(binding_structure.struct_().element_size());
        for (const auto& e_structure : binding_structure.struct_().element()) {
          elements->push_back(
              TFF_TRY(FromTensorsAndBindingStructure(e_structure, tensors)));
        }
        return ExecutorValue(elements);
      }
      default: {
        return absl::UnimplementedError(absl::StrCat(
            "Unknown output binding kind: ", binding_structure.binding_case()));
      }
    }
  }

 private:
  ExecutorValue() = delete;

  std::variant<tensorflow::Tensor, SequenceTensor, std::shared_ptr<Computation>,
               std::shared_ptr<std::vector<ExecutorValue>>>
      value_;

  static absl::Status BindKindMismatch(const absl::string_view value_kind,
                                       const v0::TensorFlow::Binding& shape) {
    return absl::InvalidArgumentError(
        absl::StrCat("Attempted to bind ", value_kind,
                     " value to argument of kind ", shape.SerializeAsString()));
  }
};

absl::StatusOr<ExecutorValue> Computation::Call(
    std::optional<ExecutorValue> arg) {
  // Skip everything if there are no outputs.
  // If `output_tensor_names` is empty, TF raises an error, so we must bypass it
  // entirely.
  if (output_tensor_names_.empty()) {
    return ExecutorValue::FromTensorsAndBindingStructure(output_shape_, {});
  }
  auto session = TFF_TRY(this->session_provider_.BorrowSession());
  if (!init_op_.empty()) {
    tensorflow::Status status = session->Run(
        /*inputs=*/{},
        /*output_tensor_names=*/{},
        /*target_tensor_names=*/{init_op_},
        /*outputs=*/nullptr);
    if (!status.ok()) {
      return absl::InternalError(ERR_LOG(absl::StrCat(
          "Failed to initialize the computation: ", status.error_message())));
    }
  }
  if (arg.has_value() != parameter_shape_.has_value()) {
    auto actual = arg.has_value() ? "was" : "wasn't";
    auto expected = parameter_shape_.has_value() ? "was" : "wasn't";
    return absl::InvalidArgumentError(
        absl::StrCat("Argument ", actual,
                     " provided to tensorflow computation, but an argument ",
                     expected, " expected."));
  }
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  if (arg.has_value()) {
    TFF_TRY(arg.value().Bind(parameter_shape_.value(), &inputs));
  }
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status status =
      session->Run(inputs, output_tensor_names_,
                   /*target_tensor_names=*/{}, &outputs);
  if (!status.ok()) {
    return absl::InternalError(ERR_LOG(
        absl::StrCat("Failed to run computation: ", status.error_message())));
  }
  // Return the session rental before computing the final ExecutorValue.
  session.ReturnRental();
  absl::Span<tensorflow::Tensor> slice(outputs);
  return ExecutorValue::FromTensorsAndBindingStructure(output_shape_, &slice);
}

using ValueFuture = std::shared_future<absl::StatusOr<ExecutorValue>>;

absl::Status MaterializeSequence(const tensorflow::Tensor& graph_def_tensor,
                                 v0::Value::Sequence* sequence_value_pb) {
  if ((graph_def_tensor.dtype() != tensorflow::DT_STRING) ||
      graph_def_tensor.shape().dims() != 0) {
    return absl::InternalError(
        absl::StrCat("Materialize sequence produced unexpected output. "
                     "Expected scalar string tensor, received tensor "
                     "with dtype [",
                     graph_def_tensor.dtype(), "] and rank ",
                     graph_def_tensor.shape().dims()));
  }
  *sequence_value_pb->mutable_serialized_graph_def() =
      graph_def_tensor.flat<tensorflow::tstring>()(0);
  return absl::OkStatus();
}

class TensorFlowExecutor : public ExecutorBase<ValueFuture> {
 public:
  TensorFlowExecutor() {}

 private:
  // A hash map of compiler generated TensorFlow function ids to already
  // construction Computation objects.
  absl::flat_hash_map<uint64, std::shared_ptr<Computation>> function_cache_
      ABSL_GUARDED_BY(function_cache_mutex_);
  absl::Mutex function_cache_mutex_;

  absl::StatusOr<ExecutorValue> CreateValueAny(const v0::Value& value_pb) {
    VLOG(2) << "Creating value: " << value_pb.Utf8DebugString();
    switch (value_pb.value_case()) {
      case v0::Value::kComputation: {
        return CreateValueComputation(value_pb.computation());
      }
      case v0::Value::kTensor: {
        return CreateValueTensor(value_pb.tensor());
      }
      case v0::Value::kStruct: {
        return CreateValueStruct(value_pb.struct_());
      }
      case v0::Value::kSequence: {
        return CreateValueSequence(value_pb.sequence());
      }
      default:
        return absl::UnimplementedError(
            absl::StrCat("Unknown value proto type ", value_pb.value_case()));
    }
  }

  absl::StatusOr<ExecutorValue> CreateValueComputation(
      const v0::Computation& comp_pb) {
    if (!comp_pb.has_tensorflow()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "`TensorFlowExecutor::CreateValue` cannot create values for "
          "non-TensorFlow computations. Found computation of type ",
          comp_pb.computation_case()));
    }
    if (!comp_pb.tensorflow().has_id() || comp_pb.tensorflow().id() == 0) {
      // No ID to use for caching, simply create a computation and skip cache
      // logic.
      return ExecutorValue(
          TFF_TRY(Computation::FromProto(comp_pb.tensorflow())));
    }
    const uint64 function_id = comp_pb.tensorflow().id();
    // Try the fast path first, reader locks are much cheaper.
    {
      absl::ReaderMutexLock reader_lock(&function_cache_mutex_);
      auto cache_iter = function_cache_.find(function_id);
      if (cache_iter != function_cache_.end()) {
        VLOG(2) << "Cache hit for function id: " << function_id;
        return ExecutorValue(cache_iter->second);
      }
    }
    // Otherwise build the cached value and insert it into the cache.
    VLOG(2) << "Cache MISS for function id: " << function_id;
    std::shared_ptr<Computation> computation =
        TFF_TRY(Computation::FromProto(comp_pb.tensorflow()));
    {
      absl::WriterMutexLock writer_lock(&function_cache_mutex_);
      auto result = function_cache_.try_emplace(function_id, computation);
      if (!result.second) {
        // Another thread beat us to creating the cache value. We end up
        // throwing away our value here, but this is fine because its cheap.
        computation = result.first->second;
      }
    }
    return ExecutorValue(computation);
  }

  absl::StatusOr<ExecutorValue> CreateValueTensor(
      const google::protobuf::Any& tensor_pb_any) {
    // TODO(b/192457597): There's an extra copy here to go from
    // `Any -> tensorflow::TensorProto -> tensorflow::Tensor`.
    // Ideally we'd store a `tensorflow::TensorProto` directly, but this is hard
    // due to `executor.proto` needing to be defined in OSS, which then means
    // we'd need to pull the tensorflow proto and all its dependencies into our
    // Github repo (or something similar). michaelreneer is probably a good
    // resource for more information here.
    tensorflow::TensorProto tensor_pb;
    if (!tensor_pb_any.UnpackTo(&tensor_pb)) {
      return absl::InvalidArgumentError(
          ERR_LOG("Could not parse `Any` as `tensorflow::TensorProto`."));
    }
    tensorflow::Tensor tensor;
    if (!tensor.FromProto(std::move(tensor_pb))) {
      return absl::InvalidArgumentError(
          ERR_LOG("Could not create `tensorflow::Tensor` from "
                  "`tensorflow::TensorProto`"));
    }
    return ExecutorValue(std::move(tensor));
  }

  absl::StatusOr<ExecutorValue> CreateValueStruct(
      const v0::Value::Struct& struct_pb) {
    auto elements = std::make_shared<std::vector<ExecutorValue>>();
    elements->reserve(struct_pb.element_size());
    for (const v0::Value::Struct::Element& element_pb : struct_pb.element()) {
      elements->push_back(TFF_TRY(CreateValueAny(element_pb.value())));
    }
    return ExecutorValue(std::move(elements));
  }

  absl::StatusOr<ExecutorValue> CreateValueSequence(
      const v0::Value::Sequence& sequence_pb) const {
    return ExecutorValue(
        SequenceTensor(tensorflow::Tensor(sequence_pb.serialized_graph_def())));
  }

  // NOTE: `value` reference must be valid until `tasks.WaitAll` is called.
  absl::Status MaterializeValue(const ExecutorValue& value, v0::Value* value_pb,
                                ParallelTasks& tasks) {
    switch (value.type()) {
      case ExecutorValue::ValueType::TENSOR: {
        tasks.add_task([value, value_pb]() {
          return SerializeTensorValue(value.tensor(), value_pb);
        });
        return absl::OkStatus();
      }
      case ExecutorValue::ValueType::SEQUENCE: {
        tasks.add_task([value, value_pb]() {
          return MaterializeSequence(value.sequence(),
                                     value_pb->mutable_sequence());
        });
        return absl::OkStatus();
      }
      case ExecutorValue::ValueType::STRUCT: {
        v0::Value::Struct* struct_pb = value_pb->mutable_struct_();
        for (const ExecutorValue& element : value.elements()) {
          // NOTE: field names are never returned.
          TFF_TRY(MaterializeValue(
              element, struct_pb->add_element()->mutable_value(), tasks));
        }
        return absl::OkStatus();
      }
      case ExecutorValue::ValueType::COMPUTATION: {
        return absl::InvalidArgumentError(
            "Cannot materialize uncalled computations");
      }
    }
  }

 protected:
  const char* ExecutorName() final { return "TensorFlowExecutor"; }
  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final {
    return ReadyFuture(TFF_TRY(CreateValueAny(value_pb)));
  }
  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture function, std::optional<ValueFuture> argument) final {
    return ThreadRun([function = std::move(function),
                      argument = std::move(
                          argument)]() -> absl::StatusOr<ExecutorValue> {
      ExecutorValue fn = TFF_TRY(Wait(function));
      std::optional<ExecutorValue> arg = std::nullopt;
      if (argument.has_value()) {
        arg = TFF_TRY(Wait(argument.value()));
      }
      if (fn.type() != ExecutorValue::ValueType::COMPUTATION) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected `function` argument to `TensorFlowExecutor::CreateCall` "
            "to be a computation, but found type ",
            fn.type()));
      }
      return fn.computation()->Call(std::move(arg));
    });
  }
  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> elements) final {
    return Map(
        std::move(elements),
        [](std::vector<ExecutorValue>&& elements)
            -> absl::StatusOr<ExecutorValue> {
          return ExecutorValue(std::make_shared<std::vector<ExecutorValue>>(
              std::move(elements)));
        });
  }
  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture value,
                                              const uint32 index) final {
    return Map(std::vector<ValueFuture>({value}),
               [index](std::vector<ExecutorValue>&& values)
                   -> absl::StatusOr<ExecutorValue> {
                 ExecutorValue& value = values[0];
                 if (value.type() != ExecutorValue::ValueType::STRUCT) {
                   return absl::InvalidArgumentError(
                       ERR_LOG("Cannot create selection on non-struct value."));
                 }
                 if (value.elements().size() <= index) {
                   return absl::InvalidArgumentError(ERR_LOG(absl::StrCat(
                       "Attempted to access index ", index, " of a ",
                       value.elements().size(), "-length struct.")));
                 }
                 return ExecutorValue(value.elements()[index]);
               });
  }
  absl::Status Materialize(ValueFuture value_fut, v0::Value* value_pb) {
    ExecutorValue value = TFF_TRY(Wait(std::move(value_fut)));
    ParallelTasks tasks;
    TFF_TRY(MaterializeValue(value, value_pb, tasks));
    TFF_TRY(tasks.WaitAll());
    return absl::OkStatus();
  }
};

}  // namespace

std::shared_ptr<Executor> CreateTensorFlowExecutor() {
  return std::make_shared<TensorFlowExecutor>();
}

}  // namespace tensorflow_federated
