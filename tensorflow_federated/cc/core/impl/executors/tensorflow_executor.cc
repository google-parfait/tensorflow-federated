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

#include <algorithm>
#include <cstdint>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/session_provider.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

namespace {

// Logs an error message to LOG(ERROR), returning the message.
#define ERR_LOG(msg)   \
  ([](std::string m) { \
    LOG(ERROR) << m;   \
    return m;          \
  }(msg))

class ExecutorValue;

constexpr char kPlaceholderOp[] = "Placeholder";
constexpr char kDatasetToGraphOp[] = "DatasetToGraphV2";
constexpr char kDatasetFromGraphOp[] = "DatasetFromGraph";

std::string GetNodeName(absl::string_view tensor_name) {
  absl::string_view::size_type pos = tensor_name.find(':');
  if (pos == absl::string_view::npos) {
    return std::string(tensor_name);
  } else {
    return std::string(tensor_name.substr(0, pos));
  }
}

struct NamesForBindingRewrite {
  // The variant dataset node name without tensor output identifier (no ":N"
  // suffix).
  std::string variant_node_name;
  // The graph serialization/deserialization node name without tensor output
  // identifier (no ":N" suffix).
  std::string graph_def_node_name;
  // The graph tensor name with a tensor output identifier (a ":N" suffix).
  std::string graph_def_tensor_name;
};

// Computes the names for the nodes and tensors we will add to the graph when
// wrapping sequence bindings in datset serialization ops.
NamesForBindingRewrite GetVariantTensorNodeNameAndReplacement(
    absl::string_view variant_tensor_name,
    absl::string_view replace_node_suffix, absl::string_view node_prefix) {
  NamesForBindingRewrite names;
  names.variant_node_name = GetNodeName(variant_tensor_name);
  names.graph_def_node_name = absl::StrCat(
      node_prefix, "/", names.variant_node_name, "/", replace_node_suffix);
  names.graph_def_tensor_name = absl::StrCat(names.graph_def_node_name, ":0");
  return names;
}

#ifdef SHARED_TF_FRAMEWORK_OBJECT
// If building with dynamically linked TensorFlow we use the runtime reflection
// of protocol buffers to access fields. Otherwise we have issues with
// interopability between libprotobuf in TFF and libprotobuf in TF.
void AddDatasetFromGraphOp(tensorflow::GraphDef& graphdef_pb,
                           const NamesForBindingRewrite& graph_names,
                           const std::string& dataset_placeholder_node_name) {
  auto* graphdef_reflection = graphdef_pb.GetReflection();
  auto repeated_nodes =
      graphdef_reflection->GetMutableRepeatedFieldRef<::google::protobuf::Message>(
          &graphdef_pb, graphdef_pb.GetDescriptor()->FindFieldByName("node"));
  std::unique_ptr<::google::protobuf::Message> graph_from_dataset_node(
      repeated_nodes.NewMessage());
  auto* node_reflection = graph_from_dataset_node->GetReflection();
  auto* node_descriptor = graph_from_dataset_node->GetDescriptor();
  node_reflection->SetString(graph_from_dataset_node.get(),
                             node_descriptor->FindFieldByName("name"),
                             std::string(graph_names.graph_def_node_name));
  node_reflection->SetString(graph_from_dataset_node.get(),
                             node_descriptor->FindFieldByName("op"),
                             std::string(kDatasetFromGraphOp));
  node_reflection
      ->GetMutableRepeatedFieldRef<std::string>(
          graph_from_dataset_node.get(),
          node_descriptor->FindFieldByName("input"))
      .Add(std::string(dataset_placeholder_node_name));
  repeated_nodes.Add(*graph_from_dataset_node);
}

void AddDatasetToGraphOp(tensorflow::GraphDef& graphdef_pb,
                         const NamesForBindingRewrite& graph_names,
                         const std::string& variant_tensor_name) {
  auto* graphdef_reflection = graphdef_pb.GetReflection();
  auto repeated_nodes =
      graphdef_reflection->GetMutableRepeatedFieldRef<::google::protobuf::Message>(
          &graphdef_pb, graphdef_pb.GetDescriptor()->FindFieldByName("node"));
  std::unique_ptr<::google::protobuf::Message> graph_to_dataset_node(
      repeated_nodes.NewMessage());
  auto* node_reflection = graph_to_dataset_node->GetReflection();
  auto* node_descriptor = graph_to_dataset_node->GetDescriptor();
  node_reflection->SetString(graph_to_dataset_node.get(),
                             node_descriptor->FindFieldByName("name"),
                             std::string(graph_names.graph_def_node_name));
  node_reflection->SetString(graph_to_dataset_node.get(),
                             node_descriptor->FindFieldByName("op"),
                             std::string(kDatasetToGraphOp));
  node_reflection
      ->GetMutableRepeatedFieldRef<std::string>(
          graph_to_dataset_node.get(),
          node_descriptor->FindFieldByName("input"))
      .Add(std::string(variant_tensor_name));
  // TODO(b/220195980): Explicitly set the default ATTRS we get from the Python
  // API.
  //
  // external_state_policy == 0 warns when state will be lost. We expect
  // the state (shuffle buffers, etc) to be lost, but it's nice to continue
  // logging when such an event occurs.
  //
  // With strip_device_assignment == true, we strip the device placement, we
  // want the session we're about to enter to provide the device placement.
  repeated_nodes.Add(*graph_to_dataset_node);
}

#else   // Building with staticly linked TensorFlow.

void AddDatasetFromGraphOp(tensorflow::GraphDef& graphdef_pb,
                           const NamesForBindingRewrite& graph_names,
                           const std::string& dataset_placeholder_node_name) {
  tensorflow::NodeDef* graph_from_dataset_node = graphdef_pb.add_node();
  graph_from_dataset_node->set_name(graph_names.graph_def_node_name);
  graph_from_dataset_node->set_op(kDatasetFromGraphOp);
  graph_from_dataset_node->add_input()->assign(dataset_placeholder_node_name);
}

void AddDatasetToGraphOp(tensorflow::GraphDef& graphdef_pb,
                         const NamesForBindingRewrite& graph_names,
                         const std::string& variant_tensor_name) {
  tensorflow::NodeDef* graph_from_dataset_node = graphdef_pb.add_node();
  graph_from_dataset_node->set_name(graph_names.graph_def_node_name);
  graph_from_dataset_node->set_op(kDatasetToGraphOp);
  graph_from_dataset_node->add_input()->assign(variant_tensor_name);
  // Explicitly set the default ATTRS used in the Python API.
  // external_state_policy == 0 warns when state will be lost. We expect
  // the state (shuffle buffers, etc) to be lost, but it's nice to continue
  // logging when such an event occurs.
  (*graph_from_dataset_node->mutable_attr())["external_state_policy"].set_i(0);
  // We strip the device placement, we want the session we're about to enter
  // to provide the device placement.
  (*graph_from_dataset_node->mutable_attr())["strip_device_assignment"].set_b(
      true);
}
#endif  // SHARED_TF_FRAMEWORK_OBJECT

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
// the reverse of `AddSerializationOpsForResults`, which is used on the result
// bindings of the function.
absl::Status AddDeserializationOpsForParameters(
    tensorflow::GraphDef& graphdef_pb, v0::TensorFlow::Binding& binding,
    absl::string_view prefix = "root") {
  switch (binding.binding_case()) {
    case v0::TensorFlow::Binding::kSequence: {
      // Get a copy of the name of the placeholder we're operating on. We're
      // going to clear/reset the binding and  then rebuild the it but re-use
      // the placholder op.
      const std::string dataset_placeholder_node_name =
          binding.sequence().variant_tensor_name();
      binding.mutable_sequence()->Clear();
      auto graph_names = GetVariantTensorNodeNameAndReplacement(
          dataset_placeholder_node_name, kDatasetFromGraphOp, prefix);
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
            [&dataset_placeholder_node_name,
             &graph_names](const std::string& input_name) -> std::string {
              if (input_name == dataset_placeholder_node_name) {
                return graph_names.graph_def_tensor_name;
              } else if (input_name == graph_names.variant_node_name) {
                return graph_names.graph_def_node_name;
              } else {
                return input_name;
              }
            });
      }
      AddDatasetFromGraphOp(graphdef_pb, graph_names,
                            dataset_placeholder_node_name);
      // Update the binding to inform the new format, but continue to use the
      // placeholder node that was originally created.
      binding.mutable_sequence()->set_graph_def_tensor_name(
          dataset_placeholder_node_name);
      return absl::OkStatus();
    }
    case v0::TensorFlow::Binding::kStruct: {
      for (int i = 0; i < binding.struct_().element_size(); ++i) {
        auto& member = *binding.mutable_struct_()->mutable_element(i);
        TFF_TRY(AddDeserializationOpsForParameters(
            graphdef_pb, member, absl::StrCat(prefix, "/", i)));
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
// issues with stateful datasets used across sessions.
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
                                           v0::TensorFlow::Binding& binding,
                                           absl::string_view prefix = "root") {
  switch (binding.binding_case()) {
    case v0::TensorFlow::Binding::kSequence: {
      if (binding.sequence().binding_case() ==
          v0::TensorFlow::SequenceBinding::kGraphDefTensorName) {
        // Already using the correct binding, simply return.
        return absl::OkStatus();
      }
      // Otherwise we need to wrap this binding with one that first calls the
      // DatasetToGraphV2 op on the variant tensor coming out of the graph.
      // First make a copy of the placeholder name because we are going to reset
      // the binding and rebuild it using the same node nade.
      const std::string variant_tensor_name =
          binding.sequence().variant_tensor_name();
      binding.mutable_sequence()->Clear();
      auto graph_names = GetVariantTensorNodeNameAndReplacement(
          variant_tensor_name, kDatasetToGraphOp, prefix);
      // We only need to add one  new node to the graph and update the binding,
      // since we'll depend on the placeholder already in the graph to feed us
      // the serialized GraphDef bytes.
      AddDatasetToGraphOp(graphdef_pb, graph_names, variant_tensor_name);
      // Update the binding to inform the new format.
      binding.mutable_sequence()->set_graph_def_tensor_name(
          graph_names.graph_def_tensor_name);
      return absl::OkStatus();
    }
    case v0::TensorFlow::Binding::kStruct: {
      for (int i = 0; i < binding.struct_().element_size(); ++i) {
        auto& member = *binding.mutable_struct_()->mutable_element(i);
        TFF_TRY(AddSerializationOpsForResults(graphdef_pb, member,
                                              absl::StrCat(prefix, "/", i)));
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

absl::Status AddDatasetSerializationToSequenceBindings(
    tensorflow::GraphDef& graphdef_pb,
    absl::optional<v0::TensorFlow::Binding>& parameter_binding,
    v0::TensorFlow::Binding& result_binding) {
  if (parameter_binding != absl::nullopt) {
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
      const v0::TensorFlow& comp_pb, absl::optional<int> max_active_sessions) {
    tensorflow::GraphDef graphdef_pb;
    if (!comp_pb.graph_def().UnpackTo(&graphdef_pb)) {
      return absl::InternalError(ERR_LOG("Could not unpack graphdef proto"));
    }
    absl::optional<v0::TensorFlow::Binding> parameter_shape;
    if (comp_pb.has_parameter()) {
      parameter_shape = comp_pb.parameter();
    }
    v0::TensorFlow::Binding result_shape = comp_pb.result();
    TFF_TRY(AddDatasetSerializationToSequenceBindings(
        graphdef_pb, parameter_shape, result_shape));
    std::vector<std::string> output_tensor_names;
    TFF_TRY(TensorNamesFromBinding(result_shape, &output_tensor_names));
    return std::make_shared<Computation>(
        std::move(graphdef_pb), comp_pb.initialize_op(),
        std::move(parameter_shape), comp_pb.result(),
        std::move(output_tensor_names), max_active_sessions);
  }

  absl::StatusOr<ExecutorValue> Call(absl::optional<ExecutorValue> arg);

  Computation(tensorflow::GraphDef graph, std::string init_op,
              absl::optional<v0::TensorFlow::Binding> parameter_shape,
              v0::TensorFlow::Binding output_shape,
              std::vector<std::string> output_tensor_names,
              absl::optional<int> max_active_sessions = absl::nullopt)
      : session_provider_(std::move(graph), max_active_sessions),
        init_op_(std::move(init_op)),
        parameter_shape_(std::move(parameter_shape)),
        output_shape_(std::move(output_shape)),
        output_tensor_names_(std::move(output_tensor_names)) {}

  std::string DebugString() const {
    return absl::StrCat("(",
                        parameter_shape_.has_value()
                            ? parameter_shape_->ShortDebugString()
                            : "",
                        " -> ", output_shape_.ShortDebugString(), ")");
  }

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
  absl::optional<v0::TensorFlow::Binding> parameter_shape_;
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
    if (absl::holds_alternative<tensorflow::Tensor>(value_)) {
      return ValueType::TENSOR;
    } else if (absl::holds_alternative<std::shared_ptr<Computation>>(value_)) {
      return ValueType::COMPUTATION;
    } else if (absl::holds_alternative<SequenceTensor>(value_)) {
      return ValueType::SEQUENCE;
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

  const std::shared_ptr<Computation>& computation() const {
    return absl::get<std::shared_ptr<Computation>>(value_);
  }

  const tensorflow::Tensor& sequence() const {
    return absl::get<SequenceTensor>(value_).as_tensor();
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
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unable to bind unknown value type: ", type()));
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

  std::string DebugString() const {
    if (absl::holds_alternative<tensorflow::Tensor>(value_)) {
      return absl::StrCat(tensorflow::DataTypeString(tensor().dtype()),
                          tensor().shape().DebugString());
    } else if (absl::holds_alternative<std::shared_ptr<Computation>>(value_)) {
      return computation()->DebugString();
    } else if (absl::holds_alternative<SequenceTensor>(value_)) {
      return absl::StrCat(tensorflow::DataTypeString(tensor().dtype()),
                          tensor().shape().DebugString(), "*");
    } else {
      auto element_formatter = [](std::string* out, const ExecutorValue& v) {
        absl::StrAppend(out, v.DebugString());
      };
      return absl::StrCat(
          "<", absl::StrJoin(elements(), ",", element_formatter), ">");
    }
  }

 private:
  ExecutorValue() = delete;

  absl::variant<tensorflow::Tensor, SequenceTensor,
                std::shared_ptr<Computation>,
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
    absl::optional<ExecutorValue> arg) {
  // Skip everything if there are no outputs.
  // If `output_tensor_names` is empty, TF raises an error, so we must bypass it
  // entirely.
  if (output_tensor_names_.empty()) {
    return ExecutorValue::FromTensorsAndBindingStructure(output_shape_, {});
  }
  auto session = TFF_TRY(this->session_provider_.BorrowSession());
  if (arg.has_value() != parameter_shape_.has_value()) {
    auto actual = arg.has_value()
                      ? absl::StrCat("of type '", arg->DebugString(), "' was")
                      : "wasn't";
    auto expected =
        parameter_shape_.has_value()
            ? absl::StrCat("matching binding '",
                           parameter_shape_->ShortDebugString(), "' was")
            : "wasn't";
    return absl::InvalidArgumentError(
        absl::StrCat("Argument ", actual,
                     " provided to tensorflow computation, but an argument ",
                     expected, " expected."));
  }
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
  if (arg.has_value()) {
    TFF_TRY(arg.value().Bind(parameter_shape_.value(), &inputs));
  }
  if (!init_op_.empty()) {
    tensorflow::Status status = session->Run(inputs,
                                             /*output_tensor_names=*/{},
                                             /*target_tensor_names=*/{init_op_},
                                             /*outputs=*/nullptr);
    if (!status.ok()) {
      return absl::InternalError(ERR_LOG(absl::StrCat(
          "Failed to initialize the computation: ", status.error_message())));
    }
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
  explicit TensorFlowExecutor(
      absl::optional<int> max_concurrent_computation_calls)
      : max_concurrent_computation_calls_(max_concurrent_computation_calls) {}

 private:
  // A hash map of compiler generated TensorFlow function ids to already
  // construction Computation objects.
  absl::flat_hash_map<uint64_t, std::shared_ptr<Computation>> function_cache_
      ABSL_GUARDED_BY(function_cache_mutex_);
  absl::Mutex function_cache_mutex_;
  absl::optional<uint16_t> max_concurrent_computation_calls_;

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
    if (!comp_pb.has_tensorflow() && !comp_pb.has_intrinsic()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "`TensorFlowExecutor::CreateValue` can only create values for "
          "TensorFlow computations and intrinsics. Found computation of type ",
          comp_pb.computation_case()));
    }
    if (comp_pb.has_intrinsic()) {
      if (comp_pb.intrinsic().uri() == "args_into_sequence") {
        return absl::UnimplementedError(
            "`federated_select` simulation is not yet supported in the TFF C++ "
            "runtime. For `federated_select` support, consider opting into the "
            "Python runtime using "
            "`tff.backends.native.set_local_python_execution_context()` (or "
            "`tff.google.backends.native.set_borg_execution_context(...)` for "
            "multi-machine uses).");
      }
      return absl::InvalidArgumentError(
          absl::StrCat("TensorFlowExecutor does not support intrinsic ",
                       comp_pb.intrinsic().uri()));
    }
    if (!comp_pb.tensorflow().has_cache_key() ||
        comp_pb.tensorflow().cache_key().id() == 0) {
      // No ID to use for caching, simply create a computation and skip cache
      // logic.
      LOG_FIRST_N(WARNING, 10) << "Skipped caching computation, no cache_key:\n"
                               << comp_pb.type().Utf8DebugString();
      return ExecutorValue(TFF_TRY(Computation::FromProto(
          comp_pb.tensorflow(), max_concurrent_computation_calls_)));
    }
    const uint64_t function_id = comp_pb.tensorflow().cache_key().id();
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
    std::shared_ptr<Computation> computation = TFF_TRY(Computation::FromProto(
        comp_pb.tensorflow(), max_concurrent_computation_calls_));
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
  absl::string_view ExecutorName() final {
    static constexpr absl::string_view kExecutorName = "TensorFlowExecutor";
    return kExecutorName;
  }
  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final {
    return ReadyFuture(TFF_TRY(CreateValueAny(value_pb)));
  }
  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture function, absl::optional<ValueFuture> argument) final {
    return ThreadRun([function = std::move(function),
                      argument = std::move(
                          argument)]() -> absl::StatusOr<ExecutorValue> {
      ExecutorValue fn = TFF_TRY(Wait(function));
      absl::optional<ExecutorValue> arg = absl::nullopt;
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
                                              const uint32_t index) final {
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

std::shared_ptr<Executor> CreateTensorFlowExecutor(
    absl::optional<int> max_concurrent_computation_calls) {
  return std::make_shared<TensorFlowExecutor>(max_concurrent_computation_calls);
}

}  // namespace tensorflow_federated
