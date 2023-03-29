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
#include "tensorflow_federated/cc/core/impl/executors/eager_computation.h"

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/mlir/python/mlir.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"

namespace tensorflow_federated {

namespace {

// A process level counter for suffixing name of tf_function created from
// GraphDef. This ensures that each tf function registered is unique and does
// not override an existing function def.
ABSL_CONST_INIT static absl::Mutex function_counter_mutex(absl::kConstInit);
int32_t GetNextFunctionId() {
  absl::MutexLock function_id_lock(&function_counter_mutex);
  static int32_t function_id = 0;
  return function_id++;
}

absl::Status AddFunctionDef(const tensorflow::FunctionDef& function,
                            TFE_Context* context, TF_Status* status) {
  if (TFE_ContextHasFunction(context, function.signature().name().c_str())) {
    return absl::OkStatus();
  }
  auto fdef = function.SerializeAsString();
  TFE_ContextAddFunctionDef(context, fdef.data(), fdef.size(), status);
  if (TF_GetCode(status) != TF_OK) {
    return absl::InternalError(absl::StrCat(
        "FunctionDef could not be registered: ", TF_Message(status)));
  }
  return absl::OkStatus();
}

absl::Status RemoveFunctionDef(const tensorflow::FunctionDef& function,
                               TFE_Context* context, TF_Status* status) {
  if (TFE_ContextHasFunction(context, function.signature().name().c_str())) {
    TFE_ContextRemoveFunction(context, function.signature().name().c_str(),
                              status);
    if (TF_GetCode(status) != TF_OK) {
      return absl::InternalError(absl::StrCat(
          "FunctionDef could not be registered: ", TF_Message(status)));
    }
  }
  return absl::OkStatus();
}

// Add a return node to graph for returning outputs. These would be converted to
// ret and output_args fields when converting to FunctionDef.
absl::Status AddReturnNodeToFunction(tensorflow::Graph* graph,
                                     std::vector<tensorflow::Node*> outputs) {
  int index = 0;
  for (auto* output_op : outputs) {
    tensorflow::NodeDefBuilder builder(absl::StrCat("op_output_", index),
                                       "_Retval");
    tensorflow::NodeDef ret_node_def;
    tensorflow::DataType output_type = output_op->output_type(0);

    auto status = builder.Attr("T", output_type)
                      .Attr("index", index++)
                      .Input(output_op->name(), 0, output_type)
                      .Finalize(&ret_node_def, /*consume=*/true);
    if (!status.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to import graph def ", status.ToString()));
    }

    tensorflow::Node* ret_node = graph->AddNode(ret_node_def, &status);

    if (!status.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to convert graph def ", status.ToString()));
    }

    graph->AddEdge(output_op, 0, ret_node, 0);
  }
  return absl::OkStatus();
}

// Add nodes for input arguments with type "_Arg". These would be converted to
// input_args fields in signature when converting to FunctionDef.
absl::Status AddInputArgNodesToFunction(
    tensorflow::Graph* graph, std::vector<tensorflow::Node*> input_args) {
  int index = 0;
  for (auto* n : input_args) {
    // Input arg node is created with the same name as existing node binding.
    tensorflow::NodeDefBuilder builder(n->name(), "_Arg");
    tensorflow::NodeDef arg_node_def;
    tensorflow::DataType input_type = n->attrs().Find("dtype")->type();

    auto status = builder.Attr("T", input_type)
                      .Attr("index", index++)
                      .Finalize(&arg_node_def, /*consume=*/true);
    if (!status.ok()) {
      return absl::InternalError(absl::StrCat(
          "Failed to create input argument node: ", status.ToString()));
    }

    // Replace placeholder input node with the new Arg node. Since, the name of
    // node is same, the graph already has correct out edges.
    graph->AddNode(arg_node_def, &status);
    if (!status.ok()) {
      return absl::InternalError(absl::StrCat(
          "Failed to add input argument node: ", status.ToString()));
    }

    // Remove the placeholder input node.
    // This is a no-op if the placeholder node is not removed, however, for
    // keeping graph tidy, remove the extraneous node.
    graph->RemoveNode(n);
  }

  return absl::OkStatus();
}

void AddControlEdgeForInitOp(tensorflow::Graph* graph,
                             tensorflow::Node* initialize_op) {
  // This function introduces a control edge from Initialization ops inputs
  // to all the nodes in graph with same input. This is to ensure variable
  // initialization ops are run before all nodes reading the variable to get
  // correct ordering of execution.
  //
  // Auto control dependencies introduce a control dependency between 2 ops if
  // they have the same input in the program order.
  // This follows similar logic.
  std::vector<tensorflow::Node*> destinations;
  absl::flat_hash_set<tensorflow::Node*> init_ops;
  for (auto* input : initialize_op->in_nodes()) {
    init_ops.insert(input);
  }
  absl::flat_hash_set<tensorflow::Node*> init_requirements;
  for (auto* node : graph->nodes()) {
    if (init_ops.contains(node)) {
      init_requirements.insert(node);
      for (auto input : node->in_nodes()) {
        init_requirements.insert(input);
      }
    }
  }

  for (auto* dest : graph->nodes()) {
    if (dest == initialize_op || init_ops.contains(dest)) {
      continue;
    }

    // Check if node has same input as initialization op, introduce a control
    // edge.
    for (auto input : dest->in_nodes()) {
      if (init_requirements.contains(input)) {
        destinations.push_back(dest);
      }
    }
  }

  for (auto dest : destinations) {
    graph->AddControlEdge(initialize_op, dest);
  }
}

absl::StatusOr<tensorflow::NodeDef> CreateRelayoutNodeDef(
    const std::string varop_node_name,
    tensorflow::dtensor::Layout varop_node_layout) {
  tensorflow::NodeDefBuilder builder(absl::StrCat(varop_node_name, "Relayout"),
                                     "Relayout");
  builder.Input(varop_node_name, 0, tensorflow::DT_RESOURCE);
  builder.Attr("layout", varop_node_layout.ToString());
  builder.Attr("T", tensorflow::DT_RESOURCE);
  tensorflow::NodeDef relayout_node_def;
  auto tf_status = builder.Finalize(&relayout_node_def);
  if (!tf_status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to create Relayout node.", tf_status.ToString()));
  }
  return relayout_node_def;
}

absl::Status InsertRelayoutNodeForVariable(
    tensorflow::Node* var_node, tensorflow::dtensor::Layout var_node_layout,
    tensorflow::Graph* graph) {
  auto relayout_node_def_or =
      CreateRelayoutNodeDef(var_node->name(), var_node_layout);
  if (!relayout_node_def_or.ok()) {
    return absl::InternalError(relayout_node_def_or.status().ToString());
  }
  tensorflow::Status tf_status;
  auto relayout_node = graph->AddNode(relayout_node_def_or.value(), &tf_status);
  if (!tf_status.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to insert Relayout node in graph.", tf_status.ToString()));
  }
  absl::flat_hash_map<tensorflow::Node*, int> out_nodes_with_dest_indx;

  for (auto* e : var_node->out_edges()) {
    if (!e->IsControlEdge()) {
      // Layouts are not required for control edge.
      out_nodes_with_dest_indx.insert(std::make_pair(e->dst(), e->dst_input()));
    }
  }
  for (auto e : out_nodes_with_dest_indx) {
    tf_status = graph->UpdateEdge(relayout_node, 0, e.first, e.second);

    if (!tf_status.ok()) {
      return absl::InternalError(absl::StrCat(
          "Failed to updated relayout edges ", tf_status.ToString()));
    }
  }
  return absl::OkStatus();
}

absl::Status PopulateBindingNames(
    const v0::TensorFlow::Binding& binding,
    std::vector<std::string>& tensor_names_from_binding) {
  switch (binding.binding_case()) {
    case v0::TensorFlow::Binding::kTensor: {
      if (!binding.tensor().has_tensor_name()) {
        return absl::InternalError("Tensor binding does not have a name.");
      }
      // Binding names have suffix ":N" in them, remove that to look up nodes.
      tensor_names_from_binding.push_back(
          GetNodeName(binding.tensor().tensor_name()));
      break;
    }
    case v0::TensorFlow::Binding::kStruct: {
      for (const auto& b : binding.struct_().element()) {
        TFF_TRY(PopulateBindingNames(b, tensor_names_from_binding));
      }
      break;
    }
    case v0::TensorFlow::Binding::kSequence: {
      return absl::UnimplementedError(
          "Only Struct and Tensor Binding support added");
    }
    default:
      // No Binding
      break;
  }
  return absl::OkStatus();
}

absl::StatusOr<tensorflow::FunctionDef> ConvertToFunctionDef(
    std::string init_op, const tensorflow::GraphDef& graphdef_pb,
    const v0::TensorFlow::Binding& input_binding,
    const v0::TensorFlow::Binding& output_binding,
    std::map<std::string, tensorflow::dtensor::Layout> layout_map) {
  tensorflow::FunctionDef func_def;
  std::vector<bool> visited(graphdef_pb.node_size());
  std::deque<const tensorflow::Node*> queue;

  tensorflow::Node* initialize_op = nullptr;
  auto graph =
      std::make_unique<tensorflow::Graph>(tensorflow::OpRegistry::Global());
  auto status = tensorflow::ImportGraphDef({}, graphdef_pb, graph.get(),
                                           nullptr, nullptr);
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to import graph def ", status.ToString()));
  }

  std::unordered_map<std::string, tensorflow::Node*> node_map;

  std::vector<tensorflow::Node*> outputs;
  std::vector<tensorflow::Node*> input_args;

  std::vector<std::string> input_bindings;
  std::vector<std::string> output_bindings;

  TFF_TRY(PopulateBindingNames(output_binding, output_bindings));
  TFF_TRY(PopulateBindingNames(input_binding, input_bindings));

  for (auto* node : graph->nodes()) {
    if (init_op == node->name()) {
      initialize_op = node;
    }
    node_map.insert(std::make_pair(node->name(), node));
  }

  for (const auto& name : output_bindings) {
    if (node_map.find(name) == node_map.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid output binding. Node name ", name,
                       " is not part of graph."));
    }
    outputs.push_back(node_map[name]);
  }

  for (const auto& name : input_bindings) {
    if (node_map.find(name) == node_map.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid input binding. Node name ", name, " is not part of graph."));
    }
    input_args.push_back(node_map[name]);
  }

  if (initialize_op != nullptr) {
    AddControlEdgeForInitOp(graph.get(), initialize_op);
  }

  if (!outputs.empty()) {
    TFF_TRY(AddReturnNodeToFunction(graph.get(), outputs));
  }

  if (!input_args.empty()) {
    TFF_TRY(AddInputArgNodesToFunction(graph.get(), input_args));
  }

  // Update Node layout for variable
  for (auto& node_layout : layout_map) {
    if (node_map.find(node_layout.first) != node_map.end()) {
      const auto& node = node_map[node_layout.first];
      if (node->IsOp() && node->op_def().name() == "VarHandleOp") {
        TFF_TRY(InsertRelayoutNodeForVariable(node, node_layout.second,
                                              graph.get()));
      } else {
        // Only Variable layouts supported for now, until we discover a
        // requirement for other nodes.
      }
    }
  }
  // NOTE: Check if there is a C++ API to prune the graph based on input
  // and output args.
  std::vector<std::string> output_names(output_bindings.begin(),
                                        output_bindings.end());
  const std::string function_name =
      absl::StrCat("tf_computation_function", GetNextFunctionId());
  status = tensorflow::GraphToFunctionDef(*graph.get(), function_name,
                                          output_names, &func_def);
  if (!status.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to convert graph def to function def: ", status.ToString()));
  }
  VLOG(2) << "Converted func def " << func_def.DebugString();
  return func_def;
}

void UpdateVarHandleOpNodesAsAnonymous(tensorflow::FunctionDef& func_def) {
  for (auto& node : *func_def.mutable_node_def()) {
    if (node.op() == std::string("VarHandleOp")) {
      (*node.mutable_attr())["shared_name"].set_s(
          tensorflow::ResourceHandle::ANONYMOUS_NAME);
    }
  }
}
absl::Status UpdateVariableLayouts(
    tensorflow::FunctionDef& func_def,
    std::map<std::string, tensorflow::dtensor::Layout> layout_map) {
  std::unordered_map<std::string, tensorflow::NodeDef> node_map;
  std::unordered_map<std::string, std::string>
      replace_input_from_var_to_relayout;
  std::vector<tensorflow::NodeDef> relayout_nodes;
  for (const auto& node_def : func_def.node_def()) {
    if (node_def.op() == std::string("VarHandleOp")) {
      if (layout_map.find(node_def.name()) != layout_map.end()) {
        auto node_def_or =
            CreateRelayoutNodeDef(node_def.name(), layout_map[node_def.name()]);
        if (node_def_or.ok()) {
          tensorflow::NodeDef relayout_node = node_def_or.value();
          // NodeDefBuilder does not insert output type and index for the
          // inputs. i.e. input to relayout_def is "VarHandleOp" instead of
          // "VarHandleOp:resource:0".
          // Manually set it here, since with FunctionDef as input, there is no
          // Graph to FunctionDef conversion in this path.
          *relayout_node.mutable_input(0) =
              absl::StrCat(node_def.name(), ":resource:0");
          relayout_nodes.push_back(relayout_node);
          replace_input_from_var_to_relayout[absl::StrCat(node_def.name(),
                                                          ":resource:0")] =
              absl::StrCat(node_def_or.value().name(), ":output:0");
        } else {
          return absl::InternalError(node_def_or.status().ToString());
        }
      }
    } else {
      // Layouts for Other node types than VarHandleOp not supported right
      // now.
    }
  }

  // Change inputs to all ops consuming varHandleOp to Relayout/
  for (auto& node_def : *func_def.mutable_node_def()) {
    for (int i = 0; i < node_def.input_size(); i++) {
      auto input = node_def.input(i);
      if (replace_input_from_var_to_relayout.find(input) !=
          replace_input_from_var_to_relayout.end()) {
        *node_def.mutable_input(i) = replace_input_from_var_to_relayout[input];
      }
    }
  }

  // Insert Relayout Nodes in the graph.
  for (auto& node_def : relayout_nodes) {
    func_def.add_node_def()->Swap(&node_def);
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<EagerComputation> EagerComputation::FromProto(
    const v0::TensorFlowFunction& comp_pb,
    std::map<std::string, tensorflow::dtensor::Layout> layout_map) {
  if (!comp_pb.function_def().Is<tensorflow::FunctionDef>()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported type in function def proto: ",
        comp_pb.function_def().type_url(), ". Only FunctionDef is supported."));
  }
  // If input is FunctionDef, directly register the functionDef for calling
  // later from Call method.
  tensorflow::FunctionDef func_def;
  if (!comp_pb.function_def().UnpackTo(&func_def)) {
    return absl::InternalError("Could not unpack FunctionDef proto");
  }

  TFF_TRY(UpdateVariableLayouts(func_def, layout_map));
  UpdateVarHandleOpNodesAsAnonymous(func_def);
  // Note: Nested function defs should be passed here for second argument.
  return EagerComputation(func_def, {});
}

absl::StatusOr<EagerComputation> EagerComputation::FromProto(
    const v0::TensorFlow& comp_pb,
    std::map<std::string, tensorflow::dtensor::Layout> layout_map) {
  if (!(comp_pb.graph_def().Is<tensorflow::GraphDef>())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported type in Graph def proto: ", comp_pb.graph_def().type_url(),
        ". Only GraphDef is supported."));
  }

  // Convert GraphDef to FunctionDefs to execute via TFE_Execute API.
  tensorflow::GraphDef graphdef_pb;

  if (!comp_pb.graph_def().UnpackTo(&graphdef_pb)) {
    return absl::InternalError("Could not unpack graphdef proto");
  }

  tensorflow::FunctionDef main_func_def = TFF_TRY(
      ConvertToFunctionDef(comp_pb.initialize_op(), graphdef_pb,
                           comp_pb.parameter(), comp_pb.result(), layout_map));

  UpdateVarHandleOpNodesAsAnonymous(main_func_def);
  // Register Function defs present in library, since these may be invoked from
  // nodes in graph def via StatefulPartitionedCall.
  std::vector<tensorflow::FunctionDef> function_defs_to_register;
  for (auto& func_def : *graphdef_pb.mutable_library()->mutable_function()) {
    UpdateVarHandleOpNodesAsAnonymous(func_def);
    function_defs_to_register.push_back(func_def);
  }

  return EagerComputation(main_func_def, function_defs_to_register);
}

EagerComputation::EagerComputation(
    tensorflow::FunctionDef main_function_def,
    std::vector<tensorflow::FunctionDef> function_defs_to_register)
    : main_function_def_(main_function_def),
      function_defs_to_register_(function_defs_to_register) {}

absl::Status EagerComputation::ExecuteFunction(
    TFE_Context* context, std::string func_name,
    std::optional<std::string> device_name, absl::Span<TFE_TensorHandle*> args,
    std::vector<TFE_TensorHandle*>* outputs) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> func_op(
      TFE_NewOp(context, func_name.c_str(), status.get()), TFE_DeleteOp);
  if (TF_GetCode(status.get()) != TF_OK) {
    return absl::InternalError(
        absl::StrCat("Failed to run computation: ", TF_Message(status.get())));
  }

  if (device_name.has_value()) {
    TFE_OpSetDevice(func_op.get(), device_name.value().c_str(), status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      return absl::InternalError(absl::StrCat("Failed to use provided device: ",
                                              TF_Message(status.get())));
    }
  }

  for (auto arg : args) {
    TFE_OpAddInput(func_op.get(), arg, status.get());
  }

  int num_unpacked_results = main_function_def_.signature().output_arg_size();
  // Use malloc to allocate results, since variable length array is not
  // supported.
  TFE_TensorHandle** raw_result = (TFE_TensorHandle**)malloc(
      sizeof(TFE_TensorHandle*) * num_unpacked_results);
  TFE_Execute(func_op.get(), raw_result, &num_unpacked_results, status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    return absl::InternalError(
        absl::StrCat("Failed to run computation: ", TF_Message(status.get())));
  }

  for (int i = 0; i < num_unpacked_results; i++) {
    outputs->push_back(raw_result[i]);
  }
  free(raw_result);
  return absl::OkStatus();
}

absl::Status EagerComputation::RegisterFunctions(TFE_Context* context) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status_ptr(
      TF_NewStatus(), TF_DeleteStatus);

  TFF_TRY(AddFunctionDef(main_function_def_, context, status_ptr.get()));
  for (const auto& func_def : function_defs_to_register_) {
    TFF_TRY(AddFunctionDef(func_def, context, status_ptr.get()));
  }
  return absl::OkStatus();
}

absl::Status EagerComputation::RemoveFunctions(TFE_Context* context) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status_ptr(
      TF_NewStatus(), TF_DeleteStatus);

  TFE_ContextRemoveFunction(
      context, main_function_def_.signature().name().c_str(), status_ptr.get());

  TFF_TRY(RemoveFunctionDef(main_function_def_, context, status_ptr.get()));
  for (const auto& func_def : function_defs_to_register_) {
    TFF_TRY(RemoveFunctionDef(func_def, context, status_ptr.get()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<TFE_TensorHandle*>> EagerComputation::Call(
    TFE_Context* context, std::optional<std::vector<TFE_TensorHandle*>> args,
    std::optional<std::string> device_name) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status_ptr(
      TF_NewStatus(), TF_DeleteStatus);

  TFF_TRY(RegisterFunctions(context));

  std::vector<TFE_TensorHandle*> outputs;
  std::vector<TFE_TensorHandle*> inputs;
  if (args.has_value()) {
    inputs = args.value();
  }

  absl::Span<TFE_TensorHandle*> args_span(inputs);

  TFF_TRY(ExecuteFunction(context, main_function_def_.signature().name(),
                          device_name, args_span, &outputs));

  return outputs;
}

}  // namespace tensorflow_federated
