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

#include "tensorflow_federated/cc/core/impl/executors/session_provider.h"

#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow_federated {

// A process level counter for SessionProvider creation. This ensures
// each session provider is creating unique containers for each session
// created in the process. This is necessary because the TesnorFlow
// ResourceManager shares resource objects across Sessions in the same process,
// but we need to ability to clear an individual Session's container after
// each call.
ABSL_CONST_INIT static absl::Mutex function_id_mutex(absl::kConstInit);
uint32_t GetNextFunctionId() {
  absl::MutexLock function_id_lock(&function_id_mutex);
  static uint32_t function_id = 0;
  return function_id++;
}

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

const absl::flat_hash_set<std::string>& GetOpsWithContainerAttr() {
  // Get the list of all ops that have an `attr` named `container`. We assume
  // this will be all ops that construct resources and allow us to configure
  // which container those resources will be placed in. This maybe fragile
  // because an op _might_ not expose the container attr, or might name it
  // something different.
  static auto* op_names = []() -> absl::flat_hash_set<std::string>* {
    std::vector<tensorflow::OpDef> ops;
    tensorflow::OpRegistry::Global()->GetRegisteredOps(&ops);
    auto op_has_container_attr =
        [](const tensorflow::OpDef& op_def_pb) -> bool {
      for (const auto& attr_def_pb : op_def_pb.attr()) {
        if (attr_def_pb.name() == "container") {
          return true;
        }
      }
      return false;
    };
    auto* op_names = new absl::flat_hash_set<std::string>;
    for (const auto& op_def_pb : ops) {
      if (op_has_container_attr(op_def_pb)) {
        op_names->emplace(op_def_pb.name());
      }
    }
    return op_names;
  }();
  return *op_names;
}

// Rewrites the `container` attr of any op that supports, creating a new
// `GraphDef` with altered `NodeDef`s.
tensorflow::GraphDef ReplaceContainers(
    const tensorflow::GraphDef& original_graph, const std::string& container) {
  // Go through the graph and explicitly set the `container` attr on any op that
  // supports a `container` attr.
  const auto& ops_with_container_attrs = GetOpsWithContainerAttr();
  tensorflow::GraphDef graph = original_graph;
  for (tensorflow::NodeDef& node_pb : *graph.mutable_node()) {
    if (ops_with_container_attrs.contains(node_pb.op())) {
      (*node_pb.mutable_attr())["container"].set_s(container);
    }
  }
  for (tensorflow::FunctionDef& function_pb :
       *graph.mutable_library()->mutable_function()) {
    for (tensorflow::NodeDef& node_pb : *function_pb.mutable_node_def()) {
      if (ops_with_container_attrs.contains(node_pb.op())) {
        (*node_pb.mutable_attr())["container"].set_s(container);
      }
    }
  }
  return graph;
}

SessionProvider::SessionProvider(tensorflow::GraphDef&& graph,
                                 absl::optional<uint16_t> max_active_sessions)
    : max_active_sessions_(max_active_sessions),
      active_sessions_(0),
      graph_(graph),
      function_id_(GetNextFunctionId()) {
  maybe_open_cpus_ = std::thread::hardware_concurrency();
}

absl::StatusOr<std::unique_ptr<tensorflow::Session>>
SessionProvider::CreateSession(const std::string& container) {
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(get_session_options()));
  if (session == nullptr) {
    return absl::InternalError("Failed to create TensorFlow session.");
  }
  auto status = session->Create(ReplaceContainers(graph_, container));
  if (!status.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to create graph in session: ", status.error_message()));
  }
  return std::move(session);
}

absl::StatusOr<SessionProvider::SessionWithResourceContainer>
SessionProvider::TakeSession() {
  lock_.LockWhen(
      absl::Condition(this, &SessionProvider::SessionOrCpuAvailable));
  active_sessions_++;
  if (!sessions_.empty()) {
    SessionProvider::SessionWithResourceContainer session(
        std::move(sessions_.back()));
    sessions_.pop_back();
    lock_.Unlock();
    return std::move(session);
  }
  maybe_open_cpus_--;
  // Build a container name based on the number of sessions created so that
  // each session gets its own container.
  std::string container =
      absl::StrCat(function_id_, "/", session_creation_counter_++);
  lock_.Unlock();
  auto session = CreateSession(container);
  lock_.Lock();
  maybe_open_cpus_++;
  lock_.Unlock();
  if (session.ok()) {
    return SessionProvider::SessionWithResourceContainer{
        TFF_TRY(std::move(session)), container};
  } else {
    return session.status();
  }
}

void SessionProvider::ReturnSession(
    SessionProvider::SessionWithResourceContainer&& session) {
  // Clear any previous resources we had in a container for this session
  // before we loan it out for execution again in the future.
  const tensorflow::DeviceMgr* device_mgr;
  tensorflow::Status status = session.session->LocalDeviceManager(&device_mgr);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to get local device manager for session:" << status;
  } else {
    device_mgr->ClearContainers({session.container_name});
  }
  lock_.Lock();
  active_sessions_--;
  sessions_.emplace_back(std::move(session));
  lock_.Unlock();
}

}  // namespace tensorflow_federated
