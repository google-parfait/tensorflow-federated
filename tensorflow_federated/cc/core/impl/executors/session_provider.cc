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
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/public/session_options.h"

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

struct AcceleratorDevices {
  const int16_t num_gpus = 0;
  const int16_t num_tpus = 0;
};

const AcceleratorDevices& GetAcceleratorDevices() {
  static const AcceleratorDevices* accelerator_devices =
      []() -> AcceleratorDevices* {
    int16_t num_gpus = 0;
    int16_t num_tpus = 0;
    std::vector<std::string> devices;
    tensorflow::Status s =
        tensorflow::DeviceFactory::ListAllPhysicalDevices(&devices);
    if (!s.ok()) {
      LOG(ERROR) << "Error detecting physical devices, defaulting to CPU only: "
                 << s;
      return new AcceleratorDevices{0};
    }
    LOG_FIRST_N(INFO, 1) << "Found devices: [" << absl::StrJoin(devices, ",")
                         << "]";
    for (absl::string_view device : devices) {
      std::vector<absl::string_view> device_parts = absl::StrSplit(device, ':');
      if (device_parts.size() != 3) {
        LOG(ERROR) << "Unknown device name format: [" << device << "]";
        continue;
      }
      auto device_type = device_parts[1];
      if (device_type == tensorflow::DEVICE_GPU) {
        LOG_FIRST_N(INFO, 1) << "Found GPU device: [" << device << "]";
        ++num_gpus;
      } else if (device_type == tensorflow::DEVICE_TPU) {
        LOG_FIRST_N(INFO, 1) << "Found TPU device: [" << device << "]";
        ++num_tpus;
      } else {
        LOG_FIRST_N(INFO, 1) << "Skipping device: [" << device << "]";
      }
    }
    return new AcceleratorDevices{num_gpus, num_tpus};
  }();
  return *accelerator_devices;
}

void SetDevice(absl::string_view device, tensorflow::GraphDef* graph_def,
               const char* device_type) {
  for (tensorflow::NodeDef& node_pb : *graph_def->mutable_node()) {
    if (!node_pb.device().empty()) {
      VLOG(5) << "Skipping already placed node [" << node_pb.name() << "] ("
              << node_pb.op() << ") on " << node_pb.device();
      continue;
    } else if (tensorflow::KernelDefAvailable(device_type, node_pb)) {
      VLOG(5) << "Placing node [" << node_pb.name() << "] (" << node_pb.op()
              << ") on device [" << device << "]";
      node_pb.set_device(device.data(), device.size());
    } else {
      VLOG(5) << "Leaving node [" << node_pb.name() << "] alone.";
    }
  }
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
SessionProvider::CreateSession(const uint16_t session_id) {
  const std::string container = absl::StrCat(function_id_, "/", session_id);
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(get_session_options()));
  if (session == nullptr) {
    return absl::InternalError("Failed to create TensorFlow session.");
  }
  tensorflow::GraphDef graph_def = ReplaceContainers(graph_, container);
  const AcceleratorDevices& devices = GetAcceleratorDevices();

  if (devices.num_gpus > 0 && devices.num_tpus > 0) {
    return absl::UnimplementedError(
        "Can't create a TensorFlow session for hardware with both GPUs and "
        "TPUs.");
  }

  if (devices.num_gpus > 0) {
    // If we have GPUs, round robin the session by explicitly setting the
    // `device` attr of the GPU-capable kernels.
    const uint16_t device_id = session_id % devices.num_gpus;
    const std::string& device =
        absl::StrCat("/device:", tensorflow::DEVICE_GPU, ":", device_id);
    VLOG(2) << "Pinning function [" << function_id_ << "] session ["
            << session_id << "] to device [" << device << "]";
    SetDevice(device, &graph_def, tensorflow::DEVICE_GPU);
  }
  if (devices.num_tpus > 0) {
    // If we have TPUs, round robin the session by explicitly setting the
    // `device` attr of the TPU-capable kernels.
    const uint16_t device_id = session_id % devices.num_tpus;
    const std::string& device =
        absl::StrCat("/device:", tensorflow::DEVICE_TPU, ":", device_id);
    VLOG(2) << "Pinning function [" << function_id_ << "] session ["
            << session_id << "] to device [" << device << "]";
    SetDevice(device, &graph_def, tensorflow::DEVICE_TPU);
  }
  auto status = session->Create(graph_def);
  if (!status.ok()) {
    LOG(ERROR) << status;
    for (absl::string_view line :
         absl::StrSplit(graph_def.Utf8DebugString(), '\n')) {
      LOG(ERROR) << line;
    }
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
  // Build a container name based on the number of sessions created so that each
  // session gets its own container.
  const uint16_t session_id = session_creation_counter_++;
  lock_.Unlock();
  auto session = CreateSession(session_id);
  lock_.Lock();
  maybe_open_cpus_++;
  lock_.Unlock();
  if (session.ok()) {
    return SessionProvider::SessionWithResourceContainer{
        TFF_TRY(std::move(session)), function_id_, session_id};
  } else {
    return session.status();
  }
}

void SessionProvider::ReturnSession(
    SessionProvider::SessionWithResourceContainer&& session) {
  session.ClearResourceContainers();
  lock_.Lock();
  active_sessions_--;
  sessions_.emplace_back(std::move(session));
  lock_.Unlock();
}

}  // namespace tensorflow_federated
