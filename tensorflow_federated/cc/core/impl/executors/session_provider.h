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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_SESSION_PROVIDER_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_SESSION_PROVIDER_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"

namespace tensorflow_federated {

inline const tensorflow::SessionOptions& get_session_options() {
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
// session to finish rather than adding extra total work. It also serves as a
// location to inject a maximum on the amount of concurrency applied at a
// per-computation level.
//
// This class is intended only to serve as a dependency of the
// TensorFlowExecutor.
class SessionProvider {
 public:
  SessionProvider(tensorflow::GraphDef&& graph,
                  absl::optional<uint16_t> max_active_sessions)
      : max_active_sessions_(max_active_sessions),
        active_sessions_(0),
        graph_(graph) {
    maybe_open_cpus_ = std::thread::hardware_concurrency();
  }

  absl::StatusOr<std::unique_ptr<tensorflow::Session>> TakeSession() {
    lock_.LockWhen(
        absl::Condition(this, &SessionProvider::SessionOrCpuAvailable));
    active_sessions_++;
    if (!sessions_.empty()) {
      std::unique_ptr<tensorflow::Session> session(std::move(sessions_.back()));
      sessions_.pop_back();
      lock_.Unlock();
      return std::move(session);
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
    active_sessions_--;
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

  bool SessionOrCpuAvailable() {
    bool under_active_session_limit =
        !max_active_sessions_.has_value() ||
        active_sessions_ < max_active_sessions_.value();
    return under_active_session_limit &&
           (!sessions_.empty() || maybe_open_cpus_ > 0);
  }

 private:
  absl::StatusOr<std::unique_ptr<tensorflow::Session>> CreateSession() {
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
    return std::move(session);
  }

  // Move-only.
  SessionProvider(SessionProvider&& other) = default;
  SessionProvider& operator=(SessionProvider&& other) = default;
  SessionProvider(const SessionProvider&) = delete;
  SessionProvider& operator=(const SessionProvider&) = delete;

  absl::Mutex lock_;
  std::vector<std::unique_ptr<tensorflow::Session>> sessions_;
  uint16_t maybe_open_cpus_;
  absl::optional<uint16_t> max_active_sessions_;
  uint16_t active_sessions_;
  tensorflow::GraphDef graph_;
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_SESSION_PROVIDER_H_
