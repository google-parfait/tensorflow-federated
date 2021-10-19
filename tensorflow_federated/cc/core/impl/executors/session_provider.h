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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"

namespace tensorflow_federated {

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
// Additionally, it rewrites the graphs of each of the session created so that
// ops that create resources create them in isolated containers. When a session
// is returned to the SessionProvider, the container for that session is
// cleared, freeing resources. This is necessary in TFF, which expects stateless
// functions and would run into issues (e.g. re-initialize lookup tables)
// otherwise.
//
// This class is intended only to serve as a dependency of the
// TensorFlowExecutor.
class SessionProvider {
 public:
  SessionProvider(tensorflow::GraphDef&& graph,
                  absl::optional<uint16_t> max_active_sessions);

  class SessionWithResourceContainer {
   public:
    SessionWithResourceContainer(std::unique_ptr<tensorflow::Session> session,
                                 uint32_t function_id, uint16_t session_id)
        : session_(std::move(session)),
          container_name_(absl::StrCat(function_id, "/", session_id)) {
      tensorflow::Status status = session_->LocalDeviceManager(&device_mgr_);
      if (!status.ok()) {
        LOG(FATAL) << "Unable to retrieve device manager for function ["
                   << function_id << "] with session [" << session_id << "]";
      }
    }
    SessionWithResourceContainer(SessionWithResourceContainer&& other) =
        default;

    void ClearResourceContainers() {
      if (device_mgr_ == nullptr) {
        LOG(ERROR) << "Tried to clear the resources of a session that had "
                   << "no device manager.";
      }
      // Clear any previous resources we had in a container for this session
      // before we loan it out for execution again in the future.
      device_mgr_->ClearContainers({container_name_});
    }

    tensorflow::Session* session_ptr() { return session_.get(); }

   private:
    std::unique_ptr<tensorflow::Session> session_;
    const std::string container_name_;
    const tensorflow::DeviceMgr* device_mgr_;
  };

  // An RAII container which returns the session to the provider on destruction.
  class SessionRental {
   public:
    SessionRental(SessionWithResourceContainer&& session,
                  SessionProvider& provider)
        : session_(std::move(session)), provider_(provider) {}

    SessionRental(SessionRental&& other)
        : session_(std::move(other.session_)), provider_(other.provider_) {}

    void ReturnRental() {
      if (session_.session_ptr() != nullptr) {
        provider_.ReturnSession(std::move(session_));
      }
    }

    ~SessionRental() { ReturnRental(); }

    tensorflow::Session* operator->() { return session_.session_ptr(); }

   private:
    SessionWithResourceContainer session_;
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

  absl::StatusOr<SessionWithResourceContainer> TakeSession();
  void ReturnSession(SessionWithResourceContainer&& session);

 private:
  absl::StatusOr<std::unique_ptr<tensorflow::Session>> CreateSession(
      const uint16_t session_id);

  // Move-only.
  SessionProvider(SessionProvider&& other) = default;
  SessionProvider& operator=(SessionProvider&& other) = default;
  SessionProvider(const SessionProvider&) = delete;
  SessionProvider& operator=(const SessionProvider&) = delete;

  absl::Mutex lock_;
  std::vector<SessionWithResourceContainer> sessions_;
  uint16_t maybe_open_cpus_;
  absl::optional<uint16_t> max_active_sessions_;
  uint16_t active_sessions_;
  const tensorflow::GraphDef graph_;
  // A prefix for all containers used by sessions created by this provider.
  const uint32_t function_id_;
  // A running count of the number of sessions created by this provider. This is
  // used for two purposes:
  // - The container name for resources used by this session (combined with
  //   `function_id_` above to uniquely identify a function copy).
  // - The accelerator device to pin this computation on. If a machine has
  //   multiple accelerators, sessions will be pinned to the
  //   `session_creation_counter_ % num_accelerators` device.
  uint16_t session_creation_counter_ ABSL_GUARDED_BY(lock_) = 0;
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_SESSION_PROVIDER_H_
