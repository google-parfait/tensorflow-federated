/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_federated/cc/core/impl/aggregation/protocol/aggregation_protocol.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/aggregation_protocol_messages.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"

namespace {

namespace py = ::pybind11;

using ::tensorflow_federated::aggregation::AggregationProtocol;

}  // namespace

PYBIND11_MODULE(aggregation_protocol, m) {
  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  auto py_aggregation_protocol =
      py::class_<AggregationProtocol>(m, "AggregationProtocol")
          .def("Start", &AggregationProtocol::Start)
          .def("AddClients", &AggregationProtocol::AddClients)
          .def("ReceiveClientMessage",
               &AggregationProtocol::ReceiveClientMessage)
          // TODO: b/319889173 - Re-enable `absl::Status` use here once the TF
          // pybind11_abseil import issue is resolved.
          .def("CloseClient",
               [](AggregationProtocol* ap, int64_t client_id,
                  const std::string& client_status_msg,
                  int32_t client_status_code) {
                 return ap->CloseClient(
                     client_id,
                     absl::Status(absl::StatusCode(client_status_code),
                                  client_status_msg));
               })
          .def("Complete", &AggregationProtocol::Complete)
          .def("Abort", &AggregationProtocol::Abort)
          .def("GetStatus", &AggregationProtocol::GetStatus)
          .def("GetResult", &AggregationProtocol::GetResult)
          .def("PollServerMessage", &AggregationProtocol::PollServerMessage);
}
