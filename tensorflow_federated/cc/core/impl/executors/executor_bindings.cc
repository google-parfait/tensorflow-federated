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

// This file contains the pybind11 defintions for exposing the C++ Executor
// interface in Python.
//
// General principles:
//   - Python methods defined here (e.g. `.def_*()`) should not contain
//     "business logic". That should be implemented on the underlying C++ class.
//     The only logic that may exist here is parameter/result conversions (e.g.
//     `OwnedValueId` -> `ValueId`, etc).

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "grpcpp/grpcpp.h"
#include "pybind11/include/pybind11/detail/common.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/wrapped_proto_caster.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/composing_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/remote_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

namespace py = ::pybind11;

namespace {

////////////////////////////////////////////////////////////////////////////////
// The Python module defintion `executor_bindings`.
//
// This will be used with `import executor_bindings` on the Python side. This
// module should _not_ be directly imported into the public pip API. The methods
// here will raise `NotOkStatus` errors from absl, which are not user friendly.
////////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(executor_bindings, m) {
  py::google::ImportStatusModule();

  m.doc() = "Bindings for the C++ ";

  // Provide an `OwnedValueId` class to handle return values from the
  // `Executor` interface.
  //
  // Note: no `init<>()` method defined, this object is only constructor from
  // Executor instances.
  py::class_<OwnedValueId>(m, "OwnedValueId")
      .def_property_readonly("ref", &OwnedValueId::ref)
      .def("__str__",
           [](const OwnedValueId& self) { return absl::StrCat(self.ref()); })
      .def("__repr__", [](const OwnedValueId& self) {
        return absl::StrCat("<OwnedValueId: ", self.ref(), ">");
      });

  // We provide ComposingChild as an opaque object so that they can be returned
  // from pybind functions.
  py::class_<ComposingChild>(m, "ComposingChild")
      .def("__repr__", [](const ComposingChild& self) {
        return absl::StrCat(
            "<ComposingChild with num clients: ", self.num_clients(), ">");
      });

  // Provide the `Executor` interface.
  //
  // A `dispose` method is purposely not exposed. Though `Executor::Dispose`
  // exists in C++, Python should call `Dispose` via the `OwnedValueId`
  // destructor during garbage collection.
  //
  // Note: no `init<>()` method defined, must be constructed useing the create_*
  // methods defined below.
  py::class_<Executor,
             // PyExecutor trampoline goes here when ready
             std::shared_ptr<Executor>>(m, "Executor")
      .def("create_value",
           pybind11::google::WithWrappedProtos(&Executor::CreateValue),
           py::arg("value_pb"), py::return_value_policy::move)
      .def("create_struct",
           pybind11::google::WithWrappedProtos(&Executor::CreateStruct),
           py::return_value_policy::move)
      .def("create_selection",
           pybind11::google::WithWrappedProtos(&Executor::CreateSelection),
           py::return_value_policy::move)
      .def("create_call",
           pybind11::google::WithWrappedProtos(&Executor::CreateCall),
           py::arg("function"),
           // Allow `argument` to be `None`.
           py::arg("argument").none(true), py::return_value_policy::move)
      .def("materialize",
           pybind11::google::WithWrappedProtos(
               [](Executor& e,
                  const ValueId& value_id) -> absl::StatusOr<v0::Value> {
                 // Construct a new `v0::Value` to write to and return it to
                 // Python.
                 v0::Value value_pb;
                 auto result = e.Materialize(value_id, &value_pb);
                 if (!result.ok()) {
                   return result;
                 }
                 return value_pb;
               }));

  // Executor construction methods.
  m.def("create_tensorflow_executor", &CreateTensorFlowExecutor,
        py::arg("max_concurrent_computation_calls") = py::none(),
        "Creates a TensorFlowExecutor.");
  m.def("create_reference_resolving_executor",
        &CreateReferenceResolvingExecutor,
        "Creates a ReferenceResolvingExecutor", py::arg("inner_executor"));
  m.def("create_federating_executor", &CreateFederatingExecutor,
        py::arg("inner_executor"), py::arg("cardinalities"),
        "Creates a FederatingExecutor.");
  m.def("create_composing_child", &ComposingChild::Make, py::arg("executor"),
        py::arg("cardinalities"), "Creates a ComposingExecutor.");
  m.def("create_composing_executor", &CreateComposingExecutor,
        py::arg("server"), py::arg("children"), "Creates a ComposingExecutor.");
  m.def("create_remote_executor",
        py::overload_cast<std::shared_ptr<grpc::ChannelInterface>,
                          const CardinalityMap&>(&CreateRemoteExecutor),
        py::arg("channel"), py::arg("cardinalities"),
        "Creates a RemoteExecutor.");

  py::class_<grpc::ChannelInterface, std::shared_ptr<grpc::ChannelInterface>>(
      m, "GRPCChannelInterface");

  m.def(
      "create_insecure_grpc_channel",
      [](const std::string& target) -> std::shared_ptr<grpc::ChannelInterface> {
        auto channel_options = grpc::ChannelArguments();
        channel_options.SetMaxSendMessageSize(
            std::numeric_limits<int32_t>::max());
        channel_options.SetMaxReceiveMessageSize(
            std::numeric_limits<int32_t>::max());
        return grpc::CreateCustomChannel(
            target, grpc::InsecureChannelCredentials(), channel_options);
      },
      pybind11::return_value_policy::take_ownership);
}

}  // namespace
}  // namespace tensorflow_federated
