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
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "include/grpcpp/impl/channel_interface.h"
#include "include/grpcpp/security/credentials.h"
#include "include/grpcpp/support/channel_arguments.h"
#include "include/pybind11/cast.h"
#include "include/pybind11/detail/common.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "include/pybind11/stl.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/safe_ptr.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/composing_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/dtensor_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/remote_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/sequence_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/streaming_remote_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/xla_executor.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow {
Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);
}  // namespace tensorflow

namespace tensorflow_federated {

namespace py = ::pybind11;

namespace {

TFE_Context* GetContextHandle(PyObject* py_context) {
  tensorflow::Safe_PyObjectPtr py_context_handle(
      PyObject_GetAttrString(py_context, "_handle"));
  if (py_context_handle == nullptr) {
    // Current Python code makes sure this never happens. If it does, or
    // becomes hard to maintain, we can call the ensure_initialized() method
    // here.
    PyErr_SetString(
        PyExc_TypeError,
        "Expected `context` argument in EagerTensor constructor to have a "
        "`_handle` attribute but it did not. Was eager Context initialized?");
    return nullptr;
  }

  auto* ctx = reinterpret_cast<TFE_Context*>(
      PyCapsule_GetPointer(py_context_handle.get(), nullptr));
  if (ctx == nullptr) {
    PyErr_SetString(PyExc_TypeError,
                    tensorflow::strings::StrCat(
                        "Expected context._handle to contain a PyCapsule "
                        "encoded pointer to TFE_Context. Got ",
                        Py_TYPE(py_context_handle.get())->tp_name)
                        .c_str());
  }
  return ctx;
}

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

  // v0::Value serialization methods.
  m.def("serialize_tensor_value",
        [](const tensorflow::Tensor& tensor) -> absl::StatusOr<v0::Value> {
          v0::Value value_pb;
          TFF_TRY(SerializeTensorValue(tensor, &value_pb));
          return value_pb;
        });
  m.def("deserialize_tensor_value", &DeserializeTensorValue);

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
      .def("create_value", &Executor::CreateValue, py::arg("value_pb"),
           py::return_value_policy::move,
           py::call_guard<py::gil_scoped_release>())
      .def("create_struct", &Executor::CreateStruct,
           py::return_value_policy::move,
           py::call_guard<py::gil_scoped_release>())
      .def("create_selection", &Executor::CreateSelection,
           py::return_value_policy::move,
           py::call_guard<py::gil_scoped_release>())
      .def("create_call", &Executor::CreateCall, py::arg("function"),
           // Allow `argument` to be `None`.
           py::arg("argument").none(true), py::return_value_policy::move,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "materialize",
          [](Executor& e,
             const ValueId& value_id) -> absl::StatusOr<v0::Value> {
            // Construct a new `v0::Value` to write to and return it to Python.
            v0::Value value_pb;
            absl::Status result = e.Materialize(value_id, &value_pb);
            if (!result.ok()) {
              return result;
            }
            return value_pb;
          },
          py::call_guard<py::gil_scoped_release>());

  // Executor construction methods.
  m.def("create_tensorflow_executor", &CreateTensorFlowExecutor,
        py::arg("max_concurrent_computation_calls") = -1,
        "Creates a TensorFlowExecutor.");
  m.def(
      "create_dtensor_executor",
      [](const std::string& device_name = "", std::string serialized_mesh = "",
         int max_concurrent_computation_calls =
             -1) -> absl::StatusOr<std::shared_ptr<Executor>> {
        PyObject* context = GetPyEagerContext();
        std::optional<tensorflow::dtensor::Mesh> mesh_opt = std::nullopt;
        if (!serialized_mesh.empty()) {
          auto mesh = tensorflow::dtensor::Mesh::FromString(serialized_mesh);
          if (!mesh.ok()) {
            return absl::InvalidArgumentError(mesh.status().ToString());
          }
          mesh_opt = mesh.value();
        }

        auto executor = CreateDTensorExecutor(
            /*context=*/GetContextHandle(context),
            /*dtensor_device_name=*/device_name.empty()
                ? std::nullopt
                : std::optional<std::string>(device_name),
            /*mesh=*/mesh_opt,
            /*dtensor_converter=*/nullptr,  // Use default converter.
            /*max_concurrent_computation_calls=*/
            max_concurrent_computation_calls);
        return executor;
      },
      "Creates a DTensorExecutor.");
  m.def("create_reference_resolving_executor",
        &CreateReferenceResolvingExecutor,
        "Creates a ReferenceResolvingExecutor", py::arg("inner_executor"));
  m.def("create_federating_executor", &CreateFederatingExecutor,
        py::arg("inner_server_executor"), py::arg("inner_client_executor"),
        py::arg("cardinalities"), "Creates a FederatingExecutor.");
  m.def("create_composing_child", &ComposingChild::Make, py::arg("executor"),
        py::arg("cardinalities"), "Creates a ComposingExecutor.");
  m.def("create_composing_executor", &CreateComposingExecutor,
        py::arg("server"), py::arg("children"), "Creates a ComposingExecutor.");
  m.def("create_remote_executor",
        py::overload_cast<std::shared_ptr<grpc::ChannelInterface>,
                          const CardinalityMap&>(&CreateRemoteExecutor),
        py::arg("channel"), py::arg("cardinalities"),
        "Creates a RemoteExecutor.");
  m.def(
      "create_streaming_remote_executor",
      py::overload_cast<std::shared_ptr<grpc::ChannelInterface>,
                        const CardinalityMap&>(&CreateStreamingRemoteExecutor),
      py::arg("channel"), py::arg("cardinalities"),
      "Creates a StreamingRemoteExecutor.");
  m.def("create_xla_executor", &CreateXLAExecutor,
        py::arg("platform_name") = "Host", "Creates an XlaExecutor.");
  m.def("create_sequence_executor", &CreateSequenceExecutor,
        py::arg("target_executor"), "Creates a SequenceExecutor.");

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

namespace pybind11 {
namespace detail {

template <>
struct type_caster<tensorflow::Tensor> {
 public:
  // Macro to create `value` variable which is used in `load` to store the
  // result of the conversion.
  PYBIND11_TYPE_CASTER(tensorflow::Tensor, const_name("Tensor"));

  // Pybind11 caster for PyArray (Python) -> tensorflow::Tensor (C++).
  bool load(handle src, bool) {
    {
      tensorflow::Safe_TF_TensorPtr tf_tensor_ptr;
      tensorflow::Status status = tensorflow::NdarrayToTensor(
          /*ctx=*/nullptr, src.ptr(), &tf_tensor_ptr);
      if (!status.ok()) {
        LOG(ERROR) << status;
        return false;
      }
      status = TF_TensorToTensor(tf_tensor_ptr.get(), &value);
      if (!status.ok()) {
        LOG(ERROR) << status;
        return false;
      }
    }
    tensorflow::ClearDecrefCache();
    return !PyErr_Occurred();
  }

  // Convert tensorflow::Tensor (C++) back to a PyArray (Python).
  static handle cast(const tensorflow::Tensor tensor, return_value_policy,
                     handle) {
    PyObject* result = nullptr;
    tensorflow::Status status = tensorflow::TensorToNdarray(tensor, &result);
    if (!status.ok()) {
      PyErr_SetString(PyExc_ValueError, "Failed to create np.ndarray");
      return nullptr;
    }
    return result;
  }
};
}  // namespace detail
}  // namespace pybind11
