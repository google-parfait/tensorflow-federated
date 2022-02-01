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

// This file contains the pybind11 definitions exposing `DataBackendExample`.
//
// It is designed to be used as an example to implementors of the `DataBackend`
// interface for how they can expose their own `DataBackend` implementations to
// Python.

#include "include/pybind11/detail/common.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "include/pybind11/stl.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/wrapped_proto_caster.h"
#include "tensorflow_federated/cc/core/impl/executors/data_backend.h"
#include "tensorflow_federated/examples/custom_data_backend/data_backend_example.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated_examples {
namespace {

using pybind11_protobuf::WithWrappedProtos;
using tensorflow_federated::DataBackend;
using tensorflow_federated::v0::Data;
using tensorflow_federated::v0::Value;
namespace py = ::pybind11;

PYBIND11_MODULE(data_backend_example_bindings, m) {
  py::google::ImportStatusModule();
  pybind11_protobuf::ImportWrappedProtoCasters();

  // Select the particular overload of `resolve_to_value` which returns
  // an `absl::StatusOr<Value>` rather than mutating the argument.
  absl::StatusOr<Value> (DataBackendExample::*resolve_to_value)(const Data&) =
      &DataBackend::ResolveToValue;

  py::class_<DataBackendExample>(m, "DataBackendExample")
      .def(py::init())
      .def("resolve_to_value", WithWrappedProtos(resolve_to_value),
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace
}  // namespace tensorflow_federated_examples
