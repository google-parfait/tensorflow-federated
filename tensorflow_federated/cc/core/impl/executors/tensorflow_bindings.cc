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

// This file contains the pybind11 definitions for exposing the C++ Executor
// interface in Python.
//
// General principles:
//   - Python methods defined here (e.g. `.def_*()`) should not contain
//     "business logic". That should be implemented on the underlying C++ class.
//     The only logic that may exist here is parameter/result conversions (e.g.
//     `OwnedValueId` -> `ValueId`, etc).

#include <cstdint>
#include <memory>

#include "include/pybind11/cast.h"
#include "include/pybind11/detail/common.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_casters.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"

namespace tensorflow_federated {

namespace py = ::pybind11;

namespace {

////////////////////////////////////////////////////////////////////////////////
// The Python module definition `tensorflow_bindings`.
//
// This will be used with `import tensorflow_bindings` on the Python side. This
// module should _not_ be directly imported into the public pip API. The methods
// here will raise `NotOkStatus` errors from absl, which are not user friendly.
////////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(tensorflow_bindings, m) {
  py::google::ImportStatusModule();

  // IMPORTANT: The binding defined in this module are dependent on the binding
  // defined in the `executor_bindings` module.
  py::module::import(
      "tensorflow_federated.cc.core.impl.executors.executor_bindings"
  );

  m.doc() = "Bindings for the C++ ";

  // Executor construction methods.
  m.def(
      "create_tensorflow_executor",
      [](int32_t max_concurrent_computation_calls,
         bool synchronous_value_creation) {
        return CreateTensorFlowExecutor(max_concurrent_computation_calls,
                                        synchronous_value_creation);
      },
      py::arg("max_concurrent_computation_calls") = -1,
      py::arg("synchronous_value_creation") = false,
      "Creates a TensorFlowExecutor.");
}

}  // namespace
}  // namespace tensorflow_federated
