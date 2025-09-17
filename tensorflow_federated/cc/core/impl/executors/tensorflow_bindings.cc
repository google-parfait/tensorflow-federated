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

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "federated_language/proto/computation.pb.h"
#include "third_party/py/federated_language_executor/executor.pb.h"
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
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"

namespace tensorflow {
absl::Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);
}  // namespace tensorflow

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
  m.def("create_tensorflow_executor", &CreateTensorFlowExecutor,
        py::arg("max_concurrent_computation_calls") = -1,
        "Creates a TensorFlowExecutor.");
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
      absl::Status status = tensorflow::NdarrayToTensor(
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
    absl::Status status = tensorflow::TensorToNdarray(tensor, &result);
    if (!status.ok()) {
      PyErr_SetString(PyExc_ValueError, "Failed to create np.ndarray");
      return nullptr;
    }
    return result;
  }
};
}  // namespace detail
}  // namespace pybind11
