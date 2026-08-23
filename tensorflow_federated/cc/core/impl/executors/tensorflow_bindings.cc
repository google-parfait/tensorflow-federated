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

#include "absl/status/status.h"
#include "federated_language/proto/array.pb.h"
#include "include/pybind11/cast.h"
#include "include/pybind11/detail/common.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"
#include "tensorflow/c/safe_ptr.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"

namespace pybind11 {
namespace detail {

template <>
struct type_caster<tensorflow::Tensor> {
 public:
  PYBIND11_TYPE_CASTER(tensorflow::Tensor, const_name("Tensor"));

  bool load(handle src, bool) {
    tensorflow::Safe_TF_TensorPtr tf_tensor_ptr;
    absl::Status status =
        tensorflow::NdarrayToTensor(/*ctx=*/nullptr, src.ptr(), &tf_tensor_ptr);
    if (!status.ok()) {
      return false;
    }
    status = TF_TensorToTensor(tf_tensor_ptr.get(), &value);
    if (!status.ok()) {
      return false;
    }
    return !PyErr_Occurred();
  }

  static handle cast(const tensorflow::Tensor& tensor, return_value_policy,
                     handle) {
    PyObject* result = nullptr;
    absl::Status status = tensorflow::TensorToNdarray(tensor, &result);
    if (!status.ok()) {
      return nullptr;
    }
    return result;
  }
};

}  // namespace detail
}  // namespace pybind11

namespace tensorflow_federated {

namespace py = ::pybind11;

namespace {

PYBIND11_MODULE(tensorflow_bindings, m) {
  tsl::ImportNumpy();
  py::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  m.doc() = "Bindings for the C++ TensorFlow serialization";

  // Serialization methods.
  m.def("tensor_from_array_content", &TensorFromArrayContent,
        py::arg("array_pb"),
        "Deserializes a tensorflow::Tensor from a federated_language::Array"
        " content.");
  m.def("array_content_from_tensor", &ArrayContentFromTensor, py::arg("tensor"),
        "Serializes a tensorflow::Tensor to a federated_language::Array "
        "content.");
  m.def("tensor_from_array", &TensorFromArray, py::arg("array_pb"),
        "Deserializes a tensorflow::Tensor from a federated_language::Array.");
  m.def("array_from_tensor", &ArrayFromTensor, py::arg("tensor"),
        "Serializes a tensorflow::Tensor to a federated_language::Array.");
}

}  // namespace
}  // namespace tensorflow_federated
