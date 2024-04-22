/* Copyright 2023, The TensorFlow Federated Authors.

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

#include "include/pybind11/pybind11.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"

namespace tensorflow_federated {

namespace {

////////////////////////////////////////////////////////////////////////////////
// The Python module definition `executor_test_utils_bindings`.
//
// This will be used with `import executor_test_utils_bindings` in Python. This
// module should _not_ be directly imported into the public pip API. The methods
// here will raise `NotOkStatus` errors from absl, which are not user friendly.
////////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(executor_test_utils_bindings, m) {
  m.doc() = "Bindings for the C++ ";

  // Executor construction methods.
  m.def("create_mock_executor", &CreateMockExecutor, "Creates a MockExecutor.");
}

}  // namespace
}  // namespace tensorflow_federated
