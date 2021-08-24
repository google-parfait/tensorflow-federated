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

#include <memory>

#include "grpcpp/grpcpp.h"
#include "absl/types/span.h"
#include "pybind11/include/pybind11/detail/common.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/proto_casters.h"
#include "tensorflow_federated/cc/core/impl/executor_stacks/remote_stacks.h"

namespace tensorflow_federated {

namespace py = ::pybind11;

namespace {

PYBIND11_MODULE(executor_stack_bindings, m) {
  m.def("create_remote_executor_stack",
        py::overload_cast<
            absl::Span<const std::shared_ptr<grpc::ChannelInterface>>,
            const CardinalityMap&>(&CreateRemoteExecutorStack),
        "Creates a C++ remote execution stack.");
}

}  // namespace
}  // namespace tensorflow_federated
