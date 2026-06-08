/*
 * Copyright 2026, The TensorFlow Federated Authors.
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

/*
 * VERS_1.0 exports public TFF library API and shared types (like absl::Status and pybind11)
 * so they can be resolved by dynamic dependencies.
 */
VERS_1.0 {
  global:
    Wrapped_PyInit_executor_bindings;
    Wrapped_PyInit_tensorflow_bindings;
    Wrapped_PyInit_executor_stack_bindings;
    *tensorflow_federated*;
    extern "C++" {
      absl::Status*;
      absl::internal_statusor::*;
      pybind11_protobuf::*;
      pybind11::google::*;
    };
    _ZTI*absl*Status*;
    _ZTS*absl*Status*;
    _ZTI*google*protobuf*;
    _ZTS*google*protobuf*;
    _ZTI*pybind11*;
    _ZTS*pybind11*;
};

/*
 * TFF_PRIVATE exports grpc::* symbols under a private version to avoid dynamic linker
 * conflicts/crashes with external Python gRPC packages (like grpcio) while allowing
 * local dynamic wrappers (like executor_bindings.so) to resolve them. Other
 * dependencies (like Abseil) are kept local (hidden) to prevent leakage.
 */
TFF_PRIVATE {
  global:
    extern "C++" {
      grpc::*;
    };
  local:
    extern "C++" {
      absl::*;
    };
    *;
};
