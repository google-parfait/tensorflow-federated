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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATA_EXECUTOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATA_EXECUTOR_H_

#include <memory>

#include "tensorflow_federated/cc/core/impl/executors/data_backend.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace tensorflow_federated {

// Returns an executor that resolves `Data` blocks using `data_backend`.
//
// Note that this executor does not transform values not at top-level, so it is
// advisable to place this beneath a `ReferenceResolvingExecutor`.
std::shared_ptr<Executor> CreateDataExecutor(
    std::shared_ptr<Executor> child, std::shared_ptr<DataBackend> data_backend);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DATA_EXECUTOR_H_
