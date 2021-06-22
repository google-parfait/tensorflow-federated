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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_REFERENCE_RESOLVING_EXECUTOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_REFERENCE_RESOLVING_EXECUTOR_H_

#include <memory>

#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace tensorflow_federated {

// Returns an executor that can resolve Lambdas and References, relying on
// a child executor to complete the computation.
std::shared_ptr<Executor> CreateReferenceResolvingExecutor(
    std::shared_ptr<Executor> child);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_REFERENCE_RESOLVING_EXECUTOR_H_
