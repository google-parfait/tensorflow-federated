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

#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"

#include <memory>

#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace tensorflow_federated {

std::shared_ptr<Executor> CreateMockExecutor() {
  return std::make_shared<MockExecutor>();
}

}  // namespace tensorflow_federated
