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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EXECUTOR_TEST_BASE_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EXECUTOR_TEST_BASE_H_

#include <memory>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/protobuf_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

class ExecutorTestBase : public ::testing::Test {
 protected:
  std::shared_ptr<Executor> test_executor_;

  void ExpectMaterialize(ValueId id, v0::Value value) {
    auto result = TFF_ASSERT_OK(test_executor_->Materialize(id));
    EXPECT_THAT(result, testing::EqualsProto(value));
  }

  void ExpectCreateMaterialize(v0::Value value) {
    auto id = TFF_ASSERT_OK(test_executor_->CreateValue(value));
    ExpectMaterialize(id, value);
  }
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EXECUTOR_TEST_BASE_H_
