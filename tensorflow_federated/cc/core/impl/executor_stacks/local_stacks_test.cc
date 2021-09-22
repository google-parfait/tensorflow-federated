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

#include "tensorflow_federated/cc/core/impl/executor_stacks/local_stacks.h"

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

using testing::MockFunction;
using testing::Return;

namespace tensorflow_federated {

class LocalStacksTest : public ::testing::Test {
 protected:
  LocalStacksTest()
      : test_executor_(std::make_shared<MockExecutor>()),
        cards_({{std::string(kClientsUri), 1}}) {}
  std::shared_ptr<Executor> test_executor_;
  CardinalityMap cards_;
};

TEST_F(LocalStacksTest, CustomLeafExecutorIsCalled) {
  MockFunction<absl::StatusOr<std::shared_ptr<Executor>>()> mock_executor_fn;
  EXPECT_CALL(mock_executor_fn, Call()).WillOnce(Return(test_executor_));
  TFF_EXPECT_OK(CreateLocalExecutor(cards_, mock_executor_fn.AsStdFunction()));
}

}  // namespace tensorflow_federated
