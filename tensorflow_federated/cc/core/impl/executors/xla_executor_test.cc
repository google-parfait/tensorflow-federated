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

#include "tensorflow_federated/cc/core/impl/executors/xla_executor.h"

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"

namespace tensorflow_federated {
namespace {

using ::tensorflow_federated::testing::TensorV;

class XLAExecutorTest : public ::testing::Test {
 public:
  XLAExecutorTest() { test_executor_ = CreateXLAExecutor(); }
  std::shared_ptr<Executor> test_executor_;
};

TEST_F(XLAExecutorTest, CreateValueSimpleTensor) {
  int8_t input_int = 9;
  v0::Value input_pb = TensorV(input_int);
  EXPECT_THAT(test_executor_->CreateValue(input_pb),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace tensorflow_federated
