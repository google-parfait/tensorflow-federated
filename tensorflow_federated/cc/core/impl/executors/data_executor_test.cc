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

#include "tensorflow_federated/cc/core/impl/executors/data_executor.h"

#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor_test_base.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_data_backend.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

namespace {

using ::tensorflow_federated::testing::TensorV;

class DataExecutorTest : public ExecutorTestBase {
 public:
  DataExecutorTest() {
    test_executor_ =
        CreateDataExecutor(mock_executor_child_, mock_data_backend_);
  }

  ~DataExecutorTest() override {}

 protected:
  std::shared_ptr<::testing::StrictMock<MockDataBackend>> mock_data_backend_ =
      std::make_shared<::testing::StrictMock<MockDataBackend>>();
  std::shared_ptr<::testing::StrictMock<MockExecutor>> mock_executor_child_ =
      std::make_shared<::testing::StrictMock<MockExecutor>>();
};

TEST_F(DataExecutorTest, CreateValueResolvesData) {
  std::string uri = "some_data_uri";
  v0::Value resolved_data_value = TensorV(22);
  mock_data_backend_->ExpectResolveToValue(uri, resolved_data_value);
  v0::Value unresolved_data_value;
  unresolved_data_value.mutable_computation()->mutable_data()->set_uri(uri);
  mock_executor_child_->ExpectCreateMaterialize(resolved_data_value);
  OwnedValueId value_id =
      TFF_ASSERT_OK(test_executor_->CreateValue(unresolved_data_value));
  ExpectMaterialize(value_id, resolved_data_value);
}

TEST_F(DataExecutorTest, CreateValueUnknownValuesDelegatesToChild) {
  v0::Value unknown_value = TensorV(5);
  mock_executor_child_->ExpectCreateMaterialize(unknown_value);
  ExpectCreateMaterialize(unknown_value);
}

TEST_F(DataExecutorTest, CreateValueNonDataComputationDelegatesToChild) {
  v0::Value non_data_computation;
  non_data_computation.mutable_computation()->mutable_intrinsic()->set_uri(
      "some_uri");
  mock_executor_child_->ExpectCreateMaterialize(non_data_computation);
  ExpectCreateMaterialize(non_data_computation);
}

TEST_F(DataExecutorTest, CreateStructDelegatesToChild) {
  v0::Value v1 = TensorV(1);
  ValueId v1_child_id = mock_executor_child_->ExpectCreateValue(v1);
  OwnedValueId v1_id = TFF_ASSERT_OK(test_executor_->CreateValue(v1));

  v0::Value v2 = TensorV(2);
  ValueId v2_child_id = mock_executor_child_->ExpectCreateValue(v2);
  OwnedValueId v2_id = TFF_ASSERT_OK(test_executor_->CreateValue(v2));

  ValueId struct_child_id =
      mock_executor_child_->ExpectCreateStruct({v1_child_id, v2_child_id});
  OwnedValueId struct_id =
      TFF_ASSERT_OK(test_executor_->CreateStruct({v1_id, v2_id}));

  v0::Value result = TensorV("result");
  mock_executor_child_->ExpectMaterialize(struct_child_id, result);
  ExpectMaterialize(struct_id, result);
}

TEST_F(DataExecutorTest, CreateSelectionDelegatesToChild) {
  v0::Value source = TensorV("source");
  ValueId source_child_id = mock_executor_child_->ExpectCreateValue(source);
  OwnedValueId source_id = TFF_ASSERT_OK(test_executor_->CreateValue(source));

  const uint32_t index = 4;
  ValueId selected_child_id =
      mock_executor_child_->ExpectCreateSelection(source_child_id, index);
  OwnedValueId selected_id =
      TFF_ASSERT_OK(test_executor_->CreateSelection(source_id, index));

  v0::Value result = TensorV("result");
  mock_executor_child_->ExpectMaterialize(selected_child_id, result);
  ExpectMaterialize(selected_id, result);
}

TEST_F(DataExecutorTest, CreateCallNoArgDelegatesToChild) {
  v0::Value fn = TensorV("fn");
  ValueId fn_child_id = mock_executor_child_->ExpectCreateValue(fn);
  OwnedValueId fn_id = TFF_ASSERT_OK(test_executor_->CreateValue(fn));

  ValueId result_child_id =
      mock_executor_child_->ExpectCreateCall(fn_child_id, absl::nullopt);
  OwnedValueId result_id =
      TFF_ASSERT_OK(test_executor_->CreateCall(fn_id, absl::nullopt));

  v0::Value result = TensorV("result");
  mock_executor_child_->ExpectMaterialize(result_child_id, result);
  ExpectMaterialize(result_id, result);
}

TEST_F(DataExecutorTest, CreateCallWithArgDelegatesToChild) {
  v0::Value fn = TensorV("fn");
  ValueId fn_child_id = mock_executor_child_->ExpectCreateValue(fn);
  OwnedValueId fn_id = TFF_ASSERT_OK(test_executor_->CreateValue(fn));

  v0::Value arg = TensorV("arg");
  ValueId arg_child_id = mock_executor_child_->ExpectCreateValue(arg);
  OwnedValueId arg_id = TFF_ASSERT_OK(test_executor_->CreateValue(arg));

  ValueId result_child_id =
      mock_executor_child_->ExpectCreateCall(fn_child_id, arg_child_id);
  OwnedValueId result_id =
      TFF_ASSERT_OK(test_executor_->CreateCall(fn_id, arg_id));

  v0::Value result = TensorV("result");
  mock_executor_child_->ExpectMaterialize(result_child_id, result);
  ExpectMaterialize(result_id, result);
}

}  // namespace

}  // namespace tensorflow_federated
