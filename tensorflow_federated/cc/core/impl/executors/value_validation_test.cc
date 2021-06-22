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

#include "tensorflow_federated/cc/core/impl/executors/value_validation.h"

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/proto/v0/computation.proto.h"
#include "tensorflow_federated/proto/v0/executor.proto.h"

namespace tensorflow_federated {

namespace {

using ::absl::StatusCode;
using testing::ClientsV;
using testing::ServerV;
using testing::TensorV;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

const uint32 NUM_CLIENTS = 5;
const v0::Value TENSOR = TensorV(1.0);

class ValueValidationTest : public ::testing::Test {};

TEST_F(ValueValidationTest, ValidateFederatedServer) {
  EXPECT_THAT(ValidateFederated(NUM_CLIENTS, ServerV(TENSOR).federated()),
              IsOkAndHolds(FederatedKind::SERVER));
}

TEST_F(ValueValidationTest, ValidateFederatedAtClients) {
  EXPECT_THAT(
      ValidateFederated(
          NUM_CLIENTS,
          ClientsV(std::vector<v0::Value>(NUM_CLIENTS, TENSOR)).federated()),
      IsOkAndHolds(FederatedKind::CLIENTS));
}

TEST_F(ValueValidationTest, ValidateFederatedAllEqualAtClients) {
  EXPECT_THAT(
      ValidateFederated(NUM_CLIENTS, ClientsV({TENSOR}, true).federated()),
      IsOkAndHolds(FederatedKind::CLIENTS_ALL_EQUAL));
}

TEST_F(ValueValidationTest, ValidateFederatedAllEqualNotLengthOne) {
  EXPECT_THAT(ValidateFederated(NUM_CLIENTS,
                                ClientsV({TENSOR, TENSOR}, true).federated()),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ValueValidationTest, ValidateFederatedErrorOnWrongNumberClients) {
  EXPECT_THAT(
      ValidateFederated(NUM_CLIENTS, ClientsV({TENSOR, TENSOR}).federated()),
      StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ValueValidationTest, ValidateFederatedErrorOnNonAllEqualServer) {
  v0::Value value_proto;
  v0::FederatedType* type_proto =
      value_proto.mutable_federated()->mutable_type();
  type_proto->set_all_equal(false);
  *type_proto->mutable_placement()->mutable_value()->mutable_uri() = "server";
  for (uint32 i = 0; i < NUM_CLIENTS; i++) {
    *value_proto.mutable_federated()->add_value() = TENSOR;
  }
  EXPECT_THAT(ValidateFederated(NUM_CLIENTS, value_proto.federated()),
              StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace

}  // namespace tensorflow_federated
