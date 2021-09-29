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

#include "tensorflow_federated/cc/core/impl/executors/session_provider.h"

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated {

namespace {

class SessionProviderTest : public ::testing::Test {
 public:
};

TEST_F(SessionProviderTest, TestStandaloneTakeSession) {
  tensorflow::GraphDef graphdef_pb;
  SessionProvider session_provider(std::move(graphdef_pb), absl::nullopt);
  TFF_ASSERT_OK(session_provider.TakeSession());
}

TEST_F(SessionProviderTest, TestSessionNotAvailableConcurrencyLimited) {
  tensorflow::GraphDef graphdef_pb;
  SessionProvider session_provider(std::move(graphdef_pb),
                                   /*max_active_sessions=*/1);
  TFF_ASSERT_OK(session_provider.TakeSession());
  EXPECT_EQ(session_provider.SessionOrCpuAvailable(), false);
}

TEST_F(SessionProviderTest, TestSessionAvailableConcurrencyUnlimited) {
  tensorflow::GraphDef graphdef_pb;
  SessionProvider session_provider(std::move(graphdef_pb), absl::nullopt);
  TFF_ASSERT_OK(session_provider.TakeSession());
  EXPECT_EQ(session_provider.SessionOrCpuAvailable(), true);
}

TEST_F(SessionProviderTest, TestReturningSessionFreesUpAvailability) {
  tensorflow::GraphDef graphdef_pb;
  SessionProvider session_provider(std::move(graphdef_pb), 1);
  absl::StatusOr<std::unique_ptr<tensorflow::Session>> taken_session =
      session_provider.TakeSession();
  TFF_ASSERT_OK(taken_session.status());
  EXPECT_EQ(session_provider.SessionOrCpuAvailable(), false);
  session_provider.ReturnSession(std::move(taken_session.value()));
  EXPECT_EQ(session_provider.SessionOrCpuAvailable(), true);
}

}  // namespace
}  // namespace tensorflow_federated
