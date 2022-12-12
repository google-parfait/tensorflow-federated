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

#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated {

namespace {

TEST(SessionProviderTest, TestStandaloneTakeSession) {
  tensorflow::GraphDef graphdef_pb;
  SessionProvider session_provider(std::move(graphdef_pb));
  TFF_ASSERT_OK(session_provider.TakeSession());
}

}  // namespace
}  // namespace tensorflow_federated
