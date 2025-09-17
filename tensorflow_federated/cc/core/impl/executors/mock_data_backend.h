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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_MOCK_DATA_BACKEND_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_MOCK_DATA_BACKEND_H_

#include <string>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "absl/status/status.h"
#include "federated_language/proto/computation.pb.h"
#include "third_party/py/federated_language_executor/executor.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/data_backend.h"
#include "tensorflow_federated/cc/testing/protobuf_matchers.h"

namespace tensorflow_federated {

using ::tensorflow_federated::testing::EqualsProto;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgReferee;

class MockDataBackend : public DataBackend {
 public:
  ~MockDataBackend() override = default;
  MOCK_METHOD(absl::Status, ResolveToValue,
              (const federated_language::Data& data_reference,
               const federated_language::Type& type_reference,
               federated_language_executor::Value& data_out),
              (override));

  inline void ExpectResolveToValue(
      std::string expected_uri, federated_language::Type expected_type,
      federated_language_executor::Value to_return) {
    federated_language::Data data;
    data.set_uri(std::move(expected_uri));
    EXPECT_CALL(*this, ResolveToValue(EqualsProto(data),
                                      EqualsProto(expected_type), ::testing::_))
        .WillOnce(DoAll(SetArgReferee<2>(std::move(to_return)),
                        Return(absl::OkStatus())));
  }
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_MOCK_DATA_BACKEND_H_
