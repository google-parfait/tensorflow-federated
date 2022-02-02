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
#include "absl/status/statusor.h"
#include "tensorflow_federated/cc/core/impl/executors/data_backend.h"
#include "tensorflow_federated/cc/core/impl/executors/protobuf_matchers.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

using ::tensorflow_federated::testing::EqualsProto;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgReferee;

class MockDataBackend : public DataBackend {
 public:
  ~MockDataBackend() override {}
  MOCK_METHOD(absl::Status, ResolveToValue,
              (const v0::Data& data_reference, const v0::Type& type_reference,
               v0::Value& data_out),
              (override));

  inline void ExpectResolveToValue(std::string expected_uri,
                                   v0::Type expected_type,
                                   v0::Value to_return) {
    v0::Data data;
    data.set_uri(std::move(expected_uri));
    EXPECT_CALL(*this, ResolveToValue(EqualsProto(data),
                                      EqualsProto(expected_type), ::testing::_))
        .WillOnce(DoAll(SetArgReferee<2>(std::move(to_return)),
                        Return(absl::OkStatus())));
  }
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_MOCK_DATA_BACKEND_H_
