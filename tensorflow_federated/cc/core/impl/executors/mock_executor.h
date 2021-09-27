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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_MOCK_EXECUTOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_MOCK_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/protobuf_matchers.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

const ::testing::Cardinality ONCE = ::testing::Exactly(1);

class MockExecutor : public Executor,
                     public std::enable_shared_from_this<Executor> {
 public:
  MockExecutor() { next_id_ = 0; }
  ~MockExecutor() override {}

  MOCK_METHOD(absl::StatusOr<OwnedValueId>, CreateValue,
              (const v0::Value& value_pb), (override));
  MOCK_METHOD(absl::StatusOr<OwnedValueId>, CreateCall,
              (const ValueId function,
               const absl::optional<const ValueId> argument),
              (override));
  MOCK_METHOD(absl::StatusOr<OwnedValueId>, CreateStruct,
              (const absl::Span<const ValueId> members), (override));
  MOCK_METHOD(absl::StatusOr<OwnedValueId>, CreateSelection,
              (const ValueId source, const uint32_t index), (override));
  MOCK_METHOD(absl::Status, Materialize,
              (const ValueId value, v0::Value* value_pb), (override));
  MOCK_METHOD(absl::Status, Dispose, (const ValueId value), (override));

  template <typename EXPECTATION>
  ValueId ReturnsNewValue(EXPECTATION& e,
                          ::testing::Cardinality repeatedly = ONCE) {
    ValueId id = next_id_++;
    auto lambda = [this, id]() { return OwnedValueId(shared_from_this(), id); };
    e.Times(repeatedly).WillRepeatedly(std::move(lambda));
    EXPECT_CALL(*this, Dispose(id)).Times(repeatedly);
    return id;
  }

  inline ValueId ExpectCreateValue(const v0::Value& expected,
                                   ::testing::Cardinality repeatedly = ONCE) {
    return ReturnsNewValue(
        EXPECT_CALL(*this, CreateValue(testing::EqualsProto(expected))),
        repeatedly);
  }

  inline ValueId ExpectCreateCall(ValueId fn_id,
                                  absl::optional<const ValueId> arg_id,
                                  ::testing::Cardinality repeatedly = ONCE) {
    return ReturnsNewValue(EXPECT_CALL(*this, CreateCall(fn_id, arg_id)),
                           repeatedly);
  }

  inline ValueId ExpectCreateSelection(ValueId source_id, uint32_t index) {
    return ReturnsNewValue(
        EXPECT_CALL(*this, CreateSelection(source_id, index)));
  }

  inline ValueId ExpectCreateStruct(absl::Span<const ValueId> elements,
                                    ::testing::Cardinality repeatedly = ONCE) {
    // Store the data behind the span since the span itself may refer to a
    // temporary which is unavailable at the time the expected call comes in.
    struct_expectations_.push_back(std::make_unique<std::vector<ValueId>>(
        elements.begin(), elements.end()));
    absl::Span<const ValueId> elements_again(*struct_expectations_.back());
    return ReturnsNewValue(EXPECT_CALL(*this, CreateStruct(elements_again)),
                           repeatedly);
  }

  inline void ExpectMaterialize(ValueId id, v0::Value to_return,
                                ::testing::Cardinality repeatedly = ONCE) {
    EXPECT_CALL(*this, Materialize(id, ::testing::_))
        .Times(repeatedly)
        .WillRepeatedly(
            ::testing::DoAll(::testing::SetArgPointee<1>(std::move(to_return)),
                             ::testing::Return(absl::OkStatus())));
  }

  inline void ExpectCreateMaterialize(
      v0::Value value_pb, ::testing::Cardinality repeatedly = ONCE) {
    auto id = ExpectCreateValue(value_pb);
    ExpectMaterialize(id, value_pb);
  }

 private:
  ValueId next_id_;
  std::vector<std::unique_ptr<std::vector<ValueId>>> struct_expectations_;
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_MOCK_EXECUTOR_H_
