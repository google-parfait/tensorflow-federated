/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/executors/protobuf_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

TEST(TensorSpecTest, Create) {
  TensorSpec tensor_spec("foo", DT_INT32, {2, 3});
  EXPECT_THAT(tensor_spec.name(), "foo");
  EXPECT_THAT(tensor_spec.dtype(), DT_INT32);
  EXPECT_THAT(tensor_spec.shape(), TensorShape({2, 3}));
}

TEST(TensorSpecTest, FromProto) {
  TensorSpecProto tensor_spec_proto;
  tensor_spec_proto.set_name("foo");
  tensor_spec_proto.set_dtype(DT_INT32);
  tensor_spec_proto.mutable_shape()->add_dim_sizes(8);
  TensorSpec expected_tensor_spec("foo", DT_INT32, {8});
  auto actual_tensor_spec = TensorSpec::FromProto(tensor_spec_proto);
  EXPECT_THAT(actual_tensor_spec, IsOk());
  EXPECT_EQ(*actual_tensor_spec, expected_tensor_spec);
}

TEST(TensorSpecTest, ToProto) {
  TensorSpec tensor_spec("foo", DT_INT32, {8});
  TensorSpecProto tensor_spec_proto;
  tensor_spec_proto.set_name("foo");
  tensor_spec_proto.set_dtype(DT_INT32);
  tensor_spec_proto.mutable_shape()->add_dim_sizes(8);
  EXPECT_THAT(tensor_spec.ToProto(), testing::EqualsProto(tensor_spec_proto));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
