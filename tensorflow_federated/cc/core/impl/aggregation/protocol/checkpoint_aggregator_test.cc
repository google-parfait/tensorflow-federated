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

#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "tensorflow_federated/cc/core/impl/aggregation/testing/parse_text_proto.h"
// clang-format on
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/scheduler.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/mocks.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::AnyOf;
using ::testing::ByMove;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::StrEq;
using testing::TestWithParam;

using CheckpointAggregatorTest = TestWithParam<bool>;

Configuration default_configuration() {
  // One "federated_sum" intrinsic with a single scalar int32 tensor.
  return PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
}

Configuration default_fedsql_configuration() {
  return PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "fedsql_group_by"
      intrinsic_args {
        input_tensor {
          name: "key1"
          dtype: DT_FLOAT
          shape { dim_sizes: -1 }
        }
      }
      output_tensors {
        name: "key1_out"
        dtype: DT_FLOAT
        shape { dim_sizes: -1 }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "val1"
            dtype: DT_FLOAT
            shape {}
          }
        }
        output_tensors {
          name: "val1_out"
          dtype: DT_FLOAT
          shape {}
        }
      }
    }
  )pb");
}

std::unique_ptr<CheckpointAggregator> Create(const Configuration& config) {
  auto result = CheckpointAggregator::Create(config);
  EXPECT_OK(result.status());
  return std::move(result.value());
}

std::unique_ptr<CheckpointAggregator> CreateWithDefaultConfig() {
  return Create(default_configuration());
}

std::unique_ptr<CheckpointAggregator> CreateWithDefaultFedSqlConfig() {
  return Create(default_fedsql_configuration());
}

TEST(CheckpointAggregatorTest, CreateFromIntrinsicsSuccess) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"federated_sum",
                        {TensorSpec("foo", DT_INT32, {})},
                        {TensorSpec("foo_out", DT_INT32, {})},
                        {},
                        {}});
  EXPECT_OK(CheckpointAggregator::Create(&intrinsics));
}

TEST(CheckpointAggregatorTest, CreateFromIntrinsicsUnsupportedNumberOfInputs) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back(
      {"federated_sum",
       {TensorSpec("foo", DT_INT32, {}), TensorSpec("bar", DT_INT32, {})},
       {TensorSpec("foo_out", DT_INT32, {})},
       {},
       {}});
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateFromIntrinsicsUnsupportedNumberOfOutputs) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"federated_sum",
                        {TensorSpec("foo", DT_INT32, {})},
                        {TensorSpec("foo_out", DT_INT32, {}),
                         TensorSpec("bar_out", DT_INT32, {})},
                        {},
                        {}});
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateFromIntrinsicsUnsupportedInputType) {
  std::vector<Intrinsic> intrinsics;
  Tensor parameter =
      Tensor::Create(DT_FLOAT, {1}, CreateTestData<float>({42})).value();
  Intrinsic intrinsic{"federated_sum",
                      {TensorSpec("foo", DT_INT32, {})},
                      {TensorSpec("foo_out", DT_INT32, {})},
                      {},
                      {}};
  intrinsic.parameters.push_back(std::move(parameter));
  intrinsics.push_back(std::move(intrinsic));
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args { parameter {} }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");

  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateFromIntrinsicsUnsupportedIntrinsicUri) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"unsupported_xyz",
                        {TensorSpec("foo", DT_INT32, {})},
                        {TensorSpec("foo_out", DT_INT32, {})},
                        {},
                        {}});
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics), StatusIs(NOT_FOUND));
}

TEST(CheckpointAggregatorTest, CreateFromIntrinsicsUnsupportedInputSpec) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"federated_sum",
                        {TensorSpec("foo", DT_INT32, {-1})},
                        {TensorSpec("foo_out", DT_INT32, {})},
                        {},
                        {}});
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest,
     CreateFromIntrinsicsMismatchingInputAndOutputDataType) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"federated_sum",
                        {TensorSpec("foo", DT_INT32, {})},
                        {TensorSpec("foo_out", DT_FLOAT, {})},
                        {},
                        {}});
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest,
     CreateFromIntrinsicsMismatchingInputAndOutputShape) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"federated_sum",
                        {TensorSpec("foo", DT_INT32, {1})},
                        {TensorSpec("foo_out", DT_INT32, {2})},
                        {},
                        {}});
  EXPECT_THAT(CheckpointAggregator::Create(&intrinsics),
              StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateSuccess) {
  EXPECT_OK(CheckpointAggregator::Create(default_configuration()));
}

TEST(CheckpointAggregatorTest, CreateUnsupportedNumberOfInputs) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      intrinsic_args {
        input_tensor {
          name: "bar"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateUnsupportedNumberOfOutputs) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
      output_tensors {
        name: "bar_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateUnsupportedInputType) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args { parameter {} }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateUnsupportedIntrinsicUri) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "unsupported_xyz"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateUnsupportedInputSpec) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape { dim_sizes: -1 }
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateMismatchingInputAndOutputDataType) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_FLOAT
        shape {}
      }
    }
  )pb");
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, CreateMismatchingInputAndOutputShape) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape { dim_sizes: 1 }
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape { dim_sizes: 2 }
      }
    }
  )pb");
  EXPECT_THAT(CheckpointAggregator::Create(config_message),
              StatusIs(INVALID_ARGUMENT));
}

TEST_P(CheckpointAggregatorTest, CreateFromIntrinsicsAccumulateSuccess) {
  std::vector<Intrinsic> intrinsics;
  intrinsics.push_back({"federated_sum",
                        {TensorSpec("foo", DT_INT32, {})},
                        {TensorSpec("foo_out", DT_INT32, {})},
                        {},
                        {}});
  auto aggregator = CheckpointAggregator::Create(&intrinsics).value();

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator =
        CheckpointAggregator::Deserialize(&intrinsics, serialized_state)
            .value();
  }
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({2}));
  }));
  TFF_EXPECT_OK(aggregator->Accumulate(parser));
}

TEST(CheckpointAggregatorTest, AccumulateMissingTensor) {
  auto aggregator = CreateWithDefaultConfig();
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo")))
      .WillOnce(Return(ByMove(absl::NotFoundError("Missing tensor foo"))));
  EXPECT_THAT(aggregator->Accumulate(parser), StatusIs(NOT_FOUND));
}

TEST(CheckpointAggregatorTest, AccumulateMismatchingTensor) {
  auto aggregator = CreateWithDefaultConfig();
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {}, CreateTestData({2.f}));
  }));
  EXPECT_THAT(aggregator->Accumulate(parser), StatusIs(INVALID_ARGUMENT));
}

TEST(CheckpointAggregatorTest, AccumulateSuccess) {
  auto aggregator = CreateWithDefaultConfig();
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({2}));
  }));
  EXPECT_OK(aggregator->Accumulate(parser));
}

TEST(CheckpointAggregatorTest, AccumulateAfterReport) {
  auto aggregator = CreateWithDefaultConfig();

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {0})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(aggregator->Report(builder));

  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({2}));
  }));
  EXPECT_THAT(aggregator->Accumulate(parser), StatusIs(ABORTED));
}

TEST(CheckpointAggregatorTest, AccumulateAfterAbort) {
  auto aggregator = CreateWithDefaultConfig();
  aggregator->Abort();
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({2}));
  }));
  EXPECT_THAT(aggregator->Accumulate(parser), StatusIs(ABORTED));
}

TEST_P(CheckpointAggregatorTest, ReportZeroInputs) {
  auto aggregator = CreateWithDefaultConfig();

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator = CheckpointAggregator::Deserialize(default_configuration(),
                                                   serialized_state)
                     .value();
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {0})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, ReportOneInput) {
  auto aggregator = CreateWithDefaultConfig();
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({2}));
  }));
  EXPECT_OK(aggregator->Accumulate(parser));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator = CheckpointAggregator::Deserialize(default_configuration(),
                                                   serialized_state)
                     .value();
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {2})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, ReportTwoInputs) {
  auto aggregator = CreateWithDefaultConfig();
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo")))
      .WillOnce(Invoke(
          [] { return Tensor::Create(DT_INT32, {}, CreateTestData({2})); }))
      .WillOnce(Invoke(
          [] { return Tensor::Create(DT_INT32, {}, CreateTestData({3})); }));
  EXPECT_OK(aggregator->Accumulate(parser));
  EXPECT_OK(aggregator->Accumulate(parser));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator = CheckpointAggregator::Deserialize(default_configuration(),
                                                   serialized_state)
                     .value();
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {5})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, ReportMultipleTensors) {
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape { dim_sizes: 3 }
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape { dim_sizes: 3 }
      }
    }
    intrinsic_configs {
      intrinsic_uri: "federated_sum"
      intrinsic_args {
        input_tensor {
          name: "bar"
          dtype: DT_FLOAT
          shape { dim_sizes: 2 dim_sizes: 2 }
        }
      }
      output_tensors {
        name: "bar_out"
        dtype: DT_FLOAT
        shape { dim_sizes: 2 dim_sizes: 2 }
      }
    })pb");
  auto aggregator = Create(config_message);

  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {3}, CreateTestData({1, 2, 3}));
  }));
  EXPECT_CALL(parser, GetTensor(StrEq("bar"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2, 2},
                          CreateTestData({1.f, 2.f, 3.f, 4.f}));
  }));
  EXPECT_OK(aggregator->Accumulate(parser));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator =
        CheckpointAggregator::Deserialize(config_message, serialized_state)
            .value();
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({3}, {1, 2, 3})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(builder, Add(StrEq("bar_out"),
                           IsTensor<float>({2, 2}, {1.f, 2.f, 3.f, 4.f})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(aggregator->Report(builder));
}

TEST(CheckpointAggregatorTest, ReportAfterReport) {
  auto aggregator = CreateWithDefaultConfig();

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {0})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(aggregator->Report(builder));
  EXPECT_THAT(aggregator->Report(builder), StatusIs(ABORTED));
}

TEST(CheckpointAggregatorTest, ReportAfterAbort) {
  auto aggregator = CreateWithDefaultConfig();
  aggregator->Abort();
  MockCheckpointBuilder builder;
  EXPECT_THAT(aggregator->Report(builder), StatusIs(ABORTED));
}

// A fake aggregator that can never report.
class CannotReportAggregator : public AggVectorAggregator<int> {
 public:
  using AggVectorAggregator<int>::AggVectorAggregator;

 private:
  void AggregateVector(const AggVector<int>& agg_vector) override {}
  bool CanReport() const override { return false; }
};

class CannotReportAggregatorFactory : public TensorAggregatorFactory {
 private:
  absl::StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    if (intrinsic.inputs[0].dtype() != DT_INT32) {
      return absl::InvalidArgumentError("Unsupported dtype: expected DT_INT32");
    }
    return std::make_unique<CannotReportAggregator>(
        intrinsic.inputs[0].dtype(), intrinsic.inputs[0].shape());
  }

  absl::StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    return TFF_STATUS(UNIMPLEMENTED);
  }
};

TEST(CheckpointAggregatorTest, ReportWithFailedCanReportPrecondition) {
  CannotReportAggregatorFactory agg_factory;
  RegisterAggregatorFactory("foo1_aggregation", &agg_factory);
  // Configuration that uses the CannotReportAggregatorFactory above.
  // This aggregation is supposed to always fail the CanReport call.
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "foo1_aggregation"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  auto aggregator = Create(config_message);

  MockCheckpointBuilder builder;
  EXPECT_THAT(aggregator->Report(builder), StatusIs(FAILED_PRECONDITION));
}

TEST_P(CheckpointAggregatorTest, ReportFedSqlZeroInputs) {
  // One intrinsic:
  //    fedsql_group_by with two grouping keys key1 and key2, only the first one
  //    of which should be output, and two inner GoogleSQL:sum intrinsics bar
  //    and baz operating on float tensors.
  Configuration config_message = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "fedsql_group_by"
      intrinsic_args {
        input_tensor {
          name: "key1"
          dtype: DT_STRING
          shape { dim_sizes: -1 }
        }
      }
      intrinsic_args {
        input_tensor {
          name: "key2"
          dtype: DT_STRING
          shape { dim_sizes: -1 }
        }
      }
      output_tensors {
        name: "key1_out"
        dtype: DT_STRING
        shape { dim_sizes: -1 }
      }
      output_tensors {
        name: ""
        dtype: DT_STRING
        shape { dim_sizes: -1 }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "bar"
            dtype: DT_FLOAT
            shape {}
          }
        }
        output_tensors {
          name: "bar_out"
          dtype: DT_FLOAT
          shape {}
        }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "baz"
            dtype: DT_FLOAT
            shape {}
          }
        }
        output_tensors {
          name: "baz_out"
          dtype: DT_FLOAT
          shape {}
        }
      }
    }
  )pb");
  auto aggregator = Create(config_message);

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator =
        CheckpointAggregator::Deserialize(config_message, serialized_state)
            .value();
  }

  MockCheckpointBuilder builder;
  // Verify that empty tensors are added to the result checkpoint.
  EXPECT_CALL(builder, Add(StrEq("key1_out"), IsTensor<string_view>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(builder, Add(StrEq("bar_out"), IsTensor<float>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(builder, Add(StrEq("baz_out"), IsTensor<float>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, ReportFedSqlsOneInput) {
  auto aggregator = CreateWithDefaultFedSqlConfig();

  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("key1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({1.f, 2.f}));
  }));
  EXPECT_CALL(parser, GetTensor(StrEq("val1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({.1f, .2f}));
  }));
  EXPECT_OK(aggregator->Accumulate(parser));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator = CheckpointAggregator::Deserialize(
                     default_fedsql_configuration(), serialized_state)
                     .value();
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("key1_out"), IsTensor<float>({2}, {1.f, 2.f})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(builder, Add(StrEq("val1_out"), IsTensor<float>({2}, {.1f, .2f})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, ReportFedSqlsTwoInputs) {
  auto aggregator = CreateWithDefaultFedSqlConfig();

  MockCheckpointParser parser1;
  EXPECT_CALL(parser1, GetTensor(StrEq("key1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {3}, CreateTestData({1.f, 2.f, 3.f}));
  }));
  EXPECT_CALL(parser1, GetTensor(StrEq("val1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {3}, CreateTestData({.1f, .2f, .3f}));
  }));
  EXPECT_OK(aggregator->Accumulate(parser1));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator = CheckpointAggregator::Deserialize(
                     default_fedsql_configuration(), serialized_state)
                     .value();
  }

  MockCheckpointParser parser2;
  EXPECT_CALL(parser2, GetTensor(StrEq("key1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({2.f, 4.f}));
  }));
  EXPECT_CALL(parser2, GetTensor(StrEq("val1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {2}, CreateTestData({.4f, .5f}));
  }));
  EXPECT_OK(aggregator->Accumulate(parser2));

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("key1_out"),
                           IsTensor<float>({4}, {1.f, 2.f, 3.f, 4.f})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(builder, Add(StrEq("val1_out"),
                           IsTensor<float>({4}, {.1f, .6f, .3f, .5f})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, ReportFedSqlsEmptyInput) {
  auto aggregator = CreateWithDefaultFedSqlConfig();

  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("key1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {0}, CreateTestData<float>({}));
  }));
  EXPECT_CALL(parser, GetTensor(StrEq("val1"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_FLOAT, {0}, CreateTestData<float>({}));
  }));
  EXPECT_OK(aggregator->Accumulate(parser));

  if (GetParam()) {
    auto serialized_state = std::move(*aggregator).Serialize().value();
    aggregator = CheckpointAggregator::Deserialize(
                     default_fedsql_configuration(), serialized_state)
                     .value();
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("key1_out"), IsTensor<float>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(builder, Add(StrEq("val1_out"), IsTensor<float>({0}, {})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(aggregator->Report(builder));
}

TEST_P(CheckpointAggregatorTest, MergeSuccess) {
  auto aggregator1 = CreateWithDefaultConfig();
  MockCheckpointParser parser1;
  EXPECT_CALL(parser1, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({5}));
  }));
  EXPECT_OK(aggregator1->Accumulate(parser1));

  auto aggregator2 = CreateWithDefaultConfig();
  MockCheckpointParser parser2;
  EXPECT_CALL(parser2, GetTensor(StrEq("foo"))).WillOnce(Invoke([] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({7}));
  }));
  EXPECT_OK(aggregator2->Accumulate(parser2));

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize().value();
    aggregator1 = CheckpointAggregator::Deserialize(default_configuration(),
                                                    serialized_state1)
                      .value();
    auto serialized_state2 = std::move(*aggregator2).Serialize().value();
    aggregator2 = CheckpointAggregator::Deserialize(default_configuration(),
                                                    serialized_state2)
                      .value();
  }

  EXPECT_OK(aggregator1->MergeWith(std::move(*aggregator2)));

  if (GetParam()) {
    auto serialized_state1 = std::move(*aggregator1).Serialize().value();
    aggregator1 = CheckpointAggregator::Deserialize(default_configuration(),
                                                    serialized_state1)
                      .value();
  }

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {12})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(aggregator1->Report(builder));
}

TEST(CheckpointAggregatorTest, MergeIncompatibleConfigs) {
  // TODO: b/316662605 - Add this test.
}

TEST(CheckpointAggregatorTest, ConcurrentAccumulationSuccess) {
  const int64_t kNumInputs = 10;
  auto aggregator = CreateWithDefaultConfig();

  // The following block will repeatedly provide scalar int tensors with
  // incrementing values.
  std::atomic<int> tensor_value = 0;
  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillRepeatedly(Invoke([&] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({++tensor_value}));
  }));

  // Schedule receiving inputs on 4 concurrent threads.
  auto scheduler = CreateThreadPoolScheduler(4);
  for (int64_t i = 0; i < kNumInputs; ++i) {
    scheduler->Schedule([&]() { EXPECT_OK(aggregator->Accumulate(parser)); });
  }
  scheduler->WaitUntilIdle();

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {55})))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(aggregator->Report(builder));
}

// A trivial test aggregator that delegates aggregation to a function.
class FunctionAggregator final : public AggVectorAggregator<int> {
 public:
  using Func = std::function<int(int, int)>;

  FunctionAggregator(DataType dtype, TensorShape shape, Func agg_function)
      : AggVectorAggregator<int>(dtype, shape), agg_function_(agg_function) {}

 private:
  void AggregateVector(const AggVector<int>& agg_vector) override {
    for (auto [i, v] : agg_vector) {
      data()[i] = agg_function_(data()[i], v);
    }
  }

  const Func agg_function_;
};

// Factory for the FunctionAggregator.
class FunctionAggregatorFactory final : public TensorAggregatorFactory {
 public:
  explicit FunctionAggregatorFactory(FunctionAggregator::Func agg_function)
      : agg_function_(agg_function) {}

 private:
  absl::StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    if (intrinsic.inputs[0].dtype() != DT_INT32) {
      return absl::InvalidArgumentError("Unsupported dtype: expected DT_INT32");
    }
    return std::make_unique<FunctionAggregator>(intrinsic.inputs[0].dtype(),
                                                intrinsic.inputs[0].shape(),
                                                agg_function_);
  }

  absl::StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    return TFF_STATUS(UNIMPLEMENTED);
  }

  const FunctionAggregator::Func agg_function_;
};

TEST(CheckpointAggregatorTest, ConcurrentAccumulationAbortWhileQueued) {
  const int64_t kNumInputs = 10;
  const int64_t kNumInputsBeforeBlocking = 3;

  // Notifies the aggregation to unblock;
  absl::Notification resume_aggregation_notification;
  absl::Notification aggregation_blocked_notification;
  std::atomic<int> agg_counter = 0;
  FunctionAggregatorFactory agg_factory([&](int a, int b) {
    if (++agg_counter > kNumInputsBeforeBlocking &&
        !aggregation_blocked_notification.HasBeenNotified()) {
      aggregation_blocked_notification.Notify();
      resume_aggregation_notification.WaitForNotification();
    }
    return a + b;
  });
  RegisterAggregatorFactory("foo2_aggregation", &agg_factory);

  auto aggregator = Create(PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "foo2_aggregation"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb"));

  MockCheckpointParser parser;
  EXPECT_CALL(parser, GetTensor(StrEq("foo"))).WillRepeatedly(Invoke([&] {
    return Tensor::Create(DT_INT32, {}, CreateTestData({1}));
  }));

  // Schedule receiving inputs on 10 concurrent threads.
  auto scheduler = CreateThreadPoolScheduler(10);
  for (int64_t i = 0; i < kNumInputs; ++i) {
    scheduler->Schedule([&]() {
      EXPECT_THAT(aggregator->Accumulate(parser),
                  AnyOf(StatusIs(OK), StatusIs(ABORTED)));
    });
  }

  aggregation_blocked_notification.WaitForNotification();
  aggregator->Abort();
  resume_aggregation_notification.Notify();
  scheduler->WaitUntilIdle();
}

TEST(CheckpointAggregatorTest, SerializeAfterReport) {
  auto aggregator = CreateWithDefaultConfig();

  MockCheckpointBuilder builder;
  EXPECT_CALL(builder, Add(StrEq("foo_out"), IsTensor<int32_t>({}, {0})))
      .WillOnce(Return(absl::OkStatus()));
  TFF_EXPECT_OK(aggregator->Report(builder));
  EXPECT_THAT(std::move(*aggregator).Serialize(), StatusIs(ABORTED));
}

TEST(CheckpointAggregatorTest, SerializeAfterAbort) {
  auto aggregator = CreateWithDefaultConfig();
  aggregator->Abort();
  EXPECT_THAT(std::move(*aggregator).Serialize(), StatusIs(ABORTED));
}

TEST(CheckpointAggregatorTest, DeserializeInvalidState) {
  std::string serialized_state = "invalid";
  EXPECT_THAT(CheckpointAggregator::Deserialize(default_configuration(),
                                                serialized_state),
              StatusIs(INVALID_ARGUMENT));
}

INSTANTIATE_TEST_SUITE_P(
    CheckpointAggregatorTestInstantiation, CheckpointAggregatorTest,
    testing::ValuesIn<bool>({false, true}),
    [](const testing::TestParamInfo<CheckpointAggregatorTest::ParamType>&
           info) { return info.param ? "SerializeDeserialize" : "None"; });

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
