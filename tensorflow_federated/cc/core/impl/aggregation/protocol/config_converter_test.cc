// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "tensorflow_federated/cc/core/impl/aggregation/testing/parse_text_proto.h"
// clang-format on
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using testing::SizeIs;

class MockFactory : public TensorAggregatorFactory {
  MOCK_METHOD(StatusOr<std::unique_ptr<TensorAggregator>>, Create,
              (const Intrinsic&), (const, override));
  MOCK_METHOD(StatusOr<std::unique_ptr<TensorAggregator>>, Deserialize,
              (const Intrinsic&, std::string), (const, override));
};

class ConfigConverterTest : public ::testing::Test {
 protected:
  ConfigConverterTest() {
    if (!is_registered_) {
      MockFactory mock_factory;
      RegisterAggregatorFactory("my_intrinsic", &mock_factory);
      RegisterAggregatorFactory("inner_intrinsic", &mock_factory);
      RegisterAggregatorFactory("outer_intrinsic", &mock_factory);
      RegisterAggregatorFactory("other_intrinsic", &mock_factory);
      RegisterAggregatorFactory("fedsql_group_by", &mock_factory);
      RegisterAggregatorFactory("fedsql_dp_group_by", &mock_factory);
      RegisterAggregatorFactory("GoogleSQL:sum", &mock_factory);
      RegisterAggregatorFactory("GoogleSQL:max", &mock_factory);
      RegisterAggregatorFactory("GoogleSQL:$differential_privacy_sum",
                                &mock_factory);
      is_registered_ = true;
    }
  }

 private:
  static bool is_registered_;
};

bool ConfigConverterTest::is_registered_ = false;

TEST_F(ConfigConverterTest, ConvertEmpty) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: { intrinsic_uri: "my_intrinsic" }
  )pb");
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  ASSERT_THAT(parsed_intrinsics.value()[0],
              EqIntrinsic(Intrinsic{"my_intrinsic", {}, {}, {}, {}}));
}

TEST_F(ConfigConverterTest, ConvertInputs) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "my_intrinsic"
      intrinsic_args {
        input_tensor {
          name: "foo"
          dtype: DT_INT32
          shape { dim_sizes: 8 }
        }
      }
      intrinsic_args {
        input_tensor {
          name: "bar"
          dtype: DT_FLOAT
          shape { dim_sizes: 2 dim_sizes: 3 }
        }
      }
    }
  )pb");

  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  ASSERT_THAT(parsed_intrinsics.value()[0],
              EqIntrinsic(Intrinsic{"my_intrinsic",
                                    {TensorSpec{"foo", DT_INT32, {8}},
                                     TensorSpec{"bar", DT_FLOAT, {2, 3}}},
                                    {},
                                    {},
                                    {}}));
}

TEST_F(ConfigConverterTest, ConvertOutputs) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "my_intrinsic"
      output_tensors {
        name: "foo_out"
        dtype: DT_INT32
        shape { dim_sizes: 16 }
      }
      output_tensors {
        name: "bar_out"
        dtype: DT_FLOAT
        shape { dim_sizes: 3 dim_sizes: 4 }
      }
    }
  )pb");
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  ASSERT_THAT(parsed_intrinsics.value()[0],
              EqIntrinsic(Intrinsic{"my_intrinsic",
                                    {},
                                    {TensorSpec{"foo_out", DT_INT32, {16}},
                                     TensorSpec{"bar_out", DT_FLOAT, {3, 4}}},
                                    {},
                                    {}}));
}

TEST_F(ConfigConverterTest, ConvertParams) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "my_intrinsic"
      intrinsic_args {
        parameter {
          dtype: DT_FLOAT
          shape { dim_sizes: 2 dim_sizes: 3 }
        }
      }
    }
  )pb");
  std::initializer_list<float> values{1, 2, 3, 4, 5, 6};
  std::string data =
      std::string(reinterpret_cast<char*>(std::vector(values).data()),
                  values.size() * sizeof(float));
  *config.mutable_intrinsic_configs(0)
       ->mutable_intrinsic_args(0)
       ->mutable_parameter()
       ->mutable_content() = data;
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  Tensor expected_tensor =
      Tensor::Create(DT_FLOAT, {2, 3}, CreateTestData<float>(values)).value();
  Intrinsic expected{"my_intrinsic", {}, {}, {}, {}};
  expected.parameters.push_back(std::move(expected_tensor));
  ASSERT_THAT(parsed_intrinsics.value()[0], EqIntrinsic(std::move(expected)));
}

TEST_F(ConfigConverterTest, ConvertInnerAggregations) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "my_intrinsic"
      inner_intrinsics {
        intrinsic_uri: "inner_intrinsic"
        intrinsic_args {
          input_tensor {
            name: "foo"
            dtype: DT_INT32
            shape { dim_sizes: 8 }
          }
        }
        output_tensors {
          name: "foo_out"
          dtype: DT_INT32
          shape { dim_sizes: 16 }
        }
      }
    }
  )pb");
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  Intrinsic expected_inner = Intrinsic{"inner_intrinsic",
                                       {TensorSpec{"foo", DT_INT32, {8}}},
                                       {TensorSpec{"foo_out", DT_INT32, {16}}},
                                       {},
                                       {}};
  Intrinsic expected{"my_intrinsic", {}, {}, {}, {}};
  expected.nested_intrinsics.push_back(std::move(expected_inner));
  ASSERT_THAT(parsed_intrinsics.value()[0], EqIntrinsic(std::move(expected)));
}

TEST_F(ConfigConverterTest, ConvertFedSql_GroupByAlreadyPresent) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "fedsql_group_by"
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
        shape { dim_sizes: -1 }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "bar"
            dtype: DT_INT32
            shape {}
          }
        }
        output_tensors {
          name: "bar_out"
          dtype: DT_INT32
          shape {}
        }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:max"
        intrinsic_args {
          input_tensor {
            name: "baz"
            dtype: DT_INT32
            shape {}
          }
        }
        output_tensors {
          name: "baz_out"
          dtype: DT_INT32
          shape {}
        }
      }
    }
  )pb");
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  Intrinsic expected_sum = Intrinsic{"GoogleSQL:sum",
                                     {TensorSpec{"bar", DT_INT32, {-1}}},
                                     {TensorSpec{"bar_out", DT_INT32, {-1}}},
                                     {},
                                     {}};
  Intrinsic expected_max = Intrinsic{"GoogleSQL:max",
                                     {TensorSpec{"baz", DT_INT32, {-1}}},
                                     {TensorSpec{"baz_out", DT_INT32, {-1}}},
                                     {},
                                     {}};

  Intrinsic expected{"fedsql_group_by",
                     {TensorSpec{"foo", DT_INT32, {-1}}},
                     {TensorSpec{"foo_out", DT_INT32, {-1}}},
                     {},
                     {}};
  expected.nested_intrinsics.push_back(std::move(expected_sum));
  expected.nested_intrinsics.push_back(std::move(expected_max));
  ASSERT_THAT(parsed_intrinsics.value()[0], EqIntrinsic(std::move(expected)));
}

TEST_F(ConfigConverterTest, ConvertFedSql_WrapWhenGroupByNotPresent) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "GoogleSQL:sum"
      intrinsic_args {
        input_tensor {
          name: "bar"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "bar_out"
        dtype: DT_INT32
        shape {}
      }
    }
    intrinsic_configs: {
      intrinsic_uri: "other_intrinsic"
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
    intrinsic_configs: {
      intrinsic_uri: "GoogleSQL:max"
      intrinsic_args {
        input_tensor {
          name: "baz"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "baz_out"
        dtype: DT_INT32
        shape {}
      }
    }
  )pb");
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  // Even though there are three top level intrinsics in the configuration, the
  // two fedsql intrinsics should be wrapped by a group by intrinsic so only two
  // toplevel intrinsics will be present in the output.
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(2));
  Intrinsic expected_sum = Intrinsic{"GoogleSQL:sum",
                                     {TensorSpec{"bar", DT_INT32, {-1}}},
                                     {TensorSpec{"bar_out", DT_INT32, {-1}}},
                                     {},
                                     {}};
  Intrinsic expected_max = Intrinsic{"GoogleSQL:max",
                                     {TensorSpec{"baz", DT_INT32, {-1}}},
                                     {TensorSpec{"baz_out", DT_INT32, {-1}}},
                                     {},
                                     {}};

  Intrinsic expected_other{"other_intrinsic",
                           {TensorSpec{"foo", DT_INT32, {}}},
                           {TensorSpec{"foo_out", DT_INT32, {}}},
                           {},
                           {}};
  Intrinsic expected_groupby{"fedsql_group_by", {}, {}, {}, {}};
  expected_groupby.nested_intrinsics.push_back(std::move(expected_sum));
  expected_groupby.nested_intrinsics.push_back(std::move(expected_max));
  ASSERT_THAT(parsed_intrinsics.value()[0],
              EqIntrinsic(std::move(expected_other)));
  ASSERT_THAT(parsed_intrinsics.value()[1],
              EqIntrinsic(std::move(expected_groupby)));
}

TEST_F(ConfigConverterTest, ConvertFedSql_WrapWhenGroupByNotPresent_Nested) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "outer_intrinsic"
      inner_intrinsics: {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "bar"
            dtype: DT_INT32
            shape {}
          }
        }
        output_tensors {
          name: "bar_out"
          dtype: DT_INT32
          shape {}
        }
      }
      inner_intrinsics: {
        intrinsic_uri: "other_intrinsic"
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
      inner_intrinsics: {
        intrinsic_uri: "GoogleSQL:max"
        intrinsic_args {
          input_tensor {
            name: "baz"
            dtype: DT_INT32
            shape {}
          }
        }
        output_tensors {
          name: "baz_out"
          dtype: DT_INT32
          shape {}
        }
      }
    }
  )pb");
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  Intrinsic expected_sum = Intrinsic{"GoogleSQL:sum",
                                     {TensorSpec{"bar", DT_INT32, {-1}}},
                                     {TensorSpec{"bar_out", DT_INT32, {-1}}},
                                     {},
                                     {}};
  Intrinsic expected_max = Intrinsic{"GoogleSQL:max",
                                     {TensorSpec{"baz", DT_INT32, {-1}}},
                                     {TensorSpec{"baz_out", DT_INT32, {-1}}},
                                     {},
                                     {}};
  Intrinsic expected_other{"other_intrinsic",
                           {TensorSpec{"foo", DT_INT32, {}}},
                           {TensorSpec{"foo_out", DT_INT32, {}}},
                           {},
                           {}};
  Intrinsic expected_groupby{"fedsql_group_by", {}, {}, {}, {}};
  expected_groupby.nested_intrinsics.push_back(std::move(expected_sum));
  expected_groupby.nested_intrinsics.push_back(std::move(expected_max));
  Intrinsic expected_outer{"outer_intrinsic", {}, {}, {}, {}};
  expected_outer.nested_intrinsics.push_back(std::move(expected_other));
  expected_outer.nested_intrinsics.push_back(std::move(expected_groupby));
  ASSERT_THAT(parsed_intrinsics.value()[0],
              EqIntrinsic(std::move(expected_outer)));
}

TEST_F(ConfigConverterTest, ConvertFedSqlDp_GroupByAlreadyPresent) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "fedsql_dp_group_by"
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
        shape { dim_sizes: -1 }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:$differential_privacy_sum"
        intrinsic_args {
          input_tensor {
            name: "bar"
            dtype: DT_INT32
            shape {}
          }
        }
        output_tensors {
          name: "bar_out"
          dtype: DT_INT64
          shape {}
        }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:$differential_privacy_sum"
        intrinsic_args {
          input_tensor {
            name: "baz"
            dtype: DT_INT32
            shape {}
          }
        }
        output_tensors {
          name: "baz_out"
          dtype: DT_INT64
          shape {}
        }
      }
    }
  )pb");
  StatusOr<std::vector<Intrinsic>> parsed_intrinsics = ParseFromConfig(config);
  ASSERT_THAT(parsed_intrinsics, IsOk());
  ASSERT_THAT(parsed_intrinsics.value(), SizeIs(1));
  Intrinsic expected_sum_1 = Intrinsic{"GoogleSQL:$differential_privacy_sum",
                                       {TensorSpec{"bar", DT_INT32, {-1}}},
                                       {TensorSpec{"bar_out", DT_INT64, {-1}}},
                                       {},
                                       {}};
  Intrinsic expected_sum_2 = Intrinsic{"GoogleSQL:$differential_privacy_sum",
                                       {TensorSpec{"baz", DT_INT32, {-1}}},
                                       {TensorSpec{"baz_out", DT_INT64, {-1}}},
                                       {},
                                       {}};

  Intrinsic expected{"fedsql_dp_group_by",
                     {TensorSpec{"foo", DT_INT32, {-1}}},
                     {TensorSpec{"foo_out", DT_INT32, {-1}}},
                     {},
                     {}};
  expected.nested_intrinsics.push_back(std::move(expected_sum_1));
  expected.nested_intrinsics.push_back(std::move(expected_sum_2));
  ASSERT_THAT(parsed_intrinsics.value()[0], EqIntrinsic(std::move(expected)));
}

TEST_F(ConfigConverterTest, ConvertFedSqlDp_GroupByNotPresent) {
  Configuration config = PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs {
      intrinsic_uri: "GoogleSQL:$differential_privacy_sum"
      intrinsic_args {
        input_tensor {
          name: "bar"
          dtype: DT_INT32
          shape {}
        }
      }
      output_tensors {
        name: "bar_out"
        dtype: DT_INT64
        shape {}
      }
    }
  )pb");
  Status s = ParseFromConfig(config).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              ::testing::HasSubstr("Inner DP SQL intrinsics must already be "
                                   "wrapped with an outer DP SQL intrinsic"));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
