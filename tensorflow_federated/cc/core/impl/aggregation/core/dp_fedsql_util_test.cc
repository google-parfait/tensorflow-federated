/*
 * Copyright 2026 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_util.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/parse_text_proto.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

class MockFactory : public TensorAggregatorFactory {
 public:
  MOCK_METHOD(StatusOr<std::unique_ptr<TensorAggregator>>, Create,
              (const Intrinsic&), (const, override));
  MOCK_METHOD(StatusOr<std::unique_ptr<TensorAggregator>>, Deserialize,
              (const Intrinsic&, std::string), (const, override));
};

REGISTER_AGGREGATOR_FACTORY("fedsql_dp_group_by", MockFactory);

namespace foo {
using FooFactory = MockFactory;
REGISTER_AGGREGATOR_FACTORY("foo", FooFactory);
}  // namespace foo

class PopulateDPParametersTest : public ::testing::Test {
 protected:
  static std::vector<Intrinsic> LoadConfig(absl::string_view config) {
    Configuration agg_config = PARSE_TEXT_PROTO(config);
    auto intrinsics = ParseFromConfig(agg_config);
    TFF_CHECK(intrinsics.ok()) << intrinsics.status();
    return std::move(*intrinsics);
  }
};

// Tune the L_1 parameter.
TEST_F(PopulateDPParametersTest, TuneL1) {
  auto intrinsics = LoadConfig(R"pb(
    intrinsic_configs:
    [ {
      intrinsic_uri: "fedsql_dp_group_by"
      intrinsic_args:
      [ { parameter { dtype: DT_DOUBLE double_val: 1.0 } }
        , { parameter { dtype: DT_DOUBLE double_val: 0.00001 } }
        , { parameter { dtype: DT_INT64 int64_val: -1 } }]
      inner_intrinsics:
      [ {
        intrinsic_args:
        [ {
          input_tensor {
            name: "input 1"
            dtype: DT_INT32
            shape { dim_sizes: -1 }
          }
        }
          , {
            input_tensor {
              name: "input 2"
              dtype: DT_INT32
              shape { dim_sizes: -1 }
            }
          }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }]
      }]
    }]
  )pb");

  // Replace the -1.0 value in position 1 of the inner parameters with 0.5.
  absl::flat_hash_map<std::string, double> parameters = {
      {"L1_0_estimated", 0.5}};
  EXPECT_OK(PopulateDPParameters(intrinsics[0], parameters));
  EXPECT_THAT(intrinsics[0].parameters,
              ElementsAre(IsScalarTensor(1.0), IsScalarTensor(0.00001),
                          IsScalarTensor(-1L)));
  EXPECT_THAT(intrinsics[0].nested_intrinsics[0].parameters,
              ElementsAre(IsScalarTensor(-1.0), IsScalarTensor(0.5),
                          IsScalarTensor(-1.0)));
}

// Tune (> 1 instances of) the L_inf parameter.
TEST_F(PopulateDPParametersTest, TuneLinf) {
  auto intrinsics = LoadConfig(R"pb(
    intrinsic_configs:
    [ {
      intrinsic_uri: "fedsql_dp_group_by"
      intrinsic_args:
      [ { parameter { dtype: DT_DOUBLE double_val: 1.0 } }
        , { parameter { dtype: DT_DOUBLE double_val: 0.00001 } }
        , { parameter { dtype: DT_INT64 int64_val: 10 } }]
      inner_intrinsics:
      [ {
        intrinsic_args:
        [ {
          input_tensor {
            name: "input 1"
            dtype: DT_INT32
            shape { dim_sizes: -1 }
          }
        }
          , {
            input_tensor {
              name: "input 2"
              dtype: DT_INT32
              shape { dim_sizes: -1 }
            }
          }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }]
      }
        , {
          intrinsic_args:
          [ {
            input_tensor {
              name: "input 1"
              dtype: DT_INT32
              shape { dim_sizes: -1 }
            }
          }
            , {
              input_tensor {
                name: "input 2"
                dtype: DT_INT32
                shape { dim_sizes: -1 }
              }
            }
            , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }
            , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }
            , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }]
        }]
    }]
  )pb");

  // Replace the -1.0 values in position 0 of inner parameters with 0.5 and 0.7.
  absl::flat_hash_map<std::string, double> parameters = {
      {"Linf_0_estimated", 0.5}, {"Linf_1_estimated", 0.7}};
  EXPECT_OK(PopulateDPParameters(intrinsics[0], parameters));

  EXPECT_THAT(intrinsics[0].parameters,
              ElementsAre(IsScalarTensor(1.0), IsScalarTensor(0.00001),
                          IsScalarTensor(10L)));
  EXPECT_THAT(intrinsics[0].nested_intrinsics[0].parameters,
              ElementsAre(IsScalarTensor(0.5), IsScalarTensor(-1.0),
                          IsScalarTensor(-1.0)));
  EXPECT_THAT(intrinsics[0].nested_intrinsics[1].parameters,
              ElementsAre(IsScalarTensor(0.7), IsScalarTensor(-1.0),
                          IsScalarTensor(-1.0)));
}

// Tune the max_groups_contributed parameter.
TEST_F(PopulateDPParametersTest, TuneL0) {
  auto intrinsics = LoadConfig(R"pb(
    intrinsic_configs:
    [ {
      intrinsic_uri: "fedsql_dp_group_by"
      intrinsic_args:
      [ { parameter { dtype: DT_DOUBLE double_val: 1.0 } }
        , { parameter { dtype: DT_DOUBLE double_val: 0.00001 } }
        , { parameter { dtype: DT_INT64 int64_val: -1 } }]
      inner_intrinsics:
      [ {
        intrinsic_args:
        [ {
          input_tensor {
            name: "input 1"
            dtype: DT_INT32
            shape { dim_sizes: -1 }
          }
        }
          , {
            input_tensor {
              name: "input 2"
              dtype: DT_INT32
              shape { dim_sizes: -1 }
            }
          }
          , { parameter { dtype: DT_DOUBLE double_val: 0.5 } }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }]
      }
        , {
          intrinsic_args:
          [ {
            input_tensor {
              name: "input 1"
              dtype: DT_INT32
              shape { dim_sizes: -1 }
            }
          }
            , {
              input_tensor {
                name: "input 2"
                dtype: DT_INT32
                shape { dim_sizes: -1 }
              }
            }
            , { parameter { dtype: DT_DOUBLE double_val: 0.7 } }
            , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }
            , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }]
        }]
    }]
  )pb");

  // Replace the -1 value in position 2 of top-level parameters with 10.
  absl::flat_hash_map<std::string, double> parameters = {
      {"max_groups_contributed_estimated", 10}};
  EXPECT_OK(PopulateDPParameters(intrinsics[0], parameters));

  EXPECT_THAT(intrinsics[0].parameters,
              ElementsAre(IsScalarTensor(1.0), IsScalarTensor(0.00001),
                          IsScalarTensor(10L)));
  EXPECT_THAT(intrinsics[0].nested_intrinsics[0].parameters,
              ElementsAre(IsScalarTensor(0.5), IsScalarTensor(-1.0),
                          IsScalarTensor(-1.0)));
  EXPECT_THAT(intrinsics[0].nested_intrinsics[1].parameters,
              ElementsAre(IsScalarTensor(0.7), IsScalarTensor(-1.0),
                          IsScalarTensor(-1.0)));
}

TEST_F(PopulateDPParametersTest, ValidateParameterMaps) {
  auto intrinsics = LoadConfig(R"pb(
    intrinsic_configs:
    [ {
      intrinsic_uri: "fedsql_dp_group_by"
      intrinsic_args:
      [ { parameter { dtype: DT_DOUBLE double_val: 1.0 } }
        , { parameter { dtype: DT_DOUBLE double_val: 0.00001 } }
        , { parameter { dtype: DT_INT64 int64_val: -1 } }]
      inner_intrinsics:
      [ {
        intrinsic_args:
        [ {
          input_tensor {
            name: "input 1"
            dtype: DT_INT32
            shape { dim_sizes: -1 }
          }
        }
          , {
            input_tensor {
              name: "input 2"
              dtype: DT_INT32
              shape { dim_sizes: -1 }
            }
          }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }]
      }]
    }]
  )pb");

  // Negative parameter values are invalid.
  EXPECT_THAT(PopulateDPParameters(intrinsics[0], {{"L1_0_estimated", -1.0}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be positive")));

  // Nested intrinsics index out of bounds.
  EXPECT_THAT(PopulateDPParameters(intrinsics[0], {{"L1_1_estimated", 1.0}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be in [0, 1) but got 1")));

  // Test all mismatch cases in the loop.
  std::vector<std::string> mismatch_cases = {
      "foo",
      "_estimated",
      "foo_estimated",
      "max_groups_contributed",
      "max_groups_contributed__estimated",
      "max_groups_contributedaaa_estimated",
      "max_groups_contributed0_estimated",
      "L1_",
      "L1__estimated_",
      "L1_aaa_estimated",
      "Linf_",
      "Linf__estimated_",
      "Linf_aaa_estimated",
  };

  for (const auto& mismatch_case : mismatch_cases) {
    EXPECT_THAT(PopulateDPParameters(intrinsics[0], {{mismatch_case, 1.0}}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("must match one of")));
  }
}

TEST_F(PopulateDPParametersTest, ValidateIntrinsicUri) {
  auto intrinsics = LoadConfig(R"pb(
    intrinsic_configs:
    [ { intrinsic_uri: "foo" }]
  )pb");
  EXPECT_THAT(
      PopulateDPParameters(intrinsics[0], {}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Unsupported intrinsic uri for DP parameter population")));
}

TEST_F(PopulateDPParametersTest, PopulatedAlreadySetParameters) {
  auto intrinsics = LoadConfig(R"pb(
    intrinsic_configs:
    [ {
      intrinsic_uri: "fedsql_dp_group_by"
      intrinsic_args:
      [ { parameter { dtype: DT_DOUBLE double_val: 1.0 } }
        , { parameter { dtype: DT_DOUBLE double_val: 0.00001 } }
        , { parameter { dtype: DT_INT64 int64_val: -1 } }]
      inner_intrinsics:
      [ {
        intrinsic_args:
        [ {
          input_tensor {
            name: "input 1"
            dtype: DT_INT32
            shape { dim_sizes: -1 }
          }
        }
          , {
            input_tensor {
              name: "input 2"
              dtype: DT_INT32
              shape { dim_sizes: -1 }
            }
          }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }
          , { parameter { dtype: DT_DOUBLE double_val: 0.5 } }
          , { parameter { dtype: DT_DOUBLE double_val: -1.0 } }]
      }]
    }]
  )pb");
  EXPECT_THAT(PopulateDPParameters(intrinsics[0], {{"L1_0_estimated", 0.5}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected untuned L_1 to be -1 "
                                 "but got 0.5 instead.")));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
