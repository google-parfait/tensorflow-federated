/*
 * Copyright 2023 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"

#include <climits>
#include <cstdint>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/core/impl/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using testing::IsFalse;
using testing::IsTrue;
using testing::TestWithParam;

using OneDimGroupingAggregatorTest = TestWithParam<bool>;

// A simple Sum Aggregator
template <typename InputT, typename OutputT = InputT>
class SumGroupingAggregator final
    : public OneDimGroupingAggregator<InputT, OutputT> {
 public:
  using OneDimGroupingAggregator<InputT, OutputT>::OneDimGroupingAggregator;
  using OneDimGroupingAggregator<InputT, OutputT>::data;

  static SumGroupingAggregator<InputT, OutputT> FromProto(
      const OneDimGroupingAggregatorState& aggregator_state) {
    return SumGroupingAggregator<InputT, OutputT>(
        MutableVectorData<OutputT>::CreateFromEncodedContent(
            aggregator_state.vector_data()),
        aggregator_state.num_inputs());
  }

 private:
  void AggregateVectorByOrdinals(
      const AggVector<int64_t>& ordinals_vector,
      const AggVector<InputT>& value_vector) override {
    auto value_it = value_vector.begin();
    for (auto o : ordinals_vector) {
      int64_t output_index = o.value;
      // If this function returned a failed Status at this point, the
      // data_vector_ may have already been partially modified, leaving the
      // GroupingAggregator in a bad state. Thus, check that the indices of the
      // ordinals tensor and the data tensor match with TFF_CHECK instead.
      //
      // TODO: b/266974165 - Revisit the constraint that the indices of the
      // values must match the indices of the ordinals when sparse tensors are
      // implemented. It may be possible for the value to be omitted for a given
      // ordinal in which case the default value should be used.
      TFF_CHECK(value_it.index() == o.index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";
      // Delegate the actual aggregation to the specific aggregation
      // intrinsic implementation.
      AggregateValue(output_index, value_it++.value());
    }
  }

  void MergeVectorByOrdinals(const AggVector<int64_t>& ordinals_vector,
                             const AggVector<OutputT>& value_vector) override {
    auto value_it = value_vector.begin();
    for (auto o : ordinals_vector) {
      int64_t output_index = o.value;
      TFF_CHECK(value_it.index() == o.index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";
      AggregateValue(output_index, value_it++.value());
    }
  }

  inline void AggregateValue(int64_t i, OutputT value) { data()[i] += value; }

  OutputT GetDefaultValue() override { return static_cast<OutputT>(0); }
};

// A simple Min Aggregator that works for int32_t
class MinGroupingAggregator final : public OneDimGroupingAggregator<int32_t> {
 public:
  using OneDimGroupingAggregator<int32_t>::OneDimGroupingAggregator;
  using OneDimGroupingAggregator<int32_t>::data;

  static MinGroupingAggregator FromProto(
      const OneDimGroupingAggregatorState& aggregator_state) {
    return MinGroupingAggregator(
        MutableVectorData<int32_t>::CreateFromEncodedContent(
            aggregator_state.vector_data()),
        aggregator_state.num_inputs());
  }

 private:
  void AggregateVectorByOrdinals(
      const AggVector<int64_t>& ordinals_vector,
      const AggVector<int32_t>& value_vector) override {
    auto value_it = value_vector.begin();
    for (auto o : ordinals_vector) {
      int64_t output_index = o.value;
      // If this function returned a failed Status at this point, the
      // data_vector_ may have already been partially modified, leaving the
      // GroupingAggregator in a bad state. Thus, check that the indices of the
      // ordinals tensor and the data tensor match with TFF_CHECK instead.
      //
      // TODO: b/266974165 - Revisit the constraint that the indices of the
      // values must match the indices of the ordinals when sparse tensors are
      // implemented. It may be possible for the value to be omitted for a given
      // ordinal in which case the default value should be used.
      TFF_CHECK(value_it.index() == o.index)
          << "Indices in AggVector of ordinals and AggVector of values "
             "are mismatched.";
      // Delegate the actual aggregation to the specific aggregation
      // intrinsic implementation.
      AggregateValue(output_index, value_it++.value());
    }
  }

  void MergeVectorByOrdinals(const AggVector<int64_t>& ordinals_vector,
                             const AggVector<int32_t>& value_vector) override {
    AggregateVectorByOrdinals(ordinals_vector, value_vector);
  }

  inline void AggregateValue(int64_t i, int32_t value) {
    if (value < data()[i]) {
      data()[i] = value;
    }
  }
  int32_t GetDefaultValue() override { return INT_MAX; }
};

TEST_P(OneDimGroupingAggregatorTest, EmptyReport) {
  SumGroupingAggregator<int32_t> aggregator;

  if (GetParam()) {
    auto state = std::move(aggregator).ToProto();
    aggregator = SumGroupingAggregator<int32_t>::FromProto(state);
  }

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor<int32_t>({0}, {}));
}

TEST_P(OneDimGroupingAggregatorTest, ScalarAggregation_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t1}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t2}), IsOk());

  if (GetParam()) {
    auto state = std::move(aggregator).ToProto();
    aggregator = SumGroupingAggregator<int32_t>::FromProto(state);
  }

  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t3}), IsOk());
  EXPECT_THAT(aggregator.CanReport(), IsTrue());

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({1}, {6}));
}

TEST_P(OneDimGroupingAggregatorTest, DenseAggregation_Succeeds) {
  const TensorShape shape = {4};
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinals =
      Tensor::Create(DT_INT64, shape, CreateTestData<int64_t>({0, 1, 2, 3}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinals, &t1}), IsOk());
  EXPECT_THAT(aggregator.Accumulate({&ordinals, &t2}), IsOk());

  if (GetParam()) {
    auto state = std::move(aggregator).ToProto();
    aggregator = SumGroupingAggregator<int32_t>::FromProto(state);
  }

  EXPECT_THAT(aggregator.Accumulate({&ordinals, &t3}), IsOk());
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor(shape, {14, 19, 23, 49}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(OneDimGroupingAggregatorTest, DifferentOrdinalsPerAccumulate_Succeeds) {
  const TensorShape shape = {4};
  SumGroupingAggregator<int32_t> aggregator;
  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, shape, CreateTestData<int64_t>({3, 3, 2, 0}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, shape, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator.Accumulate({&t1_ordinals, &t1}), IsOk());
  // Totals: [27, 0, 15, 4]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, shape, CreateTestData<int64_t>({1, 0, 1, 4}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, shape, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator.Accumulate({&t2_ordinals, &t2}), IsOk());

  if (GetParam()) {
    auto state = std::move(aggregator).ToProto();
    aggregator = SumGroupingAggregator<int32_t>::FromProto(state);
  }

  // Totals: [32, 11, 15, 4, 2]
  Tensor t3_ordinals =
      Tensor::Create(DT_INT64, shape, CreateTestData<int64_t>({2, 2, 5, 1}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, shape, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator.Accumulate({&t3_ordinals, &t3}), IsOk());
  // Totals: [32, 31, 29, 4, 2, 7]
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({6}, {32, 31, 29, 4, 2, 7}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(OneDimGroupingAggregatorTest, DifferentShapesPerAccumulate_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({2, 0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({17, 3})).value();
  EXPECT_THAT(aggregator.Accumulate({&t1_ordinals, &t1}), IsOk());
  // Totals: [3, 0, 17]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {6}, CreateTestData<int64_t>({1, 0, 1, 4, 3, 0}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 13, 2, 4, 5}))
          .value();
  EXPECT_THAT(aggregator.Accumulate({&t2_ordinals, &t2}), IsOk());

  if (GetParam()) {
    auto state = std::move(aggregator).ToProto();
    aggregator = SumGroupingAggregator<int32_t>::FromProto(state);
  }

  // Totals: [13, 23, 17, 4, 2]
  Tensor t3_ordinals =
      Tensor::Create(DT_INT64, {5}, CreateTestData<int64_t>({2, 2, 1, 0, 4}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {5}, CreateTestData({3, 11, 7, 6, 3})).value();
  EXPECT_THAT(aggregator.Accumulate({&t3_ordinals, &t3}), IsOk());
  // Totals: [13, 30, 31, 4, 2]
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {19, 30, 31, 4, 5}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(OneDimGroupingAggregatorTest,
       DifferentShapesPerAccumulate_NonzeroDefaultValue_Succeeds) {
  // Use a MinGroupingAggregator which has a non-zero default value so we can
  // test that when the output grows, elements are set to the default value.
  MinGroupingAggregator aggregator;
  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({2, 0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({17, 3})).value();
  EXPECT_THAT(aggregator.Accumulate({&t1_ordinals, &t1}), IsOk());
  // Totals: [3, INT_MAX, 17]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {6}, CreateTestData<int64_t>({0, 0, 0, 4, 4, 0}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 13, 2, 4, -50}))
          .value();
  EXPECT_THAT(aggregator.Accumulate({&t2_ordinals, &t2}), IsOk());

  if (GetParam()) {
    auto state = std::move(aggregator).ToProto();
    aggregator = MinGroupingAggregator::FromProto(state);
  }

  // Totals: [-50, INT_MAX, 17, INT_MAX, 2]
  Tensor t3_ordinals =
      Tensor::Create(DT_INT64, {5}, CreateTestData<int64_t>({2, 2, 1, 0, 4}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {5}, CreateTestData({33, 11, 7, 6, 3})).value();
  EXPECT_THAT(aggregator.Accumulate({&t3_ordinals, &t3}), IsOk());
  // Totals: [-50, 7, 11, INT_MAX, 2]
  EXPECT_THAT(aggregator.CanReport(), IsTrue());
  EXPECT_THAT(aggregator.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {-50, 7, 11, INT_MAX, 2}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(OneDimGroupingAggregatorTest, Merge_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;
  Tensor ordinal =
      Tensor::Create(DT_INT64, {1}, CreateTestData<int64_t>({0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {1}, CreateTestData({1})).value();
  Tensor t2 = Tensor::Create(DT_INT32, {1}, CreateTestData({2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {1}, CreateTestData({3})).value();
  EXPECT_THAT(aggregator1.Accumulate({&ordinal, &t1}), IsOk());
  EXPECT_THAT(aggregator2.Accumulate({&ordinal, &t2}), IsOk());
  EXPECT_THAT(aggregator2.Accumulate({&ordinal, &t3}), IsOk());

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = SumGroupingAggregator<int32_t>::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = SumGroupingAggregator<int32_t>::FromProto(state2);
  }

  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  EXPECT_THAT(aggregator2_num_inputs, Eq(2));
  auto aggregator2_result = std::move(aggregator2).Report();
  EXPECT_THAT(aggregator2_result, IsOk());
  EXPECT_THAT(aggregator2_result->size(), Eq(1));
  Tensor aggregator2_result_tensor = std::move(aggregator2_result.value()[0]);
  EXPECT_THAT(aggregator2_result_tensor, IsTensor<int32_t>({1}, {5}));
  EXPECT_THAT(aggregator1.MergeTensors({&ordinal, &aggregator2_result_tensor},
                                       aggregator2_num_inputs),
              IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor({1}, {6}));
}

TEST_P(OneDimGroupingAggregatorTest, Merge_BothEmpty_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = SumGroupingAggregator<int32_t>::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = SumGroupingAggregator<int32_t>::FromProto(state2);
  }

  // Merge the two empty aggregators together.
  auto empty_ordinals =
      Tensor::Create(DT_INT64, {0}, CreateTestData<int64_t>({})).value();
  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  EXPECT_THAT(aggregator1.MergeTensors({&empty_ordinals, &aggregator2_result},
                                       aggregator2_num_inputs),
              IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(0));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), Eq(1));
  EXPECT_THAT(result.value()[0], IsTensor<int32_t>({0}, {}));
}

TEST_P(OneDimGroupingAggregatorTest, Merge_ThisOutputEmpty_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;

  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({3, 3, 2, 0}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator2.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator2 totals: [27, 0, 15, 4]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({1, 0, 1, 4}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator2.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator2 totals: [32, 11, 15, 4, 2]

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = SumGroupingAggregator<int32_t>::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = SumGroupingAggregator<int32_t>::FromProto(state2);
  }

  // Merge aggregator2 into aggregator1 which has not received any inputs.
  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  auto ordinals_for_merge =
      Tensor::Create(DT_INT64, {5}, CreateTestData<int64_t>({2, 6, 1, 4, 3}))
          .value();
  EXPECT_THAT(
      aggregator1.MergeTensors({&ordinals_for_merge, &aggregator2_result},
                               aggregator2_num_inputs),
      IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(2));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({7}, {0, 15, 32, 2, 4, 0, 11}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(OneDimGroupingAggregatorTest, Merge_OtherOutputEmpty_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;

  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({3, 3, 2, 0}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator1 totals: [27, 0, 15, 4]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({1, 0, 1, 4}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator1 totals: [32, 11, 15, 4, 2]

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = SumGroupingAggregator<int32_t>::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = SumGroupingAggregator<int32_t>::FromProto(state2);
  }

  // Merge with aggregator2 which has not received any inputs.
  auto empty_ordinals =
      Tensor::Create(DT_INT64, {0}, CreateTestData<int64_t>({})).value();
  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  EXPECT_THAT(aggregator1.MergeTensors({&empty_ordinals, &aggregator2_result},
                                       aggregator2_num_inputs),
              IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(2));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {32, 11, 15, 4, 2}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(OneDimGroupingAggregatorTest,
       Merge_OtherOutputHasFewerElements_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;

  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({3, 3, 2, 0}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator1 totals: [27, 0, 15, 4]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({1, 0, 1, 4}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator1 totals: [32, 11, 15, 4, 2]

  Tensor t3_ordinals =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({2, 2})).value();
  Tensor t3 = Tensor::Create(DT_INT32, {2}, CreateTestData({3, 11})).value();
  EXPECT_THAT(aggregator2.Accumulate({&t3_ordinals, &t3}), IsOk());
  // aggregator2 totals: [0, 0, 14]

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = SumGroupingAggregator<int32_t>::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = SumGroupingAggregator<int32_t>::FromProto(state2);
  }

  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  auto ordinals_for_merge =
      Tensor::Create(DT_INT64, {3}, CreateTestData<int64_t>({2, 0, 1})).value();
  EXPECT_THAT(
      aggregator1.MergeTensors({&ordinals_for_merge, &aggregator2_result},
                               aggregator2_num_inputs),
      IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {32, 25, 15, 4, 2}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(OneDimGroupingAggregatorTest,
       Merge_OtherOutputHasMoreElements_Succeeds) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;

  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({3, 3, 2, 0}))
          .value();
  Tensor t1 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({1, 3, 15, 27})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator1 totals: [27, 0, 15, 4]
  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({1, 0, 1, 4}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({10, 5, 1, 2})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator1 totals: [32, 11, 15, 4, 2]

  Tensor t3_ordinals =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({2, 2, 5, 1}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {4}, CreateTestData({3, 11, 7, 20})).value();
  EXPECT_THAT(aggregator2.Accumulate({&t3_ordinals, &t3}), IsOk());
  // aggregator2 totals: [0, 20, 14, 0, 0, 7]

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = SumGroupingAggregator<int32_t>::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = SumGroupingAggregator<int32_t>::FromProto(state2);
  }

  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  auto ordinals_for_merge =
      Tensor::Create(DT_INT64, {6}, CreateTestData<int64_t>({1, 3, 5, 0, 2, 4}))
          .value();
  EXPECT_THAT(
      aggregator1.MergeTensors({&ordinals_for_merge, &aggregator2_result},
                               aggregator2_num_inputs),
      IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({6}, {32, 11, 15, 24, 9, 14}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST_P(OneDimGroupingAggregatorTest,
       Merge_OtherOutputHasMoreElements_NonzeroDefaultValue_Succeeds) {
  // Use a MinGroupingAggregator which has a non-zero default value so we can
  // test that when the output grows, elements are set to the default value.
  MinGroupingAggregator aggregator1;
  MinGroupingAggregator aggregator2;
  Tensor t1_ordinals =
      Tensor::Create(DT_INT64, {2}, CreateTestData<int64_t>({2, 0})).value();
  Tensor t1 = Tensor::Create(DT_INT32, {2}, CreateTestData({-17, 3})).value();
  EXPECT_THAT(aggregator1.Accumulate({&t1_ordinals, &t1}), IsOk());
  // aggregator1 totals: [3, INT_MAX, -17]

  Tensor t2_ordinals =
      Tensor::Create(DT_INT64, {6}, CreateTestData<int64_t>({0, 0, 0, 4, 4, 0}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {6}, CreateTestData({10, 5, 13, 2, 4, -50}))
          .value();
  EXPECT_THAT(aggregator2.Accumulate({&t2_ordinals, &t2}), IsOk());
  // aggregator2 totals: [-50, INT_MAX, INT_MAX, INT_MAX, 2]
  Tensor t3_ordinals =
      Tensor::Create(DT_INT64, {5}, CreateTestData<int64_t>({2, 2, 1, 0, 4}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {5}, CreateTestData({33, 11, 7, 6, 3})).value();
  EXPECT_THAT(aggregator2.Accumulate({&t3_ordinals, &t3}), IsOk());
  // aggregator2 totals: [-50, 7, 11, INT_MAX, 2]

  if (GetParam()) {
    auto state1 = std::move(aggregator1).ToProto();
    aggregator1 = MinGroupingAggregator::FromProto(state1);
    auto state2 = std::move(aggregator2).ToProto();
    aggregator2 = MinGroupingAggregator::FromProto(state2);
  }

  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  auto ordinals_for_merge =
      Tensor::Create(DT_INT64, {5}, CreateTestData<int64_t>({4, 3, 2, 1, 0}))
          .value();
  EXPECT_THAT(
      aggregator1.MergeTensors({&ordinals_for_merge, &aggregator2_result},
                               aggregator2_num_inputs),
      IsOk());
  EXPECT_THAT(aggregator1.CanReport(), IsTrue());
  EXPECT_THAT(aggregator1.GetNumInputs(), Eq(3));

  auto result = std::move(aggregator1).Report();
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(result.value().size(), Eq(1));
  // Verify the resulting tensor.
  EXPECT_THAT(result.value()[0], IsTensor({5}, {2, INT_MAX, -17, 7, -50}));
  // Also ensure that the resulting tensor is dense.
  EXPECT_TRUE(result.value()[0].is_dense());
}

TEST(OneDimGroupingAggregatorTest,
     Aggregate_OrdinalTensorHasIncompatibleDataType) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal =
      Tensor::Create(DT_INT32, {}, CreateTestData<int32_t>({0})).value();
  Tensor t = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({0})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}),
              StatusIs(INVALID_ARGUMENT));
}

TEST(OneDimGroupingAggregatorTest, Aggregate_IncompatibleDataType) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t = Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({0})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}),
              StatusIs(INVALID_ARGUMENT));
}

TEST(OneDimGroupingAggregatorTest,
     Aggregate_OrdinalAndValueTensorsHaveIncompatibleShapes) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t = Tensor::Create(DT_INT32, {2}, CreateTestData({0, 1})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}),
              StatusIs(INVALID_ARGUMENT));
}

TEST(OneDimGroupingAggregatorTest,
     Aggregate_MultidimensionalTensorsNotSupported) {
  SumGroupingAggregator<int32_t> aggregator;
  Tensor ordinal =
      Tensor::Create(DT_INT64, {2, 2}, CreateTestData<int64_t>({0, 0, 0, 0}))
          .value();
  Tensor t =
      Tensor::Create(DT_INT32, {2, 2}, CreateTestData({0, 1, 2, 3})).value();
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}),
              StatusIs(INVALID_ARGUMENT));
}

TEST(OneDimGroupingAggregatorTest, Merge_IncompatibleDataType) {
  SumGroupingAggregator<int32_t> aggregator1;
  SumGroupingAggregator<float> aggregator2;

  auto empty_ordinals =
      Tensor::Create(DT_INT64, {0}, CreateTestData<int64_t>({})).value();
  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  Status s = aggregator1.MergeTensors({&empty_ordinals, &aggregator2_result},
                                      aggregator2_num_inputs);
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("dtype mismatch"));
}

TEST(OneDimGroupingAggregatorTest, Merge_IncompatibleInputDataType) {
  SumGroupingAggregator<int32_t, int64_t> aggregator1;
  SumGroupingAggregator<int32_t> aggregator2;

  auto empty_ordinals =
      Tensor::Create(DT_INT64, {0}, CreateTestData<int64_t>({})).value();
  int aggregator2_num_inputs = aggregator2.GetNumInputs();
  auto aggregator2_result =
      std::move(std::move(aggregator2).Report().value()[0]);
  Status s = aggregator1.MergeTensors({&empty_ordinals, &aggregator2_result},
                                      aggregator2_num_inputs);
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("dtype mismatch"));
}

TEST(OneDimGroupingAggregatorTest, FailsAfterBeingConsumed) {
  Tensor ordinal =
      Tensor::Create(DT_INT64, {}, CreateTestData<int64_t>({0})).value();
  Tensor t = Tensor::Create(DT_INT32, {}, CreateTestData({0})).value();
  SumGroupingAggregator<int32_t> aggregator;
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}), IsOk());
  EXPECT_THAT(std::move(aggregator).Report(), IsOk());

  // Now the aggregator instance has been consumed and should fail any
  // further operations.
  EXPECT_THAT(aggregator.CanReport(), IsFalse());  // NOLINT
  EXPECT_THAT(std::move(aggregator).Report(),
              StatusIs(FAILED_PRECONDITION));         // NOLINT
  EXPECT_THAT(aggregator.Accumulate({&ordinal, &t}),  // NOLINT
              StatusIs(FAILED_PRECONDITION));
}

TEST(OneDimGroupingAggregatorTest, Serialize_Unimplmeneted) {
  SumGroupingAggregator<int32_t> aggregator;
  Status s = std::move(aggregator).Serialize().status();
  EXPECT_THAT(s, StatusIs(UNIMPLEMENTED));
}

INSTANTIATE_TEST_SUITE_P(
    OneDimGroupingAggregatorTestInstantiation, OneDimGroupingAggregatorTest,
    testing::ValuesIn<bool>({false, true}),
    [](const testing::TestParamInfo<OneDimGroupingAggregatorTest::ParamType>&
           info) { return info.param ? "SaveIntermediateState" : "None"; });

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
