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
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::Eq;

TEST(CompositeKeyCombinerTest, EmptyInput_Invalid) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT});
  StatusOr<Tensor> result = combiner.Accumulate(InputTensorList({}));
  ASSERT_THAT(result, StatusIs(INVALID_ARGUMENT));
}

TEST(CompositeKeyCombinerTest, InputWithWrongShapeTensor_Invalid) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT, DT_INT32});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {4}, CreateTestData<int32_t>({1, 2, 3, 4}))
          .value();
  StatusOr<Tensor> result = combiner.Accumulate(InputTensorList({&t1, &t2}));
  ASSERT_THAT(result, StatusIs(INVALID_ARGUMENT));
}

TEST(CompositeKeyCombinerTest, InputWithTooFewTensorsInvalid) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT, DT_INT32});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3})).value();
  StatusOr<Tensor> result = combiner.Accumulate(InputTensorList({&t1}));
  ASSERT_THAT(result, StatusIs(INVALID_ARGUMENT));
}

TEST(CompositeKeyCombinerTest, InputWithTooManyTensors_Invalid) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT, DT_INT32});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3})).value();
  Tensor t3 =
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({4, 5, 6})).value();
  StatusOr<Tensor> result =
      combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
  ASSERT_THAT(result, StatusIs(INVALID_ARGUMENT));
}

TEST(CompositeKeyCombinerTest, InputWithWrongTypes_Invalid) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT, DT_STRING});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3})).value();
  StatusOr<Tensor> result = combiner.Accumulate(InputTensorList({&t1, &t2}));
  ASSERT_THAT(result, StatusIs(INVALID_ARGUMENT));
}

TEST(CompositeKeyCombinerTest, OutputBeforeAccumulateOutputsEmptyTensor) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT});
  OutputTensorList output = combiner.GetOutputKeys();
  EXPECT_THAT(output.size(), Eq(1));
  EXPECT_THAT(output[0], IsTensor<float>({0}, {}));
}

TEST(CompositeKeyCombinerTest, AccumulateAndOutput_SingleElement) {
  CompositeKeyCombiner combiner(std::vector<DataType>{DT_FLOAT});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {1}, CreateTestData<float>({1.3})).value();
  StatusOr<Tensor> result = combiner.Accumulate(InputTensorList({&t1}));
  ASSERT_OK(result);
  EXPECT_THAT(result.value(), IsTensor<int64_t>({1}, {0}));

  OutputTensorList output = combiner.GetOutputKeys();
  EXPECT_THAT(output.size(), Eq(1));
  EXPECT_THAT(output[0], IsTensor<float>({1}, {1.3}));
}

TEST(CompositeKeyCombinerTest, AccumulateAndOutput_NumericTypes) {
  CompositeKeyCombiner combiner(
      std::vector<DataType>{DT_FLOAT, DT_INT32, DT_INT64});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {3}, CreateTestData<int32_t>({1, 2, 3})).value();
  Tensor t3 =
      Tensor::Create(DT_INT64, {3}, CreateTestData<int64_t>({4, 5, 6})).value();
  StatusOr<Tensor> result =
      combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
  ASSERT_OK(result);
  EXPECT_THAT(result.value(), IsTensor<int64_t>({3}, {0, 1, 2}));

  OutputTensorList output = combiner.GetOutputKeys();
  EXPECT_THAT(output.size(), Eq(3));
  EXPECT_THAT(output[0], IsTensor<float>({3}, {1.1, 1.2, 1.3}));
  EXPECT_THAT(output[1], IsTensor<int32_t>({3}, {1, 2, 3}));
  EXPECT_THAT(output[2], IsTensor<int64_t>({3}, {4, 5, 6}));
}

TEST(CompositeKeyCombinerTest,
     NumericTypes_SameKeysResultInSameOrdinalsAcrossAccumulateCalls) {
  CompositeKeyCombiner combiner(
      std::vector<DataType>{DT_FLOAT, DT_INT32, DT_INT64});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({1.1, 1.2, 1.1, 1.2}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_INT32, {4}, CreateTestData<int32_t>({1, 2, 3, 2}))
          .value();
  Tensor t3 =
      Tensor::Create(DT_INT64, {4}, CreateTestData<int64_t>({4, 5, 6, 5}))
          .value();
  StatusOr<Tensor> result1 =
      combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
  ASSERT_OK(result1);
  EXPECT_THAT(result1.value(), IsTensor<int64_t>({4}, {0, 1, 2, 1}));

  // Across different calls to Accumulate, tensors can have different shape.
  Tensor t4 = Tensor::Create(DT_FLOAT, {5},
                             CreateTestData<float>({1.2, 1.1, 1.1, 1.1, 1.2}))
                  .value();
  Tensor t5 =
      Tensor::Create(DT_INT32, {5}, CreateTestData<int32_t>({2, 3, 2, 3, 2}))
          .value();
  Tensor t6 =
      Tensor::Create(DT_INT64, {5}, CreateTestData<int64_t>({5, 6, 5, 6, 5}))
          .value();
  StatusOr<Tensor> result2 =
      combiner.Accumulate(InputTensorList({&t4, &t5, &t6}));
  ASSERT_OK(result2);
  EXPECT_THAT(result2.value(), IsTensor<int64_t>({5}, {1, 2, 3, 2, 1}));

  OutputTensorList output = combiner.GetOutputKeys();
  EXPECT_THAT(output.size(), Eq(3));
  EXPECT_THAT(output[0], IsTensor<float>({4}, {1.1, 1.2, 1.1, 1.1}));
  EXPECT_THAT(output[1], IsTensor<int32_t>({4}, {1, 2, 3, 2}));
  EXPECT_THAT(output[2], IsTensor<int64_t>({4}, {4, 5, 6, 5}));
}

TEST(CompositeKeyCombinerTest, AccumulateAndOutput_StringTypes) {
  CompositeKeyCombiner combiner(
      std::vector<DataType>{DT_FLOAT, DT_STRING, DT_STRING});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {3}, CreateTestData<float>({1.1, 1.2, 1.3}))
          .value();
  Tensor t2 = Tensor::Create(DT_STRING, {3},
                             CreateTestData<string_view>({"abc", "de", ""}))
                  .value();
  Tensor t3 =
      Tensor::Create(DT_STRING, {3},
                     CreateTestData<string_view>({"fghi", "jklmn", "o"}))
          .value();
  StatusOr<Tensor> result =
      combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
  ASSERT_OK(result);
  EXPECT_THAT(result.value(), IsTensor<int64_t>({3}, {0, 1, 2}));

  OutputTensorList output = combiner.GetOutputKeys();
  EXPECT_THAT(output.size(), Eq(3));
  EXPECT_THAT(output[0], IsTensor<float>({3}, {1.1, 1.2, 1.3}));
  EXPECT_THAT(output[1], IsTensor<string_view>({3}, {"abc", "de", ""}));
  EXPECT_THAT(output[2], IsTensor<string_view>({3}, {"fghi", "jklmn", "o"}));
}

TEST(CompositeKeyCombinerTest,
     StringTypes_SameCompositeKeysResultInSameOrdinalsAcrossAccumulateCalls) {
  CompositeKeyCombiner combiner(
      std::vector<DataType>{DT_FLOAT, DT_STRING, DT_STRING});
  Tensor t1 =
      Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({1.1, 1.2, 1.2, 1.3}))
          .value();
  Tensor t2 =
      Tensor::Create(DT_STRING, {4},
                     CreateTestData<string_view>({"abc", "de", "de", ""}))
          .value();
  Tensor t3 = Tensor::Create(
                  DT_STRING, {4},
                  CreateTestData<string_view>({"fghi", "jklmn", "jklmn", "o"}))
                  .value();
  StatusOr<Tensor> result1 =
      combiner.Accumulate(InputTensorList({&t1, &t2, &t3}));
  ASSERT_OK(result1);
  EXPECT_THAT(result1.value(), IsTensor<int64_t>({4}, {0, 1, 1, 2}));

  // Across different calls to Accumulate, tensors can have different shape.
  Tensor t4 = Tensor::Create(DT_FLOAT, {5},
                             CreateTestData<float>({1.3, 1.4, 1.1, 1.2, 1.1}))
                  .value();
  Tensor t5 = Tensor::Create(
                  DT_STRING, {5},
                  CreateTestData<string_view>({"", "abc", "abc", "de", "abc"}))
                  .value();
  Tensor t6 =
      Tensor::Create(
          DT_STRING, {5},
          CreateTestData<string_view>({"o", "pqrs", "fghi", "jklmn", "fghi"}))
          .value();
  StatusOr<Tensor> result2 =
      combiner.Accumulate(InputTensorList({&t4, &t5, &t6}));
  ASSERT_OK(result2);
  EXPECT_THAT(result2.value(), IsTensor<int64_t>({5}, {2, 3, 0, 1, 0}));

  OutputTensorList output = combiner.GetOutputKeys();
  EXPECT_THAT(output.size(), Eq(3));
  EXPECT_THAT(output[0], IsTensor<float>({4}, {1.1, 1.2, 1.3, 1.4}));
  EXPECT_THAT(output[1], IsTensor<string_view>({4}, {"abc", "de", "", "abc"}));
  EXPECT_THAT(output[2],
              IsTensor<string_view>({4}, {"fghi", "jklmn", "o", "pqrs"}));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
