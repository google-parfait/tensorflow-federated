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
#include "tensorflow_federated/cc/core/impl/aggregation/core/domain_iterator.h"

#include <cstdint>
#include <vector>

#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

TEST(DPClosedDomainHistogramTest, SuccessfulIteration_Keys) {
  // Create domain_tensors
  std::vector<Tensor> vector_of_domains;
  vector_of_domains.push_back(
      Tensor::Create(DT_STRING, {3},
                     CreateTestData<string_view>({"a", "b", "c"}))
          .value());
  vector_of_domains.push_back(
      Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>({"d", "e"}))
          .value());
  TensorSpan domain_tensors(vector_of_domains.data(), 2);

  // Create a DomainIterator
  // When which_key = 0, output should be a, b, c, a, b, c
  internal::DomainIteratorForKeys domain_iterator0(domain_tensors, 0);
  std::vector<string_view> expected_keys0 = {"a", "b", "c", "a", "b", "c"};
  int i = 0;
  while (!domain_iterator0.wrapped_around()) {
    const void* key_ptr = *domain_iterator0;
    EXPECT_EQ(reinterpret_cast<const string_view*>(key_ptr)->data(),
              expected_keys0[i++]);
    ++domain_iterator0;
  }
  // When which_key = 1, output should be d, d, d, e, e, e, ...
  internal::DomainIteratorForKeys domain_iterator1(domain_tensors, 1);
  std::vector<string_view> expected_keys1 = {"d", "d", "d", "e", "e", "e"};
  i = 0;
  while (!domain_iterator1.wrapped_around()) {
    const void* key_ptr = *domain_iterator1;
    EXPECT_EQ(reinterpret_cast<const string_view*>(key_ptr)->data(),
              expected_keys1[i++]);
    ++domain_iterator1;
  }
}

TEST(DPClosedDomainHistogramTest, SuccessfulIteration_Aggregations) {
  // Create domain_tensors
  std::vector<Tensor> vector_of_domains;
  vector_of_domains.push_back(
      Tensor::Create(DT_STRING, {3},
                     CreateTestData<string_view>({"a", "b", "c"}))
          .value());
  vector_of_domains.push_back(
      Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>({"d", "e"}))
          .value());
  TensorSpan domain_tensors(vector_of_domains.data(), 2);

  // Create DPCompositeKeyCombiner and populate it with (c, e) and (a, d)
  DPCompositeKeyCombiner dp_key_combiner({DT_STRING, DT_STRING});
  Tensor column0 =
      Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>({"c", "a"}))
          .value();
  Tensor column1 =
      Tensor::Create(DT_STRING, {2}, CreateTestData<string_view>({"e", "d"}))
          .value();
  InputTensorList itl({&column0, &column1});
  auto accumulate_result = dp_key_combiner.Accumulate(itl);
  TFF_ASSERT_OK(accumulate_result.status());

  // Create two aggregations
  OutputTensorList otl;
  otl.push_back(
      Tensor::Create(DT_INT32, {2}, CreateTestData<int32_t>({11, 35})).value());
  otl.push_back(
      Tensor::Create(DT_INT32, {2}, CreateTestData<int32_t>({89, -3})).value());

  // Create a DomainIterator for aggregation 0
  internal::DomainIteratorForAggregations domain_iterator0(
      domain_tensors, otl[0], dp_key_combiner);
  // The iterator goes through (a, d), (b, d), (c, d), (a, e), (b, e), (c, e)
  // which means the first outcome of operator*() should be a pointer to 35 and
  // the last should be a pointer to 11. All others should be nullptr.
  int i = 0;
  while (!domain_iterator0.wrapped_around()) {
    if (i == 0) {
      EXPECT_EQ(*reinterpret_cast<const int32_t*>(*domain_iterator0), 35);
    } else if (i == 5) {
      EXPECT_EQ(*reinterpret_cast<const int32_t*>(*domain_iterator0), 11);
    } else {
      EXPECT_EQ(*domain_iterator0, nullptr);
    }
    ++domain_iterator0;
    ++i;
  }
  // Repeat for aggregation 1. The first and last are -3 and 89 respectively.
  internal::DomainIteratorForAggregations domain_iterator1(
      domain_tensors, otl[1], dp_key_combiner);
  i = 0;
  while (!domain_iterator1.wrapped_around()) {
    if (i == 0) {
      EXPECT_EQ(*reinterpret_cast<const int32_t*>(*domain_iterator1), -3);
    } else if (i == 5) {
      EXPECT_EQ(*reinterpret_cast<const int32_t*>(*domain_iterator1), 89);
    } else {
      EXPECT_EQ(*domain_iterator1, nullptr);
    }
    ++domain_iterator1;
    ++i;
  }
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
