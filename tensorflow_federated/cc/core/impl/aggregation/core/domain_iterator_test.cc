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

#include <vector>

#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"

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

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
