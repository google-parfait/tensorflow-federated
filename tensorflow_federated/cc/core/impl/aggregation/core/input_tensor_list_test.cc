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

#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"

#include <cstdint>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::Eq;
using testing::Not;

class InputTensorListTest : public testing::Test {
 protected:
  InputTensorListTest()
      : t1_(Tensor::Create(DT_FLOAT, {1}, CreateTestData<float>({1})).value()),
        t2_(Tensor::Create(DT_INT32, {2}, CreateTestData<int32_t>({2, 3}))
                .value()),
        t3_(Tensor::Create(DT_INT64, {3}, CreateTestData<int64_t>({4, 5, 6}))
                .value()),
        t4_(Tensor::Create(DT_FLOAT, {4}, CreateTestData<float>({7, 8, 9, 10}))
                .value()),
        t5_(Tensor::Create(DT_INT32, {5},
                           CreateTestData<int32_t>({11, 12, 13, 14, 15}))
                .value()),
        t6_(Tensor::Create(DT_INT64, {6},
                           CreateTestData<int64_t>({16, 17, 18, 19, 20, 21}))
                .value()) {}

  InputTensorList CreateInlined() {
    return InputTensorList({&t1_, &t2_, &t3_});
  }

  InputTensorList CreateAllocated() {
    return InputTensorList({&t1_, &t2_, &t3_, &t4_, &t5_, &t6_});
  }

  Tensor t1_;
  Tensor t2_;
  Tensor t3_;
  Tensor t4_;
  Tensor t5_;
  Tensor t6_;
};

TEST_F(InputTensorListTest, Inlined_Size) {
  InputTensorList tensor_list = CreateInlined();
  EXPECT_THAT(tensor_list.size(), Eq(3));
}

TEST_F(InputTensorListTest, Inlined_Iterate) {
  InputTensorList tensor_list = CreateInlined();
  auto iter = tensor_list.begin();
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{1}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{2}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{3}));
  iter++;
  EXPECT_THAT(iter, Eq(tensor_list.end()));
}

TEST_F(InputTensorListTest, Inlined_MoveConstructor_Iterate) {
  InputTensorList moved_tensor_list = CreateInlined();
  InputTensorList tensor_list(std::move(moved_tensor_list));
  auto iter = tensor_list.begin();
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{1}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{2}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{3}));
  iter++;
  EXPECT_THAT(iter, Eq(tensor_list.end()));
}

TEST_F(InputTensorListTest, Inlined_MoveAssignment_Iterate) {
  InputTensorList moved_tensor_list = CreateInlined();
  // Initially, create the tensor list as an allocated tensor list before
  // assigning it to an inlined InputTensorList via move assignment.
  InputTensorList tensor_list = CreateAllocated();
  tensor_list = std::move(moved_tensor_list);
  auto iter = tensor_list.begin();
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{1}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{2}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{3}));
  iter++;
  EXPECT_THAT(iter, Eq(tensor_list.end()));

  // Assigning back to the moved variable is valid.
  moved_tensor_list = std::move(tensor_list);
  iter = moved_tensor_list.begin();
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{1}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{2}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{3}));
  iter++;
  EXPECT_THAT(iter, Eq(moved_tensor_list.end()));
}

TEST_F(InputTensorListTest, Inlined_ForEachLoop) {
  InputTensorList tensor_list = CreateInlined();
  uint64_t expected_size = 1;
  for (const Tensor* t : tensor_list) {
    EXPECT_THAT(t->num_elements(), Eq(expected_size));
    expected_size++;
  }
}

TEST_F(InputTensorListTest, Inlined_Iterate_MultiPassGuarantee) {
  // Ensure the iterator meets the multi-pass guarantee requirements required
  // by forward iterators.
  // (https://en.cppreference.com/w/cpp/iterator/forward_iterator)
  InputTensorList tensor_list = CreateInlined();
  auto iterI = tensor_list.begin();
  auto iterJ = tensor_list.begin();
  EXPECT_THAT(iterI, Eq(iterJ));
  EXPECT_THAT(*iterI, Eq(*iterJ));
  const Tensor* elem = *iterI;
  iterI++;
  // iterJ points to the same element as before even though iterI was moved
  // forward.
  EXPECT_THAT(elem, Eq(*iterJ));
  EXPECT_THAT(*iterI, Not(Eq(*iterJ)));
  // After both iterators are incremented the same number of times they should
  // again point to the same element.
  iterI++;
  iterJ++;
  iterJ++;
  EXPECT_THAT(*iterI, Eq(*iterJ));
}

TEST_F(InputTensorListTest, Inlined_Iterate_PostincrementAndPreincrement) {
  InputTensorList tensor_list = CreateInlined();
  auto iterI = tensor_list.begin();
  // If postincrement works as expected, iterJ will be set to the value of iterI
  // before it is incremented.
  auto iterJ = iterI++;
  EXPECT_THAT(iterJ, Eq(tensor_list.begin()));
  // If preincrement works as expected, iterK should be set to the value of
  // iterJ after it is incremented, which is now the same as iterI.
  auto iterK = ++iterJ;
  EXPECT_THAT(iterK, Eq(iterJ));
  EXPECT_THAT(iterK, Eq(iterI));
}

TEST_F(InputTensorListTest, Inlined_Index) {
  InputTensorList tensor_list = CreateInlined();
  EXPECT_THAT(tensor_list[0]->shape(), Eq(TensorShape{1}));
  EXPECT_THAT(tensor_list[1]->shape(), Eq(TensorShape{2}));
  EXPECT_THAT(tensor_list[2]->shape(), Eq(TensorShape{3}));
}

TEST_F(InputTensorListTest, Inlined_SizeConstructorAndMutableIndex) {
  InputTensorList tensor_list(3);
  tensor_list[0] = &t1_;
  tensor_list[1] = &t2_;
  tensor_list[2] = &t3_;

  EXPECT_THAT(tensor_list[0]->shape(), Eq(TensorShape{1}));
  EXPECT_THAT(tensor_list[1]->shape(), Eq(TensorShape{2}));
  EXPECT_THAT(tensor_list[2]->shape(), Eq(TensorShape{3}));
}

TEST_F(InputTensorListTest, Inlined_SizeConstructor_InitializesPointersToNull) {
  InputTensorList tensor_list(3);

  EXPECT_THAT(tensor_list[0], Eq(nullptr));
  EXPECT_THAT(tensor_list[1], Eq(nullptr));
  EXPECT_THAT(tensor_list[2], Eq(nullptr));
}

TEST_F(InputTensorListTest, Allocated_Size) {
  InputTensorList tensor_list = CreateAllocated();
  EXPECT_THAT(tensor_list.size(), Eq(6));
}

TEST_F(InputTensorListTest, Allocated_Iterate) {
  InputTensorList tensor_list = CreateAllocated();
  auto iter = tensor_list.begin();
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{1}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{2}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{3}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{4}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{5}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{6}));
  iter++;
  EXPECT_THAT(iter, Eq(tensor_list.end()));
}

TEST_F(InputTensorListTest, Allocated_MoveConstructor_Iterate) {
  InputTensorList moved_tensor_list = CreateAllocated();
  InputTensorList tensor_list(std::move(moved_tensor_list));
  auto iter = tensor_list.begin();
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{1}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{2}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{3}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{4}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{5}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{6}));
  iter++;
  EXPECT_THAT(iter, Eq(tensor_list.end()));
}

TEST_F(InputTensorListTest, Allocated_MoveAssignment_Iterate) {
  InputTensorList moved_tensor_list = CreateAllocated();
  // Initially, create the tensor list as an inlined tensor list before
  // assigning it to an inlined InputTensorList via move assignment.
  InputTensorList tensor_list = CreateInlined();
  tensor_list = std::move(moved_tensor_list);
  auto iter = tensor_list.begin();
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{1}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{2}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{3}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{4}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{5}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{6}));
  iter++;
  EXPECT_THAT(iter, Eq(tensor_list.end()));

  // Assigning back to the moved variable is valid.
  moved_tensor_list = std::move(tensor_list);
  iter = moved_tensor_list.begin();
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{1}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{2}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{3}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{4}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{5}));
  iter++;
  EXPECT_THAT((*iter)->shape(), Eq(TensorShape{6}));
  iter++;
  EXPECT_THAT(iter, Eq(moved_tensor_list.end()));
}

TEST_F(InputTensorListTest, Allocated_ForEachLoop) {
  InputTensorList tensor_list = CreateAllocated();
  uint64_t expected_size = 1;
  for (const Tensor* t : tensor_list) {
    EXPECT_THAT(t->num_elements(), Eq(expected_size));
    expected_size++;
  }
}

TEST_F(InputTensorListTest, Allocated_Iterate_MultiPassGuarantee) {
  // Ensure the iterator meets the multi-pass guarantee requirements required
  // by forward iterators
  // (https://en.cppreference.com/w/cpp/iterator/forward_iterator)
  InputTensorList tensor_list = CreateAllocated();
  auto iterI = tensor_list.begin();
  auto iterJ = tensor_list.begin();
  EXPECT_THAT(iterI, Eq(iterJ));
  EXPECT_THAT(*iterI, Eq(*iterJ));
  const Tensor* elem = *iterI;
  iterI++;
  // iterJ points to the same element as before even though iterI was moved
  // forward.
  EXPECT_THAT(elem, Eq(*iterJ));
  EXPECT_THAT(*iterI, Not(Eq(*iterJ)));
  // After both iterators are incremented the same number of times they should
  // again point to the same element.
  iterI++;
  iterJ++;
  iterJ++;
  EXPECT_THAT(*iterI, Eq(*iterJ));
}

TEST_F(InputTensorListTest, Allocated_Iterate_PostincrementAndPreincrement) {
  InputTensorList tensor_list = CreateAllocated();
  auto iterI = tensor_list.begin();
  // If postincrement works as expected, iterJ will be set to the value of iterI
  // before it is incremented.
  auto iterJ = iterI++;
  EXPECT_THAT(iterJ, Eq(tensor_list.begin()));
  // If preincrement works as expected, iterK should be set to the value of
  // iterJ after it is incremented, which is now the same as iterI.
  auto iterK = ++iterJ;
  EXPECT_THAT(iterK, Eq(iterJ));
  EXPECT_THAT(iterK, Eq(iterI));
}

TEST_F(InputTensorListTest, Allocated_Index) {
  InputTensorList tensor_list = CreateAllocated();
  EXPECT_THAT(tensor_list[0]->shape(), Eq(TensorShape{1}));
  EXPECT_THAT(tensor_list[1]->shape(), Eq(TensorShape{2}));
  EXPECT_THAT(tensor_list[2]->shape(), Eq(TensorShape{3}));
  EXPECT_THAT(tensor_list[3]->shape(), Eq(TensorShape{4}));
  EXPECT_THAT(tensor_list[4]->shape(), Eq(TensorShape{5}));
  EXPECT_THAT(tensor_list[5]->shape(), Eq(TensorShape{6}));
}

TEST_F(InputTensorListTest, Allocated_SizeConstructorAndMutableIndex) {
  InputTensorList tensor_list(6);
  tensor_list[0] = &t1_;
  tensor_list[1] = &t2_;
  tensor_list[2] = &t3_;
  tensor_list[3] = &t4_;
  tensor_list[4] = &t5_;
  tensor_list[5] = &t6_;

  EXPECT_THAT(tensor_list[0]->shape(), Eq(TensorShape{1}));
  EXPECT_THAT(tensor_list[1]->shape(), Eq(TensorShape{2}));
  EXPECT_THAT(tensor_list[2]->shape(), Eq(TensorShape{3}));
  EXPECT_THAT(tensor_list[3]->shape(), Eq(TensorShape{4}));
  EXPECT_THAT(tensor_list[4]->shape(), Eq(TensorShape{5}));
  EXPECT_THAT(tensor_list[5]->shape(), Eq(TensorShape{6}));
}

TEST_F(InputTensorListTest,
       Allocated_SizeConstructor_InitializesPointersToNull) {
  InputTensorList tensor_list(6);

  EXPECT_THAT(tensor_list[0], Eq(nullptr));
  EXPECT_THAT(tensor_list[1], Eq(nullptr));
  EXPECT_THAT(tensor_list[2], Eq(nullptr));
  EXPECT_THAT(tensor_list[3], Eq(nullptr));
  EXPECT_THAT(tensor_list[4], Eq(nullptr));
  EXPECT_THAT(tensor_list[5], Eq(nullptr));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated
