/*
 * Copyright 2022 Google LLC
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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TESTING_TESTING_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TESTING_TESTING_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_vector.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace tensorflow_federated::aggregation {

namespace tf = ::tensorflow;

// Convenience macros for `EXPECT_THAT(s, IsOk())`, where `s` is either
// a `Status` or a `StatusOr<T>`.
// Old versions of the protobuf library define EXPECT_OK as well, so we only
// conditionally define our version.
#if !defined(EXPECT_OK)
#define EXPECT_OK(result) EXPECT_THAT(result, IsOk())
#endif
#define ASSERT_OK(result) ASSERT_THAT(result, IsOk())

// Creates a temporary file name with given suffix unique for the running test.
std::string TemporaryTestFile(absl::string_view suffix);

template <typename T>
tf::Tensor CreateTfTensor(tf::DataType data_type,
                          std::initializer_list<int64_t> dim_sizes,
                          std::initializer_list<T> values) {
  tf::TensorShape shape;
  EXPECT_TRUE(tf::TensorShape::BuildTensorShape(dim_sizes, &shape).ok());
  tf::Tensor tensor(data_type, shape);
  T* tensor_data_ptr = reinterpret_cast<T*>(tensor.data());
  for (auto value : values) {
    *tensor_data_ptr++ = value;
  }
  return tensor;
}

tf::Tensor CreateStringTfTensor(std::initializer_list<int64_t> dim_sizes,
                                std::initializer_list<string_view> values);

// Wrapper around tf::ops::Save that sets up and runs the op.
absl::Status CreateTfCheckpoint(tf::Input filename, tf::Input tensor_names,
                                tf::InputList tensors);

// Returns a summary of the checkpoint as a map of tensor names and values.
absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
SummarizeCheckpoint(const absl::Cord& checkpoint);

// Converts a potentially sparse tensor to a flat vector of tensor values.
template <typename T>
std::vector<T> TensorValuesToVector(const Tensor& arg) {
  std::vector<T> vec(arg.num_elements());
  if (arg.num_elements() > 0) {
    AggVector<T> agg_vector = arg.AsAggVector<T>();
    for (auto [i, v] : agg_vector) {
      vec[i] = v;
    }
  }
  return vec;
}

// Writes description of a tensor shape to the ostream.
std::ostream& operator<<(std::ostream& os, const TensorShape& shape);

// Writes description of a tensor to the ostream.
template <typename T>
void DescribeTensor(::std::ostream* os, DataType dtype, TensorShape shape,
                    std::vector<T> values) {
  // Max number of tensor values to be printed.
  constexpr int kMaxValues = 100;
  // TODO: b/233915195 - Print dtype name instead of number.
  *os << "{dtype: " << dtype;
  *os << ", shape: ";
  *os << shape;
  *os << ", values: {";
  int num_values = 0;
  bool insert_comma = false;
  for (auto v : values) {
    if (++num_values > kMaxValues) {
      *os << "...";
      break;
    }
    if (insert_comma) {
      *os << ", ";
    }
    *os << v;
    insert_comma = true;
  }
  *os << "}}";
}

template <typename T>
std::string ToProtoContent(std::initializer_list<T> values) {
  return std::string(reinterpret_cast<char*>(std::vector(values).data()),
                     values.size() * sizeof(T));
}

template <>
inline std::string ToProtoContent(std::initializer_list<string_view> values) {
  // The following is the simplified version of serializing the string values
  // that works only for short strings that are shorter than 128 characters, in
  // which case string lengths can be encoded with one byte each.
  std::string content(values.size(), '\0');
  size_t index = 0;
  // Write sizes of strings first.
  for (string_view value : values) {
    TFF_CHECK(value.size() < 128);
    content[index++] = static_cast<char>(value.size());
  }
  // Append data of all strings.
  for (string_view value : values) {
    content.append(value.data(), value.size());
  }
  return content;
}

// Writes description of a tensor to the ostream.
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

// TensorMatcher implementation.
template <typename T>
class TensorMatcherImpl : public ::testing::MatcherInterface<const Tensor&> {
 public:
  TensorMatcherImpl(DataType expected_dtype, TensorShape expected_shape,
                    std::vector<T> expected_values)
      : expected_dtype_(expected_dtype),
        expected_shape_(expected_shape),
        expected_values_(expected_values) {}

  void DescribeTo(std::ostream* os) const override {
    DescribeTensor<T>(os, expected_dtype_, expected_shape_, expected_values_);
  }

  bool MatchAndExplain(
      const Tensor& arg,
      ::testing::MatchResultListener* listener) const override {
    return arg.dtype() == expected_dtype_ && arg.shape() == expected_shape_ &&
           TensorValuesToVector<T>(arg) == expected_values_;
  }

 private:
  DataType expected_dtype_;
  TensorShape expected_shape_;
  std::vector<T> expected_values_;
};

// TensorMatcher can be used to compare a tensor against an expected
// value type, shape, and the list of values.
template <typename T>
class TensorMatcher {
 public:
  explicit TensorMatcher(DataType expected_dtype, TensorShape expected_shape,
                         std::initializer_list<T> expected_values)
      : expected_dtype_(expected_dtype),
        expected_shape_(expected_shape),
        expected_values_(expected_values.begin(), expected_values.end()) {}
  // Intentionally allowed to be implicit.
  operator ::testing::Matcher<const Tensor&>() const {  // NOLINT
    return ::testing::MakeMatcher(new TensorMatcherImpl<T>(
        expected_dtype_, expected_shape_, expected_values_));
  }

 private:
  DataType expected_dtype_;
  TensorShape expected_shape_;
  std::vector<T> expected_values_;
};

template <typename T>
TensorMatcher<T> IsTensor(TensorShape expected_shape,
                          std::initializer_list<T> expected_values) {
  return TensorMatcher<T>(internal::TypeTraits<T>::kDataType, expected_shape,
                          expected_values);
}

// Writes description of an intrinsic to the ostream.
std::ostream& operator<<(std::ostream& os, const Intrinsic& intrinsic);

// IntrinsicMatcher implementation.
class IntrinsicMatcherImpl
    : public ::testing::MatcherInterface<const Intrinsic&> {
 public:
  explicit IntrinsicMatcherImpl(Intrinsic&& expected_intrinsic)
      : expected_intrinsic_(std::move(expected_intrinsic)) {
    for (int i = 0; i < expected_intrinsic_.nested_intrinsics.size(); ++i) {
      nested_intrinsic_matchers_.emplace_back(
          std::move(expected_intrinsic_.nested_intrinsics[i]));
    }
  }

  void DescribeTo(std::ostream* os) const override;

  bool MatchAndExplain(const Intrinsic& arg,
                       ::testing::MatchResultListener* listener) const override;

 private:
  Intrinsic expected_intrinsic_;
  std::vector<IntrinsicMatcherImpl> nested_intrinsic_matchers_;
};

// IntrinsicMatcher can be used to compare a tensor against an expected
// value type, shape, and the list of values.
class IntrinsicMatcher {
 public:
  explicit IntrinsicMatcher(Intrinsic expected_intrinsic)
      : matcher_(::testing::Matcher<const Intrinsic&>(
            new IntrinsicMatcherImpl(std::move(expected_intrinsic)))) {}
  operator ::testing::Matcher<const Intrinsic&>() {  // NOLINT
    return matcher_;
  }

 private:
  ::testing::Matcher<const Intrinsic&> matcher_;
};

::testing::Matcher<const Intrinsic&> EqIntrinsic(Intrinsic expected_intrinsic);

}  // namespace tensorflow_federated::aggregation

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TESTING_TESTING_H_
