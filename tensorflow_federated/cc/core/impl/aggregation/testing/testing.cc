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

#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"

#include <initializer_list>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "googletest/include/gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/platform.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated::aggregation {

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  DTYPE_CASES(tensor.dtype(), T,
              DescribeTensor<T>(&os, tensor.dtype(), tensor.shape(),
                                TensorValuesToVector<T>(tensor)));
  return os;
}

std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
  os << "{";
  bool insert_comma = false;
  for (auto dim_size : shape.dim_sizes()) {
    if (insert_comma) {
      os << ", ";
    }
    os << dim_size;
    insert_comma = true;
  }
  os << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const TensorSpec& spec) {
  os << "{ name: " << spec.name();
  os << ", shape: " << spec.shape();
  // TODO: b/233915195 - Print dtype name instead of number.
  os << ", dtype: " << spec.dtype();
  os << " }";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Intrinsic& intrinsic) {
  os << "{uri: " << intrinsic.uri;
  os << ", inputs: {";
  bool insert_comma = false;
  for (const TensorSpec& input : intrinsic.inputs) {
    if (insert_comma) {
      os << ", ";
    }
    os << input;
    insert_comma = true;
  }
  insert_comma = false;
  os << "}, outputs: {";
  for (const TensorSpec& output : intrinsic.outputs) {
    if (insert_comma) {
      os << ", ";
    }
    os << output;
    insert_comma = true;
  }
  insert_comma = false;
  os << "}, parameters: {";
  for (const Tensor& parameter : intrinsic.parameters) {
    if (insert_comma) {
      os << ", ";
    }
    os << parameter;
    insert_comma = true;
  }
  os << "}, nested_intrinsics: {";
  insert_comma = false;
  for (const Intrinsic& nested : intrinsic.nested_intrinsics) {
    if (insert_comma) {
      os << ", ";
    }
    os << nested;
    insert_comma = true;
  }
  os << "}}";
  return os;
}

std::string TestName() {
  auto test_info = testing::UnitTest::GetInstance()->current_test_info();
  return absl::StrReplaceAll(test_info->name(), {{"/", "_"}});
}

std::string TemporaryTestFile(absl::string_view suffix) {
  return ConcatPath(StripTrailingPathSeparator(testing::TempDir()),
                    absl::StrCat(TestName(), suffix));
}

void IntrinsicMatcherImpl::DescribeTo(std::ostream* os) const {
  *os << "{uri: " << expected_intrinsic_.uri;
  *os << ", inputs: {";
  bool insert_comma = false;
  for (const TensorSpec& input : expected_intrinsic_.inputs) {
    if (insert_comma) {
      *os << ", ";
    }
    *os << input;
    insert_comma = true;
  }
  insert_comma = false;
  *os << "}, outputs: {";
  for (const TensorSpec& output : expected_intrinsic_.outputs) {
    if (insert_comma) {
      *os << ", ";
    }
    *os << output;
    insert_comma = true;
  }
  insert_comma = false;
  *os << "}, parameters: {";
  for (const Tensor& parameter : expected_intrinsic_.parameters) {
    if (insert_comma) {
      *os << ", ";
    }
    DTYPE_CASES(parameter.dtype(), T,
                DescribeTensor<T>(os, parameter.dtype(), parameter.shape(),
                                  TensorValuesToVector<T>(parameter)));
    insert_comma = true;
  }
  insert_comma = false;
  *os << "}, nested_intrinsics: {";
  for (const IntrinsicMatcherImpl& nested : nested_intrinsic_matchers_) {
    if (insert_comma) {
      *os << ", ";
    }
    nested.DescribeTo(os);
    insert_comma = true;
  }
  *os << "}}";
}

bool IntrinsicMatcherImpl::MatchAndExplain(
    const Intrinsic& arg, ::testing::MatchResultListener* listener) const {
  if (expected_intrinsic_.nested_intrinsics.size() !=
      arg.nested_intrinsics.size()) {
    return false;
  }
  for (int i = 0; i < nested_intrinsic_matchers_.size(); ++i) {
    if (!nested_intrinsic_matchers_[i].MatchAndExplain(arg.nested_intrinsics[i],
                                                       listener))
      return false;
  }

  if (expected_intrinsic_.parameters.size() != arg.parameters.size()) {
    return false;
  }
  for (int i = 0; i < expected_intrinsic_.parameters.size(); ++i) {
    const Tensor& tensor = expected_intrinsic_.parameters[i];
    bool tensor_match = false;
    DTYPE_CASES(
        tensor.dtype(), T,
        tensor_match = TensorMatcherImpl<T>(tensor.dtype(), tensor.shape(),
                                            TensorValuesToVector<T>(tensor))
                           .MatchAndExplain(arg.parameters[i], listener));
    if (!tensor_match) return false;
  }

  return arg.uri == expected_intrinsic_.uri &&
         arg.inputs == expected_intrinsic_.inputs &&
         arg.outputs == expected_intrinsic_.outputs;
}

template <>
bool TensorApproximatelyMatch(const Tensor& tensor,
                              std::vector<string_view> expected_values,
                              std::optional<string_view> tolerance) {
  return TensorValuesToVector<string_view>(tensor) == expected_values;
}

::testing::Matcher<const Intrinsic&> EqIntrinsic(Intrinsic expected_intrinsic) {
  return IntrinsicMatcher(std::move(expected_intrinsic));
}
}  // namespace tensorflow_federated::aggregation
