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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DATATYPE_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DATATYPE_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"

namespace tensorflow_federated {
namespace aggregation {

using string_view = absl::string_view;

namespace internal {

// The type kind, which indicates what sort of operations are valid to
// perform on the type.
enum TypeKind {
  kUnknown = 0,
  kNumeric = 1,
  kString = 2,
};

// Returns the TypeKind for the given DataType.
TypeKind GetTypeKind(DataType dtype);

// This struct is used to map typename T to DataType and specify other traits
// of typename T.
template <typename T>
struct TypeTraits {
  constexpr static DataType kDataType = DT_INVALID;
};

#define MATCH_TYPE_AND_DTYPE(TYPE, DTYPE, TYPE_KIND) \
  template <>                                        \
  struct TypeTraits<TYPE> {                          \
    constexpr static DataType kDataType = DTYPE;     \
    constexpr static TypeKind type_kind = TYPE_KIND; \
  }

// Mapping of native types to DT_ types.
// TODO: b/222605809 - Add other types.
MATCH_TYPE_AND_DTYPE(float, DT_FLOAT, TypeKind::kNumeric);
MATCH_TYPE_AND_DTYPE(double, DT_DOUBLE, TypeKind::kNumeric);
MATCH_TYPE_AND_DTYPE(int32_t, DT_INT32, TypeKind::kNumeric);
MATCH_TYPE_AND_DTYPE(int64_t, DT_INT64, TypeKind::kNumeric);
MATCH_TYPE_AND_DTYPE(uint64_t, DT_UINT64, TypeKind::kNumeric);
MATCH_TYPE_AND_DTYPE(string_view, DT_STRING, TypeKind::kString);

// The macros DTYPE_CASE and DTYPE_CASES are used to translate Tensor DataType
// to strongly typed calls of code parameterized with the template typename
// TYPE_ARG.
//
// For example, let's say there is a function that takes an AggVector<T>:
// template <typename T>
// void DoSomething(AggVector<T> agg_vector) { ... }
//
// Given a Tensor, the following code can be used to make a DoSomething call:
// DTYPE_CASES(tensor.dtype(), T, DoSomething(tensor.AsAggVector<T>()));
//
// The second parameter specifies the type argument to be used as the template
// parameter in the statement in the third argument.

#define SINGLE_ARG(...) __VA_ARGS__
#define DTYPE_CASE(TYPE, TYPE_ARG, STMTS)       \
  case internal::TypeTraits<TYPE>::kDataType: { \
    typedef TYPE TYPE_ARG;                      \
    STMTS;                                      \
    break;                                      \
  }

#define DTYPE_CASES_BEGIN(TYPE_ENUM) switch (TYPE_ENUM) {
#define DTYPE_CASES_END(TYPE_ENUM)                      \
  case DT_INVALID:                                      \
    TFF_LOG(FATAL) << "Invalid type";                   \
    break;                                              \
  default:                                              \
    TFF_LOG(FATAL) << "Unsupported type " << TYPE_ENUM; \
    }

#define DTYPE_FLOATING_CASES(TYPE_ARG, STMTS) \
  DTYPE_CASE(float, TYPE_ARG, STMTS)          \
  DTYPE_CASE(double, TYPE_ARG, STMTS)

#define DTYPE_INTEGER_CASES(TYPE_ARG, STMTS) \
DTYPE_CASE(int32_t, TYPE_ARG, STMTS)         \
DTYPE_CASE(int64_t, TYPE_ARG, STMTS)         \
DTYPE_CASE(uint64_t, TYPE_ARG, STMTS)

#define DTYPE_NUMERICAL_CASES(TYPE_ARG, STMTS) \
  DTYPE_FLOATING_CASES(TYPE_ARG, STMTS)        \
  DTYPE_INTEGER_CASES(TYPE_ARG, STMTS)

#define DTYPE_STRING_CASES(TYPE_ARG, STMTS) \
  DTYPE_CASE(string_view, TYPE_ARG, STMTS)

// TODO: b/222605809 - Add other types.
#define DTYPE_CASES(TYPE_ENUM, TYPE_ARG, STMTS)      \
  DTYPE_CASES_BEGIN(TYPE_ENUM)                       \
  DTYPE_NUMERICAL_CASES(TYPE_ARG, SINGLE_ARG(STMTS)) \
  DTYPE_STRING_CASES(TYPE_ARG, SINGLE_ARG(STMTS))    \
  DTYPE_CASES_END(TYPE_ENUM)

#define NUMERICAL_ONLY_DTYPE_CASES(TYPE_ENUM, TYPE_ARG, STMTS) \
  DTYPE_CASES_BEGIN(TYPE_ENUM)                                 \
  DTYPE_NUMERICAL_CASES(TYPE_ARG, SINGLE_ARG(STMTS))           \
  DTYPE_CASES_END(TYPE_ENUM)

#define FLOATING_ONLY_DTYPE_CASES(TYPE_ENUM, TYPE_ARG, STMTS) \
  DTYPE_CASES_BEGIN(TYPE_ENUM)                                \
  DTYPE_FLOATING_CASES(TYPE_ARG, SINGLE_ARG(STMTS))           \
  DTYPE_CASES_END(TYPE_ENUM)

}  // namespace internal

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DATATYPE_H_
