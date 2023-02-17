/* Copyright 2021, The TensorFlow Federated Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STATUS_MACROS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STATUS_MACROS_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tensorflow_federated {
namespace status_macros {

// Internal-only helper for TFF_TRY
// Handles the no-`suffix` case.
inline absl::Status __get_status(absl::Status&& status) {
  return std::move(status);
}

// Internal-only helper for TFF_TRY
// Appends `suffix` to the message of the original status.
inline absl::Status __get_status(absl::Status&& status,
                                 const std::string_view& suffix) {
  return absl::Status(status.code(),
                      absl::StrCat(status.message(), " ", suffix));
}

// Internal-only helper for TFF_TRY
// Handles converting from a `StatusOr` to `Status` before delegating down.
template <typename T, typename... VarArgs>
inline absl::Status __get_status(absl::StatusOr<T>&& status_or,
                                 VarArgs... varargs) {
  return __get_status(std::move(status_or).status(), varargs...);
}

// Internal-only helper for TFF_TRY
inline void __void_or_result(absl::Status _) {}

// Internal-only helper for TFF_TRY
template <typename T>
inline T __void_or_result(absl::StatusOr<T>&& res) {
  return std::move(res.value());
}

// A macro that accepts either an `absl::Status` or `absl::StatusOr<T>` and
// returns early in the case of an error. If successful, the macro evaluates to
// an expression of type `T` (in the case of `absl::StatusOr<T>`) or `void`
// (in the case of `absl::Status`).
//
// The macro accepts an optional last argument for a `const std::string_view&`
// to append to the error message.
#define TFF_TRY(expr, ...)                                                     \
  ({                                                                           \
    auto __tff_expr_res = expr;                                                \
    if (!__tff_expr_res.ok()) {                                                \
      return ::tensorflow_federated::status_macros::__get_status(              \
          std::move(__tff_expr_res),                                           \
          ##__VA_ARGS__);                                                      \
    }                                                                          \
    ::tensorflow_federated::status_macros::__void_or_result(                   \
        std::move(__tff_expr_res));                                            \
  })

}  // namespace status_macros
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STATUS_MACROS_H_
