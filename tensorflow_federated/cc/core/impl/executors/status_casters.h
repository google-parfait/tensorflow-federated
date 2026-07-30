/* Copyright 2026, The TensorFlow Federated Authors.
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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STATUS_CASTERS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STATUS_CASTERS_H_

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "include/pybind11/pybind11.h"
#include "pybind11_abseil/status_not_ok_exception.h"

namespace pybind11 {
namespace detail {

template <>
struct type_caster<absl::Status> {
 public:
  PYBIND11_TYPE_CASTER(absl::Status, const_name("Status"));

  static handle cast(absl::Status src, return_value_policy /* policy */,
                     handle /* parent */) {
    if (!src.ok()) {
      throw ::pybind11::google::StatusNotOk(std::move(src));
    }
    return none().release();
  }
};

template <typename T>
struct type_caster<absl::StatusOr<T>> {
 public:
  using value_conv = make_caster<T>;
  PYBIND11_TYPE_CASTER(absl::StatusOr<T>, const_name("StatusOr[") +
                                              value_conv::name +
                                              const_name("]"));

  static handle cast(absl::StatusOr<T> src, return_value_policy policy,
                     handle parent) {
    if (!src.ok()) {
      throw ::pybind11::google::StatusNotOk(std::move(src).status());
    }
    return value_conv::cast(std::move(*src), policy, parent);
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STATUS_CASTERS_H_
