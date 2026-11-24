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

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "include/pybind11/cast.h"
#include "include/pybind11/detail/common.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"

namespace pybind11 {
namespace google {

// No-op in OSS because errors are converted without status.so.
inline void ImportStatusModule() {}

}  // namespace google

namespace detail {

inline str DecodeUtf8Replace(const std::string& text) {
  PyObject* decoded = PyUnicode_DecodeUTF8(
      text.data(), static_cast<Py_ssize_t>(text.size()), "replace");
  if (decoded == nullptr) {
    throw error_already_set();
  }
  return reinterpret_steal<str>(decoded);
}

[[noreturn]] inline void ThrowStatusException(const absl::Status& status) {
  const int code_int = static_cast<int>(status.code());
  const std::string code_str(absl::StatusCodeToString(status.code()));
  const std::string message(status.message());
  const std::string full_message =
      message.empty() ? absl::StrCat("[", code_str, "]")
                      : absl::StrCat(message, " [", code_str, "]");

  gil_scoped_acquire gil;
  str py_message = DecodeUtf8Replace(message);
  str py_full_message = DecodeUtf8Replace(full_message);

  // Raise `pybind11_abseil.status.StatusNotOk` when `pybind11_abseil.status`
  // is available and matches the expected API. Otherwise, raise
  // `builtins.Exception` with a `.status` attribute.
  PyObject* status_mod_ptr = PyImport_ImportModule("pybind11_abseil.status");
  if (status_mod_ptr != nullptr) {
    object status_not_ok_cls;
    object exc;
    bool built_status_not_ok = false;
    try {
      object status_mod = reinterpret_steal<object>(status_mod_ptr);
      object status_code = status_mod.attr("StatusCode")(code_int);
      status_not_ok_cls = status_mod.attr("StatusNotOk");
      exc = status_mod.attr("BuildStatusNotOk")(status_code, py_message);
      built_status_not_ok = true;
    } catch (const error_already_set&) {
      PyErr_Clear();
    }
    if (built_status_not_ok) {
      PyErr_SetObject(status_not_ok_cls.ptr(), exc.ptr());
      throw error_already_set();
    }
  } else {
    if (!PyErr_ExceptionMatches(PyExc_ImportError)) {
      throw error_already_set();
    }
    PyErr_Clear();
  }

  object status_obj = module_::import("types").attr("SimpleNamespace")(
      arg("code_int") = cpp_function([code_int]() { return code_int; }),
      arg("message") =
          cpp_function([message]() { return DecodeUtf8Replace(message); }),
      arg("to_string") = cpp_function(
          [full_message]() { return DecodeUtf8Replace(full_message); }));
  object exc =
      module_::import("builtins").attr("Exception")(py_full_message);
  exc.attr("status") = status_obj;
  PyErr_SetObject(PyExc_Exception, exc.ptr());
  throw error_already_set();
}

template <>
struct type_caster<absl::Status> {
 public:
  PYBIND11_TYPE_CASTER(absl::Status, const_name("Status"));

  static handle cast(const absl::Status& src,
                     return_value_policy /* policy */, handle /* parent */) {
    if (!src.ok()) {
      ThrowStatusException(src);
    }
    return none().release();
  }

  bool load(handle, bool) { return false; }
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
      ThrowStatusException(src.status());
    }
    return value_conv::cast(std::move(*src),
                            return_value_policy_override<T>::policy(policy),
                            parent);
  }

  bool load(handle, bool) { return false; }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STATUS_CASTERS_H_
