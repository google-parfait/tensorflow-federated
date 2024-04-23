/*
 * Copyright 2019 Google LLC
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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_UNIQUE_VALUE_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_UNIQUE_VALUE_H_

#include <optional>
#include <utility>

namespace tensorflow_federated {

/**
 * UniqueValue<T> provides move-only semantics for some value of type T.
 *
 * Its semantics are much like std::unique_ptr, but without requiring an
 * allocation and pointer indirection (recall a moved-from std::unique_ptr is
 * reset to nullptr).
 *
 * Instead, UniqueValue is represented just like std::optional - but
 * has_value() == false once moved-from. std::optional does *not* reset when
 * moved from (even if the wrapped type is move-only); that's consistent, but
 * not especially desirable.
 *
 * Since UniqueValue is always move-only, including a UniqueValue member is
 * sufficient for a containing aggregate to be move-only.
 */
template <typename T>
class UniqueValue {
 public:
  constexpr explicit UniqueValue(std::nullopt_t) : value_() {}
  constexpr explicit UniqueValue(T val) : value_(std::move(val)) {}

  UniqueValue(UniqueValue const&) = delete;
  UniqueValue(UniqueValue&& other) : value_(std::move(other.value_)) {
    other.value_.reset();
  }

  UniqueValue& operator=(UniqueValue other) {
    value_.swap(other.value_);
    return *this;
  }

  /**
   * Indicates if this instance holds a value (i.e. has not been moved away).
   *
   * It is an error to dereference this UniqueValue if !has_value().
   */
  constexpr bool has_value() const {
    return value_.has_value();
  }

  constexpr T Take() && {
    T v = *std::move(value_);
    value_.reset();
    return v;
  }

  constexpr T const& operator*() const & {
    return *value_;
  }

  T& operator*() & {
    return *value_;
  }

  T* operator->() {
    return &*value_;
  }

  /**
   * Replaces current value with a newly constructed one given constructor
   * arguments for T (like std::optional::emplace).
   */
  template <class... _Args>
  T& Emplace(_Args&&... __args) {
    value_.emplace(std::forward<_Args>(__args)...);
    return *value_;
  }

  /**
   * Removes (destructs) a value. No-op if absent;
   */
  void Reset() { value_.reset(); }

 private:
  std::optional<T> value_;
};

// Deduction guide allowing one to write UniqueValue(x) without an explicit
// template argument. This one would be implicitly generated; it's here to
// suppress -Wctad-maybe-unsupported.
template <typename T>
UniqueValue(T val) -> UniqueValue<T>;

/**
 * Makes a UniqueValue<T> given constructor arguments for T
 * (like std::make_unique).
 */
template <typename T, typename... Args>
constexpr UniqueValue<T> MakeUniqueValue(Args&&... args) {
  return UniqueValue<T>(T(std::forward<Args>(args)...));
}

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_UNIQUE_VALUE_H_
