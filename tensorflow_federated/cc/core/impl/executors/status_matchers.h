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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STATUS_MATCHERS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STATUS_MATCHERS_H_

// TODO(b/199461150) remove this file and the associated .cc file.

#include <cstdint>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "grpcpp/grpcpp.h"

namespace tensorflow_federated {
namespace internal_status {

using ::testing::Matcher;
using ::testing::MatcherCast;
using ::testing::MatcherInterface;
using ::testing::MatchResultListener;
using ::testing::PrintToString;
using ::testing::SafeMatcherCast;
using ::testing::StringMatchResultListener;

inline const absl::Status& GetStatus(const absl::Status& status) {
  return status;
}

template <typename T>
inline const absl::Status& GetStatus(const absl::StatusOr<T>& status) {
  return status.status();
}

////////////////////////////////////////////////////////////
// Implementation of IsOkAndHolds().

// Monomorphic implementation of matcher IsOkAndHolds(m).  StatusOrType is a
// reference to StatusOr<T>.
template <typename StatusOrType>
class IsOkAndHoldsMatcherImpl : public MatcherInterface<StatusOrType> {
 public:
  typedef
      typename std::remove_reference<StatusOrType>::type::value_type value_type;

  template <typename InnerMatcher>
  explicit IsOkAndHoldsMatcherImpl(InnerMatcher&& inner_matcher)
      : inner_matcher_(SafeMatcherCast<const value_type&>(
            std::forward<InnerMatcher>(inner_matcher))) {}

  void DescribeTo(std::ostream* os) const override {
    *os << "is OK and has a value that ";
    inner_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "isn't OK or has a value that ";
    inner_matcher_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(StatusOrType actual_value,
                       MatchResultListener* result_listener) const override {
    if (!actual_value.ok()) {
      *result_listener << "which has status " << actual_value.status();
      return false;
    }

    StringMatchResultListener inner_listener;
    const bool matches =
        inner_matcher_.MatchAndExplain(*actual_value, &inner_listener);
    const std::string inner_explanation = inner_listener.str();
    if (!inner_explanation.empty()) {
      *result_listener << "which contains value "
                       << PrintToString(*actual_value) << ", "
                       << inner_explanation;
    }
    return matches;
  }

 private:
  const Matcher<const value_type&> inner_matcher_;
};

// Implements IsOkAndHolds(m) as a polymorphic matcher.
template <typename InnerMatcher>
class IsOkAndHoldsMatcher {
 public:
  explicit IsOkAndHoldsMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(std::move(inner_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the
  // given type.  StatusOrType can be either StatusOr<T> or a
  // reference to StatusOr<T>.
  template <typename StatusOrType>
  operator Matcher<StatusOrType>() const {  // NOLINT
    return Matcher<StatusOrType>(
        new IsOkAndHoldsMatcherImpl<const StatusOrType&>(inner_matcher_));
  }

 private:
  const InnerMatcher inner_matcher_;
};

// StatusIs() is a polymorphic matcher.  This class is the common
// implementation of it shared by all types T where StatusIs() can be
// used as a Matcher<T>.
class StatusIsMatcherCommonImpl {
 public:
  StatusIsMatcherCommonImpl(Matcher<absl::StatusCode> code_matcher,
                            Matcher<const std::string&> message_matcher)
      : code_matcher_(std::move(code_matcher)),
        message_matcher_(std::move(message_matcher)) {}

  void DescribeTo(std::ostream* os) const;

  void DescribeNegationTo(std::ostream* os) const;

  bool MatchAndExplain(const absl::Status& status,
                       MatchResultListener* result_listener) const;

 private:
  const Matcher<absl::StatusCode> code_matcher_;
  const Matcher<const std::string&> message_matcher_;
};

// Monomorphic implementation of matcher StatusIs() for a given type
// T.  T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoStatusIsMatcherImpl : public MatcherInterface<T> {
 public:
  explicit MonoStatusIsMatcherImpl(StatusIsMatcherCommonImpl common_impl)
      : common_impl_(std::move(common_impl)) {}

  void DescribeTo(std::ostream* os) const override {
    common_impl_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    common_impl_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(T actual_value,
                       MatchResultListener* result_listener) const override {
    return common_impl_.MatchAndExplain(GetStatus(actual_value),
                                        result_listener);
  }

 private:
  StatusIsMatcherCommonImpl common_impl_;
};

// Implements StatusIs() as a polymorphic matcher.
class StatusIsMatcher {
 public:
  template <typename StatusCodeMatcher, typename StatusMessageMatcher>
  StatusIsMatcher(StatusCodeMatcher&& code_matcher,
                  StatusMessageMatcher&& message_matcher)
      : common_impl_(MatcherCast<absl::StatusCode>(
                         std::forward<StatusCodeMatcher>(code_matcher)),
                     MatcherCast<const std::string&>(
                         std::forward<StatusMessageMatcher>(message_matcher))) {
  }

  // Converts this polymorphic matcher to a monomorphic matcher of the
  // given type.  T can be StatusOr<>, Status, or a reference to
  // either of them.
  template <typename T>
  operator Matcher<T>() const {  // NOLINT
    return Matcher<T>(new MonoStatusIsMatcherImpl<const T&>(common_impl_));
  }

 private:
  const StatusIsMatcherCommonImpl common_impl_;
};

// CanonicalStatusIs() is a polymorphic matcher.  This class is the common
// implementation of it shared by all types T where CanonicalStatusIs() can be
// used as a Matcher<T>.
class CanonicalStatusIsMatcherCommonImpl {
 public:
  CanonicalStatusIsMatcherCommonImpl(
      Matcher<absl::StatusCode> code_matcher,
      Matcher<const std::string&> message_matcher)
      : code_matcher_(std::move(code_matcher)),
        message_matcher_(std::move(message_matcher)) {}

  void DescribeTo(std::ostream* os) const;

  void DescribeNegationTo(std::ostream* os) const;

  bool MatchAndExplain(const absl::Status& status,
                       MatchResultListener* result_listener) const;

 private:
  const Matcher<absl::StatusCode> code_matcher_;
  const Matcher<const std::string&> message_matcher_;
};

// Monomorphic implementation of matcher CanonicalStatusIs() for a given type
// T.  T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoCanonicalStatusIsMatcherImpl : public MatcherInterface<T> {
 public:
  explicit MonoCanonicalStatusIsMatcherImpl(
      CanonicalStatusIsMatcherCommonImpl common_impl)
      : common_impl_(std::move(common_impl)) {}

  void DescribeTo(std::ostream* os) const override {
    common_impl_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    common_impl_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(T actual_value,
                       MatchResultListener* result_listener) const override {
    return common_impl_.MatchAndExplain(GetStatus(actual_value),
                                        result_listener);
  }

 private:
  CanonicalStatusIsMatcherCommonImpl common_impl_;
};

// Implements CanonicalStatusIs() as a polymorphic matcher.
class CanonicalStatusIsMatcher {
 public:
  template <typename StatusCodeMatcher, typename StatusMessageMatcher>
  CanonicalStatusIsMatcher(StatusCodeMatcher&& code_matcher,
                           StatusMessageMatcher&& message_matcher)
      : common_impl_(MatcherCast<absl::StatusCode>(
                         std::forward<StatusCodeMatcher>(code_matcher)),
                     MatcherCast<const std::string&>(
                         std::forward<StatusMessageMatcher>(message_matcher))) {
  }

  // Converts this polymorphic matcher to a monomorphic matcher of the given
  // type.  T can be StatusOr<>, Status, or a reference to either of them.
  template <typename T>
  operator Matcher<T>() const {  // NOLINT
    return Matcher<T>(
        new MonoCanonicalStatusIsMatcherImpl<const T&>(common_impl_));
  }

 private:
  const CanonicalStatusIsMatcherCommonImpl common_impl_;
};

// Monomorphic implementation of matcher IsOk() for a given type T.
// T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoIsOkMatcherImpl : public MatcherInterface<T> {
 public:
  void DescribeTo(std::ostream* os) const override { *os << "is OK"; }
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "is not OK";
  }
  bool MatchAndExplain(T actual_value, MatchResultListener*) const override {
    return GetStatus(actual_value).ok();
  }
};

// Implements IsOk() as a polymorphic matcher.
class IsOkMatcher {
 public:
  template <typename T>
  operator Matcher<T>() const {  // NOLINT
    return Matcher<T>(new MonoIsOkMatcherImpl<const T&>());
  }
};

void AddFatalFailure(absl::string_view expression, const char* file,
                     uint32_t line, absl::Status status);

inline absl::Status GetStatus(absl::Status&& status) { return status; }

template <typename T>
inline absl::Status GetStatus(absl::StatusOr<T>&& status_or) {
  return std::move(status_or).status();
}

inline void VoidOrResult(absl::Status _) {}

template <typename T>
inline T VoidOrResult(absl::StatusOr<T>&& res) {
  return std::move(res.value());
}

}  // namespace internal_status

// Macros for testing the results of functions that return absl::Status or
// absl::StatusOr<T> (for any type T).
#define TFF_EXPECT_OK(expression) \
  EXPECT_THAT(expression, ::tensorflow_federated::IsOk())
#define TFF_ASSERT_OK(expression)                               \
  ({                                                            \
    auto __tff_expr_res = expression;                           \
    if (!__tff_expr_res.ok()) {                                 \
      ::tensorflow_federated::internal_status::AddFatalFailure( \
          #expression, __FILE__, __LINE__,                      \
          ::tensorflow_federated::internal_status::GetStatus(   \
              std::move(__tff_expr_res)));                      \
      return;                                                   \
    }                                                           \
    ::tensorflow_federated::internal_status::VoidOrResult(      \
        std::move(__tff_expr_res));                             \
  })

// Executes an expression that returns an absl::StatusOr, and assigns the
// contained variable to lhs if the error code is OK.
// If the Status is non-OK, generates a test failure and returns from the
// current function, which must have a void return type.
//
// Example: Declaring and initializing a new value
//   TFF_ASSERT_OK_AND_ASSIGN(const ValueType& value, MaybeGetValue(arg));
//
// Example: Assigning to an existing value
//   ValueType value;
//   TFF_ASSERT_OK_AND_ASSIGN(value, MaybeGetValue(arg));
//
// The value assignment example would expand into something like:
//   auto status_or_value = MaybeGetValue(arg);
//   TFF_ASSERT_OK(status_or_value.status());
//   value = std::move(status_or_value).ValueOrDie();
#define TFF_ASSERT_OK_AND_ASSIGN(lhs, rexpr) lhs = TFF_ASSERT_OK(rexpr)

// Returns a gMock matcher that matches a StatusOr<> whose status is
// OK and whose value matches the inner matcher.
template <typename InnerMatcher>
internal_status::IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>
IsOkAndHolds(InnerMatcher&& inner_matcher) {
  return internal_status::IsOkAndHoldsMatcher<
      typename std::decay<InnerMatcher>::type>(
      std::forward<InnerMatcher>(inner_matcher));
}

// Returns a gMock matcher that matches a Status or StatusOr<> whose
// error space matches space_matcher, whose status code matches
// code_matcher, and whose error message matches message_matcher.
template <typename ErrorSpaceMatcher, typename StatusCodeMatcher,
          typename StatusMessageMatcher>
internal_status::StatusIsMatcher StatusIs(
    ErrorSpaceMatcher&& space_matcher, StatusCodeMatcher&& code_matcher,
    StatusMessageMatcher&& message_matcher) {
  return internal_status::StatusIsMatcher(
      std::forward<ErrorSpaceMatcher>(space_matcher),
      std::forward<StatusCodeMatcher>(code_matcher),
      std::forward<StatusMessageMatcher>(message_matcher));
}

// The one and two-arg StatusIs methods may infer the expected ErrorSpace from
// the StatusCodeMatcher argument. If you call StatusIs(e) or StatusIs(e, msg)
// and the argument `e` is:
// - an enum type,
// - which is associated with a custom ErrorSpace `S`,
// - and is not "OK" (i.e. 0),
// then the matcher will match a Status or StatusOr<> whose error space is `S`.
//
// Otherwise, the expected error space is the canonical error space.

// Returns a gMock matcher that matches a Status or StatusOr<> whose error space
// is the inferred error space (see above), whose status code matches
// code_matcher, and whose error message matches message_matcher.
template <typename StatusCodeMatcher, typename StatusMessageMatcher>
internal_status::StatusIsMatcher StatusIs(
    StatusCodeMatcher&& code_matcher, StatusMessageMatcher&& message_matcher) {
  return internal_status::StatusIsMatcher(
      std::forward<StatusCodeMatcher>(code_matcher),
      std::forward<StatusMessageMatcher>(message_matcher));
}

// Returns a gMock matcher that matches a Status or StatusOr<> whose error space
// is the inferred error space (see above), and whose status code matches
// code_matcher.
template <typename StatusCodeMatcher>
internal_status::StatusIsMatcher StatusIs(StatusCodeMatcher&& code_matcher) {
  return StatusIs(std::forward<StatusCodeMatcher>(code_matcher), ::testing::_);
}

// Returns a gMock matcher that matches a Status or StatusOr<> whose canonical
// status code (i.e., Status::CanonicalCode) matches code_matcher and whose
// error message matches message_matcher.
template <typename StatusCodeMatcher, typename StatusMessageMatcher>
internal_status::CanonicalStatusIsMatcher CanonicalStatusIs(
    StatusCodeMatcher&& code_matcher, StatusMessageMatcher&& message_matcher) {
  return internal_status::CanonicalStatusIsMatcher(
      std::forward<StatusCodeMatcher>(code_matcher),
      std::forward<StatusMessageMatcher>(message_matcher));
}

// Returns a gMock matcher that matches a Status or StatusOr<> whose canonical
// status code (i.e., Status::CanonicalCode) matches code_matcher.
template <typename StatusCodeMatcher>
internal_status::CanonicalStatusIsMatcher CanonicalStatusIs(
    StatusCodeMatcher&& code_matcher) {
  return CanonicalStatusIs(std::forward<StatusCodeMatcher>(code_matcher),
                           ::testing::_);
}

// Returns a gMock matcher that matches a Status or StatusOr<> which is OK.
inline internal_status::IsOkMatcher IsOk() {
  return internal_status::IsOkMatcher();
}

// Separate matcher for `grpc::Status`, since it and `absl::Status` are not
// interconvertible in OSS.
class GrpcStatusMatcher {
 public:
  explicit GrpcStatusMatcher(grpc::StatusCode code,
                             absl::optional<std::string> message)
      : expected_code_(code), expected_message_(std::move(message)) {}

  using is_gtest_matcher = void;
  bool MatchAndExplain(grpc::Status status,
                       ::testing::MatchResultListener* os) const {
    if (status.error_code() != expected_code_) {
      *os << "the status code is " << status.error_code();
      return false;
    }
    if (expected_message_.has_value() &&
        status.error_message().find(expected_message_.value()) !=
            std::string::npos) {
      *os << "the error message is " << status.error_message();
    }
    return true;
  }

  void DescribeTo(std::ostream* os) const {
    *os << " has status code " << expected_code_;
    if (expected_message_.has_value()) {
      *os << " and a message containing: " << expected_message_.value();
    }
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << " does not have status code " << expected_code_;
    if (expected_message_.has_value()) {
      *os << " and a message containing: " << expected_message_.value();
    }
  }

 private:
  grpc::StatusCode expected_code_;
  absl::optional<std::string> expected_message_;
};

inline ::testing::Matcher<grpc::Status> GrpcStatusIs(
    grpc::StatusCode code,
    absl::optional<std::string> message = absl::nullopt) {
  return GrpcStatusMatcher(code, message);
}

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_STATUS_MATCHERS_H_
