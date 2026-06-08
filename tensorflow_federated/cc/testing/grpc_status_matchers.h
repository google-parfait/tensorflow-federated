/* Copyright 2026, The TensorFlow Federated Authors.

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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_TESTING_GRPC_STATUS_MATCHERS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_TESTING_GRPC_STATUS_MATCHERS_H_

#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "include/grpcpp/support/status.h"

namespace tensorflow_federated {

// Separate matcher for `grpc::Status`, since it and `absl::Status` are not
// interconvertible in OSS.
class GrpcStatusMatcher {
 public:
  explicit GrpcStatusMatcher(grpc::StatusCode code,
                             std::optional<std::string> message)
      : expected_code_(code), expected_message_(std::move(message)) {}

  using is_gtest_matcher = void;
  bool MatchAndExplain(const grpc::Status& status,
                       ::testing::MatchResultListener* os) const {
    if (status.error_code() != expected_code_) {
      *os << "the status code is " << status.error_code();
      return false;
    }
    if (expected_message_.has_value() &&
        status.error_message().find(expected_message_.value()) ==
            std::string::npos) {
      *os << "the error message is " << status.error_message();
      return false;
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
  std::optional<std::string> expected_message_;
};

inline ::testing::Matcher<grpc::Status> GrpcStatusIs(
    grpc::StatusCode code, std::optional<std::string> message = std::nullopt) {
  return GrpcStatusMatcher(code, message);
}

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_TESTING_GRPC_STATUS_MATCHERS_H_
