# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//tensorflow_federated/cc/core/impl/aggregation:__subpackages__"],
)

cc_library(
    name = "base",
    srcs = [
        "base_name.cc",
        "monitoring.cc",
        "platform.cc",
    ],
    hdrs = [
        "base_name.h",
        "monitoring.h",
        "move_to_lambda.h",
        "platform.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_test(
    name = "monitoring_test",
    size = "small",
    srcs = ["monitoring_test.cc"],
    deps = [
        ":base",
        "//tensorflow_federated/cc/testing:oss_test_main",
    ],
)

cc_test(
    name = "platform_test",
    size = "small",
    srcs = ["platform_test.cc"],
    deps = [
        ":base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "scheduler",
    srcs = ["scheduler.cc"],
    hdrs = ["scheduler.h"],
    deps = [
        ":base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "scheduler_test",
    size = "small",
    srcs = ["scheduler_test.cc"],
    deps = [
        ":scheduler",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "move_to_lambda_test",
    size = "small",
    srcs = ["move_to_lambda_test.cc"],
    deps = [
        ":base",
        ":unique_value",
        "//tensorflow_federated/cc/testing:oss_test_main",
    ],
)

cc_library(
    name = "unique_value",
    hdrs = ["unique_value.h"],
)

cc_test(
    name = "unique_value_test",
    srcs = ["unique_value_test.cc"],
    deps = [
        ":base",
        ":unique_value",
        "//tensorflow_federated/cc/testing:oss_test_main",
    ],
)

cc_library(
    name = "clock",
    srcs = ["clock.cc"],
    hdrs = ["clock.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "simulated_clock",
    testonly = 1,
    srcs = ["simulated_clock.cc"],
    hdrs = ["simulated_clock.h"],
    deps = [
        ":clock",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "simulated_clock_test",
    srcs = ["simulated_clock_test.cc"],
    deps = [
        ":clock",
        ":simulated_clock",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "@com_google_absl//absl/time",
    ],
)
