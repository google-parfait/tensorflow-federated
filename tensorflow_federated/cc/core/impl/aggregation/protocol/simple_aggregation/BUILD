# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "latency_aggregator",
    hdrs = ["latency_aggregator.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "latency_aggregator_test",
    srcs = ["latency_aggregator_test.cc"],
    deps = [
        ":latency_aggregator",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "cancelable_callback",
    srcs = ["cancelable_callback.cc"],
    hdrs = ["cancelable_callback.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/base:clock",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "cancelable_callback_test",
    srcs = ["cancelable_callback_test.cc"],
    deps = [
        ":cancelable_callback",
        "//tensorflow_federated/cc/core/impl/aggregation/base:clock",
        "//tensorflow_federated/cc/core/impl/aggregation/base:simulated_clock",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "simple_aggregation",
    srcs = ["simple_aggregation_protocol.cc"],
    hdrs = ["simple_aggregation_protocol.h"],
    deps = [
        ":cancelable_callback",
        ":latency_aggregator",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/base:clock",
        "//tensorflow_federated/cc/core/impl/aggregation/core:aggregator",
        "//tensorflow_federated/cc/core/impl/aggregation/core:intrinsic",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:aggregation_protocol",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:aggregation_protocol_messages_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_aggregator",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_builder",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_parser",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:configuration_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:resource_resolver",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

cc_test(
    name = "simple_aggregation_test",
    srcs = ["simple_aggregation_protocol_test.cc"],
    deps = [
        ":simple_aggregation",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/base:scheduler",
        "//tensorflow_federated/cc/core/impl/aggregation/base:simulated_clock",
        "//tensorflow_federated/cc/core/impl/aggregation/core:aggregator",
        "//tensorflow_federated/cc/core/impl/aggregation/core:intrinsic",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:aggregation_protocol_messages_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:configuration_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:mocks",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:parse_text_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
