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

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "tensorflow_testing",
    srcs = ["tensorflow_testing.cc"],
    hdrs = ["tensorflow_testing.h"],
    deps = [
        ":testing",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/tensorflow:tf_checkpoint_reader",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@pypi//tensorflow:libtensorflow_cc",
        "@pypi//tensorflow:libtensorflow_framework",
        "@pypi//tensorflow:tf_header_lib",
    ],
)

cc_library(
    name = "testing",
    srcs = ["testing.cc"],
    hdrs = ["testing.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:intrinsic",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mocks",
    hdrs = ["mocks.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_builder",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_parser",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:resource_resolver",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "parse_text_proto",
    hdrs = ["parse_text_proto.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "test_data",
    hdrs = ["test_data.h"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow_federated/cc/core/impl/aggregation/core:tensor"],
)
