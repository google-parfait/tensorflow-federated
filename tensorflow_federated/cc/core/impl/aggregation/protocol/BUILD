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

load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("//third_party/protobuf/bazel:proto_library.bzl", "proto_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

proto_library(
    name = "aggregation_protocol_messages_proto",
    srcs = ["aggregation_protocol_messages.proto"],
)

cc_proto_library(
    name = "aggregation_protocol_messages_cc_proto",
    deps = [":aggregation_protocol_messages_proto"],
)

py_proto_library(
    name = "aggregation_protocol_messages_py_pb2",
    deps = [":aggregation_protocol_messages_proto"],
)

proto_library(
    name = "configuration_proto",
    srcs = ["configuration.proto"],
    deps = ["//tensorflow_federated/cc/core/impl/aggregation/core:tensor_proto"],
)

proto_library(
    name = "checkpoint_aggregator_proto",
    srcs = ["checkpoint_aggregator.proto"],
)

cc_proto_library(
    name = "checkpoint_aggregator_cc_proto",
    deps = [":checkpoint_aggregator_proto"],
)

cc_proto_library(
    name = "configuration_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":configuration_proto"],
)

py_proto_library(
    name = "configuration_py_pb2",
    deps = [":configuration_proto"],
)

cc_library(
    name = "aggregation_protocol",
    hdrs = ["aggregation_protocol.h"],
    deps = [
        ":aggregation_protocol_messages_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "resource_resolver",
    hdrs = ["resource_resolver.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "checkpoint_builder",
    hdrs = ["checkpoint_builder.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "checkpoint_parser",
    hdrs = ["checkpoint_parser.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "checkpoint_aggregator",
    srcs = ["checkpoint_aggregator.cc"],
    hdrs = ["checkpoint_aggregator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":checkpoint_aggregator_cc_proto",
        ":checkpoint_builder",
        ":checkpoint_parser",
        ":config_converter",
        ":configuration_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:aggregation_cores",
        "//tensorflow_federated/cc/core/impl/aggregation/core:aggregator",
        "//tensorflow_federated/cc/core/impl/aggregation/core:intrinsic",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "checkpoint_aggregator_test",
    srcs = ["checkpoint_aggregator_test.cc"],
    deps = [
        ":checkpoint_aggregator",
        ":configuration_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/base:scheduler",
        "//tensorflow_federated/cc/core/impl/aggregation/core:aggregator",
        "//tensorflow_federated/cc/core/impl/aggregation/core:intrinsic",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:mocks",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:parse_text_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "config_converter",
    srcs = ["config_converter.cc"],
    hdrs = ["config_converter.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":configuration_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:aggregator",
        "//tensorflow_federated/cc/core/impl/aggregation/core:dp_fedsql_constants",
        "//tensorflow_federated/cc/core/impl/aggregation/core:fedsql_constants",
        "//tensorflow_federated/cc/core/impl/aggregation/core:intrinsic",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "federated_compute_checkpoint_builder",
    srcs = ["federated_compute_checkpoint_builder.cc"],
    hdrs = ["federated_compute_checkpoint_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":checkpoint_builder",
        ":checkpoint_header",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_test(
    name = "federated_compute_checkpoint_builder_test",
    srcs = ["federated_compute_checkpoint_builder_test.cc"],
    deps = [
        ":checkpoint_builder",
        ":checkpoint_header",
        ":federated_compute_checkpoint_builder",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "federated_compute_checkpoint_parser",
    srcs = ["federated_compute_checkpoint_parser.cc"],
    hdrs = ["federated_compute_checkpoint_parser.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":checkpoint_header",
        ":checkpoint_parser",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "checkpoint_header",
    hdrs = ["checkpoint_header.h"],
)

cc_test(
    name = "federated_compute_checkpoint_parser_test",
    srcs = ["federated_compute_checkpoint_parser_test.cc"],
    deps = [
        ":checkpoint_builder",
        ":checkpoint_parser",
        ":federated_compute_checkpoint_builder",
        ":federated_compute_checkpoint_parser",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "config_converter_test",
    srcs = ["config_converter_test.cc"],
    deps = [
        ":config_converter",
        ":configuration_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:aggregator",
        "//tensorflow_federated/cc/core/impl/aggregation/core:intrinsic",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:parse_text_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)
