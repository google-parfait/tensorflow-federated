# Copyright 2024 Google LLC
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

package(
    default_visibility = ["//visibility:public"]
)

load("@org_tensorflow//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

cc_library(
    name = "native",
    srcs = [
        "jni.cc",
    ],
    hdrs = [
        "util.h",
    ],
    copts = [
        "-Wno-narrowing",
        "-Wno-sign-compare",
        "-Wno-overloaded-virtual",
    ],
    deps = [
        ":plan_cc_proto",
        "@bazel_tools//tools/jdk:jni",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_aggregator",
        "//tensorflow_federated/cc/core/impl/aggregation/tensorflow:tensorflow_checkpoint_builder_factory",
        "//tensorflow_federated/cc/core/impl/aggregation/tensorflow:tensorflow_checkpoint_parser_factory",
    ],
    alwayslink = 1,
)

tf_proto_library(
    name = "plan_proto",
    srcs = ["plan.proto"],
    protodeps = [
        "@org_tensorflow//tensorflow/core:protos_all",
    ],
)

alias(
    name = "plan_cc_proto",
    actual = "plan_proto_cc",
)

cc_binary(
    name = "aggregation-jni",
    linkshared = 1,
    linkstatic = 1,
    deps = [":native"],
)
