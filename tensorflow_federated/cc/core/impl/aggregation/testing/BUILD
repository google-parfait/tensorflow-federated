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
    name = "testing",
    testonly = True,
    srcs = ["testing.cc"],
    hdrs = ["testing.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/core:intrinsic",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@org_tensorflow//tensorflow/c:checkpoint_reader",
        "@org_tensorflow//tensorflow/c:tf_status_headers",
        "@org_tensorflow//tensorflow/c:tf_status_helper",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core/platform:tstring",
    ],
)

cc_library(
    name = "mocks",
    testonly = True,
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
    testonly = True,
    hdrs = ["parse_text_proto.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "test_data",
    testonly = True,
    hdrs = ["test_data.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@com_google_absl//absl/strings:string_view",
    ],
)
