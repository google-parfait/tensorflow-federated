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
load("@rules_cc//cc:defs.bzl", "cc_proto_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

# TODO: b/254719929 - Create a "core" library that bundles all core libraries together.

proto_library(
    name = "tensor_proto",
    srcs = ["tensor.proto"],
)

cc_proto_library(
    name = "tensor_cc_proto",
    deps = [":tensor_proto"],
)

java_proto_library(
    name = "tensor_java_proto",
    visibility = ["//visibility:public"],
    deps = [":tensor_proto"],
)

py_proto_library(
    name = "tensor_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":tensor_proto"],
)

proto_library(
    name = "agg_core_proto",
    srcs = ["agg_core.proto"],
    deps = [":tensor_proto"],
)

cc_proto_library(
    name = "agg_core_cc_proto",
    deps = [":agg_core_proto"],
)

cc_library(
    name = "tensor",
    srcs = [
        "datatype.cc",
        "input_tensor_list.cc",
        "tensor.cc",
        "tensor_data.cc",
        "tensor_shape.cc",
        "tensor_spec.cc",
    ],
    hdrs = [
        "agg_vector.h",
        "agg_vector_iterator.h",
        "datatype.h",
        "input_tensor_list.h",
        "mutable_vector_data.h",
        "tensor.h",
        "tensor_data.h",
        "tensor_shape.h",
        "tensor_spec.h",
        "vector_string_data.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "aggregator",
    srcs = [
        "tensor_aggregator.cc",
        "tensor_aggregator_registry.cc",
    ],
    hdrs = [
        "agg_vector_aggregator.h",
        "aggregator.h",
        "one_dim_grouping_aggregator.h",
        "tensor_aggregator.h",
        "tensor_aggregator_factory.h",
        "tensor_aggregator_registry.h",
    ],
    deps = [
        ":agg_core_cc_proto",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "intrinsic",
    hdrs = ["intrinsic.h"],
    visibility = ["//visibility:public"],
    deps = [":tensor"],
)

cc_library(
    name = "dp_noise_mechanisms",
    srcs = ["dp_noise_mechanisms.cc"],
    hdrs = [
        "dp_noise_mechanisms.h",
    ],
    deps = [
        ":dp_fedsql_constants",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_cc_differential_privacy//algorithms:numerical-mechanisms",
        "@com_google_cc_differential_privacy//algorithms:partition-selection",
    ],
)

cc_test(
    name = "dp_noise_mechanisms_test",
    srcs = ["dp_noise_mechanisms_test.cc"],
    deps = [
        ":dp_fedsql_constants",
        ":dp_noise_mechanisms",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_library(
    name = "composite_key_combiner",
    srcs = ["composite_key_combiner.cc"],
    hdrs = [
        "composite_key_combiner.h",
        "tensor_aggregator.h",
    ],
    deps = [
        ":aggregator",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:node_hash_map",
    ],
)

cc_library(
    name = "dp_composite_key_combiner",
    srcs = ["dp_composite_key_combiner.cc"],
    hdrs = [
        "dp_composite_key_combiner.h",
    ],
    deps = [
        ":composite_key_combiner",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "one_dim_grouping_aggregator",
    srcs = ["one_dim_grouping_aggregator.cc"],
    hdrs = [
        "one_dim_grouping_aggregator.h",
    ],
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
    ],
)

cc_library(
    name = "grouping_federated_sum",
    srcs = ["grouping_federated_sum.cc"],
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":intrinsic",
        ":one_dim_grouping_aggregator",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
    ],
    alwayslink = 1,
)

cc_library(
    name = "group_by_aggregator",
    srcs = ["group_by_aggregator.cc"],
    hdrs = [
        "group_by_aggregator.h",
    ],
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":composite_key_combiner",
        ":fedsql_constants",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dp_group_by_factory",
    srcs = ["dp_group_by_factory.cc"],
    hdrs = [
        "dp_group_by_factory.h",
    ],
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":dp_closed_domain_histogram",
        ":dp_composite_key_combiner",
        ":dp_fedsql_constants",
        ":dp_open_domain_histogram",
        ":group_by_aggregator",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dp_grouping_federated_sum",
    srcs = ["dp_grouping_federated_sum.cc"],
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":dp_fedsql_constants",
        ":intrinsic",
        ":one_dim_grouping_aggregator",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dp_open_domain_histogram",
    srcs = ["dp_open_domain_histogram.cc"],
    hdrs = [
        "dp_open_domain_histogram.h",
    ],
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":composite_key_combiner",
        ":dp_composite_key_combiner",
        ":dp_fedsql_constants",
        ":dp_noise_mechanisms",
        ":group_by_aggregator",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dp_closed_domain_histogram",
    srcs = ["dp_closed_domain_histogram.cc"],
    hdrs = [
        "dp_closed_domain_histogram.h",
    ],
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":composite_key_combiner",
        ":dp_composite_key_combiner",
        ":dp_fedsql_constants",
        ":dp_noise_mechanisms",
        ":group_by_aggregator",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_cc_differential_privacy//algorithms:numerical-mechanisms",
    ],
)

cc_library(
    name = "aggregation_cores",
    srcs = [
        "federated_mean.cc",
        "federated_sum.cc",
    ],
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":composite_key_combiner",
        ":dp_closed_domain_histogram",
        ":dp_composite_key_combiner",
        ":dp_fedsql_constants",
        ":dp_group_by_factory",
        ":dp_grouping_federated_sum",
        ":dp_noise_mechanisms",
        ":dp_open_domain_histogram",
        ":fedsql_constants",
        ":group_by_aggregator",
        ":grouping_federated_sum",
        ":intrinsic",
        ":one_dim_grouping_aggregator",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
    ],
    alwayslink = 1,
)

cc_test(
    name = "tensor_test",
    srcs = [
        "tensor_data_test.cc",
        "tensor_shape_test.cc",
        "tensor_spec_test.cc",
        "tensor_test.cc",
    ],
    deps = [
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:protobuf_matchers",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "agg_vector_test",
    srcs = ["agg_vector_test.cc"],
    deps = [
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
    ],
)

cc_test(
    name = "aggregator_test",
    srcs = ["agg_vector_aggregator_test.cc"],
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "tensor_aggregator_registry_test",
    srcs = ["tensor_aggregator_registry_test.cc"],
    deps = [
        ":aggregator",
        ":intrinsic",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_library(
    name = "mutable_string_data",
    hdrs = ["mutable_string_data.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor",
        "@com_google_absl//absl/container:fixed_array",
    ],
)

cc_library(
    name = "fedsql_constants",
    hdrs = ["fedsql_constants.h"],
)

cc_library(
    name = "dp_fedsql_constants",
    hdrs = ["dp_fedsql_constants.h"],
)

cc_test(
    name = "federated_mean_test",
    srcs = ["federated_mean_test.cc"],
    deps = [
        ":agg_core_cc_proto",
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "federated_sum_test",
    srcs = ["federated_sum_test.cc"],
    deps = [
        ":agg_core_cc_proto",
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "input_tensor_list_test",
    srcs = ["input_tensor_list_test.cc"],
    deps = [
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
    ],
)

cc_test(
    name = "composite_key_combiner_test",
    srcs = ["composite_key_combiner_test.cc"],
    deps = [
        ":composite_key_combiner",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "dp_composite_key_combiner_test",
    srcs = ["dp_composite_key_combiner_test.cc"],
    deps = [
        ":aggregator",
        ":dp_composite_key_combiner",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "mutable_vector_data_test",
    srcs = ["mutable_vector_data_test.cc"],
    deps = [
        ":tensor",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "vector_string_data_test",
    srcs = [
        "vector_string_data.h",
        "vector_string_data_test.cc",
    ],
    deps = [
        ":tensor",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "mutable_string_data_test",
    srcs = ["mutable_string_data_test.cc"],
    deps = [
        ":mutable_string_data",
        ":tensor",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "one_dim_grouping_aggregator_test",
    srcs = ["one_dim_grouping_aggregator_test.cc"],
    deps = [
        ":agg_core_cc_proto",
        ":one_dim_grouping_aggregator",
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "group_by_aggregator_test",
    srcs = ["group_by_aggregator_test.cc"],
    deps = [
        ":aggregator",
        ":group_by_aggregator",
        ":grouping_federated_sum",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "dp_open_domain_histogram_test",
    srcs = ["dp_open_domain_histogram_test.cc"],
    deps = [
        ":aggregator",
        ":dp_fedsql_constants",
        ":dp_group_by_factory",
        ":dp_grouping_federated_sum",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "dp_closed_domain_histogram_test",
    srcs = ["dp_closed_domain_histogram_test.cc"],
    deps = [
        ":aggregator",
        ":dp_closed_domain_histogram",
        ":dp_composite_key_combiner",
        ":dp_fedsql_constants",
        ":dp_group_by_factory",
        ":dp_grouping_federated_sum",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "grouping_federated_sum_test",
    srcs = ["grouping_federated_sum_test.cc"],
    deps = [
        ":aggregator",
        ":grouping_federated_sum",
        ":intrinsic",
        ":tensor",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
    ],
)

cc_test(
    name = "dp_grouping_federated_sum_test",
    srcs = ["dp_grouping_federated_sum_test.cc"],
    deps = [
        ":aggregator",
        ":dp_fedsql_constants",
        ":dp_grouping_federated_sum",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/base",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "federated_sum_bench",
    testonly = 1,
    srcs = ["federated_sum_bench.cc"],
    linkstatic = 1,
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_binary(
    name = "group_by_bench",
    testonly = 1,
    srcs = ["group_by_bench.cc"],
    linkstatic = 1,
    deps = [
        ":aggregator",
        ":group_by_aggregator",
        ":grouping_federated_sum",
        ":intrinsic",
        ":mutable_string_data",
        ":tensor",
        ":tensor_cc_proto",
        "@com_google_benchmark//:benchmark",
    ],
)
