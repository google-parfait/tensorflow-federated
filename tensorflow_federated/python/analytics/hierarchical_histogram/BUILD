load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":hierarchical_histogram_packages",
        "//tensorflow_federated/python/analytics:analytics_users",

        # Python package namespace
        "//tensorflow_federated/python/analytics:__pkg__",
    ],
)

package_group(
    name = "hierarchical_histogram_packages",
    packages = ["//tensorflow_federated/python/analytics/hierarchical_histogram/..."],
)

licenses(["notice"])

py_library(
    name = "hierarchical_histogram",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "hierarchical_histogram_factory",
    srcs = ["hierarchical_histogram_factory.py"],
    deps = [
        ":clipping_factory",
        "//tensorflow_federated/python/aggregators:differential_privacy",
        "//tensorflow_federated/python/aggregators:secure",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "@pypi_tensorflow_privacy//:pkg",
    ],
)

py_test(
    name = "hierarchical_histogram_factory_test",
    timeout = "long",
    srcs = ["hierarchical_histogram_factory_test.py"],
    shard_count = 4,
    deps = [
        ":build_tree_from_leaf",
        ":hierarchical_histogram_factory",
        "//tensorflow_federated/python/aggregators:differential_privacy",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "hierarchical_histogram_lib",
    srcs = ["hierarchical_histogram_lib.py"],
    deps = [
        ":clipping_factory",
        ":hierarchical_histogram_factory",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "@pypi_attrs//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "hierarchical_histogram_lib_test",
    srcs = ["hierarchical_histogram_lib_test.py"],
    shard_count = 6,
    deps = [
        ":hierarchical_histogram_lib",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/test:static_assert",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "hierarchical_histogram_decoder",
    srcs = ["hierarchical_histogram_decoder.py"],
    deps = [
        ":build_tree_from_leaf",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "hierarchical_histogram_decoder_test",
    srcs = ["hierarchical_histogram_decoder_test.py"],
    deps = [
        ":build_tree_from_leaf",
        ":hierarchical_histogram_decoder",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "build_tree_from_leaf",
    srcs = ["build_tree_from_leaf.py"],
    deps = ["@pypi_numpy//:pkg"],
)

py_test(
    name = "build_tree_from_leaf_test",
    srcs = ["build_tree_from_leaf_test.py"],
    deps = [
        ":build_tree_from_leaf",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "clipping_factory",
    srcs = ["clipping_factory.py"],
    deps = [
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "clipping_factory_test",
    srcs = ["clipping_factory_test.py"],
    deps = [
        ":clipping_factory",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
