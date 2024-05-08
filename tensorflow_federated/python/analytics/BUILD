load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":analytics_packages",
        ":analytics_users",
    ],
)

package_group(
    name = "analytics_packages",
    packages = ["//tensorflow_federated/python/analytics/..."],
)

package_group(
    name = "analytics_users",
    includes = ["//tensorflow_federated/python/simulation:simulation_packages"],
)

licenses(["notice"])

py_library(
    name = "analytics",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":count_distinct",
        ":data_processing",
        ":differential_privacy",
        ":histogram_processing",
        "//tensorflow_federated/python/analytics/heavy_hitters",
        "//tensorflow_federated/python/analytics/heavy_hitters/iblt:iblt_factory",
        "//tensorflow_federated/python/analytics/hierarchical_histogram:hierarchical_histogram_decoder",
        "//tensorflow_federated/python/analytics/hierarchical_histogram:hierarchical_histogram_lib",
    ],
)

py_library(
    name = "data_processing",
    srcs = ["data_processing.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "data_processing_test",
    srcs = ["data_processing_test.py"],
    deps = [
        ":data_processing",
        ":histogram_test_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "histogram_test_utils",
    testonly = True,
    srcs = ["histogram_test_utils.py"],
    deps = ["@pypi_tensorflow//:pkg"],
)

py_library(
    name = "histogram_processing",
    srcs = ["histogram_processing.py"],
    deps = ["@pypi_tensorflow//:pkg"],
)

py_test(
    name = "histogram_processing_test",
    srcs = ["histogram_processing_test.py"],
    deps = [
        ":histogram_processing",
        ":histogram_test_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "differential_privacy",
    srcs = ["differential_privacy.py"],
)

py_library(
    name = "count_distinct",
    srcs = ["count_distinct.py"],
    deps = [
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "differential_privacy_test",
    srcs = ["differential_privacy_test.py"],
    deps = [
        ":differential_privacy",
        "@pypi_absl_py//:pkg",
    ],
)

py_test(
    name = "count_distinct_test",
    srcs = ["count_distinct_test.py"],
    deps = [
        ":count_distinct",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
