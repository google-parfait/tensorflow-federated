load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":templates_packages",
        "//tensorflow_federated/python/core:core_users",
        "//tensorflow_federated/python/core/backends:backends_packages",
    ],
)

package_group(
    name = "templates_packages",
    packages = ["//tensorflow_federated/python/core/templates/..."],
)

licenses(["notice"])

py_library(
    name = "templates",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":aggregation_process",
        ":estimation_process",
        ":iterative_process",
        ":measured_process",
    ],
)

py_library(
    name = "aggregation_process",
    srcs = ["aggregation_process.py"],
    deps = [
        ":errors",
        ":measured_process",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "aggregation_process_test",
    size = "small",
    srcs = ["aggregation_process_test.py"],
    deps = [
        ":aggregation_process",
        ":errors",
        ":measured_process",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
)

py_library(
    name = "estimation_process",
    srcs = ["estimation_process.py"],
    deps = [
        ":errors",
        ":iterative_process",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
    ],
)

py_test(
    name = "estimation_process_test",
    size = "small",
    srcs = ["estimation_process_test.py"],
    deps = [
        ":errors",
        ":estimation_process",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "iterative_process",
    srcs = ["iterative_process.py"],
    deps = [
        ":errors",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "iterative_process_test",
    size = "small",
    srcs = ["iterative_process_test.py"],
    deps = [
        ":errors",
        ":iterative_process",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "measured_process",
    srcs = ["measured_process.py"],
    deps = [
        ":errors",
        ":iterative_process",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_attrs//:pkg",
        "@pypi_dm_tree//:pkg",
    ],
)

py_test(
    name = "measured_process_test",
    size = "small",
    srcs = ["measured_process_test.py"],
    args = [
        "--golden",
        "$(location measured_process_test_goldens/composition_with_aggregation_processes.expected)",
        "--golden",
        "$(location measured_process_test_goldens/composition_with_measured_processes.expected)",
        "--golden",
        "$(location measured_process_test_goldens/concatenation_with_measured_processes.expected)",
        "--golden",
        "$(location measured_process_test_goldens/concatenation_with_aggregation_processes.expected)",
        "--verbose",
    ],
    data = [
        "measured_process_test_goldens/composition_with_aggregation_processes.expected",
        "measured_process_test_goldens/composition_with_measured_processes.expected",
        "measured_process_test_goldens/concatenation_with_aggregation_processes.expected",
        "measured_process_test_goldens/concatenation_with_measured_processes.expected",
    ],
    deps = [
        ":aggregation_process",
        ":errors",
        ":iterative_process",
        ":measured_process",
        "//tensorflow_federated/python/core/impl/compiler:compiler_test_utils",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)
