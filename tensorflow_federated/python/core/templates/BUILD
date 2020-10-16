load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core:core_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/templates/...",

        # Core Dependencies
        "//tensorflow_federated/python/core/backends/...",
        "//tensorflow_federated/python/core/utils/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "templates",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow_federated:__pkg__",
    ],
    deps = [
        ":aggregation_process",
        ":iterative_process",
        ":measured_process",
    ],
)

py_library(
    name = "aggregation_process",
    srcs = ["aggregation_process.py"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":measured_process",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:placements",
    ],
)

py_test(
    name = "aggregation_process_test",
    size = "small",
    srcs = ["aggregation_process_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregation_process",
        ":errors",
        ":measured_process",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
    srcs_version = "PY3",
)

py_library(
    name = "estimation_process",
    srcs = ["estimation_process.py"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":iterative_process",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
    ],
)

py_test(
    name = "estimation_process_test",
    size = "small",
    srcs = ["estimation_process_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":estimation_process",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "iterative_process",
    srcs = ["iterative_process.py"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "iterative_process_test",
    size = "small",
    srcs = ["iterative_process_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":iterative_process",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "measured_process",
    srcs = ["measured_process.py"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":iterative_process",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "measured_process_test",
    size = "small",
    srcs = ["measured_process_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":measured_process",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)
