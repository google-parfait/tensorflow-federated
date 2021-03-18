load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "optimizers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/common_libs:py_typecheck"],
)

py_test(
    name = "optimizer_test",
    srcs = ["optimizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "optimizer_test_utils",
    srcs = ["optimizer_test_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "optimizer_test_utils_test",
    srcs = ["optimizer_test_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizer_test_utils",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "sgdm",
    srcs = ["sgdm.py"],
    srcs_version = "PY3",
    deps = [":optimizer"],
)

py_test(
    name = "sgdm_test",
    srcs = ["sgdm_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizer_test_utils",
        ":sgdm",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)
