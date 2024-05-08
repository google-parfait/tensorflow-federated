load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":common_libs_packages",
        "//tensorflow_federated/python:internal_packages",

        # Python package namespace
        "//tensorflow_federated:__pkg__",
    ],
)

package_group(
    name = "common_libs_packages",
    packages = ["//tensorflow_federated/python/common_libs/..."],
)

licenses(["notice"])

py_library(
    name = "common_libs",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "async_utils",
    srcs = ["async_utils.py"],
    deps = [":tracing"],
)

py_test(
    name = "async_utils_test",
    srcs = ["async_utils_test.py"],
    deps = [
        ":async_utils",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "deprecation",
    srcs = ["deprecation.py"],
)

py_test(
    name = "deprecation_test",
    srcs = ["deprecation_test.py"],
    deps = [
        ":deprecation",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "golden",
    testonly = True,
    srcs = ["golden.py"],
    deps = ["@pypi_absl_py//:pkg"],
)

py_test(
    name = "golden_test",
    size = "small",
    srcs = ["golden_test.py"],
    args = [
        "--golden",
        "$(location golden_test_goldens/test_check_string_fails.expected)",
        "--golden",
        "$(location golden_test_goldens/test_check_string_succeeds.expected)",
        "--golden",
        "$(location golden_test_goldens/test_check_string_updates.expected)",
    ],
    data = [
        "golden_test_goldens/test_check_string_fails.expected",
        "golden_test_goldens/test_check_string_succeeds.expected",
        "golden_test_goldens/test_check_string_updates.expected",
    ],
    deps = [
        ":golden",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "py_typecheck",
    srcs = ["py_typecheck.py"],
    deps = [
        "@pypi_attrs//:pkg",
        "@pypi_typing_extensions//:pkg",
    ],
)

py_test(
    name = "py_typecheck_test",
    size = "small",
    srcs = ["py_typecheck_test.py"],
    deps = [
        ":py_typecheck",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "retrying",
    srcs = ["retrying.py"],
    deps = [
        ":py_typecheck",
    ],
)

py_test(
    name = "retrying_test",
    size = "small",
    srcs = ["retrying_test.py"],
    deps = [
        ":retrying",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "serializable",
    srcs = ["serializable.py"],
)

py_library(
    name = "structure",
    srcs = ["structure.py"],
    deps = [
        ":py_typecheck",
        "@pypi_attrs//:pkg",
        "@pypi_dm_tree//:pkg",
    ],
)

py_test(
    name = "structure_test",
    size = "small",
    srcs = ["structure_test.py"],
    deps = [
        ":structure",
        "@pypi_absl_py//:pkg",
        "@pypi_attrs//:pkg",
    ],
)

py_library(
    name = "tracing",
    srcs = ["tracing.py"],
    deps = [
        ":py_typecheck",
        "@pypi_absl_py//:pkg",
    ],
)

py_test(
    name = "tracing_test",
    size = "small",
    srcs = ["tracing_test.py"],
    deps = [
        ":tracing",
        "@pypi_absl_py//:pkg",
    ],
)
