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
    visibility = ["//tools/python_package:python_package_tool"],
)

py_library(
    name = "async_utils",
    srcs = ["async_utils.py"],
    deps = ["@federated_language//federated_language"],
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
        "@pypi//absl_py",
    ],
)

py_library(
    name = "golden",
    srcs = ["golden.py"],
    deps = ["@pypi//absl_py"],
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
        "@pypi//absl_py",
    ],
)

py_library(
    name = "py_typecheck",
    srcs = ["py_typecheck.py"],
    deps = [
        "@pypi//attrs",
        "@pypi//typing_extensions",
    ],
)

py_test(
    name = "py_typecheck_test",
    size = "small",
    srcs = ["py_typecheck_test.py"],
    deps = [
        ":py_typecheck",
        "@pypi//absl_py",
    ],
)

py_library(
    name = "structure",
    srcs = ["structure.py"],
    deps = [
        ":deprecation",
        "@federated_language//federated_language/common_libs:structure",
        "@pypi//attrs",  # buildcleaner: keep
        "@pypi//dm_tree",  # buildcleaner: keep
    ],
)
