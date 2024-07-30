load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":test_packages",
        "//tensorflow_federated/python/core:core_users",
    ],
)

package_group(
    name = "test_packages",
    packages = ["//tensorflow_federated/python/core/test/..."],
)

licenses(["notice"])

py_library(
    name = "test",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":static_assert",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_test_utils",
        "//tensorflow_federated/python/core/impl/context_stack:runtime_error_context",
        "//tensorflow_federated/python/core/impl/context_stack:set_default_context",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
    ],
)

py_library(
    name = "static_assert",
    srcs = ["static_assert.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
    ],
)

py_test(
    name = "static_assert_test",
    size = "small",
    srcs = ["static_assert_test.py"],
    deps = [
        ":static_assert",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
    ],
)
