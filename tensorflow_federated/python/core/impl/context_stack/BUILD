load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":context_stack_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
        "//tensorflow_federated/python/core/impl/computation:computation_packages",
        "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
        "//tensorflow_federated/python/core/impl/executors:executors_packages",
        "//tensorflow_federated/python/core/impl/federated_context:federated_context_packages",
    ],
)

package_group(
    name = "context_stack_packages",
    packages = ["//tensorflow_federated/python/core/impl/context_stack/..."],
)

licenses(["notice"])

py_library(
    name = "context_stack",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "context_base",
    srcs = ["context_base.py"],
)

py_library(
    name = "context_stack_base",
    srcs = ["context_stack_base.py"],
)

py_library(
    name = "context_stack_impl",
    srcs = ["context_stack_impl.py"],
    deps = [
        ":context_base",
        ":context_stack_base",
        ":runtime_error_context",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "context_stack_impl_test",
    size = "small",
    srcs = ["context_stack_impl_test.py"],
    deps = [
        ":context_stack_impl",
        ":context_stack_test_utils",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "context_stack_test_utils",
    srcs = ["context_stack_test_utils.py"],
    deps = [
        ":context_base",
        ":context_stack_impl",
        "@pypi_absl_py//:pkg",
    ],
)

py_test(
    name = "context_stack_test_utils_test",
    srcs = ["context_stack_test_utils_test.py"],
    deps = [
        ":context_stack_impl",
        ":context_stack_test_utils",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "get_context_stack",
    srcs = ["get_context_stack.py"],
    deps = [":context_stack_impl"],
)

py_test(
    name = "get_context_stack_test",
    size = "small",
    srcs = ["get_context_stack_test.py"],
    deps = [
        ":context_stack_impl",
        ":get_context_stack",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "runtime_error_context",
    srcs = ["runtime_error_context.py"],
    deps = [":context_base"],
)

py_library(
    name = "set_default_context",
    srcs = ["set_default_context.py"],
    deps = [
        ":context_stack_impl",
        ":runtime_error_context",
    ],
)

py_test(
    name = "set_default_context_test",
    size = "small",
    srcs = ["set_default_context_test.py"],
    deps = [
        ":context_stack_impl",
        ":context_stack_test_utils",
        ":set_default_context",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "symbol_binding_context",
    srcs = ["symbol_binding_context.py"],
    deps = [":context_base"],
)
