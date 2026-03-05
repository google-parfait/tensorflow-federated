load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":execution_contexts_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
    ],
)

package_group(
    name = "execution_contexts_packages",
    packages = ["//tensorflow_federated/python/core/impl/execution_contexts/..."],
)

licenses(["notice"])

py_library(
    name = "execution_contexts",
    srcs = ["__init__.py"],
    visibility = ["//tools/python_package:python_package_tool"],
)

py_library(
    name = "mergeable_comp_execution_context",
    srcs = ["mergeable_comp_execution_context.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:async_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "mergeable_comp_execution_context_test",
    size = "small",
    timeout = "moderate",
    srcs = ["mergeable_comp_execution_context_test.py"],
    deps = [
        ":mergeable_comp_execution_context",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
    ],
)
