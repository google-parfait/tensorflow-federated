load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = [
    ":api_packages",
    "//tensorflow_federated/python/core:core_visibility",
    "//tensorflow_federated/python/core/templates:templates_packages",
    "//tensorflow_federated/python/core/test:test_packages",

    # TODO(b/181569610): Clean up circular dependency between the `api` and
    # `impl` packages.
    "//tensorflow_federated/python/core/impl:impl_packages",
])

package_group(
    name = "api_packages",
    packages = [
        "//tensorflow_federated/experimental/python/core/api/...",
        "//tensorflow_federated/python/core/api/...",
    ],
)

licenses(["notice"])

py_library(
    name = "api",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "computations",
    srcs = ["computations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
    ],
)
