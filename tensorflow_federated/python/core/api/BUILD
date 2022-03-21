load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":api_packages",
    "//tensorflow_federated/python/core:core_visibility",
    "//tensorflow_federated/python/core/backends:backends_packages",
    "//tensorflow_federated/python/core/templates:templates_packages",
    "//tensorflow_federated/python/core/test:test_packages",
    "//tensorflow_federated/python/core/types:types_packages",

    # Namespace visibility
    "//tensorflow_federated:__pkg__",

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
    name = "computation_base",
    srcs = ["computation_base.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/types:typed_object"],
)

py_library(
    name = "computations",
    srcs = ["computations.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/wrappers:computation_wrapper_instances"],
)

py_library(
    name = "test_case",
    srcs = ["test_case.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/types:computation_types"],
)

py_test(
    name = "test_case_test",
    size = "small",
    srcs = ["test_case_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":test_case",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)
