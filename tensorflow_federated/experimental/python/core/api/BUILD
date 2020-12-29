load("@rules_python//python:defs.bzl", "py_library")

package_group(
    name = "default_visibility",
    packages = ["//tensorflow_federated/experimental/..."],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "api",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "computations",
    srcs = ["computations.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/experimental/python/core/impl/wrappers:computation_wrapper_instances"],
)
