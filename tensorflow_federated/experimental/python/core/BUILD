load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:private"])

package_group(
    name = "core_packages",
    packages = ["//tensorflow_federated/experimental/python/core/..."],
)

package_group(
    name = "core_visibility",
    includes = ["//tensorflow_federated/experimental/python/learning:learning_packages"],
)

licenses(["notice"])

py_library(
    name = "core",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)
