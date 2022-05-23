load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = [
    ":wrappers_packages",
    "//tensorflow_federated/experimental/python/core/impl:impl_visibility",
])

package_group(
    name = "wrappers_packages",
    packages = ["//tensorflow_federated/experimental/python/core/impl/wrappers/..."],
)

licenses(["notice"])

py_library(
    name = "wrappers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)
