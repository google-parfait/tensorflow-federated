load("@rules_python//python:defs.bzl", "py_library")

package_group(
    name = "default_visibility",
    packages = ["//tensorflow_federated/experimental/python/core/impl/..."],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "impl",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)
