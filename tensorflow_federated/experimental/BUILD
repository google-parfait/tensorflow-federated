load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "experimental",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = ["//tensorflow_federated/experimental/python/core/api:computations"],
)
