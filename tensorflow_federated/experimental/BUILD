load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "experimental",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//third_party/tensorflow_federated:__pkg__"],
    deps = [
        "//third_party/tensorflow_federated/experimental/python/core/api:computations",
        "//third_party/tensorflow_federated/experimental/python/learning",
    ],
)
