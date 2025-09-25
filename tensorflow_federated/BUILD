load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "tensorflow_federated",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":version",
        "//tensorflow_federated/python/aggregators",
        "//tensorflow_federated/python/analytics",
        "//tensorflow_federated/python/common_libs:async_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/backends",
        "//tensorflow_federated/python/core/environments/tensorflow",
        "//tensorflow_federated/python/core/framework",
        "//tensorflow_federated/python/core/templates",
        "//tensorflow_federated/python/learning",
        "//tensorflow_federated/python/program",
        "//tensorflow_federated/python/simulation",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
)
