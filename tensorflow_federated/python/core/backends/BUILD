load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

package_group(
    name = "backends_packages",
    packages = ["//tensorflow_federated/python/core/backends/..."],
)

licenses(["notice"])

py_library(
    name = "backends",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        "//tensorflow_federated/python/core/backends/mapreduce",
        "//tensorflow_federated/python/core/backends/native",
        "//tensorflow_federated/python/core/backends/test",
    ],
)
