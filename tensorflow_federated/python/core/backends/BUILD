load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = [
    ":backends_packages",
    "//tensorflow_federated/python/core:core_visibility",
])

package_group(
    name = "backends_packages",
    packages = ["//tensorflow_federated/python/core/backends/..."],
)

licenses(["notice"])

py_library(
    name = "backends",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        "//tensorflow_federated/python/core/backends/mapreduce",
        "//tensorflow_federated/python/core/backends/native",
        "//tensorflow_federated/python/core/backends/reference",
        "//tensorflow_federated/python/core/backends/test",
        "//tensorflow_federated/python/core/backends/xla",
    ],
)
