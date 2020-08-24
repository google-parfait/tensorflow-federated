package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "backends",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow_federated:__pkg__",
    ],
    deps = [
        "//tensorflow_federated/python/core/backends/mapreduce",
        "//tensorflow_federated/python/core/backends/native",
        "//tensorflow_federated/python/core/backends/reference",
    ],
)
