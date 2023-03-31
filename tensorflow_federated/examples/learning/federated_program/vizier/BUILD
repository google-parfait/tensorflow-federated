load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "data_sources",
    srcs = ["data_sources.py"],
    deps = ["//tensorflow_federated"],
)

py_library(
    name = "learning_process",
    srcs = ["learning_process.py"],
    deps = ["//tensorflow_federated"],
)

py_binary(
    name = "program",
    srcs = ["program.py"],
    deps = [
        ":data_sources",
        ":learning_process",
        ":vizier_service",
        "//tensorflow_federated",
    ],
)

py_library(
    name = "vizier_service",
    srcs = ["vizier_service.py"],
)
