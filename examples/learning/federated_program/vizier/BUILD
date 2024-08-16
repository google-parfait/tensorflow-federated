load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

py_library(
    name = "data_sources",
    srcs = ["data_sources.py"],
    deps = [
        "//tensorflow_federated",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "learning_process",
    srcs = ["learning_process.py"],
    deps = [
        "//tensorflow_federated",
        "@pypi_tensorflow//:pkg",
    ],
)

py_binary(
    name = "program",
    srcs = ["program.py"],
    data = [
        ":study_spec.textproto",
    ],
    deps = [
        ":data_sources",
        ":learning_process",
        ":vizier_service",
        "//tensorflow_federated",
        "@pypi_absl_py//:pkg",
        "@pypi_google_vizier//:pkg",
    ],
)

py_library(
    name = "vizier_service",
    srcs = ["vizier_service.py"],
    deps = ["@pypi_tensorflow//:pkg"],
)

py_test(
    name = "vizier_service_test",
    srcs = ["vizier_service_test.py"],
    data = [
        ":study_spec.textproto",
    ],
    deps = [
        ":vizier_service",
        "@pypi_absl_py//:pkg",
    ],
)
