load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "p13n_utils",
    srcs = ["p13n_utils.py"],
    deps = [
        "//tensorflow_federated",
        "//third_party/tensorflow",
    ],
)

py_test(
    name = "p13n_utils_test",
    srcs = ["p13n_utils_test.py"],
    deps = [
        ":p13n_utils",
        "//tensorflow_federated",
        "//third_party/tensorflow",
        "@pypi_numpy//:pkg",
    ],
)

py_binary(
    name = "emnist_p13n_main",
    srcs = ["emnist_p13n_main.py"],
    deps = [
        ":p13n_utils",
        "//tensorflow_federated",
        "//third_party/tensorflow",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)
