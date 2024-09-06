load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_binary(
    name = "emnist_fedavg_main",
    srcs = ["emnist_fedavg_main.py"],
    deps = [
        ":simple_fedavg_tff",
        "//tensorflow_federated",
        "//third_party/tensorflow",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "simple_fedavg_tf",
    srcs = ["simple_fedavg_tf.py"],
    deps = [
        "//tensorflow_federated",
        "//third_party/tensorflow",
        "@pypi_attrs//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "simple_fedavg_tff",
    srcs = ["simple_fedavg_tff.py"],
    deps = [
        ":simple_fedavg_tf",
        "//tensorflow_federated",
        "//third_party/tensorflow",
    ],
)

py_test(
    name = "simple_fedavg_test",
    size = "medium",
    srcs = ["simple_fedavg_test.py"],
    shard_count = 3,
    tags = ["nokokoro"],
    deps = [
        ":simple_fedavg_tf",
        ":simple_fedavg_tff",
        "//tensorflow_federated",
        "//third_party/tensorflow",
        "@pypi_absl_py//:pkg",
        "@pypi_attrs//:pkg",
        "@pypi_numpy//:pkg",
    ],
)
