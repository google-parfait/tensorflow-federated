load("@rules_python//python:defs.bzl", "py_binary")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_binary(
    name = "load_flair",
    srcs = ["load_flair.py"],
    deps = [
        "//tensorflow_federated",
        "@pypi_absl_py//:pkg",
    ],
)
