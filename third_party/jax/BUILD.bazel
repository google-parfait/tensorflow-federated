load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "jax",
    visibility = ["//tensorflow_federated:__subpackages__"],
    deps = [
        "@pypi_jax//:pkg",
        "@pypi_jaxlib//:pkg",
    ],
)
