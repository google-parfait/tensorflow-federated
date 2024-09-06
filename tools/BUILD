load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_python//python:defs.bzl", "py_binary")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [":tools_packages"],
)

package_group(
    name = "tools_packages",
    packages = ["//tools/..."],
)

licenses(["notice"])

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = [
        "//tensorflow_federated:__subpackages__",  # Scheuklappen: keep
    ],
    deps = [
        "//third_party/tensorflow:tensorflow_bzl",
        "@rules_python//python:defs.bzl",
    ],
)

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    deps = [
        "//third_party/py/tensorflow_docs/api_generator",
        "@pypi_absl_py//:pkg",
    ],
)

filegroup(
    name = "update_version",
    srcs = ["update_version.sh"],
    tags = ["ignore_srcs"],
)
