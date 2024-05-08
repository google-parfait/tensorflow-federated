load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_python//python:defs.bzl", "py_binary")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [":tools_packages"],
)

package_group(
    name = "tools_packages",
    packages = ["//tensorflow_federated/tools/..."],
)

licenses(["notice"])

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = [
        "//tensorflow_federated:__subpackages__",  # Scheuklappen: keep
    ],
    deps = [
        "@rules_python//python:defs.bzl",
    ],
)

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    deps = ["@pypi_absl_py//:pkg"],
)
