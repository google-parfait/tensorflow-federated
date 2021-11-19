load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = [":tools_packages"])

package_group(
    name = "tools_packages",
    packages = ["//tensorflow_federated/tools/..."],
)

licenses(["notice"])

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    deps = ["@rules_python//python:defs.bzl"],
)

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
