load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = [
    # ":data_packages",
    # ":data_users",
    "//visibility:public",
])

package_group(
    name = "data_packages",
    packages = ["//tensorflow_federated/data/..."],
)

package_group(
    name = "data_users",
    includes = [
        "//tensorflow_federated/data:data_packages",
        "//tensorflow_federated/tools:tools_packages",
    ],
)

licenses(["notice"])

py_library(
    name = "data",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

copy_file(
    name = "worker_binary2",
    src = "//tensorflow_federated/cc/simulation:worker_binary",
    out = "worker_binary",
)

genrule(
    name = "worker_binary_xz",
    srcs = ["//tensorflow_federated/cc/simulation:worker_binary"],
    outs = ["worker_binary.xz"],
    cmd = "$(location @liblzma//:xz) --extreme  --stdout $< > $@",
    executable = True,
    output_to_bindir = True,
    tools = ["@liblzma//:xz"],
)
