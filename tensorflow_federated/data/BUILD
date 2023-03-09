load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":data_packages",
        ":data_users",
    ],
)

package_group(
    name = "data_packages",
    packages = ["//tensorflow_federated/data/..."],
)

package_group(
    name = "data_users",
    includes = [
        "//tensorflow_federated/python:python_packages",
        "//tensorflow_federated/tools:tools_packages",
    ],
)

licenses(["notice"])

copy_file(
    name = "_worker_binary",
    src = "//tensorflow_federated/cc/simulation:worker_binary",
    out = "worker_binary",
)
