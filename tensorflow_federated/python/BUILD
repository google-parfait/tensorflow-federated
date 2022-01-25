load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:private"])

package_group(
    name = "python_packages",
    packages = [
        "//third_party/tensorflow_federated/experimental/python/...",
        "//third_party/tensorflow_federated/python/...",
    ],
)

package_group(
    name = "internal_packages",
    packages = [
        "//third_party/tensorflow_federated/experimental/python/...",
        "//third_party/tensorflow_federated/python/...",

        # The examples and tests in TFF should depend on the public API.
        "-//third_party/tensorflow_federated/experimental/python/examples/...",
        "-//third_party/tensorflow_federated/python/examples/...",
        "-//third_party/tensorflow_federated/python/tests/...",
    ],
)

licenses(["notice"])

py_library(
    name = "python",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//third_party/tensorflow_federated/tools/python_package:python_package_tool"],
)
