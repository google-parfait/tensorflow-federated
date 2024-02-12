load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [":tensorflow_packages"],
)

package_group(
    name = "tensorflow_packages",
    packages = ["//tensorflow_federated/python/core/environments/tensorflow/..."],
)

licenses(["notice"])

py_library(
    name = "tensorflow",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
    ],
)
