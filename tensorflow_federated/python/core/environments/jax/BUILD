load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [":jax_packages"],
)

package_group(
    name = "jax_packages",
    packages = ["//tensorflow_federated/python/core/environments/jax/..."],
)

licenses(["notice"])

py_library(
    name = "jax",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        "//tensorflow_federated/python/core/environments/jax_frontend:jax_computation",
    ],
)
