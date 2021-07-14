load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:private"])

package_group(
    name = "impl_packages",
    packages = ["//tensorflow_federated/experimental/python/core/impl/..."],
)

package_group(
    name = "impl_visibility",
    includes = [
        "//tensorflow_federated/experimental/python/core/api:api_packages",
        "//tensorflow_federated/experimental/python/learning:learning_packages",
    ],
    packages = [
        # Namespace visibility
        "//tensorflow_federated/experimental",
    ],
)

licenses(["notice"])

py_library(
    name = "impl",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)
