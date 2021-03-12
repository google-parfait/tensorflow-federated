load("@rules_python//python:defs.bzl", "py_library")

# Packages that have visibility into the `impl` package.
package_group(
    name = "impl_visibility",
    includes = ["//tensorflow_federated/python/core:core_visibility"],
    packages = [
        "//tensorflow_federated",
        "//tensorflow_federated/python/core/api/...",
        "//tensorflow_federated/python/core/backends/...",
        "//tensorflow_federated/python/core/framework/...",
        "//tensorflow_federated/python/core/templates/...",
        "//tensorflow_federated/python/core/test/...",
        "//tensorflow_federated/python/core/types/...",
        "//tensorflow_federated/python/core/utils/...",
    ],
)

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "impl",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)
