load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = [
    ":types_packages",
    "//tensorflow_federated/python/core:core_visibility",
])

package_group(
    name = "types_packages",
    packages = ["//tensorflow_federated/python/core/types/..."],
)

licenses(["notice"])

py_library(
    name = "types",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)
