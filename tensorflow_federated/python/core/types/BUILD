load("@rules_python//python:defs.bzl", "py_library")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core:core_visibility"],
    packages = [
        "//tensorflow_federated/python/core/types/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "types",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)
