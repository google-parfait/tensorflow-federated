load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_federated",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":version",
        "//third_party/tensorflow_federated/experimental",
        "//third_party/tensorflow_federated/python/aggregators",
        "//third_party/tensorflow_federated/python/analytics",
        "//third_party/tensorflow_federated/python/common_libs:structure",
        "//third_party/tensorflow_federated/python/common_libs:tracing",
        "//third_party/tensorflow_federated/python/core/api:computation_base",
        "//third_party/tensorflow_federated/python/core/api:computations",
        "//third_party/tensorflow_federated/python/core/backends",
        "//third_party/tensorflow_federated/python/core/framework",
        "//third_party/tensorflow_federated/python/core/impl/federated_context:data",
        "//third_party/tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//third_party/tensorflow_federated/python/core/impl/federated_context:value_impl",
        "//third_party/tensorflow_federated/python/core/impl/types:computation_types",
        "//third_party/tensorflow_federated/python/core/impl/types:placements",
        "//third_party/tensorflow_federated/python/core/impl/types:type_conversions",
        "//third_party/tensorflow_federated/python/core/impl/types:typed_object",
        "//third_party/tensorflow_federated/python/core/templates",
        "//third_party/tensorflow_federated/python/core/test",
        "//third_party/tensorflow_federated/python/core/types",
        "//third_party/tensorflow_federated/python/learning",
        "//third_party/tensorflow_federated/python/program",
        "//third_party/tensorflow_federated/python/simulation",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    srcs_version = "PY3",
)
