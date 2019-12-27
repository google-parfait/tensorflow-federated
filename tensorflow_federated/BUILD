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
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:typed_object",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/api:values",
        "//tensorflow_federated/python/core/backends",
        "//tensorflow_federated/python/core/framework",
        "//tensorflow_federated/python/core/utils",
        "//tensorflow_federated/python/learning",
        "//tensorflow_federated/python/simulation",
    ],
)

config_setting(
    name = "py2_mode",
    flag_values = {"@bazel_tools//tools/python:python_version": "PY2"},
    visibility = [":__subpackages__"],
)

config_setting(
    name = "py3_mode",
    flag_values = {"@bazel_tools//tools/python:python_version": "PY3"},
    visibility = [":__subpackages__"],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    srcs_version = "PY3",
)
