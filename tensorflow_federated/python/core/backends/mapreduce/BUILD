package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "mapreduce",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/core/backends:__pkg__"],
    deps = [":canonical_form"],
)

py_library(
    name = "canonical_form",
    srcs = ["canonical_form.py"],
    deps = [
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api",
    ],
)
