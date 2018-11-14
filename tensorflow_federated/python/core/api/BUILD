package(default_visibility = ["//tensorflow_federated/python/core:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# This is the import for other components of TFF that are layered upon the Core API.
py_library(
    name = "api",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":computation_base",
        ":computations",
        ":types",
    ],
)

py_library(
    name = "computation_base",
    srcs = ["computation_base.py"],
)

py_library(
    name = "computations",
    srcs = ["computations.py"],
    deps = [
        "//tensorflow_federated/python/core/impl:computation_wrapper_instances",
    ],
)

py_test(
    name = "computations_test",
    size = "small",
    srcs = ["computations_test.py"],
    deps = [
        ":computations",
        ":types",
        ":value_base",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "types_test",
    size = "small",
    srcs = ["types_test.py"],
    deps = [
        ":types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "value_base",
    srcs = ["value_base.py"],
)
