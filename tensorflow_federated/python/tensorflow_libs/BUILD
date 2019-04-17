package(default_visibility = ["//third_party/tensorflow_federated:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "tensor_utils_test",
    size = "small",
    srcs = ["tensor_utils_test.py"],
    deps = [
        ":tensor_utils",
        "//tensorflow_federated/python/common_libs:test",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "graph_keys",
    srcs = ["graph_keys.py"],
)
