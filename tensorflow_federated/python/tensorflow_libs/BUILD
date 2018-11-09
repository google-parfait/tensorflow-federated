package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tensor_utils_test",
    size = "small",
    srcs = ["tensor_utils_test.py"],
    deps = [
        ":tensor_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
