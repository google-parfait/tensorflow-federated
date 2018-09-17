package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "python",
    srcs = ["__init__.py"],
)

py_test(
    name = "sanity_test",
    size = "small",
    srcs = ["sanity_test.py"],
    deps = [
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)
