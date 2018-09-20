package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# This is the public import users should use.
py_library(
    name = "python",
    srcs = ["__init__.py"],
    deps = ["//tensorflow_federated/python/core/api:computation_base"],
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
