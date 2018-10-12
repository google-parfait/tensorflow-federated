package(default_visibility = ["//tensorflow_federated/python/core/api:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "computation_impl",
    srcs = ["computation_impl.py"],
    deps = [
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/core/api:computation_base",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "computation_test",
    size = "small",
    srcs = ["computation_test.py"],
    deps = [
        ":computation_impl",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "serialization",
    srcs = ["serialization.py"],
    deps = [
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.py"],
    deps = [
        ":serialization",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)
