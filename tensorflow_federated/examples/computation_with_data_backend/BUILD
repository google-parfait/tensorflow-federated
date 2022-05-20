load("@rules_python//python:defs.bzl", "py_binary")

licenses(["notice"])

py_binary(
    name = "data_backend_list",
    srcs = ["data_backend_list.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "@absl_py//absl:app",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "data_backend_numpy",
    srcs = ["data_backend_numpy.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "@absl_py//absl:app",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "data_backend_scalar",
    srcs = ["data_backend_scalar.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "@absl_py//absl:app",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
