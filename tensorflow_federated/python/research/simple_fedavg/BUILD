package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "simple_fedavg",
    srcs = ["simple_fedavg.py"],
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_test(
    name = "simple_fedavg_test",
    size = "large",
    srcs = ["simple_fedavg_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":simple_fedavg",
        "//tensorflow_federated",
    ],
)
