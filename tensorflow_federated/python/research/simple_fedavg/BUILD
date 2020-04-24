package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_binary(
    name = "emnist_fedavg_main",
    srcs = ["emnist_fedavg_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":simple_fedavg_tf",
        ":simple_fedavg_tff",
        "//tensorflow_federated",
    ],
)

py_library(
    name = "simple_fedavg_tf",
    srcs = ["simple_fedavg_tf.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_library(
    name = "simple_fedavg_tff",
    srcs = ["simple_fedavg_tff.py"],
    srcs_version = "PY3",
    deps = [
        ":simple_fedavg_tf",
        "//tensorflow_federated",
    ],
)

py_test(
    name = "simple_fedavg_test",
    size = "large",
    srcs = ["simple_fedavg_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["requires-gpu-sm35"],
    deps = [
        ":simple_fedavg_tf",
        ":simple_fedavg_tff",
        "//tensorflow_federated",
    ],
)
