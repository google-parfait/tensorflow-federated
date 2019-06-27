package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "baseline_fedavg",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":simple_fedavg",
    ],
)

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
    deps = [
        ":simple_fedavg",
        "//tensorflow_federated",
        "//tensorflow_federated/python/examples/mnist:models",
    ],
)
