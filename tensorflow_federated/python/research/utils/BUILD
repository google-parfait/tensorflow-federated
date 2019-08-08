package(default_visibility = ["//tensorflow_federated/python/research:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":utils"],
)

py_library(
    name = "training_loops",
    srcs = ["training_loops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
    ],
)

py_test(
    name = "training_loops_test",
    size = "large",
    srcs = ["training_loops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":training_loops",
        "//tensorflow_federated",
        "//tensorflow_federated/python/examples/mnist:models",
    ],
)
