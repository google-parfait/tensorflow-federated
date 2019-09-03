package(default_visibility = ["//tensorflow_federated:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "datasets",
    srcs = ["__init__.py"],
    deps = [
        ":dataset_utils",
        "//tensorflow_federated/python/simulation/datasets/emnist",
        "//tensorflow_federated/python/simulation/datasets/shakespeare",
        "//tensorflow_federated/python/simulation/datasets/stackoverflow",
    ],
)

py_library(
    name = "dataset_utils",
    srcs = ["dataset_utils.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "dataset_utils_test",
    size = "small",
    srcs = ["dataset_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset_utils",
        "//tensorflow_federated/python/simulation:client_data",
    ],
)
