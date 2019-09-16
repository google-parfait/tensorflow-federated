package(default_visibility = ["//tensorflow_federated:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "datasets",
    srcs = ["__init__.py"],
    deps = [
        ":dataset_utils",
        ":emnist",
        ":shakespeare",
        ":stackoverflow",
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

py_library(
    name = "emnist",
    srcs = ["emnist.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/simulation:from_tensor_slices_client_data",
        "//tensorflow_federated/python/simulation:hdf5_client_data",
        "//tensorflow_federated/python/simulation:transforming_client_data",
    ],
)

py_test(
    name = "emnist_test",
    size = "small",
    srcs = ["emnist_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":emnist"],
)

py_library(
    name = "shakespeare",
    srcs = ["shakespeare.py"],
    deps = ["//tensorflow_federated/python/simulation:hdf5_client_data"],
)

py_library(
    name = "stackoverflow",
    srcs = ["stackoverflow.py"],
    deps = ["//tensorflow_federated/python/simulation:hdf5_client_data"],
)
