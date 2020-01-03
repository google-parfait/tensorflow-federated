package(default_visibility = ["//tensorflow_federated:__subpackages__"])

licenses(["notice"])

py_library(
    name = "cifar100",
    srcs = ["cifar100.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/simulation:hdf5_client_data"],
)

py_test(
    name = "cifar100_test",
    size = "medium",
    srcs = ["cifar100_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":cifar100"],
)

py_library(
    name = "datasets",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":cifar100",
        ":dataset_utils",
        ":emnist",
        ":shakespeare",
        ":stackoverflow",
    ],
)

py_library(
    name = "dataset_utils",
    srcs = ["dataset_utils.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/simulation:hdf5_client_data"],
)

py_library(
    name = "stackoverflow",
    srcs = ["stackoverflow.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/simulation:hdf5_client_data"],
)
