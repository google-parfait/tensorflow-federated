package(default_visibility = ["//tensorflow_federated:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "emnist",
    srcs = ["__init__.py"],
    deps = [":load_data"],
)

py_library(
    name = "synthetic",
    srcs = ["synthetic.py"],
)

py_library(
    name = "load_data",
    srcs = ["load_data.py"],
    deps = [
        ":synthetic",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/simulation:from_tensor_slices_client_data",
        "//tensorflow_federated/python/simulation:hdf5_client_data",
        "//tensorflow_federated/python/simulation:transforming_client_data",
    ],
)

py_test(
    name = "load_data_test",
    size = "small",
    srcs = ["load_data_test.py"],
    deps = [
        ":load_data",
        "//third_party/py/six",
    ],
)
