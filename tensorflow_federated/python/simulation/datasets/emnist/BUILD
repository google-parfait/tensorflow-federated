package(default_visibility = ["//tensorflow_federated:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "emnist",
    srcs = ["__init__.py"],
    deps = [":load_data"],
)

py_library(
    name = "load_data",
    srcs = ["load_data.py"],
    deps = [
        "//tensorflow_federated/python/simulation:hdf5_client_data",
        "//tensorflow_federated/python/simulation:transforming_client_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
