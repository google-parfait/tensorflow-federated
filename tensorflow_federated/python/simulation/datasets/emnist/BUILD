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
    deps = ["@six_archive//:six"],
)

py_library(
    name = "load_data",
    srcs = ["load_data.py"],
    deps = [
        ":synthetic",
        "//tensorflow_federated/python/simulation:from_tensor_slices_client_data",
        "//tensorflow_federated/python/simulation:hdf5_client_data",
        "//tensorflow_federated/python/simulation:transforming_client_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "load_data_test",
    size = "small",
    srcs = ["load_data_test.py"],
    deps = [
        ":load_data",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)
