package(default_visibility = ["//tensorflow_federated/python/simulation:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "simulation",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":client_data",
        ":file_per_user_client_data",
        ":hdf5_client_data",
        "//tensorflow_federated/python/simulation/datasets",
    ],
)

py_library(
    name = "client_data",
    srcs = ["client_data.py"],
    deps = ["@six_archive//:six"],
)

py_library(
    name = "file_per_user_client_data",
    srcs = ["file_per_user_client_data.py"],
    deps = [
        ":client_data",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "file_per_user_client_data_test",
    size = "small",
    srcs = ["file_per_user_client_data_test.py"],
    deps = [
        ":file_per_user_client_data",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "hdf5_client_data",
    srcs = ["hdf5_client_data.py"],
    deps = [
        ":client_data",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "hdf5_client_data_test",
    size = "small",
    srcs = ["hdf5_client_data_test.py"],
    deps = [
        ":hdf5_client_data",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)
