package(default_visibility = ["//tensorflow_federated:__subpackages__"])

licenses(["notice"])

py_library(
    name = "simulation",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":client_data",
        ":file_per_user_client_data",
        ":from_tensor_slices_client_data",
        ":hdf5_client_data",
        ":server_utils",
        ":transforming_client_data",
        "//tensorflow_federated/python/simulation/datasets",
        "//tensorflow_federated/python/simulation/models",
    ],
)

py_library(
    name = "client_data",
    srcs = ["client_data.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/tensorflow_libs:version_check",
    ],
)

py_test(
    name = "client_data_test",
    size = "small",
    srcs = ["client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":client_data",
        "//tensorflow_federated/python/tensorflow_libs:version_check",
    ],
)

py_library(
    name = "file_per_user_client_data",
    srcs = ["file_per_user_client_data.py"],
    srcs_version = "PY3",
    deps = [
        ":client_data",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_test(
    name = "file_per_user_client_data_test",
    size = "small",
    srcs = ["file_per_user_client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":file_per_user_client_data"],
)

py_library(
    name = "from_tensor_slices_client_data",
    srcs = ["from_tensor_slices_client_data.py"],
    srcs_version = "PY3",
    deps = [
        ":client_data",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "from_tensor_slices_client_data_test",
    size = "small",
    srcs = ["from_tensor_slices_client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":from_tensor_slices_client_data"],
)

py_library(
    name = "hdf5_client_data",
    srcs = ["hdf5_client_data.py"],
    srcs_version = "PY3",
    deps = [
        ":client_data",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_test(
    name = "hdf5_client_data_test",
    size = "small",
    srcs = ["hdf5_client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":hdf5_client_data"],
)

py_library(
    name = "server_utils",
    srcs = ["server_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core",
    ],
)

py_test(
    name = "server_utils_test",
    size = "small",
    srcs = ["server_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":server_utils",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core",
    ],
)

py_library(
    name = "transforming_client_data",
    srcs = ["transforming_client_data.py"],
    srcs_version = "PY3",
    deps = [
        ":client_data",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "transforming_client_data_test",
    size = "small",
    srcs = ["transforming_client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hdf5_client_data",
        ":transforming_client_data",
    ],
)
