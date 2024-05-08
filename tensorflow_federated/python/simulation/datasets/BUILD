load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":datasets_packages",
        "//tensorflow_federated/python/simulation:simulation_users",
        "//tensorflow_federated/python/simulation/baselines:baselines_packages",
    ],
)

package_group(
    name = "datasets_packages",
    packages = ["//tensorflow_federated/python/simulation/datasets/..."],
)

licenses(["notice"])

py_library(
    name = "datasets",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/simulation:__pkg__"],
    deps = [
        "inaturalist",
        ":celeba",
        ":cifar100",
        ":client_data",
        ":dataset_utils",
        ":emnist",
        ":file_per_user_client_data",
        ":flair",
        ":from_tensor_slices_client_data",
        ":gldv2",
        ":shakespeare",
        ":sql_client_data",
        ":sql_client_data_utils",
        ":stackoverflow",
        ":transforming_client_data",
    ],
)

py_library(
    name = "celeba",
    srcs = ["celeba.py"],
    deps = [
        ":download",
        ":sql_client_data",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "celeba_test",
    size = "small",
    srcs = ["celeba_test.py"],
    deps = [
        ":celeba",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "cifar100",
    srcs = ["cifar100.py"],
    deps = [
        ":download",
        ":from_tensor_slices_client_data",
        ":sql_client_data",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "cifar100_test",
    size = "small",
    srcs = ["cifar100_test.py"],
    deps = [
        ":cifar100",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "dataset_utils",
    srcs = ["dataset_utils.py"],
    deps = ["@pypi_tensorflow//:pkg"],
)

py_test(
    name = "dataset_utils_test",
    size = "small",
    srcs = ["dataset_utils_test.py"],
    deps = [
        ":dataset_utils",
        ":from_tensor_slices_client_data",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "download",
    srcs = ["download.py"],
    deps = [
        "@pypi_tensorflow//:pkg",
        "@pypi_tqdm//:pkg",
    ],
)

py_test(
    name = "download_test",
    size = "small",
    srcs = ["download_test.py"],
    deps = [
        ":download",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "emnist",
    srcs = ["emnist.py"],
    deps = [
        ":download",
        ":from_tensor_slices_client_data",
        ":sql_client_data",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "emnist_test",
    timeout = "long",
    srcs = ["emnist_test.py"],
    deps = [
        ":emnist",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "flair",
    srcs = ["flair.py"],
    deps = [
        ":client_data",
        ":file_per_user_client_data",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "flair_test",
    srcs = ["flair_test.py"],
    deps = [
        ":flair",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "shakespeare",
    srcs = ["shakespeare.py"],
    deps = [
        ":client_data",
        ":download",
        ":from_tensor_slices_client_data",
        ":sql_client_data",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "shakespeare_test",
    size = "small",
    srcs = ["shakespeare_test.py"],
    deps = [
        ":shakespeare",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "stackoverflow",
    srcs = ["stackoverflow.py"],
    deps = [
        ":download",
        ":from_tensor_slices_client_data",
        ":sql_client_data",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "stackoverflow_test",
    size = "medium",
    srcs = ["stackoverflow_test.py"],
    deps = [
        ":stackoverflow",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "gldv2",
    srcs = ["gldv2.py"],
    deps = [
        ":client_data",
        ":from_tensor_slices_client_data",
        ":vision_datasets_utils",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "gldv2_test",
    size = "small",
    srcs = ["gldv2_test.py"],
    deps = [
        ":gldv2",
        ":vision_datasets_utils",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "inaturalist",
    srcs = ["inaturalist.py"],
    deps = [
        ":client_data",
        ":from_tensor_slices_client_data",
        ":vision_datasets_utils",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "inaturalist_test",
    size = "small",
    srcs = ["inaturalist_test.py"],
    deps = [
        ":inaturalist",
        ":vision_datasets_utils",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "vision_datasets_utils",
    srcs = ["vision_datasets_utils.py"],
    deps = [
        ":client_data",
        ":file_per_user_client_data",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "vision_datasets_utils_test",
    srcs = ["vision_datasets_utils_test.py"],
    deps = [
        ":vision_datasets_utils",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "client_data",
    srcs = ["client_data.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "client_data_test",
    size = "small",
    srcs = ["client_data_test.py"],
    deps = [
        ":client_data",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "file_per_user_client_data",
    srcs = ["file_per_user_client_data.py"],
    deps = [
        ":client_data",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "file_per_user_client_data_test",
    size = "small",
    srcs = ["file_per_user_client_data_test.py"],
    deps = [
        ":file_per_user_client_data",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "from_tensor_slices_client_data",
    srcs = ["from_tensor_slices_client_data.py"],
    deps = [
        ":client_data",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "from_tensor_slices_client_data_test",
    size = "small",
    srcs = ["from_tensor_slices_client_data_test.py"],
    deps = [
        ":from_tensor_slices_client_data",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "sql_client_data",
    srcs = ["sql_client_data.py"],
    deps = [
        ":client_data",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "sql_client_data_test",
    size = "small",
    srcs = ["sql_client_data_test.py"],
    deps = [
        ":sql_client_data",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "transforming_client_data",
    srcs = ["transforming_client_data.py"],
    deps = [
        ":client_data",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "transforming_client_data_test",
    srcs = ["transforming_client_data_test.py"],
    deps = [
        ":from_tensor_slices_client_data",
        ":transforming_client_data",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "sql_client_data_utils",
    srcs = ["sql_client_data_utils.py"],
    deps = [
        ":client_data",
        ":sql_client_data",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "sql_client_data_utils_test",
    srcs = ["sql_client_data_utils_test.py"],
    deps = [
        ":sql_client_data_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
