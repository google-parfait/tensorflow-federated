load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":baselines_packages",
        "//tensorflow_federated/python/simulation:simulation_users",
    ],
)

package_group(
    name = "baselines_packages",
    packages = ["//tensorflow_federated/python/simulation/baselines/..."],
)

licenses(["notice"])

py_library(
    name = "baselines",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/simulation:__pkg__"],
    deps = [
        ":baseline_task",
        ":client_spec",
        ":task_data",
        "//tensorflow_federated/python/simulation/baselines/cifar100",
        "//tensorflow_federated/python/simulation/baselines/emnist",
        "//tensorflow_federated/python/simulation/baselines/landmark",
        "//tensorflow_federated/python/simulation/baselines/shakespeare",
        "//tensorflow_federated/python/simulation/baselines/stackoverflow",
    ],
)

py_library(
    name = "baseline_task",
    srcs = ["baseline_task.py"],
    deps = [
        ":task_data",
        "//tensorflow_federated/python/learning/models:variable",
        "@pypi_attrs//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "baseline_task_test",
    srcs = ["baseline_task_test.py"],
    deps = [
        ":baseline_task",
        ":task_data",
        "//tensorflow_federated/python/learning/models:keras_utils",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "@pypi_attrs//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "client_spec",
    srcs = ["client_spec.py"],
    deps = ["@pypi_attrs//:pkg"],
)

py_library(
    name = "keras_metrics",
    srcs = ["keras_metrics.py"],
    deps = ["@pypi_tensorflow//:pkg"],
)

py_test(
    name = "keras_metrics_test",
    srcs = ["keras_metrics_test.py"],
    deps = [
        ":keras_metrics",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "task_data",
    srcs = ["task_data.py"],
    deps = [
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "task_data_test",
    srcs = ["task_data_test.py"],
    deps = [
        ":task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
