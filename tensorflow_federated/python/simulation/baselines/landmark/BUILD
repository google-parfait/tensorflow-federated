load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":landmark_packages",
        "//tensorflow_federated/python/simulation:simulation_users",
    ],
)

package_group(
    name = "landmark_packages",
    packages = ["//tensorflow_federated/python/simulation/baselines/landmark/..."],
)

licenses(["notice"])

py_library(
    name = "landmark",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/simulation/baselines:__pkg__"],
    deps = [
        ":landmark_tasks",
    ],
)

py_library(
    name = "landmark_tasks",
    srcs = ["landmark_tasks.py"],
    deps = [
        ":landmark_preprocessing",
        "//tensorflow_federated/python/learning/models:keras_utils",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/datasets:gldv2",
        "//tensorflow_federated/python/simulation/models:mobilenet_v2",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "landmark_tasks_test",
    srcs = ["landmark_tasks_test.py"],
    deps = [
        ":landmark_tasks",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "landmark_preprocessing",
    srcs = ["landmark_preprocessing.py"],
    deps = [
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "landmark_preprocessing_test",
    srcs = ["landmark_preprocessing_test.py"],
    deps = [
        ":landmark_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
