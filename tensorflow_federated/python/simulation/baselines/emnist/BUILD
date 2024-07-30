load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":emnist_packages",
        "//tensorflow_federated/python/simulation:simulation_users",
    ],
)

package_group(
    name = "emnist_packages",
    packages = ["//tensorflow_federated/python/simulation/baselines/emnist/..."],
)

licenses(["notice"])

py_library(
    name = "emnist",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/simulation/baselines:__pkg__"],
    deps = [
        ":autoencoder_tasks",
        ":char_recognition_tasks",
    ],
)

py_library(
    name = "autoencoder_tasks",
    srcs = ["autoencoder_tasks.py"],
    deps = [
        ":emnist_models",
        ":emnist_preprocessing",
        "//tensorflow_federated/python/learning/models:keras_utils",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/datasets:emnist",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "autoencoder_tasks_test",
    srcs = ["autoencoder_tasks_test.py"],
    deps = [
        ":autoencoder_tasks",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "char_recognition_tasks",
    srcs = ["char_recognition_tasks.py"],
    deps = [
        ":emnist_models",
        ":emnist_preprocessing",
        "//tensorflow_federated/python/learning/models:keras_utils",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/datasets:emnist",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "char_recognition_tasks_test",
    srcs = ["char_recognition_tasks_test.py"],
    deps = [
        ":char_recognition_tasks",
        ":emnist_models",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "emnist_models",
    srcs = ["emnist_models.py"],
    deps = ["@pypi_tensorflow//:pkg"],
)

py_test(
    name = "emnist_models_test",
    srcs = ["emnist_models_test.py"],
    deps = [
        ":emnist_models",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "emnist_preprocessing",
    srcs = ["emnist_preprocessing.py"],
    deps = [
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "emnist_preprocessing_test",
    size = "medium",
    srcs = ["emnist_preprocessing_test.py"],
    deps = [
        ":emnist_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
