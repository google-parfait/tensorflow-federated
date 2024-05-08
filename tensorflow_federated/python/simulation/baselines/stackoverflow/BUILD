load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":stackoverflow_packages",
        "//tensorflow_federated/python/simulation:simulation_users",
    ],
)

package_group(
    name = "stackoverflow_packages",
    packages = ["//tensorflow_federated/python/simulation/baselines/stackoverflow/..."],
)

licenses(["notice"])

py_library(
    name = "stackoverflow",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/simulation/baselines:__pkg__"],
    deps = [
        ":constants",
        ":tag_prediction_tasks",
        ":word_prediction_tasks",
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

py_library(
    name = "tag_prediction_preprocessing",
    srcs = ["tag_prediction_preprocessing.py"],
    deps = [
        ":constants",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "tag_prediction_preprocessing_test",
    size = "medium",
    srcs = ["tag_prediction_preprocessing_test.py"],
    deps = [
        ":tag_prediction_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "tag_prediction_tasks",
    srcs = ["tag_prediction_tasks.py"],
    deps = [
        ":constants",
        ":tag_prediction_preprocessing",
        "//tensorflow_federated/python/learning/models:keras_utils",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/datasets:stackoverflow",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "tag_prediction_tasks_test",
    srcs = ["tag_prediction_tasks_test.py"],
    deps = [
        ":tag_prediction_tasks",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "word_prediction_models",
    srcs = ["word_prediction_models.py"],
    deps = ["@pypi_tensorflow//:pkg"],
)

py_test(
    name = "word_prediction_models_test",
    srcs = ["word_prediction_models_test.py"],
    deps = [
        ":word_prediction_models",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "word_prediction_preprocessing",
    srcs = ["word_prediction_preprocessing.py"],
    deps = [
        ":constants",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_attrs//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "word_prediction_preprocessing_test",
    size = "medium",
    srcs = ["word_prediction_preprocessing_test.py"],
    deps = [
        ":word_prediction_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "word_prediction_tasks",
    srcs = ["word_prediction_tasks.py"],
    deps = [
        ":constants",
        ":word_prediction_models",
        ":word_prediction_preprocessing",
        "//tensorflow_federated/python/learning/models:keras_utils",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:keras_metrics",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/datasets:stackoverflow",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "word_prediction_tasks_test",
    srcs = ["word_prediction_tasks_test.py"],
    deps = [
        ":word_prediction_tasks",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
