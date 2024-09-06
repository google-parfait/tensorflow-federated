load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":shakespeare_packages",
        "//tensorflow_federated/python/simulation:simulation_users",
    ],
)

package_group(
    name = "shakespeare_packages",
    packages = ["//tensorflow_federated/python/simulation/baselines/shakespeare/..."],
)

licenses(["notice"])

py_library(
    name = "shakespeare",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/simulation/baselines:__pkg__"],
    deps = [":char_prediction_tasks"],
)

py_library(
    name = "char_prediction_models",
    srcs = ["char_prediction_models.py"],
    deps = ["//third_party/tensorflow"],
)

py_test(
    name = "char_prediction_models_test",
    srcs = ["char_prediction_models_test.py"],
    deps = [
        ":char_prediction_models",
        "//third_party/tensorflow",
    ],
)

py_library(
    name = "char_prediction_preprocessing",
    srcs = ["char_prediction_preprocessing.py"],
    deps = [
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//third_party/tensorflow",
    ],
)

py_test(
    name = "char_prediction_preprocessing_test",
    size = "medium",
    srcs = ["char_prediction_preprocessing_test.py"],
    deps = [
        ":char_prediction_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//third_party/tensorflow",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "char_prediction_tasks",
    srcs = ["char_prediction_tasks.py"],
    deps = [
        ":char_prediction_models",
        ":char_prediction_preprocessing",
        "//tensorflow_federated/python/learning/models:keras_utils",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:keras_metrics",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/datasets:shakespeare",
        "//third_party/tensorflow",
    ],
)

py_test(
    name = "char_prediction_tasks_test",
    srcs = ["char_prediction_tasks_test.py"],
    deps = [
        ":char_prediction_tasks",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//third_party/tensorflow",
        "@pypi_absl_py//:pkg",
    ],
)
