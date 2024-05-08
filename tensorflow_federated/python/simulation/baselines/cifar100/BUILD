load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":cifar100_packages",
        "//tensorflow_federated/python/simulation:simulation_users",
    ],
)

package_group(
    name = "cifar100_packages",
    packages = ["//tensorflow_federated/python/simulation/baselines/cifar100/..."],
)

licenses(["notice"])

py_library(
    name = "cifar100",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/simulation/baselines:__pkg__"],
    deps = [":image_classification_tasks"],
)

py_library(
    name = "image_classification_preprocessing",
    srcs = ["image_classification_preprocessing.py"],
    deps = [
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "image_classification_preprocessing_test",
    size = "medium",
    srcs = ["image_classification_preprocessing_test.py"],
    deps = [
        ":image_classification_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "image_classification_tasks",
    srcs = ["image_classification_tasks.py"],
    deps = [
        ":image_classification_preprocessing",
        "//tensorflow_federated/python/learning/models:keras_utils",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:cifar100",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/models:resnet_models",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "image_classification_tasks_test",
    srcs = ["image_classification_tasks_test.py"],
    deps = [
        ":image_classification_tasks",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/models:resnet_models",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
