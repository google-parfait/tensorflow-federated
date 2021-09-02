load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":baselines_packages",
    "//tensorflow_federated/python/simulation:simulation_visibility",
])

package_group(
    name = "baselines_packages",
    packages = ["//tensorflow_federated/python/simulation/baselines/..."],
)

licenses(["notice"])

py_library(
    name = "baselines",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/simulation:__pkg__"],
    deps = [
        ":baseline_task",
        ":client_spec",
        ":task_data",
        "//tensorflow_federated/python/simulation/baselines/cifar100",
        "//tensorflow_federated/python/simulation/baselines/emnist",
        "//tensorflow_federated/python/simulation/baselines/shakespeare",
        "//tensorflow_federated/python/simulation/baselines/stackoverflow",
    ],
)

py_library(
    name = "baseline_task",
    srcs = ["baseline_task.py"],
    srcs_version = "PY3",
    deps = [
        requirement("attrs"),
        requirement("tf-nightly"),
        ":task_data",
        "//tensorflow_federated/python/learning:model",
    ],
)

py_test(
    name = "baseline_task_test",
    srcs = ["baseline_task_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("attrs"),
        requirement("tf-nightly"),
        ":baseline_task",
        ":task_data",
        "//tensorflow_federated/python/learning:keras_utils",
        "//tensorflow_federated/python/simulation/datasets:client_data",
    ],
)

py_library(
    name = "client_spec",
    srcs = ["client_spec.py"],
    srcs_version = "PY3",
    deps = [requirement("attrs")],
)

py_library(
    name = "keras_metrics",
    srcs = ["keras_metrics.py"],
    srcs_version = "PY3",
    deps = [requirement("tf-nightly")],
)

py_test(
    name = "keras_metrics_test",
    srcs = ["keras_metrics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        ":keras_metrics",
    ],
)

py_library(
    name = "task_data",
    srcs = ["task_data.py"],
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tf-nightly"),
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/simulation/datasets:client_data",
    ],
)

py_test(
    name = "task_data_test",
    srcs = ["task_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        requirement("absl-py"),
        ":task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
    ],
)
