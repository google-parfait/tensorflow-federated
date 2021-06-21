load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/simulation:simulation_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/simulation/baselines/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "baselines",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/simulation:__pkg__"],
)

py_library(
    name = "baseline_task",
    srcs = ["baseline_task.py"],
    srcs_version = "PY3",
    deps = [
        requirement("attrs"),
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
        ":baseline_task",
        ":task_data",
        "//tensorflow_federated/python/learning:keras_utils",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
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
    deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
)

py_test(
    name = "keras_metrics_test",
    srcs = ["keras_metrics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_metrics",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "task_data",
    srcs = ["task_data.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "task_data_test",
    srcs = ["task_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
