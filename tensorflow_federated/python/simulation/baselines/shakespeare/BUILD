load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":shakespeare_packages",
    "//tensorflow_federated/python/simulation:simulation_visibility",
])

package_group(
    name = "shakespeare_packages",
    packages = ["//tensorflow_federated/python/simulation/baselines/shakespeare/..."],
)

licenses(["notice"])

py_library(
    name = "shakespeare",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/simulation/baselines:__pkg__"],
    deps = [":char_prediction_tasks"],
)

py_library(
    name = "char_prediction_models",
    srcs = ["char_prediction_models.py"],
    srcs_version = "PY3",
    deps = [requirement("tf-nightly")],
)

py_test(
    name = "char_prediction_models_test",
    srcs = ["char_prediction_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        ":char_prediction_models",
    ],
)

py_library(
    name = "char_prediction_preprocessing",
    srcs = ["char_prediction_preprocessing.py"],
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        "//tensorflow_federated/python/simulation/baselines:client_spec",
    ],
)

py_test(
    name = "char_prediction_preprocessing_test",
    size = "medium",
    srcs = ["char_prediction_preprocessing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        requirement("absl-py"),
        ":char_prediction_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
    ],
)

py_library(
    name = "char_prediction_tasks",
    srcs = ["char_prediction_tasks.py"],
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        ":char_prediction_models",
        ":char_prediction_preprocessing",
        "//tensorflow_federated/python/learning:keras_utils",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:keras_metrics",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/datasets:shakespeare",
    ],
)

py_test(
    name = "char_prediction_tasks_test",
    srcs = ["char_prediction_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        requirement("absl-py"),
        ":char_prediction_tasks",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
    ],
)
