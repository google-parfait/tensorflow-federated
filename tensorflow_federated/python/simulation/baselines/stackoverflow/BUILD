load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/simulation:simulation_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/simulation/baselines/stackoverflow/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "stackoverflow",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
)

py_library(
    name = "tag_prediction_preprocessing",
    srcs = ["tag_prediction_preprocessing.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
    ],
)

py_test(
    name = "tag_prediction_preprocessing_test",
    size = "medium",
    srcs = ["tag_prediction_preprocessing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tag_prediction_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
    ],
)

py_library(
    name = "tag_prediction_tasks",
    srcs = ["tag_prediction_tasks.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":tag_prediction_preprocessing",
        "//tensorflow_federated/python/learning:keras_utils",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/datasets:stackoverflow",
    ],
)

py_test(
    name = "tag_prediction_tasks_test",
    srcs = ["tag_prediction_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tag_prediction_tasks",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
    ],
)

py_library(
    name = "word_prediction_models",
    srcs = ["word_prediction_models.py"],
    srcs_version = "PY3",
)

py_test(
    name = "word_prediction_models_test",
    srcs = ["word_prediction_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":word_prediction_models"],
)

py_library(
    name = "word_prediction_preprocessing",
    srcs = ["word_prediction_preprocessing.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
    ],
)

py_test(
    name = "word_prediction_preprocessing_test",
    size = "medium",
    srcs = ["word_prediction_preprocessing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":word_prediction_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
    ],
)

py_library(
    name = "word_prediction_tasks",
    srcs = ["word_prediction_tasks.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":word_prediction_models",
        ":word_prediction_preprocessing",
        "//tensorflow_federated/python/learning:keras_utils",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:keras_metrics",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/datasets:stackoverflow",
    ],
)

py_test(
    name = "word_prediction_tasks_test",
    srcs = ["word_prediction_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":word_prediction_tasks",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
    ],
)
