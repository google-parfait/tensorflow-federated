load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":emnist_packages",
    "//tensorflow_federated/python/simulation:simulation_visibility",
])

package_group(
    name = "emnist_packages",
    packages = ["//tensorflow_federated/python/simulation/baselines/emnist/..."],
)

licenses(["notice"])

py_library(
    name = "emnist",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/simulation/baselines:__pkg__"],
    deps = [
        ":autoencoder_tasks",
        ":char_recognition_tasks",
    ],
)

py_library(
    name = "autoencoder_tasks",
    srcs = ["autoencoder_tasks.py"],
    srcs_version = "PY3",
    deps = [
        ":emnist_models",
        ":emnist_preprocessing",
        "//tensorflow_federated/python/learning:keras_utils",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/datasets:emnist",
    ],
)

py_test(
    name = "autoencoder_tasks_test",
    srcs = ["autoencoder_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":autoencoder_tasks",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "char_recognition_tasks",
    srcs = ["char_recognition_tasks.py"],
    srcs_version = "PY3",
    deps = [
        ":emnist_models",
        ":emnist_preprocessing",
        "//tensorflow_federated/python/learning:keras_utils",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/datasets:emnist",
    ],
)

py_test(
    name = "char_recognition_tasks_test",
    srcs = ["char_recognition_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":char_recognition_tasks",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:baseline_task",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "emnist_models",
    srcs = ["emnist_models.py"],
    srcs_version = "PY3",
)

py_test(
    name = "emnist_models_test",
    srcs = ["emnist_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":emnist_models"],
)

py_library(
    name = "emnist_preprocessing",
    srcs = ["emnist_preprocessing.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/simulation/baselines:client_spec"],
)

py_test(
    name = "emnist_preprocessing_test",
    size = "medium",
    srcs = ["emnist_preprocessing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":emnist_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "@absl_py//absl/testing:parameterized",
    ],
)
