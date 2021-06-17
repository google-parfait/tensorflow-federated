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
    name = "emnist",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/simulation/baselines:__pkg__"],
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
    deps = [
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
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
    ],
)

py_library(
    name = "emnist_tasks",
    srcs = ["emnist_tasks.py"],
    srcs_version = "PY3",
    deps = [
        ":emnist_models",
        ":emnist_preprocessing",
        "//tensorflow_federated/python/learning:keras_utils",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:task_data",
        "//tensorflow_federated/python/simulation/datasets:emnist",
    ],
)

py_test(
    name = "emnist_tasks_test",
    srcs = ["emnist_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":emnist_tasks",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
        "//tensorflow_federated/python/simulation/baselines:task_data",
    ],
)
