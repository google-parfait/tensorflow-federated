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
    name = "cifar100_prediction",
    srcs = ["cifar100_prediction.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/simulation:client_data",
        "//tensorflow_federated/python/simulation/datasets:cifar100",
    ],
)

py_test(
    name = "cifar100_prediction_test",
    size = "medium",
    srcs = ["cifar100_prediction_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cifar100_prediction",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation:client_data",
    ],
)

py_library(
    name = "emnist_prediction",
    srcs = ["emnist_prediction.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/simulation:client_data",
        "//tensorflow_federated/python/simulation/datasets:emnist",
    ],
)

py_test(
    name = "emnist_prediction_test",
    size = "medium",
    srcs = ["emnist_prediction_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":emnist_prediction",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation:client_data",
    ],
)
