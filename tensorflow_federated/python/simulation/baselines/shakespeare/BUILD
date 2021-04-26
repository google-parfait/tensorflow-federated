load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/simulation:simulation_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/simulation/baselines/shakespeare/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "shakespeare",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/simulation/baselines:__pkg__"],
)

py_library(
    name = "shakespeare_models",
    srcs = ["shakespeare_models.py"],
    srcs_version = "PY3",
)

py_test(
    name = "shakespeare_models_test",
    srcs = ["shakespeare_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":shakespeare_models"],
)

py_library(
    name = "shakespeare_preprocessing",
    srcs = ["shakespeare_preprocessing.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_test(
    name = "shakespeare_preprocessing_test",
    size = "medium",
    srcs = ["shakespeare_preprocessing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":shakespeare_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
    ],
)
