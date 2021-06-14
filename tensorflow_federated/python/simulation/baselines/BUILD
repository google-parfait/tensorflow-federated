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
