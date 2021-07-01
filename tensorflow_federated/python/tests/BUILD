load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package_group(
    name = "default_visibility",
    packages = ["//tensorflow_federated/python/tests/..."],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_test(
    name = "ast_generation_test",
    size = "small",
    srcs = ["ast_generation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "backend_test",
    timeout = "long",
    srcs = ["backend_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":remote_runtime_test_utils",
        ":temperature_sensor_example",
        ":test_contexts",
        "//tensorflow_federated",
    ],
)

py_cpu_gpu_test(
    name = "backend_accelerators_test",
    srcs = ["backend_accelerators_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "composite_tensor_e2e_test",
    size = "small",
    srcs = ["composite_tensor_e2e_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "map_reduce_form_test",
    size = "medium",
    srcs = ["map_reduce_form_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "perf_regression_test",
    timeout = "moderate",
    srcs = ["perf_regression_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "remote_runtime_integration_test",
    timeout = "long",
    srcs = ["remote_runtime_integration_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":remote_runtime_test_utils",
        "//tensorflow_federated",
    ],
)

py_library(
    name = "remote_runtime_test_utils",
    testonly = True,
    srcs = ["remote_runtime_test_utils.py"],
    data = [":test_aggregator"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "simulation_test",
    size = "medium",
    srcs = ["simulation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_library(
    name = "temperature_sensor_example",
    srcs = ["temperature_sensor_example.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_binary(
    name = "test_aggregator",
    srcs = ["test_aggregator.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_library(
    name = "test_contexts",
    testonly = True,
    srcs = ["test_contexts.py"],
    srcs_version = "PY3",
    deps = [
        ":remote_runtime_test_utils",
        "//tensorflow_federated",
    ],
)
