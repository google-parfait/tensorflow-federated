load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_test(
    name = "backend_test",
    timeout = "long",
    srcs = ["backend_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":remote_runtime_test_utils",
        ":temperature_sensor_example",
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
    name = "canonical_form_test",
    size = "large",
    srcs = ["canonical_form_test.py"],
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
    timeout = "moderate",
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
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_library(
    name = "temperature_sensor_example",
    srcs = ["temperature_sensor_example.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)
