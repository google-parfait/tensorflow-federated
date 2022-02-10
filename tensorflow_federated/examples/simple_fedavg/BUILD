load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "emnist_fedavg_main",
    srcs = ["emnist_fedavg_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":simple_fedavg_tff",
        "//tensorflow_federated",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "simple_fedavg_tf",
    srcs = ["simple_fedavg_tf.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "simple_fedavg_tff",
    srcs = ["simple_fedavg_tff.py"],
    srcs_version = "PY3",
    deps = [
        ":simple_fedavg_tf",
        "//tensorflow_federated",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_cpu_gpu_test(
    name = "simple_fedavg_test",
    size = "medium",
    srcs = ["simple_fedavg_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [
        ":simple_fedavg_tf",
        ":simple_fedavg_tff",
        "//tensorflow_federated",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
