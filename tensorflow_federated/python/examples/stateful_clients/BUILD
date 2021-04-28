load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "emnist_fedavg_main",
    srcs = ["emnist_fedavg_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":stateful_fedavg_tf",
        ":stateful_fedavg_tff",
        "//tensorflow_federated",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "stateful_fedavg_tf",
    srcs = ["stateful_fedavg_tf.py"],
    srcs_version = "PY3",
    deps = [
        requirement("attrs"),
        "//tensorflow_federated",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "stateful_fedavg_tff",
    srcs = ["stateful_fedavg_tff.py"],
    srcs_version = "PY3",
    deps = [
        ":stateful_fedavg_tf",
        "//tensorflow_federated",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "stateful_fedavg_test",
    size = "medium",
    srcs = ["stateful_fedavg_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":stateful_fedavg_tf",
        ":stateful_fedavg_tff",
        "//tensorflow_federated",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
