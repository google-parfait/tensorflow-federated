load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "p13n_utils",
    srcs = ["p13n_utils.py"],
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        "//tensorflow_federated",
    ],
)

py_test(
    name = "p13n_utils_test",
    srcs = ["p13n_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tf-nightly"),
        ":p13n_utils",
        "//tensorflow_federated",
    ],
)

py_binary(
    name = "emnist_p13n_main",
    srcs = ["emnist_p13n_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tf-nightly"),
        requirement("absl-py"),
        ":p13n_utils",
        "//tensorflow_federated",
    ],
)
