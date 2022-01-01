load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "p13n_utils",
    srcs = ["p13n_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras",
    ],
)

py_test(
    name = "p13n_utils_test",
    srcs = ["p13n_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":p13n_utils",
        "//tensorflow_federated",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras",
    ],
)

py_binary(
    name = "emnist_p13n_main",
    srcs = ["emnist_p13n_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":p13n_utils",
        "//tensorflow_federated",
        "@absl_py//absl:app",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras",
    ],
)
