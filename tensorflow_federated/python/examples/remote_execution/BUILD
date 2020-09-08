load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "remote_executor_example",
    srcs = ["remote_executor_example.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)
