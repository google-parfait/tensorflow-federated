load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "test",
    srcs = ["test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("grpcio"),
        "//tensorflow_federated",
    ],
)
