load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "test",
    srcs = ["test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
