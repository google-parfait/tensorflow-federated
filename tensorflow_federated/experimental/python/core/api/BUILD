load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_test(
    name = "noop_test",
    size = "small",
    srcs = ["noop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "nopresubmit",
    ],
)
