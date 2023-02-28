load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

py_library(
    name = "computations",
    srcs = ["computations.py"],
    deps = ["//tensorflow_federated"],
)

py_library(
    name = "program_logic",
    srcs = ["program_logic.py"],
    deps = ["//tensorflow_federated"],
)

py_binary(
    name = "program",
    srcs = ["program.py"],
    deps = [
        ":computations",
        ":program_logic",
        "//tensorflow_federated",
    ],
)

py_test(
    name = "program_logic_test",
    srcs = ["program_logic_test.py"],
    deps = [
        ":program_logic",
        "//tensorflow_federated",
    ],
)
