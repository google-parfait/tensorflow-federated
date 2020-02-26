package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "p13n_utils",
    srcs = ["p13n_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "p13n_utils_test",
    srcs = ["p13n_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":p13n_utils",
        "//tensorflow_federated",
    ],
)
