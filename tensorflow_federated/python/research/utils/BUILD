package(default_visibility = ["//tensorflow_federated/python/research:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":utils"],
)
