package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "mnist",
    srcs = ["__init__.py"],
    deps = [":models"],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    deps = ["//tensorflow_federated/python"],
)

py_test(
    name = "models_test",
    size = "large",
    srcs = ["models_test.py"],
    strict_deps = True,
    deps = [
        ":models",
        "//tensorflow_federated/python",
    ],
)
