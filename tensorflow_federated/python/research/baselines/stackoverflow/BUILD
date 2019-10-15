package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "models",
    srcs = ["models.py"],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":models"],
)

py_binary(
    name = "non_federated_stackoverflow",
    srcs = ["non_federated_stackoverflow.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":models",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)
