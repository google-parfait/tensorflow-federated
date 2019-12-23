package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset",
        "//tensorflow_federated/python/common_libs:test",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    srcs_version = "PY3",
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":models"],
)

py_binary(
    name = "run_centralized",
    srcs = ["run_centralized.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset",
        ":models",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)
