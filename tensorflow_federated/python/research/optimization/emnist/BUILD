package(default_visibility = ["//tensorflow_federated/python/research:__subpackages__"])

licenses(["notice"])

py_binary(
    name = "compute_yogi_init",
    srcs = ["compute_yogi_init.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset",
        ":models",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/optimization/shared:optimizer_utils",
    ],
)

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
    tags = [
        "manual",
        "nopresubmit",
    ],
    deps = [":dataset"],
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
        "//tensorflow_federated/python/research/optimization/shared:keras_callbacks",
        "//tensorflow_federated/python/research/optimization/shared:optimizer_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)

py_binary(
    name = "run_federated",
    srcs = ["run_federated.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset",
        ":models",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/optimization/shared:iterative_process_builder",
        "//tensorflow_federated/python/research/utils:training_loop",
        "//tensorflow_federated/python/research/utils:training_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)
