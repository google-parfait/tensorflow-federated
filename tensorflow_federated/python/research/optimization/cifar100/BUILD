package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "compute_yogi_init",
    srcs = ["compute_yogi_init.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/optimization/shared:optimizer_utils",
        "//tensorflow_federated/python/research/optimization/shared:resnet_models",
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
    deps = ["//tensorflow_federated/python/research/optimization/cifar100:dataset"],
)

py_binary(
    name = "run_centralized",
    srcs = ["run_centralized.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset",
        "//tensorflow_federated/python/research/optimization/shared:keras_callbacks",
        "//tensorflow_federated/python/research/optimization/shared:optimizer_utils",
        "//tensorflow_federated/python/research/optimization/shared:resnet_models",
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
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/optimization/shared:iterative_process_builder",
        "//tensorflow_federated/python/research/optimization/shared:resnet_models",
        "//tensorflow_federated/python/research/optimization/shared:training_utils",
        "//tensorflow_federated/python/research/utils:training_loop",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)
