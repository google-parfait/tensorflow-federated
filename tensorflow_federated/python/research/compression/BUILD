package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_binary(
    name = "run_experiment",
    srcs = ["run_experiment.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sparsity",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/utils:training_loop",
        "//tensorflow_federated/python/research/utils:training_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
        "//tensorflow_federated/python/research/utils/datasets:emnist_dataset",
        "//tensorflow_federated/python/research/utils/models:emnist_models",
    ],
)

py_library(
    name = "sparsity",
    srcs = ["sparsity.py"],
    srcs_version = "PY3",
)

py_test(
    name = "sparsity_test",
    srcs = ["sparsity_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":sparsity"],
)
