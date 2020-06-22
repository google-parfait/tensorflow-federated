package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "compression_process_adapter",
    srcs = ["compression_process_adapter.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/utils:adapters",
    ],
)

py_binary(
    name = "run_experiment",
    srcs = ["run_experiment.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":compression_process_adapter",
        ":sparsity",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/optimization/emnist:models",
        "//tensorflow_federated/python/research/utils:training_loop",
        "//tensorflow_federated/python/research/utils:training_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
        "//tensorflow_federated/python/research/utils/datasets:emnist_dataset",
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
