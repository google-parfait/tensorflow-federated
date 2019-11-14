package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "metrics_hook",
    srcs = ["metrics_hook.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)

py_binary(
    name = "run_experiment",
    srcs = ["run_experiment.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":metrics_hook",
        ":sparsity",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/utils:training_loops",
        "//tensorflow_federated/python/research/utils:utils_impl",
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
