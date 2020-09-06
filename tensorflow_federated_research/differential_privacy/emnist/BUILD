licenses(["notice"])

py_binary(
    name = "run_federated",
    srcs = ["run_federated.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated_research/differential_privacy:dp_utils",
        "//tensorflow_federated_research/optimization/shared:optimizer_utils",
        "//tensorflow_federated_research/utils:training_loop",
        "//tensorflow_federated_research/utils:training_utils",
        "//tensorflow_federated_research/utils:utils_impl",
        "//tensorflow_federated_research/utils/datasets:emnist_dataset",
        "//tensorflow_federated_research/utils/models:emnist_models",
    ],
)
