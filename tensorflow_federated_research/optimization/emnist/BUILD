package(default_visibility = ["//tensorflow_federated_research"])

licenses(["notice"])

py_library(
    name = "centralized_emnist",
    srcs = ["centralized_emnist.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated_research/utils:centralized_training_loop",
        "//tensorflow_federated_research/utils/datasets:emnist_dataset",
        "//tensorflow_federated_research/utils/models:emnist_models",
    ],
)

py_library(
    name = "federated_emnist",
    srcs = ["federated_emnist.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated_research/utils:training_loop",
        "//tensorflow_federated_research/utils:training_utils",
        "//tensorflow_federated_research/utils/datasets:emnist_dataset",
        "//tensorflow_federated_research/utils/models:emnist_models",
    ],
)
