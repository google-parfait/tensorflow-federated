package(default_visibility = ["//tensorflow_federated_research"])

licenses(["notice"])

py_library(
    name = "centralized_emnist_ae",
    srcs = ["centralized_emnist_ae.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated_research/utils:centralized_training_loop",
        "//tensorflow_federated_research/utils/datasets:emnist_ae_dataset",
        "//tensorflow_federated_research/utils/models:emnist_ae_models",
    ],
)

py_library(
    name = "federated_emnist_ae",
    srcs = ["federated_emnist_ae.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated_research/utils:training_loop",
        "//tensorflow_federated_research/utils:training_utils",
        "//tensorflow_federated_research/utils/datasets:emnist_ae_dataset",
        "//tensorflow_federated_research/utils/models:emnist_ae_models",
    ],
)
