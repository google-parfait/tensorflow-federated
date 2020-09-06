package(default_visibility = ["//tensorflow_federated_research"])

licenses(["notice"])

py_library(
    name = "centralized_shakespeare",
    srcs = ["centralized_shakespeare.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated_research/optimization/shared:keras_metrics",
        "//tensorflow_federated_research/utils:centralized_training_loop",
        "//tensorflow_federated_research/utils/datasets:shakespeare_dataset",
        "//tensorflow_federated_research/utils/models:shakespeare_models",
    ],
)

py_library(
    name = "federated_shakespeare",
    srcs = ["federated_shakespeare.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated_research/optimization/shared:keras_metrics",
        "//tensorflow_federated_research/utils:training_loop",
        "//tensorflow_federated_research/utils:training_utils",
        "//tensorflow_federated_research/utils/datasets:shakespeare_dataset",
        "//tensorflow_federated_research/utils/models:shakespeare_models",
    ],
)
