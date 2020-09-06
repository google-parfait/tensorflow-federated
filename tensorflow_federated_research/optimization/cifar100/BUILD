package(default_visibility = ["//tensorflow_federated_research"])

licenses(["notice"])

py_library(
    name = "centralized_cifar100",
    srcs = ["centralized_cifar100.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated_research/utils:centralized_training_loop",
        "//tensorflow_federated_research/utils/datasets:cifar100_dataset",
        "//tensorflow_federated_research/utils/models:resnet_models",
    ],
)

py_library(
    name = "federated_cifar100",
    srcs = ["federated_cifar100.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated_research/utils:training_loop",
        "//tensorflow_federated_research/utils:training_utils",
        "//tensorflow_federated_research/utils/datasets:cifar100_dataset",
        "//tensorflow_federated_research/utils/models:resnet_models",
    ],
)
