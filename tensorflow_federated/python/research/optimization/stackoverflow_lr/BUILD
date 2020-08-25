package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "centralized_stackoverflow_lr",
    srcs = ["centralized_stackoverflow_lr.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/research/utils:centralized_training_loop",
        "//tensorflow_federated/python/research/utils/datasets:stackoverflow_lr_dataset",
        "//tensorflow_federated/python/research/utils/models:stackoverflow_lr_models",
    ],
)

py_library(
    name = "federated_stackoverflow_lr",
    srcs = ["federated_stackoverflow_lr.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/utils:training_loop",
        "//tensorflow_federated/python/research/utils:training_utils",
        "//tensorflow_federated/python/research/utils/datasets:stackoverflow_lr_dataset",
        "//tensorflow_federated/python/research/utils/models:stackoverflow_lr_models",
    ],
)
