package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "cifar100_dataset",
    srcs = ["cifar100_dataset.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "cifar100_dataset_test",
    srcs = ["cifar100_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nopresubmit",
    ],
    deps = [":cifar100_dataset"],
)

py_library(
    name = "emnist_dataset",
    srcs = ["emnist_dataset.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "emnist_dataset_test",
    srcs = ["emnist_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nopresubmit",
    ],
    deps = [":emnist_dataset"],
)

py_library(
    name = "emnist_ae_dataset",
    srcs = ["emnist_ae_dataset.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "emnist_ae_dataset_test",
    srcs = ["emnist_ae_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nopresubmit",
    ],
    deps = [":emnist_ae_dataset"],
)

py_library(
    name = "shakespeare_dataset",
    srcs = ["shakespeare_dataset.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "shakespeare_dataset_test",
    srcs = ["shakespeare_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nopresubmit",
    ],
    deps = [":shakespeare_dataset"],
)

py_library(
    name = "stackoverflow_dataset",
    srcs = ["stackoverflow_dataset.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "stackoverflow_dataset_test",
    size = "medium",
    srcs = ["stackoverflow_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":stackoverflow_dataset",
        "//tensorflow_federated",
        "//tensorflow_federated/python/common_libs:test",
    ],
)

py_library(
    name = "stackoverflow_lr_dataset",
    srcs = ["stackoverflow_lr_dataset.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "stackoverflow_lr_dataset_test",
    size = "large",
    srcs = ["stackoverflow_lr_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":stackoverflow_lr_dataset",
        "//tensorflow_federated/python/common_libs:test",
    ],
)
