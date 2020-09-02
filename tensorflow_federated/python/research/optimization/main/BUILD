package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_binary(
    name = "centralized_trainer",
    srcs = ["centralized_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/research/optimization/cifar100:centralized_cifar100",
        "//tensorflow_federated/python/research/optimization/emnist:centralized_emnist",
        "//tensorflow_federated/python/research/optimization/emnist_ae:centralized_emnist_ae",
        "//tensorflow_federated/python/research/optimization/shakespeare:centralized_shakespeare",
        "//tensorflow_federated/python/research/optimization/shared:optimizer_utils",
        "//tensorflow_federated/python/research/optimization/stackoverflow:centralized_stackoverflow",
        "//tensorflow_federated/python/research/optimization/stackoverflow_lr:centralized_stackoverflow_lr",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)

py_test(
    name = "centralized_tasks_test",
    size = "large",
    srcs = ["centralized_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nopresubmit",
    ],
    deps = [
        "//tensorflow_federated/python/research/optimization/cifar100:centralized_cifar100",
        "//tensorflow_federated/python/research/optimization/emnist:centralized_emnist",
        "//tensorflow_federated/python/research/optimization/emnist_ae:centralized_emnist_ae",
        "//tensorflow_federated/python/research/optimization/shakespeare:centralized_shakespeare",
        "//tensorflow_federated/python/research/optimization/stackoverflow:centralized_stackoverflow",
        "//tensorflow_federated/python/research/optimization/stackoverflow_lr:centralized_stackoverflow_lr",
    ],
)

py_binary(
    name = "federated_trainer",
    srcs = ["federated_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/optimization/cifar100:federated_cifar100",
        "//tensorflow_federated/python/research/optimization/emnist:federated_emnist",
        "//tensorflow_federated/python/research/optimization/emnist_ae:federated_emnist_ae",
        "//tensorflow_federated/python/research/optimization/shakespeare:federated_shakespeare",
        "//tensorflow_federated/python/research/optimization/shared:fed_avg_schedule",
        "//tensorflow_federated/python/research/optimization/shared:optimizer_utils",
        "//tensorflow_federated/python/research/optimization/stackoverflow:federated_stackoverflow",
        "//tensorflow_federated/python/research/optimization/stackoverflow_lr:federated_stackoverflow_lr",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)

py_test(
    name = "federated_tasks_test",
    size = "large",
    srcs = ["federated_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nopresubmit",
    ],
    deps = [
        "//tensorflow_federated/python/research/optimization/cifar100:federated_cifar100",
        "//tensorflow_federated/python/research/optimization/emnist:federated_emnist",
        "//tensorflow_federated/python/research/optimization/emnist_ae:federated_emnist_ae",
        "//tensorflow_federated/python/research/optimization/shakespeare:federated_shakespeare",
        "//tensorflow_federated/python/research/optimization/shared:fed_avg_schedule",
        "//tensorflow_federated/python/research/optimization/stackoverflow:federated_stackoverflow",
        "//tensorflow_federated/python/research/optimization/stackoverflow_lr:federated_stackoverflow_lr",
        "//tensorflow_federated/python/research/utils:metrics_manager",
    ],
)
