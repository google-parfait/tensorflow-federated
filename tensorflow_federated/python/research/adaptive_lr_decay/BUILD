package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "adaptive_fed_avg",
    srcs = ["adaptive_fed_avg.py"],
    srcs_version = "PY3",
    deps = [
        ":callbacks",
        "//tensorflow_federated",
    ],
)

py_test(
    name = "adaptive_fed_avg_test",
    size = "large",
    srcs = ["adaptive_fed_avg_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["manual"],
    deps = [
        ":adaptive_fed_avg",
        ":callbacks",
        "//tensorflow_federated",
    ],
)

py_library(
    name = "callbacks",
    srcs = ["callbacks.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "callbacks_test",
    size = "small",
    srcs = ["callbacks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":callbacks"],
)

py_binary(
    name = "federated_trainer",
    srcs = ["federated_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adaptive_fed_avg",
        ":callbacks",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/optimization/cifar100:federated_cifar100",
        "//tensorflow_federated/python/research/optimization/emnist:federated_emnist",
        "//tensorflow_federated/python/research/optimization/emnist_ae:federated_emnist_ae",
        "//tensorflow_federated/python/research/optimization/shakespeare:federated_shakespeare",
        "//tensorflow_federated/python/research/optimization/shared:optimizer_utils",
        "//tensorflow_federated/python/research/optimization/stackoverflow:federated_stackoverflow",
        "//tensorflow_federated/python/research/optimization/stackoverflow_lr:federated_stackoverflow_lr",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)
