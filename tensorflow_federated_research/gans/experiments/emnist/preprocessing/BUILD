package(default_visibility = ["//tensorflow_federated_research/gans/experiments/emnist:__subpackages__"])

licenses(["notice"])

py_binary(
    name = "filter_examples",
    srcs = ["filter_examples.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated_research/gans/experiments/emnist:emnist_data_utils",
        "//tensorflow_federated_research/gans/experiments/emnist/classifier:emnist_classifier_model",
        "//tensorflow_federated_research/utils:utils_impl",
    ],
)

py_binary(
    name = "filter_users",
    srcs = ["filter_users.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated_research/gans/experiments/emnist:emnist_data_utils",
        "//tensorflow_federated_research/gans/experiments/emnist/classifier:emnist_classifier_model",
        "//tensorflow_federated_research/utils:utils_impl",
    ],
)

py_library(
    name = "filtered_emnist_data_utils",
    srcs = ["filtered_emnist_data_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated_research/gans/experiments/emnist:emnist_data_utils",
    ],
)
