package(default_visibility = ["//tensorflow_federated/python/research/gans/experiments/emnist:__subpackages__"])

licenses(["notice"])

py_library(
    name = "emnist_eval_util",
    srcs = ["emnist_eval_util.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "emnist_eval_util_test",
    srcs = ["emnist_eval_util_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "local",
        "manual",
        "notap",
    ],
    deps = [
        ":emnist_eval_util",
        "//tensorflow_federated/python/research/gans/experiments/emnist:emnist_data_utils",
        "//tensorflow_federated/python/research/gans/experiments/emnist/classifier:emnist_classifier_model",
    ],
)
