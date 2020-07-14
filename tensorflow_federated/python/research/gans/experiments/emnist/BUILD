package(default_visibility = ["//tensorflow_federated/python/research/gans/experiments/emnist:__subpackages__"])

licenses(["notice"])

py_library(
    name = "emnist_data_utils",
    srcs = ["emnist_data_utils.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "emnist_data_utils_test",
    srcs = ["emnist_data_utils_test.py"],
    local = True,
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":emnist_data_utils",
        "//tensorflow_federated",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":emnist_data_utils",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/gans:gan_losses",
        "//tensorflow_federated/python/research/gans:gan_training_tf_fns",
        "//tensorflow_federated/python/research/gans:tff_gans",
        "//tensorflow_federated/python/research/gans:training_loops",
        "//tensorflow_federated/python/research/gans/experiments/emnist/classifier:emnist_classifier_model",
        "//tensorflow_federated/python/research/gans/experiments/emnist/eval:emnist_eval_util",
        "//tensorflow_federated/python/research/gans/experiments/emnist/models:convolutional_gan_networks",
        "//tensorflow_federated/python/research/gans/experiments/emnist/preprocessing:filtered_emnist_data_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)

py_binary(
    name = "run_experiments",
    srcs = ["run_experiments.py"],
    data = ["//tensorflow_federated/python/research/gans/experiments/emnist:train"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/research/utils:utils_impl"],
)
