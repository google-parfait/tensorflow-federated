package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "federated_emnist_ae",
    srcs = ["federated_emnist_ae.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/utils:training_loop",
        "//tensorflow_federated/python/research/utils:training_utils",
        "//tensorflow_federated/python/research/utils/datasets:emnist_ae_dataset",
        "//tensorflow_federated/python/research/utils/models:emnist_ae_models",
    ],
)

py_binary(
    name = "run_centralized",
    srcs = ["run_centralized.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/research/optimization/shared:keras_callbacks",
        "//tensorflow_federated/python/research/optimization/shared:optimizer_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
        "//tensorflow_federated/python/research/utils/datasets:emnist_ae_dataset",
        "//tensorflow_federated/python/research/utils/models:emnist_ae_models",
    ],
)
