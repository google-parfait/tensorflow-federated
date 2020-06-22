package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "emnist_models",
    srcs = ["emnist_models.py"],
    srcs_version = "PY3",
)

py_test(
    name = "emnist_models_test",
    srcs = ["emnist_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":emnist_models"],
)

py_library(
    name = "emnist_ae_models",
    srcs = ["emnist_ae_models.py"],
    srcs_version = "PY3",
)

py_test(
    name = "emnist_ae_models_test",
    srcs = ["emnist_ae_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":emnist_ae_models"],
)

py_library(
    name = "resnet_models",
    srcs = ["resnet_models.py"],
    srcs_version = "PY3",
)

py_test(
    name = "resnet_models_test",
    size = "small",
    timeout = "moderate",
    srcs = ["resnet_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":resnet_models"],
)

py_library(
    name = "shakespeare_models",
    srcs = ["shakespeare_models.py"],
    srcs_version = "PY3",
)

py_test(
    name = "shakespeare_models_test",
    srcs = ["shakespeare_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":shakespeare_models",
        "//tensorflow_federated/python/research/optimization/shared:keras_metrics",
    ],
)

py_library(
    name = "stackoverflow_models",
    srcs = ["stackoverflow_models.py"],
    srcs_version = "PY3",
)

py_test(
    name = "stackoverflow_models_test",
    srcs = ["stackoverflow_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":stackoverflow_models"],
)

py_library(
    name = "stackoverflow_lr_models",
    srcs = ["stackoverflow_lr_models.py"],
    srcs_version = "PY3",
)

py_test(
    name = "stackoverflow_lr_models_test",
    srcs = ["stackoverflow_lr_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":stackoverflow_lr_models"],
)
