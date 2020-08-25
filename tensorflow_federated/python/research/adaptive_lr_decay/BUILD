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

py_library(
    name = "decay_iterative_process_builder",
    srcs = ["decay_iterative_process_builder.py"],
    srcs_version = "PY3",
    deps = [
        ":adaptive_fed_avg",
        ":callbacks",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/optimization/shared:optimizer_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)

py_test(
    name = "decay_iterative_process_builder_test",
    size = "large",
    srcs = ["decay_iterative_process_builder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["manual"],
    deps = [
        ":adaptive_fed_avg",
        ":callbacks",
        ":decay_iterative_process_builder",
        "//tensorflow_federated",
    ],
)

py_binary(
    name = "run_federated_cifar100",
    srcs = ["run_federated_cifar100.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adaptive_fed_avg",
        ":decay_iterative_process_builder",
        "//tensorflow_federated/python/research/utils:training_loop",
        "//tensorflow_federated/python/research/utils:training_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
        "//tensorflow_federated/python/research/utils/datasets:cifar100_dataset",
        "//tensorflow_federated/python/research/utils/models:resnet_models",
    ],
)

py_binary(
    name = "run_federated_emnist",
    srcs = ["run_federated_emnist.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adaptive_fed_avg",
        ":decay_iterative_process_builder",
        "//tensorflow_federated/python/research/utils:training_loop",
        "//tensorflow_federated/python/research/utils:training_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
        "//tensorflow_federated/python/research/utils/datasets:emnist_dataset",
        "//tensorflow_federated/python/research/utils/models:emnist_models",
    ],
)

py_binary(
    name = "run_federated_shakespeare",
    srcs = ["run_federated_shakespeare.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adaptive_fed_avg",
        ":decay_iterative_process_builder",
        "//tensorflow_federated/python/research/optimization/shared:keras_metrics",
        "//tensorflow_federated/python/research/utils:training_loop",
        "//tensorflow_federated/python/research/utils:training_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
        "//tensorflow_federated/python/research/utils/datasets:shakespeare_dataset",
        "//tensorflow_federated/python/research/utils/models:shakespeare_models",
    ],
)

py_binary(
    name = "run_federated_stackoverflow",
    srcs = ["run_federated_stackoverflow.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adaptive_fed_avg",
        ":decay_iterative_process_builder",
        "//tensorflow_federated/python/research/optimization/shared:keras_metrics",
        "//tensorflow_federated/python/research/utils:training_loop",
        "//tensorflow_federated/python/research/utils:training_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
        "//tensorflow_federated/python/research/utils/datasets:stackoverflow_dataset",
        "//tensorflow_federated/python/research/utils/models:stackoverflow_models",
    ],
)

py_binary(
    name = "run_federated_stackoverflow_lr",
    srcs = ["run_federated_stackoverflow_lr.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adaptive_fed_avg",
        ":decay_iterative_process_builder",
        "//tensorflow_federated/python/research/utils:training_loop",
        "//tensorflow_federated/python/research/utils:training_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
        "//tensorflow_federated/python/research/utils/datasets:stackoverflow_lr_dataset",
        "//tensorflow_federated/python/research/utils/models:stackoverflow_lr_models",
    ],
)
