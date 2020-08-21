package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "fed_avg_schedule",
    srcs = ["fed_avg_schedule.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_test(
    name = "fed_avg_schedule_test",
    size = "large",
    srcs = ["fed_avg_schedule_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [
        ":fed_avg_schedule",
        "//tensorflow_federated",
    ],
)

py_library(
    name = "iterative_process_builder",
    srcs = ["iterative_process_builder.py"],
    srcs_version = "PY3",
    deps = [
        ":fed_avg_schedule",
        ":optimizer_utils",
        "//tensorflow_federated",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)

py_test(
    name = "iterative_process_builder_test",
    size = "medium",
    timeout = "long",
    srcs = ["iterative_process_builder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":iterative_process_builder",
        "//tensorflow_federated",
    ],
)

py_library(
    name = "keras_metrics",
    srcs = ["keras_metrics.py"],
    srcs_version = "PY3",
)

py_test(
    name = "keras_metrics_test",
    size = "small",
    srcs = ["keras_metrics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":keras_metrics"],
)

py_library(
    name = "keras_callbacks",
    srcs = ["keras_callbacks.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/research/utils:utils_impl"],
)

py_test(
    name = "keras_callbacks_test",
    size = "small",
    srcs = ["keras_callbacks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":keras_callbacks"],
)

py_library(
    name = "optimizer_utils",
    srcs = ["optimizer_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/research/optimization/shared/keras_optimizers:lars",
        "//tensorflow_federated/python/research/optimization/shared/keras_optimizers:yogi",
    ],
)

py_test(
    name = "optimizer_utils_test",
    size = "small",
    srcs = ["optimizer_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":optimizer_utils"],
)

py_library(
    name = "schedule_utils",
    srcs = ["schedule_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "schedule_utils_test",
    size = "small",
    srcs = ["schedule_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":schedule_utils",
        "//tensorflow_federated",
    ],
)
