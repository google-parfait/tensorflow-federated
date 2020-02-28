package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_binary(
    name = "run_centralized",
    srcs = ["run_centralized.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/research/optimization/shared:keras_metrics",
        "//tensorflow_federated/python/research/optimization/stackoverflow:dataset",
        "//tensorflow_federated/python/research/optimization/stackoverflow:models",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)

py_binary(
    name = "run_federated",
    srcs = ["run_federated.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/research/optimization/shared:keras_metrics",
        "//tensorflow_federated/python/research/optimization/shared:optimizer_utils",
        "//tensorflow_federated/python/research/optimization/stackoverflow:dataset",
        "//tensorflow_federated/python/research/optimization/stackoverflow:models",
        "//tensorflow_federated/python/research/utils:adapters",
        "//tensorflow_federated/python/research/utils:training_loop",
        "//tensorflow_federated/python/research/utils:training_utils",
        "//tensorflow_federated/python/research/utils:utils_impl",
    ],
)
