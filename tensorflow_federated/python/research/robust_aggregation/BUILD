package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "robust_federated_aggregation",
    srcs = ["robust_federated_aggregation.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_federated",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "robust_federated_aggregation_test",
    srcs = ["robust_federated_aggregation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":robust_federated_aggregation",
        "//tensorflow_federated",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)
