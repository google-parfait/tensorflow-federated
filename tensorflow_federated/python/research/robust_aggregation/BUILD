package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "robust_aggregation",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":robust_federated_aggregation",
    ],
)

py_library(
    name = "robust_federated_aggregation",
    srcs = ["robust_federated_aggregation.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_federated",
    ],
)


py_test(
    name = "robust_federated_aggregation_test",
    size = "large",
    srcs = ["robust_federated_aggregation_test.py"],
    deps = [
        ":robust_federated_aggregation",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)
