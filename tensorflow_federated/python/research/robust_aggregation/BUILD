package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "robust_aggregation",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [":robust_federated_aggregation"],
)

py_library(
    name = "robust_federated_aggregation",
    srcs = ["robust_federated_aggregation.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core",
    ],
)

py_test(
    name = "robust_federated_aggregation_test",
    srcs = ["robust_federated_aggregation_test.py"],
    deps = [
        ":robust_federated_aggregation",
        "//tensorflow_federated",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/learning",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)
