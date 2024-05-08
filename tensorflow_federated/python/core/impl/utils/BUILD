load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":utils_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
        "//tensorflow_federated/python/core/impl/compiler:compiler_packages",
        "//tensorflow_federated/python/core/impl/executors:executors_packages",
        "//tensorflow_federated/python/core/impl/federated_context:federated_context_packages",
    ],
)

package_group(
    name = "utils_packages",
    packages = ["//tensorflow_federated/python/core/impl/utils/..."],
)

licenses(["notice"])

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "tensorflow_utils",
    srcs = ["tensorflow_utils.py"],
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/tensorflow_libs:graph_utils",
        "//tensorflow_federated/python/tensorflow_libs:serialization_utils",
        "@pypi_attrs//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "tensorflow_utils_test",
    size = "small",
    srcs = ["tensorflow_utils_test.py"],
    deps = [
        ":tensorflow_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/tensorflow_libs:serialization_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensorflow_test_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_attrs//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
