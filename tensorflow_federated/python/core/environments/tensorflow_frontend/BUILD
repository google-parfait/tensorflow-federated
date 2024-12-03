load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":tensorflow_frontend_packages",
        "//tensorflow_federated/python/core/environments:environments_users",
        "//tensorflow_federated/python/core/environments/tensorflow:tensorflow_packages",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_backend_packages",
    ],
)

package_group(
    name = "tensorflow_frontend_packages",
    packages = ["//tensorflow_federated/python/core/environments/tensorflow_frontend/..."],
)

licenses(["notice"])

py_library(
    name = "tensorflow_frontend",
    srcs = ["__init__.py"],
    visibility = ["//tools/python_package:python_package_tool"],
)

py_library(
    name = "tensorflow_computation",
    srcs = ["tensorflow_computation.py"],
    deps = [
        ":tensorflow_serialization",
        ":tensorflow_types",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "tensorflow_computation_test",
    size = "small",
    srcs = ["tensorflow_computation_test.py"],
    deps = [
        ":tensorflow_computation",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "tensorflow_computation_context",
    srcs = ["tensorflow_computation_context.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_utils",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "tensorflow_computation_context_test",
    size = "small",
    srcs = ["tensorflow_computation_context_test.py"],
    deps = [
        ":tensorflow_computation",
        ":tensorflow_computation_context",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_library(
    name = "tensorflow_serialization",
    srcs = ["tensorflow_serialization.py"],
    deps = [
        ":tensorflow_computation_context",
        ":variable_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:serialization_utils",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_utils",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_test(
    name = "tensorflow_serialization_test",
    size = "small",
    srcs = ["tensorflow_serialization_test.py"],
    deps = [
        ":tensorflow_serialization",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:serialization_utils",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_test_utils",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "variable_utils",
    srcs = ["variable_utils.py"],
)

py_library(
    name = "tensorflow_types",
    srcs = ["tensorflow_types.py"],
    deps = ["@federated_language//federated_language"],
)

py_test(
    name = "variable_utils_test",
    size = "small",
    timeout = "moderate",
    srcs = ["variable_utils_test.py"],
    deps = [":variable_utils"],
)

py_test(
    name = "tensorflow_types_test",
    srcs = ["tensorflow_types_test.py"],
    deps = [
        ":tensorflow_types",
        "@federated_language//federated_language",
    ],
)
