load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":executors_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
        "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_stacks_packages",
    ],
)

package_group(
    name = "executors_packages",
    packages = ["//tensorflow_federated/python/core/impl/executors/..."],
)

licenses(["notice"])

py_library(
    name = "executors",
    srcs = ["__init__.py"],
    visibility = ["//tools/python_package:python_package_tool"],
)

py_library(
    name = "cardinality",
    srcs = ["cardinality.py"],
    deps = [
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language//federated_language_executor:executor_py_pb2",
    ],
)

py_test(
    name = "cardinality_test",
    srcs = ["cardinality_test.py"],
    deps = [
        ":cardinality",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
        "@federated_language//federated_language_executor:executor_py_pb2",
    ],
)

py_library(
    name = "cpp_to_python_executor",
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    visibility = [
        ":executors_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
        "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_stacks_packages",
        "//third_party/py/federated_language_jax:__subpackages__",
        "//third_party/py/tensorflow_federated:__subpackages__",
    ],
)

py_library(name = "data_conversions")

py_library(
    name = "executor_bindings",
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
)

py_library(
    name = "executor_test_utils_bindings",
    testonly = True,
    srcs = ["executor_test_utils_bindings.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = ["//tensorflow_federated/cc/core/impl/executors:executor_test_utils_bindings"],
)

py_library(
    name = "remote_executor",
    srcs = ["remote_executor.py"],
    deps = [
        ":cardinality",
        ":remote_executor_stub",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
        "@federated_language//federated_language_executor:executor_py_pb2",
        "@federated_language//federated_language_executor:value_serialization",
    ],
)

py_test(
    name = "remote_executor_test",
    size = "small",
    srcs = ["remote_executor_test.py"],
    deps = [
        ":remote_executor",
        ":remote_executor_stub",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
        "@federated_language//federated_language_executor:executor_py_pb2",
    ],
)

py_library(
    name = "remote_executor_stub",
    srcs = ["remote_executor_stub.py"],
    deps = ["@federated_language//federated_language_executor:executor_py_pb2"],
)

py_library(
    name = "remote_executor_grpc_stub",
    srcs = ["remote_executor_grpc_stub.py"],
    deps = [
        ":remote_executor_stub",
        "@federated_language//federated_language",
        "@federated_language//federated_language_executor:executor_errors",
        "@federated_language//federated_language_executor:executor_py_pb2",
        "@federated_language//federated_language_executor:executor_py_pb2_grpc",
    ],
)

py_test(
    name = "remote_executor_grpc_stub_test",
    srcs = ["remote_executor_grpc_stub_test.py"],
    deps = [
        ":remote_executor_grpc_stub",
        "@federated_language//federated_language",
        "@federated_language//federated_language_executor:executor_py_pb2",
        "@federated_language//federated_language_executor:executor_py_pb2_grpc",
        "@federated_language//federated_language_executor:value_serialization",
    ],
)
