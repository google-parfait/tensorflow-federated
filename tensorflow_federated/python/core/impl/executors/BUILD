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
    name = "cpp_to_python_executor",
    srcs = ["cpp_to_python_executor.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":executor_bindings",
        ":executors_errors",
        ":value_serialization",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "cpp_to_python_executor_test",
    srcs = ["cpp_to_python_executor_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":cpp_to_python_executor",
        ":executor_bindings",
        ":value_serialization",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "data_conversions",
    srcs = ["data_conversions.py"],
    deps = ["@federated_language//federated_language"],
)

py_test(
    name = "data_conversions_test",
    srcs = ["data_conversions_test.py"],
    deps = [
        ":data_conversions",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "executor_bindings",
    srcs = ["executor_bindings.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":data_conversions",
        "//tensorflow_federated/cc/core/impl/executors:executor_bindings",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "executor_bindings_test",
    timeout = "moderate",
    srcs = ["executor_bindings_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":executor_bindings",
        ":executor_test_utils_bindings",
        "@federated_language//federated_language",
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
    name = "executor_utils",
    srcs = ["executor_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "executor_utils_test",
    srcs = ["executor_utils_test.py"],
    deps = [
        ":executor_utils",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "executors_errors",
    srcs = ["executors_errors.py"],
    deps = ["@federated_language//federated_language"],
)

py_library(
    name = "remote_executor",
    srcs = ["remote_executor.py"],
    deps = [
        ":remote_executor_stub",
        ":value_serialization",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "remote_executor_test",
    size = "small",
    srcs = ["remote_executor_test.py"],
    deps = [
        ":remote_executor",
        ":remote_executor_stub",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "remote_executor_stub",
    srcs = ["remote_executor_stub.py"],
    deps = ["//tensorflow_federated/proto/v0:executor_py_pb2"],
)

py_library(
    name = "remote_executor_grpc_stub",
    srcs = ["remote_executor_grpc_stub.py"],
    deps = [
        ":executors_errors",
        ":remote_executor_stub",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/proto/v0:executor_py_pb2_grpc",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "remote_executor_grpc_stub_test",
    srcs = ["remote_executor_grpc_stub_test.py"],
    deps = [
        ":executors_errors",
        ":remote_executor_grpc_stub",
        ":value_serialization",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/proto/v0:executor_py_pb2_grpc",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "value_serialization",
    srcs = ["value_serialization.py"],
    deps = [
        ":executor_utils",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:array_py_pb2",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_test(
    name = "value_serialization_test",
    size = "small",
    srcs = ["value_serialization_test.py"],
    deps = [
        ":value_serialization",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)
