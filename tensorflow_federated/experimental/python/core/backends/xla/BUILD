load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    packages = [
        "//tensorflow_federated/experimental/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "xla",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "executor",
    srcs = ["executor.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/experimental/python/core/impl/utils:xla_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl:type_utils",
        "//tensorflow_federated/python/core/impl/executors:executor_base",
        "//tensorflow_federated/python/core/impl/executors:executor_value_base",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_test(
    name = "executor_test",
    srcs = ["executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":executor",
        "//tensorflow_federated/experimental/python/core/impl/utils:xla_serialization",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)
