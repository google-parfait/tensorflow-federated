load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "backend_info",
    srcs = ["backend_info.py"],
    srcs_version = "PY3",
    tags = [
        "manual",
    ],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "backend_info_test",
    srcs = ["backend_info_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
    ],
    deps = [":backend_info"],
)

py_library(
    name = "compiler",
    srcs = ["compiler.py"],
    srcs_version = "PY3",
    tags = [
        "manual",
    ],
    deps = [
        ":computation_module",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "compiler_test",
    srcs = ["compiler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
    ],
    deps = [
        ":backend_info",
        ":compiler",
        ":computation_module",
        ":runtime",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl:computation_impl",
    ],
)

py_library(
    name = "computation_module",
    srcs = ["computation_module.py"],
    srcs_version = "PY3",
    tags = [
        "manual",
    ],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
    ],
)

py_test(
    name = "computation_module_test",
    srcs = ["computation_module_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
    ],
    deps = [
        ":computation_module",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_library(
    name = "executor",
    srcs = ["executor.py"],
    srcs_version = "PY3",
    tags = [
        "manual",
    ],
    deps = [
        ":backend_info",
        ":compiler",
        ":runtime",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl:type_utils",
        "//tensorflow_federated/python/core/impl/executors:executor_base",
        "//tensorflow_federated/python/core/impl/executors:executor_value_base",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "executor_test",
    srcs = ["executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
    ],
    deps = [
        ":backend_info",
        ":executor",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/context_stack:set_default_context",
        "//tensorflow_federated/python/core/impl/executors:execution_context",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
    ],
)

py_library(
    name = "runtime",
    srcs = ["runtime.py"],
    srcs_version = "PY3",
    tags = [
        "manual",
    ],
    deps = [
        ":backend_info",
        ":computation_module",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:typed_object",
    ],
)

py_test(
    name = "runtime_test",
    srcs = ["runtime_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
    ],
    deps = [
        ":backend_info",
        ":computation_module",
        ":runtime",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)
