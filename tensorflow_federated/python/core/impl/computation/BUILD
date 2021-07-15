load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":computation_packages",
    "//tensorflow_federated/experimental/python/core/impl/jax_context:jax_context_packages",
    "//tensorflow_federated/python/core/impl:impl_visibility",
    "//tensorflow_federated/python/core/impl/executor_stacks:executor_stacks_packages",
    "//tensorflow_federated/python/core/impl/executors:executors_packages",
    "//tensorflow_federated/python/core/impl/federated_context:federated_context_packages",
    "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_context_packages",
    "//tensorflow_federated/python/core/impl/wrappers:wrappers_packages",
    "//tensorflow_federated/python/core/impl/xla_context:xla_context_packages",
])

package_group(
    name = "computation_packages",
    packages = ["//tensorflow_federated/python/core/impl/computation/..."],
)

licenses(["notice"])

py_library(
    name = "computation",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "computation_impl",
    srcs = ["computation_impl.py"],
    srcs_version = "PY3",
    deps = [
        ":function_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_test(
    name = "computation_impl_test",
    size = "small",
    srcs = ["computation_impl_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "computation_serialization",
    srcs = ["computation_serialization.py"],
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
    ],
)

py_test(
    name = "computation_serialization_test",
    size = "small",
    srcs = ["computation_serialization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_serialization",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "function_utils",
    srcs = ["function_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:typed_object",
        "//tensorflow_federated/python/tensorflow_libs:function",
    ],
)

py_test(
    name = "function_utils_test",
    size = "small",
    srcs = ["function_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":function_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)
