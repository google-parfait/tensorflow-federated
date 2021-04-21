load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    packages = [
        "//tensorflow_federated/experimental/python/core/impl/jax_context/...",
        "//tensorflow_federated/experimental/python/core/impl/wrappers/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "jax_context",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "jax_computation_context",
    srcs = ["jax_computation_context.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/context_stack:context_base"],
)

py_test(
    name = "jax_computation_context_test",
    size = "small",
    srcs = ["jax_computation_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":jax_computation_context",
        "//tensorflow_federated/python/core/api:computations",
    ],
)

py_library(
    name = "jax_serialization",
    srcs = ["jax_serialization.py"],
    srcs_version = "PY3",
    deps = [
        ":jax_computation_context",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/backends/xla:xla_serialization",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "jax_serialization_test",
    size = "small",
    srcs = ["jax_serialization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":jax_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/backends/xla:xla_serialization",
        "//tensorflow_federated/python/core/impl/computation:function_utils",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)
