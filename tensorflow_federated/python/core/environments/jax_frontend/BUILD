load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":jax_frontend_packages",
        "//tensorflow_federated/python/core/environments:environments_users",
        "//tensorflow_federated/python/core/environments/jax:jax_packages",
        "//tensorflow_federated/python/core/environments/xla_backend:xla_backend_packages",
    ],
)

package_group(
    name = "jax_frontend_packages",
    packages = ["//tensorflow_federated/python/core/environments/jax_frontend/..."],
)

licenses(["notice"])

py_library(
    name = "jax_frontend",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "jax_computation",
    srcs = ["jax_computation.py"],
    deps = [
        ":jax_serialization",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/computation:computation_wrapper",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//third_party/jax",
        "@pypi_numpy//:pkg",
    ],
)

py_test(
    name = "jax_computation_test",
    size = "small",
    srcs = ["jax_computation_test.py"],
    deps = [
        ":jax_computation",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/computation:polymorphic_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//third_party/jax",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "jax_computation_context",
    srcs = ["jax_computation_context.py"],
    deps = ["//tensorflow_federated/python/core/impl/context_stack:context_base"],
)

py_test(
    name = "jax_computation_context_test",
    size = "small",
    srcs = ["jax_computation_context_test.py"],
    deps = [
        ":jax_computation",
        ":jax_computation_context",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "jax_serialization",
    srcs = ["jax_serialization.py"],
    deps = [
        ":jax_computation_context",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/xla_backend:xla_serialization",
        "//tensorflow_federated/python/core/impl/computation:function_utils",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:array_shape",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:typed_object",
        "//third_party/jax",
        "@pypi_numpy//:pkg",
    ],
)

py_test(
    name = "jax_serialization_test",
    size = "small",
    srcs = ["jax_serialization_test.py"],
    deps = [
        ":jax_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/xla_backend:xla_serialization",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//third_party/jax",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)
