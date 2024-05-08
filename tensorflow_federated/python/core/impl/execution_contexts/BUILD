load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":execution_contexts_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
    ],
)

package_group(
    name = "execution_contexts_packages",
    packages = ["//tensorflow_federated/python/core/impl/execution_contexts/..."],
)

licenses(["notice"])

py_library(
    name = "execution_contexts",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "async_execution_context",
    srcs = ["async_execution_context.py"],
    deps = [
        ":compiler_pipeline",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:retrying",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/executors:cardinalities_utils",
        "//tensorflow_federated/python/core/impl/executors:executor_base",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executor_value_base",
        "//tensorflow_federated/python/core/impl/executors:executors_errors",
        "//tensorflow_federated/python/core/impl/executors:ingestable_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "async_execution_context_test",
    size = "small",
    srcs = ["async_execution_context_test.py"],
    deps = [
        ":async_execution_context",
        "//tensorflow_federated/python/core/impl/executors:executors_errors",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "compiler_pipeline",
    srcs = ["compiler_pipeline.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
    ],
)

py_test(
    name = "compiler_pipeline_test",
    size = "small",
    srcs = ["compiler_pipeline_test.py"],
    deps = [
        ":compiler_pipeline",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "mergeable_comp_execution_context",
    srcs = ["mergeable_comp_execution_context.py"],
    deps = [
        ":compiler_pipeline",
        "//tensorflow_federated/python/common_libs:async_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/executors:cardinalities_utils",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:typed_object",
        "@pypi_attrs//:pkg",
    ],
)

py_test(
    name = "mergeable_comp_execution_context_test",
    size = "small",
    timeout = "moderate",
    srcs = ["mergeable_comp_execution_context_test.py"],
    deps = [
        ":mergeable_comp_execution_context",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "sync_execution_context",
    srcs = ["sync_execution_context.py"],
    deps = [
        ":async_execution_context",
        "//tensorflow_federated/python/common_libs:async_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/executors:cardinalities_utils",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
    ],
)
