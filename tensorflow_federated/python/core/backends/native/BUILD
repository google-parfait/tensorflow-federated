load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":native_packages",
        "//tensorflow_federated/python/core:core_users",
        "//tensorflow_federated/python/core/backends/xla:xla_packages",

        # TODO: b/233800373 - Remove dependency from `test` to `native`.
        "//tensorflow_federated/python/core/backends/test:test_packages",
    ],
)

package_group(
    name = "native_packages",
    packages = ["//tensorflow_federated/python/core/backends/native/..."],
)

licenses(["notice"])

py_library(
    name = "native",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/core/backends:__pkg__"],
    deps = [
        ":compiler",
        ":execution_contexts",
        ":mergeable_comp_compiler",
    ],
)

py_library(
    name = "compiler",
    srcs = ["compiler.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/backends/mapreduce:compiler",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:compiled_computation_transformations",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_tree_transformations",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:transformations",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "compiler_test",
    srcs = ["compiler_test.py"],
    deps = [
        ":compiler",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "cpp_execution_contexts",
    srcs = ["cpp_execution_contexts.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":compiler",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_executor_bindings",
        "//tensorflow_federated/python/core/impl/context_stack:set_default_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:async_execution_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:sync_execution_context",
        "//tensorflow_federated/python/core/impl/executor_stacks:cpp_executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executor_bindings",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "cpp_execution_contexts_test",
    srcs = ["cpp_execution_contexts_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":cpp_execution_contexts",
        ":execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/context_stack:get_context_stack",
        "//tensorflow_federated/python/core/impl/execution_contexts:async_execution_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:sync_execution_context",
        "//tensorflow_federated/python/core/impl/executors:executor_bindings",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "execution_contexts",
    srcs = ["execution_contexts.py"],
    deps = [
        ":compiler",
        ":mergeable_comp_compiler",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/execution_contexts:async_execution_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:mergeable_comp_execution_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:sync_execution_context",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_factory",
    ],
)

py_test(
    name = "execution_contexts_test",
    srcs = ["execution_contexts_test.py"],
    deps = [
        ":execution_contexts",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/executors:remote_executor_grpc_stub",
        "//tensorflow_federated/python/tensorflow_libs:tensorflow_test_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "mergeable_comp_compiler",
    srcs = ["mergeable_comp_compiler.py"],
    deps = [
        "//tensorflow_federated/python/core/backends/mapreduce:compiler",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_building_block_factory",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_tree_transformations",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tensorflow_computation_factory",
        "//tensorflow_federated/python/core/impl/compiler:transformations",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/execution_contexts:mergeable_comp_execution_context",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "mergeable_comp_compiler_test",
    srcs = ["mergeable_comp_compiler_test.py"],
    deps = [
        ":mergeable_comp_compiler",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/execution_contexts:async_execution_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:mergeable_comp_execution_context",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_factory",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)
