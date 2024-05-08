load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":program_packages",
        "//tensorflow_federated/python/analytics:analytics_packages",
        "//tensorflow_federated/python/learning:learning_packages",
        "//tensorflow_federated/python/simulation:simulation_packages",
    ],
)

package_group(
    name = "program_packages",
    packages = ["//tensorflow_federated/python/program/..."],
)

licenses(["notice"])

py_library(
    name = "program",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":client_id_data_source",
        ":data_source",
        ":dataset_data_source",
        ":federated_context",
        ":file_program_state_manager",
        ":file_release_manager",
        ":logging_release_manager",
        ":memory_release_manager",
        ":native_platform",
        ":program_state_manager",
        ":release_manager",
        ":tensorboard_release_manager",
        ":value_reference",
    ],
)

py_library(
    name = "client_id_data_source",
    srcs = ["client_id_data_source.py"],
    deps = [
        ":data_source",
        ":serialization_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_numpy//:pkg",
    ],
)

py_test(
    name = "client_id_data_source_test",
    srcs = ["client_id_data_source_test.py"],
    deps = [
        ":client_id_data_source",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "data_source",
    srcs = ["data_source.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:serializable",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "dataset_data_source",
    srcs = ["dataset_data_source.py"],
    deps = [
        ":data_source",
        ":serialization_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "dataset_data_source_test",
    srcs = ["dataset_data_source_test.py"],
    deps = [
        ":dataset_data_source",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "federated_context",
    srcs = ["federated_context.py"],
    deps = [
        ":structure_utils",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:get_context_stack",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "federated_context_test",
    srcs = ["federated_context_test.py"],
    deps = [
        ":federated_context",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "file_program_state_manager",
    srcs = ["file_program_state_manager.py"],
    deps = [
        ":file_utils",
        ":program_state_manager",
        ":structure_utils",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serializable",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "file_program_state_manager_test",
    srcs = ["file_program_state_manager_test.py"],
    deps = [
        ":file_program_state_manager",
        ":file_utils",
        ":program_state_manager",
        ":program_test_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "file_release_manager",
    srcs = ["file_release_manager.py"],
    deps = [
        ":file_utils",
        ":release_manager",
        ":structure_utils",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "file_release_manager_test",
    srcs = ["file_release_manager_test.py"],
    deps = [
        ":file_release_manager",
        ":file_utils",
        ":program_test_utils",
        ":release_manager",
        ":structure_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "file_utils",
    srcs = ["file_utils.py"],
    deps = [
        ":structure_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "file_utils_test",
    srcs = ["file_utils_test.py"],
    deps = [
        ":file_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "logging_release_manager",
    srcs = ["logging_release_manager.py"],
    deps = [
        ":release_manager",
        ":value_reference",
        "@pypi_absl_py//:pkg",
    ],
)

py_test(
    name = "logging_release_manager_test",
    srcs = ["logging_release_manager_test.py"],
    deps = [
        ":logging_release_manager",
        ":program_test_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "memory_release_manager",
    srcs = ["memory_release_manager.py"],
    deps = [
        ":release_manager",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "memory_release_manager_test",
    srcs = ["memory_release_manager_test.py"],
    deps = [
        ":memory_release_manager",
        ":program_test_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "native_platform",
    srcs = ["native_platform.py"],
    deps = [
        ":federated_context",
        ":structure_utils",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:async_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/execution_contexts:async_execution_context",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_test(
    name = "native_platform_test",
    srcs = ["native_platform_test.py"],
    deps = [
        ":native_platform",
        ":program_test_utils",
        ":structure_utils",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:async_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/execution_contexts:async_execution_context",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "prefetching_data_source",
    srcs = ["prefetching_data_source.py"],
    deps = [
        ":data_source",
        ":serialization_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/context_stack:get_context_stack",
        "//tensorflow_federated/python/core/impl/execution_contexts:async_execution_context",
        "//tensorflow_federated/python/core/impl/executors:cardinality_carrying_base",
        "//tensorflow_federated/python/core/impl/executors:executor_base",
        "//tensorflow_federated/python/core/impl/executors:executor_value_base",
        "//tensorflow_federated/python/core/impl/executors:ingestable_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "prefetching_data_source_test",
    srcs = ["prefetching_data_source_test.py"],
    tags = [
        "nokokoro",  # b/268497846 - Temporarily disable the `PrefetchingDataSource`.
    ],
    deps = [
        ":data_source",
        ":prefetching_data_source",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_test_utils",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "program_state_manager",
    srcs = ["program_state_manager.py"],
    deps = [
        ":structure_utils",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:serializable",
    ],
)

py_test(
    name = "program_state_manager_test",
    srcs = ["program_state_manager_test.py"],
    deps = [
        ":program_state_manager",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "program_test_utils",
    testonly = True,
    srcs = ["program_test_utils.py"],
    deps = [
        ":value_reference",
        "//tensorflow_federated/python/common_libs:serializable",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@pypi_attrs//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "release_manager",
    srcs = ["release_manager.py"],
    deps = [
        ":structure_utils",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@pypi_attrs//:pkg",
        "@pypi_dm_tree//:pkg",
    ],
)

py_test(
    name = "release_manager_test",
    srcs = ["release_manager_test.py"],
    deps = [
        ":program_test_utils",
        ":release_manager",
        "@pypi_absl_py//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "serialization_utils",
    srcs = ["serialization_utils.py"],
    deps = [
        ":structure_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:serializable",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "serialization_utils_test",
    srcs = ["serialization_utils_test.py"],
    deps = [
        ":program_test_utils",
        ":serialization_utils",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "structure_utils",
    srcs = ["structure_utils.py"],
    deps = [
        "@pypi_attrs//:pkg",
        "@pypi_dm_tree//:pkg",
    ],
)

py_test(
    name = "structure_utils_test",
    srcs = ["structure_utils_test.py"],
    deps = [
        ":program_test_utils",
        ":structure_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "tensorboard_release_manager",
    srcs = ["tensorboard_release_manager.py"],
    deps = [
        ":release_manager",
        ":structure_utils",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "tensorboard_release_manager_test",
    srcs = ["tensorboard_release_manager_test.py"],
    deps = [
        ":program_test_utils",
        ":tensorboard_release_manager",
        "@pypi_absl_py//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "value_reference",
    srcs = ["value_reference.py"],
    deps = [
        ":structure_utils",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:typed_object",
        "@pypi_numpy//:pkg",
    ],
)

py_test(
    name = "value_reference_test",
    srcs = ["value_reference_test.py"],
    deps = [
        ":program_test_utils",
        ":value_reference",
        "@pypi_absl_py//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
