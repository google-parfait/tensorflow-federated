load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core/impl:impl_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/impl/executors/...",

        # Impl Dependencies
        "//tensorflow_federated/python/core/impl/executor_stacks/...",
        "//tensorflow_federated/python/core/impl/bindings_utils/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "executors",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "caching_executor",
    srcs = ["caching_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_test(
    name = "caching_executor_test",
    size = "small",
    srcs = ["caching_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":caching_executor",
        ":eager_tf_executor",
        ":executor_base",
        ":executor_stacks",
        ":executor_test_utils",
        ":reference_resolving_executor",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "cardinalities_utils",
    srcs = ["cardinalities_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":cardinality_carrying_base",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "cardinalities_utils_test",
    srcs = ["cardinalities_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cardinalities_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "cardinality_carrying_base",
    srcs = ["cardinality_carrying_base.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "data_backend_base",
    srcs = ["data_backend_base.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "data_executor",
    srcs = ["data_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":data_backend_base",
        ":executor_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_test(
    name = "data_executor_test",
    size = "small",
    srcs = ["data_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data_backend_base",
        ":data_executor",
        ":eager_tf_executor",
        ":execution_context",
        ":executor_stacks",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/federated_context:data",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "eager_tf_executor",
    srcs = ["eager_tf_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:typed_object",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "//tensorflow_federated/python/tensorflow_libs:graph_merge",
    ],
)

py_cpu_gpu_test(
    name = "eager_tf_executor_test",
    size = "small",
    srcs = ["eager_tf_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":executor_stacks",
        ":executor_test_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_test(
    name = "eager_tf_executor_multi_gpu_test",
    size = "small",
    srcs = ["eager_tf_executor_multi_gpu_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":eager_tf_executor",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
    ],
)

py_library(
    name = "execution_context",
    srcs = ["execution_context.py"],
    srcs_version = "PY3",
    deps = [
        ":cardinalities_utils",
        ":executor_base",
        ":executor_factory",
        ":executor_value_base",
        ":ingestable_base",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/impl/compiler:compiler_pipeline",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "execution_context_test",
    size = "small",
    srcs = ["execution_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":execution_context",
        ":executor_stacks",
        ":executor_test_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "executor_base",
    srcs = ["executor_base.py"],
    srcs_version = "PY3",
    deps = [":executor_value_base"],
)

py_library(
    name = "executor_bindings",
    srcs = ["executor_bindings.py"],
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [
        "//tensorflow_federated/python/core/impl/bindings_utils:data_conversions",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "executor_bindings_test",
    srcs = ["executor_bindings_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [
        ":executor_bindings",
        ":serialization_bindings",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/bindings_utils:value_serialization",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_library(
    name = "executor_factory",
    srcs = ["executor_factory.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "executor_service",
    srcs = ["executor_service.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_factory",
        ":executor_serialization",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
    ],
)

py_test(
    name = "executor_service_test",
    size = "small",
    srcs = ["executor_service_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":executor_base",
        ":executor_factory",
        ":executor_serialization",
        ":executor_service",
        ":executor_stacks",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "executor_serialization",
    srcs = ["executor_serialization.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_transformations",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "executor_serialization_test",
    size = "small",
    srcs = ["executor_serialization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":executor_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "executor_stacks",
    srcs = ["executor_stacks.py"],
    srcs_version = "PY3",
    deps = [
        ":caching_executor",
        ":eager_tf_executor",
        ":execution_context",
        ":executor_base",
        ":executor_factory",
        ":federated_composing_strategy",
        ":federated_resolving_strategy",
        ":federating_executor",
        ":reference_resolving_executor",
        ":remote_executor",
        ":sequence_executor",
        ":sizing_executor",
        ":thread_delegating_executor",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:local_computation_factory_base",
        "//tensorflow_federated/python/core/impl/compiler:tensorflow_computation_factory",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_cpu_gpu_test(
    name = "executor_stacks_test",
    size = "small",
    srcs = ["executor_stacks_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":executor_base",
        ":executor_factory",
        ":executor_stacks",
        ":executor_test_utils",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "executor_stacks_multi_gpu_test",
    srcs = ["executor_stacks_multi_gpu_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":executor_base",
        ":executor_stacks",
        ":executor_test_utils",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "executor_test_utils",
    testonly = True,
    srcs = ["executor_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":execution_context",
        ":executor_base",
        ":executor_stacks",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/compiler:computation_factory",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:tensorflow_computation_factory",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_factory",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "executor_test_utils_test",
    size = "small",
    srcs = ["executor_test_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":executor_stacks",
        ":executor_test_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
    ],
)

py_library(
    name = "executor_utils",
    srcs = ["executor_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:local_computation_factory_base",
        "//tensorflow_federated/python/core/impl/compiler:tensorflow_computation_factory",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_factory",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "executor_utils_test",
    size = "small",
    srcs = ["executor_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":executor_test_utils",
        ":executor_utils",
        ":executor_value_base",
        ":federated_composing_strategy",
        ":federated_resolving_strategy",
        ":federating_executor",
        ":reference_resolving_executor",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "executor_value_base",
    srcs = ["executor_value_base.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/types:typed_object"],
)

py_library(
    name = "federated_composing_strategy",
    srcs = ["federated_composing_strategy.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        ":federating_executor",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:local_computation_factory_base",
        "//tensorflow_federated/python/core/impl/compiler:tensorflow_computation_factory",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "federated_composing_strategy_test",
    size = "small",
    srcs = ["federated_composing_strategy_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":federated_composing_strategy",
        ":federated_resolving_strategy",
        ":federating_executor",
        ":reference_resolving_executor",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "federated_resolving_strategy",
    srcs = ["federated_resolving_strategy.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        ":federating_executor",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/compiler:local_computation_factory_base",
        "//tensorflow_federated/python/core/impl/compiler:tensorflow_computation_factory",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "federated_resolving_strategy_test",
    size = "small",
    srcs = ["federated_resolving_strategy_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":executor_test_utils",
        ":federated_resolving_strategy",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "federating_executor",
    srcs = ["federating_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_test(
    name = "federating_executor_test",
    size = "small",
    srcs = ["federating_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":executor_test_utils",
        ":executor_value_base",
        ":federated_resolving_strategy",
        ":federating_executor",
        ":reference_resolving_executor",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "ingestable_base",
    srcs = ["ingestable_base.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/types:typed_object"],
)

py_library(
    name = "reference_resolving_executor",
    srcs = ["reference_resolving_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_test(
    name = "reference_resolving_executor_test",
    size = "small",
    srcs = ["reference_resolving_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":executor_stacks",
        ":executor_test_utils",
        ":federated_resolving_strategy",
        ":federating_executor",
        ":reference_resolving_executor",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "remote_executor",
    srcs = ["remote_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":execution_context",
        ":executor_base",
        ":executor_serialization",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "remote_executor_test",
    size = "small",
    srcs = ["remote_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":execution_context",
        ":executor_service",
        ":executor_stacks",
        ":executor_test_utils",
        ":reference_resolving_executor",
        ":remote_executor",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "sequence_executor",
    srcs = ["sequence_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "sequence_executor_test",
    size = "small",
    srcs = ["sequence_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":sequence_executor",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_factory",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "serialization_bindings",
    srcs = ["serialization_bindings.py"],
    srcs_version = "PY3",
    tags = ["nokokoro"],
)

py_test(
    name = "serialization_bindings_test",
    srcs = ["serialization_bindings_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [
        ":serialization_bindings",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "sizing_executor",
    srcs = ["sizing_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_value_base",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "sizing_executor_test",
    size = "small",
    srcs = ["sizing_executor_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":caching_executor",
        ":eager_tf_executor",
        ":reference_resolving_executor",
        ":sizing_executor",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "thread_delegating_executor",
    srcs = ["thread_delegating_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_value_base",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
    ],
)

py_test(
    name = "thread_delegating_executor_test",
    size = "small",
    srcs = ["thread_delegating_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":caching_executor",
        ":eager_tf_executor",
        ":executor_base",
        ":thread_delegating_executor",
        "//tensorflow_federated/python/core/api:computations",
    ],
)

py_library(
    name = "transforming_executor",
    srcs = ["transforming_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
    ],
)

py_test(
    name = "transforming_executor_test",
    size = "small",
    srcs = ["transforming_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":transforming_executor",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)
