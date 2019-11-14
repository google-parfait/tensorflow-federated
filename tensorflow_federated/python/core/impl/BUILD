package(default_visibility = [
    "//tensorflow_federated/python/core/api:__subpackages__",
    "//tensorflow_federated/python/core/backends:__subpackages__",
    "//tensorflow_federated/python/core/framework:__subpackages__",
    "//tensorflow_federated/python/core/impl:__subpackages__",
    "//tensorflow_federated/python/core/utils:__subpackages__",
])

licenses(["notice"])

# TODO(b/143772942): This package group is temporary and can be removed once all
# the appropriate  modules have been moved to the `compiler` package.
package_group(
    name = "compiler_visibility",
    packages = ["//tensorflow_federated/python/core/..."],
)

# TODO(b/143772942): This package group is temporary and can be removed once all
# the appropriate  modules have been moved to the `utils` package.
package_group(
    name = "utils_visibility",
    packages = ["//tensorflow_federated/python/core/..."],
)

py_library(
    name = "impl",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "caching_executor",
    srcs = ["caching_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        ":executor_base",
        ":executor_value_base",
        ":type_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
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
        ":computation_impl",
        ":eager_executor",
        ":executor_test_utils",
        ":lambda_executor",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
    ],
)

py_library(
    name = "compiled_computation_transforms",
    srcs = ["compiled_computation_transforms.py"],
    srcs_version = "PY2AND3",
    visibility = [":compiler_visibility"],
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:proto_transformations",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "//tensorflow_federated/python/tensorflow_libs:graph_merge",
    ],
)

py_test(
    name = "compiled_computation_transforms_test",
    size = "small",
    srcs = ["compiled_computation_transforms_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [":compiler_visibility"],
    deps = [
        ":compiled_computation_transforms",
        ":context_stack_impl",
        ":tensorflow_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:proto_transformations",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/wrappers:computation_wrapper_instances",
    ],
)

py_library(
    name = "compiler_pipeline",
    srcs = ["compiler_pipeline.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":computation_impl",
        ":context_stack_base",
        ":transformations",
        ":value_transformations",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
    ],
)

py_test(
    name = "compiler_pipeline_test",
    size = "small",
    srcs = ["compiler_pipeline_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":compiler_pipeline",
        ":computation_impl",
        ":context_stack_impl",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
    ],
)

py_library(
    name = "composite_executor",
    srcs = ["composite_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        ":type_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
    ],
)

py_test(
    name = "composite_executor_test",
    size = "small",
    srcs = ["composite_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":caching_executor",
        ":composite_executor",
        ":concurrent_executor",
        ":eager_executor",
        ":federated_executor",
        ":lambda_executor",
        ":set_default_executor",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)

py_library(
    name = "computation_impl",
    srcs = ["computation_impl.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":context_stack_base",
        ":type_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:function_utils",
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
        ":context_stack_impl",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
    ],
)

py_library(
    name = "concurrent_executor",
    srcs = ["concurrent_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "concurrent_executor_test",
    size = "small",
    srcs = ["concurrent_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":concurrent_executor",
        ":eager_executor",
        ":executor_base",
        "//tensorflow_federated/python/core/api:computations",
    ],
)

py_library(
    name = "context_base",
    srcs = ["context_base.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "context_stack_base",
    srcs = ["context_stack_base.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "context_stack_impl",
    srcs = ["context_stack_impl.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":compiler_pipeline",
        ":context_base",
        ":context_stack_base",
        ":reference_executor",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "context_stack_test",
    size = "small",
    srcs = ["context_stack_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_base",
        ":context_stack_impl",
        ":reference_executor",
    ],
)

py_library(
    name = "eager_executor",
    srcs = ["eager_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        ":executor_base",
        ":executor_value_base",
        ":type_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "//tensorflow_federated/python/tensorflow_libs:graph_merge",
    ],
)

py_test(
    name = "eager_executor_test",
    size = "small",
    srcs = ["eager_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        ":eager_executor",
        ":executor_test_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
    ],
)

py_test(
    name = "end_to_end_test",
    size = "small",
    srcs = ["end_to_end_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        ":context_stack_impl",
        ":tensorflow_serialization",
        "//tensorflow_federated/python/common_libs:test",
    ],
)

py_library(
    name = "execution_context",
    srcs = ["execution_context.py"],
    srcs_version = "PY3",
    deps = [
        ":context_base",
        ":executor_base",
        ":executor_value_base",
        ":runtime_utils",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
    ],
)

py_test(
    name = "execution_context_test",
    size = "small",
    srcs = ["execution_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":execution_context",
        ":executor_stacks",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)

py_library(
    name = "executor_base",
    srcs = ["executor_base.py"],
    srcs_version = "PY3",
)

py_library(
    name = "executor_service",
    srcs = ["executor_service.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_service_utils",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "executor_service_test",
    size = "small",
    srcs = ["executor_service_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_executor",
        ":executor_base",
        ":executor_service",
        ":executor_service_utils",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
    ],
)

py_library(
    name = "executor_service_utils",
    srcs = ["executor_service_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        ":tensorflow_serialization",
        ":type_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "executor_service_utils_test",
    size = "small",
    srcs = ["executor_service_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":executor_service_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)

py_library(
    name = "executor_stacks",
    srcs = ["executor_stacks.py"],
    srcs_version = "PY3",
    deps = [
        ":caching_executor",
        ":composite_executor",
        ":concurrent_executor",
        ":eager_executor",
        ":executor_base",
        ":federated_executor",
        ":lambda_executor",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
    ],
)

py_test(
    name = "executor_stacks_test",
    size = "small",
    srcs = ["executor_stacks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":executor_stacks",
        ":executor_test_utils",
        ":set_default_executor",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)

py_library(
    name = "executor_test_utils",
    testonly = True,
    srcs = ["executor_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_value_base",
        ":set_default_executor",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/utils:tf_computation_utils",
    ],
)

py_test(
    name = "executor_test_utils_test",
    size = "small",
    srcs = ["executor_test_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_executor",
        ":executor_test_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
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
        ":type_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
    ],
)

py_library(
    name = "executor_value_base",
    srcs = ["executor_value_base.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/api:typed_object"],
)

py_library(
    name = "federated_computation_context",
    srcs = ["federated_computation_context.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":context_base",
        ":context_stack_base",
        ":type_utils",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "federated_computation_context_test",
    size = "small",
    srcs = ["federated_computation_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":federated_computation_context",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:value_base",
    ],
)

py_library(
    name = "federated_computation_utils",
    srcs = ["federated_computation_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":context_stack_base",
        ":federated_computation_context",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
    ],
)

py_test(
    name = "federated_computation_utils_test",
    size = "small",
    srcs = ["federated_computation_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":federated_computation_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/utils:function_utils",
    ],
)

py_library(
    name = "federated_executor",
    srcs = ["federated_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        ":type_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
    ],
)

py_test(
    name = "federated_executor_test",
    size = "small",
    srcs = ["federated_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        ":eager_executor",
        ":executor_test_utils",
        ":federated_executor",
        ":lambda_executor",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
    ],
)

py_library(
    name = "intrinsic_bodies",
    srcs = ["intrinsic_bodies.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":context_stack_base",
        ":intrinsic_factory",
        ":type_utils",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
    ],
)

py_test(
    name = "intrinsic_bodies_test",
    size = "small",
    srcs = ["intrinsic_bodies_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":intrinsic_bodies",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
    ],
)

py_library(
    name = "intrinsic_factory",
    srcs = ["intrinsic_factory.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":context_stack_base",
        ":type_utils",
        ":value_impl",
        ":value_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)

py_library(
    name = "lambda_executor",
    srcs = ["lambda_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        ":executor_base",
        ":executor_value_base",
        ":transformations",
        ":type_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
    ],
)

py_test(
    name = "lambda_executor_test",
    size = "small",
    srcs = ["lambda_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_executor",
        ":executor_test_utils",
        ":federated_executor",
        ":lambda_executor",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)

py_library(
    name = "reference_executor",
    srcs = ["reference_executor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":compiler_pipeline",
        ":computation_impl",
        ":context_base",
        ":runtime_utils",
        ":tensorflow_deserialization",
        ":transformations",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "reference_executor_test",
    size = "small",
    srcs = ["reference_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        ":context_stack_impl",
        ":intrinsic_bodies",
        ":intrinsic_factory",
        ":reference_executor",
        ":transformations",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:test_utils",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "//tensorflow_federated/python/core/impl/wrappers:computation_wrapper_instances",
    ],
)

py_library(
    name = "runtime_utils",
    srcs = ["runtime_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
    ],
)

py_test(
    name = "runtime_utils_test",
    srcs = ["runtime_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":runtime_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
    ],
)

py_library(
    name = "remote_executor",
    srcs = ["remote_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":execution_context",
        ":executor_base",
        ":executor_service_utils",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
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
        ":lambda_executor",
        ":remote_executor",
        ":set_default_executor",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "set_default_executor",
    srcs = ["set_default_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":execution_context",
        ":executor_base",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "set_default_executor_test",
    size = "small",
    srcs = ["set_default_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":eager_executor",
        ":set_default_executor",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
    ],
)

py_library(
    name = "tensorflow_deserialization",
    srcs = ["tensorflow_deserialization.py"],
    srcs_version = "PY2AND3",
    visibility = [":utils_visibility"],
    deps = [
        ":type_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "tensorflow_deserialization_test",
    size = "small",
    srcs = ["tensorflow_deserialization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_deserialization",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
    ],
)

py_library(
    name = "tensorflow_serialization",
    srcs = ["tensorflow_serialization.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":context_stack_base",
        ":tf_computation_context",
        ":type_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:function_utils",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "//tensorflow_federated/python/tensorflow_libs:graph_keys",
    ],
)

py_test(
    name = "tensorflow_serialization_test",
    size = "small",
    srcs = ["tensorflow_serialization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":tensorflow_serialization",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
    ],
)

py_library(
    name = "tf_computation_context",
    srcs = ["tf_computation_context.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":computation_impl",
        ":context_base",
        ":tensorflow_deserialization",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
    ],
)

py_test(
    name = "tf_computation_context_test",
    size = "small",
    srcs = ["tf_computation_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tf_computation_context",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
    ],
)

py_library(
    name = "transformations",
    srcs = ["transformations.py"],
    srcs_version = "PY2AND3",
    visibility = [":compiler_visibility"],
    deps = [
        ":compiled_computation_transforms",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_block_analysis",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
    ],
)

py_test(
    name = "transformations_test",
    size = "small",
    srcs = ["transformations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [":compiler_visibility"],
    deps = [
        ":context_stack_impl",
        ":tensorflow_serialization",
        ":transformations",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/compiler:building_block_analysis",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:test_utils",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/wrappers:computation_wrapper_instances",
    ],
)

py_library(
    name = "transforming_executor",
    srcs = ["transforming_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        ":executor_base",
        ":type_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
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
        ":transformations",
        ":transforming_executor",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)

py_library(
    name = "type_utils",
    srcs = ["type_utils.py"],
    srcs_version = "PY2AND3",
    visibility = [":utils_visibility"],
    deps = [
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
    ],
)

py_test(
    name = "type_utils_test",
    size = "small",
    srcs = ["type_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [":utils_visibility"],
    deps = [
        ":type_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)

py_library(
    name = "value_impl",
    srcs = ["value_impl.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":computation_impl",
        ":context_stack_base",
        ":tensorflow_serialization",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
        "//tensorflow_federated/python/core/impl/utils:function_utils",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "value_impl_test",
    size = "small",
    srcs = ["value_impl_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":federated_computation_context",
        ":reference_executor",
        ":transformations",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
    ],
)

py_library(
    name = "value_transformations",
    srcs = ["value_transformations.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":context_stack_base",
        ":federated_computation_utils",
        ":intrinsic_bodies",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
    ],
)

py_test(
    name = "value_transformations_test",
    size = "small",
    srcs = ["value_transformations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":value_transformations",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:test_utils",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
    ],
)

py_library(
    name = "value_utils",
    srcs = ["value_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":value_impl",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
    ],
)

py_test(
    name = "value_utils_test",
    size = "small",
    srcs = ["value_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        ":context_stack_impl",
        ":federated_computation_context",
        ":transformations",
        ":value_impl",
        ":value_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
    ],
)
