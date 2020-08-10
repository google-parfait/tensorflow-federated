# Direct and indirect packages that have visibility into the `impl` package.
package_group(
    name = "impl_visibility",
    includes = [
        "//tensorflow_federated/python/core:core_visibility",
    ],
    packages = [
        "//tensorflow_federated/python/core/api/...",
        "//tensorflow_federated/python/core/backends/...",
        "//tensorflow_federated/python/core/framework/...",
        "//tensorflow_federated/python/core/templates/...",
        "//tensorflow_federated/python/core/test/...",
        "//tensorflow_federated/python/core/utils/...",
    ],
)

package_group(
    name = "default_visibility",
    includes = [":impl_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/impl/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "impl",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_test(
    name = "ast_generation_test",
    size = "small",
    srcs = ["ast_generation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
    ],
)

py_library(
    name = "computation_impl",
    srcs = ["computation_impl.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
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
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
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
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
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
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
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
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/types:type_factory",
    ],
)

py_library(
    name = "federated_computation_context",
    srcs = ["federated_computation_context.py"],
    srcs_version = "PY3",
    deps = [
        ":value_impl",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/context_stack:symbol_binding_context",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "federated_computation_context_test",
    size = "small",
    srcs = ["federated_computation_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_computation_context",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
    ],
)

py_library(
    name = "federated_computation_utils",
    srcs = ["federated_computation_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":federated_computation_context",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_test(
    name = "federated_computation_utils_test",
    size = "small",
    srcs = ["federated_computation_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_computation_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/utils:function_utils",
    ],
)

py_library(
    name = "intrinsic_bodies",
    srcs = ["intrinsic_bodies.py"],
    srcs_version = "PY3",
    deps = [
        ":intrinsic_factory",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "intrinsic_bodies_test",
    size = "medium",
    srcs = ["intrinsic_bodies_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":intrinsic_bodies",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/executors:executor_test_utils",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
    ],
)

py_library(
    name = "intrinsic_factory",
    srcs = ["intrinsic_factory.py"],
    srcs_version = "PY3",
    deps = [
        ":value_impl",
        ":value_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/context_stack:symbol_binding_context",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_factory",
    ],
)

py_test(
    name = "intrinsic_factory_test",
    srcs = ["intrinsic_factory_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_computation_context",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
    ],
)

py_library(
    name = "reference_executor",
    srcs = ["reference_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":computation_impl",
        ":tensorflow_deserialization",
        ":type_utils",
        ":value_transformations",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:compiler_pipeline",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/executors:cardinalities_utils",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_factory",
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
        ":intrinsic_bodies",
        ":intrinsic_factory",
        ":reference_executor",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:test_utils",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/core/impl/types:type_factory",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "//tensorflow_federated/python/core/impl/wrappers:computation_wrapper_instances",
    ],
)

py_library(
    name = "tensorflow_deserialization",
    srcs = ["tensorflow_deserialization.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
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
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
    ],
)

py_library(
    name = "tensorflow_serialization",
    srcs = ["tensorflow_serialization.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation_context",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:function_utils",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "//tensorflow_federated/python/tensorflow_libs:variable_utils",
    ],
)

py_test(
    name = "tensorflow_serialization_test",
    size = "small",
    srcs = ["tensorflow_serialization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_serialization",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "test",
    testonly = True,
    srcs = ["test.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/api:computations"],
)

py_library(
    name = "tree_to_cc_transformations",
    srcs = ["tree_to_cc_transformations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:compiled_computation_transforms",
    ],
)

py_test(
    name = "tree_to_cc_transformations_test",
    size = "small",
    srcs = ["tree_to_cc_transformations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tree_to_cc_transformations",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tensorflow_computation_factory",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/wrappers:computation_wrapper_instances",
    ],
)

py_library(
    name = "type_utils",
    srcs = ["type_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
    ],
)

py_test(
    name = "type_utils_test",
    size = "small",
    srcs = ["type_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":type_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
    ],
)

py_library(
    name = "value_impl",
    srcs = ["value_impl.py"],
    srcs_version = "PY3",
    deps = [
        ":tensorflow_serialization",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/context_stack:symbol_binding_context",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
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
        ":federated_computation_context",
        ":reference_executor",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
    ],
)

py_library(
    name = "value_transformations",
    srcs = ["value_transformations.py"],
    srcs_version = "PY3",
    deps = [
        ":federated_computation_utils",
        ":intrinsic_bodies",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
    ],
)

py_test(
    name = "value_transformations_test",
    size = "small",
    srcs = ["value_transformations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":value_transformations",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:test_utils",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
    ],
)

py_library(
    name = "value_utils",
    srcs = ["value_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":value_impl",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
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
        ":federated_computation_context",
        ":value_impl",
        ":value_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
    ],
)
