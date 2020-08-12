package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core/impl:impl_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/impl/compiler/...",

        # Impl Dependencies
        "//tensorflow_federated/python/core/impl/executors/...",
        "//tensorflow_federated/python/core/impl/federated_context/...",
        "//tensorflow_federated/python/core/impl/wrappers/...",

        # TODO(b/143772942): This package is temporary and can be removed once
        # all the appropriate modules have been moved out of the `impl` package.
        "//tensorflow_federated/python/core/impl",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "compiler",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "building_block_analysis",
    srcs = ["building_block_analysis.py"],
    srcs_version = "PY3",
    deps = [
        ":building_blocks",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
    ],
)

py_test(
    name = "building_block_analysis_test",
    size = "small",
    srcs = ["building_block_analysis_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":building_block_analysis",
        ":building_blocks",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_library(
    name = "building_block_factory",
    srcs = ["building_block_factory.py"],
    srcs_version = "PY3",
    deps = [
        ":building_blocks",
        ":intrinsic_defs",
        ":tensorflow_computation_factory",
        ":transformation_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "building_block_factory_test",
    size = "small",
    srcs = ["building_block_factory_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":building_block_factory",
        ":building_blocks",
        ":intrinsic_defs",
        ":test_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_library(
    name = "building_blocks",
    srcs = ["building_blocks.py"],
    srcs_version = "PY3",
    deps = [
        ":intrinsic_defs",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_test(
    name = "building_blocks_test",
    size = "small",
    srcs = ["building_blocks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":building_block_factory",
        ":building_blocks",
        ":intrinsic_defs",
        ":test_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "compiled_computation_transforms",
    srcs = ["compiled_computation_transforms.py"],
    srcs_version = "PY3",
    deps = [
        ":building_block_factory",
        ":building_blocks",
        ":tensorflow_computation_transformations",
        ":transformation_utils",
        ":tree_analysis",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "//tensorflow_federated/python/tensorflow_libs:graph_merge",
        "//tensorflow_federated/python/tensorflow_libs:graph_optimizations",
        "//tensorflow_federated/python/tensorflow_libs:graph_spec",
    ],
)

py_test(
    name = "compiled_computation_transforms_test",
    size = "small",
    srcs = ["compiled_computation_transforms_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":building_block_factory",
        ":building_blocks",
        ":compiled_computation_transforms",
        ":tensorflow_computation_factory",
        ":tensorflow_computation_transformations",
        ":test_utils",
        ":tree_analysis",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_library(
    name = "computation_factory",
    srcs = ["computation_factory.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_factory",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_test(
    name = "computation_factory_test",
    srcs = ["computation_factory_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_factory",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_factory",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "intrinsic_defs",
    srcs = ["intrinsic_defs.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_factory",
    ],
)

py_test(
    name = "intrinsic_defs_test",
    size = "small",
    srcs = ["intrinsic_defs_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":intrinsic_defs"],
)

py_library(
    name = "compiler_pipeline",
    srcs = ["compiler_pipeline.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
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
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
    ],
)

py_library(
    name = "tensorflow_computation_factory",
    srcs = ["tensorflow_computation_factory.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_transformations",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "tensorflow_computation_factory_test",
    srcs = ["tensorflow_computation_factory_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_computation_factory",
        ":test_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_factory",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "tensorflow_computation_transformations",
    srcs = ["tensorflow_computation_transformations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "tensorflow_computation_transformations_test",
    size = "small",
    srcs = ["tensorflow_computation_transformations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":building_block_analysis",
        ":building_block_factory",
        ":building_blocks",
        ":tensorflow_computation_transformations",
        ":test_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":building_block_factory",
        ":building_blocks",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:tensorflow_deserialization",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_library(
    name = "transformations",
    srcs = ["transformations.py"],
    srcs_version = "PY3",
    deps = [
        ":building_block_factory",
        ":building_blocks",
        ":compiled_computation_transforms",
        ":transformation_utils",
        ":tree_analysis",
        ":tree_transformations",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl:tree_to_cc_transformations",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "transformations_test",
    srcs = ["transformations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":building_block_analysis",
        ":building_block_factory",
        ":building_blocks",
        ":intrinsic_defs",
        ":test_utils",
        ":transformation_utils",
        ":transformations",
        ":tree_analysis",
        ":tree_transformations",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:tree_to_cc_transformations",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
    ],
)

py_library(
    name = "transformation_utils",
    srcs = ["transformation_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":building_blocks",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_test(
    name = "transformation_utils_test",
    size = "small",
    srcs = ["transformation_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":building_block_factory",
        ":building_blocks",
        ":test_utils",
        ":transformation_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
    ],
)

py_library(
    name = "tree_analysis",
    srcs = ["tree_analysis.py"],
    srcs_version = "PY3",
    deps = [
        ":building_block_analysis",
        ":building_blocks",
        ":intrinsic_defs",
        ":transformation_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "tree_analysis_test",
    srcs = ["tree_analysis_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":building_block_analysis",
        ":building_block_factory",
        ":building_blocks",
        ":intrinsic_defs",
        ":test_utils",
        ":tree_analysis",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_library(
    name = "tree_transformations",
    srcs = ["tree_transformations.py"],
    srcs_version = "PY3",
    deps = [
        ":building_block_analysis",
        ":building_block_factory",
        ":building_blocks",
        ":intrinsic_defs",
        ":transformation_utils",
        ":tree_analysis",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_transformations",
    ],
)

py_test(
    name = "tree_transformations_test",
    size = "small",
    srcs = ["tree_transformations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":building_block_analysis",
        ":building_block_factory",
        ":building_blocks",
        ":intrinsic_defs",
        ":test_utils",
        ":transformation_utils",
        ":tree_analysis",
        ":tree_transformations",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
    ],
)
