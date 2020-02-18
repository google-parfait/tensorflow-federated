package(default_visibility = [
    "//tensorflow_federated/python/core/api:__subpackages__",
    "//tensorflow_federated/python/core/backends:__subpackages__",
    "//tensorflow_federated/python/core/framework:__subpackages__",
    "//tensorflow_federated/python/core/impl/compiler:__subpackages__",
    "//tensorflow_federated/python/core/impl/context_stack:__subpackages__",
    "//tensorflow_federated/python/core/impl/executors:__subpackages__",
    "//tensorflow_federated/python/core/impl/wrappers:__subpackages__",
    "//tensorflow_federated/python/core/utils:__subpackages__",
    # TODO(b/143772942): This package is temporary and can be removed once all
    # the appropriate  modules have been moved out of the `impl` package.
    "//tensorflow_federated/python/core/impl:__pkg__",
])

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
        ":type_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
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
        ":placement_literals",
        ":transformation_utils",
        ":type_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:type_utils",
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
        ":placement_literals",
        ":test_utils",
        ":type_factory",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)

py_library(
    name = "building_blocks",
    srcs = ["building_blocks.py"],
    srcs_version = "PY3",
    deps = [
        ":intrinsic_defs",
        ":placement_literals",
        ":type_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
        "//tensorflow_federated/python/core/impl:type_utils",
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
        ":type_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
    ],
)

py_library(
    name = "intrinsic_defs",
    srcs = ["intrinsic_defs.py"],
    srcs_version = "PY3",
    deps = [
        ":type_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "intrinsic_defs_test",
    size = "small",
    srcs = ["intrinsic_defs_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":intrinsic_defs",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)

py_library(
    name = "placement_literals",
    srcs = ["placement_literals.py"],
    srcs_version = "PY3",
)

py_test(
    name = "placement_literals_test",
    size = "small",
    srcs = ["placement_literals_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":placement_literals"],
)

py_library(
    name = "proto_transformations",
    srcs = ["proto_transformations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "proto_transformations_test",
    size = "small",
    srcs = ["proto_transformations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":building_block_analysis",
        ":building_block_factory",
        ":building_blocks",
        ":proto_transformations",
        ":test_utils",
        ":type_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
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
        ":type_serialization",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl:tensorflow_deserialization",
        "//tensorflow_federated/python/core/impl:type_utils",
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
        ":transformation_utils",
        ":tree_analysis",
        ":tree_transformations",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:compiled_computation_transforms",
        "//tensorflow_federated/python/core/impl:tree_to_cc_transformations",
        "//tensorflow_federated/python/core/impl:type_utils",
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
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl:tree_to_cc_transformations",
    ],
)

py_library(
    name = "transformation_utils",
    srcs = ["transformation_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":building_blocks",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl:type_utils",
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
        ":placement_literals",
        ":test_utils",
        ":transformation_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
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
        ":placement_literals",
        ":transformation_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
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
        ":type_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
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
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:type_utils",
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
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)

py_library(
    name = "type_factory",
    srcs = ["type_factory.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
    ],
)

py_test(
    name = "type_factory_test",
    size = "small",
    srcs = ["type_factory_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":type_factory"],
)

py_library(
    name = "type_serialization",
    srcs = ["type_serialization.py"],
    srcs_version = "PY3",
    deps = [
        ":placement_literals",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "type_serialization_test",
    size = "small",
    srcs = ["type_serialization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":type_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)
