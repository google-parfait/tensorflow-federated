package(default_visibility = ["//tensorflow_federated/python/core/api:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "compiler_pipeline",
    srcs = ["compiler_pipeline.py"],
    deps = [
        ":computation_building_blocks",
        ":computation_impl",
        ":context_stack_base",
        ":intrinsic_bodies",
        ":transformations",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "@six_archive//:six",
    ],
)

py_test(
    name = "compiler_pipeline_test",
    size = "small",
    srcs = ["compiler_pipeline_test.py"],
    deps = [
        ":compiler_pipeline",
        ":computation_building_blocks",
        ":computation_impl",
        ":context_stack_impl",
        ":intrinsic_defs",
        ":transformations",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "computation_building_blocks",
    srcs = ["computation_building_blocks.py"],
    deps = [
        ":placement_literals",
        ":type_serialization",
        ":type_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
        "@six_archive//:six",
    ],
)

py_test(
    name = "computation_building_blocks_test",
    size = "small",
    srcs = ["computation_building_blocks_test.py"],
    deps = [
        ":computation_building_blocks",
        ":context_stack_impl",
        ":tensorflow_serialization",
        ":type_serialization",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "computation_constructing_utils",
    srcs = ["computation_constructing_utils.py"],
    deps = [
        ":computation_building_blocks",
        ":intrinsic_defs",
        ":placement_literals",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "@six_archive//:six",
    ],
)

py_test(
    name = "computation_constructing_utils_test",
    size = "small",
    srcs = ["computation_constructing_utils_test.py"],
    deps = [
        ":computation_building_blocks",
        ":computation_constructing_utils",
        ":context_stack_impl",
        ":placement_literals",
        ":type_utils",
        ":value_impl",
        "//tensorflow_federated/python/core/api:computation_types",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "computation_impl",
    srcs = ["computation_impl.py"],
    deps = [
        ":context_stack_base",
        ":function_utils",
        ":type_serialization",
        ":type_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "computation_impl_test",
    size = "small",
    srcs = ["computation_impl_test.py"],
    deps = [
        ":computation_impl",
        ":context_stack_impl",
        ":type_serialization",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/core/api:computation_types",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "computation_wrapper",
    srcs = ["computation_wrapper.py"],
    deps = [
        ":function_utils",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "computation_wrapper_test",
    size = "small",
    srcs = ["computation_wrapper_test.py"],
    deps = [
        ":computation_wrapper",
        ":context_base",
        ":context_stack_impl",
        ":function_utils",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "computation_wrapper_instances",
    srcs = ["computation_wrapper_instances.py"],
    deps = [
        ":computation_impl",
        ":computation_wrapper",
        ":context_stack_impl",
        ":federated_computation_utils",
        ":tensorflow_serialization",
        ":type_utils",
    ],
)

py_test(
    name = "computation_wrapper_instances_test",
    size = "small",
    srcs = ["computation_wrapper_instances_test.py"],
    deps = [
        ":computation_building_blocks",
        ":computation_impl",
        ":computation_wrapper_instances",
        ":placement_literals",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "context_base",
    srcs = ["context_base.py"],
    deps = ["@six_archive//:six"],
)

py_library(
    name = "context_stack_base",
    srcs = ["context_stack_base.py"],
    deps = ["@six_archive//:six"],
)

py_library(
    name = "context_stack_impl",
    srcs = ["context_stack_impl.py"],
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
    deps = [
        ":context_base",
        ":context_stack_impl",
        ":reference_executor",
        "@absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "dtype_utils",
    srcs = ["dtype_utils.py"],
)

py_library(
    name = "federated_computation_context",
    srcs = ["federated_computation_context.py"],
    deps = [
        ":context_base",
        ":context_stack_base",
        ":type_utils",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "@six_archive//:six",
    ],
)

py_test(
    name = "federated_computation_context_test",
    size = "small",
    srcs = ["federated_computation_context_test.py"],
    deps = [
        ":context_stack_impl",
        ":federated_computation_context",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:value_base",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "federated_computation_utils",
    srcs = ["federated_computation_utils.py"],
    deps = [
        ":computation_building_blocks",
        ":context_stack_base",
        ":federated_computation_context",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "@six_archive//:six",
    ],
)

py_test(
    name = "federated_computation_utils_test",
    size = "small",
    srcs = ["federated_computation_utils_test.py"],
    deps = [
        ":context_stack_impl",
        ":federated_computation_utils",
        ":function_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "function_utils",
    srcs = ["function_utils.py"],
    deps = [
        ":context_base",
        ":context_stack_base",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:value_base",
        "@org_tensorflow//tensorflow/python:framework_for_generated_wrappers",
        "@six_archive//:six",
    ],
)

py_test(
    name = "function_utils_test",
    size = "small",
    srcs = ["function_utils_test.py"],
    deps = [
        ":context_base",
        ":context_stack_impl",
        ":function_utils",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python:framework_for_generated_wrappers",
        "@six_archive//:six",
    ],
)

py_library(
    name = "graph_utils",
    srcs = ["graph_utils.py"],
    deps = [
        ":dtype_utils",
        ":type_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "graph_utils_test",
    size = "small",
    srcs = ["graph_utils_test.py"],
    deps = [
        ":graph_utils",
        ":type_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "intrinsic_bodies",
    srcs = ["intrinsic_bodies.py"],
    deps = [
        ":context_stack_base",
        ":intrinsic_defs",
        ":intrinsic_factory",
        ":intrinsic_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "intrinsic_bodies_test",
    size = "small",
    srcs = ["intrinsic_bodies_test.py"],
    deps = [
        ":computation_building_blocks",
        ":computation_impl",
        ":context_stack_impl",
        ":intrinsic_bodies",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "intrinsic_factory",
    srcs = ["intrinsic_factory.py"],
    deps = [
        ":computation_building_blocks",
        ":context_stack_base",
        ":intrinsic_defs",
        ":type_constructors",
        ":type_utils",
        ":value_impl",
        ":value_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:value_base",
        "@six_archive//:six",
    ],
)

py_library(
    name = "intrinsic_defs",
    srcs = ["intrinsic_defs.py"],
    deps = [
        ":type_constructors",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "@six_archive//:six",
    ],
)

py_test(
    name = "intrinsic_defs_test",
    size = "small",
    srcs = ["intrinsic_defs_test.py"],
    deps = [
        ":intrinsic_defs",
        ":type_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "intrinsic_utils",
    srcs = ["intrinsic_utils.py"],
    deps = [
        ":computation_building_blocks",
        ":context_stack_base",
        ":intrinsic_defs",
        ":type_constructors",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "intrinsic_utils_test",
    size = "small",
    srcs = ["intrinsic_utils_test.py"],
    deps = [
        ":context_stack_impl",
        ":intrinsic_utils",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "placement_literals",
    srcs = ["placement_literals.py"],
)

py_test(
    name = "placement_literals_test",
    size = "small",
    srcs = ["placement_literals_test.py"],
    deps = [
        ":placement_literals",
        "@absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "reference_executor",
    srcs = ["reference_executor.py"],
    deps = [
        ":compiler_pipeline",
        ":computation_building_blocks",
        ":computation_impl",
        ":context_base",
        ":dtype_utils",
        ":graph_utils",
        ":intrinsic_defs",
        ":placement_literals",
        ":tensorflow_deserialization",
        ":transformations",
        ":type_constructors",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "reference_executor_test",
    size = "small",
    srcs = ["reference_executor_test.py"],
    deps = [
        ":computation_building_blocks",
        ":computation_impl",
        ":context_stack_impl",
        ":graph_utils",
        ":intrinsic_utils",
        ":reference_executor",
        ":type_constructors",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "tensorflow_deserialization",
    srcs = ["tensorflow_deserialization.py"],
    deps = [
        ":graph_utils",
        ":type_serialization",
        ":type_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "tensorflow_deserialization_test",
    size = "small",
    srcs = ["tensorflow_deserialization_test.py"],
    deps = [
        ":context_stack_impl",
        ":tensorflow_deserialization",
        ":tensorflow_serialization",
        "//tensorflow_federated/python/common_libs:test",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "tensorflow_serialization",
    srcs = ["tensorflow_serialization.py"],
    deps = [
        ":context_stack_base",
        ":graph_utils",
        ":tf_computation_context",
        ":type_serialization",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/tensorflow_libs:graph_keys",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tensorflow_serialization_test",
    size = "small",
    srcs = ["tensorflow_serialization_test.py"],
    deps = [
        ":context_stack_impl",
        ":tensorflow_serialization",
        ":type_serialization",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "tf_computation_context",
    srcs = ["tf_computation_context.py"],
    deps = [
        ":computation_impl",
        ":context_base",
        ":tensorflow_deserialization",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tf_computation_context_test",
    size = "small",
    srcs = ["tf_computation_context_test.py"],
    deps = [
        ":tf_computation_context",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "transformations",
    srcs = ["transformations.py"],
    deps = [
        ":computation_building_blocks",
        ":context_stack_base",
        ":federated_computation_utils",
        ":intrinsic_defs",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "@six_archive//:six",
    ],
)

py_test(
    name = "transformations_test",
    size = "small",
    srcs = ["transformations_test.py"],
    deps = [
        ":computation_building_blocks",
        ":computation_impl",
        ":context_stack_impl",
        ":intrinsic_bodies",
        ":intrinsic_defs",
        ":tensorflow_serialization",
        ":transformations",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "type_constructors",
    srcs = ["type_constructors.py"],
    deps = [
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
    ],
)

py_test(
    name = "type_constructors_test",
    size = "small",
    srcs = ["type_constructors_test.py"],
    deps = [
        ":type_constructors",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "type_serialization",
    srcs = ["type_serialization.py"],
    deps = [
        ":placement_literals",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "type_serialization_test",
    size = "small",
    srcs = ["type_serialization_test.py"],
    deps = [
        ":type_serialization",
        ":type_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "type_utils",
    srcs = ["type_utils.py"],
    deps = [
        ":placement_literals",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
        "//tensorflow_federated/python/core/api:value_base",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "type_utils_test",
    size = "small",
    srcs = ["type_utils_test.py"],
    deps = [
        ":computation_building_blocks",
        ":context_stack_impl",
        ":type_utils",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "value_impl",
    srcs = ["value_impl.py"],
    deps = [
        ":computation_building_blocks",
        ":computation_constructing_utils",
        ":computation_impl",
        ":context_stack_base",
        ":dtype_utils",
        ":function_utils",
        ":graph_utils",
        ":intrinsic_defs",
        ":placement_literals",
        ":tensorflow_serialization",
        ":type_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:value_base",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "value_impl_test",
    size = "small",
    srcs = ["value_impl_test.py"],
    deps = [
        ":computation_building_blocks",
        ":context_stack_impl",
        ":federated_computation_context",
        ":transformations",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:value_base",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "value_utils",
    srcs = ["value_utils.py"],
    deps = [
        ":computation_building_blocks",
        ":intrinsic_defs",
        ":type_utils",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:value_base",
        "@six_archive//:six",
    ],
)

py_test(
    name = "value_utils_test",
    size = "small",
    srcs = ["value_utils_test.py"],
    deps = [
        ":computation_building_blocks",
        ":computation_impl",
        ":context_stack_impl",
        ":federated_computation_context",
        ":transformations",
        ":value_impl",
        ":value_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)
