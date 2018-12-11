# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//tensorflow_federated/python/core/api:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "compiler_pipeline",
    srcs = ["compiler_pipeline.py"],
    deps = [
        ":computation_building_blocks",
        ":intrinsic_bodies",
        ":intrinsic_defs",
        ":transformations",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
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
        ":intrinsic_defs",
        ":transformations",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:types",
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
        "//tensorflow_federated/python/core/api:types",
        "@six_archive//:six",
    ],
)

py_test(
    name = "computation_building_blocks_test",
    size = "small",
    srcs = ["computation_building_blocks_test.py"],
    deps = [
        ":computation_building_blocks",
        ":tensorflow_serialization",
        ":type_serialization",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:types",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "computation_building_utils",
    srcs = ["computation_building_utils.py"],
    deps = [
        ":computation_building_blocks",
        ":context_stack",
        ":federated_computation_context",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:types",
    ],
)

py_test(
    name = "computation_building_utils_test",
    srcs = ["computation_building_utils_test.py"],
    deps = [
        ":computation_building_blocks",
        ":computation_building_utils",
        ":func_utils",
        ":value_impl",
        "//tensorflow_federated/python/core/api:types",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "computation_impl",
    srcs = ["computation_impl.py"],
    deps = [
        ":context_stack",
        ":func_utils",
        ":type_serialization",
        ":type_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "computation_impl_test",
    size = "small",
    srcs = ["computation_impl_test.py"],
    deps = [
        ":computation_impl",
        ":type_serialization",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "computation_wrapper",
    srcs = ["computation_wrapper.py"],
    deps = [
        ":func_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:types",
    ],
)

py_library(
    name = "computation_wrapper_instances",
    srcs = ["computation_wrapper_instances.py"],
    deps = [
        ":computation_building_utils",
        ":computation_impl",
        ":computation_wrapper",
    ],
)

py_test(
    name = "computation_wrapper_instances_test",
    size = "small",
    srcs = ["computation_wrapper_instances_test.py"],
    deps = [
        ":computation_wrapper_instances",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "computation_wrapper_test",
    size = "small",
    srcs = ["computation_wrapper_test.py"],
    deps = [
        ":computation_wrapper",
        ":type_utils",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "context_base",
    srcs = ["context_base.py"],
    deps = ["@six_archive//:six"],
)

py_library(
    name = "context_stack",
    srcs = ["context_stack.py"],
    deps = [
        ":context_base",
        ":default_context",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "context_stack_test",
    size = "small",
    srcs = ["context_stack_test.py"],
    deps = [
        ":context_base",
        ":context_stack",
        ":default_context",
        "@absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "default_context",
    srcs = ["default_context.py"],
    deps = [":context_base"],
)

py_library(
    name = "federated_computation_context",
    srcs = ["federated_computation_context.py"],
    deps = [
        ":context_base",
        ":value_impl",
    ],
)

py_test(
    name = "federated_computation_context_test",
    size = "small",
    srcs = ["federated_computation_context_test.py"],
    deps = [
        ":federated_computation_context",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:value_base",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "func_utils",
    srcs = ["func_utils.py"],
    deps = [
        ":type_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:types",
        "@six_archive//:six",
    ],
)

py_test(
    name = "func_utils_test",
    size = "small",
    srcs = ["func_utils_test.py"],
    deps = [
        ":func_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:types",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "graph_utils",
    srcs = ["graph_utils.py"],
    deps = [
        ":type_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:types",
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
        ":test_utils",
        ":type_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "intrinsic_bodies",
    srcs = ["intrinsic_bodies.py"],
    deps = [
        ":intrinsic_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
    ],
)

py_test(
    name = "intrinsic_bodies_test",
    size = "small",
    srcs = ["intrinsic_bodies_test.py"],
    deps = [
        ":computation_building_blocks",
        ":computation_impl",
        ":intrinsic_bodies",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:types",
        "@absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "intrinsic_defs",
    srcs = ["intrinsic_defs.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:types",
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
        ":intrinsic_defs",
        ":type_utils",
        ":value_impl",
        "//tensorflow_federated/python/core/api:types",
    ],
)

py_test(
    name = "intrinsic_utils_test",
    size = "small",
    srcs = ["intrinsic_utils_test.py"],
    deps = [
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
    name = "tensorflow_deserialization",
    srcs = ["tensorflow_deserialization.py"],
    deps = [
        ":graph_utils",
        ":type_serialization",
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
        ":tensorflow_deserialization",
        ":tensorflow_serialization",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "tensorflow_serialization",
    srcs = ["tensorflow_serialization.py"],
    deps = [
        ":context_stack",
        ":graph_utils",
        ":tf_computation_context",
        ":type_serialization",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tensorflow_serialization_test",
    size = "small",
    srcs = ["tensorflow_serialization_test.py"],
    deps = [
        ":tensorflow_serialization",
        ":type_serialization",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
)

py_test(
    name = "test_utils_test",
    size = "small",
    srcs = ["test_utils_test.py"],
    deps = [
        ":test_utils",
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
        ":context_stack",
        ":tf_computation_context",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:types",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "transformations",
    srcs = ["transformations.py"],
    deps = [
        ":computation_building_blocks",
        ":computation_impl",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:types",
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
        ":intrinsic_bodies",
        ":intrinsic_defs",
        ":transformations",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:types",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "type_serialization",
    srcs = ["type_serialization.py"],
    deps = [
        ":placement_literals",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "type_serialization_test",
    size = "small",
    srcs = ["type_serialization_test.py"],
    deps = [
        ":type_serialization",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "type_utils",
    srcs = ["type_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:types",
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
        ":test_utils",
        ":type_utils",
        ":value_impl",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:types",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "value_impl",
    srcs = ["value_impl.py"],
    deps = [
        ":computation_building_blocks",
        ":computation_impl",
        ":func_utils",
        ":intrinsic_defs",
        ":placement_literals",
        ":tensorflow_serialization",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:types",
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
        ":value_impl",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:types",
        "//tensorflow_federated/python/core/api:value_base",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
