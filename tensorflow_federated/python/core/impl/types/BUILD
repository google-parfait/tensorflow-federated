load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":types_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
        "//tensorflow_federated/python/core/impl/compiler:compiler_packages",
        "//tensorflow_federated/python/core/impl/computation:computation_packages",
        "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_stacks_packages",
        "//tensorflow_federated/python/core/impl/executors:executors_packages",
        "//tensorflow_federated/python/core/impl/federated_context:federated_context_packages",
        "//tensorflow_federated/python/core/impl/utils:utils_packages",
    ],
)

package_group(
    name = "types_packages",
    packages = ["//tensorflow_federated/python/core/impl/types/..."],
)

licenses(["notice"])

py_library(
    name = "types",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":array_shape",
        ":computation_types",
        ":type_analysis",
        ":type_conversions",
        ":type_serialization",
    ],
)

py_library(
    name = "array_shape",
    srcs = ["array_shape.py"],
    deps = [
        "//tensorflow_federated/proto/v0:array_py_pb2",
        "//tensorflow_federated/proto/v0:data_type_py_pb2",
    ],
)

py_test(
    name = "array_shape_test",
    srcs = ["array_shape_test.py"],
    deps = [
        ":array_shape",
        "//tensorflow_federated/proto/v0:array_py_pb2",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "computation_types",
    srcs = ["computation_types.py"],
    deps = [
        ":array_shape",
        ":placements",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "@pypi_attrs//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
        "@pypi_typing_extensions//:pkg",
    ],
)

py_test(
    name = "computation_types_test",
    size = "small",
    srcs = ["computation_types_test.py"],
    args = [
        "--golden",
        "$(location computation_types_test_goldens/container_types_full_repr.expected)",
        "--golden",
        "$(location computation_types_test_goldens/long_formatted_with_diff.expected)",
        "--golden",
        "$(location computation_types_test_goldens/short_compact_repr.expected)",
    ],
    data = [
        "computation_types_test_goldens/container_types_full_repr.expected",
        "computation_types_test_goldens/long_formatted_with_diff.expected",
        "computation_types_test_goldens/short_compact_repr.expected",
    ],
    deps = [
        ":computation_types",
        ":placements",
        "//tensorflow_federated/python/common_libs:golden",
        "//tensorflow_federated/python/common_libs:structure",
        "@pypi_absl_py//:pkg",
        "@pypi_attrs//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "dtype_utils",
    srcs = ["dtype_utils.py"],
    deps = [
        "//tensorflow_federated/proto/v0:data_type_py_pb2",
        "@pypi_numpy//:pkg",
    ],
)

py_test(
    name = "dtype_utils_test",
    srcs = ["dtype_utils_test.py"],
    deps = [
        ":dtype_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "placements",
    srcs = ["placements.py"],
)

py_test(
    name = "placements_test",
    size = "small",
    srcs = ["placements_test.py"],
    deps = [
        ":placements",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "type_analysis",
    srcs = ["type_analysis.py"],
    deps = [
        ":array_shape",
        ":computation_types",
        ":placements",
        ":type_conversions",
        ":type_transformations",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "@pypi_numpy//:pkg",
    ],
)

py_test(
    name = "type_analysis_test",
    size = "small",
    srcs = ["type_analysis_test.py"],
    deps = [
        ":computation_types",
        ":placements",
        ":type_analysis",
        "//tensorflow_federated/python/common_libs:structure",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "type_conversions",
    srcs = ["type_conversions.py"],
    deps = [
        ":computation_types",
        ":dtype_utils",
        ":typed_object",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "@pypi_attrs//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "type_conversions_test",
    size = "small",
    srcs = ["type_conversions_test.py"],
    deps = [
        ":computation_types",
        ":placements",
        ":type_conversions",
        ":type_test_utils",
        ":typed_object",
        "//tensorflow_federated/python/common_libs:structure",
        "@pypi_absl_py//:pkg",
        "@pypi_attrs//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "type_factory",
    srcs = ["type_factory.py"],
    deps = [":computation_types"],
)

py_test(
    name = "type_factory_test",
    size = "small",
    srcs = ["type_factory_test.py"],
    deps = [
        ":computation_types",
        ":type_factory",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "type_serialization",
    srcs = ["type_serialization.py"],
    deps = [
        ":array_shape",
        ":computation_types",
        ":dtype_utils",
        ":placements",
        "//tensorflow_federated/proto/v0:array_py_pb2",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_test(
    name = "type_serialization_test",
    size = "small",
    srcs = ["type_serialization_test.py"],
    deps = [
        ":array_shape",
        ":computation_types",
        ":dtype_utils",
        ":placements",
        ":type_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/proto/v0:data_type_py_pb2",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "type_test_utils",
    srcs = ["type_test_utils.py"],
    deps = [":computation_types"],
)

py_library(
    name = "type_transformations",
    srcs = ["type_transformations.py"],
    deps = [
        ":computation_types",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_test(
    name = "type_transformations_test",
    size = "small",
    srcs = ["type_transformations_test.py"],
    deps = [
        ":computation_types",
        ":placements",
        ":type_transformations",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "typed_object",
    srcs = ["typed_object.py"],
    deps = [":computation_types"],
)
