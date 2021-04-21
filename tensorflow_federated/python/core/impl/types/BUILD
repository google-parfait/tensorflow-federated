load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core/impl:impl_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/impl/types/...",

        # Impl Dependencies
        "//tensorflow_federated/python/core/impl/compiler/...",
        "//tensorflow_federated/python/core/impl/computation/...",
        "//tensorflow_federated/python/core/impl/context_stack/...",
        "//tensorflow_federated/python/core/impl/executors/...",
        "//tensorflow_federated/python/core/impl/federated_context/...",
        "//tensorflow_federated/python/core/impl/tensorflow_context/...",
        "//tensorflow_federated/python/core/impl/utils/...",
        "//tensorflow_federated/python/core/impl/wrappers/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "types",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "placements",
    srcs = ["placements.py"],
    srcs_version = "PY3",
)

py_test(
    name = "placements_test",
    size = "small",
    srcs = ["placements_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":placements"],
)

py_library(
    name = "type_analysis",
    srcs = ["type_analysis.py"],
    srcs_version = "PY3",
    deps = [
        ":placements",
        ":type_conversions",
        ":type_transformations",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "type_analysis_test",
    size = "small",
    srcs = ["type_analysis_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":placements",
        ":type_analysis",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_library(
    name = "type_conversions",
    srcs = ["type_conversions.py"],
    srcs_version = "PY3",
    deps = [
        ":typed_object",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "type_conversions_test",
    size = "small",
    srcs = ["type_conversions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":placements",
        ":type_conversions",
        ":typed_object",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "type_factory",
    srcs = ["type_factory.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/api:computation_types"],
)

py_test(
    name = "type_factory_test",
    size = "small",
    srcs = ["type_factory_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":type_factory",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_library(
    name = "type_serialization",
    srcs = ["type_serialization.py"],
    srcs_version = "PY3",
    deps = [
        ":placements",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
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
        ":placements",
        ":type_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "type_transformations",
    srcs = ["type_transformations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "type_transformations_test",
    size = "small",
    srcs = ["type_transformations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":placements",
        ":type_transformations",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_library(
    name = "typed_object",
    srcs = ["typed_object.py"],
    srcs_version = "PY3",
)
