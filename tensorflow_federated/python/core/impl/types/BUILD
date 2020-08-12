package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core/impl:impl_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/impl/types/...",

        # Impl Dependencies
        "//tensorflow_federated/python/core/impl/compiler/...",
        "//tensorflow_federated/python/core/impl/context_stack/...",
        "//tensorflow_federated/python/core/impl/executors/...",
        "//tensorflow_federated/python/core/impl/federated_context/...",
        "//tensorflow_federated/python/core/impl/tensorflow_context/...",
        "//tensorflow_federated/python/core/impl/utils/...",
        "//tensorflow_federated/python/core/impl/wrappers/...",

        # TODO(b/143772942): This package is temporary and can be removed once
        # all the appropriate modules have been moved out of the `impl` package.
        "//tensorflow_federated/python/core/impl",
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
    name = "type_analysis",
    srcs = ["type_analysis.py"],
    srcs_version = "PY3",
    deps = [
        ":placement_literals",
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
        ":placement_literals",
        ":type_analysis",
        ":type_factory",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_library(
    name = "type_conversions",
    srcs = ["type_conversions.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
    ],
)

py_test(
    name = "type_conversions_test",
    size = "small",
    srcs = ["type_conversions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":placement_literals",
        ":type_conversions",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
    ],
)

py_library(
    name = "type_factory",
    srcs = ["type_factory.py"],
    srcs_version = "PY3",
    deps = [
        ":placement_literals",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "type_factory_test",
    size = "small",
    srcs = ["type_factory_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":placement_literals",
        ":type_factory",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_library(
    name = "type_serialization",
    srcs = ["type_serialization.py"],
    srcs_version = "PY3",
    deps = [
        ":placement_literals",
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
        ":placement_literals",
        ":type_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
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
        ":placement_literals",
        ":type_transformations",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)
