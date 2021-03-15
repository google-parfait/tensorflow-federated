load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core:core_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/api/...",

        # Core Dependencies
        "//tensorflow_federated/python/core/backends/...",
        "//tensorflow_federated/python/core/templates/...",
        "//tensorflow_federated/python/core/utils/...",
        "//tensorflow_federated/python/core/test/...",
        "//tensorflow_federated/python/core/types/...",

        # Namespace Dependencies
        "//tensorflow_federated",

        # TODO(b/181569610): Clean up circular dependency between the `api` and
        # `impl` packages.
        "//tensorflow_federated/python/core/impl/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "api",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "computation_base",
    srcs = ["computation_base.py"],
    srcs_version = "PY3",
    deps = [
        ":typed_object",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
    ],
)

py_library(
    name = "computation_types",
    srcs = ["computation_types.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_test(
    name = "computation_types_test",
    size = "small",
    srcs = ["computation_types_test.py"],
    args = [
        "--golden",
        "$(location computation_types_test_goldens/constructor_argument_normalization_error.expected)",
        "--golden",
        "$(location computation_types_test_goldens/container_types_full_repr.expected)",
        "--golden",
        "$(location computation_types_test_goldens/long_formatted_with_diff.expected)",
        "--golden",
        "$(location computation_types_test_goldens/short_compact_repr.expected)",
    ],
    data = [
        "computation_types_test_goldens/constructor_argument_normalization_error.expected",
        "computation_types_test_goldens/container_types_full_repr.expected",
        "computation_types_test_goldens/long_formatted_with_diff.expected",
        "computation_types_test_goldens/short_compact_repr.expected",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_types",
        ":placements",
        "//tensorflow_federated/python/common_libs:golden",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_library(
    name = "computations",
    srcs = ["computations.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/wrappers:computation_wrapper_instances"],
)

py_library(
    name = "intrinsics",
    srcs = ["intrinsics.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsic_factory",
    ],
)

py_test(
    name = "intrinsics_test",
    size = "large",
    srcs = ["intrinsics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_types",
        ":computations",
        ":intrinsics",
        ":placements",
        ":test_case",
        ":value_base",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
    ],
)

py_library(
    name = "placements",
    srcs = ["placements.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/types:placement_literals"],
)

py_library(
    name = "test_case",
    srcs = ["test_case.py"],
    srcs_version = "PY3",
    deps = [":computation_types"],
)

py_test(
    name = "test_case_test",
    size = "small",
    srcs = ["test_case_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_types",
        ":test_case",
    ],
)

py_library(
    name = "typed_object",
    srcs = ["typed_object.py"],
    srcs_version = "PY3",
)

py_library(
    name = "value_base",
    srcs = ["value_base.py"],
    srcs_version = "PY3",
    deps = [":typed_object"],
)

py_library(
    name = "values",
    srcs = ["values.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/federated_context:value_impl",
    ],
)

py_test(
    name = "values_test",
    size = "small",
    srcs = ["values_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":value_base",
        ":values",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation_context",
    ],
)
