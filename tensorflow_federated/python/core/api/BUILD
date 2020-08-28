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

        # Namespace Dependencies
        "//tensorflow_federated",

        # TODO(b/143772942): This visibility is circular, it would be clearer if
        # the `api` package could depend on the `impl` package but not the other
        # way around.
        "//tensorflow_federated/python/core/impl/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "api",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow_federated:__pkg__",
        "//tensorflow_federated/tools:__subpackages__",
    ],
)

py_library(
    name = "computation_base",
    srcs = ["computation_base.py"],
    srcs_version = "PY3",
    deps = [":typed_object"],
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_types",
        ":placements",
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
        "//tensorflow_federated/python/core/impl:intrinsic_factory",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
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
        ":value_base",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:test",
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
        "//tensorflow_federated/python/core/impl:value_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
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
