package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core/impl:impl_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/impl/tensorflow_contet/...",

        # Impl Dependencies
        "//tensorflow_federated/python/core/impl/executors/...",
        "//tensorflow_federated/python/core/impl/wrappers/...",

        # TODO(b/143772942): This package is temporary and can be removed once
        # all the appropriate modules have been moved out of the `impl` package.
        "//tensorflow_federated/python/core/impl",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "tensorflow_context",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "tensorflow_computation_context",
    srcs = ["tensorflow_computation_context.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl:tensorflow_deserialization",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_test(
    name = "tensorflow_computation_context_test",
    size = "small",
    srcs = ["tensorflow_computation_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_computation_context",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
    ],
)

py_library(
    name = "tensorflow_serialization",
    srcs = ["tensorflow_serialization.py"],
    srcs_version = "PY3",
    deps = [
        ":tensorflow_computation_context",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
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
