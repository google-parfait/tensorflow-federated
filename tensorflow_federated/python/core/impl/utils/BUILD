package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core/impl:impl_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/impl/utils/...",

        # Impl Dependencies
        "//tensorflow_federated/python/core/impl/compiler/...",
        "//tensorflow_federated/python/core/impl/executors/...",
        "//tensorflow_federated/python/core/impl/federated_context/...",
        "//tensorflow_federated/python/core/impl/tensorflow_context/...",
        "//tensorflow_federated/python/core/impl/wrappers/...",

        # TODO(b/143772942): This package is temporary and can be removed once
        # all the appropriate modules have been moved out of the `impl` package.
        "//tensorflow_federated/python/core/impl",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "function_utils",
    srcs = ["function_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/tensorflow_libs:function",
    ],
)

py_test(
    name = "function_utils_test",
    size = "small",
    srcs = ["function_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":function_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_library(
    name = "tensorflow_utils",
    srcs = ["tensorflow_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":function_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_test(
    name = "tensorflow_utils_test",
    size = "small",
    srcs = ["tensorflow_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)
