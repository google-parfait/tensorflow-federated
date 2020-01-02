package(default_visibility = ["//tensorflow_federated:__subpackages__"])

licenses(["notice"])

# This is the import for other components of TFF that are layered upon the Core API.
py_library(
    name = "api",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":computation_base",
        ":computation_types",
        ":computations",
        ":intrinsics",
        ":placements",
        ":typed_object",
        ":value_base",
        ":values",
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
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
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
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
    ],
)

py_library(
    name = "computations",
    srcs = ["computations.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/wrappers:computation_wrapper_instances"],
)

py_test(
    name = "computations_test",
    size = "small",
    srcs = ["computations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":api",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/impl:executor_stacks",
        "//tensorflow_federated/python/core/utils:test",
    ],
)

py_library(
    name = "intrinsics",
    srcs = ["intrinsics.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/impl:context_stack_impl",
        "//tensorflow_federated/python/core/impl:intrinsic_factory",
    ],
)

py_test(
    name = "intrinsics_test",
    size = "large",
    srcs = ["intrinsics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":api",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/impl:executor_stacks",
        "//tensorflow_federated/python/core/utils:test",
    ],
)

py_library(
    name = "placements",
    srcs = ["placements.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/compiler:placement_literals"],
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
        "//tensorflow_federated/python/core/impl:context_stack_impl",
        "//tensorflow_federated/python/core/impl:value_impl",
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
    ],
)
