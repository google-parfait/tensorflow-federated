# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//tensorflow_federated/python/core:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# This is the import for other components of TFF that are layered upon the Core API.
py_library(
    name = "api",
    srcs = ["__init__.py"],
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
    deps = [
        ":typed_object",
        "@six_archive//:six",
    ],
)

py_library(
    name = "computation_types",
    srcs = ["computation_types.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl:placement_literals",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "computation_types_test",
    size = "small",
    srcs = ["computation_types_test.py"],
    default_python_version = "PY3",
    deps = [
        ":computation_types",
        ":placements",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "computations",
    srcs = ["computations.py"],
    deps = [
        "//tensorflow_federated/python/core/impl:computation_wrapper_instances",
    ],
)

py_test(
    name = "computations_test",
    size = "small",
    srcs = ["computations_test.py"],
    default_python_version = "PY3",
    deps = [
        ":computation_types",
        ":computations",
        ":value_base",
        "//tensorflow_federated/python/common_libs:test_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "intrinsics",
    srcs = ["intrinsics.py"],
    deps = [
        "//tensorflow_federated/python/core/impl:context_stack_impl",
        "//tensorflow_federated/python/core/impl:intrinsic_factory",
    ],
)

py_test(
    name = "intrinsics_test",
    size = "large",
    srcs = ["intrinsics_test.py"],
    default_python_version = "PY3",
    deps = [
        ":computation_types",
        ":computations",
        ":intrinsics",
        ":placements",
        "//tensorflow_federated/python/core/impl:context_stack_impl",
        "//tensorflow_federated/python/core/impl:federated_computation_context",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "placements",
    srcs = ["placements.py"],
    deps = [
        "//tensorflow_federated/python/core/impl:placement_literals",
    ],
)

py_library(
    name = "typed_object",
    srcs = ["typed_object.py"],
    deps = ["@six_archive//:six"],
)

py_library(
    name = "value_base",
    srcs = ["value_base.py"],
    deps = [
        ":typed_object",
        "@six_archive//:six",
    ],
)

py_library(
    name = "values",
    srcs = ["values.py"],
    deps = [
        "//tensorflow_federated/python/core/impl:context_stack_impl",
        "//tensorflow_federated/python/core/impl:value_impl",
    ],
)

py_test(
    name = "values_test",
    size = "small",
    srcs = ["values_test.py"],
    default_python_version = "PY3",
    deps = [
        ":value_base",
        ":values",
        "@absl_py//absl/testing:absltest",
    ],
)
