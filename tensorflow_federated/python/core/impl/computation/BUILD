load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":computation_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
        "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
        "//tensorflow_federated/python/core/impl/executors:executors_packages",
        "//tensorflow_federated/python/core/impl/federated_context:federated_context_packages",
    ],
)

package_group(
    name = "computation_packages",
    packages = ["//tensorflow_federated/python/core/impl/computation/..."],
)

licenses(["notice"])

py_library(
    name = "computation",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "computation_base",
    srcs = ["computation_base.py"],
    deps = [
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_library(
    name = "computation_impl",
    srcs = ["computation_impl.py"],
    deps = [
        ":computation_base",
        ":function_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_test(
    name = "computation_impl_test",
    size = "small",
    srcs = ["computation_impl_test.py"],
    deps = [
        ":computation_impl",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "computation_serialization",
    srcs = ["computation_serialization.py"],
    deps = [
        ":computation_base",
        ":computation_impl",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
    ],
)

py_test(
    name = "computation_serialization_test",
    size = "small",
    srcs = ["computation_serialization_test.py"],
    deps = [
        ":computation_base",
        ":computation_impl",
        ":computation_serialization",
        "//tensorflow_federated/python/core/impl/compiler:computation_factory",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "computation_wrapper",
    srcs = ["computation_wrapper.py"],
    deps = [
        ":computation_base",
        ":computation_impl",
        ":polymorphic_computation",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_test(
    name = "computation_wrapper_test",
    size = "small",
    srcs = ["computation_wrapper_test.py"],
    deps = [
        ":computation_impl",
        ":computation_wrapper",
        ":function_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "function_utils",
    srcs = ["function_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "function_utils_test",
    size = "small",
    srcs = ["function_utils_test.py"],
    deps = [
        ":function_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "polymorphic_computation",
    srcs = ["polymorphic_computation.py"],
    deps = [
        ":computation_impl",
        ":function_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_test(
    name = "polymorphic_computation_test",
    srcs = ["polymorphic_computation_test.py"],
    deps = [
        ":computation_impl",
        ":polymorphic_computation",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)
