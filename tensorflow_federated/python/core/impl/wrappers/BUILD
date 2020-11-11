load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core/impl:impl_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/impl/wrappers/...",

        # TODO(b/143772942): This package is temporary and can be removed once
        # all the appropriate modules have been moved out of the `impl` package.
        "//tensorflow_federated/python/core/impl",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "wrappers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "computation_wrapper",
    srcs = ["computation_wrapper.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/utils:function_utils",
        "//tensorflow_federated/python/tensorflow_libs:function",
    ],
)

py_test(
    name = "computation_wrapper_test",
    size = "small",
    srcs = ["computation_wrapper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_wrapper",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/utils:function_utils",
    ],
)

py_library(
    name = "computation_wrapper_instances",
    srcs = ["computation_wrapper_instances.py"],
    srcs_version = "PY3",
    deps = [
        ":computation_wrapper",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation_utils",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_serialization",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_test(
    name = "computation_wrapper_instances_test",
    size = "small",
    srcs = ["computation_wrapper_instances_test.py"],
    args = [
        "--golden",
        "$(location computation_wrapper_instances_test_goldens/tensorflow_wrapper_traceback.expected)",
        "--golden",
        "$(location computation_wrapper_instances_test_goldens/federated_computation_wrapper_traceback.expected)",
        "--golden",
        "$(location computation_wrapper_instances_test_goldens/returns_type_container_mismatch_traceback.expected)",
    ],
    data = [
        "computation_wrapper_instances_test_goldens/federated_computation_wrapper_traceback.expected",
        "computation_wrapper_instances_test_goldens/returns_type_container_mismatch_traceback.expected",
        "computation_wrapper_instances_test_goldens/tensorflow_wrapper_traceback.expected",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_wrapper_instances",
        "//tensorflow_federated/python/common_libs:golden",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
    ],
)
