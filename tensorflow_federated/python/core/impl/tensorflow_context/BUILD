load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":tensorflow_context_packages",
    "//tensorflow_federated/python/core/impl:impl_visibility",
    "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
    "//tensorflow_federated/python/core/impl/executor_stacks:executor_stacks_packages",
    "//tensorflow_federated/python/core/impl/executors:executors_packages",
])

package_group(
    name = "tensorflow_context_packages",
    packages = ["//tensorflow_federated/python/core/impl/tensorflow_context/..."],
)

licenses(["notice"])

py_library(
    name = "tensorflow_context",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "tensorflow_computation",
    srcs = ["tensorflow_computation.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/computation:computation_wrapper",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_serialization",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "tensorflow_computation_test",
    size = "small",
    srcs = ["tensorflow_computation_test.py"],
    args = [
        "--golden",
        "$(location tensorflow_computation_test_goldens/non_callable_non_type_traceback.expected)",
        "--golden",
        "$(location tensorflow_computation_test_goldens/returns_type_container_mismatch_traceback.expected)",
        "--golden",
        "$(location tensorflow_computation_test_goldens/tensorflow_computation_traceback.expected)",
    ],
    data = [
        "tensorflow_computation_test_goldens/non_callable_non_type_traceback.expected",
        "tensorflow_computation_test_goldens/returns_type_container_mismatch_traceback.expected",
        "tensorflow_computation_test_goldens/tensorflow_computation_traceback.expected",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_computation",
        "//tensorflow_federated/python/common_libs:golden",
        "//tensorflow_federated/python/core/impl/computation:computation_wrapper",
        "//tensorflow_federated/python/core/impl/context_stack:get_context_stack",
        "//tensorflow_federated/python/core/impl/context_stack:runtime_error_context",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "tensorflow_computation_context",
    srcs = ["tensorflow_computation_context.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
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
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
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
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "//tensorflow_federated/python/tensorflow_libs:variable_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
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
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensorflow_test_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
