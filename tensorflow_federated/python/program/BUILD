load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":program_packages",
    "//tensorflow_federated/python/analytics:analytics_packages",
    "//tensorflow_federated/python/learning:learning_packages",
    "//tensorflow_federated/python/simulation:simulation_packages",
])

package_group(
    name = "program_packages",
    packages = ["//tensorflow_federated/python/program/..."],
)

licenses(["notice"])

py_library(
    name = "program",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":data_source",
        ":federated_context",
        ":file_program_state_manager",
        ":file_release_manager",
        ":logging_release_manager",
        ":memory_release_manager",
        ":native_platform",
        ":program_state_manager",
        ":release_manager",
        ":tensorboard_release_manager",
        ":value_reference",
    ],
)

py_library(
    name = "data_source",
    srcs = ["data_source.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/types:computation_types"],
)

py_library(
    name = "federated_context",
    srcs = ["federated_context.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:get_context_stack",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "federated_context_test",
    srcs = ["federated_context_test.py"],
    deps = [
        ":federated_context",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "file_program_state_manager",
    srcs = ["file_program_state_manager.py"],
    srcs_version = "PY3",
    deps = [
        ":file_utils",
        ":program_state_manager",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@absl_py//absl/logging",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "file_program_state_manager_test",
    srcs = ["file_program_state_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":file_program_state_manager",
        ":file_utils",
        ":program_state_manager",
        ":program_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "file_release_manager",
    srcs = ["file_release_manager.py"],
    srcs_version = "PY3",
    deps = [
        ":file_utils",
        ":release_manager",
        ":structure_utils",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "file_release_manager_test",
    srcs = ["file_release_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":file_release_manager",
        ":file_utils",
        ":program_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "file_utils",
    srcs = ["file_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "file_utils_test",
    srcs = ["file_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":file_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "logging_release_manager",
    srcs = ["logging_release_manager.py"],
    srcs_version = "PY3",
    deps = [
        ":release_manager",
        ":value_reference",
        "@absl_py//absl/logging",
    ],
)

py_test(
    name = "logging_release_manager_test",
    srcs = ["logging_release_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":logging_release_manager",
        ":program_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "memory_release_manager",
    srcs = ["memory_release_manager.py"],
    srcs_version = "PY3",
    deps = [
        ":release_manager",
        ":value_reference",
    ],
)

py_test(
    name = "memory_release_manager_test",
    srcs = ["memory_release_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":memory_release_manager",
        ":program_test_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "native_platform",
    srcs = ["native_platform.py"],
    srcs_version = "PY3",
    deps = [
        ":data_source",
        ":federated_context",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:async_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "native_platform_test",
    srcs = ["native_platform_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_context",
        ":native_platform",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "program_state_manager",
    srcs = ["program_state_manager.py"],
    srcs_version = "PY3",
)

py_test(
    name = "program_state_manager_test",
    srcs = ["program_state_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":program_state_manager",
        "@absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "program_test_utils",
    testonly = True,
    srcs = ["program_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":value_reference",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "release_manager",
    srcs = ["release_manager.py"],
    srcs_version = "PY3",
)

py_library(
    name = "structure_utils",
    srcs = ["structure_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "structure_utils_test",
    srcs = ["structure_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":program_test_utils",
        ":structure_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "tensorboard_release_manager",
    srcs = ["tensorboard_release_manager.py"],
    srcs_version = "PY3",
    deps = [
        ":release_manager",
        ":structure_utils",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tensorboard_release_manager_test",
    srcs = ["tensorboard_release_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":program_test_utils",
        ":tensorboard_release_manager",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "value_reference",
    srcs = ["value_reference.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "value_reference_test",
    srcs = ["value_reference_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":program_test_utils",
        ":value_reference",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
