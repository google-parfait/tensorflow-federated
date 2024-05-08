load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [":programs_packages"],
)

package_group(
    name = "programs_packages",
    packages = ["//tensorflow_federated/python/learning/programs/..."],
)

licenses(["notice"])

py_library(
    name = "programs",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":evaluation_program_logic",
        ":training_program_logic",
        ":vizier_program_logic",
    ],
)

py_library(
    name = "evaluation_program_logic",
    srcs = ["evaluation_program_logic.py"],
    deps = [
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/program:data_source",
        "//tensorflow_federated/python/program:federated_context",
        "//tensorflow_federated/python/program:file_program_state_manager",
        "//tensorflow_federated/python/program:release_manager",
        "//tensorflow_federated/python/program:value_reference",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_test(
    name = "evaluation_program_logic_test",
    timeout = "moderate",
    srcs = ["evaluation_program_logic_test.py"],
    deps = [
        ":evaluation_program_logic",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/program:data_source",
        "//tensorflow_federated/python/program:federated_context",
        "//tensorflow_federated/python/program:file_program_state_manager",
        "//tensorflow_federated/python/program:native_platform",
        "//tensorflow_federated/python/program:release_manager",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "program_logic",
    srcs = ["program_logic.py"],
    deps = [
        ":evaluation_program_logic",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/program:data_source",
        "//tensorflow_federated/python/program:program_state_manager",
        "//tensorflow_federated/python/program:release_manager",
    ],
)

py_library(
    name = "training_program_logic",
    srcs = ["training_program_logic.py"],
    deps = [
        ":evaluation_program_logic",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/program:data_source",
        "//tensorflow_federated/python/program:federated_context",
        "//tensorflow_federated/python/program:program_state_manager",
        "//tensorflow_federated/python/program:release_manager",
        "//tensorflow_federated/python/program:value_reference",
        "@pypi_absl_py//:pkg",
    ],
)

py_test(
    name = "training_program_logic_test",
    timeout = "moderate",
    srcs = ["training_program_logic_test.py"],
    deps = [
        ":evaluation_program_logic",
        ":program_logic",
        ":training_program_logic",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_test_utils",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/program:data_source",
        "//tensorflow_federated/python/program:federated_context",
        "//tensorflow_federated/python/program:native_platform",
        "//tensorflow_federated/python/program:program_state_manager",
        "//tensorflow_federated/python/program:release_manager",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "vizier_program_logic",
    srcs = ["vizier_program_logic.py"],
    deps = [
        ":evaluation_program_logic",
        ":program_logic",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/program:data_source",
        "//tensorflow_federated/python/program:program_state_manager",
        "//tensorflow_federated/python/program:release_manager",
        "//tensorflow_federated/python/program:structure_utils",
        "//tensorflow_federated/python/program:value_reference",
        "@pypi_google_vizier//:pkg",
    ],
)

py_test(
    name = "vizier_program_logic_test",
    srcs = ["vizier_program_logic_test.py"],
    deps = [
        ":evaluation_program_logic",
        ":vizier_program_logic",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_test_utils",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/program:data_source",
        "//tensorflow_federated/python/program:native_platform",
        "//tensorflow_federated/python/program:program_state_manager",
        "//tensorflow_federated/python/program:release_manager",
        "@pypi_absl_py//:pkg",
        "@pypi_google_vizier//:pkg",
    ],
)
