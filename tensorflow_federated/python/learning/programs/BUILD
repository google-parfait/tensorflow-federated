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
        "//tensorflow_federated/python/program:file_program_state_manager",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "evaluation_program_logic_test",
    timeout = "moderate",
    srcs = ["evaluation_program_logic_test.py"],
    deps = [
        ":evaluation_program_logic",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/program:file_program_state_manager",
        "//tensorflow_federated/python/program:native_platform",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "program_logic",
    srcs = ["program_logic.py"],
    deps = [
        ":evaluation_program_logic",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "training_program_logic",
    srcs = ["training_program_logic.py"],
    deps = [
        ":evaluation_program_logic",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "@federated_language//federated_language",
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
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/program:native_platform",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "vizier_program_logic",
    srcs = ["vizier_program_logic.py"],
    deps = [
        ":evaluation_program_logic",
        ":program_logic",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/program:structure_utils",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "vizier_program_logic_test",
    srcs = ["vizier_program_logic_test.py"],
    deps = [
        ":evaluation_program_logic",
        ":vizier_program_logic",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/program:native_platform",
        "@federated_language//federated_language",
    ],
)
