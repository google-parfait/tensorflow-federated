load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":program_packages",
    "//tensorflow_federated/python/analytics:analytics_packages",
    "//tensorflow_federated/python/cloud:cloud_packages",
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
        ":file_program_state_manager",
        ":file_release_manager",
        ":logging_release_manager",
        ":memory_release_manager",
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
    name = "file_program_state_manager",
    srcs = ["file_program_state_manager.py"],
    srcs_version = "PY3",
    deps = [
        ":file_utils",
        ":program_state_manager",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "file_program_state_manager_test",
    srcs = ["file_program_state_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":file_program_state_manager",
        ":program_state_manager",
        ":test_utils",
    ],
)

py_library(
    name = "file_release_manager",
    srcs = ["file_release_manager.py"],
    deps = [
        ":file_utils",
        ":release_manager",
        ":structure_utils",
        ":value_reference",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "file_release_manager_test",
    srcs = ["file_release_manager_test.py"],
    deps = [
        ":file_release_manager",
        ":test_utils",
    ],
)

py_library(
    name = "file_utils",
    srcs = ["file_utils.py"],
    deps = ["//tensorflow_federated/python/common_libs:py_typecheck"],
)

py_test(
    name = "file_utils_test",
    srcs = ["file_utils_test.py"],
    deps = [":file_utils"],
)

py_library(
    name = "logging_release_manager",
    srcs = ["logging_release_manager.py"],
    srcs_version = "PY3",
    deps = [
        ":release_manager",
        ":value_reference",
    ],
)

py_test(
    name = "logging_release_manager_test",
    srcs = ["logging_release_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":logging_release_manager",
        ":test_utils",
    ],
)

py_library(
    name = "memory_release_manager",
    srcs = ["memory_release_manager.py"],
    deps = [
        ":release_manager",
        ":value_reference",
    ],
)

py_test(
    name = "memory_release_manager_test",
    srcs = ["memory_release_manager_test.py"],
    deps = [
        ":memory_release_manager",
        ":test_utils",
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
    deps = [":program_state_manager"],
)

py_library(
    name = "release_manager",
    srcs = ["release_manager.py"],
    srcs_version = "PY3",
)

py_library(
    name = "structure_utils",
    srcs = ["structure_utils.py"],
)

py_test(
    name = "structure_utils_test",
    srcs = ["structure_utils_test.py"],
    deps = [
        ":structure_utils",
        ":test_utils",
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
    ],
)

py_test(
    name = "tensorboard_release_manager_test",
    srcs = ["tensorboard_release_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorboard_release_manager",
        ":test_utils",
    ],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    deps = [
        ":value_reference",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "value_reference",
    srcs = ["value_reference.py"],
    deps = ["//tensorflow_federated/python/core/impl/types:typed_object"],
)

py_test(
    name = "value_reference_test",
    srcs = ["value_reference_test.py"],
    deps = [
        ":test_utils",
        ":value_reference",
    ],
)
