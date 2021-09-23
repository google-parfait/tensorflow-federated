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
        ":file_program_state_manager",
        ":program_state_manager",
        ":release_manager",
    ],
)

py_library(
    name = "file_program_state_manager",
    srcs = ["file_program_state_manager.py"],
    srcs_version = "PY3",
    deps = [
        ":program_state_manager",
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
    ],
)

py_library(
    name = "logging_release_manager",
    srcs = ["logging_release_manager.py"],
    srcs_version = "PY3",
    deps = [
        ":release_manager",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "logging_release_manager_test",
    srcs = ["logging_release_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":logging_release_manager"],
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
