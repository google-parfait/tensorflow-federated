load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":program_packages",
        "//tensorflow_federated/python/analytics:analytics_packages",
        "//tensorflow_federated/python/learning:learning_packages",
        "//tensorflow_federated/python/simulation:simulation_packages",
    ],
)

package_group(
    name = "program_packages",
    packages = ["//tensorflow_federated/python/program/..."],
)

licenses(["notice"])

py_library(
    name = "program",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":client_id_data_source",
        ":dataset_data_source",
        ":file_program_state_manager",
        ":file_release_manager",
        ":tensorboard_release_manager",
    ],
)

py_library(
    name = "client_id_data_source",
    srcs = ["client_id_data_source.py"],
    deps = [
        ":serialization_utils",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "client_id_data_source_test",
    srcs = ["client_id_data_source_test.py"],
    deps = [
        ":client_id_data_source",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "dataset_data_source",
    srcs = ["dataset_data_source.py"],
    deps = [
        ":serialization_utils",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "dataset_data_source_test",
    srcs = ["dataset_data_source_test.py"],
    deps = [
        ":dataset_data_source",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "file_program_state_manager",
    srcs = ["file_program_state_manager.py"],
    deps = [
        ":file_utils",
        ":structure_utils",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "file_program_state_manager_test",
    srcs = ["file_program_state_manager_test.py"],
    deps = [
        ":file_program_state_manager",
        ":file_utils",
        ":program_test_utils",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "file_release_manager",
    srcs = ["file_release_manager.py"],
    deps = [
        ":file_utils",
        ":structure_utils",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "file_release_manager_test",
    srcs = ["file_release_manager_test.py"],
    deps = [
        ":file_release_manager",
        ":file_utils",
        ":program_test_utils",
        ":structure_utils",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "file_utils",
    srcs = ["file_utils.py"],
    deps = [":structure_utils"],
)

py_test(
    name = "file_utils_test",
    srcs = ["file_utils_test.py"],
    deps = [":file_utils"],
)

py_library(
    name = "program_test_utils",
    testonly = True,
    srcs = ["program_test_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "serialization_utils",
    srcs = ["serialization_utils.py"],
    deps = [
        ":structure_utils",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_test(
    name = "serialization_utils_test",
    srcs = ["serialization_utils_test.py"],
    deps = [
        ":program_test_utils",
        ":serialization_utils",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "structure_utils",
    srcs = ["structure_utils.py"],
    deps = ["//tensorflow_federated/python/common_libs:py_typecheck"],
)

py_test(
    name = "structure_utils_test",
    srcs = ["structure_utils_test.py"],
    deps = [
        ":program_test_utils",
        ":structure_utils",
    ],
)

py_library(
    name = "tensorboard_release_manager",
    srcs = ["tensorboard_release_manager.py"],
    deps = [
        ":structure_utils",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "tensorboard_release_manager_test",
    srcs = ["tensorboard_release_manager_test.py"],
    deps = [
        ":program_test_utils",
        ":tensorboard_release_manager",
    ],
)
