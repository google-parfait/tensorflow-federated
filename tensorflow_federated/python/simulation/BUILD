load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":simulation_packages",
        ":simulation_users",
    ],
)

package_group(
    name = "simulation_packages",
    packages = ["//tensorflow_federated/python/simulation/..."],
)

package_group(
    name = "simulation_users",
    includes = ["//tools:tools_packages"],
)

licenses(["notice"])

py_library(
    name = "simulation",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":iterative_process_compositions",
        ":sampling_utils",
        ":training_loop",
        "//tensorflow_federated/python/simulation/baselines",
        "//tensorflow_federated/python/simulation/datasets",
        "//tensorflow_federated/python/simulation/models",
    ],
)

py_library(
    name = "iterative_process_compositions",
    srcs = ["iterative_process_compositions.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "iterative_process_compositions_test",
    size = "small",
    srcs = ["iterative_process_compositions_test.py"],
    deps = [
        ":iterative_process_compositions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "sampling_utils",
    srcs = ["sampling_utils.py"],
)

py_test(
    name = "sampling_utils_test",
    size = "small",
    srcs = ["sampling_utils_test.py"],
    deps = [":sampling_utils"],
)

py_library(
    name = "training_loop",
    srcs = ["training_loop.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "training_loop_test",
    size = "medium",
    srcs = ["training_loop_test.py"],
    deps = [
        ":training_loop",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "@federated_language//federated_language",
    ],
)
