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
    includes = ["//tensorflow_federated/tools:tools_packages"],
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
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/learning/templates:learning_process",
    ],
)

py_test(
    name = "iterative_process_compositions_test",
    size = "small",
    srcs = ["iterative_process_compositions_test.py"],
    deps = [
        ":iterative_process_compositions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "sampling_utils",
    srcs = ["sampling_utils.py"],
    deps = ["@pypi_numpy//:pkg"],
)

py_test(
    name = "sampling_utils_test",
    size = "small",
    srcs = ["sampling_utils_test.py"],
    deps = [
        ":sampling_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "training_loop",
    srcs = ["training_loop.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/program:program_state_manager",
        "//tensorflow_federated/python/program:release_manager",
        "@pypi_absl_py//:pkg",
    ],
)

py_test(
    name = "training_loop_test",
    size = "medium",
    srcs = ["training_loop_test.py"],
    deps = [
        ":training_loop",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)
