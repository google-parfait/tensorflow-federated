load("//tensorflow_federated/tools:build_defs.bzl", "cc_cpu_gpu_test")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

licenses(["notice"])

package_group(
    name = "executors_visibility",
    packages = [
        "//tensorflow_federated/cc/core/impl/executor_stacks/...",
        "//tensorflow_federated/cc/core/impl/executors/...",
        "//tensorflow_federated/python/core/impl/executors/...",
    ],
)

package(default_visibility = [":executors_visibility"])

cc_library(
    name = "cardinalities",
    srcs = ["cardinalities.cc"],
    # hdrs = ["cardinalities.h"],
    tags = ["nokokoro"],
)

cc_library(
    name = "composing_executor",
    srcs = ["composing_executor.cc"],
    # hdrs = ["composing_executor.h"],
    tags = ["nokokoro"],
    deps = [
        ":cardinalities",
        ":computations",
        ":executor",
        ":federated_intrinsics",
        ":status_macros",
        ":threading",
        ":value_validation",
    ],
)

cc_test(
    name = "composing_executor_test",
    srcs = ["composing_executor_test.cc"],
    tags = ["nokokoro"],
    deps = [
        ":composing_executor",
        ":computations",
        ":executor_test_base",
        ":federated_intrinsics",
        ":mock_executor",
        ":value_test_utils",
    ],
)

cc_library(
    name = "computations",
    # hdrs = ["computations.h"],
    tags = ["nokokoro"],
)

pybind_extension(
    name = "executor_bindings",
    srcs = ["executor_bindings.cc"],
    tags = ["nokokoro"],
    deps = [
        ":composing_executor",
        ":executor",
        ":federating_executor",
        ":reference_resolving_executor",
        ":remote_executor",
        ":tensorflow_executor",
    ],
)

cc_library(
    name = "executor",
    srcs = ["executor.cc"],
    # hdrs = ["executor.h"],
    tags = ["nokokoro"],
    deps = [":status_macros"],
)

cc_library(
    name = "executor_service",
    srcs = ["executor_service.cc"],
    # hdrs = ["executor_service.h"],
    tags = ["nokokoro"],
    deps = [
        ":cardinalities",
        ":executor",
        ":status_macros",
    ],
)

cc_test(
    name = "executor_service_test",
    srcs = ["executor_service_test.cc"],
    tags = ["nokokoro"],
    deps = [
        ":executor",
        ":executor_service",
        ":mock_executor",
        ":value_test_utils",
    ],
)

cc_library(
    name = "executor_test_base",
    testonly = True,
    srcs = [],
    # hdrs = ["executor_test_base.h"],
    tags = ["nokokoro"],
    deps = [":mock_executor"],
)

cc_library(
    name = "federated_intrinsics",
    srcs = ["federated_intrinsics.cc"],
    # hdrs = ["federated_intrinsics.h"],
    tags = ["nokokoro"],
)

cc_library(
    name = "federating_executor",
    srcs = ["federating_executor.cc"],
    # hdrs = ["federating_executor.h"],
    tags = ["nokokoro"],
    deps = [
        ":cardinalities",
        ":executor",
        ":federated_intrinsics",
        ":status_macros",
        ":threading",
        ":value_validation",
    ],
)

cc_test(
    name = "federating_executor_test",
    srcs = ["federating_executor_test.cc"],
    tags = ["nokokoro"],
    deps = [
        ":executor_test_base",
        ":federating_executor",
        ":mock_executor",
        ":value_test_utils",
    ],
)

cc_library(
    name = "mock_executor",
    testonly = True,
    # hdrs = ["mock_executor.h"],
    tags = ["nokokoro"],
    deps = [":executor"],
)

cc_library(
    name = "reference_resolving_executor",
    srcs = ["reference_resolving_executor.cc"],
    # hdrs = ["reference_resolving_executor.h"],
    tags = ["nokokoro"],
    deps = [
        ":executor",
        ":status_macros",
    ],
)

cc_test(
    name = "reference_resolving_executor_test",
    timeout = "short",
    srcs = ["reference_resolving_executor_test.cc"],
    tags = ["nokokoro"],
    deps = [
        ":executor",
        ":executor_test_base",
        ":mock_executor",
        ":reference_resolving_executor",
        ":value_test_utils",
    ],
)

cc_library(
    name = "remote_executor",
    srcs = ["remote_executor.cc"],
    # hdrs = ["remote_executor.h"],
    tags = ["nokokoro"],
    deps = [
        ":cardinalities",
        ":executor",
        ":status_macros",
        ":threading",
    ],
)

cc_test(
    name = "remote_executor_test",
    timeout = "short",
    srcs = ["remote_executor_test.cc"],
    tags = ["nokokoro"],
    deps = [
        ":executor",
        ":remote_executor",
        ":value_test_utils",
    ],
)

pybind_extension(
    name = "serialization_bindings",
    srcs = ["serialization_bindings.cc"],
    tags = ["nokokoro"],
    deps = [
        ":status_macros",
        ":tensor_serialization",
    ],
)

cc_library(
    name = "status_macros",
    # hdrs = ["status_macros.h"],
    tags = ["nokokoro"],
)

cc_library(
    name = "tensor_serialization",
    srcs = ["tensor_serialization.cc"],
    # hdrs = ["tensor_serialization.h"],
    tags = ["nokokoro"],
    deps = [":status_macros"],
)

cc_library(
    name = "tensorflow_executor",
    srcs = ["tensorflow_executor.cc"],
    # hdrs = ["tensorflow_executor.h"],
    tags = ["nokokoro"],
    deps = [
        ":executor",
        ":status_macros",
        ":tensor_serialization",
        ":threading",
    ],
)

cc_cpu_gpu_test(
    name = "tensorflow_executor_test",
    srcs = ["tensorflow_executor_test.cc"],
    tags = ["nokokoro"],
    deps = [
        ":executor",
        ":status_macros",
        ":tensorflow_executor",
        ":value_test_utils",
    ],
)

cc_library(
    name = "threading",
    srcs = ["threading.cc"],
    # hdrs = ["threading.h"],
    tags = ["nokokoro"],
    deps = [":status_macros"],
)

cc_test(
    name = "threading_test",
    timeout = "short",
    srcs = ["threading_test.cc"],
    tags = ["nokokoro"],
    deps = [":threading"],
)

cc_library(
    name = "value_test_utils",
    testonly = True,
    # hdrs = ["value_test_utils.h"],
    tags = ["nokokoro"],
    deps = [":executor"],
)

cc_library(
    name = "value_validation",
    srcs = ["value_validation.cc"],
    # hdrs = ["value_validation.h"],
    tags = ["nokokoro"],
    deps = [
        ":cardinalities",
        ":federated_intrinsics",
    ],
)

cc_test(
    name = "value_validation_test",
    timeout = "short",
    srcs = ["value_validation_test.cc"],
    tags = ["nokokoro"],
    deps = [
        ":value_test_utils",
        ":value_validation",
    ],
)
