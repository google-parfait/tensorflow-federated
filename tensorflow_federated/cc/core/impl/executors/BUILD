load("//tensorflow_federated/tools:build_defs.bzl", "cc_cpu_gpu_test")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = [
    ":executors_packages",
    "//tensorflow_federated/cc/core/impl:impl_visibility",
    "//tensorflow_federated/cc/core/impl/executor_stacks:executor_stacks_packages",
])

package_group(
    name = "executors_packages",
    packages = [
        "//tensorflow_federated/cc/core/impl/executors/...",
        "//tensorflow_federated/python/core/impl/executors/...",
    ],
)

licenses(["notice"])

cc_library(
    name = "cardinalities",
    srcs = ["cardinalities.cc"],
    hdrs = ["cardinalities.h"],
    tags = ["nokokoro"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "composing_executor",
    srcs = ["composing_executor.cc"],
    hdrs = ["composing_executor.h"],
    tags = ["nokokoro"],
    deps = [
        ":cardinalities",
        ":computations",
        ":executor",
        ":federated_intrinsics",
        ":status_macros",
        ":threading",
        ":value_validation",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_protobuf//:protobuf",
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
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "computations",
    hdrs = ["computations.h"],
    tags = ["nokokoro"],
    deps = ["//tensorflow_federated/proto/v0:computation_cc_proto"],
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
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:wrapped_proto_caster",
    ],
)

cc_library(
    name = "executor",
    srcs = ["executor.cc"],
    hdrs = ["executor.h"],
    tags = ["nokokoro"],
    deps = [
        ":status_macros",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "executor_service",
    srcs = ["executor_service.cc"],
    hdrs = ["executor_service.h"],
    tags = ["nokokoro"],
    deps = [
        ":cardinalities",
        ":executor",
        ":status_macros",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
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
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "executor_test_base",
    testonly = True,
    srcs = [],
    hdrs = ["executor_test_base.h"],
    tags = ["nokokoro"],
    deps = [
        ":mock_executor",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "federated_intrinsics",
    srcs = ["federated_intrinsics.cc"],
    hdrs = ["federated_intrinsics.h"],
    tags = ["nokokoro"],
    deps = ["@com_google_absl//absl/status:statusor"],
)

cc_library(
    name = "federating_executor",
    srcs = ["federating_executor.cc"],
    hdrs = ["federating_executor.h"],
    tags = ["nokokoro"],
    deps = [
        ":cardinalities",
        ":executor",
        ":federated_intrinsics",
        ":status_macros",
        ":threading",
        ":value_validation",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mock_executor",
    testonly = True,
    hdrs = ["mock_executor.h"],
    tags = ["nokokoro"],
    deps = [
        ":executor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_grpc",
    testonly = True,
    hdrs = ["mock_grpc.h"],
    tags = ["nokokoro"],
    deps = [
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "reference_resolving_executor",
    srcs = ["reference_resolving_executor.cc"],
    hdrs = ["reference_resolving_executor.h"],
    tags = ["nokokoro"],
    deps = [
        ":executor",
        ":status_macros",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core/platform:tstring",
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
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/cc:array_ops",
        "@org_tensorflow//tensorflow/cc:math_ops",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "remote_executor",
    srcs = ["remote_executor.cc"],
    hdrs = ["remote_executor.h"],
    tags = ["nokokoro"],
    deps = [
        ":cardinalities",
        ":executor",
        ":status_macros",
        ":threading",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "remote_executor_test",
    timeout = "short",
    srcs = ["remote_executor_test.cc"],
    tags = ["nokokoro"],
    deps = [
        ":executor",
        ":mock_grpc",
        ":remote_executor",
        ":value_test_utils",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/cc:array_ops",
        "@org_tensorflow//tensorflow/cc:math_ops",
    ],
)

pybind_extension(
    name = "serialization_bindings",
    srcs = ["serialization_bindings.cc"],
    tags = ["nokokoro"],
    deps = [
        ":status_macros",
        ":tensor_serialization",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@org_tensorflow//tensorflow/c/eager:c_api",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core/common_runtime/eager:tensor_handle",
        "@org_tensorflow//tensorflow/python/eager:pywrap_tfe_lib",
        "@org_tensorflow//tensorflow/python/lib/core:ndarray_tensor_headers",
        "@org_tensorflow//tensorflow/python/lib/core:safe_ptr",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:wrapped_proto_caster",
    ],
)

cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
    tags = ["nokokoro"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "tensor_serialization",
    srcs = ["tensor_serialization.cc"],
    hdrs = ["tensor_serialization.h"],
    tags = ["nokokoro"],
    deps = [
        ":status_macros",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tensorflow_executor",
    srcs = ["tensorflow_executor.cc"],
    hdrs = ["tensorflow_executor.h"],
    tags = ["nokokoro"],
    deps = [
        ":executor",
        ":status_macros",
        ":tensor_serialization",
        ":threading",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/cc:array_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/cc:dataset_ops_internal",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow_opensource",
        "@org_tensorflow//tensorflow/core/platform:errors",
        "@org_tensorflow//tensorflow/core/platform:tstring",
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
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/cc:array_ops",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/cc:dataset_ops_internal",
        "@org_tensorflow//tensorflow/cc:math_ops",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/cc:resource_variable_ops",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "threading",
    srcs = ["threading.cc"],
    hdrs = ["threading.h"],
    tags = ["nokokoro"],
    deps = [
        ":status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "threading_test",
    timeout = "short",
    srcs = ["threading_test.cc"],
    tags = ["nokokoro"],
    deps = [
        ":threading",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "value_test_utils",
    testonly = True,
    hdrs = ["value_test_utils.h"],
    tags = ["nokokoro"],
    deps = [
        ":executor",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/cc:const_op",
        "@org_tensorflow//tensorflow/cc:dataset_ops_internal",
        "@org_tensorflow//tensorflow/cc:scope",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "value_validation",
    srcs = ["value_validation.cc"],
    hdrs = ["value_validation.h"],
    tags = ["nokokoro"],
    deps = [
        ":cardinalities",
        ":federated_intrinsics",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status:statusor",
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
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
