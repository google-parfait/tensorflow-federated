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
    deps = [
        ":cardinalities",
        ":computations",
        ":executor",
        ":federated_intrinsics",
        ":status_macros",
        ":threading",
        ":value_validation",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.8.0.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.8.0.dylib",
    ],
)

cc_test(
    name = "composing_executor_test",
    srcs = [
        "composing_executor_test.cc",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.dylib",
    ],
    deps = [
        ":composing_executor",
        ":computations",
        ":executor",
        ":executor_test_base",
        ":federated_intrinsics",
        ":mock_executor",
        ":status_matchers",
        ":value_test_utils",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "computations",
    hdrs = ["computations.h"],
    deps = ["//tensorflow_federated/proto/v0:computation_cc_proto"],
)

pybind_extension(
    name = "executor_bindings",
    srcs = ["executor_bindings.cc"],
    deps = [
        ":cardinalities",
        ":composing_executor",
        ":executor",
        ":federating_executor",
        ":reference_resolving_executor",
        ":remote_executor",
        ":tensorflow_executor",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:wrapped_proto_caster",
    ],
)

cc_library(
    name = "executor",
    srcs = ["executor.cc"],
    hdrs = ["executor.h"],
    deps = [
        ":status_macros",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.8.0.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.8.0.dylib",
        "@org_tensorflow//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "executor_service",
    srcs = ["executor_service.cc"],
    hdrs = ["executor_service.h"],
    deps = [
        ":cardinalities",
        ":executor",
        ":status_conversion",
        ":status_macros",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_test(
    name = "executor_service_test",
    srcs = [
        "executor_service_test.cc",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.dylib",
    ],
    deps = [
        ":executor",
        ":executor_service",
        ":mock_executor",
        ":protobuf_matchers",
        ":status_conversion",
        ":status_matchers",
        ":value_test_utils",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "executor_test_base",
    testonly = True,
    srcs = [],
    hdrs = ["executor_test_base.h"],
    deps = [
        ":executor",
        ":mock_executor",
        ":protobuf_matchers",
        ":status_matchers",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "federated_intrinsics",
    srcs = ["federated_intrinsics.cc"],
    hdrs = ["federated_intrinsics.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "federating_executor",
    srcs = ["federating_executor.cc"],
    hdrs = ["federating_executor.h"],
    deps = [
        ":cardinalities",
        ":executor",
        ":federated_intrinsics",
        ":status_macros",
        ":threading",
        ":value_validation",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.8.0.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.8.0.dylib",
    ],
)

cc_test(
    name = "federating_executor_test",
    srcs = [
        "federating_executor_test.cc",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.dylib",
    ],
    deps = [
        ":executor",
        ":executor_test_base",
        ":federating_executor",
        ":mock_executor",
        ":status_matchers",
        ":value_test_utils",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mock_executor",
    testonly = True,
    hdrs = ["mock_executor.h"],
    deps = [
        ":executor",
        ":protobuf_matchers",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_grpc",
    testonly = True,
    hdrs = ["mock_grpc.h"],
    deps = [
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "protobuf_matchers",
    testonly = True,
    hdrs = ["protobuf_matchers.h"],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "reference_resolving_executor",
    srcs = ["reference_resolving_executor.cc"],
    hdrs = ["reference_resolving_executor.h"],
    deps = [
        ":executor",
        ":status_macros",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.8.0.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.8.0.dylib",
    ],
)

cc_test(
    name = "reference_resolving_executor_test",
    timeout = "short",
    srcs = [
        "reference_resolving_executor_test.cc",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.dylib",
    ],
    deps = [
        ":executor",
        ":executor_test_base",
        ":mock_executor",
        ":protobuf_matchers",
        ":reference_resolving_executor",
        ":status_matchers",
        ":value_test_utils",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "remote_executor",
    srcs = ["remote_executor.cc"],
    hdrs = ["remote_executor.h"],
    deps = [
        ":cardinalities",
        ":executor",
        ":status_conversion",
        ":status_macros",
        ":threading",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_test(
    name = "remote_executor_test",
    timeout = "short",
    srcs = [
        "remote_executor_test.cc",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.dylib",
    ],
    deps = [
        ":executor",
        ":mock_grpc",
        ":protobuf_matchers",
        ":remote_executor",
        ":status_matchers",
        ":value_test_utils",
        "//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "serialization_bindings",
    srcs = ["serialization_bindings.cc"],
    deps = [
        ":status_macros",
        ":tensor_serialization",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.8.0.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.8.0.dylib",
        "@org_tensorflow//tensorflow/python/lib/core:ndarray_tensor_headers",
        "@org_tensorflow//tensorflow/python/lib/core:safe_ptr",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:wrapped_proto_caster",
    ],
)

cc_library(
    name = "status_conversion",
    srcs = ["status_conversion.cc"],
    hdrs = ["status_conversion.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "status_matchers",
    testonly = True,
    srcs = ["status_matchers.cc"],
    hdrs = ["status_matchers.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tensor_serialization",
    srcs = ["tensor_serialization.cc"],
    hdrs = ["tensor_serialization.h"],
    deps = [
        ":status_macros",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.8.0.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.8.0.dylib",
    ],
)

cc_library(
    name = "session_provider",
    srcs = ["session_provider.cc"],
    hdrs = ["session_provider.h"],
    deps = [
        ":status_macros",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.8.0.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.8.0.dylib",
    ],
)

cc_test(
    name = "session_provider_test",
    srcs = [
        "session_provider_test.cc",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.dylib",
    ],
    deps = [
        ":session_provider",
        ":status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensorflow_executor",
    srcs = ["tensorflow_executor.cc"],
    hdrs = ["tensorflow_executor.h"],
    deps = [
        ":executor",
        ":session_provider",
        ":status_macros",
        ":tensor_serialization",
        ":threading",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.8.0.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.8.0.dylib",
    ],
)

cc_cpu_gpu_test(
    name = "tensorflow_executor_test",
    srcs = [
        "tensorflow_executor_test.cc",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.dylib",
    ],
    deps = [
        ":executor",
        ":protobuf_matchers",
        ":status_macros",
        ":status_matchers",
        ":tensorflow_executor",
        ":value_test_utils",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "threading",
    srcs = ["threading.cc"],
    hdrs = ["threading.h"],
    deps = [
        ":status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "threading_test",
    timeout = "short",
    srcs = [
        "threading_test.cc",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.dylib",
    ],
    deps = [
        ":status_matchers",
        ":threading",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "value_test_utils",
    testonly = True,
    hdrs = ["value_test_utils.h"],
    deps = [
        ":executor",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.8.0.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.8.0.dylib",
    ],
)

cc_library(
    name = "value_validation",
    srcs = ["value_validation.cc"],
    hdrs = ["value_validation.h"],
    deps = [
        ":cardinalities",
        ":federated_intrinsics",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "value_validation_test",
    timeout = "short",
    srcs = [
        "value_validation_test.cc",
        "@org_tensorflow//tensorflow:libtensorflow_cc.2.dylib",
        "@org_tensorflow//tensorflow:libtensorflow_framework.2.dylib",
    ],
    deps = [
        ":status_matchers",
        ":value_test_utils",
        ":value_validation",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
