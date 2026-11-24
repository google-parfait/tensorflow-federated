load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":executor_stacks_packages",
        "//tensorflow_federated/cc/core/impl:impl_users",
    ],
)

package_group(
    name = "executor_stacks_packages",
    packages = [
        "//tensorflow_federated/cc/core/impl/executor_stacks/...",
        "//tensorflow_federated/python/core/impl/executor_stacks/...",
    ],
)

licenses(["notice"])

cc_library(
    name = "local_stacks",
    srcs = ["local_stacks.cc"],
    hdrs = ["local_stacks.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "//tensorflow_federated/cc/core/impl/executors:executor",
        "//tensorflow_federated/cc/core/impl/executors:federating_executor",
        "//tensorflow_federated/cc/core/impl/executors:reference_resolving_executor",
        "//tensorflow_federated/cc/core/impl/executors:sequence_executor",
        "//tensorflow_federated/cc/core/impl/executors:status_macros",
        "//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "local_stacks_test",
    srcs = ["local_stacks_test.cc"],
    deps = [
        ":local_stacks",
        "//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "//tensorflow_federated/cc/core/impl/executors:executor",
        "//tensorflow_federated/cc/core/impl/executors:mock_executor",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "remote_stacks",
    srcs = ["remote_stacks.cc"],
    hdrs = ["remote_stacks.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "//tensorflow_federated/cc/core/impl/executors:composing_executor",
        "//tensorflow_federated/cc/core/impl/executors:executor",
        "//tensorflow_federated/cc/core/impl/executors:reference_resolving_executor",
        "//tensorflow_federated/cc/core/impl/executors:remote_executor",
        "//tensorflow_federated/cc/core/impl/executors:remote_executor_stack",
        "//tensorflow_federated/cc/core/impl/executors:status_macros",
        "//tensorflow_federated/cc/core/impl/executors:streaming_remote_executor",
        "//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "remote_stacks_test",
    srcs = ["remote_stacks_test.cc"],
    deps = [
        ":remote_stacks",
        "//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "//tensorflow_federated/cc/core/impl/executors:composing_executor",
        "//tensorflow_federated/cc/core/impl/executors:executor",
        "//tensorflow_federated/cc/core/impl/executors:mock_executor",
        "//tensorflow_federated/cc/testing:oss_test_main",
        "//tensorflow_federated/cc/testing:status_matchers",
        "@com_github_grpc_grpc//:gpr",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
