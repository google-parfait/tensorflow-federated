load("//tensorflow_federated/tools:build_defs.bzl", "tff_cc_test_with_tf_deps")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = [
    ":executor_stacks_packages",
    "//tensorflow_federated/cc/core/impl:impl_visibility",
])

package_group(
    name = "executor_stacks_packages",
    packages = [
        "//tensorflow_federated/cc/core/impl/executor_stacks/...",
        "//tensorflow_federated/python/core/impl/executor_stacks/...",
    ],
)

licenses(["notice"])

pybind_extension(
    name = "executor_stack_bindings",
    srcs = ["executor_stack_bindings.cc"],
    deps = [
        ":remote_stacks",
        "//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "@com_github_grpc_grpc//:grpc++",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)

cc_library(
    name = "local_stacks",
    srcs = ["local_stacks.cc"],
    hdrs = ["local_stacks.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "//tensorflow_federated/cc/core/impl/executors:executor",
        "//tensorflow_federated/cc/core/impl/executors:federating_executor",
        "//tensorflow_federated/cc/core/impl/executors:reference_resolving_executor",
        "//tensorflow_federated/cc/core/impl/executors:status_macros",
        "//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
        "@com_google_absl//absl/status:statusor",
    ],
)

tff_cc_test_with_tf_deps(
    name = "local_stacks_test",
    srcs = ["local_stacks_test.cc"],
    deps = [
        ":local_stacks",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
        "//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "//tensorflow_federated/cc/core/impl/executors:executor",
        "//tensorflow_federated/cc/core/impl/executors:mock_executor",
        "//tensorflow_federated/cc/core/impl/executors:status_matchers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "remote_stacks",
    srcs = ["remote_stacks.cc"],
    hdrs = ["remote_stacks.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "//tensorflow_federated/cc/core/impl/executors:composing_executor",
        "//tensorflow_federated/cc/core/impl/executors:executor",
        "//tensorflow_federated/cc/core/impl/executors:federating_executor",
        "//tensorflow_federated/cc/core/impl/executors:reference_resolving_executor",
        "//tensorflow_federated/cc/core/impl/executors:remote_executor",
        "//tensorflow_federated/cc/core/impl/executors:status_macros",
        "//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
        "//tensorflow_federated/cc/core/impl/executors:threading",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tff_cc_test_with_tf_deps(
    name = "remote_stacks_test",
    srcs = ["remote_stacks_test.cc"],
    tags = ["nomacos"],
    deps = [
        ":remote_stacks",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
        "//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "//tensorflow_federated/cc/core/impl/executors:composing_executor",
        "//tensorflow_federated/cc/core/impl/executors:executor",
        "//tensorflow_federated/cc/core/impl/executors:mock_executor",
        "//tensorflow_federated/cc/core/impl/executors:status_matchers",
        "//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
