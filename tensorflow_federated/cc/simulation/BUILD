load("//tensorflow_federated/tools:build_defs.bzl", "tff_cc_binary_with_tf_deps")

package(default_visibility = [
    ":simulation_packages",
    ":simulation_visibility",
])

package_group(
    name = "simulation_visibility",
    includes = [
        "//tensorflow_federated/python/simulation:simulation_packages",
        "//tensorflow_federated/python/tests:tests_packages",
    ],
)

package_group(
    name = "simulation_packages",
    packages = ["//tensorflow_federated/cc/simulation/..."],
)

licenses(["notice"])

cc_library(
    name = "servers",
    srcs = ["servers.cc"],
    hdrs = ["servers.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/executor_stacks:local_stacks",
        "//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "//tensorflow_federated/cc/core/impl/executors:executor",
        "//tensorflow_federated/cc/core/impl/executors:executor_service",
        "//tensorflow_federated/cc/core/impl/executors:status_macros",
        "//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "worker_main",
    srcs = ["worker_main.cc"],
    deps = [
        ":servers",
        "//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

# TODO(b/234147708): If / when we move to a static build of this directory, this target can be replaced with cc_binary.
tff_cc_binary_with_tf_deps(
    name = "worker_binary",
    linkopts = ["-lrt"],
    deps = [
        ":worker_main",
    ],
)
