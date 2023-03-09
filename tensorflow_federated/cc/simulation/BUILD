package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":simulation_packages",
        ":simulation_users",
    ],
)

package_group(
    name = "simulation_packages",
    packages = ["//tensorflow_federated/cc/simulation/..."],
)

package_group(
    name = "simulation_users",
    includes = [
        "//tensorflow_federated/data:data_packages",
        "//tensorflow_federated/python/simulation:simulation_packages",
        "//tensorflow_federated/python/tests:tests_packages",
        "//tensorflow_federated/tools:tools_packages",
    ],
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

cc_binary(
    name = "worker_binary",
    linkopts = ["-lrt"],
    deps = [
        ":worker_main",
        "@tensorflow_compression//tensorflow_compression/cc:run_length_gamma_kernels",  # buildcleaner: keep, RunLengthGammaEncode
    ],
)
