package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":testing_packages",
        "//tensorflow_federated/cc/core/impl:impl_packages",
        "//tensorflow_federated/cc/simulation:simulation_packages",
    ],
)

package_group(
    name = "testing_packages",
    packages = [
        "//tensorflow_federated/cc/testing/...",
    ],
)

licenses(["notice"])

cc_library(
    name = "oss_test_main",
    testonly = 1,
    srcs = ["oss_test_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/flags:parse",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "protobuf_matchers",
    testonly = True,
    hdrs = ["protobuf_matchers.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "status_matchers",
    testonly = True,
    srcs = ["status_matchers.cc"],
    hdrs = ["status_matchers.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@googletest//:gtest",
    ],
)
