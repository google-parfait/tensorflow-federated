package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//tensorflow_federated/cc/core/impl/aggregation:__subpackages__"],
)

cc_library(
    name = "base",
    srcs = [
        "base_name.cc",
        "monitoring.cc",
        "platform.cc",
    ],
    hdrs = [
        "base_name.h",
        "monitoring.h",
        "move_to_lambda.h",
        "platform.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_test(
    name = "monitoring_test",
    size = "small",
    srcs = ["monitoring_test.cc"],
    deps = [
        ":base",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "platform_test",
    size = "small",
    srcs = ["platform_test.cc"],
    deps = [
        ":base",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
        "//tensorflow_federated/cc/core/impl/aggregation/testing",
        "//tensorflow_federated/cc/core/impl/executors:status_matchers",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "scheduler",
    srcs = ["scheduler.cc"],
    hdrs = ["scheduler.h"],
    deps = [
        ":base",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "scheduler_test",
    size = "small",
    srcs = ["scheduler_test.cc"],
    deps = [
        ":base",
        ":scheduler",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "move_to_lambda_test",
    size = "small",
    srcs = ["move_to_lambda_test.cc"],
    deps = [
        ":base",
        ":unique_value",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
    ],
)

cc_library(
    name = "unique_value",
    hdrs = ["unique_value.h"],
)

cc_test(
    name = "unique_value_test",
    srcs = ["unique_value_test.cc"],
    deps = [
        ":base",
        ":unique_value",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "clock",
    srcs = ["clock.cc"],
    hdrs = ["clock.h"],
    deps = [
        ":base",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "simulated_clock",
    testonly = 1,
    srcs = ["simulated_clock.cc"],
    hdrs = ["simulated_clock.h"],
    deps = [":clock"],
)

cc_test(
    name = "simulated_clock_test",
    srcs = ["simulated_clock_test.cc"],
    deps = [
        ":simulated_clock",
        "//tensorflow_federated/cc/common_libs:oss_test_main",
        "@com_google_absl//absl/time",
    ],
)
