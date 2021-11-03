licenses(["notice"])

cc_library(
    name = "oss_test_main",
    testonly = 1,
    srcs = [
        "oss_test_main.cc",
    ],
    tags = ["keep_dep"],  # Tell build_cleaner to keep dependencies on this.
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
    ],
)
