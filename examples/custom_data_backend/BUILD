load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_applicable_licenses = ["//:package_license"])

licenses(["notice"])

cc_library(
    name = "data_backend_example_cc",
    srcs = ["data_backend_example.cc"],
    hdrs = ["data_backend_example.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/executors:data_backend",
        "//tensorflow_federated/cc/core/impl/executors:tensorflow_utils",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@federated_language//federated_language/proto:array_cc_proto",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@pypi_tensorflow//:libtensorflow_cc",
        "@pypi_tensorflow//:libtensorflow_framework",
        "@pypi_tensorflow//:tf_header_lib",
    ],
)
