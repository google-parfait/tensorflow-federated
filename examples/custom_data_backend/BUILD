licenses(["notice"])

cc_library(
    name = "data_backend_example_cc",
    srcs = ["data_backend_example.cc"],
    hdrs = ["data_backend_example.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/executors:data_backend",
        "//tensorflow_federated/cc/core/impl/executors:tensorflow_utils",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@federated_language//federated_language/proto:array_cc_proto",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)
