licenses(["notice"])

cc_library(
    name = "data_backend_example_cc",
    srcs = ["data_backend_example.cc"],
    hdrs = ["data_backend_example.h"],
    deps = [
        "//tensorflow_federated/cc/core/impl/executors:data_backend",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)
