load("//tensorflow_federated:build_defs.bzl", "federated_py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

proto_library(
    name = "tensorflow_federated_v0_proto",
    srcs = [
        "computation.proto",
    ],
    deps = [],
)

federated_py_proto_library(
    name = "tensorflow_federated_v0_py_pb2",
    srcs = [
        "computation.proto",
    ],
    deps = [":tensorflow_federated_v0_proto"],
)
