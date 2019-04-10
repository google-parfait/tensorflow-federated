load("@protobuf_archive//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_proto_library(
    name = "tensorflow_federated_v0_py_pb2",
    srcs = ["computation.proto"],
    default_runtime = "@protobuf_archive//:protobuf_python",
    protoc = "@protobuf_archive//:protoc",
    srcs_version = "PY2AND3",
    deps = ["@protobuf_archive//:protobuf_python"],
)
