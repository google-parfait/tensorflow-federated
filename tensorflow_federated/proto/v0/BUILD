load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "v0",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_proto_library(
    name = "tensorflow_federated_v0_py_pb2",
    srcs = ["computation.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY2AND3",
    deps = ["@com_google_protobuf//:protobuf_python"],
)
