load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "v0",
    srcs = ["__init__.py"],
)

py_proto_library(
    name = "computation_py_pb2",
    srcs = ["computation.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY3",
    deps = ["@com_google_protobuf//:protobuf_python"],
)

py_proto_library(
    name = "executor_py_pb2",
    srcs = ["executor.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY3",
    use_grpc_plugin = True,
    deps = [
        "@com_google_protobuf//:protobuf_python",
        ":computation_py_pb2",
    ],
)

filegroup(
    name = "proto_files",
    srcs = [
        "computation.proto",
        "executor.proto",
    ],
)
