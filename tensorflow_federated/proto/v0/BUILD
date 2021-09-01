load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "v0",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

proto_library(
    name = "computation_proto",
    srcs = ["computation.proto"],
    deps = ["@com_google_protobuf//:any_proto"],
)

py_proto_library(
    name = "computation_py_pb2",
    deps = [":computation_proto"],
)

py_test(
    name = "computation_pb2_test",
    srcs = ["computation_pb2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "nofixdeps",
        "nokokoro",
    ],
    deps = [
        ":computation_py_pb2",
        "//third_party/tensorflow/core:protos_all_py_pb2",
        "@absl_py//absl/testing:absltest",
    ],
)

cc_proto_library(
    name = "computation_cc_proto",
    deps = [":computation_proto"],
)

proto_library(
    name = "executor_proto",
    srcs = ["executor.proto"],
    deps = [
        ":computation_proto",
        "@com_google_protobuf//:any_proto",
    ],
)

py_proto_library(
    name = "executor_py_pb2",
    deps = [":executor_proto"],
)

py_grpc_library(
    name = "executor_py_pb2_grpc",
    srcs = [":executor_proto"],
    deps = [":executor_py_pb2"],
)

cc_proto_library(
    name = "executor_cc_proto",
    deps = [":executor_proto"],
)

cc_grpc_library(
    name = "executor_cc_grpc_proto",
    srcs = [":executor_proto"],
    grpc_only = True,
    deps = [":executor_cc_proto"],
)

filegroup(
    name = "proto_files",
    srcs = [
        "computation.proto",
        "executor.proto",
    ],
)
