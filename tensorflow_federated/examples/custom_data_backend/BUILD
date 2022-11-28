load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

cc_library(
    name = "data_backend_example_cc",
    srcs = ["data_backend_example.cc"],
    hdrs = ["data_backend_example.h"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        "//tensorflow_federated/cc/core/impl/executors:data_backend",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

pybind_extension(
    name = "data_backend_example_bindings",
    srcs = ["data_backend_example_bindings.cc"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":data_backend_example_cc",
        "//tensorflow_federated/cc/core/impl/executors:data_backend",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:wrapped_proto_caster",
    ],
)

py_library(
    name = "data_backend_example_py",
    srcs = ["data_backend_example.py"],
    srcs_version = "PY3",
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":data_backend_example_bindings",
        "//tensorflow_federated",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
    ],
)

py_test(
    name = "data_backend_example_test",
    srcs = ["data_backend_example_test.py"],
    data = ["@pybind11_abseil//pybind11_abseil:status.so"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":data_backend_example_py",
        "//tensorflow_federated",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
    ],
)
