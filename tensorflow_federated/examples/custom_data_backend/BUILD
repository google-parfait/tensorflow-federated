load("//tensorflow_federated/tools:build_defs.bzl", "tff_cc_library_with_tf_deps", "tff_pybind_extension_with_tf_deps")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

tff_cc_library_with_tf_deps(
    name = "data_backend_example_cc",
    srcs = ["data_backend_example.cc"],
    hdrs = ["data_backend_example.h"],
    tf_deps = [
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:protos_all_cc",
    ],
    deps = [
        "//tensorflow_federated/cc/core/impl/executors:data_backend",
        "//tensorflow_federated/proto/v0:computation_cc_proto",
        "//tensorflow_federated/proto/v0:executor_cc_proto",
        "@com_google_absl//absl/status",
    ],
)

tff_pybind_extension_with_tf_deps(
    name = "data_backend_example_bindings",
    srcs = ["data_backend_example_bindings.cc"],
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
    tags = ["nokokoro"],
    deps = [
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
    tags = ["nokokoro"],
    deps = [
        ":data_backend_example_py",
        "//tensorflow_federated",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
