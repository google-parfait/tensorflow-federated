load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

pybind_extension(
    name = "aggregation_protocols",
    srcs = ["aggregation_protocols.cc"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:aggregation_protocol",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:configuration_cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:resource_resolver",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol/simple_aggregation",
        "//tensorflow_federated/cc/core/impl/aggregation/tensorflow:tensorflow_checkpoint_builder_factory",
        "//tensorflow_federated/cc/core/impl/aggregation/tensorflow:tensorflow_checkpoint_parser_factory",
        "@com_google_absl//absl/status:statusor",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_test(
    name = "aggregation_protocols_test",
    srcs = ["aggregation_protocols_test.py"],
    data = [
        ":aggregation_protocols.so",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol/python:aggregation_protocol.so",
    ],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor_py_pb2",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:configuration_py_pb2",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:py_pb2",
    ],
)
