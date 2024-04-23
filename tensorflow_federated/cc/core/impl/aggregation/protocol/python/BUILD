load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:public"],
)

pybind_extension(
    name = "aggregation_protocol",
    srcs = ["aggregation_protocol.cc"],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:aggregation_protocol",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:cc_proto",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:configuration_cc_proto",
        "@com_google_absl//absl/status",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)
