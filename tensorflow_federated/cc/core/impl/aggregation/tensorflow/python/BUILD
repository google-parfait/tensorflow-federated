# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
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
        "@pybind11_abseil//pybind11_abseil:status.so",
    ],
    tags = [
        "nokokoro",  # b/193543632
    ],
    deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/core:tensor_py_pb2",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:aggregation_protocol_messages_py_pb2",
        "//tensorflow_federated/cc/core/impl/aggregation/protocol:configuration_py_pb2",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
