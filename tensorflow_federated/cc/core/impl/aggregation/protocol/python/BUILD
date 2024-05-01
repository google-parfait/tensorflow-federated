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
