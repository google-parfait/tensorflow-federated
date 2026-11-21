# Copyright 2026, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Repository rule providing hermetic Python headers for @system_python."""

_BUILD_FILE = """\
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

alias(
    name = "python_headers",
    actual = "@rules_python//python/cc:current_py_cc_headers",
    visibility = ["//visibility:public"],
)

string_flag(
    name = "internal_python_support",
    build_setting_default = "Supported",
    values = [
        "None",
        "Supported",
        "Unsupported",
    ],
)

config_setting(
    name = "none",
    flag_values = {":internal_python_support": "None"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "supported",
    flag_values = {":internal_python_support": "Supported"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "unsupported",
    flag_values = {":internal_python_support": "Unsupported"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "exists",
    match_any = [":supported", ":unsupported"],
    visibility = ["//visibility:public"],
)
"""

def _hermetic_system_python_impl(repository_ctx):
    repository_ctx.file("BUILD.bazel", _BUILD_FILE)

hermetic_system_python = repository_rule(
    implementation = _hermetic_system_python_impl,
)
