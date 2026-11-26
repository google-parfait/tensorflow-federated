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
"""Stages py_package files for py_wheel and strips C++ .so extensions."""

load("@rules_cc//cc:action_names.bzl", "STRIP_ACTION_NAME")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

def _strip_wheel_extensions_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    feature_cfg = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    strip_tool = cc_common.get_tool_for_action(
        feature_configuration = feature_cfg,
        action_name = STRIP_ACTION_NAME,
    )
    if not strip_tool:
        fail("CC toolchain does not provide a strip executable.")

    strip_flag = "-x" if ctx.attr.is_macos else "--strip-unneeded"
    prefix = ctx.attr.staging_prefix

    staged_files = []
    stripped_count = 0
    for f in ctx.attr.src[DefaultInfo].files.to_list():
        if not f.basename.endswith(".so"):
            staged_files.append(f)
            continue
        if f.short_path.startswith("../"):
            fail("Unexpected external .so in wheel package: %s" % f.short_path)
        out = ctx.actions.declare_file(prefix + "/" + f.short_path)
        args = ctx.actions.args()
        args.add(strip_flag)
        args.add("-o", out)
        args.add(f)
        ctx.actions.run(
            mnemonic = "StripWheelExtension",
            executable = strip_tool,
            arguments = [args],
            inputs = depset([f], transitive = [cc_toolchain.all_files]),
            outputs = [out],
            progress_message = "Stripping %s for wheel" % f.short_path,
        )
        staged_files.append(out)
        stripped_count += 1

    if stripped_count == 0:
        fail("Expected at least one .so extension in %s." % ctx.attr.src.label)

    return [DefaultInfo(files = depset(staged_files))]

strip_wheel_extensions = rule(
    implementation = _strip_wheel_extensions_impl,
    attrs = {
        "src": attr.label(mandatory = True),
        "is_macos": attr.bool(default = False),
        "staging_prefix": attr.string(default = "stripped_pkg"),
    },
    toolchains = use_cc_toolchain(),
    fragments = ["cpp"],
)
