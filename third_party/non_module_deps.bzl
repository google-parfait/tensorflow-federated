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
"""Module extension for loading non-BCR differential-privacy archives."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _non_module_deps_impl(_ctx):
    http_archive(
        name = "com_google_differential_privacy",
        integrity = "sha256-bm4c16gZaVyq5Aj0+pOBKat6hug/4kEBN8heUBMau+A=",
        strip_prefix = "differential-privacy-3.0.0",
        urls = [
            "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
        ],
    )
    http_archive(
        name = "com_google_cc_differential_privacy",
        integrity = "sha256-bm4c16gZaVyq5Aj0+pOBKat6hug/4kEBN8heUBMau+A=",
        strip_prefix = "differential-privacy-3.0.0/cc",
        urls = [
            "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
        ],
    )

non_module_deps = module_extension(implementation = _non_module_deps_impl)
