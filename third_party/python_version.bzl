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
"""Repository rule selecting the hermetic Python version from HERMETIC_PYTHON_VERSION."""

_SUPPORTED_VERSIONS = {
    "3.12": ("3_12", "//:requirements_lock_3_12.txt"),
    "3.13": ("3_13", "//:requirements_lock_3_13.txt"),
}

_DEFAULT_VERSION = "3.12"

def _hermetic_python_version_impl(repository_ctx):
    version = repository_ctx.os.environ.get("HERMETIC_PYTHON_VERSION", _DEFAULT_VERSION)
    if not version:
        version = _DEFAULT_VERSION
    if version not in _SUPPORTED_VERSIONS:
        fail(
            "Unsupported HERMETIC_PYTHON_VERSION=%r; supported versions: %s" %
            (version, sorted(_SUPPORTED_VERSIONS.keys())),
        )
    version_underscore, requirements_lock = _SUPPORTED_VERSIONS[version]
    repository_ctx.file(
        "BUILD.bazel",
        content = (
            "alias(\n" +
            '    name = "python_files",\n' +
            '    actual = "@python_{underscore}//:files",\n' +
            '    visibility = ["//visibility:public"],\n' +
            ")\n"
        ).format(underscore = version_underscore),
    )
    repository_ctx.file(
        "defs.bzl",
        content = (
            'PYTHON_VERSION = "{version}"\n' +
            'PYTHON_VERSION_UNDERSCORE = "{underscore}"\n' +
            'REQUIREMENTS_LOCK = "{lock}"\n'
        ).format(
            version = version,
            underscore = version_underscore,
            lock = requirements_lock,
        ),
    )

hermetic_python_version = repository_rule(
    implementation = _hermetic_python_version_impl,
    environ = ["HERMETIC_PYTHON_VERSION"],
)
