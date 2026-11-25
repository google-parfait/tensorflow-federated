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
"""Minimal setuptools entry point marking wheels as platform/ABI-specific.

Bazel compiles `executor_bindings.so` and `tensorflow_bindings.so`, and
`pyproject.toml` includes them as `package-data` with an empty `ext_modules`
list. Without this file, `setuptools` marks the wheel as pure Python
(`Root-Is-Purelib: true`) and emits a `py3-none` tag. The pybind11 extensions
link against version-specific CPython C-API symbols, not `abi3`. A `py3-none`
tag lets `pip` install a CPython 3.12 extension into Python 3.13 and fail at
import time.

PEP 621 and `[tool.setuptools]` do not expose `distclass`, `has_ext_modules`,
or `root_is_pure`. This shim overrides `Distribution.has_ext_modules()` to
return `True` so `bdist_wheel` sets `Root-Is-Purelib: false` and emits CPython
ABI-specific tags (`cp312-cp312`, `cp313-cp313`).

Keep all project metadata, dependencies, and package configuration in
`pyproject.toml`. Do not add package metadata to this file.
"""

import setuptools
from setuptools import dist


class BinaryDistribution(dist.Distribution):
  """Distribution that forces a CPython and platform-specific wheel tag."""

  def has_ext_modules(self) -> bool:
    return True


setuptools.setup(distclass=BinaryDistribution)
