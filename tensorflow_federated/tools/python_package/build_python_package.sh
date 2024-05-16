#!/usr/bin/env bash
# Copyright 2019, The TensorFlow Federated Authors.
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
#
# Tool to build the TensorFlow Federated pip package.
set -e

main() {
  # DO_NOT_SUBMIT: Check GLIBC version.
  # ldd --version

  # Build the Python package.
  pip install --upgrade setuptools wheel
  # The manylinux tag should match GLIBC version returned by `ldd --version`.
  python "tensorflow_federated/tools/python_package/setup.py" bdist_wheel \
      --plat-name=manylinux_2_31_x86_64

  ls -la "."
  ls -la "dist"

  # DO_NOT_SUBMIT: Check Python package sizes.
  # local package="$(ls "dist/tensorflow_federated-"*".whl" | head -n1)"
  # local actual_size="$(du -b "${package}" | cut -f1)"
  # local maximum_size=80000000  # 80 MiB
  # if [ "${actual_size}" -ge "${maximum_size}" ]; then
  #   echo "Error: expected $(basename ${package}) to be less than ${maximum_size} bytes; it was ${actual_size}." 1>&2
  #   exit 1
  # fi
}

main "$@"
