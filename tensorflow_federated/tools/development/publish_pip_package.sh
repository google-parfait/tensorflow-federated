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
# Tool to publish the TensorFlow Federated pip package.
#
# Usage:
#   bazel run //tensorflow_federated/tools/development:publish_pip_package -- \
#       "/tmp/tensorflow_federated/tensorflow_federated-"*".whl"
#
# Arguments:
#   package: A path to a local pip package.
set -e

main() {
  local package="$1"

  if [[ ! -f "${package}" ]]; then
    die "The package '${package}' does not exist."
  fi

  local temp_dir="$(mktemp -d)"
  trap "rm -rf ${temp_dir}" EXIT
  pushd "${temp_dir}"

  # Create a virtual environment
  virtualenv --python=python3 "venv"
  source "venv/bin/activate"
  pip install --upgrade pip

  # Publish pip package
  pip install --upgrade "twine"
  twine check "${package}"
  twine upload "${package}"
}

main "$@"
