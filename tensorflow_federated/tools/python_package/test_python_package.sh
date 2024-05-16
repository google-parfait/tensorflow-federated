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
# Tool to test the TensorFlow Federated Python package.
set -e

main() {
  # Create a working directory.
  local temp_dir="$(mktemp -d)"
  trap "rm -rf ${temp_dir}" EXIT

  # Create a Python environment.
  python3 -m venv "${temp_dir}/venv"
  source "${temp_dir}/venv/bin/activate"
  python --version
  pip install --upgrade "pip"
  pip --version

  # Get the Python package.
  local package="$(ls "dist/tensorflow_federated-"*".whl" | head -n1)"
  if [[ ! -f "${package}" ]]; then
    echo "error: the file '${package}' does not exist." 1>&2
    exit 1
  fi

  # Test the Python package.
  pip install --upgrade "${package}"
  pip freeze
  python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"
  python -c "import tensorflow_federated as tff; print(tff.__version__)"

  # Cleanup.
  deactivate
  popd
}

main "$@"
