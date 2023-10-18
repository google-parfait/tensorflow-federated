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
# Tool to test the TensorFlow Federated pip package.
set -e

usage() {
  local script_name=$(basename "${0}")
  local options=(
      "--python=python3.11"
      "--package=<path>"
  )
  echo "usage: ${script_name} ${options[@]}"
  echo "  --python=python3.11  The Python version used by the environment to"
  echo "                       build the Python package."
  echo "  --package=<path>     A path to a local pip package."
  exit 1
}

main() {
  # Parse the arguments.
  local python="python3.11"
  local package=""

  while [[ "$#" -gt 0 ]]; do
    option="$1"
    case "${option}" in
      --python=*)
        python="${option#*=}"
        shift
        ;;
      --package=*)
        package="${option#*=}"
        shift
        ;;
      *)
        echo "error: unrecognized option '${option}'" 1>&2
        usage
        ;;
    esac
  done

  if [[ -z "${package}" ]]; then
    echo "error: required option `--package`" 1>&2
    usage
  elif [[ ! -f "${package}" ]]; then
    echo "error: the file '${package}' does not exist" 1>&2
    usage
  fi

  # Create a working directory.
  local temp_dir="$(mktemp -d)"
  trap "rm -rf ${temp_dir}" EXIT
  pushd "${temp_dir}"

  # Create a Python environment.
  "${python}" -m venv "venv"
  source "venv/bin/activate"
  python --version
  pip install --upgrade "pip"
  pip --version

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
