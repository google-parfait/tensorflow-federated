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

usage() {
  local script_name=$(basename "${0}")
  echo "usage: ${script_name} --package=<path>"
  echo "  --package=<path>  A path to a local Python package."
}

main() {
  # Parse the arguments.
  local package=""

  while [[ "$#" -gt 0 ]]; do
    option="$1"
    case "${option}" in
      --package=*)
        package="${option#*=}"
        shift
        ;;
      *)
        echo "error: unrecognized option '${option}'" 1>&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z "${package}" ]]; then
    echo "error: expected a 'package'" 1>&2
    usage
    exit 1
  elif [[ ! -f "${package}" ]]; then
    echo "error: expected the package '${package}' to exist" 1>&2
    usage
    exit 1
  fi

  # Check Python package sizes.
  local actual_size="$(du -b "${package}" | cut -f1)"
  local maximum_size=80000000  # 80 MiB
  if [[ "${actual_size}" -ge "${maximum_size}" ]]; then
    echo "error: expected '${package}' to be less than '${maximum_size}' bytes, it was '${actual_size}' bytes" 1>&2
    exit 1
  fi

  # Create a temp directory.
  local temp_dir="$(mktemp --directory)"
  trap "rm -rf ${temp_dir}" EXIT

  # Create a Python environment.
  python3 -m venv "${temp_dir}/venv"
  source "${temp_dir}/venv/bin/activate"
  python --version
  pip install --upgrade "pip"
  pip --version

  # Test the Python package.
  pip install --upgrade "${package}"
  pip freeze
  python -c "import tensorflow_federated as tff; print(tff.__version__)"
}

main "$@"
