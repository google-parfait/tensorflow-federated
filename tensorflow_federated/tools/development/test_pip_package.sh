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
#
# Usage:
#   bazel run //tensorflow_federated/tools/development:test_pip_package -- \
#       --package "/tmp/tensorflow_federated/"*".whl"
#
# Arguments:
#   package: A path to a local pip package.
set -ex

die() {
  echo >&2 "$@"
  exit 1
}

usage() {
  local script_name=$(basename "${0}")
  echo "usage: ${script_name} [--package PATH]" 1>&2
}

main() {
  # Parse arguments
  local package=""

  while [[ "$#" -gt 0 ]]; do
    opt="$1"
    case "${opt}" in
      --package)
        package="$2"
        shift
        # Shift might exit with an error code if no output_dir was provided.
        shift || break
        ;;
      *)
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z ${package} ]]; then
    usage
    exit 1
  fi

  echo "*** 1"
  pip freeze

  # Create working directory
  local temp_dir="$(mktemp -d)"
  trap "rm -rf ${temp_dir}" EXIT
  pushd "${temp_dir}"

  echo "*** 2"
  pip freeze

  if [[ $? -ne 0 ]]; then
      return_code=$?
      echo "error code 2 ='$?'"
      exit "${return_code}"
  fi

  echo "*** 3"
  pip freeze

  # Create a virtual environment
  virtualenv --python=python3 "venv"
  source "venv/bin/activate"
  pip install --upgrade pip

  echo "*** 4"
  pip freeze

  if [[ $? -ne 0 ]]; then
      return_code=$?
      echo "error code 4 ='$?'"
      exit "${return_code}"
  fi

  echo "*** 5"
  pip freeze

  # Test pip package
  pip install --upgrade "${package}"
  python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"

  echo "*** 6"
  pip freeze

  if [[ $? -ne 0 ]]; then
      return_code=$?
      echo "error code 6 ='$?'"
      exit "${return_code}"
  fi
}

main "$@"
