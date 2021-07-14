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

script="$(readlink -f "$0")"
script_dir="$(dirname "${script}")"
source "${script_dir}/common.sh"

usage() {
  local script_name=$(basename "${0}")
  local options=(
      "--latest|--nightly"
      "--python=python3"
      "--output_dir=<path>"
  )
  echo "usage: ${script_name} ${options[@]}"
  echo "  --latest|--nightly   Flags indicating whether to build the latest or"
  echo "                       nightly version of the TFF Python package."
  echo "  --python=python3     The Python version used by the environment to"
  echo "                       build the Python package."
  echo "  --output_dir=<path>  An output directory."
  exit 1
}

main() {
  # Parse arguments
  local latest=0
  local nightly=0
  local python="python3"
  local output_dir=""

  while [[ "$#" -gt 0 ]]; do
    option="$1"
    case "${option}" in
      --latest)
        latest=1
        shift
        ;;
      --nightly)
        nightly=1
        shift
        ;;
      --python=*)
        python="${option#*=}"
        shift
        ;;
      --output_dir=*)
        output_dir="${option#*=}"
        shift
        ;;
      *)
        error_unrecognized "${option}"
        usage
        ;;
    esac
  done

  if [[ "${latest}" == "1" ]] && [[ "${nightly}" == "1" ]]; then
    error_exclusive "--latest" "--nightly"
    usage
  elif [[ "${latest}" == "0" ]] && [[ "${nightly}" == "0" ]]; then
    error_required "--latest" "--nightly"
    usage
  fi

  if [[ -z "${output_dir}" ]]; then
    error_required "--output_dir"
    usage
  elif [[ ! -d "${output_dir}" ]]; then
    error_directory_does_not_exist "${output_dir}"
    usage
  fi

  # Create working directory
  local temp_dir="$(mktemp -d)"
  trap "rm -rf ${temp_dir}" EXIT
  cp -LR "tensorflow_federated" "${temp_dir}"
  pushd "${temp_dir}"

  # Create a virtual environment
  virtualenv --python="${python}" "venv"
  source "venv/bin/activate"
  python --version
  pip install --upgrade pip
  pip --version

  # Build pip package
  pip install --upgrade setuptools wheel
  flags=()
  if [[ "${latest}" == "1" ]]; then
    :  # pass
  elif [[ "${nightly}" == "1" ]]; then
    flags+=("--nightly")
  fi
  python "tensorflow_federated/tools/python_package/setup.py" bdist_wheel \
      --universal \
      "${flags[@]}"
  cp "${temp_dir}/dist/"* "${output_dir}"

  # Cleanup
  deactivate
  popd
}

main "$@"
