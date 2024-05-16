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
# Tool to build the TensorFlow Federated Python package.
set -e

usage() {
  local script_name=$(basename "${0}")
  local options=(
      "--output_dir=<path>"
  )
  echo "Usage: ${script_name} ${options[@]}"
  echo "  --output_dir=<path>   An output directory."
  exit 1
}

main() {

  # DO_NOT_SUBMIT: Debugging
  echo "---"
  pwd
  echo "---"
  echo "${RUNFILES}"
  echo "---"
  echo "${BUILD_WORKSPACE_DIRECTORY}"
  echo "---"
  echo "${BUILD_WORKING_DIRECTORY}"
  echo "---"
  printenv

  # Parse the arguments.
  local output_dir=""

  while [[ "$#" -gt 0 ]]; do
    option="$1"
    case "${option}" in
      --output_dir=*)
        output_dir="${option#*=}"
        shift
        ;;
      *)
        echo "Error: Unrecognized option '${option}'." 1>&2
        usage
        ;;
    esac
  done

  if [[ -z "${output_dir}" ]]; then
    echo "Error: Required option '--output_dir'." 1>&2
    usage
  elif [[ ! -d "${output_dir}" ]]; then
    mkdir --parents "${output_dir}"
  fi

  # Create a working directory.
  local temp_dir="$(mktemp -d)"
  trap "rm -rf ${temp_dir}" EXIT

  # Create a Python environment.
  python3 -m venv "${temp_dir}/venv"
  source "${temp_dir}/venv/bin/activate"
  python --version
  pip install --upgrade "pip"
  pip --version

  # Check the GLIBC version.
  local expected_glibc="2.35"
  if ! ldd --version | grep --quiet "${expected_glibc}"; then
    echo "Error: Expected GLIBC version to be '${expected_glibc}', found:" 1>&2
    ldd --version 1>&2
    exit 1
  fi

  # Build the Python package.
  pip install --upgrade setuptools wheel
  python "tensorflow_federated/tools/python_package/setup.py" bdist_wheel \
      --plat-name=manylinux_2_31_x86_64
  cp "dist/"* "${output_dir}"

  # Check the Python package sizes.
  local package="$(ls "dist/tensorflow_federated-"*".whl" | head -n1)"
  local actual_size="$(du -b "${package}" | cut -f1)"
  local maximum_size=80000000  # 80 MiB
  if [[ "${actual_size}" -ge "${maximum_size}" ]]; then
    echo "Error: Expected '${package}' to be less than '${maximum_size}' bytes, it was '${actual_size}' bytes." 1>&2
    exit 1
  fi
}

main "$@"
