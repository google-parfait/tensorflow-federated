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
  echo "usage: ${script_name} --output_dir=<path>"
  echo "  --output_dir=<path>  An optional output directory (defaults to"
  echo "                       '{BUILD_WORKING_DIRECTORY}/dist')."
}

# Returns 0 if $1 >= $2, 1 otherwise.
#
# This function uses `sort -V` (version sort) to compare version strings.
# `sort -V` orders version numbers naturally (e.g., 2.27 < 2.31).
# By printing the expected version and the current version, and piping to `sort -V`,
# the smaller version will appear first. If the expected version is the first line
# of the sorted output, then the current version is greater than or equal to the expected version.
version_ge() {
  local current="$1"
  local expected="$2"
  [[ "$(printf '%s\n%s\n' "${expected}" "${current}" | sort -V | head -n1)" == "${expected}" ]]
}

main() {
  # Parse the arguments.
  local output_dir="${BUILD_WORKING_DIRECTORY}/dist"

  while [[ "$#" -gt 0 ]]; do
    option="$1"
    case "${option}" in
      --output_dir=*)
        output_dir="${option#*=}"
        shift
        ;;
      *)
        echo "error: unrecognized option '${option}'" 1>&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z "${output_dir}" ]]; then
    echo "error: expected an 'output_dir'" 1>&2
    usage
    exit 1
  fi

  # Check the GLIBC version.
  local expected_glibc="2.27"
  local current_glibc=$(ldd --version | head -n1 | awk '{print $NF}')

  if [[ -z "${current_glibc}" ]] || ! version_ge "${current_glibc}" "${expected_glibc}"; then
    echo "error: expected GLIBC version to be at least '${expected_glibc}', found: ${current_glibc}" 1>&2
    ldd --version 1>&2
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

  # Build the Python package.
  pip install --upgrade "build"
  pip freeze
  python -m build --outdir "${output_dir}"
}

main "$@"
