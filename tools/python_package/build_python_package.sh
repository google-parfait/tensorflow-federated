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
  glibc_version=$(ldd --version 2>&1 | grep "GLIBC" | awk '{print $NF}')

  # Error handling if GLIBC version couldn't be determined.
  if [[ -z "$glibc_version" ]]; then
    echo "error: Could not determine GLIBC version." 1>&2
    exit 1
  fi

  echo "Detected GLIBC version: $glibc_version"

  # Extract major and minor version numbers for manylinux tag.
  IFS='.' read -r glibc_major glibc_minor <<< "$glibc_version"
  manylinux_version="${glibc_major}_${glibc_minor}"

  # Detect architecture.
  arch=$(uname -m)
  case "$arch" in
    aarch64|x86_64) ;; # Supported architectures
    *) echo "error: Unsupported architecture: $arch" >&2; exit 1 ;;
  esac

  plat_name="manylinux_${manylinux_version}_${arch}"


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
  pip install --upgrade "build" "toml-cli"

  # Update wheel platform
  toml set --toml-path "pyproject.toml" "tool.distutils.bdist_wheel.plat-name" "$plat_name"
  pip freeze
  python -m build --outdir "${output_dir}"
}

main "$@"
