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

# --- begin runfiles.bash initialization v2 ---
# Copy-pasted from the Bazel Bash runfiles library v2.
set -uo pipefail; f=bazel_tools/tools/bash/runfiles/runfiles.bash
#shellcheck source=/dev/null
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(dirname "$0")/$f" 2>/dev/null || \
  source "$(dirname "$(realpath "$0")")/$f" 2>/dev/null || \
  { echo >&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v2 ---

# Ensure RUNFILES_DIR is populated and exported if runfiles.bash didn't set it.
if [[ -z "${RUNFILES_DIR:-}" ]]; then
  if [[ -d "$0.runfiles" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  elif [[ -d "$(dirname "$0")/.runfiles" ]]; then
    export RUNFILES_DIR="$(dirname "$0")/.runfiles"
  elif [[ -d "$(dirname "$(realpath "$0")").runfiles" ]]; then
    export RUNFILES_DIR="$(dirname "$(realpath "$0")").runfiles"
  fi
fi

runfiles_export_envvars

usage() {
  local script_name=$(basename "${0}")
  echo "usage: ${script_name} --output_dir=<path>"
  echo "  --output_dir=<path>  An optional output directory (defaults to"
  echo "                       '{BUILD_WORKING_DIRECTORY}/dist')."
}

main() {
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
        echo "error: unrecognized option '${option}'" 1>&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z "${output_dir}" ]]; then
    local -r build_working_directory="${BUILD_WORKING_DIRECTORY:-}"
    if [[ -z "${build_working_directory}" ]]; then
      echo "error: --output_dir must be specified if not running under Bazel" 1>&2
      exit 1
    fi
    output_dir="${build_working_directory}/dist"
  fi

  # Load shared python helper from runfiles.
  local helper_path="$(rlocation org_tensorflow_federated/tools/python_package/get_hermetic_python.sh)"
  if [[ -z "${helper_path}" || ! -f "${helper_path}" ]]; then
    echo "ERROR: get_hermetic_python.sh not found in runfiles" >&2
    exit 1
  fi
  source "${helper_path}"

  # Check the GLIBC version.
  local expected_glibc="2.27"
  local current_glibc=$(ldd --version | head -n1 | awk '{print $NF}')

  if [[ -z "${current_glibc}" ]] || ! version_ge "${current_glibc}" "${expected_glibc}"; then
    echo "error: expected GLIBC version to be at least '${expected_glibc}', found: ${current_glibc}" 1>&2
    ldd --version 1>&2
    exit 1
  fi

  mkdir -p "${output_dir}"

  local python_cmd=$(setup_hermetic_python "${RUNFILES_DIR:-}")

  # Create a temp directory.
  local temp_dir="$(mktemp --directory)"
  trap "rm -rf ${temp_dir}" EXIT

  # Create a Python environment.
  "${python_cmd}" -m venv "${temp_dir}/venv"
  source "${temp_dir}/venv/bin/activate"

  # Unset PYTHONHOME to avoid conflicts with venv
  if [[ "${python_cmd}" != "python3" ]]; then
    unset PYTHONHOME
  fi
  python --version
  pip install --extra-index-url https://pypi.org/simple --upgrade "pip"
  pip --version

  # Copy package sources and C++ binaries from runfiles to a temp src directory.
  # This ensures we package the C++ binaries (.so) which are only present in runfiles.
  local src_dir="${temp_dir}/src"
  mkdir -p "${src_dir}"

  if [[ -z "${RUNFILES_DIR:-}" || ! -d "${RUNFILES_DIR:-}" ]]; then
     echo "error: runfiles directory not found, cannot package C++ binaries" 1>&2
     exit 1
  fi

  local org_tff_dir="${RUNFILES_DIR:-}/org_tensorflow_federated"

  if [[ -z "${org_tff_dir}" || ! -d "${org_tff_dir}" ]]; then
     echo "error: org_tensorflow_federated not found in runfiles" 1>&2
     exit 1
  fi

  echo "Copying package files from ${org_tff_dir}..."
  cp -RL "${org_tff_dir}/tensorflow_federated" "${src_dir}/"
  cp -L "${org_tff_dir}/pyproject.toml" "${src_dir}/"
  cp -L "${org_tff_dir}/LICENSE" "${src_dir}/"
  cp -L "${org_tff_dir}/README.md" "${src_dir}/"

  echo "Stripping C++ binaries..."
  find "${src_dir}/tensorflow_federated" -name "*.so" -exec chmod +w {} \; -exec strip -g {} \;

  # Build the Python package in the temp src directory.
  cd "${src_dir}"
  pip install --extra-index-url https://pypi.org/simple --upgrade "build" "setuptools" "wheel"
  pip freeze
  python -m build --no-isolation --outdir "${temp_dir}/dist"

  # Copy the generated wheel to the output directory.
  cp "${temp_dir}/dist"/* "${output_dir}/"
}

main "$@"
