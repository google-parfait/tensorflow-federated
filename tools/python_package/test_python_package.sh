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
  echo "usage: ${script_name} --package=<path>"
  echo "  --package=<path>  A path to a local Python package."
}
main() {
  # Load shared python helper from runfiles.
  local helper_path="$(rlocation org_tensorflow_federated/tools/python_package/get_hermetic_python.sh)"
  if [[ -z "${helper_path}" || ! -f "${helper_path}" ]]; then
    echo "ERROR: get_hermetic_python.sh not found in runfiles" >&2
    exit 1
  fi
  source "${helper_path}"

  local python_cmd=$(setup_hermetic_python "${RUNFILES_DIR:-}")

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
  local actual_size=$(wc -c < "${package}" | tr -d ' ')
  local maximum_size=80000000  # 80 MiB
  if [[ "${actual_size}" -ge "${maximum_size}" ]]; then
    echo "error: expected '${package}' to be less than '${maximum_size}' bytes, it was '${actual_size}' bytes" 1>&2
    exit 1
  fi

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

  # Test the Python package.
  pip install --extra-index-url https://pypi.org/simple --upgrade "${package}"
  pip freeze
  python -c "import tensorflow_federated as tff; print(tff.__version__)"

  # Run a federated computation to exercise the C++ executor bindings.
  local test_script="$(rlocation org_tensorflow_federated/tools/python_package/test_computation.py)"
  if [[ -z "${test_script}" || ! -f "${test_script}" ]]; then
    echo "ERROR: test_computation.py not found in runfiles" >&2
    exit 1
  fi
  python "${test_script}"
}

main "$@"
