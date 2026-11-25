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
# shellcheck source=/dev/null
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(dirname "$0")/$f" 2>/dev/null || \
  source "$(dirname "$(realpath "$0")")/$f" 2>/dev/null || \
  { echo >&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v2 ---

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
  local script_name
  script_name=$(basename "${0}")
  echo "usage: ${script_name} --output_dir=<path>"
  echo "  --output_dir=<path>  An optional output directory (defaults to"
  echo "                       '{BUILD_WORKING_DIRECTORY}/dist')."
}

main() {
  local output_dir=""

  while [[ "$#" -gt 0 ]]; do
    local option="$1"
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

  local runfiles_dir="${RUNFILES_DIR:-}"
  if [[ -z "${runfiles_dir}" ]]; then
    if [[ -d "${0}.runfiles" ]]; then
      runfiles_dir="${0}.runfiles"
    elif [[ -d "${BASH_SOURCE[0]}.runfiles" ]]; then
      runfiles_dir="${BASH_SOURCE[0]}.runfiles"
    else
      local dir_name
      dir_name=$(dirname "${BASH_SOURCE[0]}")
      if [[ "${dir_name}" == *.runfiles* ]]; then
        runfiles_dir="${dir_name%%.runfiles*}.runfiles"
      fi
    fi
  fi

  if [[ -z "${runfiles_dir}" || ! -d "${runfiles_dir}" ]]; then
    echo "error: runfiles directory not found, cannot package C++ bindings" 1>&2
    exit 1
  fi

  local helper_path=""
  local org_tff_dir=""
  helper_path="$(rlocation org_tensorflow_federated/tools/python_package/get_hermetic_python.sh)"
  local pyproject_path
  pyproject_path="$(rlocation org_tensorflow_federated/pyproject.toml)"
  org_tff_dir="$(dirname "${pyproject_path}")"

  if [[ -z "${helper_path}" || ! -f "${helper_path}" ]]; then
    echo "error: get_hermetic_python.sh not found in runfiles" 1>&2
    exit 1
  fi
  if [[ -z "${org_tff_dir}" || ! -d "${org_tff_dir}/tensorflow_federated" ]]; then
    echo "error: tensorflow_federated package files not found in runfiles" 1>&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${helper_path}"

  # Check the GLIBC version on Linux.
  if [[ "$(uname -s)" == "Linux" ]]; then
    local expected_glibc="2.27"
    local current_glibc
    current_glibc=$(ldd --version | awk 'NR == 1 {print $NF}')
    if [[ -z "${current_glibc}" ]] || ! version_ge "${current_glibc}" "${expected_glibc}"; then
      echo "error: expected GLIBC version to be at least '${expected_glibc}', found: ${current_glibc}" 1>&2
      ldd --version 1>&2
      exit 1
    fi
  fi

  mkdir -p "${output_dir}"
  output_dir="$(cd "${output_dir}" && pwd)"

  local python_cmd
  python_cmd=$(setup_hermetic_python "${runfiles_dir}")

  # Create a temp directory.
  local temp_dir
  temp_dir="$(mktemp -d)"
  trap "rm -rf \"${temp_dir}\"" EXIT

  # Create a Python environment.
  "${python_cmd}" -m venv "${temp_dir}/venv"
  # shellcheck source=/dev/null
  source "${temp_dir}/venv/bin/activate"

  python --version
  pip install --upgrade "pip"
  pip --version

  # Stage package sources and C++ extension libraries from runfiles.
  local src_dir="${temp_dir}/src"
  mkdir -p "${src_dir}"

  echo "Copying package files from ${org_tff_dir}..."
  cp -RL "${org_tff_dir}/tensorflow_federated" "${src_dir}/"
  cp -L "${org_tff_dir}/pyproject.toml" "${src_dir}/"
  cp -L "${org_tff_dir}/setup.py" "${src_dir}/"
  cp -L "${org_tff_dir}/LICENSE" "${src_dir}/"
  cp -L "${org_tff_dir}/README.md" "${src_dir}/"
  chmod -R u+w "${src_dir}/tensorflow_federated"

  # Verify excluded artifacts are not staged into the wheel source tree.
  if find "${src_dir}/tensorflow_federated" -name "worker_binary" -print -quit | grep -q .; then
    echo "error: unexpected worker_binary found in staged package" 1>&2
    exit 1
  fi
  if [[ -d "${src_dir}/tensorflow_federated/cc/core/impl/aggregation" ]]; then
    echo "error: unexpected cc/core/impl/aggregation found in staged package" 1>&2
    exit 1
  fi
  if find "${src_dir}/tensorflow_federated/cc" -name "*.h" -print -quit | grep -q .; then
    echo "error: unexpected C++ header found under cc/ in staged package" 1>&2
    exit 1
  fi

  echo "Stripping C++ extension libraries..."
  if [[ "$(uname -s)" == "Darwin" ]]; then
    find "${src_dir}/tensorflow_federated" -type f -name "*.so" -exec strip -x {} +
  else
    find "${src_dir}/tensorflow_federated" -type f -name "*.so" -exec strip --strip-unneeded {} +
  fi

  # Build the Python package.
  cd "${src_dir}"
  pip install --upgrade "build" "setuptools" "wheel"
  pip freeze
  local build_args=(--no-isolation --wheel --outdir "${temp_dir}/dist")
  if [[ "$(uname -s)" == "Linux" ]]; then
    local arch
    arch="$(uname -m)"
    build_args+=(
        "--config-setting=--build-option=--plat-name=manylinux_2_27_${arch}"
    )
  fi
  python -m build "${build_args[@]}"

  local wheels=("${temp_dir}/dist"/tensorflow_federated-*-cp31[23]-cp31[23]-*.whl)
  if [[ "${#wheels[@]}" -ne 1 || ! -f "${wheels[0]}" ]]; then
    echo "error: expected one CPython 3.12/3.13 wheel in ${temp_dir}/dist" 1>&2
    ls -la "${temp_dir}/dist" 1>&2
    exit 1
  fi
  cp "${wheels[0]}" "${output_dir}/"
}

main "$@"
