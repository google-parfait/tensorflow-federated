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

# Initialize Bazel runfiles when running under Bazel; allow standalone runs.
f=bazel_tools/tools/bash/runfiles/runfiles.bash
# shellcheck source=/dev/null
if source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
   source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
   source "$0.runfiles/$f" 2>/dev/null || \
   source "$(dirname "$0")/$f" 2>/dev/null || \
   source "$(dirname "$(realpath "$0")")/$f" 2>/dev/null; then
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
fi
unset f

usage() {
  local script_name
  script_name=$(basename "${0}")
  echo "usage: ${script_name} --package=<path> [--max_glibc=<major.minor>]"
  echo "  --package=<path>          A path to a local Python package."
  echo "  --max_glibc=<major.minor> Optional maximum Linux GLIBC symbol version."
}

verify_linux_elf_symbols() {
  local pkg_dir="$1"
  if ! command -v readelf >/dev/null 2>&1 || ! command -v nm >/dev/null 2>&1; then
    echo "error: readelf and nm (binutils) are required for Linux ELF checks" 1>&2
    exit 1
  fi

  local tf_so_matches=("${pkg_dir}/cc/core/impl/executors"/tensorflow_bindings*.so)
  local exec_so_matches=("${pkg_dir}/cc/core/impl/executors"/executor_bindings*.so)
  local stack_so_matches=("${pkg_dir}/cc/core/impl/executor_stacks"/executor_stack_bindings*.so)

  if [[ "${#tf_so_matches[@]}" -ne 1 || ! -f "${tf_so_matches[0]}" ]]; then
    echo "error: expected one tensorflow_bindings*.so under ${pkg_dir}" 1>&2
    exit 1
  fi
  if [[ "${#exec_so_matches[@]}" -ne 1 || ! -f "${exec_so_matches[0]}" ]]; then
    echo "error: expected one executor_bindings*.so under ${pkg_dir}" 1>&2
    exit 1
  fi
  if [[ -e "${stack_so_matches[0]}" ]]; then
    echo "error: unexpected executor_stack_bindings*.so under ${pkg_dir}" 1>&2
    exit 1
  fi

  local tf_so="${tf_so_matches[0]}"
  local exec_so="${exec_so_matches[0]}"

  local dynamic_info
  dynamic_info="$(readelf -d "${tf_so}")"
  if ! grep -Eq '\((RUNPATH|RPATH)\).*\$ORIGIN/\.\./\.\./\.\./\.\./\.\./tensorflow' <<< "${dynamic_info}"; then
    echo "error: ${tf_so} missing expected \$ORIGIN/../../../../../tensorflow RPATH/RUNPATH" 1>&2
    echo "${dynamic_info}" 1>&2
    exit 1
  fi
  if ! grep -Fq 'libtensorflow_framework.so.2' <<< "${dynamic_info}"; then
    echo "error: ${tf_so} missing NEEDED libtensorflow_framework.so.2" 1>&2
    echo "${dynamic_info}" 1>&2
    exit 1
  fi
  if ! grep -Fq 'libtensorflow_cc.so.2' <<< "${dynamic_info}"; then
    echo "error: ${tf_so} missing NEEDED libtensorflow_cc.so.2" 1>&2
    echo "${dynamic_info}" 1>&2
    exit 1
  fi

  local tf_undef_syms
  tf_undef_syms="$(nm -D -C --undefined-only "${tf_so}")"
  if ! grep -Eq 'MixingHashState.*kSeed' <<< "${tf_undef_syms}"; then
    echo "error: ${tf_so} missing undefined MixingHashState::kSeed symbol" 1>&2
    exit 1
  fi
  if grep -Fq 'DescriptorPool::generated_pool' <<< "${tf_undef_syms}"; then
    echo "error: ${tf_so} unexpectedly has undefined DescriptorPool::generated_pool" 1>&2
    exit 1
  fi

  local so_path
  local init_sym
  for so_path in "${exec_so}" "${tf_so}"; do
    if [[ "${so_path}" == "${exec_so}" ]]; then
      init_sym="PyInit_executor_bindings"
    else
      init_sym="PyInit_tensorflow_bindings"
    fi
    local defined_syms
    defined_syms="$(nm -D -C --defined-only "${so_path}")"
    if ! grep -Fq "${init_sym}" <<< "${defined_syms}"; then
      echo "error: ${so_path} missing exported ${init_sym}" 1>&2
      echo "${defined_syms}" 1>&2
      exit 1
    fi
    if grep -Eq 'tensorflow_federated::|google::protobuf::' <<< "${defined_syms}"; then
      echo "error: ${so_path} leaks hidden C++ symbols" 1>&2
      grep -E 'tensorflow_federated::|google::protobuf::' <<< "${defined_syms}" | head -n 20 1>&2
      exit 1
    fi
    if readelf -S -W "${so_path}" | grep -Eq '\s\.symtab\s'; then
      echo "error: ${so_path} was not stripped (.symtab still present)" 1>&2
      exit 1
    fi
  done

  echo "Verified Linux ELF RPATH, NEEDED, visibility, and stripped symbol checks."
}

main() {
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

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  local helper_path=""
  local test_script=""
  local verify_script=""
  if declare -F rlocation >/dev/null 2>&1; then
    helper_path="$(rlocation org_tensorflow_federated/tools/python_package/get_hermetic_python.sh || true)"
    test_script="$(rlocation org_tensorflow_federated/tools/python_package/test_computation.py || true)"
    verify_script="$(rlocation org_tensorflow_federated/tools/python_package/verify_python_package.py || true)"
  fi
  if [[ -z "${helper_path}" || ! -f "${helper_path}" ]]; then
    helper_path="${script_dir}/get_hermetic_python.sh"
  fi
  if [[ -z "${test_script}" || ! -f "${test_script}" ]]; then
    test_script="${script_dir}/test_computation.py"
  fi
  if [[ -z "${verify_script}" || ! -f "${verify_script}" ]]; then
    verify_script="${script_dir}/verify_python_package.py"
  fi

  if [[ -z "${helper_path}" || ! -f "${helper_path}" ]]; then
    echo "error: get_hermetic_python.sh not found" 1>&2
    exit 1
  fi
  if [[ -z "${test_script}" || ! -f "${test_script}" ]]; then
    echo "error: test_computation.py not found" 1>&2
    exit 1
  fi
  if [[ -z "${verify_script}" || ! -f "${verify_script}" ]]; then
    echo "error: verify_python_package.py not found" 1>&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${helper_path}"

  local python_cmd
  python_cmd=$(setup_hermetic_python "${runfiles_dir}")

  local package=""
  local max_glibc=""

  while [[ "$#" -gt 0 ]]; do
    local option="$1"
    case "${option}" in
      --package=*)
        package="${option#*=}"
        shift
        ;;
      --max_glibc=*)
        max_glibc="${option#*=}"
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
  local actual_size
  actual_size=$(wc -c < "${package}" | tr -d ' ')
  local maximum_size=20000000  # 20 MB
  if [[ "${actual_size}" -ge "${maximum_size}" ]]; then
    echo "error: expected '${package}' to be less than '${maximum_size}' bytes, it was '${actual_size}' bytes" 1>&2
    exit 1
  fi

  # Verify wheel tags, metadata, and contents when testing a .whl file.
  if [[ "${package}" == *.whl ]]; then
    "${python_cmd}" "${verify_script}" wheel "${package}"
  fi

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

  # Test the Python package.
  pip install --upgrade "${package}"
  pip freeze

  if [[ "$(uname -s)" == "Linux" ]]; then
    local tff_pkg_dir
    tff_pkg_dir="$(python -I -c "import os, tensorflow_federated as tff; print(os.path.dirname(tff.__file__))")"
    verify_linux_elf_symbols "${tff_pkg_dir}"
  fi

  python -I -c "import tensorflow_federated as tff; print(tff.__version__)"
  python -I "${test_script}"
  if [[ -n "${max_glibc}" ]]; then
    python -I "${verify_script}" bindings "--max_glibc=${max_glibc}"
  else
    python -I "${verify_script}" bindings
  fi
}

main "$@"
