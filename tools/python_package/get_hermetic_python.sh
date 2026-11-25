#!/usr/bin/env bash
# Copyright 2026, The TensorFlow Federated Authors.
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

# Returns 0 if $1 >= $2, 1 otherwise.
version_ge() {
  local current="$1"
  local expected="$2"
  [[ "$(printf '%s\n%s\n' "${expected}" "${current}" | sort -V | head -n1)" == "${expected}" ]]
}

# Finds the hermetic Python executable in Bazel runfiles (or falls back to
# python3), verifies Python >= 3.12 and < 3.14, and prints the path to stdout.
setup_hermetic_python() {
  local runfiles_dir="${1:-}"
  local hermetic_python=""
  if [[ -n "${runfiles_dir}" && -d "${runfiles_dir}" ]]; then
    if [[ -x "${runfiles_dir}/python/bin/python3" ]]; then
      hermetic_python="${runfiles_dir}/python/bin/python3"
    elif [[ -x "${runfiles_dir}/python/bin/python" ]]; then
      hermetic_python="${runfiles_dir}/python/bin/python"
    else
      local dir
      for dir in "${runfiles_dir}"/*python_3_1[23]* "${runfiles_dir}"/python_* "${runfiles_dir}"/*python*; do
        if [[ -d "${dir}" ]]; then
          if [[ -x "${dir}/bin/python3" ]]; then
            hermetic_python="${dir}/bin/python3"
            break
          elif [[ -x "${dir}/bin/python" ]]; then
            hermetic_python="${dir}/bin/python"
            break
          fi
        fi
      done
    fi
  fi

  local python_cmd
  if [[ -n "${hermetic_python}" ]]; then
    python_cmd="${hermetic_python}"
    echo "Using hermetic Python: ${python_cmd}" >&2
  else
    python_cmd="python3"
  fi

  local python_version
  python_version=$("${python_cmd}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
  if ! version_ge "${python_version}" "3.12" || version_ge "${python_version}" "3.14"; then
    echo "error: This script requires Python >= 3.12 and < 3.14. Found Python ${python_version}." >&2
    exit 1
  fi

  echo "${python_cmd}"
}
