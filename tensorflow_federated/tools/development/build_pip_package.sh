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
#
# Usage:
#   bazel run //tensorflow_federated/tools/development:build_pip_package -- \
#       --output_dir "/tmp/tensorflow_federated"
#   bazel run //tensorflow_federated/tools/development:build_pip_package -- \
#       --nightly \
#       --output_dir "/tmp/tensorflow_federated"
#
# Arguments:
#   nightly: A flag indicating whether or not to build the nightly version of
#     the pip package.
#   output_dir: An output directory.
set -e

die() {
  echo >&2 "$@"
  exit 1
}

usage() {
  local script_name=$(basename "${0}")
  echo "usage: ${script_name} [--nightly] [--output_dir PATH]" 1>&2
}

main() {
  # Parse arguments
  local nightly=0
  local output_dir=""

  while [[ "$#" -gt 0 ]]; do
    opt="$1"
    case "${opt}" in
      --nightly)
        nightly="1"
        shift
        ;;
      --output_dir)
        output_dir="$2"
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

  if [[ -z ${output_dir} ]]; then
    usage
    exit 1
  fi

  if [[ ! -d "${output_dir}" ]]; then
    die "The output directory '${output_dir}' does not exist."
  fi

  # Create working directory
  local temp_dir="$(mktemp -d)"
  trap "rm -rf ${temp_dir}" EXIT
  cp -LR "tensorflow_federated" "${temp_dir}"
  pushd "${temp_dir}"

  # Create a virtual environment
  virtualenv --python=python3.6 "venv"
  source "venv/bin/activate"
  pip install --upgrade pip

  # Build pip package
  flags=()
  if [[ ${nightly} == "1" ]]; then
    flags+=("--nightly")
  fi
  pip install --upgrade setuptools wheel
  python "tensorflow_federated/tools/development/setup.py" bdist_wheel \
      --universal \
      "${flags[@]}"
  popd

  cp "${temp_dir}/dist/"* "${output_dir}"
}

main "$@"
