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
#   bazel build //tensorflow_federated/tools/development:build_pip_package
#   bazel-bin/tensorflow_federated/tools/development/build_pip_package "/tmp/tensorflow_federated"
#
# Arguments:
#   output_dir: An output directory directory.
#   project_name: A project name, defaults to `tensorflow_federated`.
set -e

die() {
  echo >&2 "$@"
  exit 1
}

main() {
  local output_dir="$1"
  local project_name="$2"

  if [[ ! -d "${output_dir}" ]]; then
    die "The output directory '${output_dir}' does not exist."
  fi

  if [[ -z "${project_name}" ]]; then
    project_name="tensorflow_federated"
  fi

  if [[ ! -d "bazel-bin/tensorflow_federated" ]]; then
    die "Could not find bazel-bin. Did you run from the root of the build tree?"
  fi

  local temp_dir="$(mktemp -d)"
  trap "rm -rf ${temp_dir}" EXIT

  local runfiles="bazel-bin/tensorflow_federated/tools/development/build_pip_package.runfiles"
  cp -LR \
      "${runfiles}/org_tensorflow_federated/tensorflow_federated" \
      "${temp_dir}"
  cp "${runfiles}/org_tensorflow_federated/tensorflow_federated/development/build/setup.py" \
      "${temp_dir}"

  pushd "${temp_dir}" > /dev/null

  # Create a virtual environment
  virtualenv --python=python3 "venv"
  source "venv/bin/activate"
  pip install --upgrade pip

  # Build pip package
  pip install --upgrade setuptools wheel
  python setup.py bdist_wheel \
      --universal \
      --project_name "${project_name}"
  cp "dist/"* "${output_dir}"
}

main "$@"
