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
set -ex

function main() {
  local output_dir="$1"
  local project_name="$2"

  if [[ ! -d "${output_dir}" ]]; then
    echo "`${output_dir}` does not exist."
    exit 1
  fi

  if [[ -z "${project_name}" ]]; then
    project_name="tensorflow_federated"
  fi

  if [[ ! -d "bazel-bin/tensorflow_federated" ]]; then
    echo "Could not find bazel-bin. Did you run from the root of the build tree?"
    exit 1
  fi

  local temp_dir="$(mktemp --directory)"

  local runfiles="bazel-bin/tensorflow_federated/tools/build_pip_package.runfiles"
  cp --dereference --recursive \
      "${runfiles}/org_tensorflow_federated/tensorflow_federated" \
      "${temp_dir}"
  cp "${runfiles}/org_tensorflow_federated/tensorflow_federated/tools/setup.py" "${temp_dir}"

  pushd "${temp_dir}" > /dev/null
  virtualenv --python=python3 "venv"
  source "venv/bin/activate"

  pip install --upgrade pip setuptools wheel
  python setup.py bdist_wheel \
      --universal \
      --project_name "${project_name}"
  cp "dist/"* "${output_dir}"

  deactivate
  popd > /dev/null
  rm -rf "${temp_dir}"
}

main "$@"
