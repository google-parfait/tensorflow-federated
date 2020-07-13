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
# Tool to build the TensorFlow Federated client image.
#
# Usage:
#   bazel run //tensorflow_federated/tools/client:build_image
#
# Arguments:
#   artifacts_dir: A directory to use when generating intermediate artifacts,
#       which can be useful during debugging. If no directory is specified, a
#       temproary directory will be used and cleaned up when this command exits.
set -e

main() {
  local artifacts_dir="$1"

  if [[ -z "${artifacts_dir}" ]]; then
    artifacts_dir="$(mktemp -d)"
    trap "rm -rf ${artifacts_dir}" EXIT
  fi

  cp -LR "tensorflow_federated" "${artifacts_dir}"
  pushd "${artifacts_dir}"

  # Build the TensorFlow Federated package
  tensorflow_federated/tools/development/build_pip_package \
      "${artifacts_dir}"

  # Build the TensorFlow Federated runtime image
  docker build \
      --file "tensorflow_federated/tools/client/Dockerfile" \
      --tag tff-client \
      .
}

main "$@"
