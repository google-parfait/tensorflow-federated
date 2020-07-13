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
# Tool to deploy the TensorFlow Federated endpoint on GCP.
#
# Usage:
#   bazel run //tensorflow_federated/tools/runtime/gcp:deploy_endpoint
#
# Arguments:
#   artifacts_dir: A directory to use when generating intermediate artifacts,
#       which can be useful during debugging. If no directory is specified, a
#       temproary directory will be used and cleaned up when this command exits.
set -e

main() {
  local artifacts_dir="$1"

  if [[ -z "${artifacts_dir}" ]]; then
    local artifacts_dir="$(mktemp -d)"
    trap "rm -rf ${artifacts_dir}" EXIT
  fi

  cp -LR "tensorflow_federated" "${artifacts_dir}"
  pushd "${artifacts_dir}"

  # Create a virtual environment
  virtualenv --python=python3 "venv"
  source "venv/bin/activate"
  pip install --upgrade pip

  # Install gRPC
  pip install --upgrade grpcio grpcio-tools

  # Create the descriptor file
  mkdir "generated_pb2"
  python -m grpc_tools.protoc \
      --include_imports \
      --include_source_info \
      --proto_path=. \
      --descriptor_set_out="api_descriptor.pb" \
      --python_out="generated_pb2" \
      --grpc_python_out="generated_pb2" \
      "tensorflow_federated/proto/v0/executor.proto"

  # Deploy the Endpoints configuration
  gcloud endpoints services deploy \
      "api_descriptor.pb" \
      "tensorflow_federated/tools/runtime/gcp/worker.yaml"
}

main "$@"
