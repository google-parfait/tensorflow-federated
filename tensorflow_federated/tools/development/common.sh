#!/usr/bin/env bash
# Copyright 2021, The TensorFlow Federated Authors.
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
set -e

error_directory_does_not_exist() {
  echo "error: the directory '$@' does not exist" 1>&2
}

error_exclusive() {
  local options=()
  for option in "$@"; do
    options+=("and '${option}'")
  done
  options="${options[@]}"
  options="${options#and }"
  echo "error: can not specify both options ${options}" 1>&2
}

error_file_does_not_exist() {
  echo "error: the file '$@' does not exist" 1>&2
}

error_required() {
  local options=()
  for option in "$@"; do
    options+=("or '${option}'")
  done
  options="${options[@]}"
  options="${options#or }"
  echo "error: required option ${options}" 1>&2
}

error_unrecognized() {
  echo "error: unrecognized option '$@'" 1>&2
}
