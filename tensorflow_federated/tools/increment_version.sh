#!/usr/bin/env bash
# Copyright 2024, The TensorFlow Federated Authors.
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

unreleased_notes() {
  sed --silent '0,/^# Release /p' "${release_file}" \
      | head --lines=-1
}

latest_version() {
  sed --silent --regexp-extended 's/^# Release (.*)$/\1/p' "${release_file}" \
      | head --lines=1
}

usage() {
  local script_name=$(basename "${0}")
  local options=(
      "[VERSION]"
  )
  echo "usage: ${script_name} ${options[@]}"
  exit 1
}

main() {
  # Parse the arguments.
  local version="$@"
  if [[ -z "${version}" ]]; then
    echo "error: Expected a VERSION." 1>&2
    usage
  fi

  local release_file="tensorflow_federated/RELEASE.md"
  if [[ ! -f "${release_file}" ]]; then
    echo "error: Expected '${release_file}' to exist." 1>&2
  fi

  # Grab the latest version before begining to update the version.
  local latest_version="$(latest_version)"
  echo "Updating version from ${latest_version} to ${version}"

  # Confirm release notes.
  local unreleased_notes="$(unreleased_notes)"
  echo "${unreleased_notes}" | sed "s/# Unreleased/# Release ${version}/g"
  read -r -p "Do these release notes look correct (yes/no)? "
  if [[ "${REPLY}" != "yes" ]]; then
    exit 1
  fi

  # Update RELEASE.md.
  sed --in-place \
      "s/# Unreleased/# Unreleased\n\n# Release ${version}/g" \
      "${release_file}"

  # Update version.
  find "tensorflow_federated/" -type f \
      -not -path "${release_file}" \
      | xargs sed --in-place "s/${latest_version}/${version}/g"
}

main "$@"
