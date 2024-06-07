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
#
# Tool to update the version in TensorFlow Federated.
set -e

usage() {
  local script_name=$(basename "${0}")
  echo "usage: ${script_name} [OPTION]... [VERSION]"
  echo "  --yes  Answer 'yes' to all prompts."
}

get_latest_version() {
  sed --silent --regexp-extended 's/^# Release (.*)$/\1/p' "${release_file}" \
      | head --lines=1
}

get_unreleased_notes() {
  sed --silent '0,/^# Release /p' "${release_file}" \
      | head --lines=-1
}

update_file() {
  local file="$@"

  # Confirm update.
  if [[ "${yes}" != "true" ]]; then
    sed "s/${latest_version//./\\.}/${version}/g" \
        "${file}" \
        | diff --unified --color "${file}" - \
        || true
    echo ""
    read -r -p "Do these changes look correct (y/n)?" < /dev/tty
    if [[ "${REPLY}" != "y" ]]; then
      return 0
    fi
  fi

  # Update file.
  sed --in-place \
      "s/${latest_version//./\\.}/${version}/g" \
      "${file}"
}

main() {
  # Parse the arguments.
  local yes="false"
  local positional_args=()

  while [[ "$#" -gt 0 ]]; do
    option="$1"
    case "${option}" in
      --yes)
        yes="true"
        shift
        ;;
      --*|-*)
        echo "error: unrecognized option '${option}'" 1>&2
        usage
        exit 1
        ;;
      *)
        positional_args+=("${option}")
        shift
        ;;
    esac
  done

  set -- "${positional_args[@]}"
  local version="$@"

  if [[ -z "${version}" ]]; then
    echo "error: expected a 'VERSION'" 1>&2
    usage
    exit 1
  fi

  local release_file="tensorflow_federated/RELEASE.md"
  if [[ ! -f "${release_file}" ]]; then
    echo "error: expected the release file '${release_file}' to exist" 1>&2
    exit 1
  fi

  # Get the latest version from `RELEASE.md` before updating `RELEASE.md`.
  local latest_version="$(get_latest_version)"

  # Confirm release notes.
  if [[ "${yes}" != "true" ]]; then
    get_unreleased_notes | sed "s/# Unreleased/# Release ${version}/"
    read -r -p "Do these release notes look correct (y/n)?"
    if [[ "${REPLY}" != "y" ]]; then
      exit 0
    fi
  fi

  # Update RELEASE.md.
  sed --in-place \
      "s/# Unreleased/# Unreleased\n\n# Release ${version}/" \
      "${release_file}"

  # Update files.
  find "tensorflow_federated/" \
      -type f \
      -not -path "${release_file}" \
      -print0 \
      | xargs -0 grep \
      --files-with-matches \
      --binary-files=without-match \
      "${latest_version//./\\.}" \
      | while IFS= read -r file; do
            update_file "${file}" &
            wait $!
        done
}

main "$@"
