# Lint as: python3
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
"""A tool for stripping skipped cells from Colab notebooks."""

import json
import sys
from typing import Union

SKIP_ANNOTATION = '#@test {"skip": true}\n'

JsonValue = Union[dict, list, str, int, float, bool, None]


def strip_json(obj: JsonValue) -> JsonValue:
  """Remove skipped cells from a JSON object."""
  if isinstance(obj, list):
    if SKIP_ANNOTATION in obj:
      return []
    else:
      return [strip_json(x) for x in obj]
  elif isinstance(obj, dict):
    return {k: strip_json(v) for k, v in obj.items()}
  else:
    return obj


def strip_file(path: str):
  """Strip skipped cells in a Colab notebook at path `path`."""
  with open(path, 'r+') as f:
    unstripped = json.load(f)
    stripped = strip_json(unstripped)
    f.seek(0)
    json.dump(stripped, f, sort_keys=True, indent=2)
    f.truncate()


def main():
  args = sys.argv
  args.pop(0)  # remove program name
  for path in args:
    strip_file(path)


if __name__ == '__main__':
  main()
