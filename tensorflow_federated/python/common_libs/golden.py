# Copyright 2020, The TensorFlow Federated Authors.
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
"""Utilities for asserting against golden files."""

import difflib
import os.path
from typing import Dict, Optional

from absl import flags

flags.DEFINE_multi_string('golden', [], 'List of golden files available.')
flags.DEFINE_bool('update_goldens', False, 'Set true to update golden files.')
flags.DEFINE_bool('verbose', False, 'Set true to show golden diff output.')

FLAGS = flags.FLAGS


class MismatchedGoldenError(RuntimeError):
  pass


_filename_to_golden_map: Optional[Dict[str, str]] = None


def _filename_to_golden_path(filename: str) -> str:
  """Retrieve the `--golden` path flag for a golden file named `filename`."""
  # Parse out all of the golden files once
  global _filename_to_golden_map
  if _filename_to_golden_map is None:
    _filename_to_golden_map = {}
    for golden_path in FLAGS.golden:
      name = os.path.basename(golden_path)
      if name in _filename_to_golden_map:
        raise RuntimeError(
            f'Multiple golden files provided for filename {name}:\n'
            f'{golden_path} and\n'
            f'{_filename_to_golden_map[name]}\n'
            'Golden file names in the same test target must be unique.')
      _filename_to_golden_map[name] = golden_path
  return _filename_to_golden_map[filename]


def check_string(filename: str, value: str):
  """Check that `value` matches the contents of the golden file `filename`."""
  # Append a newline to the end of `value` to work around lint against
  # text files with no trailing newline.
  if not value.endswith('\n'):
    value = value + '\n'
  golden_path = _filename_to_golden_path(filename)
  if FLAGS.update_goldens:
    with open(golden_path, 'w') as f:
      f.write(value)
    return
  with open(golden_path, 'r') as f:
    golden_contents = f.read()
  if value == golden_contents:
    return
  message = (f'The contents of golden file {filename} '
             'no longer match the current value.\n'
             'To update the golden file, rerun this target with:\n'
             '`--test_arg=--update_goldens --test_strategy=local`\n')
  if FLAGS.verbose:
    message += 'Full diff:\n'
    split_value = value.split('\n')
    split_golden = golden_contents.split('\n')
    message += '\n'.join(difflib.unified_diff(split_golden, split_value))
  else:
    message += 'To see the full diff, rerun this target with:\n'
    message += '`--test_arg=--verbose\n'
  raise MismatchedGoldenError(message)
