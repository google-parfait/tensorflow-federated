## Copyright 2020, The TensorFlow Federated Authors.
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

import contextlib
import difflib
import io
import os.path
import re
import sys
import traceback
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
      old_path = _filename_to_golden_map.get(name)
      if old_path is not None and old_path != golden_path:
        raise RuntimeError(
            f'Multiple golden files provided for filename {name}:\n'
            f'{old_path} and\n'
            f'{golden_path}\n'
            'Golden file names in the same test target must be unique.')
      _filename_to_golden_map[name] = golden_path
  if filename not in _filename_to_golden_map:
    raise RuntimeError(f'No `--golden` files found with filename {filename}')
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


def traceback_string(exc_type, exc_value, tb) -> str:
  """Generates a standardized stringified version of an exception traceback."""
  exception_string_io = io.StringIO()
  traceback.print_exception(exc_type, exc_value, tb, file=exception_string_io)
  exception_string = exception_string_io.getvalue()
  # Strip path to TFF to normalize error messages
  # First in filepaths.
  without_filepath = re.sub(r'\/\S*\/tensorflow_federated\/', '',
                            exception_string)
  # Then also in class paths.
  without_classpath = re.sub(r'(\S*\.)+?(?=tensorflow_federated)', '',
                             without_filepath)
  # Strip line numbers to avoid churn
  without_linenumber = re.sub(r', line \d*', '', without_classpath)
  return without_linenumber


def check_raises_traceback(filename: str,
                           exception) -> contextlib.AbstractContextManager:
  """Check for `exception` to be raised, generating a golden traceback."""
  # Note: does not use `@contextlib.contextmanager` because that adds
  # this function to the traceback.
  return _TracebackManager(filename, exception)


class _TracebackManager():
  """Context manager for collecting tracebacks and comparing them to goldens."""

  def __init__(self, filename, exception):
    self._filename = filename
    self._exception = exception

  def __enter__(self):
    pass

  def __exit__(self, exc_type, exc_value, tb):
    # Note: How Python constructs tracebacks changes in Python 3.8 and later.
    # For now, we disable testing in these environments.
    if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
      return True

    if not issubclass(exc_type, self._exception):
      message = f'Exception `{self._exception.__name__}` was not thrown.'
      if exc_value is not None:
        message += ' A different exception was thrown, and can be seen above.'
      raise RuntimeError(message)
    traceback_str = traceback_string(exc_type, exc_value, tb)
    check_string(self._filename, traceback_str)
    return True
