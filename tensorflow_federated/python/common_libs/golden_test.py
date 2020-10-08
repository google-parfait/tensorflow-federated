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
"""Tests for `golden` library."""

from absl.testing import absltest
from absl.testing import flagsaver

from tensorflow_federated.python.common_libs import golden


class GoldenTest(absltest.TestCase):

  def test_check_string_succeeds(self):
    golden.check_string('test_check_string_succeeds.expected',
                        'foo\nbar\nbaz\nfizzbuzz')

  def test_check_string_fails(self):
    with self.assertRaises(golden.MismatchedGoldenError):
      golden.check_string('test_check_string_fails.expected',
                          'not\nwhat\nyou\nexpected')

  def test_check_string_updates(self):
    filename = 'test_check_string_updates.expected'
    golden_path = golden._filename_to_golden_path(filename)
    old_contents = 'old\ndata\n'
    new_contents = 'new\ndata\n'
    # Attempt to reset the contents of the file to their checked-in state.
    try:
      with open(golden_path, 'w') as f:
        f.write(old_contents)
    except (OSError, PermissionError):
      # We're running without `--test_strategy=local`, and so can't test
      # updates properly because these files are read-only.
      return
    # Check for a mismatch when `--update_goldens` isn't passed.
    with self.assertRaises(golden.MismatchedGoldenError):
      golden.check_string(filename, new_contents)
    # Rerun with `--update_goldens`.
    with flagsaver.flagsaver(update_goldens=True):
      golden.check_string(filename, new_contents)
    # Check again without `--update_goldens` now that they have been updated.
    try:
      golden.check_string(filename, new_contents)
    except golden.MismatchedGoldenError as e:
      self.fail(f'Unexpected mismatch after update: {e}')
    # Reset the contents of the file to their checked-in state.
    with open(golden_path, 'w') as f:
      f.write(old_contents)

  def test_check_raises_traceback(self):
    with golden.check_raises_traceback('test_check_raises_traceback.expected',
                                       RuntimeError):
      raise RuntimeError()


if __name__ == '__main__':
  absltest.main()
