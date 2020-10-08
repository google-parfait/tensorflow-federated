# Copyright 2018, The TensorFlow Federated Authors.
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
"""Base class for TFF test cases."""

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types


class TestCase(tf.test.TestCase, absltest.TestCase):
  """Base class for TensroFlow Federated tests."""

  def setUp(self):
    super().setUp()
    tf.keras.backend.clear_session()

  def assert_type_assignable_from(self, target_type, source_type):
    # Reraise the exception outside of `except` so as to avoid setting
    # the `.__cause__` on the final error (repeating everything).
    message = None
    try:
      target_type.check_assignable_from(source_type)
    except computation_types.TypeNotAssignableError as e:
      message = e.message
    if message is not None:
      self.fail(message)

  def assert_types_equivalent(self, first_type, second_type):
    message = None
    try:
      first_type.check_equivalent_to(second_type)
    except computation_types.TypesNotEquivalentError as e:
      message = e.message
    if message is not None:
      self.fail(message)

  def assert_types_identical(self, first_type, second_type):
    message = None
    try:
      first_type.check_identical_to(second_type)
    except computation_types.TypesNotIdenticalError as e:
      message = e.message
    if message is not None:
      self.fail(message)

  def assert_nested_struct_eq(self, x, y):
    """Asserts that nested structures 'x' and 'y' are the same.

    Args:
      x: One nested structure.
      y: Another nested structure.

    Raises:
      ValueError: if the structures are not the same.
    """
    try:
      tf.nest.assert_same_structure(x, y)
    except ValueError:
      self.fail('Expected structures to have the same shape.')
    xl = tf.nest.flatten(x)
    yl = tf.nest.flatten(y)
    if len(xl) != len(yl):
      self.fail('The sizes of structures {} and {} mismatch.'.format(
          str(len(xl)), str(len(yl))))
    for xe, ye in zip(xl, yl):
      if xe != ye:
        self.fail('Mismatching elements {} and {}.'.format(str(xe), str(ye)))


def main():
  """Runs all unit tests with TF 2.0 features enabled.

  This function should only be used if TensorFlow code is being tested.
  """
  tf.test.main()
