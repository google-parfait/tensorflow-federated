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
"""General purpose test utils for TFF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from six.moves import zip
import tensorflow as tf


class TestCase(tf.test.TestCase, absltest.TestCase):
  """Base class for TensroFlow Federated tests."""

  def setUp(self):
    super(TestCase, self).setUp()
    tf.keras.backend.clear_session()


def main():
  """Runs all unit tests with TF 2.0 features enabled.

  This function should only be used if TensorFlow code is being tested.
  """
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()


def graph_mode_test(test_fn):
  """Decorator for a test to be executed in graph mode.

  This decorator is used to write graph-mode tests when eager execution is
  enabled.

  This introduces a default `tf.Graph`, which tests annotated with
  `@graph_mode_test` may use or ignore by creating their own Graphs.

  Args:
    test_fn: A test function to be decorated.

  Returns:
    The decorated test_fn.
  """

  def wrapped_test_fn(*args, **kwargs):
    with tf.Graph().as_default():
      test_fn(*args, **kwargs)

  return wrapped_test_fn


def assert_nested_struct_eq(x, y):
  """Asserts that nested structures 'x' and 'y' are the same.

  Args:
    x: One nested structure.
    y: Another nested structure.

  Raises:
    ValueError: if the structures are not the same.
  """
  tf.contrib.framework.nest.assert_same_structure(x, y)
  xl = tf.contrib.framework.nest.flatten(x)
  yl = tf.contrib.framework.nest.flatten(y)
  if len(xl) != len(yl):
    raise ValueError('The sizes of structures {} and {} mismatch.'.format(
        str(len(xl)), str(len(yl))))
  for xe, ye in zip(xl, yl):
    if xe != ye:
      raise ValueError('Mismatching elements {} and {}.'.format(
          str(xe), str(ye)))
