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
"""Heavy hitter discovery TestCase."""

import tensorflow as tf


class HeavyHittersTest(tf.test.TestCase):
  """TestCase for Heavy Hitters."""

  def assertSetAllEqual(self, x, y):
    """Assert two tensor lists contain the same set of items.

    Allow items to be in different order in these two lists.

    Args:
      x: A 1D tf.string
      y: A 1D tf.string

    Raises:
      ValueError: If x and y are different sets.
    """
    x = tf.expand_dims(x, axis=0)
    y = tf.expand_dims(y, axis=0)
    self.assertEqual(
        tf.size(tf.sets.difference(x, y)), 0, msg='Input sets are not equal.')
    self.assertEqual(
        tf.size(tf.sets.difference(y, x)), 0, msg='Input sets are not equal.')
