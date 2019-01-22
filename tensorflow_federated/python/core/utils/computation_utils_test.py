# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tensorflow_federated.python.core.utils.computation_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from six import assertRaisesRegex
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.core import api as tff
from tensorflow_federated.python.core.utils import computation_utils


# Create two tff.Computations that perform sum on a sequence: initializes the
# state to 0 and add each item in a sequence to the state.
@tff.tf_computation
def initialize():
  return tf.constant(0)


@tff.tf_computation([tf.int32, tf.int32])
def add_int32(current, val):
  return current + val


@tff.tf_computation(tf.int32)
def count_int32(current):
  return current + 1


class ComputationUtilsTest(absltest.TestCase):

  def test_iterative_process_state_only(self):
    iterative_process = computation_utils.IterativeProcess(
        initialize, count_int32)

    state = iterative_process.initialize()
    iterations = 10
    for _ in range(iterations):
      # TODO(b/122321354): remove the .item() call on `state` once numpy.int32
      # type is supported.
      state = iterative_process.next(state.item())
    self.assertEqual(state, iterations)

  def test_iterative_process_state_tuple(self):
    iterative_process = computation_utils.IterativeProcess(
        initialize, add_int32)

    state = iterative_process.initialize()
    iterations = 10
    for val in range(iterations):
      # TODO(b/122321354): remove the .item() call on `state` once numpy.int32
      # type is supported.
      state = iterative_process.next(state.item(), val)
    self.assertEqual(state, sum(range(iterations)))

  def test_iterative_process_initialize_bad_type(self):
    with assertRaisesRegex(self, TypeError, r'Expected .*\.Computation, .*'):
      _ = computation_utils.IterativeProcess(
          initialize_fn=None, next_fn=add_int32)

    with assertRaisesRegex(self, TypeError,
                           r'initialize_fn must be a no-arg tff.Computation'):

      @tff.federated_computation(tf.int32)
      def one_arg_initialize(one_arg):
        del one_arg  # unused
        return tff.to_value(0)

      _ = computation_utils.IterativeProcess(
          initialize_fn=one_arg_initialize, next_fn=add_int32)

  def test_iterative_process_next_bad_type(self):
    with assertRaisesRegex(self, TypeError, r'Expected .*\.Computation, .*'):
      _ = computation_utils.IterativeProcess(
          initialize_fn=initialize, next_fn=None)

  def test_iterative_process_type_mismatch(self):
    with assertRaisesRegex(self, TypeError,
                           r'The return type of initialize_fn should match.*'):

      @tff.federated_computation([tf.float32, tf.float32])
      def add_float32(current, val):
        return current + val

      _ = computation_utils.IterativeProcess(
          initialize_fn=initialize, next_fn=add_float32)

    with assertRaisesRegex(
        self, TypeError,
        'The return type of next_fn should match the first parameter'):

      @tff.federated_computation(tf.int32)
      def add_bad_result(_):
        return 0.0

      _ = computation_utils.IterativeProcess(
          initialize_fn=initialize, next_fn=add_bad_result)


if __name__ == '__main__':
  absltest.main()
