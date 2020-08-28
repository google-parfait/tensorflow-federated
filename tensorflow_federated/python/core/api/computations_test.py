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
"""Integration tests for federated and tensorflow computations.

These tests test the public TFF core API surface by defining and executing
computations; tests are grouped into `TestCase`s based on the kind of
computation. Many of these tests are parameterized to test different parts of
the  TFF implementation, for example different executor stacks.
"""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test as common_test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import do_not_use_compiler
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_test_utils


class TensorFlowComputationsWithDatasetsTest(parameterized.TestCase):

  @executor_test_utils.executors
  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @common_test.skip_test_for_gpu
  def test_with_tf_datasets(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int64))
    def consume(ds):
      return ds.reduce(np.int64(0), lambda x, y: x + y)

    self.assertEqual(str(consume.type_signature), '(int64* -> int64)')

    @computations.tf_computation
    def produce():
      return tf.data.Dataset.range(10)

    self.assertEqual(str(produce.type_signature), '( -> int64*)')

    self.assertEqual(consume(produce()), 45)

  # TODO(b/131363314): The reference executor should support generating and
  # returning infinite datasets
  @executor_test_utils.executors(
      ('local', executor_stacks.local_executor_factory(1)),)
  def test_consume_infinite_tf_dataset(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int64))
    def consume(ds):
      # Consume the first 10 elements of the dataset.
      return ds.take(10).reduce(np.int64(0), lambda x, y: x + y)

    self.assertEqual(consume(tf.data.Dataset.range(10).repeat()), 45)

  # TODO(b/131363314): The reference executor should support generating and
  # returning infinite datasets
  @executor_test_utils.executors(
      ('local', executor_stacks.local_executor_factory(1)),)
  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @common_test.skip_test_for_gpu
  def test_produce_and_consume_infinite_tf_dataset(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int64))
    def consume(ds):
      # Consume the first 10 elements of the dataset.
      return ds.take(10).reduce(np.int64(0), lambda x, y: x + y)

    @computations.tf_computation
    def produce():
      # Produce an infinite dataset.
      return tf.data.Dataset.range(10).repeat()

    self.assertEqual(consume(produce()), 45)

  @executor_test_utils.executors
  def test_with_sequence_of_pairs(self):
    pairs = tf.data.Dataset.from_tensor_slices(
        (list(range(5)), list(range(5, 10))))

    @computations.tf_computation
    def process_pairs(ds):
      return ds.reduce(0, lambda state, pair: state + pair[0] + pair[1])

    self.assertEqual(process_pairs(pairs), 45)

  @executor_test_utils.executors
  def test_tf_comp_with_sequence_inputs_and_outputs_does_not_fail(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int32))
    def _(x):
      return x

  @executor_test_utils.executors
  def test_with_four_element_dataset_pipeline(self):

    @computations.tf_computation
    def comp1():
      return tf.data.Dataset.range(5)

    @computations.tf_computation(computation_types.SequenceType(tf.int64))
    def comp2(ds):
      return ds.map(lambda x: tf.cast(x + 1, tf.float32))

    @computations.tf_computation(computation_types.SequenceType(tf.float32))
    def comp3(ds):
      return ds.repeat(5)

    @computations.tf_computation(computation_types.SequenceType(tf.float32))
    def comp4(ds):
      return ds.reduce(0.0, lambda x, y: x + y)

    @computations.tf_computation
    def comp5():
      return comp4(comp3(comp2(comp1())))

    self.assertEqual(comp5(), 75.0)


if __name__ == '__main__':
  do_not_use_compiler._do_not_use_set_local_execution_context()
  common_test.main()
