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

import collections

import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import tensorflow_serialization


class IntegrationTest(test.TestCase):

  def test_fetch_value_with_nested_datasets(self):

    def return_two_datasets():
      return [tf.data.Dataset.range(5), tf.data.Dataset.range(5)]

    executable_return_two_datasets = computation_impl.ComputationImpl(
        tensorflow_serialization.serialize_py_fn_as_tf_computation(
            return_two_datasets, None, context_stack_impl.context_stack)[0],
        context_stack_impl.context_stack)

    x = executable_return_two_datasets()
    self.assertEqual([i for i in iter(x[0])], list(range(5)))
    self.assertEqual([i for i in iter(x[1])], list(range(5)))

  def test_fetch_value_with_dataset_and_tensor(self):

    def return_dataset_and_tensor():
      return [tf.constant(0), tf.data.Dataset.range(5), tf.constant(5)]

    executable_return_dataset_and_tensor = computation_impl.ComputationImpl(
        tensorflow_serialization.serialize_py_fn_as_tf_computation(
            return_dataset_and_tensor, None,
            context_stack_impl.context_stack)[0],
        context_stack_impl.context_stack)

    x = executable_return_dataset_and_tensor()
    self.assertEqual(x[0], 0)
    self.assertEqual([x for x in iter(x[1])], list(range(5)))
    self.assertEqual(x[2], 5)

  def test_fetch_value_with_datasets_nested_at_second_level(self):

    def return_two_datasets():
      return [
          tf.constant(0), [tf.data.Dataset.range(5),
                           tf.data.Dataset.range(5)]
      ]

    executable_return_two_datasets = computation_impl.ComputationImpl(
        tensorflow_serialization.serialize_py_fn_as_tf_computation(
            return_two_datasets, None, context_stack_impl.context_stack)[0],
        context_stack_impl.context_stack)

    x = executable_return_two_datasets()
    self.assertEqual(x[0], 0)
    self.assertEqual([i for i in iter(x[1][0])], list(range(5)))
    self.assertEqual([i for i in iter(x[1][1])], list(range(5)))

  def test_fetch_value_with_empty_dataset_and_tensors(self):

    def return_dataset():
      ds1 = tf.data.Dataset.from_tensor_slices([[1, 1], [1, 1]])
      return [tf.constant([0., 0.]), ds1.batch(5).take(0)]

    executable_return_dataset = computation_impl.ComputationImpl(
        tensorflow_serialization.serialize_py_fn_as_tf_computation(
            return_dataset, None, context_stack_impl.context_stack)[0],
        context_stack_impl.context_stack)

    x = executable_return_dataset()
    self.assertAllEqual(x[0], [0., 0.])
    self.assertEqual(
        tf.data.experimental.get_structure(x[1]),
        tf.TensorSpec(shape=(None, 2), dtype=tf.int32))
    with self.assertRaises(StopIteration):
      _ = next(iter(x[1]))

  def test_fetch_value_with_empty_structured_dataset_and_tensors(self):

    def return_dataset():
      ds1 = tf.data.Dataset.from_tensor_slices(
          collections.OrderedDict([('a', [1, 1]), ('b', [1, 1])]))
      return [tf.constant([0., 0.]), ds1.batch(5).take(0)]

    executable_return_dataset = computation_impl.ComputationImpl(
        tensorflow_serialization.serialize_py_fn_as_tf_computation(
            return_dataset, None, context_stack_impl.context_stack)[0],
        context_stack_impl.context_stack)

    x = executable_return_dataset()
    self.assertAllEqual(x[0], [0., 0.])
    self.assertEqual(
        tf.data.experimental.get_structure(x[1]),
        collections.OrderedDict([
            ('a', tf.TensorSpec(shape=(None,), dtype=tf.int32)),
            ('b', tf.TensorSpec(shape=(None,), dtype=tf.int32)),
        ]))
    with self.assertRaises(StopIteration):
      _ = next(iter(x[1]))


if __name__ == '__main__':
  test.main()
