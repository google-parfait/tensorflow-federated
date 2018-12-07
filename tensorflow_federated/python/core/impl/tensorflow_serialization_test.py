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
"""Tests for tensorflow_serialization.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import tensorflow_serialization
from tensorflow_federated.python.core.impl import type_serialization


class TensorFlowSerializationTest(tf.test.TestCase):

  def test_serialize_tensorflow_with_no_parameter(self):
    comp = tensorflow_serialization.serialize_py_func_as_tf_computation(
        lambda: tf.constant(99))
    self.assertEqual(
        str(type_serialization.deserialize_type(comp.type)), '( -> int32)')
    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    results = tf.Session().run(tf.import_graph_def(
        comp.tensorflow.graph_def, None, [
            comp.tensorflow.result.tensor.tensor_name]))
    self.assertEqual(results, [99])

  def test_serialize_tensorflow_with_simple_add_three_lambda(self):
    comp = tensorflow_serialization.serialize_py_func_as_tf_computation(
        lambda x: x + 3, tf.int32)
    self.assertEqual(
        str(type_serialization.deserialize_type(comp.type)), '(int32 -> int32)')
    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    parameter = tf.constant(1000)
    results = tf.Session().run(tf.import_graph_def(
        comp.tensorflow.graph_def,
        {comp.tensorflow.parameter.tensor.tensor_name: parameter},
        [comp.tensorflow.result.tensor.tensor_name]))
    self.assertEqual(results, [1003])

  def test_serialize_tensorflow_with_data_set_sum_lambda(self):
    def _legacy_dataset_reducer_example(ds):
      return ds.reduce(np.int64(0), lambda x, y: x + y)
    comp = tensorflow_serialization.serialize_py_func_as_tf_computation(
        _legacy_dataset_reducer_example, types.SequenceType(tf.int64))
    self.assertEqual(
        str(type_serialization.deserialize_type(comp.type)),
        '(int64* -> int64)')
    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    parameter = tf.data.Dataset.range(5)
    results = tf.Session().run(tf.import_graph_def(
        comp.tensorflow.graph_def,
        {comp.tensorflow.parameter.sequence.iterator_string_handle_name: (
            parameter.make_one_shot_iterator().string_handle())},
        [comp.tensorflow.result.tensor.tensor_name]))
    self.assertEqual(results, [10])


if __name__ == '__main__':
  tf.test.main()
