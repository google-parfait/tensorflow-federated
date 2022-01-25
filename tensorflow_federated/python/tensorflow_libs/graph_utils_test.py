# Copyright 2022, The TensorFlow Federated Authors.
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
"""Tests for graph_utils."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs import graph_utils


class IsControlDependencyTest(tf.test.TestCase):

  def test_is_true(self):
    self.assertTrue(graph_utils.is_control_dependency('^foo'))

  def test_is_false(self):
    self.assertFalse(graph_utils.is_control_dependency('foo'))
    self.assertFalse(graph_utils.is_control_dependency('foo:0'))
    self.assertFalse(graph_utils.is_control_dependency('foo:output:1'))


class GetNodeNameTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('implicit_tensor', 'foo', 'foo'),
      ('control_dep', '^foo', 'foo'),
      ('explicit_graph_tensor', 'foo:0', 'foo'),
      ('explicit_function_tensor', 'foo:output:1', 'foo'),
  )
  def test_get_node_name(self, name, expected):
    self.assertEqual(graph_utils.get_node_name(name), expected)


class MakeControlDependencyTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('implicit_tensor', 'foo', '^foo'),
      ('control_dep', '^foo', '^foo'),
      ('explicit_graph_tensor', 'foo:0', '^foo'),
      ('explicit_function_tensor', 'foo:output:1', '^foo'),
  )
  def test_make_control_dep(self, name, expected):
    self.assertEqual(graph_utils.make_control_dependency(name), expected)


class AddControlDepMappingsTest(tf.test.TestCase):

  def test_add_mappings(self):
    with tf.Graph().as_default():
      bar = tf.compat.v1.placeholder(name='bar', dtype=tf.int64)
      test2 = tf.compat.v1.placeholder(name='test2', dtype=tf.float32)
    self.assertEqual(
        graph_utils.add_control_dep_mappings({
            'foo:0': bar,
            'test': test2,
        }), {
            'foo:0': bar,
            '^foo': '^bar',
            'test': test2,
            '^test': '^test2',
        })

  def test_add_mappings_skip_existing_control_deps(self):
    input_map = {'^foo': '^bar'}
    self.assertEqual(graph_utils.add_control_dep_mappings(input_map), input_map)


if __name__ == '__main__':
  tf.test.main()
