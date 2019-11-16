# Lint as: python3
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

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.impl.compiler import type_factory


class TypeConstructorsTest(absltest.TestCase):

  def test_reduction_op(self):
    self.assertEqual(
        str(type_factory.reduction_op(tf.float32, tf.int32)),
        '(<float32,int32> -> float32)')

  def test_unary_op(self):
    self.assertEqual(str(type_factory.unary_op(tf.bool)), '(bool -> bool)')

  def test_binary_op(self):
    self.assertEqual(
        str(type_factory.binary_op(tf.bool)), '(<bool,bool> -> bool)')

  def test_at_server(self):
    self.assertEqual(str(type_factory.at_server(tf.bool)), 'bool@SERVER')

  def test_at_clients(self):
    self.assertEqual(str(type_factory.at_clients(tf.bool)), '{bool}@CLIENTS')


if __name__ == '__main__':
  absltest.main()
