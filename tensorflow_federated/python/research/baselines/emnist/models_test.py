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

import tensorflow.compat.v2 as tf

from tensorflow_federated.python.research.baselines.emnist import models


class ModelCollectionTest(tf.test.TestCase):

  def test_conv_dropout_only_digits_shape(self):
    image = tf.random.normal([4, 28 * 28])
    model = models.create_conv_dropout_model(only_digits=True)
    logits = model(image)
    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, [4, 10])

  def test_conv_dropout_shape(self):
    image = tf.random.normal([3, 28 * 28])
    model = models.create_conv_dropout_model(only_digits=False)
    logits = model(image)

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, [3, 62])

  def test_2nn_number_of_parameters(self):
    model = models.create_two_hidden_layer_model(
        only_digits=True, hidden_units=200)

    # We calculate the number of parameters based on the fact that given densely
    # connected layers of size n and m with bias units, there are (n+1)m
    # parameters between these layers. The network above should have layers of
    # size 28*28, 200, 200, and 10.
    num_model_params = (28 * 28 + 1) * 200 + 201 * 200 + 201 * 10
    self.assertEqual(model.count_params(), num_model_params)

  def test_resnet_block_parameters(self):
    resnet_model_1 = models.create_resnet(only_digits=False, num_blocks=5)
    resnet_model_2 = models.create_resnet(only_digits=False, num_blocks=9)
    self.assertLess(resnet_model_1.count_params(),
                    resnet_model_2.count_params())


if __name__ == '__main__':
  tf.test.main()
