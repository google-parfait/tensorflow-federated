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
"""Tests for ResNet v2 models."""

import tensorflow as tf

from tensorflow_federated_research.utils.models import resnet_models


class ResnetModelTest(tf.test.TestCase):

  def test_resnet18_imagenet_inputs(self):
    resnet18 = resnet_models.create_resnet18(
        input_shape=(224, 224, 3), num_classes=1000)
    self.assertIsInstance(resnet18, tf.keras.Model)

  def test_resnet34_imagenet_inputs(self):
    resnet34 = resnet_models.create_resnet34(
        input_shape=(224, 224, 3), num_classes=1000)
    self.assertIsInstance(resnet34, tf.keras.Model)

  def test_resnet50_imagenet_inputs(self):
    resnet50 = resnet_models.create_resnet50(
        input_shape=(224, 224, 3), num_classes=1000)
    self.assertIsInstance(resnet50, tf.keras.Model)

  def test_resnet152_imagenet_inputs(self):
    resnet152 = resnet_models.create_resnet152(
        input_shape=(224, 224, 3), num_classes=1000)
    self.assertIsInstance(resnet152, tf.keras.Model)

  def test_bad_input_raises_exception(self):
    with self.assertRaises(Exception):
      resnet_models.create_resnet50(input_shape=(1, 1), num_classes=10)

  def test_batch_norm_constructs(self):
    batch_resnet = resnet_models.create_resnet(
        input_shape=(32, 32, 3), num_classes=10, norm='batch')
    self.assertIsInstance(batch_resnet, tf.keras.Model)

  def test_group_norm_constructs(self):
    group_resnet = resnet_models.create_resnet(
        input_shape=(32, 32, 3), num_classes=10, norm='group')
    self.assertIsInstance(group_resnet, tf.keras.Model)

  def test_basic_fewer_parameters_than_bottleneck(self):
    input_shape = (32, 32, 3)
    num_classes = 10
    basic_resnet = resnet_models.create_resnet(
        input_shape, num_classes, block='basic')
    bottleneck_resnet = resnet_models.create_resnet(
        input_shape, num_classes, block='bottleneck')

    self.assertLess(basic_resnet.count_params(),
                    bottleneck_resnet.count_params())

  def test_repetitions_increase_number_parameters(self):
    input_shape = (32, 32, 3)
    num_classes = 10
    small_resnet = resnet_models.create_resnet(
        input_shape, num_classes, repetitions=[1, 1])
    big_resnet = resnet_models.create_resnet(
        input_shape, num_classes, repetitions=[2, 2])
    self.assertLess(small_resnet.count_params(), big_resnet.count_params())


if __name__ == '__main__':
  tf.test.main()
