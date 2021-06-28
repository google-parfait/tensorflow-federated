# Copyright 2019, Google LLC.
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

import tensorflow as tf

from tensorflow_federated.python.simulation.baselines.cifar100 import resnet_models


class CreateResnetInputValidationTest(tf.test.TestCase):

  def test_non_iterable_input_shape_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'input_shape must be an iterable of length 3 containing '
        'only positive integers'):
      resnet_models.create_resnet(input_shape=100)

  def test_input_shape_with_negative_values_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'input_shape must be an iterable of length 3 containing '
        'only positive integers'):
      resnet_models.create_resnet(input_shape=(10, -1, 2))

  def test_non_length_3_input_shape_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'input_shape must be an iterable of length 3 containing '
        'only positive integers'):
      resnet_models.create_resnet(input_shape=(10, 10))

  def test_negative_num_classes_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'num_classes must be a positive integer'):
      resnet_models.create_resnet(input_shape=(32, 32, 3), num_classes=-5)

  def test_unsupported_residual_block_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'residual_block must be of type `ResidualBlock`'):
      resnet_models.create_resnet(
          input_shape=(32, 32, 3), residual_block='bad_block')

  def test_non_iterable_repetitions_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'repetitions must be None or an iterable containing '
        'positive integers'):
      resnet_models.create_resnet(input_shape=(32, 32, 3), repetitions=5)

  def test_repetitions_with_negative_values_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'repetitions must be None or an iterable containing '
        'positive integers'):
      resnet_models.create_resnet(input_shape=(32, 32, 3), repetitions=[2, -1])

  def test_negative_initial_filters_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'initial_filters must be a positive integer'):
      resnet_models.create_resnet(input_shape=(32, 32, 3), initial_filters=-2)

  def test_non_iterable_initial_strides_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'initial_strides must be an iterable of length 2 containing'
        ' only positive integers'):
      resnet_models.create_resnet(input_shape=(32, 32, 3), initial_strides=10)

  def test_initial_strides_with_negative_values_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'initial_strides must be an iterable of length 2 containing'
        ' only positive integers'):
      resnet_models.create_resnet(
          input_shape=(32, 32, 3), initial_strides=(3, -1))

  def test_initial_strides_with_length_not_2_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'initial_strides must be an iterable of length 2 containing'
        ' only positive integers'):
      resnet_models.create_resnet(
          input_shape=(32, 32, 3), initial_strides=(3, 3, 4))

  def test_non_iterable_initial_kernel_size_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'initial_kernel_size must be an iterable of length 2 '
        'containing only positive integers'):
      resnet_models.create_resnet(
          input_shape=(32, 32, 3), initial_kernel_size=10)

  def test_initial_kernel_size_with_negative_values_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'initial_kernel_size must be an iterable of length 2 '
        'containing only positive integers'):
      resnet_models.create_resnet(
          input_shape=(32, 32, 3), initial_kernel_size=(3, -1))

  def test_initial_kernel_size_with_length_not_2_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'initial_kernel_size must be an iterable of length 2 '
        'containing only positive integers'):
      resnet_models.create_resnet(
          input_shape=(32, 32, 3), initial_kernel_size=(3, 3, 4))

  def test_unsupported_norm_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'norm_layer must be of type `NormLayer`'):
      resnet_models.create_resnet(
          input_shape=(32, 32, 3), norm_layer='bad_norm')


class ResNetConstructionTest(tf.test.TestCase):

  def test_resnet_constructs_with_batch_norm(self):
    batch_resnet = resnet_models.create_resnet(
        input_shape=(32, 32, 3),
        num_classes=10,
        norm_layer=resnet_models.NormLayer.batch_norm)
    self.assertIsInstance(batch_resnet, tf.keras.Model)

  def test_resnet_constructs_with_group_norm(self):
    group_resnet = resnet_models.create_resnet(
        input_shape=(32, 32, 3),
        num_classes=10,
        norm_layer=resnet_models.NormLayer.group_norm)
    self.assertIsInstance(group_resnet, tf.keras.Model)

  def test_basic_block_has_fewer_parameters_than_bottleneck(self):
    input_shape = (32, 32, 3)
    num_classes = 10
    basic_resnet = resnet_models.create_resnet(
        input_shape,
        num_classes,
        residual_block=resnet_models.ResidualBlock.basic)
    bottleneck_resnet = resnet_models.create_resnet(
        input_shape,
        num_classes,
        residual_block=resnet_models.ResidualBlock.bottleneck)

    self.assertLess(basic_resnet.count_params(),
                    bottleneck_resnet.count_params())

  def test_repetitions_increases_number_parameters(self):
    input_shape = (32, 32, 3)
    num_classes = 10
    small_resnet = resnet_models.create_resnet(
        input_shape, num_classes, repetitions=[1, 1])
    big_resnet = resnet_models.create_resnet(
        input_shape, num_classes, repetitions=[2, 2])
    self.assertLess(small_resnet.count_params(), big_resnet.count_params())


class CreateSpecializedResnetTest(tf.test.TestCase):

  def test_resnet18_constructs_with_cifar_inputs(self):
    resnet18 = resnet_models.create_resnet18(
        input_shape=(32, 32, 3), num_classes=100)
    self.assertIsInstance(resnet18, tf.keras.Model)

  def test_resnet34_constructs_with_cifar_inputs(self):
    resnet34 = resnet_models.create_resnet34(
        input_shape=(32, 32, 3), num_classes=100)
    self.assertIsInstance(resnet34, tf.keras.Model)

  def test_resnet50_constructs_with_cifar_inputs(self):
    resnet50 = resnet_models.create_resnet50(
        input_shape=(32, 32, 3), num_classes=100)
    self.assertIsInstance(resnet50, tf.keras.Model)

  def test_resnet152_constructs_with_cifar_inputs(self):
    resnet152 = resnet_models.create_resnet152(
        input_shape=(32, 32, 3), num_classes=100)
    self.assertIsInstance(resnet152, tf.keras.Model)


if __name__ == '__main__':
  tf.test.main()
