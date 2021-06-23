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

import collections

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines.cifar import cifar_preprocessing


TEST_DATA = collections.OrderedDict(
    coarse_label=([tf.constant(1, dtype=tf.int64)]),
    image=([tf.zeros((32, 32, 3), dtype=tf.uint8)]),
    label=([tf.constant(1, dtype=tf.int64)]),
)


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class PreprocessFnTest(tf.test.TestCase, parameterized.TestCase):

  def test_raises_non_iterable_crop(self):
    preprocess_spec = client_spec.ClientSpec(num_epochs=1, batch_size=1)
    with self.assertRaisesRegex(TypeError, 'crop_shape must be an iterable'):
      cifar_preprocessing.create_preprocess_fn(preprocess_spec, crop_shape=32)

  def test_raises_iterable_length_2_crop(self):
    preprocess_spec = client_spec.ClientSpec(num_epochs=1, batch_size=1)
    with self.assertRaisesRegex(ValueError,
                                'The crop_shape must have length 3'):
      cifar_preprocessing.create_preprocess_fn(
          preprocess_spec, crop_shape=(32, 32))

  @parameterized.named_parameters(
      ('num_epochs_1_batch_size_1', 1, 1),
      ('num_epochs_4_batch_size_2', 4, 2),
      ('num_epochs_9_batch_size_3', 9, 3),
      ('num_epochs_12_batch_size_1', 12, 1),
      ('num_epochs_3_batch_size_5', 3, 5),
      ('num_epochs_7_batch_size_2', 7, 2),
  )
  def test_ds_length_is_ceil_num_epochs_over_batch_size(self, num_epochs,
                                                        batch_size):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_spec = client_spec.ClientSpec(
        num_epochs=num_epochs, batch_size=batch_size)
    preprocess_fn = cifar_preprocessing.create_preprocess_fn(preprocess_spec)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        _compute_length_of_dataset(preprocessed_ds),
        tf.cast(tf.math.ceil(num_epochs / batch_size), tf.int32))

  @parameterized.named_parameters(
      ('crop_shape_1_no_distort', (32, 32, 3), False),
      ('crop_shape_2_no_distort', (28, 28, 3), False),
      ('crop_shape_3_no_distort', (24, 26, 3), False),
      ('crop_shape_1_distort', (32, 32, 3), True),
      ('crop_shape_2_distort', (28, 28, 3), True),
      ('crop_shape_3_distort', (24, 26, 3), True),
  )
  def test_preprocess_fn_returns_correct_element(self, crop_shape,
                                                 distort_image):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=1, shuffle_buffer_size=1)
    preprocess_fn = cifar_preprocessing.create_preprocess_fn(
        preprocess_spec, crop_shape=crop_shape, distort_image=distort_image)
    preprocessed_ds = preprocess_fn(ds)
    expected_element_spec_shape = (None,) + crop_shape
    self.assertEqual(
        preprocessed_ds.element_spec,
        (tf.TensorSpec(shape=expected_element_spec_shape, dtype=tf.float32),
         tf.TensorSpec(shape=(None,), dtype=tf.int64)))

    expected_element_shape = (1,) + crop_shape
    element = next(iter(preprocessed_ds))
    expected_element = (tf.zeros(
        shape=expected_element_shape,
        dtype=tf.float32), tf.ones(shape=(1,), dtype=tf.int32))
    self.assertAllClose(self.evaluate(element), expected_element)

  def test_preprocess_is_no_op_for_normalized_image(self):
    crop_shape = (1, 1, 3)
    x = tf.constant([[[1.0, -1.0, 0.0]]])  # Has shape (1, 1, 3), mean 0
    x = x / tf.math.reduce_std(x)  # x now has variance 1
    simple_example = collections.OrderedDict(image=x, label=0)
    image_map = cifar_preprocessing.build_image_map(crop_shape, distort=False)
    cropped_example = image_map(simple_example)

    self.assertEqual(cropped_example[0].shape, crop_shape)
    self.assertAllClose(x, cropped_example[0], rtol=1e-03)
    self.assertEqual(cropped_example[1], 0)

  @parameterized.named_parameters(
      ('max_elements1', 1),
      ('max_elements3', 3),
      ('max_elements7', 7),
      ('max_elements11', 11),
      ('max_elements18', 18),
  )
  def test_ds_length_with_max_elements(self, max_elements):
    repeat_size = 10
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA).repeat(repeat_size)
    preprocess_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=1, max_elements=max_elements)
    preprocess_fn = cifar_preprocessing.create_preprocess_fn(preprocess_spec)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        _compute_length_of_dataset(preprocessed_ds),
        min(repeat_size, max_elements))


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
