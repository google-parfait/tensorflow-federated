# Copyright 2022, Google LLC.
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
from tensorflow_federated.python.simulation.baselines.landmark import landmark_preprocessing

TEST_DATA = collections.OrderedDict({
    'image/decoded': tf.zeros(shape=(200, 200, 3), dtype=tf.int32),
    'class': tf.zeros(shape=(1), dtype=tf.int64),
})
_IMAGE_SIZE = landmark_preprocessing.IMAGE_SIZE


class LandmarkPreprocessingTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('train', landmark_preprocessing._create_model(is_training=True)),
      ('test', landmark_preprocessing._create_model(is_training=False)),
  )
  def test_map_fn_returns_correct_output_shape_and_dtype(self, model):
    map_fn = lambda element: landmark_preprocessing._map_fn(element, model)
    ds = tf.data.Dataset.from_tensors(TEST_DATA)
    image, label = iter(ds.map(map_fn)).next()
    self.assertListEqual(
        image.shape.as_list(),
        [_IMAGE_SIZE, _IMAGE_SIZE, 3],
    )
    self.assertListEqual(label.shape.as_list(), [1])
    self.assertEqual(image.dtype, tf.float32)
    self.assertEqual(label.dtype, tf.int64)

  @parameterized.named_parameters(
      ('train1', True, 1, 20, 1),
      ('train2', True, 1, 1, None),
      ('train3', True, 200, 1, None),
      ('train4', True, 200, 1, 100),
      ('train5', True, 100, 1, 200),
      ('test1', False, 1, 20, 1),
      ('test2', False, 1, 1, None),
      ('test3', False, 200, 1, None),
      ('test4', False, 200, 1, 100),
      ('test5', False, 100, 1, 200),
  )
  def test_create_preprocess_fn_returns_correct_output(
      self, is_training, num_epochs, batch_size, max_elements
  ):
    ds = tf.data.Dataset.from_tensors(TEST_DATA)
    preprocess_spec = client_spec.ClientSpec(
        num_epochs=num_epochs, batch_size=batch_size, max_elements=max_elements
    )
    preprocess_fn = landmark_preprocessing.create_preprocess_fn(
        preprocess_spec=preprocess_spec, is_training=is_training
    )
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        preprocessed_ds.element_spec,
        (
            tf.TensorSpec(
                shape=(None, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        ),
    )

    element = iter(preprocessed_ds).next()
    if max_elements is None:
      expected_length = num_epochs
    else:
      expected_length = min(num_epochs, max_elements)
    self.assertLen(preprocessed_ds, expected_length)

    expected_element = (
        tf.convert_to_tensor([[[[-1.0] * 3] * _IMAGE_SIZE] * _IMAGE_SIZE] * 1),
        tf.zeros(shape=(1, 1), dtype=tf.int64),
    )
    self.assertAllClose(element, expected_element)


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
