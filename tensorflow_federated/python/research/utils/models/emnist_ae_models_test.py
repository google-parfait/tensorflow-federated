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

import tensorflow as tf

from tensorflow_federated.python.research.utils.models import emnist_ae_models


class ModelCollectionTest(tf.test.TestCase):

  def test_ae_model(self):
    image = tf.random.normal([4, 28*28])
    model = emnist_ae_models.create_autoencoder_model()
    reconstructed_image = model(image)
    num_model_params = 2837314
    self.assertIsNotNone(reconstructed_image)
    self.assertEqual(reconstructed_image.shape, [4, 28*28])
    self.assertEqual(model.count_params(), num_model_params)


if __name__ == '__main__':
  tf.test.main()
