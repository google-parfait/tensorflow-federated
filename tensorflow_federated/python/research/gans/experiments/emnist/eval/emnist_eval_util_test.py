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
"""Test the GAN evaluation metrics for EMNIST."""

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.research.gans.experiments.emnist import emnist_data_utils
from tensorflow_federated.python.research.gans.experiments.emnist.classifier import emnist_classifier_model as ecm

from tensorflow_federated.python.research.gans.experiments.emnist.eval import emnist_eval_util as eeu


class EmnistEvalUtilTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    client_data = emnist_data_utils.create_real_images_tff_client_data(
        split='synthetic')
    images_ds = client_data.create_tf_dataset_for_client(
        client_data.client_ids[0])
    images_ds = emnist_data_utils.preprocess_img_dataset(
        images_ds, shuffle=False)
    images_ds_iterator = iter(images_ds)
    self.real_images = next(images_ds_iterator)

    np.random.seed(seed=123456)
    self.fake_images = tf.constant(
        np.random.random((32, 28, 28, 1)), dtype=tf.float32)

  def test_emnist_score(self):
    score = eeu.emnist_score(self.fake_images,
                             ecm.get_trained_emnist_classifier_model())
    self.assertAllClose(score, 1.1598, rtol=0.0001, atol=0.0001)

    score = eeu.emnist_score(self.real_images,
                             ecm.get_trained_emnist_classifier_model())
    self.assertAllClose(score, 3.9547, rtol=0.0001, atol=0.0001)

  def test_emnist_frechet_distance(self):
    distance = eeu.emnist_frechet_distance(
        self.real_images, self.fake_images,
        ecm.get_trained_emnist_classifier_model())
    self.assertAllClose(distance, 568.6883, rtol=0.0001, atol=0.0001)

    distance = eeu.emnist_frechet_distance(
        self.real_images, self.real_images,
        ecm.get_trained_emnist_classifier_model())
    self.assertAllClose(distance, 0.0)


if __name__ == '__main__':
  tf.test.main()
