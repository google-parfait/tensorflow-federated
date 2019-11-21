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
"""GAN evaluation metrics for EMNIST.

Built on top of TF-GAN library (https://github.com/tensorflow/gan).
"""

import tensorflow as tf
import tensorflow_gan as tfgan


@tf.function
def _emnist_classifier(images, emnist_classifier):
  return emnist_classifier(images)


@tf.function
def emnist_score(images, emnist_classifier):
  """EMNIST classifier score for evaluating a generative model.

  This is based on the Inception Score, but for an arbitrary classifier.

  This technique is described in detail in https://arxiv.org/abs/1606.03498. In
  summary, this function calculates

  exp( E[ KL(p(y|x) || p(y)) ] )

  which captures how different the network's classification prediction is from
  the prior distribution over classes.

  This method wraps an implementation of classifier score provided in the
  TF-GAN library (https://github.com/tensorflow/gan). Please see the TF-GAN
  implementation for further details
  (https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/eval/
  classifier_metrics.py).

  Args:
    images: Images of shape [None, 28, 28, 1] to calculate the classifier score
      for. These images should be preprocessed to the format that is expected by
      the trained classifier model provided via the emnist_classifier argument.
    emnist_classifier: A Keras model instance of a trained EMNIST classifier.
      One is available via the ../classifier/emnist_classifier_model.py library.

  Returns:
    The classifier score. A floating-point scalar.
  """
  images.shape.assert_is_compatible_with([None, 28, 28, 1])

  emnist_classifier_logits = _emnist_classifier(images, emnist_classifier)

  score = tfgan.eval.classifier_score_from_logits(emnist_classifier_logits)
  score.shape.assert_is_compatible_with([])
  return score


@tf.function
def emnist_frechet_distance(real_images, generated_images, emnist_classifier):
  """EMNIST Frechet distance between real and generated images.

  This technique is described in detail in https://arxiv.org/abs/1706.08500.

  This method wraps an implementation of Frechet classifier distance provided in
  the TF-GAN library (https://github.com/tensorflow/gan). Please see the TF-GAN
  implementation for further details
  (https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/eval/
  classifier_metrics.py).

  Args:
    real_images: Real images of shape [None, 28, 28, 1] to use to compute
      Frechet Inception distance. These images should be preprocessed to the
      format that is expected by the trained classifier model provided via the
      emnist_classifier argument.
    generated_images: Generated images of shape [None, 28, 28, 1] to use to
      compute Frechet Inception distance.
    emnist_classifier: A Keras model instance of a trained EMNIST classifier.
      One is available via the ../classifier/emnist_classifier_model.py library.

  Returns:
    The Frechet distance. A floating-point scalar.
  """
  real_images.shape.assert_is_compatible_with([None, 28, 28, 1])
  generated_images.shape.assert_is_compatible_with([None, 28, 28, 1])

  real_images_activations = _emnist_classifier(real_images, emnist_classifier)
  generated_images_activations = _emnist_classifier(generated_images,
                                                    emnist_classifier)

  frechet_distance = tfgan.eval.frechet_classifier_distance_from_activations(
      real_images_activations, generated_images_activations)
  frechet_distance.shape.assert_is_compatible_with([])
  return frechet_distance
