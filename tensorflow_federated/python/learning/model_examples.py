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
"""Simple examples implementing the Model interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow as tf

from tensorflow_federated.python.learning import model


class LinearRegression(model.Model):
  """Example of a simple linear regression implemented directly."""

  # A tuple (x, y), where 'x' represent features, and 'y' represent labels.
  Batch = collections.namedtuple('Batch', ['x', 'y'])  # pylint: disable=invalid-name

  def __init__(self, feature_dim=1):
    # Define all the variables, similar to what Keras Layers and Models
    # do in build().
    self._num_examples = tf.Variable(0, name='num_examples', trainable=False)
    self._num_batches = tf.Variable(0, name='num_batches', trainable=False)
    self._loss_sum = tf.Variable(0.0, name='loss_sum', trainable=False)
    self._a = tf.Variable(
        tf.zeros(shape=(feature_dim, 1)), name='a', trainable=True)
    self._b = tf.Variable(0.0, name='b', trainable=True)

  @property
  def trainable_variables(self):
    return [self._a, self._b]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def local_variables(self):
    return [self._num_examples, self._num_batches, self._loss_sum]

  @tf.contrib.eager.defun
  def _predict(self, x):
    return tf.matmul(x, self._a) + self._b

  @tf.contrib.eager.defun
  def forward_pass(self, batch, training=True):
    del training  # Unused

    predictions = self._predict(batch.x)
    residuals = predictions - batch.y
    num_examples = tf.gather(tf.shape(predictions), 0)
    total_loss = 0.5 * tf.reduce_sum(tf.pow(residuals, 2))

    tf.assign_add(self._loss_sum, total_loss)
    tf.assign_add(self._num_examples, num_examples)
    tf.assign_add(self._num_batches, 1)

    average_loss = total_loss / tf.to_float(num_examples)
    return model.BatchOutput(loss=average_loss, predictions=predictions)

  @tf.contrib.eager.defun
  def aggregated_outputs(self):
    return collections.OrderedDict(
        [('num_examples', self._num_examples), ('num_batches',
                                                self._num_batches),
         ('loss', self._loss_sum / tf.to_float(self._num_examples))])

  @classmethod
  def make_batch(cls, x, y):
    """Returns a `Batch` to pass to the forward pass."""
    return cls.Batch(x, y)


class TrainableLinearRegression(LinearRegression, model.TrainableModel):
  """A LinearRegression with trivial SGD training."""

  @tf.contrib.eager.defun
  def train_on_batch(self, batch):
    # Most users won't implement this, and let us provide the optimizer.
    fp = self.forward_pass(batch)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer.minimize(fp.loss, var_list=self.trainable_variables)
    return fp
