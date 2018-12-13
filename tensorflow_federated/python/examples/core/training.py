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
"""A small self-contained training example built directly on top of Core API.

This example contains a simpler form of federated averaging logic, similar to
that which one might find in `../../learning`, but optimized for simplicity and
compactness and illustrating the use of basic mechanisms provided by the Core
API. The user is encouraged to study the simplified structure of this example
first as a stepping stone towards the more general implementation in `learning`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow as tf

from tensorflow_federated import python as tff

# TODO(b/120157713): Transform this into a notebook eventually while retaining
# the ability to utilize it as an extra unit test of the Core API.

# TODO(b/120157713): TBD whether it's desirable or not to standardize on the
# same model in all tutorials and examples.

# The model we will train is a simple linear classifier that given a discrete
# input class named 'X' predicts a discrete output class named 'Y'. The input
# class is an integer from 0 to NUM_X_CLASSES - 1, and the output is an integer
# from 0 to NUM_Y_CLASSES.
NUM_X_CLASSES = 7
NUM_Y_CLASSES = 3

# The samples of data used to train or evaluate the model will consists of the
# values of the 2 features 'X' and 'Y'. The data will arrive in batches (e.g.,
# as is typicallyh the case for the output of 'tf.parse_example'). Every batch
# of samples is represented as a TFF named tuple ewith 2 elements 'X' and 'Y',
# each of which is a tf.int32 tensor with a single dimension (so conceptually,
# a vector). The two tensors should have the same number of elements (this is
# not expressed in the type below). The number of elements is unspecified (the
# shape of both scalars is [None]), since the individual batches of data might
# contain unequal numbers of samples not known in advance.
BATCH_TYPE = tff.NamedTupleType([
    ('X', tff.TensorType(tf.int32, shape=[None])),
    ('Y', tff.TensorType(tf.int32, shape=[None])),
])  # pyformat: disable


# The input to training/evaluation is simply a sequence of such batches.
INPUT_TYPE = tff.SequenceType(BATCH_TYPE)

# The parameters of the model consist of a weight matrix and a bias vector, to
# be applied to a one-hot encoding of the inputs.
MODEL_TYPE = tff.NamedTupleType([
    ('weights', tff.TensorType(tf.float32, [NUM_X_CLASSES, NUM_Y_CLASSES])),
    ('bias', tff.TensorType(tf.float32, NUM_Y_CLASSES)),
])  # pyformat: disable


# A simple TensorFlow computation that computes loss and accuracy metrics on a
# batch of features.
#
# The function decorator 'tf_computation' transforms the Python function into a
# 'Computation', a unit of composition in TFF. When a Python function is wrapped
# as a computation, one can think of it as conceptually consuming and returning
# TFF values that have TFF types. The TFF type of the parameter of computation
# 'extract_features' is a tensor type declared as the argument to the decorator,
# see above for the definition of BATCH_TYPE. The TFF type of the value returned
# by the 'extract_features' computation is determined automatically. In this
# case, it is a TFF named tuple with 2 named elements constructed from elements
# of the returned Python dictionary.
@tff.tf_computation([BATCH_TYPE, MODEL_TYPE])
def forward_pass(features, model):
  """Computes loss and accuracy metrics for the given sample batch and model.

  Args:
    features: A named tuple with 2 elements `X` and `Y` that represent the two
      feature vectors in a single batch. The elements are represented as
      `tf.Tensor`s of the dtypes and shapes as defined in `BATCH_TYPE`.
    model: A named tuple with 2 elements `weights` and `bias` that represent the
      model parameters. Each of the parameters is represented here as a
      `tf.Tensor`, with dtypes and shapes as defined in `MODEL_TYPE` above.

  Returns:
    An instance of `collections.OrderedDict` with `tf.Tensor`s containing the
    two computed metrics `loss` and `accuracy`, each an instance of `tf.Tensor`
    with dtype `tf.float32` and shape `[]` (a scalar). This dictionary is
    reinterpreted by TFF as a named TFF tuple with tensor elements.
  """
  encoded_x = tf.one_hot(features.X, NUM_X_CLASSES)
  encoded_y = tf.one_hot(features.Y, NUM_Y_CLASSES)
  softmax_y = tf.nn.softmax(tf.matmul(encoded_x, model.weights) + model.bias)
  loss = tf.reduce_mean(
      -tf.reduce_sum(encoded_y * tf.log(softmax_y), reduction_indices=1))
  prediction = tf.to_int32(tf.argmax(softmax_y, 1))
  is_correct = tf.equal(prediction, features.Y)
  accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
  count = tf.size(features.X)
  return collections.OrderedDict([('loss', loss), ('accuracy', accuracy),
                                  ('count', count)])


# Capturing type signature of the named tuple of statistics (metrics, counters)
# computed by the model.
STATS_TYPE = forward_pass.type_signature.result


# TODO(b/120157713): Implement the remainder of this example: gradient descent,
# local training loop, federated averaging of model parameters, etc.
