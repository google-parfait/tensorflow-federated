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
"""Utilities for building broadcast and aggregate functions with encoding.

This file contains utilities for building `StatefulBroadcastFn` and
`StatefulAggregateFn` utilizing `Encoder` class from `tensor_encoding` project,
to realize encoding (compression) of values being communicated between `SERVER`
and `CLIENTS`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model_utils


# TODO(b/138081552): Move to tff.learning when ready.
def build_encoded_broadcast_from_model(model_fn, encoder_fn):
  """Builds `StatefulBroadcastFn` for weights of model returned by `model_fn`.

  This method creates a `SimpleEncoder` for every weight of model created by
  `model_fn`, as returned by `encoder_fn`.

  Args:
    model_fn: A Python callable with no arguments function that returns a
      `tff.learning.Model`.
    encoder_fn: A Python callable with a single argument, which is expected to
      be a `tf.Tensor` of shape and dtype to be encoded. The function must
      return a `tensor_encoding.core.SimpleEncoder`, which expects a `tf.Tensor`
      with compatible type as the input to its `encode` method.

  Returns:
    A `StatefulBroadcastFn` for encoding and broadcasting the weights of model
    created by `model_fn`.

  Raises:
    TypeError: If `model_fn` or `encoder_fn` are not callable objects.
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_callable(encoder_fn)
  # TODO(b/144382142): Keras name uniquification is probably the main reason we
  # still need this.
  with tf.Graph().as_default():
    values = model_utils.enhance(model_fn()).weights
  encoders = tf.nest.map_structure(encoder_fn, values)
  return tff.utils.build_encoded_broadcast(values, encoders)


# TODO(b/138081552): Move to tff.learning when ready.
def build_encoded_sum_from_model(model_fn, encoder_fn):
  """Builds `StatefulAggregateFn` for weights of model returned by `model_fn`.

  This method creates a `GatherEncoder` for every trainable weight of model
  created by `model_fn`, as returned by `encoder_fn`.

  Args:
    model_fn: A Python callable with no arguments function that returns a
      `tff.learning.Model`.
    encoder_fn: A Python callable with a single argument, which is expected to
      be a `tf.Tensor` of shape and dtype to be encoded. The function must
      return a `tensor_encoding.core.SimpleEncoder`, which expects a `tf.Tensor`
      with compatible type as the input to its `encode` method.

  Returns:
    A `StatefulAggregateFn` for encoding and summing the weights of model
    created by `model_fn`.

  Raises:
    TypeError: If `model_fn` or `encoder_fn` are not callable objects.
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_callable(encoder_fn)
  # TODO(b/144382142): Keras name uniquification is probably the main reason we
  # still need this.
  with tf.Graph().as_default():
    values = model_utils.enhance(model_fn()).weights.trainable
  encoders = tf.nest.map_structure(encoder_fn, values)
  return tff.utils.build_encoded_sum(values, encoders)


# TODO(b/138081552): Move to tff.learning when ready.
def build_encoded_mean_from_model(model_fn, encoder_fn):
  """Builds `StatefulAggregateFn` for weights of model returned by `model_fn`.

  This method creates a `GatherEncoder` for every trainable weight of model
  created by `model_fn`, as returned by `encoder_fn`.

  Args:
    model_fn: A Python callable with no arguments function that returns a
      `tff.learning.Model`.
    encoder_fn: A Python callable with a single argument, which is expected to
      be a `tf.Tensor` of shape and dtype to be encoded. The function must
      return a `tensor_encoding.core.SimpleEncoder`, which expects a `tf.Tensor`
      with compatible type as the input to its `encode` method.

  Returns:
    A `StatefulAggregateFn` for encoding and averaging the weights of model
    created by `model_fn`.

  Raises:
    TypeError: If `model_fn` or `encoder_fn` are not callable objects.
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_callable(encoder_fn)
  # TODO(b/144382142): Keras name uniquification is probably the main reason we
  # still need this.
  with tf.Graph().as_default():
    values = model_utils.enhance(model_fn()).weights.trainable
  encoders = tf.nest.map_structure(encoder_fn, values)
  return tff.utils.build_encoded_mean(values, encoders)
