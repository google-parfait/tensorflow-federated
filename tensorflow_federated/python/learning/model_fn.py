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
"""Defines ModelFn and ModelSpec, which define models for federated computation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import contextlib

# Dependency imports
import six
import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs import tensor_utils

_MODEL_INPUT_TENSOR = None


def model_input_tensor():
  """Returns a batched input tensor to be processed by a model.

  This method is usually called from inside ModelFn.build() to read input data,
  which usually consists of serialized training examples, typically tf.Examples.

  Returns:
    A tensor where the first dimension represents any batching
    that has occurred outside of tensorflow.
  """
  assert _MODEL_INPUT_TENSOR is not None, (
      'No model_input_tensor found. Perhaps you are calling ModelFn.build() '
      'outside of a set_model_input_tensor context manager?')
  return _MODEL_INPUT_TENSOR


@contextlib.contextmanager
def set_model_input_tensor(input_tensor):
  """Sets the tensor returned by model_input_tensor().

  N.B. This method is not thread-safe.

  Args:
    input_tensor: The tensor model_input_tensor() should return.

  Yields:
    None.
  """
  # TODO(b/112710276): Make this context manager thread safe.
  global _MODEL_INPUT_TENSOR
  assert _MODEL_INPUT_TENSOR is None, (
      'Nested calls to set_model_input_tensor are not allowed. '
      'Found existing model_input_tensor {}'.format(_MODEL_INPUT_TENSOR))
  _MODEL_INPUT_TENSOR = input_tensor
  try:
    yield None
  finally:
    _MODEL_INPUT_TENSOR = None


@six.add_metaclass(abc.ABCMeta)
class ModelFn(object):
  """Defines a model used for training or evaluation."""

  @abc.abstractmethod
  def build(self):
    """Mutates the current default graph and returns a corresponding ModelSpec.

    Input to the model is obtained by calling model_input_tensor().

    Returns:
      A ModelSpec object providing information about the model stamped out
      in the current graph.
    """
    # Note to callers: While build() can of course be called direclty, a ModelFn
    # is usually called by using the build_model_fn() function defined below,
    # which captures additional information about variables added to the graph,
    # and other metadata that can be inferred automatically after the model is
    # created.
    pass


class AggregationSpec(object):
  """A specification of how to aggregate a Metric across multiple clients."""

  def __init__(self, compute_sum, compute_average_with_weight,
               compute_samples=True):
    """Constructs an AggregationSpec.

    Args:
      compute_sum: If True, compute a sum of the metric across clients; this
        is usually True for counter metrics like 'num_examples' or
        'num_minibatches'.
      compute_average_with_weight: A tensor weight (could be the constant 1),
        or None. If not None and the provided tensor has value 'w', the metric
        is 'v', and we index clients with i, we compute (sum_i w_i v_i) / (sum_i
        w_i), with the name '{Name}Average'. Typically only one of compute_sum
        and average_with_weight is supplied.
        N. B. This should usually be a variable holding the total sum of
        a counter like num_examples across all minibatches, not a tensor
        holding e.g. the number of examples in a single minibatch.
      compute_samples: If True, collect a sample of per-client values. This may
        be done in an approximate or privatized fashion.
    """

    self._compute_sum = compute_sum
    self._compute_average_with_weight = compute_average_with_weight
    self._compute_samples = compute_samples

  @property
  def compute_sum(self):
    return self._compute_sum

  @property
  def compute_average_with_weight(self):
    return self._compute_average_with_weight

  @property
  def compute_samples(self):
    return self._compute_samples


class Metric(object):
  """Metadata about a metric to be aggregated."""

  def __init__(self, name, value_tensor, aggregation_spec):
    """Initializes the Metric.

    It is often easier to use the named constructors Metric.average or
    Metric.sum rather than calling this constructor directly.

    Args:
      name: A human-readable string name for this metric.
      value_tensor: The tensor to read the final value of the metric after
        all minibatches (items from the dataset) have been processed.
      aggregation_spec: A specification of how to aggregate
        the metric across multiple clients.
    Raises:
      ValueError: The value argument is not a scalar.
    """
    self._name = name
    self._value_tensor = tf.convert_to_tensor(value_tensor)
    if not tensor_utils.is_scalar(self._value_tensor):
      raise ValueError('The value_tensor must be a scalar, but found {}'.format(
          self._value_tensor))
    self._aggregation_spec = aggregation_spec

  # Helper constructors for easily creating weighted average and counter (sum)
  # metrics.
  @classmethod
  def average(cls, name, value_tensor, weight):
    """Returns a Metric for a weighted average metric.

    Args:
      name: A human-readable string name for this metric.
      value_tensor: The tensor to read the final value of the metric after
        all minibatches (items from the dataset) have been processed.
      weight: A tensor holding a weight used to weight value_tensor when
        averaging across clients. See compute_average_with_weight on
        AggregationSpec.

    Raises:
      ValueError: The value argument is not a scalar.

    Returns:
      A Metric.
    """
    return cls(
        name, value_tensor,
        AggregationSpec(
            compute_sum=False,
            compute_average_with_weight=weight,
            compute_samples=True))

  @classmethod
  def sum(cls, name, value_tensor):
    """Returns a Metric for a sum metric (counter).

    Args:
      name: A human-readable string name for this metric.
      value_tensor: The tensor to read the final value of the metric after
         all minibatches (items from the dataset) have been processed.

    Raises:
      ValueError: The value argument is not a scalar.

    Returns:
      A Metric.
    """
    return cls(
        name, value_tensor,
        AggregationSpec(
            compute_sum=True,
            compute_average_with_weight=None,
            compute_samples=True))

  @property
  def name(self):
    return self._name

  @property
  def value_tensor(self):
    return self._value_tensor

  @property
  def aggregation_spec(self):
    return self._aggregation_spec


class ModelSpec(object):
  """Metadata about a model that has been built by a ModelFn."""

  def __init__(self, loss, minibatch_update_ops, metrics):
    """Constructs a ModelSpec.

    Arguments correspond to the properties exposed by this object, see
    comments on those for details.

    Args:
      loss: A tensor defining the loss on the minibatch.
      minibatch_update_ops: A list of tf operations to run per minibatch.
      metrics: A list of Metric objects.
    """
    self._loss = loss
    self._minibatch_update_ops = minibatch_update_ops
    self._metrics = metrics

  @property
  def loss(self):
    """Returns the tensor holding the loss on the current item in the dataset.

    Typically, this is the average loss on the current minibatch.

    Note: The loss is not automatically provided as a metric, you should
    specify it explicitly if you want to visualize the loss.
    """
    return self._loss

  @property
  def minibatch_update_ops(self):
    """Returns a list of ops that should be run for each item in the dataset.

    Typically, this will include the update ops returned from metrics like
    tf.metrics.accuracy or tf.metrics.auc.
    """
    return self._minibatch_update_ops

  @property
  def metrics(self):  # See discussion in next section.
    """Returns a list of Metrics objects.

    These represent any statistics computed on the complete stream of
    minibatches that should be aggregated across clients.
    """
    return self._metrics
