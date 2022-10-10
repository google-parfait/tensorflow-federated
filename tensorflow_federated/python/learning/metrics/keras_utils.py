# Copyright 2022, The TensorFlow Federated Authors.
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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Module for Keras metrics integration."""

import collections
import functools
import itertools
from typing import Any, Callable, List, OrderedDict, Tuple, TypeVar, Union

import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs import variable_utils

StateVar = TypeVar('StateVar')
MetricConstructor = Callable[[], tf.keras.metrics.Metric]
MetricConstructors = OrderedDict[str, MetricConstructor]
MetricStructure = OrderedDict[str, tf.keras.metrics.Metric]
MetricsConstructor = Callable[[], MetricStructure]


def create_functional_metric_fns(
    metrics_constructor: Union[MetricConstructor, MetricsConstructor,
                               MetricConstructors]
) -> Tuple[Callable[[], StateVar], Callable[[StateVar, ...], StateVar],
           Callable[[StateVar], Any]]:
  """Turn a Keras metric construction method into a tuple of pure functions.

  This can be used to convert Keras metrics for use in
  `tff.learning.models.FunctionalModel`. The method traces the metric logic into
  three `tf.function` with explicit `state` parameters that replace the
  closure over internal `tf.Variable` of the `tf.keras.metrics.Metric`.

  IMPORTANT: Only metrics whose `tf.keras.metrics.Metric.update_state` method
  take two arguments (`y_true` and `y_pred`) are supported.

  Example:

    >>> metric = tf.keras.metrics.Accuracy()
    >>> metric.update_state([1.0, 1.0], [0.0, 1.0])
    >>> metric.result()  # == 0.5
    >>>
    >>> metric_fns = tff.learning.metrics.create_functional_metric_fns(
    >>>    tf.keras.metrics.Accuracy)
    >>> initialize, update, finalize = metric_fns
    >>> state = initialize()
    >>> state = update(state, [1.0, 1.0], [0.0, 1.0])
    >>> finalize(state)  # == 0.5

  Args:
    metrics_constructor: Either a no-arg callable that returns a
      `tf.keras.metrics.Metric` or an `OrderedDict` of `str` names to
      `tf.keras.metrics.Metric`, or `OrderedDict` of no-arg callables returning
      `tf.keras.metrics.Metric` instances.  The no-arg callables can be the
      metric class itself (e.g.  `tf.keras.metrics.Accuracy`) in which case the
      default metric configuration will be used. It also supports lambdas or
      `functools.partial` to provide alternate metric configurations.

  Returns:
    A 3-tuple of `tf.function`s namely `(initialize, update, finalize)`.
    `initialize` is a no-arg function used to create the algrebraic "zero"
    before reducing the metric over batches of examples. `update` is a function
    with the same signature as the `metric_constructor().update_state` method
    with a preprended `state` argumetn first, and is used to add an observation
    to the metric. `finalize` only takes a `state` argument and returns the
    final metric value based on observations previously added.

  Raises:
    TypeError: If `metrics_constructor` is not a callable or `OrderedDict`, or
      if `metrics_constructor` is a callable returning values of the wrong type.
  """
  if isinstance(metrics_constructor, collections.OrderedDict):
    metrics_constructor = functools.partial(tf.nest.map_structure,
                                            lambda m: m(), metrics_constructor)
  if not callable(metrics_constructor):
    raise TypeError('`metrics_constructor` must be a callable or '
                    f'`collections.OrderedDict`. Got {metrics_constructor!r} '
                    f'which is a {type(metrics_constructor)!r}.')
  try:
    # Eagerly validate that the metrics_constructor returns values that
    # have the expected properties to provide better debugging messages to
    # caller.
    def check_keras_metric_type(obj):
      if not isinstance(obj, tf.keras.metrics.Metric):
        raise TypeError('Found non-tf.keras.metrics.Metric value: '
                        f'{type(obj)}: {obj!r}.')

    with tf.Graph().as_default():
      metrics_structure = metrics_constructor()
      tf.nest.map_structure(check_keras_metric_type, metrics_structure)
  except ValueError as e:
    raise ValueError(
        '`metrics_constructor` must return a `tf.keras.metrics.Metric` '
        'instance, or an OrderedDict of string to keras metrics.') from e
  if isinstance(metrics_structure, collections.OrderedDict):
    non_string_keys = [
        name for name in metrics_structure.keys() if not isinstance(name, str)
    ]
    if non_string_keys:
      raise TypeError('`metrics_constructor` must return an `OrdredDict` keyed '
                      f'by `str`. Got keys {non_string_keys} that were not '
                      'type `str`.')
  del metrics_structure

  # IMPORTANT: the following code relies on the order of the `tf.Variable`s in
  # `tf.keras.metrics.Metric.variables` to match the order that they are created
  # at runtime. If this changes, `build_replace_variable_with_parameter_creator`
  # will yield the wrong parameters in `update` and `finalize` calls.

  @tf.function
  def initialize():
    with tf.variable_creator_scope(variable_utils.create_tensor_variable):
      return tf.nest.map_structure(
          lambda m: [v.read_value() for v in m.variables],
          metrics_constructor())

  def build_replace_variable_with_parameter_creator(parameters):
    """Create a creation function that replaces variables with parameters.

    Args:
      parameters: Either a list of `tf.function` parameters ordered in the same
        order that `tf.Variable` will be created, or a `collections.OrderedDict`
        whose depth-first traversal matches the order of `tf.Variable` creation.

    Returns:
      A callable that can be used in a `tf.variable_creator_scope` to replace
      `tf.Variable` creation with `tf.function` parameters.
    """
    if isinstance(parameters, collections.OrderedDict):
      parameter_iterator = itertools.chain(*parameters.values())
    elif isinstance(parameters, list):
      parameter_iterator = iter(parameters)
    else:
      raise TypeError('Internal coding error: `parameters` must be a `list` or '
                      f'`OrderedDict` type, got {type(parameters)}.')

    def creator_fn(next_creator_fn, **kwargs):
      del next_creator_fn  # Unused.
      kwargs.pop('initial_value')
      return variable_utils.TensorVariable(
          initial_value=next(parameter_iterator), **kwargs)

    return creator_fn

  def _get_unwrapped_py_func(fn: Any) -> Callable[..., Any]:
    """Unwraps a `tf.function` decorated method."""
    # Its possible a function was decorated with `tf.function` more than once,
    # so we need to loop here.
    while hasattr(fn, '__original_wrapped__'):
      fn = fn.__original_wrapped__
    return fn

  @tf.function
  def update(state, y_true, y_pred, sample_weight=None):
    del sample_weight  # Unused.

    def inner_update(metric: tf.keras.metrics.Metric) -> List[tf.Tensor]:
      # We must unwrap `update_state` here because the `TensorVariable` is
      # created in the outer `update` FuncGraph and since it is not constant
      # it can't be closed over in the `update_state` FuncGraph. The
      # `tf.function` was used to get ACD, which the `TensorVariable` provides
      # for us, so we simply unwrap the function and call the Python method
      # directly.
      update_state_fn = _get_unwrapped_py_func(metric.update_state)
      update_state_fn(y_true=y_true, y_pred=y_pred)
      return [v.read_value() for v in metric.variables]

    with tf.variable_creator_scope(
        build_replace_variable_with_parameter_creator(state)):
      return tf.nest.map_structure(inner_update, metrics_constructor())

  @tf.function
  def finalize(state):
    with tf.variable_creator_scope(
        build_replace_variable_with_parameter_creator(state)):
      return tf.nest.map_structure(lambda metric: metric.result(),
                                   metrics_constructor())

  return initialize, update, finalize
