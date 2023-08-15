# Copyright 2021, The TensorFlow Federated Authors.
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
"""Helper functions for creating metric finalizers."""

from collections.abc import Callable
import inspect
from typing import Any, Union

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck

# A finalizer of a Keras metric is a `tf.function` decorated callable that takes
# in the unfinalized values of this Keras metric (i.e., the tensor values of the
# variables in `keras_metric.variables`), and returns the value of
# `keras_metric.result()`.
KerasMetricFinalizer = Callable[[list[tf.Tensor]], Any]


# TODO: b/197746608 - removes the code path that takes in a constructed Keras
# metric, because reconstructing metric via `from_config` can cause problems.
def create_keras_metric_finalizer(
    metric: Union[
        tf.keras.metrics.Metric, Callable[[], tf.keras.metrics.Metric]
    ]
) -> KerasMetricFinalizer:
  """Creates a finalizer function for the given Keras metric.

  Args:
    metric: An instance of `tf.keras.metrics.Metric` or a no-arg callable that
      constructs a `tf.keras.metrics.Metric`.

  Returns:
    A `tf.function` decorated callable that takes in the unfinalized metric
    values (i.e., tensor values of the variables in `keras_metric.variables`),
    and returns the value of `keras_metric.result()`.

  Raises:
    TypeError: If `metric` is not a `tf.keras.metrics.Metric` and not a no-arg
      callable that returns a `tf.keras.metrics.Metric`.
  """

  @tf.function
  def finalizer(unfinalized_metric_values: list[tf.Tensor]):
    # Construct a new keras metirc here, which is necessary because this
    # `tf.function` may be invoked in a different context as the `model_fn`, and
    # we need the `tf.Variable`s to be created in the current scope in order to
    # use `keras_metric.result()`.
    with tf.init_scope():
      keras_metric = create_keras_metric(metric)
    py_typecheck.check_type(unfinalized_metric_values, list)
    if len(keras_metric.variables) != len(unfinalized_metric_values):
      raise ValueError(
          'The input to the finalizer should be a list of `tf.Tensor`s matching'
          f' the variables of the Keras metric {keras_metric.name}. Expected '
          f'a list of `tf.Tensor`s of length {len(keras_metric.variables)}, '
          f'found a list of length {len(unfinalized_metric_values)}.'
      )
    for v, a in zip(keras_metric.variables, unfinalized_metric_values):
      py_typecheck.check_type(a, tf.Tensor)
      if v.shape != a.shape or v.dtype != a.dtype:
        raise ValueError(
            'The input to the finalizer should be a list of `tf.Tensor`s '
            f'matching the variables of the Keras metric {keras_metric.name}. '
            f'Expected a `tf.Tensor` of shape {v.shape} and dtype {v.dtype!r}, '
            f'found a `tf.Tensor` of shape {a.shape} and dtype {a.dtype!r}.'
        )
      v.assign(a)
    return keras_metric.result()

  return finalizer


def _check_keras_metric_config_constructable(metric: tf.keras.metrics.Metric):
  """Checks that a Keras metric is constructable from the `get_config()` method.

  Args:
    metric: A single `tf.keras.metrics.Metric`.

  Raises:
    TypeError: If the metric is not an instance of `tf.keras.metrics.Metric`, if
    the metric is not constructable from the `get_config()` method.
  """
  if not isinstance(metric, tf.keras.metrics.Metric):
    raise TypeError(
        f'Metric {type(metric)} is not a `tf.keras.metrics.Metric` '
        'to be constructable from the `get_config()` method.'
    )

  metric_type_str = type(metric).__name__

  if not hasattr(tf.keras.metrics, metric_type_str):
    _, init_fn = tf.__internal__.decorator.unwrap(metric.__init__)
    init_args = inspect.getfullargspec(init_fn).args
    init_args.remove('self')
    get_config_args = metric.get_config().keys()
    extra_args = [arg for arg in init_args if arg not in get_config_args]
    if extra_args:
      # TODO: b/197746608 - Remove the suggestion of updating `get_config` if
      # that code path is removed.
      raise TypeError(
          f'Metric {metric_type_str} is not constructable from the '
          '`get_config()` method, because `__init__` takes extra arguments '
          f'that are not included in the `get_config()`: {extra_args}. '
          'Pass the metric constructor instead, or update the `get_config()` '
          'in the metric class to include these extra arguments.\n'
          'Example:\n'
          'class CustomMetric(tf.keras.metrics.Metric):\n'
          '  def __init__(self, arg1):\n'
          '    self._arg1 = arg1\n\n'
          '  def get_config(self)\n'
          '    config = super().get_config()\n'
          "    config['arg1'] = self._arg1\n"
          '    return config'
      )


def create_keras_metric(
    metric: Union[
        tf.keras.metrics.Metric, Callable[[], tf.keras.metrics.Metric]
    ]
) -> tf.keras.metrics.Metric:
  """Create a `tf.keras.metrics.Metric` from a `tf.keras.metrics.Metric`.

  So the `tf.Variable`s in the metric can get created in the right scope in TFF.

  Args:
    metric: A single `tf.keras.metrics.Metric` or a no-arg callable that creates
      a `tf.keras.metrics.Metric`.

  Returns:
    A `tf.keras.metrics.Metric` object.

  Raises:
    TypeError: If input metric is neither a `tf.keras.metrics.Metric` or a
    no-arg callable that creates a `tf.keras.metrics.Metric`.
  """
  keras_metric = None
  if isinstance(metric, tf.keras.metrics.Metric):
    _check_keras_metric_config_constructable(metric)
    keras_metric = type(metric).from_config(metric.get_config())
  elif callable(metric):
    keras_metric = metric()
    if not isinstance(keras_metric, tf.keras.metrics.Metric):
      raise TypeError(
          'Expected input `metric` to be either a `tf.keras.metrics.Metric` '
          'or a no-arg callable that creates a `tf.keras.metrics.Metric`, '
          'found a callable that returns a '
          f'{py_typecheck.type_string(type(keras_metric))}.'
      )
  else:
    raise TypeError(
        'Expected input `metric` to be either a `tf.keras.metrics.Metric` '
        'or a no-arg callable that constructs a `tf.keras.metrics.Metric`, '
        f'found a non-callable {py_typecheck.type_string(type(metric))}.'
    )
  return keras_metric
