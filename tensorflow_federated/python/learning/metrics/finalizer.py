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

from typing import Any, Callable, List, Union
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck

# A finalizer of a Keras metric is a `tf.function` decorated callable that takes
# in the unfinalized values of this Keras metric (i.e., the tensor values of the
# variables in `keras_metric.variables`), and returns the value of
# `keras_metric.result()`.
KerasMetricFinalizer = Callable[[List[tf.Tensor]], Any]


# TODO(b/197746608): removes the code path that takes in a constructed Keras
# metric, because reconstructing metric via `from_config` can cause problems.
def create_keras_metric_finalizer(
    metric: Union[tf.keras.metrics.Metric, Callable[[],
                                                    tf.keras.metrics.Metric]]
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
  def finalizer(unfinalized_metric_values: List[tf.Tensor]):

    # Construct a new keras metirc here, which is necessary because this
    # `tf.function` may be invoked in a different context as the `model_fn`, and
    # we need the `tf.Variable`s to be created in the current scope in order to
    # use `keras_metric.result()`.
    with tf.init_scope():
      if isinstance(metric, tf.keras.metrics.Metric):
        keras_metric = type(metric).from_config(metric.get_config())
      elif callable(metric):
        keras_metric = metric()
        if not isinstance(keras_metric, tf.keras.metrics.Metric):
          raise TypeError(
              'Expected input `metric` to be either a `tf.keras.metrics.Metric`'
              ' or a no-arg callable that creates a `tf.keras.metrics.Metric`, '
              'found a callable that returns a '
              f'{py_typecheck.type_string(type(keras_metric))}.')
      else:
        raise TypeError(
            'Expected input `metric` to be either a `tf.keras.metrics.Metric` '
            'or a no-arg callable that constructs a `tf.keras.metrics.Metric`, '
            f'found a non-callable {py_typecheck.type_string(type(metric))}.')
    py_typecheck.check_type(unfinalized_metric_values, list)
    if len(keras_metric.variables) != len(unfinalized_metric_values):
      raise ValueError(
          'The input to the finalizer should be a list of `tf.Tensor`s matching'
          f' the variables of the Keras metric {keras_metric.name}. Expected '
          f'a list of `tf.Tensor`s of length {len(keras_metric.variables)}, '
          f'found a list of length {len(unfinalized_metric_values)}.')
    for v, a in zip(keras_metric.variables, unfinalized_metric_values):
      py_typecheck.check_type(a, tf.Tensor)
      if v.shape != a.shape or v.dtype != a.dtype:
        raise ValueError(
            'The input to the finalizer should be a list of `tf.Tensor`s '
            f'matching the variables of the Keras metric {keras_metric.name}. '
            f'Expected a `tf.Tensor` of shape {v.shape} and dtype {v.dtype!r}, '
            f'found a `tf.Tensor` of shape {a.shape} and dtype {a.dtype!r}.')
      v.assign(a)
    return keras_metric.result()

  return finalizer
