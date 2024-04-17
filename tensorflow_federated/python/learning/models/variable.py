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
"""Abstractions for variable-based models for use in federated learning."""

import abc
import collections
from collections.abc import Sequence
from typing import Any

import tensorflow as tf

from tensorflow_federated.python.learning.metrics import types

MODEL_ARG_NAME = 'x'
MODEL_LABEL_NAME = 'y'

BatchOutput = collections.namedtuple(
    'BatchOutput',
    ['loss', 'predictions', 'num_examples'],
    defaults=[None, None, None],
)
BatchOutput.__doc__ = """A structure that holds the output of a `tff.learning.models.VariableModel`.

  Note: All fields are optional (may be None).

  Attributes:
    loss: The scalar mean loss on the examples in the batch. If the model has
      multiple losses, it is the sum of all the individual losses.
    predictions: Tensor of predictions on the examples. The first dimension must
      be the same size (the size of the batch).
    num_examples: Number of examples seen in the batch.
  """


class VariableModel(metaclass=abc.ABCMeta):
  """Represents a variable-based model for use in TensorFlow Federated.

  Each `VariableModel` will work on a set of `tf.Variables`, and each method
  should be a computation that can be implemented as a `tf.function`; this
  implies the class should essentially be stateless from a Python perspective,
  as each method will generally only be traced once (per set of arguments) to
  create the corresponding TensorFlow graph functions. Thus, `VariableModel`
  instances should behave as expected in both eager and graph (TF 1.0) usage.

  In general, `tf.Variables` may be either:

    * Weights, the variables needed to make predictions with the model.
    * Local variables, e.g. to accumulate aggregated metrics across
      calls to forward_pass.

  The weights can be broken down into trainable variables (variables
  that can and should be trained using gradient-based methods), and
  non-trainable variables (which could include fixed pre-trained layers,
  or static model data). These variables are provided via the
  `trainable_variables`, `non_trainable_variables`, and `local_variables`
  properties, and must be initialized by the user of the `VariableModel`.

  In federated learning, model weights will generally be provided by the
  server, and updates to trainable model variables will be sent back to the
  server. Local variables are not transmitted, and are instead initialized
  locally on the device, and then used to produce `aggregated_outputs` which
  are sent to the server.

  All `tf.Variables` should be introduced in `__init__`; this could move to a
  `build` method more inline with Keras (see
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) in
  the future.
  """

  @property
  @abc.abstractmethod
  def trainable_variables(self) -> Sequence[tf.Variable]:
    """An iterable of `tf.Variable` objects, see class comment for details."""
    pass

  @property
  @abc.abstractmethod
  def non_trainable_variables(self) -> Sequence[tf.Variable]:
    """An iterable of `tf.Variable` objects, see class comment for details."""
    pass

  @property
  @abc.abstractmethod
  def local_variables(self) -> Sequence[tf.Variable]:
    """An iterable of `tf.Variable` objects, see class comment for details."""
    pass

  @property
  @abc.abstractmethod
  def input_spec(self):
    """The type specification of the `batch_input` parameter for `forward_pass`.

    A nested structure of `tf.TensorSpec` objects, that matches the structure of
    arguments that will be passed as the `batch_input` argument of
    `forward_pass`. The tensors must include a batch dimension as the first
    dimension, but the batch dimension may be undefined.

    If `input_spec` is an instance of `collections.abc.Mapping`, this mapping
    must have an `{}` element which corresponds to the input to
    `predict_on_batch`  and a `{}` element containing the batch labels.
    Otherwise the first positional element of `input_spec` must correspond to
    the input to `predict_on_batch`, and the second positional element the
    labels.

    Similar in spirit to `tf.keras.models.Model.input_spec`.
    """.format(
        MODEL_ARG_NAME, MODEL_LABEL_NAME
    )
    pass

  @abc.abstractmethod
  def forward_pass(self, batch_input, training=True) -> BatchOutput:
    """Runs the forward pass and returns results.

    This method must be serializable in a `tff.tensorflow.computation` or other
    backend decorator. Any pure-Python or unserializable logic will not be
    runnable in the federated system.

    This method should not modify any variables that are part of the model
    parameters, that is, variables that influence the predictions (exceptions
    being updated, rather than learned, parameters such as BatchNorm means and
    variances). Rather, this is done by the training loop. However, this method
    may update aggregated metrics computed across calls to `forward_pass`; the
    final values of such metrics can be accessed via `aggregated_outputs`.

    Uses in TFF:

      * To implement model evaluation.
      * To implement federated gradient descent and other
        non-Federated-Averaging algorithms, where we want the model to run the
        forward pass and update metrics, but there is no optimizer
        (we might only compute gradients on the returned loss).
      * To implement Federated Averaging.

    Args:
      batch_input: A nested structure that matches the structure of
        `VariableModel.input_spec` and each tensor in `batch_input` satisfies
        `tf.TensorSpec.is_compatible_with()` for the corresponding
        `tf.TensorSpec` in `VariableModel.input_spec`.
      training: If `True`, run the training forward pass, otherwise, run in
        evaluation mode. The semantics are generally the same as the `training`
        argument to `keras.Model.call`; this might e.g. influence how dropout or
        batch normalization is handled.

    Returns:
      A `BatchOutput` object. The object must include the `loss` tensor if the
      model will be trained via a gradient-based algorithm.
    """
    pass

  @abc.abstractmethod
  def predict_on_batch(self, batch_input, training=True):
    """Performs inference on a batch, produces predictions.

    Unlike `forward_pass`, this function must _not_ mutate any variables
    (including metrics) when `training=False`, as it must support conversion to
    a TFLite flatbuffer for inference. When `training=True` this supports cases
    such as BatchNorm mean and variance updates or dropout. In many cases this
    method will be called from `forward_pass` to produce the predictions, and
    `forward_pass` will further compute loss and metrics updates.

    Note that this implies `batch_input` will have a *different* signature for
    `predict_on_batch` than for `forward_pass`; see the args section of this
    documentation for a specification of the relationship.

    Args:
      batch_input: A nested structure of tensors that holds the prediction
        inputs for the model. The structure must match the first element of the
        structure of `Model.input_spec`, or the '{}' key if `Model.input_spec`
        is a mapping. Each tensor in `x` satisfies
        `tf.TensorSpec.is_compatible_with()` for the corresponding
        `tf.TensorSpec` in `Model.input_spec`.
      training: If `True`, allow updatable variables (e.g. BatchNorm variances
        and means) to be updated. Otherwise, run in inferece only mode with no
        variables mutated. The semantics are generally the same as the
        `training` argument to `keras.Model.`; this might e.g. influence how
        dropout or batch normalization is handled.

    Returns:
      The model's inference result. The value must be understood by the loss
      function that will be used during training. In most cases this value will
      be the logits or probabilities of the last layer in the model, however
      writers are not restricted to these, the only requirement is their loss
      function understands the result.
    """.format(
        MODEL_ARG_NAME
    )

  @abc.abstractmethod
  def report_local_unfinalized_metrics(
      self,
  ) -> collections.OrderedDict[str, Any]:
    """Creates an `collections.OrderedDict` of metric names to unfinalized values.

    For a metric, its unfinalized values are given as a structure (typically a
    list) of tensors representing values from aggregating over *all* previous
    `forward_pass` calls, unless the `reset_metrics` is called. Each time the
    `reset_metrics` is called, the local metric variables will be reset, and
    `report_local_unfinalized_metrics` only reports metrics aggregated from the
    `forward_pass` calls since the *last* `reset_metrics` call. For a Keras
    metric, its unfinalized values are typically the tensor values of its state
    variables. In general, the tensors can be an arbitrary function of all the
    `tf.Variable`s of this model.

    The metric names returned by this method should be the same as those
    expected by the `metric_finalizers()`; one should be able to use the
    unfinalized values as input to the finalizers to get the finalized values.
    Taking `tf.keras.metrics.CategoricalAccuracy` as an example, its unfinalized
    values can be a list of two tensors (from its state variables): `total` and
    `count`, and the finalizer function performs a `tf.math.divide_no_nan`.

    In federated learning, this method returns the local results from clients,
    which will typically be further aggregated across clients and made available
    on the server. This method and the `metric_finalizers()` method will be used
    together to build a cross-client metrics aggregator. For example, a simple
    "sum_then_finalize" aggregator will first sum the unfinalized metric values
    from clients, and then call the finalizer functions at the server.

    Because both of this method and the `metric_finalizers()` method are defined
    in a per-metric manner, users have the flexiblity to call finalizer at the
    clients or at the server for different metrics. Users also have the freedom
    to defined a cross-client metrics aggregator that aggregates a single metric
    in multiple ways.

    Returns:
      An `collections.OrderedDict` of metric names to unfinalized values. The
      metric names
      must be the same as those expected by the `metric_finalizers()` method.
      One should be able to use the unfinalized metric values (returned by this
      method) as the input to the finalizers (returned by `metric_finalizers()`)
      to get the finalized metrics. This method and the `metric_finalizers()`
      method will be used together to build a cross-client metrics aggregator
      when defining the federated training processes or evaluation computations.
    """
    pass

  # TODO: b/233054212 - re-enable lint
  @abc.abstractmethod
  def metric_finalizers(self) -> types.MetricFinalizersType:  # pylint: disable=g-bare-generic
    """Creates an `collections.OrderedDict` of metric names to finalizers.

    This method and the `report_local_unfinalized_metrics()` method should have
    the same keys (i.e., metric names). A finalizer returned by this method is a
    function (typically a `tf.function` decorated callable or a
    `tff.tensorflow.computation` decorated TFF Computation) that takes in a
    metric's unfinalized values (returned by
    `report_local_unfinalized_metrics()`), and returns the finalized metric
    values.

    This method and the `report_local_unfinalized_metrics()` method will be used
    together to build a cross-client metrics aggregator. See the documentation
    of `report_local_unfinalized_metrics()` for more information.

    Returns:
      An `collections.OrderedDict` of metric names to finalizers. The metric
      names must be
      the same as those from the `report_local_unfinalized_metrics()` method. A
      finalizer is a `tf.function` (or `tff.tensorflow.computation`) decorated
      callable that takes in a metric's unfinalized values, and returns the
      finalized values. This method and the `report_local_unfinalized_metrics()`
      method will be used together to build a cross-client metrics aggregator in
      federated training processes or evaluation computations.
    """
    pass

  @abc.abstractmethod
  def reset_metrics(self) -> None:
    """Resets metrics variables to initial value.

    This method is a `tf.function`. It is used to reset the metrics variables
    between different stages in client's *local* computation. Each time the
    `reset_metrics` is called, the *local* metric variables will be reset, and
    `report_local_unfinalized_metrics` only reports metrics aggregated from the
    `forward_pass` calls since the *last* `reset_metrics` call. If the
    `reset_metrics` is never called, `report_local_unfinalized_metrics` will
    report metrics aggregated over *all* previous `forward_pass` calls.
    """
    pass
