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
"""Abstractions for models used in federated learning."""

import abc
import attr


@attr.s(frozen=True, slots=True, eq=False)
class BatchOutput():
  """A structure that holds the output of a `tff.learning.Model`.

  Note: All fields are optional (may be None).

  Attributes:
    loss: The scalar mean loss on the examples in the batch. If the model
      has multiple losses, it is the sum of all the individual losses.
    predictions: Tensor of predictions on the examples. The first dimension
      must be the same size (the size of the batch).
    num_examples: Number of examples seen in the batch.
  """
  loss = attr.ib()
  predictions = attr.ib()
  num_examples = attr.ib()


class Model(object, metaclass=abc.ABCMeta):
  """Represents a model for use in TensorFlow Federated.

  Each `Model` will work on a set of `tf.Variables`, and each method should be
  a computation that can be implemented as a `tf.function`; this implies the
  class should essentially be stateless from a Python perspective, as each
  method will generally only be traced once (per set of arguments) to create the
  corresponding TensorFlow graph functions. Thus, `Model` instances should
  behave as expected in both eager and graph (TF 1.0) usage.

  In general, `tf.Variables` may be either:

    * Weights, the variables needed to make predictions with the model.
    * Local variables, e.g. to accumulate aggregated metrics across
      calls to forward_pass.

  The weights can be broken down into trainable variables (variables
  that can and should be trained using gradient-based methods), and
  non-trainable variables (which could include fixed pre-trained layers,
  or static model data). These variables are provided via the
  `trainable_variables`, `non_trainable_variables`, and `local_variables`
  properties, and must be initialized by the user of the `Model`.

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

  @abc.abstractproperty
  def trainable_variables(self):
    """An iterable of `tf.Variable` objects, see class comment for details."""
    pass

  @abc.abstractproperty
  def non_trainable_variables(self):
    """An iterable of `tf.Variable` objects, see class comment for details."""
    pass

  @abc.abstractproperty
  def local_variables(self):
    """An iterable of `tf.Variable` objects, see class comment for details."""
    pass

  @abc.abstractproperty
  def input_spec(self):
    """The type specification of the `batch_input` parameter for `forward_pass`.

    A nested structure of `tf.TensorSpec` objects, that matches the structure of
    arguments that will be passed as the `batch_input` argument of
    `forward_pass`. The tensors must include a batch dimension as the first
    dimension, but the batch dimension may be undefined.

    Similar in spirit to `tf.keras.models.Model.input_spec`.
    """
    pass

  @abc.abstractmethod
  def forward_pass(self, batch_input, training=True):
    """Runs the forward pass and returns results.

    This method should not modify any variables that are part of the model
    parameters, that is, variables that influence the predictions. Rather, this
    is done by the training loop.

    However, this method may update aggregated metrics computed across calls to
    `forward_pass`; the final values of such metrics can be accessed via
    `aggregated_outputs`.

    Uses in TFF:

      * To implement model evaluation.
      * To implement federated gradient descent and other
        non-Federated-Averaging algorithms, where we want the model to run the
        forward pass and update metrics, but there is no optimizer
        (we might only compute gradients on the returned loss).
      * To implement Federated Averaging.

    Args:
      batch_input: a nested structure that matches the structure of
        `Model.input_spec` and each tensor in `batch_input` satisfies
        `tf.TensorSpec.is_compatible_with()` for the corresponding
        `tf.TensorSpec` in `Model.input_spec`.
      training: If `True`, run the training forward pass, otherwise, run in
        evaluation mode. The semantics are generally the same as the `training`
        argument to `keras.Model.__call__`; this might e.g. influence how
        dropout or batch normalization is handled.

    Returns:
      A `BatchOutput` object. The object must include the `loss` tensor if the
      model will be trained via a gradient-based algorithm.
    """
    pass

  @abc.abstractmethod
  def report_local_outputs(self):
    """Returns tensors representing values aggregated over `forward_pass` calls.

    In federated learning, the values returned by this method will typically
    be further aggregated across clients and made available on the server.

    This method returns results from aggregating across *all* previous calls
    to `forward_pass`, most typically metrics like accuracy and loss. If needed,
    we may add a `clear_aggregated_outputs` method, which would likely just
    run the initializers on the `local_variables`.

    In general, the tensors returned can be an arbitrary function of all
    the `tf.Variables` of this model, not just the `local_variables`; for
    example, this could return tensors measuring the total L2 norm of the model
    (which might have been updated by training).

    This method may return arbitrarily shaped tensors, not just scalar metrics.
    For example, it could return the average feature vector or a count of
    how many times each feature exceed a certain magnitude.

    Returns:
      A structure of tensors (as supported by `tf.nest`)
      to be aggregated across clients.
    """
    pass

  @abc.abstractproperty
  def federated_output_computation(self):
    """Performs federated aggregation of the `Model's` `local_outputs`.

    This is typically used to aggregate metrics across many clients, e.g. the
    body of the computation might be:

    ```python
    return {
        'num_examples': tff.federated_sum(local_outputs.num_examples),
        'loss': tff.federated_mean(local_outputs.loss)
    }
    ```

    N.B. It is assumed all TensorFlow computation happens in the
    `report_local_outputs` method, and this method only uses TFF constructs to
    specify aggregations across clients.

    Returns:
      Either a `tff.Computation`, or None if no federated aggregation is needed.

      The `tff.Computation` should take as its single input a
      `tff.CLIENTS`-placed `tff.Value` corresponding to the return value of
      `Model.report_local_outputs`, and return an `OrderedDict` (possibly
      nested) of `tff.SERVER`-placed values. The consumer of this
      method should generally provide these server-placed values as outputs of
      the overall computation consuming the model. Using an `OrderedDict`
      allows the value returned by TFF executor to be converted back to an
      `OrderedDict` via the `._asdict(recursive=True)` member function.
    """
    pass
