# Copyright 2020, The TensorFlow Federated Authors.
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
"""Abstractions for Federated Reconstruction Models."""

import abc
import collections
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple, Optional

import tensorflow as tf

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning.models import model_weights


class ReconstructionBatchOutput(NamedTuple):
  """A structure for the output of a `tff.learning.models.ReconstructionModel`.

  Note: All fields are optional (may be None).

  Attributes:
    predictions: A `tf.Tensor` of predictions on the examples.
    labels: A `tf.Tensor` of labels for the examples.
    num_examples: A `tf.int32` scalar number of examples seen in the batch.
  """

  predictions: Any
  labels: Any
  num_examples: Any


class ReconstructionModel(metaclass=abc.ABCMeta):
  """Represents a reconstruction model for use in Tensorflow Federated.

  `tff.learning.models.ReconstructionModel`s are used to train models that
  reconstruct a set of their variables on device, never sharing those variables
  with the
  server.

  Each `tff.learning.models.ReconstructionModel` will work on a set of
  `tf.Variables`, and each method should be a computation that can be
  implemented as a `tf.function`; this implies the class should essentially be
  stateless from a Python perspective, as each method will generally only be
  traced once (per set of arguments) to create the corresponding TensorFlow
  graph functions. Thus, `tff.learning.models.ReconstructionModel` instances
  should behave as expected in both eager and graph (TF 1.0) usage.

  In general, `tf.Variables` may be either:
    * Weights, the variables needed to make predictions with the model.
    * Local variables, e.g. to accumulate aggregated metrics across
      calls to forward_pass.

  The weights can be broken down into:
    * Global variables: Variables that are allowed to be aggregated on the
      server.
    * Local variables: Variables that cannot leave the device.

  Furthermore, both of these types of variables can be:
    * Trainable variables: These can and should be trained using gradient-based
      methods.
    * Non-trainable variables: Could include fixed pre-trained layers or static
      model data.

  These variables are provided via:

    * `global_trainable_variables`
    * `global_non_trainable_variables`
    * `local_trainable_variables`
    * `local_non_trainable_variables`

  properties, and must be initialized by the user of the
  `tff.learning.models.ReconstructionModel`.

  While training a reconstruction model, global trainable variables will
  generally be provided by the server. Local trainable variables will then be
  reconstructed locally. Updates to the global trainable variables will be sent
  back to the server. Local variables are not transmitted.

  All `tf.Variables` should be introduced in `__init__`; this could move to a
  `build` method more inline with Keras (see
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) in
  the future.
  """

  @property
  @abc.abstractmethod
  def global_trainable_variables(self):
    """An iterable of `tf.Variable` objects, see class comment for details."""

  @property
  @abc.abstractmethod
  def global_non_trainable_variables(self):
    """An iterable of `tf.Variable` objects, see class comment for details."""

  @property
  @abc.abstractmethod
  def local_trainable_variables(self):
    """An iterable of `tf.Variable` objects, see class comment for details."""

  @property
  @abc.abstractmethod
  def local_non_trainable_variables(self):
    """An iterable of `tf.Variable` objects, see class comment for details."""

  @property
  @abc.abstractmethod
  def input_spec(self):
    """The type specification of the `batch_input` parameter for `forward_pass`.

    A nested structure of `tf.TensorSpec` objects, that matches the structure of
    arguments that will be passed as the `batch_input` argument of
    `forward_pass`. The tensors must include a batch dimension as the first
    dimension, but the batch dimension may be undefined.
    """

  @abc.abstractmethod
  def forward_pass(self, batch_input, training=True):
    """Runs the forward pass and returns results.

    This method should not modify any variables that are part of the model
    parameters, that is, variables that influence the predictions. Rather, this
    is done by the training loop.

    Args:
      batch_input: A nested structure that matches the structure of
        `Model.input_spec` and each tensor in `batch_input` satisfies
        `tf.TensorSpec.is_compatible_with()` for the corresponding
        `tf.TensorSpec` in `Model.input_spec`.
      training: If `True`, run the training forward pass, otherwise, run in
        evaluation mode. The semantics are generally the same as the `training`
        argument to `keras.Model.__call__`; this might e.g. influence how
        dropout or batch normalization is handled.

    Returns:
      A `ReconstructionBatchOutput` object.
    """

  @classmethod
  def get_global_variables(
      cls, model: 'ReconstructionModel'
  ) -> model_weights.ModelWeights:
    """Gets global variables from `model` as `ModelWeights`."""
    del cls  # Unused.
    return model_weights.ModelWeights(
        trainable=model.global_trainable_variables,
        non_trainable=model.global_non_trainable_variables,
    )

  @classmethod
  def get_local_variables(
      cls, model: 'ReconstructionModel'
  ) -> model_weights.ModelWeights:
    """Gets local variables from a `Model` as `ModelWeights`."""
    del cls  # Unused.
    return model_weights.ModelWeights(
        trainable=model.local_trainable_variables,
        non_trainable=model.local_non_trainable_variables,
    )

  @classmethod
  def has_only_global_variables(cls, model: 'ReconstructionModel') -> bool:
    """Returns `True` if the model has no local variables."""
    del cls  # Unused.
    return bool(model.local_trainable_variables) or bool(
        model.local_non_trainable_variables
    )

  # Type alias for a function that takes in a TF dataset and produces two TF
  # datasets. This is consumed by training and evaluation computation builders.
  # The first is iterated over during reconstruction and the second is iterated
  # over post-reconstruction, for both training and evaluation. This can be
  # useful for e.g. splitting the dataset into disjoint halves for each stage,
  # doing multiple local epochs of reconstruction/training, skipping
  # reconstruction entirely, etc. See `build_dataset_split_fn` for a builder,
  # although users can also specify their own `DatasetSplitFn`s (see
  # `simple_dataset_split_fn` for an example).
  # pylint: disable=invalid-name
  DatasetSplitFn = Callable[
      [tf.data.Dataset], tuple[tf.data.Dataset, tf.data.Dataset]
  ]
  # pylint: enable=invalid-name

  @classmethod
  def simple_dataset_split_fn(
      cls, client_dataset: tf.data.Dataset
  ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """An example of a `DatasetSplitFn` that returns the original client data.

    Both the reconstruction data and post-reconstruction data will result from
    iterating over the same tf.data.Dataset. Note that depending on any
    preprocessing steps applied to client tf.data.Datasets, this may not produce
    exactly the same data in the same order for both reconstruction and
    post-reconstruction. For example, if
    `client_dataset.shuffle(reshuffle_each_iteration=True)` was applied,
    post-reconstruction data will be in a different order than reconstruction
    data.

    Args:
      client_dataset: `tf.data.Dataset` representing client data.

    Returns:
      A tuple of two `tf.data.Datasets`, the first to be used for
      reconstruction, the second to be used for post-reconstruction.
    """
    del cls  # Unused.
    return client_dataset, client_dataset

  @classmethod
  def build_dataset_split_fn(
      cls,
      recon_epochs: int = 1,
      recon_steps_max: Optional[int] = None,
      post_recon_epochs: int = 1,
      post_recon_steps_max: Optional[int] = None,
      split_dataset: bool = False,
  ) -> DatasetSplitFn:
    """Builds a `DatasetSplitFn` for Federated Reconstruction training/evaluation.

    Returned `DatasetSplitFn` parameterizes training and evaluation computations
    and enables reconstruction for multiple local epochs, multiple epochs of
    post-reconstruction training, limiting the number of steps for both stages,
    and splitting client datasets into disjoint halves for each stage.

    Note that the returned function is used during both training and evaluation:
    during training, "post-reconstruction" refers to training of global
    variables, and during evaluation, it refers to calculation of metrics using
    reconstructed local variables and fixed global variables.

    Args:
      recon_epochs: The integer number of iterations over the dataset to make
        during reconstruction.
      recon_steps_max: If not None, the integer maximum number of steps
        (batches) to iterate through during reconstruction. This maximum number
        of steps is across all reconstruction iterations, i.e. it is applied
        after `recon_epochs`. If None, this has no effect.
      post_recon_epochs: The integer constant number of iterations to make over
        client data after reconstruction.
      post_recon_steps_max: If not None, the integer maximum number of steps
        (batches) to iterate through after reconstruction. This maximum number
        of steps is across all post-reconstruction iterations, i.e. it is
        applied after `post_recon_epochs`. If None, this has no effect.
      split_dataset: If True, splits `client_dataset` in half for each user,
        using even-indexed entries in reconstruction and odd-indexed entries
        after reconstruction. If False, `client_dataset` is used for both
        reconstruction and post-reconstruction, with the above arguments
        applied. If True, splitting requires that mupltiple iterations through
        the dataset yield the same ordering. For example if
        `client_dataset.shuffle(reshuffle_each_iteration=True)` has been called,
        then the split datasets may have overlap. If True, note that the dataset
        should have more than one batch for reasonable results, since the
        splitting does not occur within batches.

    Returns:
      A `SplitDatasetFn`.
    """
    del cls  # Unused.
    # Functions for splitting dataset if needed.
    recon_condition = lambda i, _: tf.equal(tf.math.floormod(i, 2), 0)
    post_recon_condition = lambda i, _: tf.greater(tf.math.floormod(i, 2), 0)
    get_entry = lambda _, entry: entry

    def dataset_split_fn(
        client_dataset: tf.data.Dataset,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
      """A `DatasetSplitFn` built with the given arguments.

      Args:
        client_dataset: `tf.data.Dataset` representing client data.

      Returns:
        A tuple of two `tf.data.Datasets`, the first to be used for
        reconstruction, the second to be used post-reconstruction.
      """
      # Split dataset if needed. This assumes the dataset has a consistent
      # order across iterations.
      if split_dataset:
        recon_dataset = (
            client_dataset.enumerate().filter(recon_condition).map(get_entry)
        )
        post_recon_dataset = (
            client_dataset.enumerate()
            .filter(post_recon_condition)
            .map(get_entry)
        )
      else:
        recon_dataset = client_dataset
        post_recon_dataset = client_dataset

      # Apply `recon_epochs` before limiting to a maximum number of batches
      # if needed.
      recon_dataset = recon_dataset.repeat(recon_epochs)
      if recon_steps_max is not None:
        recon_dataset = recon_dataset.take(recon_steps_max)

      # Do the same for post-reconstruction.
      post_recon_dataset = post_recon_dataset.repeat(post_recon_epochs)
      if post_recon_steps_max is not None:
        post_recon_dataset = post_recon_dataset.take(post_recon_steps_max)

      return recon_dataset, post_recon_dataset

    return dataset_split_fn

  @classmethod
  def read_metric_variables(
      cls, metrics: list[tf.keras.metrics.Metric]
  ) -> collections.OrderedDict[str, list[tf.Tensor]]:
    """Reads values from Keras metric variables."""
    del cls  # Unused.
    metric_variables = collections.OrderedDict()
    for metric in metrics:
      if metric.name in metric_variables:
        raise ValueError(
            f'Duplicate metric name detected: {metric.name}. '
            f'Already saw metrics {list(metric_variables.keys())}'
        )
      metric_variables[metric.name] = [v.read_value() for v in metric.variables]
    return metric_variables

  @classmethod
  def from_keras_model(
      cls,
      keras_model: tf.keras.Model,
      *,  # Caller passes below args by name.
      global_layers: Iterable[tf.keras.layers.Layer],
      local_layers: Iterable[tf.keras.layers.Layer],
      input_spec: Any,
  ) -> 'ReconstructionModel':
    """Builds a `tff.learning.models.ReconstructionModel` from a `tf.keras.Model`.

    The `tff.learning.models.ReconstructionModel` returned by this function uses
    `keras_model` for its forward pass and autodifferentiation steps. During
    reconstruction, variables in `local_layers` are initialized and trained.
    Post-reconstruction, variables in `global_layers` are trained and aggregated
    on the server. All variables must be partitioned between global and local
    layers, without overlap.

    Note: This function does not currently accept subclassed `tf.keras.Models`,
    as it makes assumptions about presence of certain attributes which are
    guaranteed to exist through the functional or Sequential API but are
    not necessarily present for subclassed models.

    Args:
      keras_model: A `tf.keras.Model` object that is not compiled.
      global_layers: Iterable of global layers to be aggregated across users.
        All trainable and non-trainable model variables that can be aggregated
        on the server should be included in these layers.
      local_layers: Iterable of local layers not shared with the server. All
        trainable and non-trainable model variables that should not be
        aggregated on the server should be included in these layers.
      input_spec: A structure of `tf.TensorSpec`s specifying the type of
        arguments the model expects. Notice this must be a compound structure of
        two elements, specifying both the data fed into the model to generate
        predictions, as its first element, as well as the expected type of the
        ground truth as its second.

    Returns:
      A `tff.learning.models.ReconstructionModel` object.

    Raises:
      TypeError: If `keras_model` is not an instance of `tf.keras.Model`.
      ValueError: If `keras_model` was compiled.
    """
    del cls  # Unused.
    if not isinstance(keras_model, tf.keras.Model):
      raise TypeError(
          'Expected `keras_model` to be type `tf.keras.Model`, '
          f'found {type(keras_model)}'
      )
    if len(input_spec) != 2:
      raise ValueError(
          'The top-level structure in `input_spec` must contain '
          'exactly two elements, as it must specify type '
          'information for both inputs to and predictions from the '
          'model.'
      )

    if keras_model._is_compiled:  # pylint: disable=protected-access
      raise ValueError('`keras_model` must not be compiled')

    return _KerasReconstructionModel(
        inner_model=keras_model,
        global_layers=global_layers,
        local_layers=local_layers,
        input_spec=input_spec,
    )


class _KerasReconstructionModel(ReconstructionModel):
  """Internal wrapper class for `tf.keras.Model` objects.

  Wraps uncompiled Keras models as `tff.learning.models.ReconstructionModel`s.
  Tracks global and local layers of the model. Parameters contained in global
  layers are sent to the server and aggregated across users normally, and
  parameters contained in local layers are reconstructed at the beginning of
  each round and not sent to the server. The loss function and metrics are
  passed to a `tff.templates.IterativeProcess` wrapping this model and computed
  there for both training and evaluation.
  """

  def __init__(
      self,
      inner_model: tf.keras.Model,
      global_layers: Iterable[tf.keras.layers.Layer],
      local_layers: Iterable[tf.keras.layers.Layer],
      input_spec: computation_types.Type,
  ):
    self._keras_model = inner_model
    self._global_layers = list(global_layers)
    self._local_layers = list(local_layers)
    self._input_spec = input_spec

    # Ensure global_layers and local_layers include exactly the Keras model's
    # trainable and non-trainable variables. Use hashable refs to uniquely
    # compare variables, and track variable names for informative error
    # messages.
    global_and_local_variables = set()
    for layer in self._global_layers + self._local_layers:
      global_and_local_variables.update(
          (var.ref(), var.name)
          for var in layer.trainable_variables + layer.non_trainable_variables
      )

    keras_variables = set(
        (var.ref(), var.name)
        for var in inner_model.trainable_variables
        + inner_model.non_trainable_variables
    )

    if global_and_local_variables != keras_variables:
      # Use a symmetric set difference to compare the variables, since either
      # set may include variables not present in the other.
      variables_difference = global_and_local_variables ^ keras_variables
      raise ValueError(
          'Global and local layers must include all trainable '
          'and non-trainable variables in the Keras model. '
          'Difference: {d}, Global and local layers vars: {v}, '
          'Keras vars: {k}'.format(
              d=variables_difference,
              v=global_and_local_variables,
              k=keras_variables,
          )
      )

  @property
  def global_trainable_variables(self) -> list[tf.Variable]:
    variables = []
    for layer in self._global_layers:
      variables.extend(layer.trainable_variables)
    return variables

  @property
  def global_non_trainable_variables(self) -> list[tf.Variable]:
    variables = []
    for layer in self._global_layers:
      variables.extend(layer.non_trainable_variables)
    return variables

  @property
  def local_trainable_variables(self) -> list[tf.Variable]:
    variables = []
    for layer in self._local_layers:
      variables.extend(layer.trainable_variables)
    return variables

  @property
  def local_non_trainable_variables(self) -> list[tf.Variable]:
    variables = []
    for layer in self._local_layers:
      variables.extend(layer.non_trainable_variables)
    return variables

  @property
  def input_spec(self):
    return self._input_spec

  @tf.function
  def forward_pass(
      self, batch_input, training=True
  ) -> ReconstructionBatchOutput:
    if hasattr(batch_input, '_asdict'):
      batch_input = batch_input._asdict()
    if isinstance(batch_input, Mapping):
      inputs = batch_input.get('x')
    else:
      inputs = batch_input[0]
    if inputs is None:
      raise KeyError(
          'Received a batch_input that is missing required key `x`. '
          'Instead have keys {}'.format(list(batch_input.keys()))
      )
    predictions = self._keras_model(inputs, training=training)

    if isinstance(batch_input, Mapping):
      y_true = batch_input.get('y')
    else:
      y_true = batch_input[1]

    return ReconstructionBatchOutput(
        predictions=predictions,
        labels=y_true,
        num_examples=tf.shape(tf.nest.flatten(inputs)[0])[0],
    )
