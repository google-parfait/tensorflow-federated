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
"""Module for creating functional implementations of a `tff.learning.Model`.

This version of the model parameterizes its `forward_pass` and
`predict_on_batch` methods by model weights, rather than storing them in the
model. This allows for greater flexibility in model portability.

To use with `tff.learning.algorithms` and other APIs that
construct learning processes expecting stateful models, wrap the functional
model with `tff.learning.models.model_from_functional`.
"""

import collections
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, TypeVar, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning.metrics import keras_finalizer
from tensorflow_federated.python.learning.metrics import keras_utils
from tensorflow_federated.python.learning.metrics import types
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.tensorflow_libs import variable_utils

Weight = Union[np.ndarray, int, float]
WeightStruct = Union[Sequence[Weight], Mapping[str, Weight]]
ModelWeights = tuple[WeightStruct, WeightStruct]
InitializeMetricsStateFn = Callable[[], types.MetricsState]
UpdateMetricsStateFn = Callable[
    [types.MetricsState, Any, Any, Any], types.MetricsState
]
FinalizeMetricsFn = Callable[[types.MetricsState], Any]
GenericMetricsState = TypeVar('GenericMetricsState', bound=types.MetricsState)


class CallableNotTensorFlowFunctionError(TypeError):
  """Error raised when a callable is not decorated as a tf.function."""


class ValueMustNotBeTFError(TypeError):
  """Error raised a value must not be a `tf.Tensor` or `tf.Variable`."""


@tf.function
def empty_metrics_state() -> types.MetricsState:
  return collections.OrderedDict()


@tf.function
def noop_update_metrics(
    state: types.MetricsState,
    labels: Any,
    batch_output: variable.BatchOutput,
    sample_weight: Optional[Any] = None,
) -> types.MetricsState:
  del state  # Unused.
  del labels  # Unused.
  del batch_output  # Unused.
  del sample_weight  # Unused.
  return collections.OrderedDict()


@tf.function
def noop_finalize_metrics(
    state: types.MetricsState,
) -> collections.OrderedDict[str, Any]:
  del state  # Unused.
  return collections.OrderedDict()


class FunctionalModel:
  """A model that parameterizes forward pass by model weights."""

  def __init__(
      self,
      *,  # Require all arguments be named.
      initial_weights: ModelWeights,
      forward_pass_fn: Callable[
          [ModelWeights, Any, bool], variable.BatchOutput
      ],
      predict_on_batch_fn: Callable[[ModelWeights, Any, bool], Any],
      metrics_fns: tuple[
          InitializeMetricsStateFn, UpdateMetricsStateFn, FinalizeMetricsFn
      ] = (empty_metrics_state, noop_update_metrics, noop_finalize_metrics),
      input_spec: Any,
  ):
    """Initializes a `FunctionalModel`.

    Example model implementing linear regression:

    ```
    w, b = np.zeros(shape=[1,3]), np.zeros([1])
    trainable_weights = (w, b)
    non_trainable_weights = ()
    initial_weights = (trainable_weights, non_trainable_weights)

    @tf.function
    def predict_on_batch(model_weights, x, training):
      del training  # Unused.
      trainable, non_trainable = model_weights
      w, b = trainable
      return tf.matmul(x, w, transpose_b=True) + b

    @tf.function
    def forward_pass(model_weights, batch_input, training):
      x, y = batch_input
      predictions = predict_on_batch(model_weights, , training)
      residuals = predictions - y
      total_loss = tf.reduce_sum(tf.pow(residuals, 2.))
      num_examples = tf.shape(predictions)[0]
      average_loss = total_loss / tf.cast(num_examples, tf.float32)
      return tff.learning.BatchOutput(
        loss=average_loss, predictions=predictions, num_examples=num_examples)

    model = FunctionalModel(
      initial_weights, forward_pass, predict_on_batch,
      (tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
       tf.TensorSpec(shape=[None, 1], dtype=tf.float32))
    )
    ```

    Args:
      initial_weights: A 2-tuple `(trainable, non_trainable)` where the two
        elements are sequences of weights. Weights must be values convertable to
        `tf.Tensor` (e.g. `numpy.ndarray`, Python sequences, etc), but _not_
        `tf.Tensor` values.
      forward_pass_fn: A `tf.function` decorated callable that takes three
        arguments, `model_weights` the same structure as `initial_weights`,
        `batch_input` a nested structure of tensors matching `input_spec`, and
        `training` a boolean determinig whether the call is during a training
        pass (e.g. for Dropout, BatchNormalization, etc).
      predict_on_batch_fn: A `tf.function` decorated callable that takes three
        arguments, `model_weights` the same structure as `initial_weights`, `x`
        the first element of `batch_input` (or `input_spec`), and `training` a
        boolean determinig whether the call is during a training pass (e.g. for
        Dropout, BatchNormalization, etc).
      metrics_fns: A 3-tuple of callables that initialize the metrics state,
        update the metrics state, and finalize the metrics values respectively.
        This can be the result of `
        tff.learning.metrics.create_functional_metric_fns`or custom user written
        callables.
      input_spec: A 2-tuple of `(x, y)` where each element is a nested structure
        of `tf.TensorSpec` that defines the shape and dtypes of `batch_input` to
        `forward_pass_fn`. `x` corresponds to batched model inputs and `y`
        corresponds to batched labels for those inputs.
    """

    def check_tf_function_decorated(fn: Any, arg_name: str) -> None:
      if not hasattr(fn, 'get_concrete_function'):
        type_string = py_typecheck.type_string(type(fn))
        raise CallableNotTensorFlowFunctionError(
            f'{arg_name} does not have a `get_concrete_function` attribute '
            'meaning it is not a callable decorated with `tf.function`. '
            f'Got a {type_string} with value {fn!r}.'
        )

    def check_non_tf_value(value):
      if tf.is_tensor(value) or isinstance(value, tf.Variable):
        raise ValueMustNotBeTFError(
            'initial_weights may not contain TensorFlow values '
            f'(tf.Tensor or tf.Variable). Got: {type(value)!r}. Try '
            'converting to a np.ndarray by using the `.numpy()` '
            'attribute for tf.Tensor, or `.read_value().numpy()` '
            'for tf.Variable.'
        )

    tf.nest.map_structure(check_non_tf_value, initial_weights)
    self._initial_weights = initial_weights
    check_tf_function_decorated(forward_pass_fn, 'forward_pass_fn')
    self._forward_pass_fn = forward_pass_fn
    check_tf_function_decorated(predict_on_batch_fn, 'predict_on_batch_fn')
    self._predict_on_batch_fn = predict_on_batch_fn
    self._input_spec = input_spec
    (
        self._initialize_metrics_state,
        self._update_metrics_state,
        self._finalize_metrics,
    ) = metrics_fns

  @property
  def initial_weights(self) -> ModelWeights:
    return self._initial_weights

  @tf.function
  def forward_pass(
      self, model_weights: ModelWeights, batch_input: Any, training: bool = True
  ) -> variable.BatchOutput:
    """Runs the forward pass and returns results."""
    return self._forward_pass_fn(model_weights, batch_input, training)

  @tf.function
  def predict_on_batch(
      self, model_weights: ModelWeights, x: Any, training: bool = True
  ):
    """Returns tensor(s) interpretable by the loss function."""
    return self._predict_on_batch_fn(model_weights, x, training)

  @tf.function
  def initialize_metrics_state(self) -> types.MetricsState:
    return self._initialize_metrics_state()

  @tf.function
  def update_metrics_state(
      self,
      state: GenericMetricsState,
      labels: Any,
      batch_output: variable.BatchOutput,
      sample_weight: Optional[Any] = None,
  ) -> GenericMetricsState:
    return self._update_metrics_state(
        state,
        labels=labels,
        batch_output=batch_output,
        sample_weight=sample_weight,
    )

  @tf.function
  def finalize_metrics(
      self, state: types.MetricsState
  ) -> collections.OrderedDict[str, Any]:
    return self._finalize_metrics(state)

  @property
  def input_spec(self):
    return self._input_spec


class _ModelFromFunctional(variable.VariableModel):
  """A `tff.learning.Model` wrapping a `tff.learning.model.FunctionalModel`."""

  def __init__(
      self,
      functional_model: FunctionalModel,
      metric_builders: Sequence[Callable[[], tf.keras.metrics.Metric]] = (),
  ):
    self._functional_model = functional_model
    # Construct `tf.Variable` to optimize during the learning process.
    trainable, non_trainable = functional_model.initial_weights
    self._trainable_variables = tuple(tf.Variable(x) for x in trainable)
    self._non_trainable_variables = tuple(
        tf.Variable(x, trainable=False) for x in non_trainable
    )
    self._model_weights = (
        self._trainable_variables,
        self._non_trainable_variables,
    )
    self._num_examples = tf.Variable(0, trainable=False)
    self._loss_sum = tf.Variable(0.0, trainable=False)
    if not metric_builders:
      self._metric_builders = []
      self._metrics = []
    else:
      self._metric_builders = metric_builders
      self._metrics = [constructor() for constructor in metric_builders]
      # Raise an error if there are duplicate metric names
      metric_names = [metric.name for metric in self._metrics]
      duplicates = set(
          name for name in metric_names if metric_names.count(name) > 1
      )
      if duplicates:
        raise ValueError(
            f'{duplicates} appeared in the metric names more than once, '
            'each metric should have a unique name.'
        )

  @property
  def trainable_variables(self) -> tuple[tf.Variable, ...]:
    return self._trainable_variables

  @property
  def non_trainable_variables(self) -> tuple[tf.Variable, ...]:
    return self._non_trainable_variables

  @property
  def local_variables(self) -> tuple[tf.Variable, ...]:
    metrics_variables = [self._loss_sum, self._num_examples]
    for metric in self._metrics:
      metrics_variables.extend(metric.variables)
    return tuple(metrics_variables)

  @property
  def input_spec(self):
    return self._functional_model.input_spec

  @tf.function
  def forward_pass(self, batch_input, training=True):
    batch_output = self._functional_model.forward_pass(
        model_weights=tf.nest.map_structure(
            lambda v: v.read_value(), self._model_weights
        ),
        batch_input=batch_input,
        training=training,
    )
    self._num_examples.assign_add(batch_output.num_examples)
    self._loss_sum.assign_add(
        batch_output.loss * tf.cast(batch_output.num_examples, tf.float32)
    )
    if isinstance(batch_input, Mapping):
      y_true = batch_input.get('y')
    else:
      y_true = batch_input[1]
    for metric in self._metrics:
      metric.update_state(y_true=y_true, y_pred=batch_output.predictions)
    return batch_output

  @tf.function
  def predict_on_batch(self, x, training=True):
    return self._functional_model.predict_on_batch(
        model_weights=tf.nest.map_structure(
            lambda v: v.read_value(), self._model_weights
        ),
        x=x,
        training=training,
    )

  @tf.function
  def report_local_unfinalized_metrics(
      self,
  ) -> collections.OrderedDict[str, list[tf.Tensor]]:
    outputs = collections.OrderedDict(
        loss=[self._loss_sum, tf.cast(self._num_examples, tf.float32)]
    )
    for metric in self._metrics:
      outputs[metric.name] = [v.read_value() for v in metric.variables]
    return outputs

  def metric_finalizers(
      self,
  ) -> collections.OrderedDict[str, keras_finalizer.KerasMetricFinalizer]:
    finalizers = collections.OrderedDict(
        # `loss` result is computed by `loss_sum` / `num_examples`.
        loss=tf.function(func=lambda x: x[0] / x[1])
    )
    for metric_builder in self._metric_builders:
      metric_name = metric_builder().name
      finalizers[metric_name] = keras_finalizer.create_keras_metric_finalizer(
          metric_builder
      )
    return finalizers

  @tf.function
  def reset_metrics(self):
    for metric in self._metrics:
      metric.reset_state()
    additional_metrics_variables = [self._loss_sum, self._num_examples]
    for var in additional_metrics_variables:
      var.assign(tf.zeros_like(var))


def model_from_functional(
    functional_model: FunctionalModel,
    metric_constructors: Sequence[Callable[[], tf.keras.metrics.Metric]] = (),
) -> variable.VariableModel:
  """Converts a `FunctionalModel` to a `tff.learning.Model`.

  WARNING: The `metrics_constructors` argument will *replace* any metrics that
  were originally attached to the `FunctionalModel` with new metrics.

  Args:
      functional_model: A `tff.learning.models.FunctionalModel` to convert.
      metric_constructors: An optional sequence of callables that return newly
        constructed `tf.keras.metrics.Metric` objects to attached to the output
        `tff.learning.Model`.

  Returns:
    A new `tff.learning.Model` with the same behavior as `functional_model`.
  """
  return _ModelFromFunctional(functional_model, metric_constructors)


class KerasFunctionalModelError(Exception):
  """An error raised when a FunctionalModel backed by Keras is used outside TFF."""


def functional_model_from_keras(
    keras_model: Union[tf.keras.Model, Callable[[], tf.keras.Model]],
    loss_fn: tf.keras.losses.Loss,
    input_spec: Union[Sequence[Any], Mapping[str, Any]],
    metrics_constructor: Optional[
        Union[
            keras_utils.MetricConstructor,
            keras_utils.MetricsConstructor,
            keras_utils.MetricConstructors,
        ]
    ] = None,
) -> FunctionalModel:
  """Converts a `tf.keras.Model` to a `tff.learning.models.FunctionalModel`.

  NOTE: This method only supports models where calling that model with
  `training=True` and `training=False` produce the same graph. Keras layers
  such as batch normalization will fail because they require updating internal
  state when `training=True` which is not suported.

  IMPORTANT: The returned model must only be used in a graph context (for
  example inside a `tff.tf_computation` decorated callable). It will raise an
  error otherwise.

  Args:
    keras_model: A `tf.keras.Model` object, should be uncompiled. If compiled,
      the metrics, optimizer, and loss function will be ignored. Note: models
      that have multiple outputs will send all outputs to the `loss_fn`.
    loss_fn: A `tf.keras.losses.Loss` object.
    input_spec: A structure of `tf.TensorSpec` defining the input to the model.
    metrics_constructor: An optional callable that must be compatible with
      `tff.learning.metrics.create_functional_metric_fns`.

  Returns:
    A `tff.learning.models.FunctionalModel`.

  Raises:
    KerasFunctionalModelError: the model has a batch normalization layer.
  """
  # We're going to do something fancy here:
  #
  # 1. Get a copy of all the variables, in the order they are created during
  #    model construction, when in a graph context.
  # 2. Use this ordering to construct a type signature of the model weights in
  #    such a way that we can inject TENSORS (those that are coming in as
  #    arguments) in place of variable creation during a call to
  #    `tf.keras.models.clone_model()`, which gives us a newly constructed Keras
  #    model in the context we want.
  # 3. Profit by having variableless graphs!
  #
  # **WARNING** Caveats:
  #
  # 1. This model _must_ be used inside a graph context (e.g. a
  #    `tff.tf_computation` decorated callable, aka a `tff.Computation`). Keras
  #    appears to create extra variables in the eager context that are not part
  #    of the user specified model, and end up not being compatible.
  #
  # 2. We have found that this trick does NOT work with non-trainable variables
  #    that are updated during training. Namely layers such as
  #    BatchNormalization try to update means/variances during training and are
  #    not compatible with this approach. We generally recommend
  #    GroupNormalization in place of BatchNormalization at the current time.
  #
  # 3. This does not support multiple outputs with different loss functions, or
  #    laywerise regularization losses TODO(b/156629927).
  if isinstance(keras_model, tf.keras.Model):
    for layer in keras_model.layers:
      # There may be other layers that are problematic, at this time updating
      # the mean/variance in batchnorm layer is the only known such instance.
      if isinstance(layer, tf.keras.layers.BatchNormalization):
        raise KerasFunctionalModelError(
            'Keras model contains a batch normalization layer, which is '
            'incompatible with `tff.learning.models.FunctionalModel`. Consider '
            'using group normalization instead.'
        )
    if keras_model.non_trainable_variables:
      raise KerasFunctionalModelError(
          'Received a Keras model with non-trainable variables. Keras models'
          ' with non-trainable variables are currently not supported by'
          ' FunctionalModel. Most training algorithms (e.g. Federated'
          ' Averaging) will not aggregate them, and they are not updated'
          ' locally by the optimizer. We can relax this in the future if we'
          ' have APIs that support updating non-trainable variables.'
      )
  elif not callable(keras_model):
    raise ValueError(
        '`keras_model` must be a `tf.keras.Model` or a no-arg '
        'callable that returns a `tf.keras.Model`.'
    )

  # Clone the keras model inside a graph context so that we only get the
  # variables for the layers (otherwise keras adds other non-user variables). We
  # also setup ops to inject the current model weights, because the cloned model
  # will be re-initialized from scratch.
  with tf.Graph().as_default() as g:
    with variable_utils.record_variable_creation_scope() as captured_variables:
      if isinstance(keras_model, tf.keras.Model):
        try:
          cloned_model = tf.keras.models.clone_model(keras_model)
        except RuntimeError as e:
          raise KerasFunctionalModelError(
              'Encountered a error converting the Keras model. Often this '
              'occurs when the `tf.keras.Model` has a layer that receives '
              'inputs from other layers directly (e.g. shared embeddings).'
              'To avoid the problem, wrap the `tf.keras.Model` construction in '
              'a no-arg callable (e.g. lambda) and pass that callable to '
              '`functional_model_from_keras`'
          ) from e
        if len(cloned_model.variables) != len(keras_model.variables):
          raise KerasFunctionalModelError(
              'The input Keras model is likely sharing variables across layers '
              'which is unsupported. Cloning the model will duplicate these '
              'variables and result in unexpected training gradients.'
          )
      else:
        cloned_model = keras_model()

      # Ensure our cloned model has the same weights as the current model.
      # We'll feed in the current model waits into the palceholders for
      # assignmnet in a session below.
      def assign_placeholder(v):
        p = tf.compat.v1.placeholder(dtype=v.dtype)
        return v.assign(p), p

      assign_ops, placeholders = zip(
          *(assign_placeholder(v) for v in cloned_model.variables)
      )
  trainable_variables = tuple(v for v in captured_variables if v.trainable)
  non_trainable_variables = tuple(
      v for v in captured_variables if not v.trainable
  )

  # Here we get the initial weights from the incoming keras model in the order
  # they are constructed; and also ensure that the values are set to the
  # incoming model weights rather than their fresh initialization.
  if isinstance(keras_model, tf.keras.Model):
    model_for_variables = keras_model
  else:
    model_for_variables = keras_model()
  current_model_weights = tf.nest.map_structure(
      lambda v: v.read_value().numpy(), model_for_variables.variables
  )
  with tf.compat.v1.Session(graph=g) as sess:
    sess.run(tf.compat.v1.initializers.variables(captured_variables))
    sess.run(
        fetches=assign_ops,
        feed_dict=dict(zip(placeholders, current_model_weights)),
    )
    initial_weights = sess.run(
        fetches=(trainable_variables, non_trainable_variables)
    )

  @tf.function
  def predict_on_batch(
      model_weights: ModelWeights, x: Any, training: bool = True
  ) -> Any:
    with tf.init_scope():
      if tf.executing_eagerly():
        raise KerasFunctionalModelError(
            'tf.keras.Model used as a FunctionalModel is only usable inside a '
            'tff.tf_computation decorated callable or a graph context.'
        )
    # Make a copy of the weights container; can't mutate Python containers
    # inside a tf.function.
    trainable, non_trainable = (list(w) for w in model_weights)

    # Here were intercept variable creation requests during the
    # `tf.keras.models.clone_model()` call.
    #
    # Instead of forwarding the variable request to TF core and getting a
    # `tf.Variable` back, we skip that and return only the `tf.Tensor` that
    # corresponds to the `tf.Variable` recreation request (avoiding any variable
    # creation). This works because TF operations that accept `tf.Variable`
    # inputs automatically call `variable.read_value()` and then operate on that
    # resulting tensor. We're relying on shortcutting that and providing the
    # tensor straight away.
    #
    # For example, `tf.matmul` doesn't notice its input is `tf.Variable` or
    # `tf.Tensor`:
    #
    #   v = tf.Variable([[1], [2], [3]])
    #   tf.matmul(v, [[4, 5, 6]])
    #
    #   and
    #
    #   v = tf.constant([[1], [2], [3]])
    #   tf.matmul(v, [[4, 5, 6]])
    #
    #   both result in:
    #
    #   <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    #   array([[ 4,  5,  6],
    #          [ 8, 10, 12],
    #          [12, 15, 18]], dtype=int32)>
    def swap_tensor_parameter_for_variable(_, **kwargs):
      if kwargs.get('trainable', True):
        return trainable.pop(0)
      else:
        return non_trainable.pop(0)

    with tf.variable_creator_scope(swap_tensor_parameter_for_variable):
      if isinstance(keras_model, tf.keras.Model):
        variableless_model = tf.keras.models.clone_model(keras_model)
      else:
        variableless_model = keras_model()
    return variableless_model(x, training)

  @tf.function
  def forward_pass(
      model_weights: ModelWeights, batch_input: Any, training: bool = True
  ) -> variable.BatchOutput:
    if isinstance(batch_input, Mapping):
      x = batch_input['x']
      y = batch_input['y']
    elif isinstance(batch_input, Sequence):
      x, y = batch_input
    else:
      raise ValueError(
          '`batch_input` must be either a mapping with keys `x` '
          f'and `y` or a sequence of `(x, y)`. Got: {batch_input!r}'
      )
    predictions = predict_on_batch(model_weights, x, training)
    batch_loss = loss_fn(y_true=y, y_pred=predictions)

    # TODO(b/207033265): more work needed to support models with multiple loss
    # functions.

    def nrows(t):
      return t.nrows() if isinstance(t, tf.RaggedTensor) else tf.shape(t)[0]

    return variable.BatchOutput(
        loss=batch_loss,
        predictions=predictions,
        num_examples=nrows(tf.nest.flatten(batch_input)[0]),
    )

  if metrics_constructor is not None:
    metrics_fns = keras_utils.create_functional_metric_fns(metrics_constructor)
  else:
    metrics_fns = (
        empty_metrics_state,
        noop_update_metrics,
        noop_finalize_metrics,
    )

  return FunctionalModel(
      initial_weights=initial_weights,
      forward_pass_fn=forward_pass,
      predict_on_batch_fn=predict_on_batch,
      metrics_fns=metrics_fns,
      input_spec=input_spec,
  )


def keras_model_from_functional_weights(
    *, model_weights: ModelWeights, keras_model: tf.keras.Model
) -> tf.keras.Model:
  """Creates a new Keras model using the model weights from a `FunctionalModel`.

  This method is effectively the reverse of `functional_model_from_keras`. Since
  the trained weights are external to the model, this method expects a nested
  structure of tensors that was used to train a `FunctionalModel`

  IMPORTANT: this method must be run in a graph context (e.g. inside a
  `tf.Graph` context, or a `tff.tf_computation` decorated callable), otherwise
  the Keras model construction will differ from how the `FunctionalModel` was
  originally created.

  Args:
    model_weights: A nested structure of tensors matching the structure of
      `tff.learning.models.FunctionalModel.initial_weights` for a model
      constructed using `functional_model_from_keras`.
    keras_model: A Keras model to use for cloning a new model but with the input
      weights.

  Returns:
    A newly constructed `tf.keras.Model` that matches the architecture of
    the input `keras_model` argument but with the weight values from
    `model_weights.
  """
  if tf.compat.v1.executing_eagerly_outside_functions():
    raise ValueError(
        '`keras_model_from_functional_weights()` can only be called from within'
        ' a graph context.'
    )

  # Convert to mutable lists that we can `pop` weights off of.
  trainable_weights, non_trainable_weights = [list(w) for w in model_weights]

  def variable_creator_with_weights(next_creator_fn, **kwargs):
    try:
      if kwargs.get('trainable', True):
        weight = trainable_weights.pop(0)
      else:
        weight = non_trainable_weights.pop(0)
    except IndexError:
      raise ValueError(
          '`model_weights` contains fewer weights than `keras_model` uses, '
          'check that the weights argument is matching the model type.'
      ) from None
    if 'initial_value' in kwargs:
      kwargs['initial_value'] = weight
    elif 'initializer' in kwargs:
      kwargs['initializer'] = tf.compat.v1.constant_initializer(value=weight)
    else:
      raise ValueError(
          "Can't set weights to a Keras model that creates variables without "
          'initial values.'
      )
    return next_creator_fn(**kwargs)

  with tf.variable_creator_scope(variable_creator_with_weights):
    new_model = tf.keras.models.clone_model(keras_model)
  if trainable_weights or non_trainable_weights:
    raise ValueError(
        '`model_weights` contained more variables than `keras_model` uses, '
        'check that the weights argument is matching the model type.'
    )
  return new_model
