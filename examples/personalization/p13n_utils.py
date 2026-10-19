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
"""An example of personalization strategy."""

import collections
from collections.abc import Callable
from typing import Any, Optional, Union

import federated_language
import keras
import tensorflow as tf
import tensorflow_federated as tff

_OptimizerFn = Union[
    tff.learning.optimizers.Optimizer,
    Callable[
        [],
        Union[
            tff.learning.optimizers.Optimizer,
            tf.keras.optimizers.Optimizer,
            keras.optimizers.Optimizer,
        ],
    ],
]
_PersonalizeFn = Callable[
    ...,
    collections.OrderedDict[str, tf.Tensor],
]
_EVAL_BATCH_SIZE = 1  # Batch size used when evaluating a dataset.
_SHUFFLE_BUFFER_SIZE = 1000  # Buffer size used when shuffling a dataset.


def build_personalize_fn(
    optimizer_fn: _OptimizerFn,
    batch_size: int,
    num_epochs: int,
    num_epochs_per_eval: int,
    shuffle: bool = True,
) -> _PersonalizeFn:
  """Builds a `tf.function` that represents a personalization strategy.

  The returned `tf.function` represents the optimization algorithm to run on
  a client in order to personalize a given model. It takes a
  `tff.learning.models.FunctionalModel`, initial `model_weights`, an unbatched
  training dataset, an unbatched test dataset, and an optional `context`
  (e.g., extra datasets) as input, trains a personalized model on the training
  dataset for `num_epochs`, evaluates the model on the test dataset every
  `num_epochs_per_eval`, and returns the evaluation metrics. The evaluation
  metrics are computed by `evaluate_fn` (see its docstring for more details).

  This builder function only serves as an example. Customers are allowed to
  write any personalization strategy as long as it satisfies the function
  signature specified by `_PERSONALIZE_FN_TYPE`.

  Args:
    optimizer_fn: A `tff.learning.optimizers.Optimizer` or a no-argument
      function that returns an optimizer.
    batch_size: An `int` specifying the batch size used in training.
    num_epochs: An `int` specifying the number of epochs used in training a
      personalized model.
    num_epochs_per_eval: An `int` specifying the frequency that a personalized
      model gets evaluated during the process of training.
    shuffle: A `bool` specifying whether to shuffle train data in every epoch.

  Returns:
    A `tf.function` that trains a personalized model, evaluates the model every
    `num_epochs_per_eval` epochs, and returns the evaluation metrics.
  """
  optimizer = optimizer_fn() if callable(optimizer_fn) else optimizer_fn
  if not isinstance(optimizer, tff.learning.optimizers.Optimizer):
    lr = float(optimizer.learning_rate)
    if isinstance(optimizer, (tf.keras.optimizers.Adam, keras.optimizers.Adam)):
      optimizer = tff.learning.optimizers.build_adam(learning_rate=lr)
    else:
      optimizer = tff.learning.optimizers.build_sgdm(learning_rate=lr)

  @tf.function
  def personalize_fn(
      model: tff.learning.models.FunctionalModel,
      model_weights: Optional[Any] = None,
      train_data: Optional[tf.data.Dataset] = None,
      test_data: Optional[tf.data.Dataset] = None,
      context: Optional[Any] = None,
  ) -> collections.OrderedDict[str, Any]:
    """A personalization strategy that trains a model and returns the metrics.

    Args:
      model: A `tff.learning.models.FunctionalModel`.
      model_weights: A `(trainable_weights, non_trainable_weights)` tuple. If
        omitted, defaults to `model.initial_weights`.
      train_data: An unbatched `tf.data.Dataset` used for training.
      test_data: An unbatched `tf.data.Dataset` used for evaluation.
      context: An optional object (e.g., extra dataset) used in personalization.

    Returns:
      An `OrderedDict` that maps metric names to `tf.Tensor`s or structures of
      `tf.Tensor`s containing the training and evaluation metrics.
    """
    del context  # This example does not use extra context.
    if isinstance(model_weights, tf.data.Dataset):
      # Called positionally as `(model, train_data, test_data)`.
      test_data = train_data
      train_data = model_weights
      model_weights = None
    assert train_data is not None
    assert test_data is not None
    if model_weights is None:
      trainable_weights = tuple(
          tf.convert_to_tensor(w) for w in model.initial_weights[0]
      )
      non_trainable_weights = tuple(
          tf.convert_to_tensor(w) for w in model.initial_weights[1]
      )
    else:
      trainable_weights, non_trainable_weights = model_weights
      trainable_weights = tuple(
          tf.convert_to_tensor(w) for w in trainable_weights
      )
      non_trainable_weights = tuple(
          tf.convert_to_tensor(w) for w in non_trainable_weights
      )

    opt_state = optimizer.initialize(
        tf.nest.map_structure(
            lambda t: federated_language.TensorType(
                t.dtype.as_numpy_dtype, t.shape
            ),
            trainable_weights,
        )
    )

    def train_one_batch(state, batch):
      """Run gradient descent on a batch."""
      num_examples_sum, current_trainable_weights, current_opt_state = state
      if isinstance(batch, collections.abc.Mapping):
        x = batch['x']
        y = batch['y']
      else:
        x, y = batch
      with tf.GradientTape() as tape:
        tape.watch(current_trainable_weights)
        predictions = model.predict_on_batch(
            model_weights=(current_trainable_weights, non_trainable_weights),
            x=x,
            training=True,
        )
        loss = model.loss(output=predictions, label=y)
      grads = tape.gradient(loss, current_trainable_weights)
      next_opt_state, next_trainable_weights = optimizer.next(
          current_opt_state, current_trainable_weights, grads
      )
      batch_num_examples = tf.shape(tf.nest.flatten(x)[0])[0]
      return (
          num_examples_sum + batch_num_examples,
          next_trainable_weights,
          next_opt_state,
      )

    # Start training.
    training_state = (tf.constant(0), trainable_weights, opt_state)
    metrics_dict = collections.OrderedDict()
    for epoch_idx in range(1, num_epochs + 1):
      if shuffle:
        train_data = train_data.shuffle(_SHUFFLE_BUFFER_SIZE)
      training_state = train_data.batch(batch_size).reduce(
          initial_state=training_state, reduce_func=train_one_batch
      )
      # Evaluate the trained model every `num_epochs_per_eval` epochs.
      if (epoch_idx % num_epochs_per_eval == 0) or (epoch_idx == num_epochs):
        current_weights = (training_state[1], non_trainable_weights)
        metrics_dict[f'epoch_{epoch_idx}'] = evaluate_fn(
            model, current_weights, test_data
        )

    # Save the training statistics.
    metrics_dict['num_train_examples'] = training_state[0]

    return metrics_dict

  return personalize_fn


@tf.function
def evaluate_fn(
    model: tff.learning.models.FunctionalModel,
    model_weights: Optional[Any] = None,
    dataset: Optional[tf.data.Dataset] = None,
) -> collections.OrderedDict[str, tf.Tensor]:
  """Evaluates a model on the given dataset.

  The returned metrics include those given by `model.finalize_metrics`. The
  returned metrics also contain an integer metric with name 'num_test_examples'.

  Args:
    model: A `tff.learning.models.FunctionalModel` created by
      `tff.learning.models.functional_model_from_keras`.
    model_weights: A `(trainable_weights, non_trainable_weights)` tuple. If
      omitted, defaults to `model.initial_weights`.
    dataset: An unbatched `tf.data.Dataset`.

  Returns:
    An `OrderedDict` of metric names to scalar `tf.Tensor`s.
  """
  if dataset is None and isinstance(model_weights, tf.data.Dataset):
    dataset = model_weights
    model_weights = None
  assert dataset is not None
  if model_weights is None:
    model_weights = (
        tuple(tf.convert_to_tensor(w) for w in model.initial_weights[0]),
        tuple(tf.convert_to_tensor(w) for w in model.initial_weights[1]),
    )

  def reduce_fn(state, batch):
    num_examples_sum, metrics_state = state
    if isinstance(batch, collections.abc.Mapping):
      x = batch['x']
      y = batch['y']
    else:
      x, y = batch
    predictions = model.predict_on_batch(
        model_weights=model_weights, x=x, training=False
    )
    loss = model.loss(output=predictions, label=y)
    batch_num_examples = tf.shape(tf.nest.flatten(x)[0])[0]
    batch_output = tff.learning.models.BatchOutput(
        loss=loss, predictions=predictions, num_examples=batch_num_examples
    )
    metrics_state = model.update_metrics_state(
        metrics_state, batch_output=batch_output, labels=y
    )
    return num_examples_sum + batch_num_examples, metrics_state

  num_examples_sum, unfinalized_metrics = dataset.batch(
      _EVAL_BATCH_SIZE
  ).reduce(
      initial_state=(tf.constant(0), model.initialize_metrics_state()),
      reduce_func=reduce_fn,
  )
  eval_metrics = collections.OrderedDict()
  eval_metrics['num_test_examples'] = num_examples_sum
  eval_metrics.update(model.finalize_metrics(unfinalized_metrics))
  return eval_metrics
