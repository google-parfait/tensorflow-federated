# Lint as: python3
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
"""An example of personalization strategy.

A personalization strategy is represented as a `tf.function` called
`personalze_fn` below: it accepts a `tff.learning.Model` (with weights already
initialized to the initial model weights when users invoke the tff.Computation),
an unbatched `tf.data.Dataset` for train, an unbatched `tf.data.Dataset` for
test, and an arbitrary `context` object (which is used to hold any extra
information that a personalization strategy may use), trains a personalized
model, and returns the evaluation metrics.

The example below trains a personalized model for `max_num_epochs` epochs, and
evaluates the model every `num_epochs_per_eval` epoch, and records the metrics.
The final evaluation metrics are represented as a nested `OrderedDict` of string
metric names to scalar `tf.Tensor`s.
"""

import collections

import tensorflow as tf


def build_personalize_fn(optimizer_fn,
                         train_batch_size,
                         max_num_epochs,
                         num_epochs_per_eval,
                         test_batch_size,
                         shuffle=True,
                         shuffle_buffer_size=1000):
  """Example of a builder function that constructs a `personalize_fn`.

  The returned function represents the optimization algorithm to run on each
  client in order to personalize a model for those clients.

  Args:
    optimizer_fn: A no-argument function that returns a
      `tf.keras.optimizers.Optimizer`.
    train_batch_size: An `int` specifying the batch size used in training the
      personalized model.
    max_num_epochs: An `int` specifying the maximum number of epochs used in
      training a personalized model.
    num_epochs_per_eval: An `int` specifying the frequency that a personalized
      model gets evaluated during the process of training.
    test_batch_size: An `int` specifying the batch size used in evaluation.
    shuffle: A `bool` specifying whether to shuffle train data in every epoch.
    shuffle_buffer_size: An `int` specifying the buffer size used in shuffling
      the train data when `shuffle=True`.

  Returns:
    A `tf.function` that trains a personalized model, evaluates the model every
    `num_epochs_per_eval` epochs, and returns the evaluation metrics.
  """
  # Create the `optimizer` here instead of inside the `tf.function` below,
  # because a `tf.function` generally does not allow creating new variables.
  optimizer = optimizer_fn()

  @tf.function
  def personalize_fn(model, train_data, test_data, context=None):
    """A personalization strategy that trains a model and returns the metrics.

    Args:
      model: A `tff.learning.Model`.
      train_data: An unbatched `tf.data.Dataset` used for training.
      test_data: An unbatched `tf.data.Dataset` used for evaluation.
      context: An optional object (e.g., extra dataset) used in personalization.

    Returns:
      An `OrderedDict` that maps a metric name to `tf.Tensor`s containing the
      evaluation metrics.
    """
    del context  # This example does not use extra context.

    def train_one_batch(state, batch):
      """Run gradient descent on a batch."""
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)

      grads = tape.gradient(output.loss, model.trainable_variables)
      optimizer.apply_gradients(
          zip(
              tf.nest.flatten(grads),
              tf.nest.flatten(model.trainable_variables)))
      # Update the number of examples and the number of batches.
      next_state = (state[0] + output.num_examples, state[1] + 1)
      return next_state

    def train_several_epochs(num_epochs, state):
      """Train the model for several epochs on `train_data`."""
      data = train_data.repeat(num_epochs)
      if shuffle:
        data = data.shuffle(shuffle_buffer_size)
      data = data.batch(train_batch_size)
      return data.reduce(initial_state=state, reduce_func=train_one_batch)

    # Start training.
    training_state = (0, 0)  # (number of examples, number of batches)

    # Compute the number of times that the model gets evaluated during training.
    num_evals, remainder_epochs = divmod(max_num_epochs, num_epochs_per_eval)

    # Create a nested structure of `tf.TensorArray`s that has the same nested
    # strucutre of the evaluation metrics returned by the `evaluate_fn`. In this
    # case, the nested strucutre is an `OrderedDict` that maps names to values.
    metrics_tensorarrays = tf.nest.map_structure(
        lambda v: tf.TensorArray(v.dtype, size=num_evals),
        evaluate_fn(model, test_data, test_batch_size))

    for i in tf.range(num_evals):
      training_state = train_several_epochs(num_epochs_per_eval, training_state)
      # Evaluate the current trained model.
      current_metrics = evaluate_fn(model, test_data, test_batch_size)
      # Write the current result to the corresponding `tf.TensorArray`s.
      metrics_tensorarrays = tf.nest.map_structure(
          lambda ta, v: ta.write(i, v),  # pylint: disable=cell-var-from-loop
          metrics_tensorarrays,
          current_metrics)

    # Finish the remaining training epochs.
    training_state = train_several_epochs(remainder_epochs, training_state)

    # Convert the nested `tf.TensorArray`s to the desired structure: here we
    # create a dict by grouping the metric values at the same epoch together.
    metrics_dict = collections.OrderedDict()
    for j in range(num_evals):
      epoch_idx = (j + 1) * num_epochs_per_eval
      metrics_dict[f'epoch_{epoch_idx}'] = tf.nest.map_structure(
          lambda ta: ta.read(j),  # pylint: disable=cell-var-from-loop
          metrics_tensorarrays)

    # Save the training statistics.
    metrics_dict['num_examples'] = training_state[0]
    metrics_dict['num_batches'] = training_state[1]

    # Evaluate the final model.
    metrics_dict['final_model'] = evaluate_fn(model, test_data, test_batch_size)

    return metrics_dict

  return personalize_fn


@tf.function
def evaluate_fn(model, dataset, batch_size):
  """Evaluate a model on the given dataset.

  Note: The returned metrics are defined in `model.report_local_outputs`, which
  can be specified by the `metrics` argument when using
  `tff.learning.from_keras_model` to build the input `tff.learning.Model`. In
  addition to passing the `metrics` argument to `tff.learning.from_keras_model`,
  users can always define extra metrics they want to evaluate inside this
  `evaluate_fn` function.

  Args:
    model: A `tff.learning.Model`.
    dataset: An unbatched `tf.data.Dataset`.
    batch_size: An `int` specifying the batch size used in evaluation.

  Returns:
    An `OrderedDict` of metric names to scalar `tf.Tensor`s containing the
    evaluation metrics defined in `model.report_local_outputs`.
  """
  # Reset the model's local variables. This is necessary because
  # `model.report_local_outputs()` aggregates the metrics from *all* previous
  # calls to `forward_pass` (which include the metrics computed in training).
  # Resetting ensures that the returned metrics are computed on test data.
  # Similar to the `reset_states` method of `tf.metrics.Metric`.
  for var in model.local_variables:
    if var.initial_value is not None:
      var.assign(var.initial_value)
    else:
      var.assign(tf.zeros_like(var))

  def reduce_fn(dummy_input, batch):
    model.forward_pass(batch, training=False)
    return dummy_input

  batched_dataset = dataset.batch(batch_size)
  # Running `reduce_fn` over the input dataset. The aggregated metrics can be
  # accessed via `model.report_local_outputs()`.
  batched_dataset.reduce(initial_state=tf.constant(0), reduce_func=reduce_fn)

  results = collections.OrderedDict()
  local_outputs = model.report_local_outputs()
  for name, metric in local_outputs.items():
    if isinstance(metric, list) and (len(metric) == 2):
      # Some metrics returned by `report_local_outputs()` can have two scalars:
      # one represents `sum`, and the other represents `count`. Ideally we want
      # to return a single scalar for each metric.
      results[name] = metric[0] / metric[1]
    else:
      results[name] = metric[0] if isinstance(metric, list) else metric
  return results
