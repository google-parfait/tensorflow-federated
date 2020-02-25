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
"""An implementation of federated personalization evaluation."""

import collections

import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model_utils


def build_personalization_eval(model_fn,
                               personalize_fn_dict,
                               baseline_evaluate_fn,
                               max_num_samples=100,
                               context_tff_type=None):
  """Builds the TFF computation for evaluating personalization strategies.

  The returned TFF computation broadcasts model weights from SERVER to CLIENTS.
  Each client evaluates the personalization strategies given in
  `personalize_fn_dict`. Evaluation metrics from at most `max_num_samples`
  participating clients are collected to the SERVER.

  NOTE: The functions in `personalize_fn_dict` and `baseline_evaluate_fn` are
  expected to take as input *unbatched* datasets, and are responsible for
  applying batching, if any, to the provided input datasets.

  Args:
    model_fn: A no-argument function that returns a `tff.learning.Model`.
    personalize_fn_dict: An `OrderedDict` that maps a `string` (representing a
      strategy name) to a no-argument function that returns a `tf.function`.
      Each `tf.function` represents a personalization strategy: it accepts a
      `tff.learning.Model` (with weights already initialized to the given model
      weights when users invoke the returned TFF computation), an unbatched
      `tf.data.Dataset` for train, an unbatched `tf.data.Dataset` for test, and
      an arbitrary context object (which is used to hold any extra information
      that a personalization strategy may use), trains a personalized model, and
      returns the evaluation metrics. The evaluation metrics are represented as
      an `OrderedDict` (or a nested `OrderedDict`) of `string` metric names to
      scalar `tf.Tensor`s.
    baseline_evaluate_fn: A `tf.function` that accepts a `tff.learning.Model`
      (with weights already initialized to the provided model weights when users
      invoke the returned TFF computation), and an unbatched `tf.data.Dataset`,
      evaluates the model on the dataset, and returns the evaluation metrics.
      The evaluation metrics are represented as an `OrderedDict` (or a nested
      `OrderedDict`) of `string` metric names to scalar `tf.Tensor`s. This
      function is *only* used to compute the baseline metrics of the initial
      model.
    max_num_samples: A positive `int` specifying the maximum number of metric
      samples to collect in a round. Each sample contains the personalization
      metrics from a single client. If the number of participating clients in a
      round is smaller than this value, all clients' metrics are collected.
    context_tff_type: A `tff.Type` of the optional context object used by the
      personalization strategies defined in `personalization_fn_dict`. We use a
      context object to hold any extra information (in addition to the training
      dataset) that personalization may use. If context is used in
      `personalization_fn_dict`, its `tff.Type` must be provided here.

  Returns:
    A federated `tff.Computation` that maps
    <model_weights@SERVER, input@CLIENTS> -> personalization_metrics@SERVER,
    where:
    - model_weights is a `tff.learning.framework.ModelWeights`.
    - each client's input is an `OrderedDict` of at least two keys `train_data`
      and `test_data`; each key is mapped to an unbatched `tf.data.Dataset`. If
      extra context is used in `personalize_fn_dict`, then client input has a
      third key `context` that is mapped to a object whose `tff.Type` is
      provided by the `context_tff_type` argument.
    - personazliation_metrics is an `OrderedDict` that maps a key
      'baseline_metrics' to the evaluation metrics of the initial model
      (computed by `baseline_evaluate_fn`), and maps keys (strategy names) in
      `personalize_fn_dict` to the evaluation metrics of the corresponding
      personalization strategies.
    - Note: only metrics from at most `max_num_samples` participating clients
      are collected to the SERVER. All collected metrics are stored in a
      single `OrderedDict` (the personalization_metrics shown above), where each
      metric is mapped to a list of scalars (each scalar comes from one client).
      Metric values at the same position, e.g., metric_1[i], metric_2[i]..., all
      come from the same client.

  Raises:
    TypeError: If arguments are of the wrong types.
    ValueError: If `baseline_metrics` is used as a key in `personalize_fn_dict`.
    ValueError: If `max_num_samples` is not positive.
  """
  # Obtain the types by constructing the model first.
  # TODO(b/124477628): Replace it with other ways of handling metadata.
  with tf.Graph().as_default():
    py_typecheck.check_callable(model_fn)
    model = model_utils.enhance(model_fn())
    model_weights_type = tff.framework.type_from_tensors(model.weights)
    batch_type = model.input_spec

  # Define the `tff.Type` of each client's input. Since batching (as well as
  # other preprocessing of datasets) is done within each personalization
  # strategy (i.e., by functions in `personalize_fn_dict`), the client-side
  # input should contain unbatched elements.
  element_type = _remove_batch_dim(batch_type)
  client_input_type = collections.OrderedDict([
      ('train_data', tff.SequenceType(element_type)),
      ('test_data', tff.SequenceType(element_type))
  ])
  if context_tff_type is not None:
    py_typecheck.check_type(context_tff_type, tff.Type)
    client_input_type['context'] = context_tff_type
  client_input_type = tff.to_type(client_input_type)

  @tff.tf_computation(model_weights_type, client_input_type)
  def _client_computation(initial_model_weights, client_input):
    """TFF computation that runs on each client."""
    model = model_fn()
    train_data = client_input['train_data']
    test_data = client_input['test_data']
    context = client_input.get('context', None)
    return _client_fn(model, initial_model_weights, train_data, test_data,
                      personalize_fn_dict, baseline_evaluate_fn, context)

  py_typecheck.check_type(max_num_samples, int)
  if max_num_samples <= 0:
    raise ValueError('max_num_samples must be a positive integer.')

  @tff.federated_computation(
      tff.FederatedType(model_weights_type, tff.SERVER),
      tff.FederatedType(client_input_type, tff.CLIENTS))
  def personalization_eval(server_model_weights, federated_client_input):
    """TFF orchestration logic."""
    client_init_weights = tff.federated_broadcast(server_model_weights)
    client_final_metrics = tff.federated_map(
        _client_computation, (client_init_weights, federated_client_input))

    # WARNING: Collecting information from clients can be risky. Users have to
    # make sure that it is proper to collect those metrics from clients.
    # TODO(b/147889283): Add a link to the TFF doc once it exists.
    results = tff.utils.federated_sample(client_final_metrics, max_num_samples)
    return results

  return personalization_eval


def _remove_batch_dim(spec):
  """Creates a nested `tf.TensorSpec` by removing the batch dimension of `spec`.

  Args:
    spec: A `tf.TensorSpec` or a nested `tf.TensorSpec`, with the first
      dimension being the batch dimension.

  Returns:
    A `tf.TensorSpec` or a nested `tf.TensorSpec` that has the same structure of
    the input `spec` object, with the batch dimension removed.

  Raises:
    TypeError: If the argument has the wrong type.
    ValueError: If the `tf.TensorSpec` does not have the first dimension.
  """

  def _remove_first_dim_for_tensorspec(ts):
    """Return a new `tf.TensorSpec` after removing the first dimension."""
    py_typecheck.check_type(ts, tf.TensorSpec)
    if (ts.shape.rank is not None) and (ts.shape.rank >= 1):
      return tf.TensorSpec(shape=ts.shape[1:], dtype=ts.dtype, name=ts.name)
    else:
      raise ValueError('Provided shape must have rank 1 or higher.')

  return tf.nest.map_structure(_remove_first_dim_for_tensorspec, spec)


@tf.function
def _client_fn(model,
               initial_model_weights,
               train_data,
               test_data,
               personalize_fn_dict,
               baseline_evaluate_fn,
               context=None):
  """The main `tf.function` that runs on device.

  This function first evalautes the initial model and gets the baseline metrics.
  Then starting from the same initial model, this function iterates over the
  personalization strategies defined in `personalize_fn_dict`, trains and
  evaluates the personalized models, and returns the evaluation metrics.

  Args:
    model: A `tff.learning.Model`.
    initial_model_weights: A `tff.learning.framework.ModelWeights` containing
      `tf.Tensor`s that hold trainable and non-trainable weights.
    train_data: A `tf.data.Dataset` used for training.
    test_data: A `tf.data.Dataset` used for evaluation.
    personalize_fn_dict: This is the same argument specified in the function
      `build_personalization_eval` above; see its documentation for details.
    baseline_evaluate_fn: This is the same argument specified in the function
      `build_personalization_eval` above; see its documentation for details.
    context: An optional object used in `personalize_fn_dict`. If used, its
      `tff.Type` must be provided by passing the correct `context_tff_type`
      argument to the `build_personalization_eval` function.

  Returns:
    An `OrderedDict` that maps a string 'baseline_metrics' to the evaluation
    metrics of the initial model (computed by `baseline_evaluate_fn`), and maps
    keys (strategy names) in `personalize_fn_dict` to the evaluation metrics of
    the corresponding personalization strategies.

  Raises:
    TypeError: If arguments are of the wrong types.
    ValueError: If `baseline_metrics` is used as a key in `personalize_fn_dict`.
  """
  # Wrap the input model as an `EnhancedModel` for easy access of its weights.
  model = model_utils.enhance(model)

  final_metrics = collections.OrderedDict()
  tff.utils.assign(model.weights, initial_model_weights)
  py_typecheck.check_callable(baseline_evaluate_fn)
  final_metrics['baseline_metrics'] = baseline_evaluate_fn(model, test_data)

  py_typecheck.check_type(personalize_fn_dict, collections.OrderedDict)
  if 'baseline_metrics' in personalize_fn_dict:
    raise ValueError('baseline_metrics should not be used as a key in '
                     'personalize_fn_dict.')

  for name, personalize_fn_builder in personalize_fn_dict.items():
    py_typecheck.check_type(name, str)
    tff.utils.assign(model.weights, initial_model_weights)

    # Construct the `personalize_fn` (and the associated `tf.Variable`s) here.
    # Once `_client_fn` is decorated with `tff.tf_computation`, construction of
    # the new variables will happen in a scope controlled by TFF. Ensuring
    # `tf.Variable`s are created in the graphs that TFF controls is the reason
    # we need `personalize_fn_dict` to contain no-argument functions that build
    # the desired `tf.function`s, rather than already built `tf.function`s.
    py_typecheck.check_callable(personalize_fn_builder)
    personalize_fn = personalize_fn_builder()

    py_typecheck.check_callable(personalize_fn)
    final_metrics[name] = personalize_fn(model, train_data, test_data, context)

  return final_metrics
