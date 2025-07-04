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
from collections.abc import Callable, Mapping
from typing import Any

import federated_language
import tensorflow as tf

from tensorflow_federated.python.aggregators import sampling
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.models import variable


_MetricsType = Mapping[str, Any]
_SplitFnType = Callable[
    [tf.data.Dataset], tuple[tf.data.Dataset, tf.data.Dataset]
]
_FinetuneEvalFnType = Callable[
    [variable.VariableModel, tf.data.Dataset, tf.data.Dataset],
    _MetricsType,
]

_TRAIN_DATA_KEY = 'train_data'
_EVAL_DATA_KEY = 'test_data'


def build_personalization_eval_computation(
    model_fn: Callable[[], variable.VariableModel],
    personalize_fn_dict: Mapping[str, Callable[[], _FinetuneEvalFnType]],
    baseline_evaluate_fn: Callable[
        [variable.VariableModel, tf.data.Dataset], _MetricsType
    ],
    max_num_clients: int = 100,
) -> federated_language.framework.Computation:
  """Builds the TFF computation for evaluating personalization strategies.

  The returned TFF computation broadcasts model weights from
  `federated_language.SERVER` to
  `federated_language.CLIENTS`. Each client evaluates the personalization
  strategies given in
  `personalize_fn_dict`. Evaluation metrics from at most `max_num_clients`
  participating clients are collected to the server.

  NOTE: The functions in `personalize_fn_dict` and `baseline_evaluate_fn` are
  expected to take as input *unbatched* datasets, and are responsible for
  applying batching, if any, to the provided input datasets.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`. This method must *not* capture
      TensorFlow tensors or variables and use them. The model must be
      constructed entirely from scratch on each invocation, returning the same
      pre-constructed model each call will result in an error.
    personalize_fn_dict: An `OrderedDict` that maps a `string` (representing a
      strategy name) to a no-argument function that returns a `tf.function`.
      Each `tf.function` represents a personalization strategy - it accepts a
      `tff.learning.models.VariableModel` (with weights already initialized to
      the given model weights when users invoke the returned TFF computation),
      an unbatched `tf.data.Dataset` for train, and an unbatched
      `tf.data.Dataset` for test, trains a personalized model, and returns the
      evaluation metrics. The evaluation metrics are represented as an
      `OrderedDict` (or a nested `OrderedDict`) of `string` metric names to
      scalar `tf.Tensor`s.
    baseline_evaluate_fn: A `tf.function` that accepts a
      `tff.learning.models.VariableModel` (with weights already initialized to
      the provided model weights when users invoke the returned TFF
      computation), and an unbatched `tf.data.Dataset`, evaluates the model on
      the dataset, and returns the evaluation metrics. The evaluation metrics
      are represented as an `OrderedDict` (or a nested `OrderedDict`) of
      `string` metric names to scalar `tf.Tensor`s. This function is *only* used
      to compute the baseline metrics of the initial model.
    max_num_clients: A positive `int` specifying the maximum number of clients
      to collect metrics in a round (default is 100). The clients are sampled
      without replacement. For each sampled client, all the personalization
      metrics from this client will be collected. If the number of participating
      clients in a round is smaller than this value, then metrics from all
      clients will be collected.

  Returns:
    A federated `federated_language.Computation` with the functional type
    signature
    `(<model_weights@SERVER, input@CLIENTS> -> personalization_metrics@SERVER)`:

    *   `model_weights` is a `tff.learning.models.ModelWeights`.
    *   Each client's input is an `OrderedDict` of two required keys
        `train_data` and `test_data`; each key is mapped to an unbatched
        `tf.data.Dataset`.
    *   `personalization_metrics` is an `OrderedDict` that maps a key
        'baseline_metrics' to the evaluation metrics of the initial model
        (computed by `baseline_evaluate_fn`), and maps keys (strategy names) in
        `personalize_fn_dict` to the evaluation metrics of the corresponding
        personalization strategies.
    *   Note: only metrics from at most `max_num_clients` participating clients
        (sampled without replacement) are collected to the SERVER. All collected
        metrics are stored in a single `OrderedDict` (`personalization_metrics`
        shown above), where each metric is mapped to a list of scalars (each
        scalar comes from one client). Metric values at the same position, e.g.,
        metric_1[i], metric_2[i]..., all come from the same client.

  Raises:
    TypeError: If arguments are of the wrong types.
    ValueError: If `baseline_metrics` is used as a key in `personalize_fn_dict`.
    ValueError: If `max_num_clients` is not positive.
  """
  # Obtain the types by constructing the model first.
  # TODO: b/124477628 - Replace it with other ways of handling metadata.
  with tf.Graph().as_default():
    model = model_fn()
    model_weights_type = model_weights_lib.weights_type_from_model(model)
    batch_tff_type = tensorflow_types.to_type(model.input_spec)

  # Define the `federated_language.Type` of each client's input. Since batching
  # (as well as other preprocessing of datasets) is done within each
  # personalization strategy (i.e., by functions in `personalize_fn_dict`), the
  # client-side input should contain unbatched elements.
  element_tff_type = _remove_batch_dim(batch_tff_type)
  client_input_type = collections.OrderedDict(
      train_data=federated_language.SequenceType(element_tff_type),
      test_data=federated_language.SequenceType(element_tff_type),
  )
  client_input_type = federated_language.to_type(client_input_type)

  py_typecheck.check_type(max_num_clients, int)
  if max_num_clients <= 0:
    raise ValueError('max_num_clients must be a positive integer.')

  client_computation = _build_client_computation(
      model_weights_type=model_weights_type,
      client_data_type=client_input_type,
      model_fn=model_fn,
      personalize_fn_dict=personalize_fn_dict,
      baseline_evaluate_fn=baseline_evaluate_fn,
  )

  reservoir_sampling_factory = sampling.UnweightedReservoirSamplingFactory(
      sample_size=max_num_clients
  )
  aggregation_process = reservoir_sampling_factory.create(
      client_computation.type_signature.result
  )

  @federated_language.federated_computation(
      federated_language.FederatedType(
          model_weights_type, federated_language.SERVER
      ),
      federated_language.FederatedType(
          client_input_type, federated_language.CLIENTS
      ),
  )
  def personalization_eval(server_model_weights, federated_client_input):
    """TFF orchestration logic."""
    client_init_weights = federated_language.federated_broadcast(
        server_model_weights
    )
    client_final_metrics = federated_language.federated_map(
        client_computation, (client_init_weights, federated_client_input)
    )

    # WARNING: Collecting information from clients can be risky. Users have to
    # make sure that it is proper to collect those metrics from clients.
    # TODO: b/147889283 - Add a link to the TFF doc once it exists.
    sampling_output = aggregation_process.next(
        aggregation_process.initialize(), client_final_metrics  # No state.
    )
    # In the future we may want to output `sampling_output.measurements` also
    # but currently it is empty.
    return sampling_output.result

  return personalization_eval


def _build_client_computation(
    model_weights_type: federated_language.Type,
    client_data_type: federated_language.Type,
    model_fn: Callable[[], variable.VariableModel],
    personalize_fn_dict: Mapping[str, Callable[[], _FinetuneEvalFnType]],
    baseline_evaluate_fn: Callable[
        [variable.VariableModel, tf.data.Dataset], _MetricsType
    ],
):
  """Return an unplaced federated_language.Computation representing the client computation."""

  py_typecheck.check_type(personalize_fn_dict, collections.OrderedDict)
  if 'baseline_metrics' in personalize_fn_dict:
    raise ValueError(
        'baseline_metrics should not be used as a key in personalize_fn_dict.'
    )

  @tensorflow_computation.tf_computation(model_weights_type, client_data_type)
  def _client_computation(initial_model_weights, client_input):
    """A computation performing personalization evaluation on a client."""
    train_data = client_input[_TRAIN_DATA_KEY]
    test_data = client_input[_EVAL_DATA_KEY]

    final_metrics = collections.OrderedDict()
    # Compute the evaluation metrics of the initial model.
    final_metrics['baseline_metrics'] = _compute_baseline_metrics(
        model_fn, initial_model_weights, test_data, baseline_evaluate_fn
    )

    # Compute the evaluation metrics of the personalized models. The returned
    # `p13n_metrics` is an `OrderedDict` that maps keys (strategy names) in
    # `personalize_fn_dict` to the evaluation metrics of the corresponding
    # personalization strategies.
    p13n_metrics = _compute_p13n_metrics(
        model_fn,
        initial_model_weights,
        train_data,
        test_data,
        personalize_fn_dict,
    )
    final_metrics.update(p13n_metrics)
    return final_metrics

  return _client_computation


def _remove_batch_dim(
    type_spec: federated_language.Type,
) -> federated_language.Type:
  """Removes the batch dimension from the `federated_language.TensorType`s in `type_spec`.

  Args:
    type_spec: A `federated_language.Type` containing
      `federated_language.TensorType`s as leaves. The first dimension in the
      leaf `federated_language.TensorType` is the batch dimension.

  Returns:
    A `federated_language.Type` of the same structure as `type_spec`, with no
    batch dimensions
    in all the leaf `federated_language.TensorType`s.

  Raises:
    TypeError: If the argument has the wrong type.
    ValueError: If the `federated_language.TensorType` does not have the first
    dimension.
  """

  def _remove_first_dim_in_tensortype(tensor_type):
    """Return a new `federated_language.TensorType` after removing the first dimension."""
    py_typecheck.check_type(tensor_type, federated_language.TensorType)
    if (
        tensor_type.shape is not None
        and not federated_language.array_shape_is_scalar(tensor_type.shape)
    ):
      return federated_language.TensorType(
          shape=tensor_type.shape[1:], dtype=tensor_type.dtype
      )
    else:
      raise ValueError('Provided shape must have rank 1 or higher.')

  return structure._map_structure(  # pylint: disable=protected-access
      _remove_first_dim_in_tensortype,
      type_spec,  # pytype: disable=wrong-arg-types
  )


def _compute_baseline_metrics(
    model_fn,
    initial_model_weights,
    test_data,
    baseline_evaluate_fn: Callable[..., object],
):
  """Evaluate the model with weights being the `initial_model_weights`."""
  model = model_fn()
  model_weights = model_weights_lib.ModelWeights.from_model(model)

  @tf.function
  def assign_and_compute():
    tf.nest.map_structure(
        lambda v, t: v.assign(t), model_weights, initial_model_weights
    )
    return baseline_evaluate_fn(model, test_data)

  return assign_and_compute()


def _compute_p13n_metrics(
    model_fn,
    initial_model_weights,
    train_data,
    test_data,
    personalize_fn_dict: Mapping[str, Callable[[], Callable[..., object]]],
):
  """Train and evaluate the personalized models."""
  model = model_fn()
  model_weights = model_weights_lib.ModelWeights.from_model(model)
  # Construct the `personalize_fn` (and the associated `tf.Variable`s) here.
  # This ensures that the new variables are created in the graphs that TFF
  # controls. This is the key reason why we need `personalize_fn_dict` to
  # contain no-argument functions that build the desired `tf.function`s, rather
  # than already built `tf.function`s. Note that this has to be done outside the
  # `tf.function` `loop_and_compute` below, because `tf.function` usually does
  # not allow creation of new variables.
  personalize_fns: dict[str, Callable[..., object]] = collections.OrderedDict()
  for name, personalize_fn_builder in personalize_fn_dict.items():
    py_typecheck.check_type(name, str)
    personalize_fns[name] = personalize_fn_builder()

  @tf.function
  def loop_and_compute():
    p13n_metrics = collections.OrderedDict()
    for name, personalize_fn in personalize_fns.items():
      tf.nest.map_structure(
          lambda v, t: v.assign(t), model_weights, initial_model_weights
      )
      p13n_metrics[name] = personalize_fn(model, train_data, test_data)
    return p13n_metrics

  return loop_and_compute()
