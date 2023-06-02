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
from typing import Any, Optional

import tensorflow as tf

from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import sampling
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.models import reconstruction_model
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.learning.templates import learning_process


_MetricsType = Mapping[str, Any]
_SplitFnType = Callable[
    [tf.data.Dataset], tuple[tf.data.Dataset, tf.data.Dataset]
]
_MetricsProcessFnType = Callable[[_MetricsType], _MetricsType]
_FinetuneEvalFnType = Callable[
    [variable.VariableModel, tf.data.Dataset, tf.data.Dataset, Any],
    _MetricsType,
]

_TRAIN_DATA_KEY = 'train_data'
_EVAL_DATA_KEY = 'test_data'
_CONTEXT_KEY = 'context'


def build_personalization_eval_computation(
    model_fn: Callable[[], variable.VariableModel],
    personalize_fn_dict: Mapping[str, Callable[[], _FinetuneEvalFnType]],
    baseline_evaluate_fn: Callable[
        [variable.VariableModel, tf.data.Dataset], _MetricsType
    ],
    max_num_clients: int = 100,
    context_tff_type: computation_types.Type = None,
) -> computation_base.Computation:
  """Builds the TFF computation for evaluating personalization strategies.

  The returned TFF computation broadcasts model weights from `tff.SERVER` to
  `tff.CLIENTS`. Each client evaluates the personalization strategies given in
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
      an unbatched `tf.data.Dataset` for train, an unbatched `tf.data.Dataset`
      for test, and an arbitrary context object (which is used to hold any extra
      information that a personalization strategy may use), trains a
      personalized model, and returns the evaluation metrics. The evaluation
      metrics are represented as an `OrderedDict` (or a nested `OrderedDict`) of
      `string` metric names to scalar `tf.Tensor`s.
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
    context_tff_type: A `tff.Type` of the optional context object used by the
      personalization strategies defined in `personalization_fn_dict`. We use a
      context object to hold any extra information (in addition to the training
      dataset) that personalization may use. If context is used in
      `personalization_fn_dict`, its `tff.Type` must be provided here.

  Returns:
    A federated `tff.Computation` with the functional type signature
    `(<model_weights@SERVER, input@CLIENTS> -> personalization_metrics@SERVER)`:

    *   `model_weights` is a `tff.learning.models.ModelWeights`.
    *   Each client's input is an `OrderedDict` of two required keys
        `train_data` and `test_data`; each key is mapped to an unbatched
        `tf.data.Dataset`. If extra context (e.g., extra datasets) is used in
        `personalize_fn_dict`, then client input has a third key `context` that
        is mapped to a object whose `tff.Type` is provided by the
        `context_tff_type` argument.
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
  # TODO(b/124477628): Replace it with other ways of handling metadata.
  with tf.Graph().as_default():
    py_typecheck.check_callable(model_fn)
    model = model_fn()
    model_weights_type = model_weights_lib.weights_type_from_model(model)
    batch_tff_type = computation_types.to_type(model.input_spec)

  # Define the `tff.Type` of each client's input. Since batching (as well as
  # other preprocessing of datasets) is done within each personalization
  # strategy (i.e., by functions in `personalize_fn_dict`), the client-side
  # input should contain unbatched elements.
  element_tff_type = _remove_batch_dim(batch_tff_type)
  client_input_type = collections.OrderedDict(
      train_data=computation_types.SequenceType(element_tff_type),
      test_data=computation_types.SequenceType(element_tff_type),
  )
  if context_tff_type is not None:
    py_typecheck.check_type(context_tff_type, computation_types.Type)
    client_input_type[_CONTEXT_KEY] = context_tff_type
  client_input_type = computation_types.to_type(client_input_type)

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

  @federated_computation.federated_computation(
      computation_types.FederatedType(model_weights_type, placements.SERVER),
      computation_types.FederatedType(client_input_type, placements.CLIENTS),
  )
  def personalization_eval(server_model_weights, federated_client_input):
    """TFF orchestration logic."""
    client_init_weights = intrinsics.federated_broadcast(server_model_weights)
    client_final_metrics = intrinsics.federated_map(
        client_computation, (client_init_weights, federated_client_input)
    )

    # WARNING: Collecting information from clients can be risky. Users have to
    # make sure that it is proper to collect those metrics from clients.
    # TODO(b/147889283): Add a link to the TFF doc once it exists.
    sampling_output = aggregation_process.next(
        aggregation_process.initialize(), client_final_metrics  # No state.
    )
    # In the future we may want to output `sampling_output.measurements` also
    # but currently it is empty.
    return sampling_output.result

  return personalization_eval


def _build_client_computation(
    model_weights_type: computation_types.Type,
    client_data_type: computation_types.Type,
    model_fn: Callable[[], variable.VariableModel],
    personalize_fn_dict: Mapping[str, Callable[[], _FinetuneEvalFnType]],
    baseline_evaluate_fn: Callable[
        [variable.VariableModel, tf.data.Dataset], _MetricsType
    ],
):
  """Return an unplaced tff.Computation representing the client computation."""

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
    context = client_input.get(_CONTEXT_KEY)

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
        context,
    )
    final_metrics.update(p13n_metrics)
    return final_metrics

  return _client_computation


def _remove_batch_dim(
    type_spec: computation_types.Type,
) -> computation_types.Type:
  """Removes the batch dimension from the `tff.TensorType`s in `type_spec`.

  Args:
    type_spec: A `tff.Type` containing `tff.TensorType`s as leaves. The first
      dimension in the leaf `tff.TensorType` is the batch dimension.

  Returns:
    A `tff.Type` of the same structure as `type_spec`, with no batch dimensions
    in all the leaf `tff.TensorType`s.

  Raises:
    TypeError: If the argument has the wrong type.
    ValueError: If the `tff.TensorType` does not have the first dimension.
  """

  def _remove_first_dim_in_tensortype(tensor_type):
    """Return a new `tff.TensorType` after removing the first dimension."""
    py_typecheck.check_type(tensor_type, computation_types.TensorType)
    if (tensor_type.shape.rank is not None) and (tensor_type.shape.rank >= 1):
      return computation_types.TensorType(
          shape=tensor_type.shape[1:], dtype=tensor_type.dtype
      )
    else:
      raise ValueError('Provided shape must have rank 1 or higher.')

  return structure.map_structure(_remove_first_dim_in_tensortype, type_spec)


def _compute_baseline_metrics(
    model_fn, initial_model_weights, test_data, baseline_evaluate_fn
):
  """Evaluate the model with weights being the `initial_model_weights`."""
  model = model_fn()
  model_weights = model_weights_lib.ModelWeights.from_model(model)

  @tf.function
  def assign_and_compute():
    tf.nest.map_structure(
        lambda v, t: v.assign(t), model_weights, initial_model_weights
    )
    py_typecheck.check_callable(baseline_evaluate_fn)
    return baseline_evaluate_fn(model, test_data)

  return assign_and_compute()


def _compute_p13n_metrics(
    model_fn,
    initial_model_weights,
    train_data,
    test_data,
    personalize_fn_dict,
    context,
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
  personalize_fns = collections.OrderedDict()
  for name, personalize_fn_builder in personalize_fn_dict.items():
    py_typecheck.check_type(name, str)
    py_typecheck.check_callable(personalize_fn_builder)
    personalize_fns[name] = personalize_fn_builder()

  @tf.function
  def loop_and_compute():
    p13n_metrics = collections.OrderedDict()
    for name, personalize_fn in personalize_fns.items():
      tf.nest.map_structure(
          lambda v, t: v.assign(t), model_weights, initial_model_weights
      )
      py_typecheck.check_callable(personalize_fn)
      p13n_metrics[name] = personalize_fn(model, train_data, test_data, context)
    return p13n_metrics

  return loop_and_compute()


def _build_personalization_eval_client_work(
    model_fn: Callable[[], variable.VariableModel],
    personalize_fn_dict: Mapping[str, Callable[[], _FinetuneEvalFnType]],
    baseline_evaluate_fn: Callable[
        [variable.VariableModel, tf.data.Dataset], _MetricsType
    ],
    max_num_clients: int,
    split_data_fn: _SplitFnType,
    derived_metrics_processing_fn: _MetricsProcessFnType,
) -> client_works.ClientWorkProcess:
  """Builds a `ClientWorkProcess` that performs model evaluation at clients."""

  with tf.Graph().as_default():
    model = model_fn()
    model_weights_type = model_weights_lib.weights_type_from_model(model)
    batch_type = computation_types.to_type(model.input_spec)

  unsplit_client_data_type = computation_types.SequenceType(batch_type)

  # Define the `tff.Type` of each client's input. Since batching (as well as
  # other preprocessing of datasets) is done within each personalization
  # strategy (i.e., by functions in `personalize_fn_dict`), the client-side
  # input should contain unbatched elements.
  element_type = _remove_batch_dim(batch_type)
  unbatched_split_client_data_type = collections.OrderedDict(
      train_data=computation_types.SequenceType(element_type),
      test_data=computation_types.SequenceType(element_type),
  )
  unbatched_split_client_data_type = computation_types.to_type(
      unbatched_split_client_data_type
  )

  py_typecheck.check_type(max_num_clients, int)
  if max_num_clients <= 0:
    raise ValueError('max_num_clients must be a positive integer.')

  @federated_computation.federated_computation
  def init_fn() -> Any:  # Returns a federated value.
    return intrinsics.federated_value((), placements.SERVER)  # No state.

  client_computation = _build_client_computation(
      model_weights_type=model_weights_type,
      client_data_type=unbatched_split_client_data_type,
      model_fn=model_fn,
      personalize_fn_dict=personalize_fn_dict,
      baseline_evaluate_fn=baseline_evaluate_fn,
  )

  @tensorflow_computation.tf_computation(unsplit_client_data_type)
  def split_data_for_client(
      unsplit_dataset: tf.data.Dataset,
  ) -> Mapping[str, Any]:
    train_dataset, test_dataset = split_data_fn(unsplit_dataset)
    return collections.OrderedDict(
        train_data=train_dataset,
        test_data=test_dataset,
    )

  reservoir_sampling_factory = sampling.UnweightedReservoirSamplingFactory(
      sample_size=max_num_clients
  )
  aggregation_process = reservoir_sampling_factory.create(
      client_computation.type_signature.result
  )

  @tensorflow_computation.tf_computation()
  def _metrics_postprocessing(
      raw_sampling_metrics: _MetricsType,
  ) -> _MetricsType:
    return derived_metrics_processing_fn(raw_sampling_metrics)

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.at_clients(model_weights_type),
      computation_types.at_clients(unsplit_client_data_type),
  )
  def next_fn(
      state: Any,
      model_weights: model_weights_lib.ModelWeights,
      unsplit_client_data: tf.data.Dataset,
  ) -> measured_process.MeasuredProcessOutput:
    del state

    unbatched_split_client_data = intrinsics.federated_map(
        split_data_for_client, unsplit_client_data
    )

    client_final_metrics = intrinsics.federated_map(
        client_computation, (model_weights, unbatched_split_client_data)
    )

    # These are the 'raw' metrics, where the values are lists showing numbers
    # from each selected client (so with cardinality of `max_num_clients`).
    raw_sampling_metrics = aggregation_process.next(
        aggregation_process.initialize(), client_final_metrics  # No state.
    ).result
    # These are additional metrics derived from the 'raw' per-client metrics.
    derived_metrics = intrinsics.federated_map(
        _metrics_postprocessing, raw_sampling_metrics
    )

    measurements = intrinsics.federated_zip(
        collections.OrderedDict(
            # Note: This key name ('eval'), and the 'current_round_metrics'
            # and 'total_rounds_metrics' key names in the associated dictionary
            # value, are necessary to match the expectations of the
            # `tff.learning.programs.EvaluationManager` class, which may consume
            # and use this learning process.
            eval=collections.OrderedDict(
                current_round_metrics=collections.OrderedDict(
                    raw=raw_sampling_metrics,
                    derived=derived_metrics,
                ),
                total_rounds_metrics=(),
            )
        )
    )

    # Return empty state as no state is tracked round-over-round.
    empty_state = intrinsics.federated_value((), placements.SERVER)
    # Return empty result as no model update will be performed for evaluation.
    empty_client_result = intrinsics.federated_value(
        client_works.ClientResult(update=(), update_weight=()),
        placements.CLIENTS,
    )
    return measured_process.MeasuredProcessOutput(
        empty_state, empty_client_result, measurements
    )

  return client_works.ClientWorkProcess(init_fn, next_fn)


def build_personalization_eval(
    model_fn: Callable[[], variable.VariableModel],
    personalize_fn_dict: Mapping[str, Callable[[], _FinetuneEvalFnType]],
    baseline_evaluate_fn: Callable[
        [variable.VariableModel, tf.data.Dataset], _MetricsType
    ],
    max_num_clients: int = 100,
    split_data_fn: Optional[_SplitFnType] = None,
    derived_metrics_processing_fn: Optional[_MetricsProcessFnType] = None,
    model_distributor: Optional[distributors.DistributionProcess] = None,
) -> learning_process.LearningProcess:
  """Builds a learning process that performs personalization evaluation.

  This function creates a `tff.learning.templates.LearningProcess` that performs
  personalization evaluation on clients. The learning process has the following
  methods inherited from `tff.learning.templates.LearningProcess`:

  *   `initialize`: A `tff.Computation` with type signature `( -> S@SERVER)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState`
      representing the initial state of the server. In the case of
      personalization evaluation, this state as unused and empty.
  *   `next`: A `tff.Computation` with type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <L@SERVER>)` where `S` is a
      `LearningAlgorithmState` whose type matches that of the output
      of `initialize`, and `{B*}@CLIENTS` represents the client datasets, where
      `B` is the type of a single batch. The output `L` contains an (empty)
      server state, as well as the aggregated personalization evaluation metrics
      at the server.
  *   `get_model_weights`: A `tff.Computation` with type signature `(S -> M)`,
      where `S` is a `tff.learning.templates.LearningAlgorithmState` whose type
      matches the output of `initialize` and `next`, and `M` represents the type
      of the model weights used during evaluation.
  *   `set_model_weights`: A `tff.Computation` with type signature
      `(<S, M> -> S)`, where `S` is a
      `tff.learning.templates.LearningAlgorithmState` whose type matches the
      output of `initialize` and `M` represents the type of the model weights
      used during evaluation.

  Each time `next` is called, the server model is broadcast to each client using
  a distributor. Each client personalizes the model, evaluates both the
  baseline model and the personalized model, and then reports these local
  unfinalized metrics. The local unfinalized metrics are then aggregated and
  finalized at server. Both current round and total rounds
  metrics will be produced. There are no updates of the server model during the
  evaluation process.

  Each client evaluates the personalization strategies given in
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
      an unbatched `tf.data.Dataset` for train, an unbatched `tf.data.Dataset`
      for test, trains a personalized model, and returns the evaluation metrics.
      The evaluation metrics are represented as an `OrderedDict` (or a nested
      `OrderedDict`) of `string` metric names to scalar `tf.Tensor`s.
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
    split_data_fn: An optional function which takes a single dataset and returns
      a tuple of two datasets, to be used for splitting each client's dataset
      into train and test datasets. The first dataset in the returned tuple will
      be used for training, and the second will be used for evaluation. Note
      that both train and test datasets are expected to be *unbatched*, so that
      batching is under the control of the personalization strategies specified
      via the `personalize_fn_dict` argument. Consequently, if this learning
      process will be used as part of a script or higher order abstraction which
      passes client datasets in batched form, then this argument needs to
      perform the unbatching. If unspecified, the default function used will
      unbatch an (assumed batched) input dataset and then split the examples
      into halves.
    derived_metrics_processing_fn: An optional function which takes a dictionary
      of 'raw' personalization evaluation metrics for a round (i.e., metrics
      which are lists of per-client values), and calculates additional metrics
      derived from the 'raw' metrics. This could be used e.g. to aggregate
      per-client values to scalars: one could take the per-client difference in
      accuracy between a personalization strategy and the baseline, and
      calculate two 'derived' scalar metrics which are the counts of clients
      which saw improvement/no improvement via personalization. Note that the
      processing performed here is dependent on the raw metrics defined in the
      the `personalize_fn_dict` and `baseline_evaluate_fn` arguments; errors
      will occur e.g. if a raw metrics is expected to exist here but was not
      defined in `personalize_fn_dict`/`baseline_evaluate_fn`. If unspecified,
      *no* derived metrics will be calculated.
    model_distributor: An optional `tff.learning.templates.DistributionProcess`
      that broadcasts the model weights on the server to the clients. It must
      support the signature `(input_values@SERVER -> output_values@CLIENTS)` and
      have empty state. If None, the server model is broadcast to the clients
      using the default `tff.federated_broadcast`.

  Returns:
    A `tff.learning.templates.LearningProcess` that performs personalization
    evaluation on clients, and returns metrics.

  Raises:
    TypeError: If arguments are of the wrong types.
    ValueError: If `baseline_metrics` is used as a key in `personalize_fn_dict`.
    ValueError: If `max_num_clients` is not positive.
  """

  @tensorflow_computation.tf_computation()
  def initial_model_weights_fn():
    return model_weights_lib.ModelWeights.from_model(model_fn())

  model_weights_type = initial_model_weights_fn.type_signature.result

  if split_data_fn is None:
    split_data_fn_after_unbatching = (
        reconstruction_model.ReconstructionModel.build_dataset_split_fn(
            split_dataset=True
        )
    )
    split_data_fn = lambda ds: split_data_fn_after_unbatching(ds.unbatch())

  if derived_metrics_processing_fn is None:
    derived_metrics_processing_fn = lambda x: collections.OrderedDict()

  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(model_weights_type)
  else:
    py_typecheck.check_type(model_distributor, distributors.DistributionProcess)

  client_work = _build_personalization_eval_client_work(
      model_fn=model_fn,
      personalize_fn_dict=personalize_fn_dict,
      baseline_evaluate_fn=baseline_evaluate_fn,
      max_num_clients=max_num_clients,
      split_data_fn=split_data_fn,
      derived_metrics_processing_fn=derived_metrics_processing_fn,
  )

  client_work_result_type = computation_types.at_clients(
      client_works.ClientResult(update=(), update_weight=())
  )
  model_update_type = client_work_result_type.member.update
  model_update_weight_type = client_work_result_type.member.update_weight
  model_aggregator_factory = mean.MeanFactory()
  model_aggregator = model_aggregator_factory.create(
      model_update_type, model_update_weight_type
  )

  # The finalizer performs no update on model weights.
  finalizer = finalizers.build_identity_finalizer(
      model_weights_type,
      model_aggregator.next.type_signature.result.result.member,
  )

  return composers.compose_learning_process(
      initial_model_weights_fn,
      model_distributor,
      client_work,
      model_aggregator,
      finalizer,
  )
