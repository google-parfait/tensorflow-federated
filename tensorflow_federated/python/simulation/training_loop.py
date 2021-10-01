# Copyright 2019, The TensorFlow Federated Authors.
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
"""Training loops for iterative process simulations."""

import collections
import pprint
import time
import typing
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Optional, Tuple
import warnings

from absl import logging

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.program import file_program_state_manager as file_program_state_manager_lib
from tensorflow_federated.python.program import program_state_manager as program_state_manager_lib
from tensorflow_federated.python.program import release_manager as release_manager_lib
from tensorflow_federated.python.simulation import checkpoint_manager
from tensorflow_federated.python.simulation import metrics_manager as metrics_manager_lib

MetricsType = MutableMapping[str, Any]
FileCheckpointManager = checkpoint_manager.FileCheckpointManager
MetricsManager = metrics_manager_lib.MetricsManager
ValidationFnType = Callable[[Any, int], MetricsType]

ROUND_TIME_KEY = 'round_time_in_seconds'
VALIDATION_METRICS_PREFIX = 'validation/'
VALIDATION_TIME_KEY = 'validation_time_in_seconds'

ROUND_NUMBER_KEY = 'round_number'
TRAINING_TIME_KEY = 'training_time_in_seconds'
EVALUATION_METRICS_PREFIX = 'evaluation/'
EVALUATION_TIME_KEY = 'evaluation_time_in_seconds'


def _load_initial_checkpoint(
    template_state: Any,
    file_checkpoint_manager: FileCheckpointManager) -> Tuple[Any, int]:
  """Loads a server state and starting round number from a checkpoint manager.

  This method loads a starting state for the iterative process and a starting
  round number indicating the first round to begin the entire training
  process. If a checkpoint is found, the starting state is set to the checkpoint
  state, and the next round to run is set to the round directly after the
  checkpoint round.

  If no checkpoint is found, the starting state is set to `template_state` and
  the starting round is set to `0`.

  Args:
    template_state: A nested structure to use as a template when reconstructing
      a checkpoint.
    file_checkpoint_manager: A `tff.simulation.FileCheckpointManager` used to
      load a checkpoint.

  Returns:
    A tuple of `(state, start_round)`, where `state` matches the Python
    structure in `initial_state`, and `start_round` is a nonnegative integer
    indicating the round at which training starts.
  """
  ckpt_state, ckpt_round = file_checkpoint_manager.load_latest_checkpoint(
      template_state)
  if ckpt_state is None:
    start_state = template_state
    start_round = 0
  else:
    start_state = ckpt_state
    start_round = ckpt_round + 1
  return start_state, start_round


def _compute_validation_metrics(state: Any, round_num: int,
                                validation_fn: ValidationFnType) -> MetricsType:
  """Computes validation metrics for a given server state and round number.

  Specifically, this will return an ordered dictionary of metrics. The keys in
  the output of `validation_fn` will be prefixed with
  `tff.simulation.VALIDATION_METRICS_PREFIX`. Additionally, the dictionary will
  contain a metric representing the number of seconds required to compute the
  validation metrics, with key `tff.simulation.VALIDATION_TIME_KEY`, prefixed
  by `tff.simulation.VALIDATION_METRICS_PREFIX`.

  Args:
    state: The current state of a simulation.
    round_num: An integer indicating the current round number.
    validation_fn: A callable accepting `state` and `round_num`, and returning a
      mapping of metrics with string-valued keys.

  Returns:
    A mapping of validation metrics, where each key has been prefixed by
    `tff.simulation.VALIDATION_METRICS_PREFIX`.
  """
  validation_start_time = time.time()
  validation_metrics = validation_fn(state, round_num)
  validation_time = time.time() - validation_start_time
  prefixed_validation_metrics = collections.OrderedDict()
  prefixed_validation_metrics[VALIDATION_METRICS_PREFIX +
                              VALIDATION_TIME_KEY] = validation_time
  for key, value in validation_metrics.items():
    prefixed_validation_metrics[VALIDATION_METRICS_PREFIX + key] = value
  return prefixed_validation_metrics


def _create_on_loop_start_fn(
    file_checkpoint_manager: Optional[FileCheckpointManager] = None,
    metrics_managers: Optional[List[MetricsManager]] = None,
    validation_fn: Optional[ValidationFnType] = None):
  """Creates a pre-loop callback function.

  This pre-loop callback performs a number of tasks depending on its input
  arguments. In its full generality, the callback will attempt to load a
  starting state and round number from a checkpoint, and clear all metrics saved
  after that starting round.

  If no checkpoint is available, we assume that no training has occurred, in
  which case we perform pre-training tasks. These include (in order, depending
  on the input arguments) computing validation, metrics on the starting state,
  saving those validation metrics via metrics managers, and saving an initial
  checkpoint. Note that if the validation and metrics writing occurs, we use a
  round number of `0`, which is reserved for pre-training tasks.

  Once the tasks above (or some subset of the tasks, depending on which
  arguments are supplied) are completed, the pre-loop callback returns the
  starting state and round number for training.

  Args:
    file_checkpoint_manager: An optional `tff.simulation.FileCheckpointManager`
      used to load an initial checkpoint, and save an initial checkpoint if no
      such checkpoint is found.
    metrics_managers: An optional list of `tff.simulation.MetricsManager`
      instances used to save initial validation metrics. Note that this occurs
      only if `validation_fn` is not `None.
    validation_fn: A callable accepting the training state and a nonnegative
      integer round number, and returning a python mapping of metrics with
      string-valued keys.

  Returns:
    A callable that accepts the initial state of an iterative process. The
    callable performs the tasks descreibed above, and returns a starting state
    and a positive integer round number at which the training loop should start.
  """
  if metrics_managers is None:
    metrics_managers = []

  def on_loop_start(initial_state):
    """Attempts to load a checkpoint before resuming training."""

    if file_checkpoint_manager is not None:
      start_state, start_round = _load_initial_checkpoint(
          initial_state, file_checkpoint_manager)
    else:
      start_state = initial_state
      start_round = 0

    for metrics_manager in metrics_managers:
      metrics_manager.clear_metrics(start_round)

    if start_round == 0:
      # Perform pre-training actions, including computing initial validation
      # metrics and saving an initial checkpoint.
      if validation_fn is not None:
        validation_metrics = _compute_validation_metrics(
            start_state, 0, validation_fn)
        for metrics_manager in metrics_managers:
          metrics_manager.save_metrics(validation_metrics, 0)

      if file_checkpoint_manager is not None:
        file_checkpoint_manager.save_checkpoint(start_state, round_num=0)
      start_round = 1

    return start_state, start_round

  return on_loop_start


def _create_on_round_end_fn(
    file_checkpoint_manager: Optional[FileCheckpointManager] = None,
    metrics_managers: Optional[List[MetricsManager]] = None,
    validation_fn: Optional[ValidationFnType] = None):
  """Creates a on-round-end callback function.

  In its full generality, this on-round-end callback computes validation metrics
  on the state of an iterative process at a given round number, updates an
  input mapping of metrics with these validation metrics, saves the metrics
  via `tff.simulation.MetricsManager` objects, and saves a checkpoint via a
  `tff.simulation.FileCheckpointManager`.

  Args:
    file_checkpoint_manager: An optional `tff.simulation.FileCheckpointManager`
      used to save a checkpoint. If `None`, no checkpoint saving occurs.
    metrics_managers: An optional list of `tff.simulation.MetricsManager`
      instances used to save metrics.
    validation_fn: An optional callable accepting the training state and a
      nonnegative integer round number, and returning a python mapping of
      metrics with string-valued keys.

  Returns:
    A callable accepting the state of an iterative process an integer round
    number, and a mapping of metrics with key-valued strings. The callable
    performs the tasks listed above, and returns the same state and a
    mapping of metrics with key-valued strings, potentially updated to include
    validation metrics.
  """
  if metrics_managers is None:
    metrics_managers = []

  def on_round_end(state: Any, round_num: int,
                   round_metrics: MetricsType) -> Tuple[Any, MetricsType]:
    if validation_fn is not None:
      validation_metrics = _compute_validation_metrics(state, round_num,
                                                       validation_fn)
      round_metrics.update(validation_metrics)

    for metrics_manager in metrics_managers:
      metrics_manager.save_metrics(round_metrics, round_num)

    if file_checkpoint_manager is not None:
      file_checkpoint_manager.save_checkpoint(state, round_num)

    return state, round_metrics

  return on_round_end


def run_simulation(
    process: iterative_process.IterativeProcess,
    client_selection_fn: Callable[[int], Any],
    total_rounds: int,
    file_checkpoint_manager: Optional[FileCheckpointManager] = None,
    metrics_managers: Optional[List[MetricsManager]] = None,
    validation_fn: Optional[ValidationFnType] = None):
  """Runs a federated training simulation for a given iterative process.

  DEPRECATED: `tff.simulation.run_simulation` is deprecated, please use
  `tff.simulation.run_training_process` instead.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> state)`.
    *   `next`: `<state, client_data> -> <state, metrics>` where state matches
        the output type of `initialize`, and `metrics` has member that is a
        python mapping with string-valued keys.

  This method performs up to `total_rounds` updates to the `state` of `process`.
  At each `round_num`, this update occurs by applying `process.next` to
  `state` and the output of ``client_selection_fn(round_num)`. We refer to this
  as a single "training step".

  This method also records how long it takes (in seconds) to call
  `client_selection_fn` and `process.next` at each round and add this to the
  round metrics with key `tff.simulation.ROUND_TIME_KEY`. Note this does not
  include validation time.

  In full generality, after each round, we compute validation metrics via
  `validation_fn` (if not `None`), add these to the metrics created by
  `process.next` (prefixing with `tff.simulation.VALIDATION_METRICS_KEY`), save
  the combined metrics using the `metrics_managers` (if not `None`), and save a
  checkpoint via `file_checkpoint_manager` (if not `None`).

  Args:
    process: A `tff.templates.IterativeProcess` instance to run.
    client_selection_fn: Callable accepting an integer round number, and
      returning a list of client data to use as federated data for that round.
    total_rounds: The number of federated training rounds to perform.
    file_checkpoint_manager: An optional `tff.simulation.FileCheckpointManager`
      used to periodically save checkpoints of the iterative process state.
    metrics_managers: An optional list of `tff.simulation.MetricsManager`
      objects used to save training metrics throughout the simulation.
    validation_fn: An optional callable accepting the current state of the
      iterative process (ie. the first output argument of
      `iterative_process.next`) and the current round number, and returning a
      mapping of validation metrics.

  Returns:
    The `state` of the iterative process after training.
  """
  warnings.warn(
      '`tff.simulation.run_simulation` is deprecated, please use '
      '`tff.simulation.run_training_process` instead.', DeprecationWarning)
  on_loop_start = _create_on_loop_start_fn(file_checkpoint_manager,
                                           metrics_managers, validation_fn)
  on_round_end = _create_on_round_end_fn(file_checkpoint_manager,
                                         metrics_managers, validation_fn)
  return run_simulation_with_callbacks(process, client_selection_fn,
                                       total_rounds, on_loop_start,
                                       on_round_end)


def run_simulation_with_callbacks(
    process: iterative_process.IterativeProcess,
    client_selection_fn: Callable[[int], Any],
    total_rounds: int,
    on_loop_start: Optional[Callable[[Any], Tuple[Any, int]]] = None,
    on_round_end: Optional[Callable[[Any, int, MetricsType],
                                    Tuple[Any, MetricsType]]] = None):
  """Runs federated training for a given `tff.templates.IterativeProcess`.

  DEPRECATED: `tff.simulation.run_simulation_with_callbacks` is deprecated,
  please use `tff.simulation.run_training_process` instead.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> state)`.
    *   `next`: `<state, client_data> -> <state, metrics>` where state matches
        the output type of `initialize`, and `metrics` has member that is a
        python mapping with string-valued keys.

  This method performs up to `total_rounds` updates to the `state` of `process`.
  At each `round_num`, this update occurs by applying `process.next` to
  `state` and the output of ``client_selection_fn(round_num)`. We refer to this
  as a single "training step".

  This method also records how long it takes (in seconds) to call
  `client_selection_fn` and `process.next` at each round and add this to the
  round metrics with key `tff.simulation.ROUND_TIME_KEY`.

  This method uses up to two callbacks. The first, `on_loop_start`, accepts the
  initial state of `process`, and returns a starting `state` and `round_num` for
  the training loop. The callback can be used for things such as loading
  checkpoints.

  The second callback, `on_round_end` is called after each training step. It
  accepts the output state and metrics of `process.next`, and the current round
  number, and returns a new state and metrics mapping. This can be used for
  computing and saving additional metrics.

  WARNING: These callbacks can access and mutate state and are intended for more
  advanced simulations where the state can be mutated outside of calling
  `process.next`. For example, the `on_round_end` callback can be used to
  mutate state according to the training metrics, enabling various kinds of
  adaptive simulations. If your simulation does not require such mutation, we
  recommend `tff.simulation.run_simulation` instead.

  Args:
    process: A `tff.templates.IterativeProcess` instance to run. Must meet the
      type signature requirements documented above.
    client_selection_fn: Callable accepting an integer round number, and
      returning a list of client data to use as federated data for that round.
    total_rounds: The number of federated training rounds to perform.
    on_loop_start: An optional callable accepting the initial `state` of the
      iterative process, and returning a (potentially updated) `state` and an
      integer `round_num` used to determine where to resume the simulation loop.
    on_round_end: An optional callable accepting the `state` of the iterative
      process, an integer round number, and a mapping of metrics. The callable
      returns a (potentially updated) `state` of the same type, and a
      (potentially updated) mapping of metrics.

  Returns:
    The `state` of the iterative process after training.
  """
  warnings.warn(
      '`tff.simulation.run_simulation_with_callbacks` is deprecated, please '
      'use `tff.simulation.run_training_process` instead.', DeprecationWarning)

  logging.info('Initializing simulation process')
  initial_state = process.initialize()

  if on_loop_start is not None:
    logging.info('Running on loop start callback')
    state, start_round = on_loop_start(initial_state)
  else:
    state = initial_state
    start_round = 1

  for round_num in range(start_round, total_rounds + 1):
    logging.info('Executing round %d', round_num)
    round_metrics = collections.OrderedDict(round_num=round_num)

    train_start_time = time.time()
    federated_train_data = client_selection_fn(round_num)

    state, metrics = process.next(state, federated_train_data)
    train_time = time.time() - train_start_time
    round_metrics[ROUND_TIME_KEY] = train_time
    round_metrics.update(metrics)

    if on_round_end is not None:
      logging.info('running round end callback')
      state, round_metrics = on_round_end(state, round_num, round_metrics)

    logging.info('Output metrics at round {:d}:\n{!s}'.format(
        round_num, pprint.pformat(round_metrics)))

  return state


def run_stateless_simulation(
    computation: computation_base.Computation,
    client_selection_fn: Callable[[int], Any],
    total_rounds: int,
    metrics_managers: Optional[List[MetricsManager]] = None):
  """Runs a federated computation on a given set of client data.

  This method performs `total_rounds` calls to the `computation`. At each round,
  this method samples client data via `client_selection_fn(round_num)`, and uses
  this as input to `computation`. The output of `computation` is assumed to be
  a mutable mapping with string-valued keys.

  This method also records how long it takes (in seconds) to call
  `client_selection_fn` and `computation` at each round and adds this to a
  dictionary of  round metrics with key `tff.simulation.ROUND_TIME_KEY`.

  Args:
    computation: A `tff.Computation` to be executed. Must accept a single
      argument (placed or unplaced).
    client_selection_fn: Callable accepting an integer round number, and
      returning a list of client data to use as federated data for that round.
    total_rounds: The number of federated training rounds to perform.
    metrics_managers: An optional list of `tff.simulation.MetricsManager`
      objects used to save metrics throughout the simulation.

  Returns:
    An dictionary, keyed by round number, with values corresponding to the
      outputs of each round's computation, with extra keys for timing
      information.
  """
  # TODO(b/194841884): Add an optional checkpoint manager argument once the
  # checkpoint managers have compatibility with "stateless" structures.
  start_round = 0
  if metrics_managers is not None:
    for manager in metrics_managers:
      manager.clear_metrics(start_round)

  all_metrics = collections.OrderedDict()
  for round_num in range(start_round, total_rounds):
    round_metrics = collections.OrderedDict(round_num=round_num)
    computation_start_time = time.time()

    federated_data = client_selection_fn(round_num)
    output = computation(federated_data)
    computation_time = time.time() - computation_start_time
    logging.info('Computation completed, took %.4f seconds', computation_time)

    round_metrics.update(output)
    round_metrics[ROUND_TIME_KEY] = computation_time

    if metrics_managers is not None:
      for manager in metrics_managers:
        manager.save_metrics(round_metrics, round_num)

    all_metrics[round_num] = round_metrics

  return all_metrics


def _run_training(training_fn: computation_base.Computation,
                  client_selection_fn: Callable[[int], Any], state: Any,
                  round_num: int) -> Tuple[Any, Mapping[str, Any]]:
  """Runs one round of federated training."""
  logging.info('Running training at round %d', round_num)
  metrics = collections.OrderedDict()
  training_time_start = time.time()
  training_data = client_selection_fn(round_num)
  state, training_metrics = training_fn(state, training_data)
  training_time = time.time() - training_time_start
  metrics.update(training_metrics)
  metrics[TRAINING_TIME_KEY] = training_time
  metrics[ROUND_NUMBER_KEY] = round_num
  return state, metrics


def _run_evaluation(evaluation_fn: computation_base.Computation,
                    client_selection_fn: Callable[[int], Any], state: Any,
                    round_num: int) -> Mapping[str, Any]:
  """Runs one round of federated evaluation."""
  logging.info('Running evaluation at round %d', round_num)
  metrics = collections.OrderedDict()
  evaluation_time_start = time.time()
  evaluation_data = client_selection_fn(round_num)
  evaluation_metrics = evaluation_fn(state, evaluation_data)
  evaluation_time = time.time() - evaluation_time_start
  metrics.update(evaluation_metrics)
  metrics[EVALUATION_TIME_KEY] = evaluation_time
  return {EVALUATION_METRICS_PREFIX + k: v for (k, v) in metrics.items()}


def run_training_process(
    training_process: iterative_process.IterativeProcess,
    training_selection_fn: Callable[[int], Any],
    total_rounds: int,
    evaluation_fn: Optional[computation_base.Computation] = None,
    evaluation_selection_fn: Optional[Callable[[int], Any]] = None,
    rounds_per_evaluation: int = 1,
    program_state_manager: Optional[
        program_state_manager_lib.ProgramStateManager] = None,
    rounds_per_saving_program_state: int = 1,
    metrics_managers: Optional[Iterable[
        release_manager_lib.ReleaseManager]] = None):
  """Runs a federated `training_process`.

  The following `tff.Computation` types signaures are required:

  *   `training_process.initialize`: `( -> state)`.
  *   `training_process.next`: `<state, client_data> -> <state, metrics>`
  *   `evaulation_fn`:  `<state, client_data> -> metrics`

  This function performs up to `total_rounds` updates to the `state` of the
  given `training_process`. At each training round, this update occurs by
  invoking `training_process.next` with `state` and the output of
  `training_selection_fn`. Depending on `rounds_per_evaluation` and
  `rounds_per_saving_program_state`, each training round may be followed by an
  invocation of the `evaluation_fn` and by saving the program state.

  Note: Round 0 represents saving an initial program model state and computing
  initial evaluation metrics and round 1 through total_rounds + 1 represent the
  training rounds.

  In addition to the training metrics and evaluation metrics, this function adds
  the following performance metrics (key and descriptions):

  * tff.simulation.ROUND_NUMBER_KEY: The round number.
  * tff.simulation.TRAINING_TIME_KEY: The amount of time (in seconds) it takes
    to run one round of training.
  * tff.simulation.EVALUATION_TIME_KEY: The amount of time (in seconds) it takes
    to run one round of evaluation.

  Args:
    training_process: A `tff.templates.IterativeProcess` to run for training.
    training_selection_fn: A `Callable` accepting an integer round number, and
      returning a list of client data to use for trainig in that round.
    total_rounds: The number of training rounds to run.
    evaluation_fn: An optional `tff.Computation` to run for evaluation.
    evaluation_selection_fn: A optional `Callable` accepting an integer round
      number, and returning a list of client data to use for evaluation in that
      round.
    rounds_per_evaluation: The number of training rounds to run between each
      invocation of `evaluation_fn`.
    program_state_manager: An optional `tff.program.ProgramStateManager` to use
      to save program state for fault tolerance.
    rounds_per_saving_program_state: The number of training rounds to run
      between saving program state.
    metrics_managers: An optional list of `tff.program.ReleaseManagers`s to use
      to save metrics.

  Returns:
    The `state` of the training process after training.
  """
  logging.info('Running training process')

  # TODO(b/199737690): Update `FileProgramStateManager` to not require a
  # structure to load program state; once this is fixed, we can move the
  # initialize invocation down so it's only called if required.
  initial_state = training_process.initialize()
  if isinstance(program_state_manager,
                file_program_state_manager_lib.FileProgramStateManager):
    file_program_state_manager = typing.cast(
        file_program_state_manager_lib.FileProgramStateManager,
        program_state_manager)
    file_program_state_manager.set_structure(initial_state)

  if program_state_manager is not None:
    program_state, version = program_state_manager.load_latest()
  else:
    program_state = None
  if program_state is not None:
    logging.info('Loaded program state at version %d', version)
    state = program_state
    start_round = version
  else:
    logging.info('Initializing training process')
    state = initial_state
    start_round = 1

    if evaluation_fn is not None and evaluation_selection_fn is not None:
      evaluation_metrics = _run_evaluation(evaluation_fn,
                                           evaluation_selection_fn, state, 0)

      if metrics_managers is not None:
        for metrics_manager in metrics_managers:
          metrics_manager.release(evaluation_metrics, 0)

    if program_state_manager is not None:
      program_state_manager.save(state, 0)

  for round_num in range(start_round, total_rounds + 1):
    logging.info('Starting round %d', round_num)
    round_metrics = collections.OrderedDict()
    state, training_metrics = _run_training(training_process.next,
                                            training_selection_fn, state,
                                            round_num)
    round_metrics.update(training_metrics)

    if evaluation_fn is not None and evaluation_selection_fn is not None:
      if round_num % rounds_per_evaluation == 0:
        evaluation_metrics = _run_evaluation(evaluation_fn,
                                             evaluation_selection_fn, state,
                                             round_num)
        round_metrics.update(evaluation_metrics)

    if metrics_managers is not None:
      for metrics_manager in metrics_managers:
        metrics_manager.release(round_metrics, round_num)

    if program_state_manager is not None:
      if round_num % rounds_per_saving_program_state == 0:
        program_state_manager.save(state, round_num)

  return state
