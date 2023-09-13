# Copyright 2022, The TensorFlow Federated Authors.
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
"""Program logic for the training and optionally evaluating models.

This program logic performs both federated model training and optionally
federated model evaluation. It is organized to have a main training loop that
runs sequentially and occasionally "forks off" side evaluation loops based on
criteria specified in the `evaluation_periodicity` parameter, using
`EvaluationManager`' to run and manage all the evaluation loops, which run for a
duration of time based on the `evaluation_period` parameter.
"""

import asyncio
from collections.abc import Coroutine
import datetime
from typing import NamedTuple, Optional, Union

from absl import logging

from tensorflow_federated.python.learning.programs import evaluation_program_logic
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.program import data_source
from tensorflow_federated.python.program import federated_context
from tensorflow_federated.python.program import program_state_manager as program_state_manager_lib
from tensorflow_federated.python.program import release_manager
from tensorflow_federated.python.program import value_reference


class ProgramState(NamedTuple):
  """A structure representing the program state."""

  state: composers.LearningAlgorithmState
  round_number: int
  next_evaluation_timestamp_seconds: Optional[int]


class TaskManager:
  """A manager for inflight tasks.

  This object holds references to asyncio Tasks so that they are not garbage
  collected. It registers a callback to remove the reference when the task
  finishes, and provides an interface to wait for all inflight tasks to finish
  before proceeding.

  This is very similar to the `asyncio.TaskGroup` method, which will be availabe
  in Python 3.11.
  """

  def __init__(self):
    self._pending_tasks: set[asyncio.Task] = set()

  async def wait_for_all_tasks(self):
    if not self._pending_tasks:
      return
    await asyncio.wait(
        self._pending_tasks, timeout=None, return_when=asyncio.ALL_COMPLETED
    )

  def _finalize_task(self, task: asyncio.Task) -> None:
    self._pending_tasks.remove(task)
    task.result()  # Trigger any potentially stored exceptions.

  def add_task(self, coro: Coroutine[None, None, None]) -> None:
    new_task = asyncio.create_task(coro)
    new_task.add_done_callback(self._finalize_task)
    self._pending_tasks.add(new_task)


# TODO: b/284509457 - Revisit this API when `initialize` is changed to be a
# value instead of `tff.Computation`.
async def train_model(
    *,
    train_process: learning_process.LearningProcess,
    initial_train_state: Optional[composers.LearningAlgorithmState] = None,
    train_data_source: data_source.FederatedDataSource,
    train_per_round_clients: int,
    train_total_rounds: int,
    program_state_manager: program_state_manager_lib.ProgramStateManager,
    model_output_manager: release_manager.ReleaseManager[
        release_manager.ReleasableStructure, str
    ],
    train_metrics_manager: Optional[
        release_manager.ReleaseManager[release_manager.ReleasableStructure, int]
    ] = None,
    evaluation_manager: Optional[evaluation_program_logic.EvaluationManager],
    evaluation_periodicity: Union[int, datetime.timedelta],
) -> None:
  """Runs specified rounds of training and optionally evaluates the model.

  This method will create an initial training state and repeatedly call
  `train_process.next`, advancing the state of the training process. Depending
  on the configuration of `evaluation_manager`, asynchronous evaluation loops
  will be spawned and executed in parallel.

  This method will save the initial state (result of `train_process.initialize`
  or passed via `initial_train_state`) using `program_state_manager`. If the
  state manager is  configured to keep the first version (e.g.
  `tff.program.FileStateProgramManager`'s `keep_first` parameter), then round
  zero (the initialization) will be retained so that future experiments can use
  the same starting point.

  If the `initial_train_state` is not None, its type signature should be the
  same as the type_signature of the result of `train_process.initialize`.

  Args:
    train_process: A `tff.learning.templates.LearningProcess` to run for
      training. The state type of the `train_process` should be a
      `tff.learning.templates.LearningAlgorithmState`, and the initial train
      state can be provided using the `initial_train_state` argument.
    initial_train_state: (Optional) A
      `tff.learning.templates.LearningAlgorithmState` of the initial state of
      the train process. Its type signature should match the `type_signature` of
      the result of `train_process.initialize`. If not specified, use the
      retsult of `train_process.initialize`.
    train_data_source: A `tff.program.FederatedDataSource` which returns client
      data used during training.
    train_per_round_clients: The number of clients per round of training.
    train_total_rounds: Total number of rounds of training.
    program_state_manager: A `tff.program.ProgramStateManager` used to save
      program state for fault tolerance.
    model_output_manager: A `tff.program.ReleaseManager` to release the model,
      the results can be used for building inference models after training, or
      warm-starting future training loops.
    train_metrics_manager: A `tff.program.ReleaseManager` to release metrics of
      training. Use `tff.program.GroupingReleaseManager` to supply multiple
      release managers.
    evaluation_manager: An `EvaluationManager` used to create a state manager
      for each evaluation loop that is forked off from the training loop.
    evaluation_periodicity: Either a integer number of rounds or
      `datetime.timedelta` to await before sending a new training checkpoint to
      `evaluation_manager.start_evaluation`. Note that the last training round
      will always be evaluated even if it does not satisfy the periodicity.

  Raises:
    ValueError: If the train state is None.
  """
  federated_context.check_in_federated_context()

  # A list of pending tasks (evaluation, value releases, etc) that we must await
  # before shutting down the program.
  task_manager = TaskManager()

  # If this job is restarting, resume any previous evaluations.
  if evaluation_manager is not None:
    logging.info('Looking for previous evaluation states...')
    await evaluation_manager.resume_from_previous_state()

  # Try to load the latest program state; if the program logic failed on a
  # previous run, this program state can be used to restore the execution of
  # this program logic and skip unnecessary steps.
  if initial_train_state is None:
    initial_train_state = await value_reference.materialize_value(
        train_process.initialize()
    )
  train_state = initial_train_state
  if train_state is None:
    raise ValueError('The initial train state is None.')
  program_state, version = await program_state_manager.load_latest(
      ProgramState(train_state, 0, 0)
  )
  if program_state is not None:
    train_state = program_state.state
    start_round = program_state.round_number
    next_evaluation_timestamp_seconds = (
        program_state.next_evaluation_timestamp_seconds
    )
    logging.info('Found previous program state version %d', version)
    if start_round < train_total_rounds:
      logging.info(
          'Resuming from training round %d,running until round %d',
          start_round,
          train_total_rounds,
      )
    else:
      logging.info(
          (
              'Loaded previously completed round %d, but only '
              'requested training until round %d, will not run training.'
          ),
          start_round,
          train_total_rounds,
      )
      if evaluation_manager is not None:
        logging.info('Checking for remaining evaluations need to finish.')
        await evaluation_manager.wait_for_evaluations_to_finish()
      return
  else:
    start_round = 0
    logging.info(
        'Starting program without previous state, saving initial state.'
    )
    # Ensure the initial state (round 0) is saved before any training occurs.
    # The program manager `keep_first=True` parameterization will enable users
    # to start future experiments from the same initialization.
    next_evaluation_timestamp_seconds = None
    await program_state_manager.save(
        ProgramState(
            train_state, start_round, next_evaluation_timestamp_seconds
        ),
        version=start_round,
    )

  train_state_type, _ = train_process.next.type_signature.result  # pytype: disable=attribute-error
  train_data_iterator = train_data_source.iterator()

  # Track a future time after which an evaluation should be started. This will
  # be `evaluation_periodicity` after the most recent evaluation time.
  next_evaluation_time = (
      datetime.datetime.fromtimestamp(next_evaluation_timestamp_seconds)
      if next_evaluation_timestamp_seconds
      else None
  )

  def should_evaluate_round(
      round_num: int, train_round_finished_time: datetime.datetime
  ) -> bool:
    is_last_round = round_num == train_total_rounds
    if is_last_round:
      return True
    elif isinstance(evaluation_periodicity, int):
      return round_num % evaluation_periodicity == 0
    elif isinstance(evaluation_periodicity, datetime.timedelta):
      nonlocal next_evaluation_time
      if (
          next_evaluation_time is None
          or next_evaluation_time < train_round_finished_time
      ):
        next_evaluation_time = (
            train_round_finished_time + evaluation_periodicity
        )
        return True
      return False
    else:
      raise ValueError(
          '`evaluation_periodicity` must be an `int` or a '
          '`datetime.timedelta` type. Got '
          f'{type(evaluation_periodicity)}'
      )

  # This is the main training loop. It sequentially performs federated learning,
  # feeding each rounds state into the next round. Occasionally a "sub-loop"
  # for evaluation is created for a giving training checkpoint, that will run
  # evaluation computations in parallel.
  for round_num in range(start_round + 1, train_total_rounds + 1):
    logging.info('Running train round %d', round_num)
    round_participants_data = train_data_iterator.select(
        train_per_round_clients
    )
    train_result = await value_reference.materialize_value(
        train_process.next(train_state, round_participants_data)
    )
    logging.info('Finished train round %d', round_num)
    if not isinstance(train_result, learning_process.LearningProcessOutput):
      raise TypeError(
          'FederatedContext returned unexpected result type after '
          'training computation invocation. Expected a '
          '`tff.learning.templates.LearningProcessOutput`, got '
          f'{type(train_result)}'
      )
    train_state = train_result.state
    train_metrics = train_result.metrics

    train_round_finished_time = datetime.datetime.now()
    if evaluation_manager is not None and should_evaluate_round(
        round_num, train_round_finished_time
    ):
      model_weights = train_process.get_model_weights(train_state)
      await evaluation_manager.start_evaluation(
          round_num, int(train_round_finished_time.timestamp()), model_weights
      )
      logging.info('Added evaluation for training round %d', round_num)

    next_evaluation_timestamp_seconds = (
        int(next_evaluation_time.timestamp()) if next_evaluation_time else None
    )
    task_manager.add_task(
        program_state_manager.save(
            ProgramState(
                train_state, round_num, next_evaluation_timestamp_seconds
            ),
            version=round_num,
        )
    )
    if train_metrics_manager is not None:
      try:
        released_train_metrics, released_train_metrics_type = (
            evaluation_program_logic.extract_and_rewrap_metrics(
                train_metrics, path=('client_work', 'train')
            )
        )
      except KeyError as e:
        raise KeyError(
            '`train_model` requires the `train_process` argument to be a '
            '`tff.learning.templates.LearningProcess` whose `next` computation '
            'metrics result has a `train` field. Instead got a '
            '`tff.Computation` whose result signature was: '
            f'{train_process.next.type_signature.result}'
        ) from e
      task_manager.add_task(
          train_metrics_manager.release(
              released_train_metrics,
              released_train_metrics_type,
              key=round_num,
          )
      )

  task_manager.add_task(
      model_output_manager.release(
          train_state,
          train_state_type,
          key=f'final_training_checkpoint_round_{train_total_rounds}',
      )
  )
  # Wait for all pending tasks to complete before exiting the program.
  await task_manager.wait_for_all_tasks()
  if evaluation_manager is not None:
    await evaluation_manager.wait_for_evaluations_to_finish()
