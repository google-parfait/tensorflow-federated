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
import datetime
from typing import Optional, Union

from absl import logging

from tensorflow_federated.python.learning.programs import evaluation_program_logic
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.program import data_source
from tensorflow_federated.python.program import federated_context
from tensorflow_federated.python.program import program_state_manager as program_state_manager_lib
from tensorflow_federated.python.program import release_manager
from tensorflow_federated.python.program import value_reference


async def _finalize_tasks(tasks: set[asyncio.Task]):
  """Finalize asynchronous tasks."""
  # Calling `result()` will rasie an error if the underlying task raised an
  # error, otherwise the error is silently swallowed.
  for task in tasks:
    task.result()


async def _wait_for_tasks_to_finish(pending_tasks: set[asyncio.Task]):
  if not pending_tasks:
    return
  done_tasks, _ = await asyncio.wait(
      pending_tasks, timeout=None, return_when=asyncio.ALL_COMPLETED
  )
  await _finalize_tasks(done_tasks)


async def _clear_finished_tasks(
    pending_tasks: set[asyncio.Task],
) -> set[asyncio.Task]:
  """Removes finished tasks, returns the set of futures that are still pending.

  This is intended as a "clean-up" method that will remove items from
  `pending_tasks` that are complete which frees the associated memory and
  also surfaces any errors that may have occur in that task. Without this, the
  Python runtime will need to hold onto all memory for every round until the
  end of the program.

  This method does _not_ wait for tasks to complete, its alright to check back
  on a task later.

  Args:
    pending_tasks: A set of futures for in-progress tasks.

  Returns:
    A set of tasks containing only the tasks in `pending_tasks` that are not
    ready yet.
  """
  if not pending_tasks:
    # Nothing to do.
    return pending_tasks
  # Wait 1 second for any potentially finished task.
  one_second = 1
  done_tasks, pending_tasks = await asyncio.wait(
      pending_tasks, timeout=one_second, return_when=asyncio.FIRST_COMPLETED
  )
  await _finalize_tasks(done_tasks)
  return pending_tasks


async def train_model(
    *,
    train_process: learning_process.LearningProcess,
    train_data_source: data_source.FederatedDataSource,
    train_per_round_clients: int,
    train_total_rounds: int,
    program_state_manager: program_state_manager_lib.ProgramStateManager,
    model_output_manager: release_manager.ReleaseManager,
    train_metrics_manager: Optional[release_manager.ReleaseManager] = None,
    evaluation_manager: Optional[evaluation_program_logic.EvaluationManager],
    evaluation_periodicity: Union[int, datetime.timedelta],
) -> None:
  """Runs specified rounds of training and optionally evaluates the model.

  This method will create an initial training state and repeatedly call
  `train_process.next`, advancing the state of the training process. Depending
  on the configuration of `evaluation_manager`, asynchronous evaluation loops
  will be spawned and executed in parallel.

  This method will save the initial state (result of `train_process.initialize`)
  using `program_state_manager`. If the state manager is  configured to keep the
  first version (e.g.  `tff.program.FileStateProgramManager`'s `keep_first`
  parameter), then round zero (the initialization) will be retained so that
  future experiments can use the same starting point.

  Args:
    train_process: A `tff.learning.templates.LearningProcess` to run for
      training.
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
  """
  federated_context.check_in_federated_context()

  # A list of pending tasks (evaluation, value releases, etc) that we must await
  # before shutting down the program.
  pending_tasks: set[asyncio.Task] = set([])

  # If this job is restarting, resume any previous evaluations.
  if evaluation_manager is not None:
    logging.info('Looking for previous evaluation states...')
    await evaluation_manager.resume_from_previous_state()

  # Try to load the latest program state; if the program logic failed on a
  # previous run, this program state can be used to restore the execution of
  # this program logic and skip unnecessary steps.
  train_state = await value_reference.materialize_value(
      train_process.initialize()
  )
  program_state, version = await program_state_manager.load_latest(
      (train_state, 0)
  )
  if program_state is not None:
    train_state, start_round = program_state
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
    await program_state_manager.save(
        (train_state, start_round), version=start_round
    )

  train_state_type, _ = train_process.next.type_signature.result
  train_data_iterator = train_data_source.iterator()

  # Track a future time after which an evaluation should be started. This will
  # be `evaluation_periodicity` after the most recent evaluation time.
  next_evaluation_time = None

  def should_evaluate_round(
      round_num: int, train_round_finished_time: datetime.datetime
  ) -> bool:
    is_last_round = round_num == train_total_rounds + 1
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
    train_result = train_process.next(train_state, round_participants_data)
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
    # Save the current program state. We await here to avoid the situation
    # were we start the next round, but saving fails, and we end up rolling
    # back to an even earlier round on resumption (a trade-off of speed for
    # potential wasted work).
    await program_state_manager.save(
        (train_state, round_num), version=round_num
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
      pending_tasks.add(
          asyncio.create_task(
              train_metrics_manager.release(
                  released_train_metrics,
                  released_train_metrics_type,
                  key=round_num,
              )
          )
      )
    train_round_finished_time = datetime.datetime.now()
    if evaluation_manager is not None and should_evaluate_round(
        round_num, train_round_finished_time
    ):
      model_weights = train_process.get_model_weights(train_state)
      await evaluation_manager.start_evaluation(
          round_num, int(train_round_finished_time.timestamp()), model_weights
      )
      logging.info(
          'Added evaluation for training round %d. Pending tasks: %s',
          round_num,
          pending_tasks,
      )
    # Clean-up any tasks that have finished in the meantime.
    pending_tasks = await _clear_finished_tasks(pending_tasks)

  pending_tasks.add(
      asyncio.create_task(
          model_output_manager.release(
              train_state,
              train_state_type,
              key=f'final_training_checkpoint_round_{train_total_rounds}',
          )
      )
  )
  # Wait for all pending tasks to complete before exiting the program.
  await _wait_for_tasks_to_finish(pending_tasks)
  if evaluation_manager is not None:
    await evaluation_manager.wait_for_evaluations_to_finish()
