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
"""Program logic for evaluating federated learning.

This package contains program logic and abstractions for
managing federated model evaluation. It exposed following symbols:

*   `EvaluationManager` which manages "forking off" side evaluation loops that
    run the evaluation loop for a duration of time based on the
    `evaluation_period` parameter. It keeps track of evaluation state enabling
    resumption of evaluations in the case the program exits prematurely.

This logic is intended to be paired with another set of program logic that
produces model weights to evaluate, typically a training program. For example
if a program is configured to evaluate every other training round for two
evaluation rounds, the sequence may look something like this:

```
   ┌────────┐  ┌──────────┐        ┌──────────┐
   │Training│  │Evaluation│        │Evaluation│
   └───┬────┘  └───┬──────┘        └───┬──────┘
       │           │                   │
       │──┐        │                   │
       │ Round 1   │                   │
       │<─┘        │                   │
       │──┐        │                   │
       │ Round 2──>│                   │
       │<─┘        │──┐                │
       │──┐        │  │ Eval Round 1   │
       │ Round 3   │<─┘                │
       │<─┘        │──┐                │
       │──┐        │  │ Eval Round 2   │
       │ Round 4──────────────────────>│
       │<─┘        │<─┘                │──┐
       │──┐        │                   │  │ Eval Round 1
       │ Round 5   │                   │<─┘
       │<─┘        │                   │──┐
       │──┐        │                   │  │ Eval Round 2
       │ Round 6────────────────────────────────────────────>
       │<─┘        │                   │<─┘
       ...
```

Where the Evaluation sequences are managed by the `EvaluationManager`.

Note: "Eval Round N" is output as `Training Round + Eval Round - 1`. This way
the first evaluation round for an evaluation loop will align on the x-axis with
the training round that provides the checkpoint that is being evaluated.
"""

import asyncio
from collections.abc import Callable, Mapping, MutableMapping, Sequence
import datetime
import string
import time
import typing
from typing import Any, Optional, Union

from absl import logging as _logging
import federated_language
import numpy as np

from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.program import file_program_state_manager

# The prefix path for metrics exported to TensorBoard. This will group all the
# metrics under tab with the same name.
MODEL_METRICS_PREFIX = 'model_metrics'
# This path matches the `tff.learning.algorithms.build_fed_eval` signature. It
# is the prefix path to the current round and total round metrics.
_EVAL_METRICS_PATH_COMPONENTS = ('client_work', 'eval')

# This path matches the `tff.learning.algorithms.build_fed_multi_model_eval`
# signature for multi-model evaluation. It is the prefix path to the current
# round and total round metrics for each model. For example, the metrics for
# model 'a' will be stored under 'client_work/a/eval/...'.
_MULTI_MODEL_EVAL_METRICS_PATH_COMPONENTS_BEFORE_MODEL_ID = ('client_work',)
_MULTI_MODEL_EVAL_METRICS_PATH_COMPONENTS_AFTER_MODEL_ID = ('eval',)


def _append_integer(integer_array: np.ndarray, new_value: int) -> np.ndarray:
  return np.concatenate(
      [integer_array, np.asarray([new_value], dtype=np.int32)]
  )


def _pop_value(
    round_array: np.ndarray, time_array: np.ndarray, value: int
) -> tuple[np.ndarray, np.ndarray]:
  _logging.vlog(5, 'Popping %s from %s', value, round_array)
  if np.size(round_array) == 0:
    raise ValueError(f'round_array was empty, but tried to pop value: {value}')
  rounds_to_keep = round_array != value
  return round_array[rounds_to_keep], time_array[rounds_to_keep]


_EVAL_MANAGER_KEY = 'eval_manager'
_EVAL_NAME_PATTERN = 'evaluation_of_train_round_{round_num:05d}'


class AutoVersionAdvanceingStateManager:
  """A file state manager that automatically advances the version number."""

  def __init__(
      self,
      state_manager: file_program_state_manager.FileProgramStateManager,
  ):
    """Initializes the AutoVersionAdvanceingStateManager.

    Args:
      state_manager: The file state manager to use for saving and loading state.
    """
    self._state_manager = state_manager
    self._next_version = 0
    self._lock = asyncio.Lock()  # Lock for concurrency safety.

  async def load_latest(
      self, structure: federated_language.program.ProgramStateStructure
  ) -> federated_language.program.ProgramStateStructure:
    """Returns the latest program state.

    Args:
      structure: The structure of the saved program state for the given
        `version` used to support serialization and deserialization of
        user-defined classes in the structure.
    """
    async with self._lock:
      state, version = await self._state_manager.load_latest(structure)
      self._next_version = version + 1
      return state

  async def save(
      self,
      program_state: federated_language.program.ProgramStateStructure,
  ) -> None:
    """Saves `program_state` and automatically advances the version number.

    Args:
      program_state: A `federated_language.program.ProgramStateStructure` to
        save.
    """
    async with self._lock:
      await self._state_manager.save(program_state, version=self._next_version)
      self._next_version += 1


class EvaluationManager:
  """A manager for facilitating multiple in-progress evaluations.

  This manager performs three responsbilities:

  1.  Prepares, starts and tracks new evaluation loops. This involves creating
      a new evaluation process and state manager for that process, adding
      the new process to the list of tracked inprocess evaluations, and
      creating a new `asyncio.Task` to run the evaluation loop.
  2.  Record evaluations that have finished. This removes the evaluation from
      the list of in-progresss evaluations.
  3.  If the program has restarted, load the most recent state of in-progress
      evaluations and restart each of the evaluations.

  This class uses N + 1 `federated_language.program.ProgramStateManagers` to
  enable
  resumable
  evaluations.

  *   The first state managers is for this class itself, and manages the list of
      in-progress evaluations via two tensor objects. Tensor objects must be
      used (rather than Python lists) because
      `tff.program.FileProgramStateManager` does not support state objects that
      change Python _structure_ across versions (e.g. to load the next version,
      we must known its shape, but after a restart we don't know).
      Alternatively, we can use tensor or ndarray objects with shape `[None]` to
      support changing shapes of structure's leaf elements.
  *   The next N state managers manage the cross-round metric aggregation for
      each evaluation process started. One for each evaluation process.
  """

  def __init__(
      self,
      data_source: federated_language.program.FederatedDataSource,
      aggregated_metrics_manager: Optional[
          federated_language.program.ReleaseManager[
              federated_language.program.ReleasableStructure, int
          ]
      ],
      create_state_manager_fn: Callable[
          [str], file_program_state_manager.FileProgramStateManager
      ],
      create_process_fn: Callable[
          [str],
          tuple[
              learning_process.LearningProcess,
              Optional[
                  federated_language.program.ReleaseManager[
                      federated_language.program.ReleasableStructure, int
                  ]
              ],
          ],
      ],
      cohort_size: int,
      duration: datetime.timedelta = datetime.timedelta(hours=24),
  ):
    """Creates an EvaluationManager.

    Args:
      data_source: A `federated_language.program.FederatedDataSource` that the
        manager will use to create iterators for evaluation loops.
      aggregated_metrics_manager: A `federated_language.program.ReleaseManager`
        for releasing the total aggregated metrics at the end of the evaluation
        loop.
      create_state_manager_fn: A callable that returns a
        `tff.program.FileProgramStateManager` that will be used to create the
        overall evaluation manager's state manager, and each per evaluation loop
        state manager that will enable resuming and checkpointing.
      create_process_fn: A callable that returns a 2-tuple of
        `tff.learning.templates.LearningProcess` and
        `federated_language.program.ReleaseManager` for the per-evaluation round
        metrics releasing that will used be to start each evaluation loop.
      cohort_size: An integer denoting the size of each evaluation round to
        select from the iterator created from `data_source`.
      duration: The `datetime.timedelta` duration to run each evaluation loop.
    """
    self._data_source = data_source
    self._aggregated_metrics_manager = aggregated_metrics_manager
    self._create_state_manager_fn = create_state_manager_fn
    self._create_evaluation_process_fn = create_process_fn
    self._cohort_size = cohort_size
    self._duration = duration
    self._state_manager = AutoVersionAdvanceingStateManager(
        create_state_manager_fn(_EVAL_MANAGER_KEY)
    )
    self._evaluating_training_checkpoints = np.zeros([0], np.int32)
    self._evaluation_start_timestamp_seconds = np.zeros([0], np.int32)
    self._pending_tasks: set[asyncio.Task] = set()

  @property
  def data_source(self) -> federated_language.program.FederatedDataSource:
    """A data source used to create iterators each evaluation loop."""
    return self._data_source

  @property
  def aggregated_metrics_manager(
      self,
  ) -> Optional[
      federated_language.program.ReleaseManager[
          federated_language.program.ReleasableStructure, int
      ]
  ]:
    """A manager for releasing metrics at the end of each evaluation loop."""
    return self._aggregated_metrics_manager

  @property
  def create_state_manager_fn(
      self,
  ) -> Callable[[str], file_program_state_manager.FileProgramStateManager]:
    """A callable that returns a program state manager each evaluation loop."""
    return self._create_state_manager_fn

  @property
  def create_process_fn(
      self,
  ) -> Callable[
      [str],
      tuple[
          learning_process.LearningProcess,
          Optional[
              federated_language.program.ReleaseManager[
                  federated_language.program.ReleasableStructure, int
              ]
          ],
      ],
  ]:
    """A callable that returns a process and manager each evaluation loop."""
    return self._create_evaluation_process_fn

  @property
  def cohort_size(self) -> int:
    """The size of each evaluation round to select from the iterator."""
    return self._cohort_size

  @property
  def duration(self) -> datetime.timedelta:
    """The duration to run each evaluation loop."""
    return self._duration

  async def wait_for_evaluations_to_finish(self) -> None:
    """Creates an awaitable that blocks until all evaluations are finished."""
    if not self._pending_tasks:
      return
    done_tasks, self._pending_tasks = await asyncio.wait(
        self._pending_tasks, timeout=None, return_when=asyncio.ALL_COMPLETED
    )
    if self._pending_tasks:
      raise RuntimeError(
          'Error in Python runtime while waiting for tall tasks to complete" '
          'Expected all tasks to be complete, but wait() returned with still '
          'unfinished tasks.'
      )
    for task in done_tasks:
      task.result()  # Trigger any potentially stored exceptions.

  def _finalize_task(self, task: asyncio.Task):
    """Calls result() on tasks to ensure errors are propagated."""
    _logging.info('Finalizing task: %s', task)
    task.result()  # Trigger any potentially stored exceptions.
    self._pending_tasks.remove(task)

  async def resume_from_previous_state(self) -> bool:
    """Load the most recent state and restart in-progress evaluations.

    Returns:
      True if there was previous state to resume from, False otherwise.
    """
    loaded_state = await self._state_manager.load_latest((
        self._evaluating_training_checkpoints,
        self._evaluation_start_timestamp_seconds,
    ))
    if loaded_state is None:
      _logging.info('No previous evaluations found, nothing to resume.')
      return False
    (
        self._evaluating_training_checkpoints,
        self._evaluation_start_timestamp_seconds,
    ) = loaded_state
    train_round_nums = self._evaluating_training_checkpoints.tolist()
    _logging.info(
        'Resuming previous evaluations found for training rounds: %s',
        train_round_nums,
    )
    evaluation_end_times = [
        datetime.datetime.fromtimestamp(start_timestamp) + self._duration
        for start_timestamp in self._evaluation_start_timestamp_seconds
    ]
    for train_round_num, evaluation_end_time in zip(
        train_round_nums, evaluation_end_times
    ):
      evaluation_name = _EVAL_NAME_PATTERN.format(round_num=train_round_num)
      # Note: `start_evaluation` has already created the initial evaluation
      # state and set the training model weights, saving it to version 0 of the
      # state manager. This will resume from there.
      state_manager = self._create_state_manager_fn(evaluation_name)
      evaluation_process, metrics_manager = self._create_evaluation_process_fn(
          evaluation_name
      )
      self._start_evaluation_from_saved_model_weights(
          train_round_num,
          evaluation_process,
          metrics_manager,
          state_manager,
          evaluation_end_time,
      )
    return True

  def _start_evaluation_from_saved_model_weights(
      self,
      train_round_num: int,
      eval_process: learning_process.LearningProcess,
      per_round_metrics_manager: Optional[
          federated_language.program.ReleaseManager[
              federated_language.program.ReleasableStructure, int
          ]
      ],
      state_manager: file_program_state_manager.FileProgramStateManager,
      evaluation_end_time: datetime.datetime,
  ) -> None:
    """Starts an asyncio.Task, adding it to self._pending_tasks.

    Args:
      train_round_num: The training round that will be evaluated.
      eval_process: The evaluation logic that will iteratively evaluate the
        training output.
      per_round_metrics_manager: An optional release manager for releasing
        per-round metrics during evaluation.
      state_manager: The evaluation state manager. This *must* have already had
        version zero saved, which contains the training checkpoint to be
        evaluated.
      evaluation_end_time: The expected end time of the evaluation.
    """

    async def run_and_record_completion():
      # TODO: b/150782658 - re-enable pytype when fixed.
      await _run_evaluation(  # pytype: disable=bad-return-type
          train_round_num,
          state_manager,
          evaluation_process=eval_process,
          evaluation_name=_EVAL_NAME_PATTERN.format(round_num=train_round_num),
          evaluation_data_source=self._data_source,
          evaluation_per_round_clients_number=self._cohort_size,
          evaluation_end_time=evaluation_end_time,
          per_round_metrics_manager=per_round_metrics_manager,
          aggregated_metrics_manager=self._aggregated_metrics_manager,
      )
      await self.record_evaluations_finished(train_round_num)
      # Clean-up the statemanager output, which will no longer be used. This
      # must happy only after this evaluation has been removed from the
      # evaluation managers state during `record_evaluations_finished`.
      await state_manager.remove_all()

    new_task = asyncio.create_task(run_and_record_completion())
    new_task.add_done_callback(self._finalize_task)
    self._pending_tasks.add(new_task)

  async def start_evaluation(
      self,
      train_round: int,
      start_timestamp_seconds: int,
      model_weights: Union[
          model_weights_lib.ModelWeights,
          dict[str, model_weights_lib.ModelWeights],
      ],
  ) -> None:
    """Starts a new evaluation loop for the incoming model_weights."""
    self._evaluating_training_checkpoints = _append_integer(
        self._evaluating_training_checkpoints, train_round
    )
    self._evaluation_start_timestamp_seconds = _append_integer(
        self._evaluation_start_timestamp_seconds, start_timestamp_seconds
    )
    _logging.vlog(
        5,
        'In-flight evaluations: %s, %s',
        self._evaluating_training_checkpoints,
        self._evaluation_start_timestamp_seconds,
    )
    # Create and start an evaluation process. Here we create the initial state
    # and set the training model weights, then immediate save version 0 of the
    # state. This will allow resuming this evaluation even if the program
    # restarts before completing the first evaluation round.
    evaluation_name = _EVAL_NAME_PATTERN.format(round_num=train_round)
    state_manager = self._create_state_manager_fn(evaluation_name)
    evaluation_process, metrics_manager = self._create_evaluation_process_fn(
        evaluation_name
    )
    eval_state = await federated_language.program.materialize_value(
        evaluation_process.initialize()
    )
    eval_state = await federated_language.program.materialize_value(
        evaluation_process.set_model_weights(eval_state, model_weights)
    )
    await state_manager.save(eval_state, version=0)
    evaluation_end_time = (
        datetime.datetime.fromtimestamp(start_timestamp_seconds)
        + self._duration
    )
    self._start_evaluation_from_saved_model_weights(
        train_round,
        evaluation_process,
        metrics_manager,
        state_manager,
        evaluation_end_time,
    )
    # Record that the evaluation process has started.
    await self._state_manager.save(
        (
            self._evaluating_training_checkpoints,
            self._evaluation_start_timestamp_seconds,
        ),
    )

  async def record_evaluations_finished(self, train_round: int) -> None:
    """Removes evaluation for `train_round` from the internal state manager.

    Args:
      train_round: The integer round number of the training round that has
        finished evaluation.

    Raises:
      RuntimeError: If `train_round` was not currently being tracked as an
        in-progress evaluation.
    """
    try:
      (
          self._evaluating_training_checkpoints,
          self._evaluation_start_timestamp_seconds,
      ) = _pop_value(
          self._evaluating_training_checkpoints,
          self._evaluation_start_timestamp_seconds,
          train_round,
      )
    except ValueError as e:
      raise RuntimeError(
          'An internal error occurred where the EvaluationManager is trying to '
          f'record an evaluation for train round [{train_round}] finished '
          'but also has no state about in-progress evaluation rounds.'
      ) from e
    await self._state_manager.save(
        (
            self._evaluating_training_checkpoints,
            self._evaluation_start_timestamp_seconds,
        ),
    )


def extract_and_rewrap_metrics(
    metrics_structure: Mapping[str, Any],
    *,
    path: Sequence[str],
) -> Mapping[str, Any]:
  """Extracts a sub-structure and re-wraps it with a new prefix.

  This is used to normalize outputs from the training and evaluation
  computations. For example, we are interested in the following substructure
  from the evaluation and training tasks:

  ```
    client_work/eval/current_round_metrics/...
    client_work/eval/total_rounds_metrics/...
    client_work/train/...
  ```

  And we like to "re-home" these structures under:

  ```
    model_metrics/...
  ```

  while leaving all the other metrics alone. This can be thought of as a
  "subtree promotion" method.

  This is used for grouping metrics so that they appear aligned in
  http://tensorboard/ in a uniform way. TensorBoard uses the name up to the
  first `/` as a tab separator in the UI.

  Args:
    metrics_structure: The nested structure of tensor-like metric values.
    path: A sequence of strings that will be used to traverse the keys of
      `metrics_structure` and identify the substructure of interest.

  Returns:
    A structure of metrics of `metrics_substructure`.

  Raises:
    ValueError: If `path` is empty.
    KeyError: If any key in `path` sequence does not exist in the
      `metrics_structure`.
  """
  if not path:
    raise ValueError(
        '`path` is empty, must be a sequence of at least one element'
    )
  current_structure = typing.cast(
      MutableMapping[str, Any], metrics_structure.copy()
  )
  structure_copy = current_structure
  *path_parts, last_part = path
  for path_part in path_parts:
    part = current_structure.get(path_part)
    if part is None:
      raise KeyError(
          f'[{path_part}] of path {path} did not exist in structure: '
          f'{structure_copy}'
      )
    part = part.copy()
    current_structure[path_part] = part
    current_structure = part
  if (substructure := current_structure.get(last_part)) is None:
    raise KeyError(
        f'[{last_part}] of path {path} did not exist in structure: '
        f'{structure_copy}'
    )
  del current_structure[last_part]
  structure_copy[MODEL_METRICS_PREFIX] = substructure

  return structure_copy


def _extract_and_rewrap_metrics_for_multi_model_evaluation(
    metrics_structure: Mapping[str, Any],
    *,
    path_before_model_id: Sequence[str],
    path_after_model_id: Sequence[str],
    model_ids: Sequence[str],
) -> Mapping[str, Any]:
  """Extracts and re-wraps metrics for multi-model evaluation.

  This is used to normalize outputs from the multi-model evaluation
  computations. For example, we are interested in the following substructure
  from the evaluation task:

  ```
    client_work/a/eval/current_round_metrics/...
    client_work/b/eval/total_rounds_metrics/...
  ```

  And we like to "re-home" these structures under:

  ```
    model_metrics/a/...
    model_metrics/b/...
  ```

  while leaving all the other metrics alone. This can be thought of as a
  "subtree promotion" method.

  This is used for grouping metrics so that they appear aligned in
  http://tensorboard/ in a uniform way. TensorBoard uses the name up to the
  first `/` as a tab separator in the UI.

  Args:
    metrics_structure: The nested structure of tensor-like metric values.
    path_before_model_id: A sequence of strings that will be used to traverse
      the keys of `metrics_structure` and identify the substructure of interest
      before the model ID.
    path_after_model_id: A sequence of strings that will be used to traverse the
      keys of `metrics_structure` and identify the substructure of interest
      after the model ID.
    model_ids: A sequence of model IDs that will be used to extract the
      substructure

  Returns:
    A structure of metrics of `metrics_substructure`.

  Raises:
    KeyError: If any key in `path_before_model_id` or `path_after_model_id`
      sequence does not exist in the `metrics_structure`.
  """
  current_structure = typing.cast(
      MutableMapping[str, Any], metrics_structure.copy()
  )
  structure_copy = current_structure
  for model_id in model_ids:
    full_path = (
        list(path_before_model_id) + [model_id] + list(path_after_model_id)
    )
    *path_parts, last_part = full_path
    for path_part in path_parts:
      part = current_structure.get(path_part)
      if part is None:
        raise KeyError(
            f'[{path_part}] of path {full_path} did not exist in'
            f' structure: {structure_copy}'
        )
      part = part.copy()
      current_structure[path_part] = part
      current_structure = part
    if (substructure := current_structure.get(last_part)) is None:
      raise KeyError(
          f'[{last_part}] of path {full_path} did not exist in'
          f' structure: {structure_copy}'
      )
    del current_structure[last_part]
    if structure_copy.get(MODEL_METRICS_PREFIX) is None:
      structure_copy[MODEL_METRICS_PREFIX] = {}
    structure_copy[MODEL_METRICS_PREFIX][model_id] = substructure
    # Reset the current structure to the root of the structure.
    current_structure = structure_copy

  return structure_copy


def _get_model_ids_for_multi_model_evaluation(
    metrics_structure: Mapping[str, Any],
) -> Optional[Sequence[str]]:
  """Returns model IDs if the metrics structure is for multi-model evaluation.

  The model IDs are expected to be in the format of lowercase letters (i.e.,
  'a', 'b', 'c', etc.).

  Args:
    metrics_structure: The nested structure of tensor-like metric values.

  Returns:
    Model IDs if the metrics structure is for multi-model evaluation, otherwise
    `None`.
  """
  current_structure = metrics_structure
  for path_part in _MULTI_MODEL_EVAL_METRICS_PATH_COMPONENTS_BEFORE_MODEL_ID:
    part = current_structure.get(path_part)
    if part is None:
      return None
    current_structure = part
  for key in current_structure.keys():
    if key not in string.ascii_lowercase:
      return None
    return list(current_structure.keys())


async def _run_evaluation(
    train_round_num: int,
    state_manager: file_program_state_manager.FileProgramStateManager,
    evaluation_process: learning_process.LearningProcess,
    evaluation_name: str,
    evaluation_data_source: federated_language.program.FederatedDataSource,
    evaluation_per_round_clients_number: int,
    evaluation_end_time: datetime.datetime,
    per_round_metrics_manager: Optional[
        federated_language.program.ReleaseManager[
            federated_language.program.ReleasableStructure, int
        ]
    ],
    aggregated_metrics_manager: Optional[
        federated_language.program.ReleaseManager[
            federated_language.program.ReleasableStructure, int
        ]
    ],
) -> None:
  """Runs evaluation for one training state.

  Args:
    train_round_num: The round number of the training checkpoint that will be
      evaluated.
    state_manager: A `tff.program.FileProgramStateManager` that will manage the
      state of this evaluation. This will be used to resume evaluation if the
      evaluation loop is interrupted for any reason.
    evaluation_process: A `tff.learning.templates.LearningProcess` to invoke to
      evaluate the model produced after training. This process must have been
      created using `tff.learning.algorithms.build_fed_eval`.
    evaluation_name: A str name of the evaluation computation.
    evaluation_data_source: A `federated_language.program.FederatedDataSource`
      which returns client data used during evaluation.
    evaluation_per_round_clients_number: Number of clients to evaluate in each
      round.
    evaluation_end_time: Expected end time for running the evaluation. Multiple
      evaluation rounds will be run until the `evaluation_end_time` has reached.
      If the `evaluation_end_time` has passed, only one round will be run.
    per_round_metrics_manager: A `federated_language.program.ReleaseManager`
      that releases the per-round evaluation metrics from platform to user
      storage. Use a `tff.programs.GroupingReleaseManager` to utilize multiple
      release managers. If `None`, per-round metrics are not released.
    aggregated_metrics_manager: A `federated_language.program.ReleaseManager`
      that releases the evaluation metrics aggregated across the entire
      evaluation loop from platform to user storage. Use a
      `tff.programs.GroupingReleaseManager` to utilize multiple release
      managers. If `None`, aggregated evaluation metrics are not released.

  Raises:
    TypeError: If result of `evaluation_process` is not a value of
      `tff.learning.templates.LearningProcessOutput` type.
    ValueError: If no previous state found for evaluation.
  """
  federated_language.program.check_in_federated_context()

  evaluation_data_iterator = evaluation_data_source.iterator()

  async def invoke_evaluation(evaluation_state, eval_round_num):
    round_start = time.monotonic()
    _logging.info(
        'Starting evaluation of `%s`, round %d',
        evaluation_name,
        eval_round_num - train_round_num,
    )
    evaluation_data = evaluation_data_iterator.select(
        evaluation_per_round_clients_number
    )
    evaluation_result = await federated_language.program.materialize_value(
        evaluation_process.next(evaluation_state, evaluation_data)
    )
    if isinstance(evaluation_result, learning_process.LearningProcessOutput):
      evaluation_state = evaluation_result.state
      evaluation_metrics = evaluation_result.metrics
    else:
      raise TypeError(
          'FederatedContext returned unexpected result type after '
          'evaluation computation invocation. Expected a '
          '`tff.learning.templates.LearningProcessOutput`, got '
          f'{type(evaluation_result)}'
      )
    # Only output the `current_round_metrics` here. The total_rounds_metrics
    # will be output once at the end of the evaluation loop.
    if per_round_metrics_manager is not None:
      model_ids = _get_model_ids_for_multi_model_evaluation(evaluation_metrics)
      if model_ids is not None:
        current_round_eval_metrics = _extract_and_rewrap_metrics_for_multi_model_evaluation(
            evaluation_metrics,
            path_before_model_id=_MULTI_MODEL_EVAL_METRICS_PATH_COMPONENTS_BEFORE_MODEL_ID,
            path_after_model_id=_MULTI_MODEL_EVAL_METRICS_PATH_COMPONENTS_AFTER_MODEL_ID
            + ('current_round_metrics',),
            model_ids=model_ids,
        )
      else:
        current_round_eval_metrics = extract_and_rewrap_metrics(
            evaluation_metrics,
            path=_EVAL_METRICS_PATH_COMPONENTS + ('current_round_metrics',),
        )
      await per_round_metrics_manager.release(
          current_round_eval_metrics,
          key=eval_round_num,
      )
    elapsed_round_seconds = time.monotonic() - round_start
    _logging.info(
        'Finished evaluation of `%s`, round %d, duration %.2f seconds)',
        evaluation_name,
        eval_round_num - train_round_num,
        elapsed_round_seconds,
    )
    return evaluation_state, evaluation_metrics, eval_round_num + 1

  # Read the initial state from the manager. If this is the first evaluation,
  # the zeroth version should contain the initial state.
  evaluation_state, version = await state_manager.load_latest(
      await federated_language.program.materialize_value(
          evaluation_process.initialize()
      )
  )
  if evaluation_state is None:
    raise ValueError(
        'No previous state found for evaluation. Evaluations '
        'must previously have at least version 0 saved.'
    )

  # Set the eval_round_num to start from the train_round_num, so that in the
  # `step` view of TensorBoard the evaluation begins at the same point as the
  # training run. Then additionally at the `version`, which is `0` if evaluation
  # just started, or potentially higher if the statemanager previously loaded
  # a past state.
  eval_round_num = train_round_num + version
  # Run at least one evaluation round.
  evaluation_state, evaluation_metrics, eval_round_num = (
      await invoke_evaluation(evaluation_state, eval_round_num)
  )
  version += 1
  await state_manager.save(evaluation_state, version=version)
  # Now run evaluations over the entire evaluation period.
  while datetime.datetime.now() < evaluation_end_time:
    evaluation_state, evaluation_metrics, eval_round_num = (
        await invoke_evaluation(evaluation_state, eval_round_num)
    )
    version += 1
    await state_manager.save(evaluation_state, version=version)
  _logging.info(
      'Finished evaluation of %s at end time %s',
      evaluation_name,
      evaluation_end_time,
  )
  if aggregated_metrics_manager is not None:
    model_ids = _get_model_ids_for_multi_model_evaluation(evaluation_metrics)
    if model_ids is not None:
      total_rounds_eval_metrics = _extract_and_rewrap_metrics_for_multi_model_evaluation(
          evaluation_metrics,
          path_before_model_id=_MULTI_MODEL_EVAL_METRICS_PATH_COMPONENTS_BEFORE_MODEL_ID,
          path_after_model_id=_MULTI_MODEL_EVAL_METRICS_PATH_COMPONENTS_AFTER_MODEL_ID
          + ('total_rounds_metrics',),
          model_ids=model_ids,
      )
    else:
      total_rounds_eval_metrics = extract_and_rewrap_metrics(
          evaluation_metrics,
          path=_EVAL_METRICS_PATH_COMPONENTS + ('total_rounds_metrics',),
      )
    await aggregated_metrics_manager.release(
        total_rounds_eval_metrics,
        key=train_round_num,
    )
