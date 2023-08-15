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
"""An example of program logic to use in a federated program.

The program logic is abstracted into a separate function to illustrate the
boundary between the program and the program logic. Note the Python types of the
function signature, this program logic only depends on the abstract interfaces
defined by the TFF's federated program API and does not depend and the platform,
therefore this program logic is portable across platforms.

Note: This example focuses on the federated program API and does not use TFF's
domain specific APIs (e.g. `tff.learning`), though it is an example of a
federated learning training loop.
"""

import asyncio
import typing
from typing import NamedTuple, Optional

import tensorflow_federated as tff


class UnexpectedTypeSignatureError(Exception):
  pass


def _check_expected_type_signatures(
    *,
    initialize: tff.Computation,
    train: tff.Computation,
    train_data_source: tff.program.FederatedDataSource,
    evaluation: tff.Computation,
    evaluation_data_source: tff.program.FederatedDataSource,
) -> None:
  """Checks the computations and data sources for the expected type signatures.

  Note: These kind of checks may not be useful for all program logic. For
  example, if you are using a `tff.learning.templates.LearningProcess` as an
  input to the program logic, then these checks might not make sense because the
  the `tff.learning.templates.LearningProcess` has already validated that those
  `tff.Computation` have the expected type signatures.

  See `train_federated_model` for more information on the expected type
  signatures of the computations and data sources.

  Args:
    initialize: A `tff.Computation` to invoke before training.
    train: A `tff.Computation` to invoke during training.
    train_data_source: A `tff.program.FederatedDataSource` which returns client
      data used during training.
    evaluation: A `tff.Computation` to invoke to evaluate the model produced
      after training.
    evaluation_data_source: A `tff.program.FederatedDataSource` which returns
      client data used during evaluation.

  Raises:
    UnexpectedTypeSignatureError: If the computations or data sources have an
      unexpected type signature.
  """
  try:
    # Check initialize type.
    initialize.type_signature.check_function()

    # Check initialize parameter type.
    if initialize.type_signature.parameter is not None:
      raise UnexpectedTypeSignatureError(
          'Expected `initialize` to have no parameters, found '
          f'{initialize.type_signature.parameter}.'
      )

    # Check initialize result type.
    initialize.type_signature.result.check_federated()
    if initialize.type_signature.result.placement is not tff.SERVER:  # pytype: disable=attribute-error
      raise UnexpectedTypeSignatureError(
          'Expected the result of `initialize` to be placed at `tff.SERVER`, '
          f'found {initialize.type_signature.result.placement}.'  # pytype: disable=attribute-error
      )

    # Check train data source type.
    if train_data_source.federated_type.placement is not tff.CLIENTS:
      raise UnexpectedTypeSignatureError(
          'Expected the data returned by `train_data_source` to be placed at '
          '`tff.CLIENTS`, found '
          f'{train_data_source.federated_type.placement}.'
      )

    # Check train type.
    train.type_signature.check_function()

    # Check train result type.
    train.type_signature.result.check_struct()
    if len(train.type_signature.result) != 2:  # pytype: disable=wrong-arg-types
      raise UnexpectedTypeSignatureError(
          'Expected `train` to return two values, found '
          f'{train.type_signature.result}.'
      )
    train_result_state_type, train_result_metrics_type = (
        train.type_signature.result
    )  # pytype: disable=attribute-error

    # Check train result state type.
    train_result_state_type.check_federated()
    if train_result_state_type.placement is not tff.SERVER:
      raise UnexpectedTypeSignatureError(
          'Expected the first result of `train` to be placed at `tff.SERVER`, '
          f'found {train_result_state_type.placement}.'
      )

    # Check train result metrics type.
    train_result_metrics_type.check_federated()
    if train_result_metrics_type.placement is not tff.SERVER:
      raise UnexpectedTypeSignatureError(
          'Expected the second result of `train` to be placed at `tff.SERVER`, '
          f'found {train_result_metrics_type.placement}.'
      )

    # Check train parameter type.
    train.type_signature.parameter.check_struct()  # pytype: disable=attribute-error
    if len(train.type_signature.parameter) != 2:  # pytype: disable=wrong-arg-types
      raise UnexpectedTypeSignatureError(
          'Expected `train` to have two parameters, found '
          f'{train.type_signature.parameter}.'
      )
    train_parameter_state_type, train_parameter_client_data_type = (
        train.type_signature.parameter
    )  # pytype: disable=attribute-error

    # Check train parameter state type.
    train_parameter_state_type.check_federated()
    if train_parameter_state_type.placement is not tff.SERVER:
      raise UnexpectedTypeSignatureError(
          'Expected the first parameter of `train` to be placed at'
          f' `tff.SERVER`, found {train_parameter_state_type.placement}.'
      )
    train_parameter_state_type.check_assignable_from(
        initialize.type_signature.result
    )
    train_parameter_state_type.check_assignable_from(train_result_state_type)

    # Check train parameter client data type.
    train_parameter_client_data_type.check_federated()
    if train_parameter_client_data_type.placement is not tff.CLIENTS:
      raise UnexpectedTypeSignatureError(
          'Expected the second parameter of `train` to be placed at '
          f'`tff.CLIENTS`, found {train_parameter_client_data_type.placement}.'
      )
    train_parameter_client_data_type.check_assignable_from(
        train_data_source.federated_type
    )

    # Check evaluation data source type.
    if evaluation_data_source.federated_type.placement is not tff.CLIENTS:
      raise UnexpectedTypeSignatureError(
          'Expected the data returned by `evaluation_data_source` to be placed '
          'at `tff.CLIENTS`, found '
          f'{evaluation_data_source.federated_type.placement}.'
      )

    # Check evaluation type.
    evaluation.type_signature.check_function()

    # Check evaluation result type.
    evaluation.type_signature.result.check_federated()
    if evaluation.type_signature.result.placement is not tff.SERVER:  # pytype: disable=attribute-error
      raise UnexpectedTypeSignatureError(
          'Expected the result of `evaluation` to be placed at `tff.SERVER`, '
          f'found {evaluation.type_signature.result.placement}.'  # pytype: disable=attribute-error
      )

    # Check evaluation parameter type.
    evaluation.type_signature.parameter.check_struct()  # pytype: disable=attribute-error
    if len(evaluation.type_signature.parameter) != 2:  # pytype: disable=wrong-arg-types
      raise UnexpectedTypeSignatureError(
          'Expected `evaluation` to have two parameters, found '
          f'{evaluation.type_signature.parameter}.'
      )
    evaluation_parameter_state_type, evaluation_parameter_client_data_type = (
        evaluation.type_signature.parameter
    )  # pytype: disable=attribute-error

    # Check evaluation parameter state type.
    evaluation_parameter_state_type.check_federated()
    if evaluation_parameter_state_type.placement is not tff.SERVER:
      raise UnexpectedTypeSignatureError(
          'Expected the first parameter of `evaluation` to be placed at '
          f'`tff.SERVER`, found {evaluation_parameter_state_type.placement}.'
      )
    evaluation_parameter_state_type.check_assignable_from(
        train_result_state_type
    )

    # Check evaluation parameter client data type.
    evaluation_parameter_client_data_type.check_federated()
    if evaluation_parameter_client_data_type.placement is not tff.CLIENTS:
      raise UnexpectedTypeSignatureError(
          'Expected the second parameter of `evaluation` to be placed at '
          '`tff.CLIENTS`, found '
          f'{evaluation_parameter_client_data_type.placement}.'
      )
    evaluation_parameter_client_data_type.check_assignable_from(
        evaluation_data_source.federated_type
    )
  except TypeError as e:
    raise UnexpectedTypeSignatureError() from e


class _TaskGroup:
  """An asynchronous context manager holding a group of tasks.

  Tasks are used to schedule coroutines concurrently. Tasks can be added to the
  group using `_TaskGroup.create_task()`. All tasks are awaited when the context
  manager exits.

  This is a simplified version of
  [`asyncio.TaskGroup`](https://docs.python.org/3/library/asyncio-task.html#task-groups)
  with less sophisticated error handling. It can be removed once Python 3.11 is
  the minimum supported version of Python.
  """

  def __init__(self):
    self._tasks = set()

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc_value, traceback):
    if self._tasks:
      await asyncio.wait(self._tasks)

  def create_task(self, coro):
    task = asyncio.create_task(coro)
    self._tasks.add(task)

    def _task_done(task):
      self._tasks.discard(task)

    task.add_done_callback(_task_done)
    return task


class _ProgramState(NamedTuple):
  """Defines the intermediate program state of the program logic.

  The program logic is responsible for defining the data required to restore the
  execution of the program logic after a failure.

  Important: Updating the fields of this program state will impact the ability
  of the program logic to load previously saved program state. If this is
  required it may be useful to version the structure of the program state.

  Attributes:
    state: The server state produced at `round_num`.
    round_num: The training round.
  """

  state: object
  round_num: int


async def train_federated_model(
    *,
    initialize: tff.Computation,
    train: tff.Computation,
    train_data_source: tff.program.FederatedDataSource,
    evaluation: tff.Computation,
    evaluation_data_source: tff.program.FederatedDataSource,
    total_rounds: int,
    num_clients: int,
    train_metrics_manager: Optional[
        tff.program.ReleaseManager[tff.program.ReleasableStructure, int]
    ] = None,
    evaluation_metrics_manager: Optional[
        tff.program.ReleaseManager[tff.program.ReleasableStructure, int]
    ] = None,
    model_output_manager: Optional[
        tff.program.ReleaseManager[
            tff.program.ReleasableStructure, Optional[object]
        ]
    ] = None,
    program_state_manager: Optional[
        tff.program.ProgramStateManager[tff.program.ProgramStateStructure]
    ] = None,
) -> None:
  """Trains a federated model for some number of rounds.

  The following types signatures are required:

  1.  `initialize`: `( -> S@SERVER)`
  2.  `train`:      `(<S@SERVER, D1@CLIENTS> -> <S@SERVER, M1@SERVER>)`
  3.  `evaluation`: `(<S@SERVER, D2@CLIENTS> -> M2@SERVER)`

  And

  4.  `train_data_source`:      `D1@CLIENTS`
  5.  `evaluation_data_source`: `D2@CLIENTS`

  Where:

  *   `S`: The server state.
  *   `M1`: The train metrics.
  *   `M2`: The evaluation metrics.
  *   `D1`: The train client data.
  *   `D2`: The evaluation client data.

  Note: `S`, `D1`, and `D2` are only required to be assignable as described
  below, not necessarily identical.

  This function invokes `initialize` to construct a local `state` and then runs
  `total_rounds` rounds updating this `state`. At each round, this update occurs
  by invoking `train` with the `state` and the `client_data` selected from the
  `train_data_source`. Each round, the training metrics are released to the
  `train_metrics_managers` and the updated `state` used in the next round of
  training.

  *   Round 0 represents the initialized state
  *   Round 1 through `total_rounds` represent the training rounds

  After training, this function invokes `evaluation` once with the updated
  `state` and the `client_data` selected from the `evaluation_data_source`; and
  the evaluation metrics are released to the `evaluation_metrics_managers`.

  Finally, `state` is released to the `model_output_manager`.

  Args:
    initialize: A `tff.Computation` to invoke before training.
    train: A `tff.Computation` to invoke during training.
    train_data_source: A `tff.program.FederatedDataSource` which returns client
      data used during training.
    evaluation: A `tff.Computation` to invoke to evaluate the model produced
      after training.
    evaluation_data_source: A `tff.program.FederatedDataSource` which returns
      client data used during evaluation.
    total_rounds: The number of training rounds to run.
    num_clients: The number of clients for each round of training and for
      evaluation.
    train_metrics_manager: An optional `tff.program.ReleaseManager` used to
      release training metrics.
    evaluation_metrics_manager: An optional `tff.program.ReleaseManager` used to
      release evaluation metrics.
    model_output_manager: An optional `tff.program.ReleaseManager` used to
      release training output.
    program_state_manager: An optional `tff.program.ProgramStateManager` used to
      save program state for fault tolerance.
  """
  tff.program.check_in_federated_context()
  _check_expected_type_signatures(
      initialize=initialize,
      train=train,
      train_data_source=train_data_source,
      evaluation=evaluation,
      evaluation_data_source=evaluation_data_source,
  )

  # Cast the `program_state_manager` to a more specific type: a manager that
  # loads and saves `_ProgramState`s instead of a manager that loads and saves
  # `tff.program.ProgramStateStructure`s. This allows the program logic to:
  # *   Keep `_ProgramState` private.
  # *   Have static typing within the program logic.
  # *   Require callers to provide a `program_state_manager` capable of handling
  #     any `tff.program.ProgramStateStructure`.
  program_state_manager = typing.cast(
      Optional[tff.program.ProgramStateManager[_ProgramState]],
      program_state_manager,
  )

  initial_state = initialize()

  # Try to load the latest program state. If the program logic failed on a
  # previous run, this program state can be used to restore the execution of
  # this program logic and skip unnecessary steps.
  if program_state_manager is not None:
    initial_state = await tff.program.materialize_value(initial_state)
    structure = _ProgramState(initial_state, round_num=0)
    program_state, version = await program_state_manager.load_latest(structure)

    # TODO: b/271445312 - Cast `program_state` to `_ProgramState`. `TypeVar`s
    # are lost from async function signatures.
    program_state = typing.cast(_ProgramState, program_state)
  else:
    program_state = None
    version = 0

  # Assign the inputs to the program logic using the loaded program state if
  # available or the initialized state.
  if program_state is not None:
    state = program_state.state
    start_round = program_state.round_num + 1
  else:
    state = initial_state
    start_round = 1

  # Construct a async context manager to group and run tasks concurrently.
  # Program logic will release values and save program state, these functions
  # are asynchronous and can be run concurrently. However, it is possible to
  # schedule these functions differently using
  # [asyncio](https://docs.python.org/3/library/asyncio.html).
  async with _TaskGroup() as task_group:
    # Construct an iterator from the `train_data_source` which returns client
    # data used during training.
    train_data_iterator = train_data_source.iterator()

    # Train `state` for some number of rounds. Both `state` and `start_round`
    # are inputs to this loop and are saved using the `program_state_manager`.
    # This means that if there is a failure during training, previously trained
    # rounds will be skipped.
    for round_num in range(start_round, total_rounds + 1):

      # Run one round of training.
      train_data = train_data_iterator.select(num_clients)
      state, metrics = train(state, train_data)

      # Release the training metrics.
      if train_metrics_manager is not None:
        _, metrics_type = train.type_signature.result  # pytype: disable=attribute-error
        metrics_type = metrics_type.member
        task_group.create_task(
            train_metrics_manager.release(metrics, metrics_type, round_num)
        )

      # Save the current program state.
      if program_state_manager is not None:
        program_state = _ProgramState(state, round_num)
        version = version + 1
        task_group.create_task(
            program_state_manager.save(program_state, version)
        )

    # Run one round of evaluation. This is similar to running one round of
    # training above, except using the `evaluation` computation and the
    # `evaluation_data_source`.
    evaluation_data_iterator = evaluation_data_source.iterator()
    evaluation_data = evaluation_data_iterator.select(num_clients)
    evaluation_metrics = evaluation(state, evaluation_data)

    # Release the evaluation metrics.
    if evaluation_metrics_manager is not None:
      evaluation_metrics_type = evaluation.type_signature.result.member  # pytype: disable=attribute-error
      task_group.create_task(
          evaluation_metrics_manager.release(
              evaluation_metrics, evaluation_metrics_type, total_rounds + 1
          )
      )

    # Release the model output.
    if model_output_manager is not None:
      _, state_type = train.type_signature.result  # pytype: disable=attribute-error
      state_type = state_type.member
      task_group.create_task(
          model_output_manager.release(state, state_type, None)
      )
