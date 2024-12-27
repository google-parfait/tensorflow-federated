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
defined by the TFF's federated program API and does not depend on the platform,
therefore this program logic is portable across platforms.

Note: This example focuses on the federated program API and does not use TFF's
domain specific APIs (e.g. `tff.learning`), though it is an example of a
federated learning training loop.
"""

import asyncio
import typing
from typing import NamedTuple, Optional

import federated_language
import tensorflow_federated as tff


class UnexpectedTypeSignatureError(Exception):
  pass


def _check_expected_type_signatures(
    *,
    initialize: tff.Computation,
    train: tff.Computation,
    train_data_source: federated_language.program.FederatedDataSource,
    evaluation: tff.Computation,
    evaluation_data_source: federated_language.program.FederatedDataSource,
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
    train_data_source: A `federated_language.program.FederatedDataSource` which
      returns client data used during training.
    evaluation: A `tff.Computation` to invoke to evaluate the model produced
      after training.
    evaluation_data_source: A `federated_language.program.FederatedDataSource`
      which returns client data used during evaluation.

  Raises:
    UnexpectedTypeSignatureError: If the computations or data sources have an
      unexpected type signature.
  """

  # Check initialize type.
  if not isinstance(initialize.type_signature, federated_language.FunctionType):
    raise UnexpectedTypeSignatureError(
        'Expected `initialize` to be a `federated_language.FunctionType`,'
        f' found {initialize.type_signature}.'
    )

  # Check initialize parameter type.
  if initialize.type_signature.parameter is not None:
    raise UnexpectedTypeSignatureError(
        'Expected `initialize` to have no parameters, found '
        f'{initialize.type_signature.parameter}.'
    )

  # Check initialize result type.
  if (
      not isinstance(
          initialize.type_signature.result, federated_language.FederatedType
      )
      or initialize.type_signature.result.placement
      is not federated_language.SERVER
  ):
    raise UnexpectedTypeSignatureError(
        'Expected `initialize` to return a `federated_language.FederatedType`'
        ' placed at `federated_language.SERVER, found'
        f' {initialize.type_signature.result}.'
    )

  # Check train data source type.
  if (
      train_data_source.federated_type.placement
      is not federated_language.CLIENTS
  ):
    raise UnexpectedTypeSignatureError(
        'Expected `train_data_source` to yield data placed at'
        ' `federated_language.CLIENTS`, found'
        f' {train_data_source.federated_type.placement}.'
    )

  # Check train type.
  if not isinstance(train.type_signature, federated_language.FunctionType):
    raise UnexpectedTypeSignatureError(
        'Expected `train` to be a `federated_language.FunctionType`, found '
        f'{train.type_signature}.'
    )

  # Check train result type.
  if (
      not isinstance(train.type_signature.result, federated_language.StructType)
      or len(train.type_signature.result) != 2
  ):
    raise UnexpectedTypeSignatureError(
        'Expected `train` to return two results, found '
        f'{train.type_signature.result}.'
    )
  train_result_state_type, train_result_metrics_type = (
      train.type_signature.result
  )

  # Check train result state type.
  if (
      not isinstance(train_result_state_type, federated_language.FederatedType)
      or train_result_state_type.placement is not federated_language.SERVER
  ):
    raise UnexpectedTypeSignatureError(
        'Expected the first result of `train` to be a'
        ' `federated_language.FederatedType` placed at'
        f' `federated_language.SERVER, found {train_result_state_type}.'
    )

  # Check train result metrics type.
  if (
      not isinstance(
          train_result_metrics_type, federated_language.FederatedType
      )
      or train_result_metrics_type.placement is not federated_language.SERVER
  ):
    raise UnexpectedTypeSignatureError(
        'Expected the second result of `train` to be a'
        ' `federated_language.FederatedType` placed at'
        f' `federated_language.SERVER, found {train_result_metrics_type}.'
    )

  # Check train parameter type.
  if (
      not isinstance(
          train.type_signature.parameter, federated_language.StructType
      )
      or len(train.type_signature.parameter) != 2
  ):
    raise UnexpectedTypeSignatureError(
        'Expected `train` to have two parameters, found '
        f'{train.type_signature.parameter}.'
    )
  train_parameter_state_type, train_parameter_client_data_type = (
      train.type_signature.parameter
  )

  # Check train parameter state type.
  if (
      not isinstance(
          train_parameter_state_type, federated_language.FederatedType
      )
      or train_parameter_state_type.placement is not federated_language.SERVER
  ):
    raise UnexpectedTypeSignatureError(
        'Expected the first parameter of `train` to be a'
        ' `federated_language.FederatedType` placed at'
        f' `federated_language.SERVER, found {train_parameter_state_type}.'
    )
  if not train_parameter_state_type.is_assignable_from(
      initialize.type_signature.result
  ):
    raise UnexpectedTypeSignatureError(
        'Expected the first parameter of `train` to be assignable from the '
        'result of `initialize`.\n '
        f'The first parameter of `train:` {train_parameter_state_type}\n'
        f'The result of `initialize:` {train_data_source.federated_type}\n'
    )
  if not train_parameter_state_type.is_assignable_from(train_result_state_type):
    raise UnexpectedTypeSignatureError(
        'Expected the first parameter of `train` to be assignable from the '
        'first result of `train`.\n'
        f'The first parameter of `train:` {train_parameter_state_type}\n'
        f'The first result of `train:` {train_result_state_type}\n'
    )

  # Check train parameter client data type.
  if (
      not isinstance(
          train_parameter_client_data_type, federated_language.FederatedType
      )
      or train_parameter_client_data_type.placement
      is not federated_language.CLIENTS
  ):
    raise UnexpectedTypeSignatureError(
        'Expected the second parameter of `train` to be a'
        ' `federated_language.FederatedType` placed at'
        ' `federated_language.CLIENTS, found'
        f' {train_parameter_client_data_type}.'
    )
  if not train_parameter_client_data_type.is_assignable_from(
      train_data_source.federated_type
  ):
    raise UnexpectedTypeSignatureError(
        'Expected the second parameter of `train` to be assignable from the '
        'data yielded from `train_data_source`.\n'
        f'The second parameter of `train:` {train_parameter_client_data_type}\n'
        'The data yielded from `tratrain_data_sourcein:` '
        f'{train_data_source.federated_type}\n'
    )

  # Check evaluation data source type.
  if (
      evaluation_data_source.federated_type.placement
      is not federated_language.CLIENTS
  ):
    raise UnexpectedTypeSignatureError(
        'Expected `evaluation_data_source` to yield data placed at '
        '`federated_language.CLIENTS`, found'
        f' {evaluation_data_source.federated_type.placement}.'
    )

  # Check evaluation type.
  if not isinstance(evaluation.type_signature, federated_language.FunctionType):
    raise UnexpectedTypeSignatureError(
        'Expected `evaluation` to be a `federated_language.FunctionType`,'
        f' found {evaluation.type_signature}.'
    )

  # Check evaluation parameter type.
  if (
      not isinstance(
          evaluation.type_signature.parameter, federated_language.StructType
      )
      or len(evaluation.type_signature.parameter) != 2
  ):
    raise UnexpectedTypeSignatureError(
        'Expected `evaluation` to have two parameters, found '
        f'{evaluation.type_signature.parameter}.'
    )
  evaluation_parameter_state_type, evaluation_parameter_client_data_type = (
      evaluation.type_signature.parameter
  )

  # Check evaluation parameter state type.
  if (
      not isinstance(
          evaluation_parameter_state_type, federated_language.FederatedType
      )
      or evaluation_parameter_state_type.placement
      is not federated_language.SERVER
  ):
    raise UnexpectedTypeSignatureError(
        'Expected the first parameter of `evaluation` to be a'
        ' `federated_language.FederatedType` placed at'
        f' `federated_language.SERVER, found {evaluation_parameter_state_type}.'
    )
  if not evaluation_parameter_state_type.is_assignable_from(
      train_result_state_type
  ):
    raise UnexpectedTypeSignatureError(
        'Expected the first parameter of `evaluation` to be assignable from '
        'the first result of `train`.\n'
        'The first parameter of `evaluation:` '
        f'{evaluation_parameter_state_type}\n'
        f'The first result of `train:` {train_result_state_type}\n'
    )

  # Check evaluation parameter client data type.
  if (
      not isinstance(
          evaluation_parameter_client_data_type,
          federated_language.FederatedType,
      )
      or evaluation_parameter_client_data_type.placement
      is not federated_language.CLIENTS
  ):
    raise UnexpectedTypeSignatureError(
        'Expected the second parameter of `evaluation` to be a'
        ' `federated_language.FederatedType` placed at'
        ' `federated_language.CLIENTS, found'
        f' {evaluation_parameter_client_data_type}.'
    )
  if not evaluation_parameter_client_data_type.is_assignable_from(
      evaluation_data_source.federated_type
  ):
    raise UnexpectedTypeSignatureError(
        'Expected the second parameter of `evaluation` to be assignable from '
        'the data yielded from `evaluation_data_source`.\n'
        'The second parameter of `evaluation:` '
        f'{evaluation_parameter_client_data_type}\n'
        'The data yielded from `evaluation_data_source:` '
        f'{evaluation_data_source.federated_type}\n'
    )

  # Check evaluation result type.
  if (
      not isinstance(
          evaluation.type_signature.result, federated_language.FederatedType
      )
      or evaluation.type_signature.result.placement
      is not federated_language.SERVER
  ):
    raise UnexpectedTypeSignatureError(
        'Expected the result of `train` to be a'
        ' `federated_language.FederatedType` placed at'
        ' `federated_language.SERVER, found'
        f' {evaluation.type_signature.result}.'
    )


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
    iterator: The training
      `federated_language.program.FederatedDataSourceIterator`.
  """

  state: object
  round_num: int
  iterator: federated_language.program.FederatedDataSourceIterator


async def train_federated_model(
    *,
    initialize: tff.Computation,
    train: tff.Computation,
    train_data_source: federated_language.program.FederatedDataSource,
    evaluation: tff.Computation,
    evaluation_data_source: federated_language.program.FederatedDataSource,
    total_rounds: int,
    num_clients: int,
    train_metrics_manager: Optional[
        federated_language.program.ReleaseManager[
            federated_language.program.ReleasableStructure, int
        ]
    ] = None,
    evaluation_metrics_manager: Optional[
        federated_language.program.ReleaseManager[
            federated_language.program.ReleasableStructure, int
        ]
    ] = None,
    model_output_manager: Optional[
        federated_language.program.ReleaseManager[
            federated_language.program.ReleasableStructure, Optional[object]
        ]
    ] = None,
    program_state_manager: Optional[
        federated_language.program.ProgramStateManager[
            federated_language.program.ProgramStateStructure
        ]
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
    train_data_source: A `federated_language.program.FederatedDataSource` which
      returns client data used during training.
    evaluation: A `tff.Computation` to invoke to evaluate the model produced
      after training.
    evaluation_data_source: A `federated_language.program.FederatedDataSource`
      which returns client data used during evaluation.
    total_rounds: The number of training rounds to run.
    num_clients: The number of clients for each round of training and for
      evaluation.
    train_metrics_manager: An optional
      `federated_language.program.ReleaseManager` used to release training
      metrics.
    evaluation_metrics_manager: An optional
      `federated_language.program.ReleaseManager` used to release evaluation
      metrics.
    model_output_manager: An optional
      `federated_language.program.ReleaseManager` used to release training
      output.
    program_state_manager: An optional
      `federated_language.program.ProgramStateManager` used to save program
      state for fault tolerance.
  """
  federated_language.program.check_in_federated_context()
  _check_expected_type_signatures(
      initialize=initialize,
      train=train,
      train_data_source=train_data_source,
      evaluation=evaluation,
      evaluation_data_source=evaluation_data_source,
  )

  # Cast the `program_state_manager` to a more specific type: a manager that
  # loads and saves `_ProgramState`s instead of a manager that loads and saves
  # `federated_language.program.ProgramStateStructure`s. This allows the program
  # logic to:
  # *   Keep `_ProgramState` private.
  # *   Have static typing within the program logic.
  # *   Require callers to provide a `program_state_manager` capable of handling
  #     any `federated_language.program.ProgramStateStructure`.
  program_state_manager = typing.cast(
      Optional[federated_language.program.ProgramStateManager[_ProgramState]],
      program_state_manager,
  )

  # Initialize the inputs of the program logic.
  state = initialize()
  start_round = 1
  train_data_iterator = train_data_source.iterator()

  # Try to load the latest program state. If the program logic failed on a
  # previous run, this program state can be used to restore the execution of
  # this program logic and skip unnecessary steps.
  if program_state_manager is not None:
    state = await federated_language.program.materialize_value(state)
    structure = _ProgramState(
        state=state,
        round_num=0,
        iterator=train_data_iterator,
    )
    program_state, version = await program_state_manager.load_latest(structure)

    # TODO: b/271445312 - Cast `program_state` to `_ProgramState`. `TypeVar`s
    # are lost from async function signatures.
    program_state = typing.cast(_ProgramState, program_state)
  else:
    program_state = None
    version = 0

  # Assign the inputs of the program logic using the loaded program state if
  # available.
  if program_state is not None:
    state = program_state.state
    start_round = program_state.round_num + 1
    train_data_iterator = program_state.iterator

  # Construct a async context manager to group and run tasks concurrently.
  # Program logic will release values and save program state, these functions
  # are asynchronous and can be run concurrently. However, it is possible to
  # schedule these functions differently using
  # [asyncio](https://docs.python.org/3/library/asyncio.html).
  async with _TaskGroup() as task_group:

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
        task_group.create_task(
            train_metrics_manager.release(metrics, key=round_num)
        )

      # Save the current program state.
      if program_state_manager is not None:
        program_state = _ProgramState(
            state=state,
            round_num=round_num,
            iterator=train_data_iterator,
        )
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
      task_group.create_task(
          evaluation_metrics_manager.release(
              evaluation_metrics, key=total_rounds + 1
          )
      )

    # Release the model output.
    if model_output_manager is not None:
      task_group.create_task(model_output_manager.release(state, key=None))
