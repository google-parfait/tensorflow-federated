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
"""

import functools
from typing import Any, Optional

from absl import logging
import tensorflow_federated as tff


class TrainFederatedModelUnexpectedTypeSingatureError(Exception):
  pass


def _check_expected_type_signatures(
    initialize: tff.Computation, train: tff.Computation,
    train_data_source: tff.program.FederatedDataSource,
    evaluation: tff.Computation,
    evaluation_data_source: tff.program.FederatedDataSource) -> None:
  """Checks the computations and data sources for the expected type signatures.

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
    TrainFederatedModelUnexpectedTypeSingatureError: If the computations or data
      sources have an unexpected type signature.
  """
  try:
    # Check initialize type.
    initialize.type_signature.check_function()

    # Check initialize parameter type.
    if initialize.type_signature.parameter is not None:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected `initialize` to have no parameters, found '
          f'{initialize.type_signature.parameter}.')

    # Check initialize result type.
    initialize.type_signature.result.check_federated()
    if initialize.type_signature.result.placement is not tff.SERVER:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected the result of `initialize` to be placed at `tff.SERVER`, '
          f'found {initialize.type_signature.result.placement}.')

    # Check train data source type.
    if train_data_source.federated_type.placement is not tff.CLIENTS:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected the data returned by `train_data_source` to be placed at '
          '`tff.CLIENTS`, found '
          f'{train_data_source.federated_type.placement}.')

    # Check train type.
    train.type_signature.check_function()

    # Check train result type.
    train.type_signature.result.check_struct()
    if len(train.type_signature.result) != 2:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected `train` to return two values, found '
          f'{train.type_signature.result}.')
    train_result_state_type, train_result_metrics_type = train.type_signature.result

    # Check train result state type.
    train_result_state_type.check_federated()
    if train_result_state_type.placement is not tff.SERVER:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected the first result of `train` to be placed at `tff.SERVER`, '
          f'found {train_result_state_type.placement}.')

    # Check train result metrics type.
    train_result_metrics_type.check_federated()
    if train_result_metrics_type.placement is not tff.SERVER:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected the second result of `train` to be placed at `tff.SERVER`, '
          f'found {train_result_metrics_type.placement}.')
    train_result_metrics_type.member.check_struct()

    # Check train parameter type.
    train.type_signature.parameter.check_struct()
    if len(train.type_signature.parameter) != 2:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected `train` to have two parameters, found '
          f'{train.type_signature.parameter}.')
    train_parameter_state_type, train_parameter_client_data_type = train.type_signature.parameter

    # Check train parameter state type.
    train_parameter_state_type.check_federated()
    if train_parameter_state_type.placement is not tff.SERVER:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected the first parameter of `train` to be placed at `tff.SERVER`, '
          f'found {train_parameter_state_type.placement}.')
    train_parameter_state_type.check_assignable_from(
        initialize.type_signature.result)
    train_parameter_state_type.check_assignable_from(train_result_state_type)

    # Check train parameter client data type.
    train_parameter_client_data_type.check_federated()
    if train_parameter_client_data_type.placement is not tff.CLIENTS:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected the second parameter of `train` to be placed at '
          f'`tff.CLIENTS`, found {train_parameter_client_data_type.placement}.')
    train_parameter_client_data_type.check_assignable_from(
        train_data_source.federated_type)

    # Check evaluation data source type.
    if evaluation_data_source.federated_type.placement is not tff.CLIENTS:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected the data returned by `evaluation_data_source` to be placed '
          'at `tff.CLIENTS`, found '
          f'{evaluation_data_source.federated_type.placement}.')

    # Check evaluation type.
    evaluation.type_signature.check_function()

    # Check evaluation result type.
    evaluation.type_signature.result.check_federated()
    if evaluation.type_signature.result.placement is not tff.SERVER:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected the result of `evaluation` to be placed at `tff.SERVER`, '
          f'found {evaluation.type_signature.result.placement}.')
    evaluation.type_signature.result.member.check_struct()

    # Check evaluation parameter type.
    evaluation.type_signature.parameter.check_struct()
    if len(evaluation.type_signature.parameter) != 2:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected `evaluation` to have two parameters, found '
          f'{evaluation.type_signature.parameter}.')
    evaluation_parameter_state_type, evaluation_parameter_client_data_type = evaluation.type_signature.parameter

    # Check evaluation parameter state type.
    evaluation_parameter_state_type.check_federated()
    if evaluation_parameter_state_type.placement is not tff.SERVER:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected the first parameter of `evaluation` to be placed at '
          f'`tff.SERVER`, found {evaluation_parameter_state_type.placement}.')
    evaluation_parameter_state_type.check_assignable_from(
        train_result_state_type)

    # Check evaluation parameter client data type.
    evaluation_parameter_client_data_type.check_federated()
    if evaluation_parameter_client_data_type.placement is not tff.CLIENTS:
      raise TrainFederatedModelUnexpectedTypeSingatureError(
          'Expected the second parameter of `evaluation` to be placed at '
          '`tff.CLIENTS`, found '
          f'{evaluation_parameter_client_data_type.placement}.')
    evaluation_parameter_client_data_type.check_assignable_from(
        evaluation_data_source.federated_type)
  except TypeError as e:
    raise TrainFederatedModelUnexpectedTypeSingatureError() from e


async def train_federated_model(
    initialize: tff.Computation,
    train: tff.Computation,
    train_data_source: tff.program.FederatedDataSource,
    evaluation: tff.Computation,
    evaluation_data_source: tff.program.FederatedDataSource,
    total_rounds: int,
    num_clients: int,
    train_metrics_manager: Optional[tff.program.ReleaseManager[
        tff.program.ReleasableStructure, int]] = None,
    evaluation_metrics_manager: Optional[tff.program.ReleaseManager[
        tff.program.ReleasableStructure, int]] = None,
    model_output_manager: Optional[tff.program.ReleaseManager[
        tff.program.ReleasableStructure, Any]] = None,
    program_state_manager: Optional[tff.program.ProgramStateManager[
        tff.program.ProgramStateStructure]] = None
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
  below, not necessarily identical

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
    num_clients: The number of clients per round of training.
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
  _check_expected_type_signatures(initialize, train, train_data_source,
                                  evaluation, evaluation_data_source)
  logging.info('Running program logic')

  initial_state = initialize()

  # Try to load the latest program state; if the program logic failed on a
  # previous run, this program state can be used to restore the execution of
  # this program logic and skip unnecessary steps.
  if program_state_manager is not None:
    initial_state = await tff.program.materialize_value(initial_state)
    structure = initial_state, 0
    program_state, version = await program_state_manager.load_latest(structure)
  else:
    program_state = None

  # Assign the inputs to the program logic using the loaded program state if
  # avilable or the initialized state.
  if program_state is not None:
    logging.info('Loaded program state at version %d', version)
    # Unpack the program state; the program logic is responsible for determining
    # how to pack and unpack program state and these functions are dependent on
    # eachother. In this example the logic is simple, the unpacking logic is
    # inlined here and the packing logic is inlined below. If the logic is more
    # complicated it may be helpful to express these as dedicated functions.
    state, round_number = program_state
    start_round = round_number + 1
  else:
    logging.info('Initialized state')
    state = initial_state
    start_round = 1

  # Construct a context manager to group and run tasks sequentially; often
  # program logic will release values and save program state, asynchronous and
  # synchronous functions (e.g. logging) can be executed sequentially using a
  # `tff.async_utils.OrderedTasks` context manager.
  async with tff.async_utils.ordered_tasks() as tasks:

    # Construct an iterator from the `train_data_source` which returns client
    # data used during training.
    train_data_iterator = train_data_source.iterator()

    # Train `state` for some number of rounds; note that both `state` and
    # `start_round` are inputs to this loop and are saved using the
    # `program_state_manager`. This means that if there is a failure during
    # training, previously trained rounds will be skipped.
    for round_number in range(start_round, total_rounds + 1):
      # Run one round of training.
      tasks.add_callable(
          functools.partial(logging.info, 'Running round %d of training',
                            round_number))
      train_data = train_data_iterator.select(num_clients)
      state, metrics = train(state, train_data)

      # Release the training metrics.
      if train_metrics_manager is not None:
        _, metrics_type = train.type_signature.result
        metrics_type = metrics_type.member
        tasks.add(
            train_metrics_manager.release(metrics, metrics_type, round_number))

      # Save the current program state.
      if program_state_manager is not None:
        # Pack the program state; the program logic should save only what is
        # required to restore the exection of this program logic after a
        # failure.
        program_state = (state, round_number)
        version = round_number
        tasks.add(program_state_manager.save(program_state, version))

    # Run one round of evaluation; similar to running one round of training
    # above, except using the `evaluation` computaiton and the
    # `evaluation_data_source`.
    tasks.add_callable(
        functools.partial(logging.info, 'Running one round of evaluation'))
    evaluation_data_iterator = evaluation_data_source.iterator()
    evaluation_data = evaluation_data_iterator.select(num_clients)
    evaluation_metrics = evaluation(state, evaluation_data)

    # Release the evaluation metrics.
    if evaluation_metrics_manager is not None:
      evaluation_metrics_type = evaluation.type_signature.result.member
      tasks.add(
          evaluation_metrics_manager.release(evaluation_metrics,
                                             evaluation_metrics_type,
                                             total_rounds + 1))

    # Release the model output.
    if model_output_manager is not None:
      _, state_type = train.type_signature.result
      state_type = state_type.member
      tasks.add(model_output_manager.release(state, state_type, None))
