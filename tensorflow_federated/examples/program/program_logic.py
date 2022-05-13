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
from typing import List, Optional

from absl import logging
import tensorflow_federated as tff


async def train_federated_model(
    initialize: tff.Computation,
    train: tff.Computation,
    train_data_source: tff.program.FederatedDataSource,
    evaluation: tff.Computation,
    evaluation_data_source: tff.program.FederatedDataSource,
    total_rounds: int,
    number_of_clients: int,
    train_output_managers: Optional[List[tff.program.ReleaseManager]] = None,
    evaluation_output_managers: Optional[List[
        tff.program.ReleaseManager]] = None,
    model_output_manager: Optional[tff.program.ReleaseManager] = None,
    program_state_manager: Optional[tff.program.ProgramStateManager] = None):
  """Trains a federated model for some number of rounds.

  The following `tff.Computation` types signatures are required:

  *   `initialize`: `( -> state)`.
  *   `train`:      `(<state, client_data> -> <state, metrics>)`
  *   `evaluation`: `(<state, client_data> -> metrics)`

  This function invokes `initialize` to construct a local `state` and then runs
  `total_rounds` rounds updating this `state`. At each round, this update occurs
  by invoking `train` with the `state` and the `client_data` selected from the
  `train_data_source`. Each round, the training metrics are released to the
  `train_output_managers` and the updated `state` used in the next round of
  training.

  *   Round 0 represents the initialized state
  *   Round 1 through `total_rounds` represent the training rounds

  After training, this function invokes `evaluation` once with the updated
  `state` and the `client_data` selected from the `evaluation_data_source`; and
  the evaluation metrics are released to the `evaluation_output_managers`.

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
    number_of_clients: The number of clients per round of training.
    train_output_managers: An optional list of `tff.program.ReleaseManagers`s
      used to release training output.
    evaluation_output_managers: An optional list of
      `tff.program.ReleaseManagers`s used to release evaluation output.
    model_output_manager: An optional `tff.program.ReleaseManagers`s used to
      release training output.
    program_state_manager: An optional `tff.program.ProgramStateManager` used to
      save program state for fault tolerance.
  """
  tff.program.check_in_federated_context()
  logging.info('Running program logic')

  # Try to load the latest program state; if the program logic failed on a
  # previous run, this program state can be used to restore the execution of
  # this program logic and skip unnecessary steps.
  if program_state_manager is not None:
    structure = initialize()
    program_state, version = await program_state_manager.load_latest(structure)
  else:
    program_state = None

  # Initialize or load `state` and the inputs to the program logic.
  if program_state is not None:
    logging.info('Loaded program state at version %d', version)
    # Unpack the program state; the program logic is responsible for determining
    # how to pack and unpack program state and these functions are dependent on
    # eachother. In this example the logic is simple, the unpacking logic is
    # inlined here and the packing logic is inlined below. If the logic is more
    # complicated it may be helpful to express these as dedicated functions.
    state, start_round = program_state
  else:
    logging.info('Initializing state')
    state = initialize()
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
      train_data = train_data_iterator.select(number_of_clients)
      state, metrics = train(state, train_data)

      # Release the training metrics.
      if train_output_managers is not None:
        tasks.add_all(
            *[m.release(metrics, round_number) for m in train_output_managers])

      # Save the current program state.
      if program_state_manager is not None:
        # Pack the program state; the program logic should save only what is
        # required to restore the exection of this program logic after a
        # failure.
        program_state = (state, start_round)
        tasks.add(program_state_manager.save(program_state, round_number))

    # Run one round of evaluation; similar to running one round of training
    # above, except using the `evaluation` computaiton and the
    # `evaluation_data_source`.
    tasks.add_callable(
        functools.partial(logging.info, 'Running one round of evaluation'))
    evaluation_data_iterator = evaluation_data_source.iterator()
    evaluation_data = evaluation_data_iterator.select(number_of_clients)
    evaluation_metrics = evaluation(state, evaluation_data)

    # Release the evaluation metrics.
    if evaluation_output_managers is not None:
      tasks.add_all(*[
          m.release(evaluation_metrics, round_number)
          for m in train_output_managers
      ])

    # Release the model output.
    if model_output_manager is not None:
      tasks.add(model_output_manager.release(state))
