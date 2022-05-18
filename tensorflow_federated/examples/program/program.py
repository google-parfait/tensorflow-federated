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
"""An example of a federated program using TFFs native platform.

This federated program, from the perspective of the components:

*   executes the computations defined by `computations`
*   using program logic defined by this example
*   with the platform-specific components provided by `tff.program`
*   and platform-agnostic components provided by `tff.program`
*   given parameterizations set by the program
*   and parameterizations set by the customer
*   when the customer runs the program in this example
*   and materializes data in platform storage to implement fault tolerance
*   and release data to customer storage as metrics

From the perspective of roles/ownership:

*   You are the customer running the program
*   TFF is the platform
*   TFF is the library
*   This example is the author of the program and the program logic

Please read
https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/program/README.md for
more documentation about this example.
"""

import asyncio
import os.path
from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.examples.program import computations
from tensorflow_federated.examples.program import program_logic

_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'The output path.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  total_rounds = 10
  number_of_clients = 3

  # Configure the platform-specific components; in this example, the TFF native
  # platform is used, but this example could use any platform that conforms to
  # the approprate abstract interfaces.

  # Create a context in which to execute the program logic.
  context = tff.google.backends.native.create_local_async_cpp_execution_context(
  )
  context = tff.program.NativeFederatedContext(context)
  tff.framework.set_default_context(context)

  # Create data sources that are compatible with the context and computations.
  to_int32 = lambda x: tf.cast(x, tf.int32)
  datasets = [tf.data.Dataset.range(10).map(to_int32)] * 3
  train_data_source = tff.program.DatasetDataSource(datasets)
  evaluation_data_source = tff.program.DatasetDataSource(datasets)

  # Create computations that are compatible with the context and data sources.
  initialize = computations.initialize
  train = computations.train
  evaluation = computations.evaluation

  # Configure the platform-agnostic components.

  # Create release managers with access to customer storage.
  train_output_managers = [tff.program.LoggingReleaseManager()]
  evaluation_output_managers = [tff.program.LoggingReleaseManager()]
  model_output_manager = tff.program.LoggingReleaseManager()

  if _OUTPUT_DIR.value is not None:
    summary_dir = os.path.join(_OUTPUT_DIR.value, 'summary')
    tensorboard_manager = tff.program.TensorBoardReleaseManager(summary_dir)
    train_output_managers.append(tensorboard_manager)

    csv_path = os.path.join(_OUTPUT_DIR.value, 'evaluation_metrics.csv')
    csv_manager = tff.program.CSVFileReleaseManager(csv_path)
    evaluation_output_managers.append(csv_manager)

  # Create a program state manager with access to platform storage.
  program_state_manager = None

  if _OUTPUT_DIR.value is not None:
    program_state_dir = os.path.join(_OUTPUT_DIR.value, 'program_state')
    program_state_manager = tff.program.FileProgramStateManager(
        program_state_dir)

  # Execute the program logic; the program logic is abstracted into a separate
  # function to illustrate the boundary between the program and the program
  # logic. This program logic is declared as an async def, and needs to be
  # executed in an asyncio event loop.
  asyncio.run(
      program_logic.train_federated_model(
          initialize=initialize,
          train=train,
          train_data_source=train_data_source,
          evaluation=evaluation,
          evaluation_data_source=evaluation_data_source,
          total_rounds=total_rounds,
          number_of_clients=number_of_clients,
          train_output_managers=train_output_managers,
          evaluation_output_managers=evaluation_output_managers,
          model_output_manager=model_output_manager,
          program_state_manager=program_state_manager))


if __name__ == '__main__':
  app.run(main)
