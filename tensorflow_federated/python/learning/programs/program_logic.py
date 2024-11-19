# Copyright 2023, The TensorFlow Federated Authors.
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
"""An example of program logic to use in a federated program."""

import datetime
import typing
from typing import Optional, Protocol, Union

import federated_language

from tensorflow_federated.python.learning.programs import evaluation_program_logic
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import learning_process


@typing.runtime_checkable
class TrainModelProgramLogic(Protocol):
  """Defines the API for some basic program logic.

  The purpose of defining this API is so that federated programs and program
  logic can compose program logic that conforms with this protocol with other
  program logic.
  """

  async def __call__(
      self,
      *,
      train_process: learning_process.LearningProcess,
      initial_train_state: composers.LearningAlgorithmState,
      train_data_source: federated_language.program.FederatedDataSource,
      train_per_round_clients: int,
      train_total_rounds: int,
      program_state_manager: federated_language.program.ProgramStateManager,
      model_output_manager: federated_language.program.ReleaseManager[
          federated_language.program.ReleasableStructure, str
      ],
      train_metrics_manager: Optional[
          federated_language.program.ReleaseManager[
              federated_language.program.ReleasableStructure, int
          ]
      ] = None,
      evaluation_manager: Optional[evaluation_program_logic.EvaluationManager],
      evaluation_periodicity: Union[int, datetime.timedelta],
  ) -> None:
    pass
