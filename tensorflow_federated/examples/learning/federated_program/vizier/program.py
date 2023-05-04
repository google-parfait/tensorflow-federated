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
r"""An example of a federated program using `tff.learning` and Vizier.

Usage:

```
bazel run //tensorflow_federated/examples/learning/federated_program/vizier:program -- \
    --study_owner="<STUDY_OWNER>" \
    --output_dir="/tmp/example_vizier_program" \
    --alsologtostderr
```
"""

import asyncio
from collections.abc import Sequence
import datetime
import os.path
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import tensorflow_federated as tff
from vizier import pyvizier
from vizier.client import client_abc

from tensorflow_federated.examples.learning.federated_program.vizier import data_sources
from tensorflow_federated.examples.learning.federated_program.vizier import learning_process
from tensorflow_federated.examples.learning.federated_program.vizier import vizier_service


_TOTAL_ROUNDS = 2
_NUM_CLIENTS = 3
_EVALUATION_PERIODICITY = 2
_EVALUATION_DURATION = datetime.timedelta(seconds=30)

_MAX_NUM_TRIALS = 10
_PROBLEM_STATEMENT_PARAMETERS = ['finalizer/learning_rate']
_PROBLEM_STATEMENT_METRICS = ['eval/loss']

_STUDY_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    'study_spec.textproto',
)

_STUDY_OWNER = flags.DEFINE_string(
    name='study_owner',
    default=None,
    help='The owner of the Vizier study.',
    required=True,
)

_OUTPUT_DIR = flags.DEFINE_string(
    name='output_dir',
    default=None,
    help='The output path.',
    required=True,
)


def _check_problem_statement(problem: pyvizier.ProblemStatement):
  actual_parameters = list(problem.search_space.parameter_names)
  if actual_parameters != _PROBLEM_STATEMENT_PARAMETERS:
    raise ValueError(
        'Expected the problem statement to define exactly the parameters'
        f'`{_PROBLEM_STATEMENT_PARAMETERS}`, found `{actual_parameters}`.'
    )
  actual_metrics = [x.name for x in problem.metric_information]
  if actual_metrics != _PROBLEM_STATEMENT_METRICS:
    raise ValueError(
        'Expected the problem statement to define exactly the metrics'
        f'`{_PROBLEM_STATEMENT_METRICS}`, found `{actual_metrics}`.'
    )


@tff.tf_computation
def _update_hparams(hparams, parameters):
  hparams['finalizer']['learning_rate'] = parameters['finalizer/learning_rate']
  return hparams


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  context = tff.backends.native.create_async_local_cpp_execution_context()
  context = tff.program.NativeFederatedContext(context)
  tff.framework.set_default_context(context)

  timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
  experiment_name = f'vizier_example_{timestamp}'

  study_config = vizier_service.create_study_config(_STUDY_CONFIG_PATH)
  _check_problem_statement(study_config)
  study = vizier_service.create_study(
      study_config=study_config,
      name=experiment_name,
      owner=_STUDY_OWNER.value,
  )
  logging.info(
      'Created Vizier Study at http://vizier/#/study/%s', study.resource_name
  )

  data_source, _, input_spec = data_sources.create_data_sources()
  train_process, evaluation_process = (
      learning_process.create_learning_processes(input_spec)
  )

  model_output_manager = tff.program.LoggingReleaseManager()

  def _metrics_manager_factory(
      trial: client_abc.TrialInterface,
  ) -> tff.program.ReleaseManager[tff.program.ReleasableStructure, int]:
    del trial  # Unused.
    return tff.program.LoggingReleaseManager()

  def _program_state_manager_factory(
      trial: client_abc.TrialInterface,
  ) -> tff.program.ProgramStateManager:
    trial_name = f'trial_{trial.id}'
    root_dir = os.path.join(_OUTPUT_DIR.value, experiment_name, trial_name)
    return tff.program.FileProgramStateManager(root_dir)

  def _evaluation_manager_factory(
      trial: client_abc.TrialInterface,
  ) -> tff.learning.programs.EvaluationManager:
    aggregated_metrics_manager = tff.program.LoggingReleaseManager()

    def _create_state_manager_fn(
        name: str,
    ) -> tff.program.FileProgramStateManager:
      trial_name = f'trial_{trial.id}'
      root_dir = os.path.join(
          _OUTPUT_DIR.value,
          experiment_name,
          trial_name,
          name,
      )
      return tff.program.FileProgramStateManager(root_dir)

    def _create_process_fn(
        name: str,
    ) -> tuple[
        tff.learning.templates.LearningProcess,
        Optional[
            tff.program.ReleaseManager[tff.program.ReleasableStructure, int]
        ],
    ]:
      del name  # Unused.
      release_manager = tff.program.LoggingReleaseManager()
      return (evaluation_process, release_manager)

    return tff.learning.programs.EvaluationManager(
        data_source=data_source,
        aggregated_metrics_manager=aggregated_metrics_manager,
        create_state_manager_fn=_create_state_manager_fn,
        create_process_fn=_create_process_fn,
        cohort_size=_NUM_CLIENTS,
        duration=_EVALUATION_DURATION,
    )

  asyncio.run(
      tff.learning.programs.train_model_with_vizier(
          study=study,
          total_trials=_MAX_NUM_TRIALS,
          update_hparams=_update_hparams,
          train_model_program_logic=tff.learning.programs.train_model,
          train_process=train_process,
          train_data_source=data_source,
          total_rounds=_TOTAL_ROUNDS,
          num_clients=_NUM_CLIENTS,
          program_state_manager_factory=_program_state_manager_factory,
          model_output_manager=model_output_manager,
          train_metrics_manager_factory=_metrics_manager_factory,
          evaluation_manager_factory=_evaluation_manager_factory,
          evaluation_periodicity=1,
      )
  )

  for trial in study.optimal_trials():
    logging.info('Trial %s is the best trial.', trial.id)


if __name__ == '__main__':
  app.run(main)
