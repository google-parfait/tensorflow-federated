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
"""Program logic for tuning other program logic using Vizier."""

import asyncio
import collections
import datetime
from typing import Optional, Protocol, Union

from vizier import pyvizier
from vizier.client import client_abc

from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.learning.programs import evaluation_program_logic
from tensorflow_federated.python.learning.programs import program_logic
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.program import data_source
from tensorflow_federated.python.program import program_state_manager as program_state_manager_lib
from tensorflow_federated.python.program import release_manager
from tensorflow_federated.python.program import structure_utils
from tensorflow_federated.python.program import value_reference


class IntReleaseManagerFactory(Protocol):

  def __call__(
      self, trial: client_abc.TrialInterface
  ) -> release_manager.ReleaseManager[release_manager.ReleasableStructure, int]:
    pass


class StrReleaseManagerFactory(Protocol):

  def __call__(
      self, trial: client_abc.TrialInterface
  ) -> release_manager.ReleaseManager[release_manager.ReleasableStructure, str]:
    pass


class ProgramStateManagerFactory(Protocol):

  def __call__(
      self, trial: client_abc.TrialInterface
  ) -> program_state_manager_lib.ProgramStateManager:
    pass


class EvaluationManagerFactory(Protocol):

  def __call__(
      self,
      trial: client_abc.TrialInterface,
  ) -> evaluation_program_logic.EvaluationManager:
    pass


class TrainProcessFactory(Protocol):

  def __call__(
      self, trial: client_abc.TrialInterface
  ) -> learning_process.LearningProcess:
    pass


async def _create_measurement(
    value: value_reference.MaterializableStructure,
    steps: int,
    creation_time: datetime.datetime,
) -> pyvizier.Measurement:
  """Creates a Vizier Measurement for the given `value`."""
  materialized_value = await value_reference.materialize_value(value)
  flattened_value = structure_utils.flatten_with_name(materialized_value)
  metrics = {k: v for k, v in flattened_value}
  elapsed_time = datetime.datetime.now().astimezone() - creation_time
  elapsed_secs = elapsed_time.total_seconds()
  return pyvizier.Measurement(
      metrics=metrics, elapsed_secs=elapsed_secs, steps=steps
  )


class _IntermediateMeasurementReleaseManager(
    release_manager.ReleaseManager[release_manager.ReleasableStructure, int]
):
  """Releases metrics as a trial's intermediate measurement."""

  def __init__(self, trial: client_abc.TrialInterface):
    self._trial = trial

  async def release(
      self, value: release_manager.ReleasableStructure, key: int
  ) -> None:
    creation_time = self._trial.materialize().creation_time
    measurement = await _create_measurement(
        value=value, steps=key, creation_time=creation_time
    )
    if not self._trial.check_early_stopping():
      self._trial.add_measurement(measurement)

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, _IntermediateMeasurementReleaseManager):
      return NotImplemented
    return self._trial == other._trial


class _FinalMeasurementReleaseManager(
    release_manager.ReleaseManager[release_manager.ReleasableStructure, int]
):
  """Releases metrics as a trial's final measurement."""

  def __init__(self, trial: client_abc.TrialInterface):
    self._trial = trial

  async def release(
      self, value: release_manager.ReleasableStructure, key: int
  ) -> None:
    creation_time = self._trial.materialize().creation_time
    measurement = await _create_measurement(
        value=value, steps=key, creation_time=creation_time
    )
    self._trial.complete(measurement)

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, _FinalMeasurementReleaseManager):
      return NotImplemented
    return self._trial == other._trial


async def train_model_with_vizier(
    *,
    study: client_abc.StudyInterface,
    total_trials: int,
    num_parallel_trials: int = 1,
    update_hparams: computation_base.Computation,
    train_model_program_logic: program_logic.TrainModelProgramLogic,
    train_process_factory: TrainProcessFactory,
    train_data_source: data_source.FederatedDataSource,
    total_rounds: int,
    num_clients: int,
    program_state_manager_factory: ProgramStateManagerFactory,
    model_output_manager_factory: StrReleaseManagerFactory,
    train_metrics_manager_factory: Optional[IntReleaseManagerFactory] = None,
    evaluation_manager_factory: EvaluationManagerFactory,
    evaluation_periodicity: Union[int, datetime.timedelta],
) -> None:
  """Trains and tunes a federated model using Vizier.

  Args:
    study: The Vizier study to use to to tune `train_model_program_logic`.
    total_trials: The number of Vizier trials.
    num_parallel_trials: The number of Vizier trials to be evaluated in
      parallel. Default is 1.
    update_hparams: A `tff.Computation` to use to update the models hparams
      using a trials parameters.
    train_model_program_logic: The program logic to use for training and
      evaluating the model.
    train_process_factory: A factory for creating
      `tff.learning.templates.LearningProcess` to run for training.
    train_data_source: A `tff.program.FederatedDataSource` which returns client
      data used during training.
    total_rounds: The number of rounds of training.
    num_clients: The number of clients per round of training.
    program_state_manager_factory: A factory for creating
      `tff.program.ProgramStateManager`s for each trail.
    model_output_manager_factory: A factory for creating
      `tff.program.ReleaseManager`s used to release the model.
    train_metrics_manager_factory: A factory for creating
      `tff.program.ReleaseManager`s used to release training metrics for each
      trail.
    evaluation_manager_factory: A factory for creating
      `tff.learning.programs.EvaluationManager`s for each trail.
    evaluation_periodicity: Either a integer number of rounds or
      `datetime.timedelta` to await before sending a new training checkpoint to
      `evaluation_manager.start_evaluation`.
  """

  # TODO: b/238909797 - Vizier does not support `StudyStoppingConfig` in OSS.
  # For now, this program logic only supports stopping the study when the
  # `total_trials` have been completed.
  async def train_model_with_vizier_one_worker(vizier_worker_name):
    while len(list(study.trials().get())) < total_trials:
      trials = study.suggest(count=1, client_id=vizier_worker_name)
      if not trials:
        break
      trial = list(trials)[0]

      train_process = train_process_factory(trial)
      initial_train_state = train_process.initialize()
      hparams = train_process.get_hparams(initial_train_state)
      hparams = update_hparams(
          hparams, collections.OrderedDict(trial.parameters)
      )
      initial_train_state = train_process.set_hparams(
          initial_train_state, hparams
      )

      program_state_manager = program_state_manager_factory(trial)
      model_output_manager = model_output_manager_factory(trial)

      intermediate_release_manager = _IntermediateMeasurementReleaseManager(
          trial
      )
      if train_metrics_manager_factory is not None:
        manager = train_metrics_manager_factory(trial)
        train_metrics_manager = release_manager.GroupingReleaseManager([
            manager,
            intermediate_release_manager,
        ])
      else:
        train_metrics_manager = intermediate_release_manager

      final_release_manager = _FinalMeasurementReleaseManager(trial)
      evaluation_manager = evaluation_manager_factory(trial)
      if evaluation_manager.aggregated_metrics_manager is not None:
        aggregated_metrics_manager = release_manager.GroupingReleaseManager([
            evaluation_manager.aggregated_metrics_manager,
            final_release_manager,
        ])
      else:
        aggregated_metrics_manager = final_release_manager
      evaluation_manager = evaluation_program_logic.EvaluationManager(
          data_source=evaluation_manager.data_source,
          aggregated_metrics_manager=aggregated_metrics_manager,
          create_state_manager_fn=evaluation_manager.create_state_manager_fn,
          create_process_fn=evaluation_manager.create_process_fn,
          cohort_size=evaluation_manager.cohort_size,
          duration=evaluation_manager.duration,
      )

      await train_model_program_logic(
          initial_train_state=initial_train_state,
          train_process=train_process,
          train_data_source=train_data_source,
          train_per_round_clients=num_clients,
          train_total_rounds=total_rounds,
          program_state_manager=program_state_manager,
          model_output_manager=model_output_manager,
          train_metrics_manager=train_metrics_manager,
          evaluation_manager=evaluation_manager,
          evaluation_periodicity=evaluation_periodicity,
      )

  await asyncio.gather(
      *[
          train_model_with_vizier_one_worker(
              vizier_worker_name=f'vizier_worker_{i}'
          )
          for i in range(num_parallel_trials)
      ]
  )
