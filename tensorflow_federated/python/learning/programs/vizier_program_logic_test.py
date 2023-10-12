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

import collections
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from vizier.client import client_abc

from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.learning.programs import evaluation_program_logic
from tensorflow_federated.python.learning.programs import vizier_program_logic
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.program import data_source
from tensorflow_federated.python.program import native_platform
from tensorflow_federated.python.program import program_state_manager
from tensorflow_federated.python.program import release_manager


def _create_mock_context() -> mock.Mock:
  return mock.create_autospec(
      native_platform.NativeFederatedContext, spec_set=True, instance=True
  )


def _create_mock_train_process() -> mock.Mock:
  mock_process = mock.create_autospec(
      learning_process.LearningProcess, instance=True, spec_set=True
  )
  empty_state = composers.LearningAlgorithmState(
      global_model_weights=(),
      distributor=(),
      client_work=(),
      aggregator=(),
      finalizer=(),
  )
  mock_process.initialize.return_value = empty_state
  mock_process.next.return_value = learning_process.LearningProcessOutput(
      state=empty_state,
      metrics=collections.OrderedDict(
          distributor=(),
          client_work=collections.OrderedDict(train=collections.OrderedDict()),
          aggregator=(),
          finalizer=(),
      ),
  )
  mock_process.get_hparams.return_value = collections.OrderedDict()
  mock_process.set_hparams.return_value = empty_state
  return mock_process


class TrainModelWithVizierTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @context_stack_test_utils.with_context(_create_mock_context)
  async def test_calls_program_components(self):
    total_trials = 10
    total_rounds = 10
    num_clients = 3
    evaluation_periodicity = 1
    num_parallel_trials = 2

    mock_trials = [
        mock.create_autospec(
            client_abc.TrialInterface, spec_set=True, instance=True
        )
        for _ in range(total_trials)
    ]
    mock_study = mock.create_autospec(
        client_abc.StudyInterface, spec_set=True, instance=True
    )

    suggested_trials = []

    def suggest(*args, **kwargs):
      del args, kwargs
      for mock_trial in mock_trials:
        suggested_trials.append(mock_trial)
        yield [mock_trial]
      return []

    mock_study.suggest.side_effect = suggest()

    mock_study.trials().get.side_effect = lambda: suggested_trials
    mock_update_hparams = mock.create_autospec(
        computation_base.Computation, spec_set=True
    )
    mock_train_model_program_logic = mock.AsyncMock()
    mock_train_process = _create_mock_train_process()
    mock_train_process_factory = mock.Mock(return_value=mock_train_process)
    mock_train_data_source = mock.create_autospec(
        data_source.FederatedDataSource, spec_set=True, instance=True
    )
    mock_program_state_manager = mock.create_autospec(
        program_state_manager.ProgramStateManager, spec_set=True, instance=True
    )
    mock_program_state_manager_factory = mock.Mock(
        return_value=mock_program_state_manager
    )
    mock_model_output_manager = mock.create_autospec(
        release_manager.ReleaseManager, spec_set=True, instance=True
    )
    mock_model_output_manager_factory = mock.Mock(
        return_value=mock_model_output_manager
    )
    mock_train_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, spec_set=True, instance=True
    )
    mock_train_metrics_manager_factory = mock.Mock(
        return_value=mock_train_metrics_manager
    )
    mock_aggregated_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, spec_set=True, instance=True
    )
    mock_evaluation_manager = mock.create_autospec(
        evaluation_program_logic.EvaluationManager, spec_set=True, instance=True
    )
    type(mock_evaluation_manager).aggregated_metrics_manager = (
        mock.PropertyMock(return_value=mock_aggregated_metrics_manager)
    )
    mock_evaluation_manager_factory = mock.Mock(
        return_value=mock_evaluation_manager
    )

    await vizier_program_logic.train_model_with_vizier(
        study=mock_study,
        total_trials=total_trials,
        num_parallel_trials=num_parallel_trials,
        update_hparams=mock_update_hparams,
        train_model_program_logic=mock_train_model_program_logic,
        train_process_factory=mock_train_process_factory,
        train_data_source=mock_train_data_source,
        total_rounds=total_rounds,
        num_clients=num_clients,
        program_state_manager_factory=mock_program_state_manager_factory,
        model_output_manager_factory=mock_model_output_manager_factory,
        train_metrics_manager_factory=mock_train_metrics_manager_factory,
        evaluation_manager_factory=mock_evaluation_manager_factory,
        evaluation_periodicity=evaluation_periodicity,
    )

    # Assert that `get_hparams`, `set_hparams`, and `update_hparams` are called
    # for each trial.
    self.assertLen(mock_train_process.get_hparams.mock_calls, total_trials)
    self.assertLen(mock_train_process.set_hparams.mock_calls, total_trials)
    self.assertLen(mock_update_hparams.mock_calls, total_trials)

    # Assert that the `program_state_manager_factory` is invoked for each trial.
    # expected_calls = [mock.call(x) for x in mock_trials]
    # self.assertLen(
    #     mock_program_state_manager_factory.mock_calls, len(expected_calls)
    # )
    # mock_program_state_manager_factory.assert_has_calls(expected_calls)
    # mock_learning_process.assert_has_calls(expected_calls)

    # Assert that the `mock_train_process_factory` is invoked for each trial.
    expected_calls = [mock.call(x) for x in mock_trials]
    self.assertLen(mock_train_process_factory.mock_calls, len(expected_calls))
    mock_train_process_factory.assert_has_calls(expected_calls)

    # Assert that the `mock_train_metrics_manager_factory` is invoked for each
    # trial.
    expected_calls = [mock.call(x) for x in mock_trials]
    self.assertLen(
        mock_train_metrics_manager_factory.mock_calls, len(expected_calls)
    )
    mock_train_metrics_manager_factory.assert_has_calls(expected_calls)

    # Assert that the `mock_evaluation_manager_factory` is invoked for each
    # trial.
    expected_calls = [mock.call(x) for x in mock_trials]
    self.assertLen(
        mock_evaluation_manager_factory.mock_calls, len(expected_calls)
    )
    mock_evaluation_manager_factory.assert_has_calls(expected_calls)

    # Assert that the `train_federated_model` is invoked for each trial.
    expected_calls = []
    for _ in range(total_trials):
      call = mock.call(
          initial_train_state=mock_train_process.set_hparams.return_value,
          train_process=mock_train_process,
          train_data_source=mock_train_data_source,
          train_per_round_clients=num_clients,
          train_total_rounds=total_rounds,
          program_state_manager=mock_program_state_manager,
          model_output_manager=mock_model_output_manager,
          train_metrics_manager=mock.ANY,
          evaluation_manager=mock.ANY,
          evaluation_periodicity=evaluation_periodicity,
      )
      expected_calls.append(call)
    self.assertLen(
        mock_train_model_program_logic.mock_calls, len(expected_calls)
    )
    mock_train_model_program_logic.assert_has_calls(expected_calls)

    for trial, call in zip(
        mock_trials, mock_train_model_program_logic.mock_calls
    ):
      _, _, kwargs = call

      # Assert the `train_metrics_manager` contains a
      # `_IntermediateMeasurementReleaseManager` for each trail.
      actual_train_metrics_manager = kwargs['train_metrics_manager']
      expected_intermediate_measurement_release_manager = (
          vizier_program_logic._IntermediateMeasurementReleaseManager(trial)
      )

      if mock_train_metrics_manager_factory is not None:
        self.assertIsInstance(
            actual_train_metrics_manager, release_manager.GroupingReleaseManager
        )
        self.assertIn(
            expected_intermediate_measurement_release_manager,
            actual_train_metrics_manager._release_managers,
        )
      else:
        self.assertEqual(
            actual_train_metrics_manager,
            expected_intermediate_measurement_release_manager,
        )

      # Assert the `evaluation_manager` contains a
      # `_FinalMeasurementReleaseManager` for each trail.
      actual_evaluation_manager = kwargs['evaluation_manager']
      actual_aggregated_metrics_manager = (
          actual_evaluation_manager.aggregated_metrics_manager
      )

      expected_final_measurement_release_manager = (
          vizier_program_logic._FinalMeasurementReleaseManager(trial)
      )

      if mock_evaluation_manager.aggregated_metrics_manager is not None:
        self.assertIsInstance(
            actual_aggregated_metrics_manager,
            release_manager.GroupingReleaseManager,
        )
        self.assertIn(
            expected_final_measurement_release_manager,
            actual_aggregated_metrics_manager._release_managers,
        )
      else:
        self.assertEqual(
            actual_aggregated_metrics_manager,
            expected_final_measurement_release_manager,
        )


if __name__ == '__main__':
  absltest.main()
