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

import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from vizier.client import client_abc

from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.learning.programs import evaluation_program_logic
from tensorflow_federated.python.learning.programs import vizier_program_logic
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.program import data_source
from tensorflow_federated.python.program import native_platform
from tensorflow_federated.python.program import program_state_manager
from tensorflow_federated.python.program import release_manager


def _create_mock_context() -> mock.Mock:
  return mock.create_autospec(
      native_platform.NativeFederatedContext, spec_set=True, instance=True
  )


class TrainModelWithVizierTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @context_stack_test_utils.with_context(_create_mock_context)
  async def test_calls_program_components(self):
    total_trials = 10
    total_rounds = 10
    num_clients = 3
    evaluation_periodicity = 1

    mock_trials = [
        mock.create_autospec(
            client_abc.TrialInterface, spec_set=True, instance=True
        )
        for _ in range(total_trials)
    ]
    mock_study = mock.create_autospec(
        client_abc.StudyInterface, spec_set=True, instance=True
    )
    mock_study.trials().get.side_effect = [
        mock_trials[0:i] for i in range(total_trials + 1)
    ]
    mock_study.suggest.side_effect = [[x] for x in mock_trials]
    mock_update_hparams = mock.create_autospec(
        computation_base.Computation, spec_set=True
    )
    mock_train_model_program_logic = mock.AsyncMock()
    mock_train_process = mock.create_autospec(
        learning_process.LearningProcess, spec_set=True, instance=True
    )
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
    mock_initialize = mock.create_autospec(
        computation_base.Computation, spec_set=True
    )
    mock_initialize_factory = mock.Mock(return_value=mock_initialize)
    patched_vizier_program_logic = mock.patch.object(
        vizier_program_logic,
        '_create_initialize_factory',
        return_value=mock_initialize_factory,
    )
    patched_learning_process = mock.patch.object(
        learning_process,
        'LearningProcess',
        return_value=mock_train_process,
    )

    with (
        patched_vizier_program_logic,
        patched_learning_process as mock_learning_process,
    ):
      await vizier_program_logic.train_model_with_vizier(
          study=mock_study,
          total_trials=total_trials,
          update_hparams=mock_update_hparams,
          train_model_program_logic=mock_train_model_program_logic,
          train_process=mock_train_process,
          train_data_source=mock_train_data_source,
          total_rounds=total_rounds,
          num_clients=num_clients,
          program_state_manager_factory=mock_program_state_manager_factory,
          model_output_manager=mock_model_output_manager,
          train_metrics_manager_factory=mock_train_metrics_manager_factory,
          evaluation_manager_factory=mock_evaluation_manager_factory,
          evaluation_periodicity=evaluation_periodicity,
      )

    # Assert that the `initialize_factory` is invoked for each trial.
    expected_calls = [mock.call(x) for x in mock_trials]
    self.assertLen(mock_initialize_factory.mock_calls, len(expected_calls))
    mock_initialize_factory.assert_has_calls(expected_calls)

    # Assert that a `LearningProcess` is constructed with a new initialize for
    # each trial.
    expected_calls = []
    for _ in range(total_trials):
      call = mock.call(
          initialize_fn=mock_initialize,
          next_fn=mock.ANY,
          get_model_weights=mock.ANY,
          set_model_weights=mock.ANY,
          get_hparams_fn=mock.ANY,
          set_hparams_fn=mock.ANY,
      )
      expected_calls.append(call)
    self.assertLen(mock_learning_process.mock_calls, len(expected_calls))
    mock_learning_process.assert_has_calls(expected_calls)

    # Assert that the `program_state_manager_factory` is invoked for each trial.
    # expected_calls = [mock.call(x) for x in mock_trials]
    # self.assertLen(
    #     mock_program_state_manager_factory.mock_calls, len(expected_calls)
    # )
    # mock_program_state_manager_factory.assert_has_calls(expected_calls)
    # mock_learning_process.assert_has_calls(expected_calls)

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
