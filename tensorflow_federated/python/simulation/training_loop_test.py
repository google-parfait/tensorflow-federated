# Copyright 2021, The TensorFlow Federated Authors.
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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.simulation import checkpoint_manager
from tensorflow_federated.python.simulation import metrics_manager as metrics_manager_lib
from tensorflow_federated.python.simulation import training_loop


class LoadInitialCheckpointTest(parameterized.TestCase):

  def test_returns_input_state_and_zero_if_checkpoint_is_none(self):
    file_checkpoint_manager = mock.create_autospec(
        checkpoint_manager.FileCheckpointManager)
    file_checkpoint_manager.load_latest_checkpoint.return_value = (None, 10)
    input_state = 'input_state'
    state, round_num = training_loop._load_initial_checkpoint(
        input_state, file_checkpoint_manager)
    file_checkpoint_manager.load_latest_checkpoint.assert_called_once_with(
        input_state)
    self.assertEqual(input_state, state)
    self.assertEqual(round_num, 0)

  @parameterized.named_parameters(
      ('checkpoint_round_1', 'state', 0),
      ('checkpoint_round_2', {}, 5),
      ('checkpoint_round_3', '0.12', 10),
      ('checkpoint_round_4', 2, 2),
  )
  def test_checkpoint_not_none(self, state, round_num):
    file_checkpoint_manager = mock.create_autospec(
        checkpoint_manager.FileCheckpointManager)
    file_checkpoint_manager.load_latest_checkpoint.return_value = (state,
                                                                   round_num -
                                                                   1)
    input_state = 'input_state'
    actual_state, actual_round = training_loop._load_initial_checkpoint(
        input_state, file_checkpoint_manager)
    file_checkpoint_manager.load_latest_checkpoint.assert_called_once_with(
        input_state)

    self.assertEqual(actual_state, state)
    self.assertEqual(actual_round, round_num)


class ComputeValidationMetricsTest(absltest.TestCase):

  def test_validation_function_called_once(self):
    validation_fn = mock.MagicMock()
    input_state = 'state'
    round_num = 0
    training_loop._compute_validation_metrics(input_state, round_num,
                                              validation_fn)
    validation_fn.assert_called_once_with(input_state, round_num)

  def test_runs_with_empty_dict(self):
    validation_fn = lambda x, y: {}
    actual_metrics = training_loop._compute_validation_metrics(
        'state', 0, validation_fn)
    self.assertIn(
        training_loop.VALIDATION_METRICS_PREFIX +
        training_loop.VALIDATION_TIME_KEY, actual_metrics.keys())
    actual_metrics.pop(training_loop.VALIDATION_METRICS_PREFIX +
                       training_loop.VALIDATION_TIME_KEY)
    expected_metrics = {}
    self.assertDictEqual(actual_metrics, expected_metrics)

  def test_prefixes_keys_with_validation_string(self):
    metrics = {'metric_1': 0, 'metric_2': 1.0, 'metric_3': 'metric_3'}
    validation_fn = lambda x, y: metrics
    actual_metrics = training_loop._compute_validation_metrics(
        'state', 0, validation_fn)
    self.assertIn(
        training_loop.VALIDATION_METRICS_PREFIX +
        training_loop.VALIDATION_TIME_KEY, actual_metrics.keys())
    actual_metrics.pop(training_loop.VALIDATION_METRICS_PREFIX +
                       training_loop.VALIDATION_TIME_KEY)

    expected_metrics = {}
    for (key, value) in metrics.items():
      expected_metrics[training_loop.VALIDATION_METRICS_PREFIX + key] = value
    self.assertDictEqual(actual_metrics, expected_metrics)


class BuildOnLoopStartFnTest(parameterized.TestCase):

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._load_initial_checkpoint')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._compute_validation_metrics')
  def test_calls_with_no_input_args(self, mock_compute_validation,
                                    mock_initialize):
    on_loop_start_fn = training_loop._create_on_loop_start_fn()
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)
    mock_initialize.assert_not_called()
    mock_compute_validation.assert_not_called()

    expected_state = on_loop_start_input
    expected_round = 1
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._load_initial_checkpoint')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._compute_validation_metrics')
  def test_calls_with_only_checkpoint_manager_and_zero_checkpoint_round(
      self, mock_compute_validation, mock_initialize):
    file_checkpoint_manager = mock.create_autospec(
        checkpoint_manager.FileCheckpointManager)
    expected_state = 'state'
    expected_round = 1
    mock_initialize.return_value = (expected_state, expected_round - 1)
    on_loop_start_fn = training_loop._create_on_loop_start_fn(
        file_checkpoint_manager=file_checkpoint_manager)
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)
    mock_initialize.assert_called_once_with(on_loop_start_input,
                                            file_checkpoint_manager)
    mock_compute_validation.assert_not_called()
    file_checkpoint_manager.save_checkpoint.assert_called_once_with(
        expected_state, expected_round - 1)
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._load_initial_checkpoint')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._compute_validation_metrics')
  def test_calls_with_only_checkpoint_manager_and_non_zero_checkpoint_round(
      self, mock_compute_validation, mock_initialize):
    file_checkpoint_manager = mock.create_autospec(
        checkpoint_manager.FileCheckpointManager)
    expected_state = 'state'
    expected_round = 3
    mock_initialize.return_value = (expected_state, expected_round)
    on_loop_start_fn = training_loop._create_on_loop_start_fn(
        file_checkpoint_manager=file_checkpoint_manager)
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)
    mock_initialize.assert_called_once_with(on_loop_start_input,
                                            file_checkpoint_manager)
    mock_compute_validation.assert_not_called()
    file_checkpoint_manager.save_checkpoint.assert_not_called()
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._load_initial_checkpoint')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._compute_validation_metrics')
  def test_calls_with_only_metrics_managers(self, mock_compute_validation,
                                            mock_initialize):
    metric_manager1 = mock.create_autospec(metrics_manager_lib.MetricsManager)
    metric_manager2 = mock.create_autospec(metrics_manager_lib.MetricsManager)
    metrics_managers = [metric_manager1, metric_manager2]
    on_loop_start_fn = training_loop._create_on_loop_start_fn(
        metrics_managers=metrics_managers)
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)

    mock_initialize.assert_not_called()
    mock_compute_validation.assert_not_called()
    expected_state = on_loop_start_input
    expected_round = 1
    for metr_mngr in metrics_managers:
      metr_mngr.clear_metrics.assert_called_once_with(expected_round - 1)
      metr_mngr.save_metrics.assert_not_called()
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._load_initial_checkpoint')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._compute_validation_metrics')
  def test_calls_with_only_validation_fn(self, mock_compute_validation,
                                         mock_initialize):
    validation_fn = mock.MagicMock()
    on_loop_start_fn = training_loop._create_on_loop_start_fn(
        validation_fn=validation_fn)
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)

    mock_initialize.assert_not_called()
    expected_state = on_loop_start_input
    expected_round = 1
    mock_compute_validation.assert_called_once_with(expected_state,
                                                    expected_round - 1,
                                                    validation_fn)
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._load_initial_checkpoint')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._compute_validation_metrics')
  def test_calls_with_metrics_managers_and_validation_fn(
      self, mock_compute_validation, mock_initialize):
    metric_manager1 = mock.create_autospec(metrics_manager_lib.MetricsManager)
    metric_manager2 = mock.create_autospec(metrics_manager_lib.MetricsManager)
    metrics_managers = [metric_manager1, metric_manager2]
    validation_fn = mock.MagicMock()
    metrics = {'metric1': 2}
    mock_compute_validation.return_value = metrics
    on_loop_start_fn = training_loop._create_on_loop_start_fn(
        metrics_managers=metrics_managers, validation_fn=validation_fn)
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)
    mock_initialize.assert_not_called()
    expected_state = on_loop_start_input
    expected_round = 1
    mock_compute_validation.assert_called_once_with(expected_state,
                                                    expected_round - 1,
                                                    validation_fn)
    for metr_mngr in metrics_managers:
      metr_mngr.clear_metrics.assert_called_once_with(0)
      metr_mngr.save_metrics.assert_called_once_with(metrics, 0)
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._load_initial_checkpoint')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._compute_validation_metrics')
  def test_calls_with_non_zero_checkpoint_and_validation_fn(
      self, mock_compute_validation, mock_initialize):
    file_checkpoint_manager = mock.create_autospec(
        checkpoint_manager.FileCheckpointManager)
    validation_fn = mock.MagicMock()
    expected_state = 'state'
    expected_round = 2
    mock_initialize.return_value = (expected_state, expected_round)
    on_loop_start_fn = training_loop._create_on_loop_start_fn(
        file_checkpoint_manager=file_checkpoint_manager,
        validation_fn=validation_fn)
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)
    mock_initialize.assert_called_once_with(on_loop_start_input,
                                            file_checkpoint_manager)
    mock_compute_validation.assert_not_called()
    file_checkpoint_manager.save_checkpoint.assert_not_called()
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)


class CreateOnRoundEndTest(absltest.TestCase):

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._compute_validation_metrics')
  def test_calls_with_no_input_args(self, mock_compute_validation):
    on_round_end_fn = training_loop._create_on_round_end_fn()
    state = 'state'
    round_num = 1
    metrics = {'metric': 1}
    actual_state, actual_metrics = on_round_end_fn(state, round_num, metrics)
    mock_compute_validation.assert_not_called()
    self.assertEqual(actual_state, state)
    self.assertEqual(actual_metrics, metrics)

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._compute_validation_metrics')
  def test_calls_with_only_checkpoint_manager(self, mock_compute_validation):
    file_checkpoint_manager = mock.create_autospec(
        checkpoint_manager.FileCheckpointManager)
    on_round_end_fn = training_loop._create_on_round_end_fn(
        file_checkpoint_manager=file_checkpoint_manager)
    state = 'state'
    round_num = 1
    metrics = {'metric': 1}
    actual_state, actual_metrics = on_round_end_fn(state, round_num, metrics)
    mock_compute_validation.assert_not_called()
    file_checkpoint_manager.load_latest_checkpoint.assert_not_called()
    file_checkpoint_manager.save_checkpoint.assert_called_once_with(
        state, round_num)
    self.assertEqual(actual_state, state)
    self.assertEqual(actual_metrics, metrics)

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._compute_validation_metrics')
  def test_calls_with_only_metrics_managers(self, mock_compute_validation):
    mock_metrics_manager1 = mock.create_autospec(
        metrics_manager_lib.MetricsManager)
    mock_metrics_manager2 = mock.create_autospec(
        metrics_manager_lib.MetricsManager)
    metrics_managers = [mock_metrics_manager1, mock_metrics_manager2]
    on_round_end_fn = training_loop._create_on_round_end_fn(
        metrics_managers=metrics_managers)
    state = 'state'
    round_num = 1
    metrics = {'metric': 1}
    actual_state, actual_metrics = on_round_end_fn(state, round_num, metrics)
    mock_compute_validation.assert_not_called()
    for mock_metrics_manager in metrics_managers:
      mock_metrics_manager.clear_metrics.assert_not_called()
      mock_metrics_manager.save_metrics.assert_called_once_with(
          metrics, round_num)
    self.assertEqual(actual_state, state)
    self.assertEqual(actual_metrics, metrics)

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._compute_validation_metrics')
  def test_calls_with_only_validation_fn(self, mock_compute_validation):
    validation_fn = mock.MagicMock()
    mock_compute_validation.return_value = {'validation_metric': 2}
    on_round_end_fn = training_loop._create_on_round_end_fn(
        validation_fn=validation_fn)
    state = 'state'
    round_num = 1
    metrics = {'metric': 1}
    actual_state, actual_metrics = on_round_end_fn(state, round_num, metrics)
    mock_compute_validation.assert_called_once_with(state, round_num,
                                                    validation_fn)
    self.assertEqual(actual_state, state)
    expected_metrics = {'metric': 1, 'validation_metric': 2}
    self.assertDictEqual(actual_metrics, expected_metrics)

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._compute_validation_metrics')
  def test_calls_with_validation_fn_and_metrics_managers(
      self, mock_compute_validation):
    mock_metrics_manager1 = mock.create_autospec(
        metrics_manager_lib.MetricsManager)
    mock_metrics_manager2 = mock.create_autospec(
        metrics_manager_lib.MetricsManager)
    metrics_managers = [mock_metrics_manager1, mock_metrics_manager2]
    validation_fn = mock.MagicMock()
    mock_compute_validation.return_value = {'validation_metric': 2}
    on_round_end_fn = training_loop._create_on_round_end_fn(
        metrics_managers=metrics_managers, validation_fn=validation_fn)

    state = 'input_state'
    round_num = 1
    metrics = collections.OrderedDict(metric=1)
    actual_state, actual_metrics = on_round_end_fn(state, round_num, metrics)
    mock_compute_validation.assert_called_once_with(state, round_num,
                                                    validation_fn)
    expected_metrics = {'metric': 1, 'validation_metric': 2}
    for mock_metrics_manager in metrics_managers:
      mock_metrics_manager.clear_metrics.assert_not_called()
      mock_metrics_manager.save_metrics.assert_called_once_with(
          expected_metrics, round_num)
    self.assertEqual(actual_state, state)
    self.assertEqual(actual_metrics, expected_metrics)


class RunSimulationTest(parameterized.TestCase):

  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop.run_simulation_with_callbacks')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._create_on_round_end_fn')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._create_on_loop_start_fn')
  def test_run_simulation_passes_correctly_with_no_optional_arguments(
      self, mock_create_on_loop_start, mock_create_on_round_end,
      mock_run_simulation_with_callbacks):
    process = mock.create_autospec(iterative_process.IterativeProcess)
    client_selection_fn = lambda x: ()
    total_rounds = 10
    on_loop_start = 'on_loop_start'
    mock_create_on_loop_start.return_value = on_loop_start
    on_round_end = 'on_round_end'
    mock_create_on_round_end.return_value = on_round_end

    training_loop.run_simulation(process, client_selection_fn, total_rounds)
    mock_create_on_loop_start.assert_called_once_with(None, None, None)
    mock_create_on_round_end.assert_called_once_with(None, None, None)
    mock_run_simulation_with_callbacks.assert_called_once_with(
        process, client_selection_fn, total_rounds, on_loop_start, on_round_end)

  @parameterized.named_parameters(
      ('optional_inputs_0', None, None, None),
      ('optional_inputs_1', 'arg1', None, None),
      ('optional_inputs_2', None, 'arg2', None),
      ('optional_inputs_3', None, None, 'arg3'),
      ('optional_inputs_4', 'arg1', 'arg2', None),
      ('optional_inputs_5', 'arg1', None, 'arg3'),
      ('optional_inputs_6', None, 'arg2', 'arg3'),
      ('optional_inputs_7', 'arg1', 'arg2', 'arg3'),
  )
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop.run_simulation_with_callbacks')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._create_on_round_end_fn')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._create_on_loop_start_fn')
  def test_run_simulation_passes_unnamed_optional_arguments_correctly(
      self, file_checkpoint_manager, metrics_managers, validation_fn,
      mock_create_on_loop_start, mock_create_on_round_end,
      mock_run_simulation_with_callbacks):
    process = mock.create_autospec(iterative_process.IterativeProcess)
    client_selection_fn = lambda x: ()
    total_rounds = 10
    on_loop_start = 'on_loop_start'
    mock_create_on_loop_start.return_value = on_loop_start
    on_round_end = 'on_round_end'
    mock_create_on_round_end.return_value = on_round_end

    training_loop.run_simulation(process, client_selection_fn, total_rounds,
                                 file_checkpoint_manager, metrics_managers,
                                 validation_fn)
    mock_create_on_loop_start.assert_called_once_with(file_checkpoint_manager,
                                                      metrics_managers,
                                                      validation_fn)
    mock_create_on_round_end.assert_called_once_with(file_checkpoint_manager,
                                                     metrics_managers,
                                                     validation_fn)
    mock_run_simulation_with_callbacks.assert_called_once_with(
        process, client_selection_fn, total_rounds, on_loop_start, on_round_end)

  @parameterized.named_parameters(
      ('optional_inputs_0', None, None, None),
      ('optional_inputs_1', 'arg1', None, None),
      ('optional_inputs_2', None, 'arg2', None),
      ('optional_inputs_3', None, None, 'arg3'),
      ('optional_inputs_4', 'arg1', 'arg2', None),
      ('optional_inputs_5', 'arg1', None, 'arg3'),
      ('optional_inputs_6', None, 'arg2', 'arg3'),
      ('optional_inputs_7', 'arg1', 'arg2', 'arg3'),
  )
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop.run_simulation_with_callbacks')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._create_on_round_end_fn')
  @mock.patch('tensorflow_federated.python.simulation.'
              'training_loop._create_on_loop_start_fn')
  def test_run_simulation_passes_named_optional_arguments_correctly(
      self, file_checkpoint_manager, metrics_managers, validation_fn,
      mock_create_on_loop_start, mock_create_on_round_end,
      mock_run_simulation_with_callbacks):
    process = mock.create_autospec(iterative_process.IterativeProcess)
    client_selection_fn = lambda x: ()
    total_rounds = 10
    on_loop_start = 'on_loop_start'
    mock_create_on_loop_start.return_value = on_loop_start
    on_round_end = 'on_round_end'
    mock_create_on_round_end.return_value = on_round_end

    training_loop.run_simulation(
        process,
        client_selection_fn,
        total_rounds,
        file_checkpoint_manager=file_checkpoint_manager,
        metrics_managers=metrics_managers,
        validation_fn=validation_fn)
    mock_create_on_loop_start.assert_called_once_with(file_checkpoint_manager,
                                                      metrics_managers,
                                                      validation_fn)
    mock_create_on_round_end.assert_called_once_with(file_checkpoint_manager,
                                                     metrics_managers,
                                                     validation_fn)
    mock_run_simulation_with_callbacks.assert_called_once_with(
        process, client_selection_fn, total_rounds, on_loop_start, on_round_end)


class RunSimulationWithCallbacksTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('rounds_1', 1),
      ('rounds_2', 2),
      ('rounds_3', 3),
      ('rounds_0', 0),
  )
  def test_next_calls_total_rounds_times(self, total_rounds):
    process = mock.create_autospec(iterative_process.IterativeProcess)
    process.next.return_value = ('0', {})
    client_selection_fn = mock.MagicMock()
    training_loop.run_simulation_with_callbacks(process, client_selection_fn,
                                                total_rounds)
    self.assertEqual(process.next.call_count, total_rounds)
    self.assertEqual(client_selection_fn.call_count, total_rounds)

  @parameterized.named_parameters(
      ('rounds_1', 1),
      ('rounds_2', 2),
      ('rounds_3', 3),
      ('rounds_5', 5),
  )
  def test_round_num_is_passed_to_client_selection_fn(self, total_rounds):
    process = mock.create_autospec(iterative_process.IterativeProcess)
    process.next.return_value = ('0', {})
    client_selection_fn = mock.MagicMock()
    training_loop.run_simulation_with_callbacks(process, client_selection_fn,
                                                total_rounds)
    expected_calls = [mock.call(i) for i in range(1, total_rounds + 1)]
    self.assertEqual(expected_calls, client_selection_fn.mock_calls)

  @parameterized.named_parameters(
      ('rounds_1', 1),
      ('rounds_2', 2),
      ('rounds_3', 3),
      ('rounds_5', 5),
  )
  def test_on_round_end_called_after_each_round(self, total_rounds):
    process = mock.create_autospec(iterative_process.IterativeProcess)
    mock_state = 2.0
    mock_metrics = {'mock_train_metric': 1}
    process.next.return_value = (mock_state, mock_metrics)

    client_selection_fn = mock.MagicMock()
    on_round_end = mock.MagicMock()
    on_round_end.return_value = (3.0, {'validation/metric': 5})
    training_loop.run_simulation_with_callbacks(
        process, client_selection_fn, total_rounds, on_round_end=on_round_end)
    for i in range(1, total_rounds + 1):
      round_end_call_args = on_round_end.call_args_list[i - 1][0]
      self.assertEqual(round_end_call_args[0], mock_state)
      self.assertEqual(round_end_call_args[1], i)
      self.assertDictContainsSubset({
          'round_num': i,
          'mock_train_metric': 1
      }, round_end_call_args[2])

  @parameterized.named_parameters(
      ('rounds_1', 1),
      ('rounds_2', 2),
      ('rounds_3', 3),
      ('rounds_5', 5),
  )
  def test_on_loop_start_only_called_once(self, total_rounds):
    process = mock.create_autospec(iterative_process.IterativeProcess)
    process.next.return_value = (0, {})
    initialize_return_value = 'initial_state'
    process.initialize.return_value = initialize_return_value
    client_selection_fn = mock.MagicMock()
    on_loop_start = mock.MagicMock()
    on_loop_start.return_value = (0, 0)
    training_loop.run_simulation_with_callbacks(
        process, client_selection_fn, total_rounds, on_loop_start=on_loop_start)
    on_loop_start.assert_called_once_with(initialize_return_value)

  @mock.patch('time.time')
  def test_train_step_timing_metrics_correctly_added(self, mock_time):
    process = mock.create_autospec(iterative_process.IterativeProcess)
    mock_state = 2.0
    mock_metrics = {'mock_train_metric': 1}
    process.next.return_value = (mock_state, mock_metrics)
    client_selection_fn = lambda x: ()
    mock_time.return_value = 0

    # We use `on_round_end` to pass the metrics as an output
    on_round_end = mock.MagicMock()
    on_round_end.return_value = ((), {})
    training_loop.run_simulation_with_callbacks(
        process, client_selection_fn, 1, on_round_end=on_round_end)

    expected_metrics_passed_to_round_end = {
        'round_num': 1,
        'mock_train_metric': 1,
        training_loop.ROUND_TIME_KEY: 0,
        training_loop.ROUNDS_PER_HOUR_KEY: None
    }
    actual_metrics_passed_to_round_end = on_round_end.call_args_list[0][0][-1]
    self.assertDictEqual(actual_metrics_passed_to_round_end,
                         expected_metrics_passed_to_round_end)


class RunStatelessSimulationTest(absltest.TestCase):

  @mock.patch('time.time')
  def test_metrics_passed_to_output(self, mock_time):
    mock_time.return_value = 0
    computation = mock.MagicMock(return_value={'a': 4, 'b': 5})
    client_selection_fn = mock.MagicMock()
    output = training_loop.run_stateless_simulation(
        computation, client_selection_fn, total_rounds=1)
    expected_round_output = {
        'a': 4,
        'b': 5,
        'round_num': 0,
        training_loop.ROUND_TIME_KEY: 0,
        training_loop.ROUNDS_PER_HOUR_KEY: None
    }
    self.assertEqual(list(output.keys()), [0])
    self.assertDictEqual(output[0], expected_round_output)

  @mock.patch('time.time')
  def test_metrics_passed_to_output_with_multiple_rounds(self, mock_time):
    mock_time.return_value = 0
    computation = mock.MagicMock(return_value={'a': 4, 'b': 5})
    client_selection_fn = mock.MagicMock()
    output = training_loop.run_stateless_simulation(
        computation, client_selection_fn, total_rounds=5)
    self.assertEqual(list(output.keys()), list(range(5)))
    for i in range(5):
      expected_round_output = {
          'a': 4,
          'b': 5,
          'round_num': i,
          training_loop.ROUND_TIME_KEY: 0,
          training_loop.ROUNDS_PER_HOUR_KEY: None
      }
      self.assertDictEqual(output[i], expected_round_output)

  @mock.patch('time.time')
  def test_metrics_passed_to_metrics_managers(self, mock_time):
    mock_time.return_value = 0
    computation = mock.MagicMock(return_value={'a': 4, 'b': 5})
    client_selection_fn = mock.MagicMock()
    metric_manager1 = mock.create_autospec(metrics_manager_lib.MetricsManager)
    metric_manager2 = mock.create_autospec(metrics_manager_lib.MetricsManager)
    metrics_managers = [metric_manager1, metric_manager2]
    training_loop.run_stateless_simulation(
        computation,
        client_selection_fn,
        total_rounds=1,
        metrics_managers=metrics_managers)
    for manager in metrics_managers:
      manager.clear_metrics.assert_called_once_with(0)
      manager.save_metrics.assert_called_once()
      save_call = manager.save_metrics.call_args
      self.assertEmpty(save_call[1])
      unnamed_args = save_call[0]
      self.assertLen(unnamed_args, 2)
      self.assertEqual(unnamed_args[1], 0)
      expected_metrics = {
          'a': 4,
          'b': 5,
          'round_num': 0,
          training_loop.ROUND_TIME_KEY: 0,
          training_loop.ROUNDS_PER_HOUR_KEY: None
      }
      self.assertDictEqual(expected_metrics, unnamed_args[0])

  def test_client_data_gets_passed_to_computation(self):
    client_selection_fn = lambda x: x
    computation = mock.MagicMock()
    training_loop.run_stateless_simulation(
        computation, client_selection_fn, total_rounds=5)
    expected_call_args_list = [((x,),) for x in range(5)]
    self.assertEqual(computation.call_args_list, expected_call_args_list)


class RunTrainingProcessTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
      ('10', 10),
  )
  def test_training_fns_called(self, total_rounds):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = ('update', {'metric': 0})
    training_selection_fn = mock.MagicMock()
    training_selection_fn.return_value = [0]

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds)

    self.assertEqual(training_process.initialize.call_count, 1)
    calls = []
    for round_num in range(1, total_rounds + 1):
      call = mock.call(round_num)
      calls.append(call)
    training_selection_fn.assert_has_calls(calls)
    calls = []
    for round_num in range(1, total_rounds + 1):
      if round_num == 1:
        state = 'initialize'
      else:
        state = 'update'
      call = mock.call(state, [0])
      calls.append(call)
    training_process.next.assert_has_calls(calls)

  @parameterized.named_parameters(
      ('0_1', 0, 1),
      ('1_1', 1, 1),
      ('1_2', 1, 2),
      ('2_1', 2, 1),
      ('2_2', 2, 2),
      ('10_1', 10, 1),
      ('10_5', 10, 5),
  )
  def test_evaluation_fns_called(self, total_rounds, rounds_per_evaluation):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = ('update', {'metric': 0})
    training_selection_fn = mock.MagicMock()
    evaluation_fn = mock.create_autospec(
        computation_base.Computation, return_value={'metric': 0})
    evaluation_selection_fn = mock.MagicMock()
    evaluation_selection_fn.return_value = [0]

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds,
        evaluation_fn=evaluation_fn,
        evaluation_selection_fn=evaluation_selection_fn,
        rounds_per_evaluation=rounds_per_evaluation)

    calls = [mock.call(0)]
    for round_num in range(1, total_rounds + 1):
      if round_num % rounds_per_evaluation == 0:
        call = mock.call(round_num)
        calls.append(call)
    evaluation_selection_fn.assert_has_calls(calls)
    calls = [mock.call('initialize', [0])]
    for round_num in range(1, total_rounds + 1):
      if round_num % rounds_per_evaluation == 0:
        call = mock.call('update', [0])
        calls.append(call)
    evaluation_fn.assert_has_calls(calls)

  @parameterized.named_parameters(
      ('without_program_state_0_1', 0, 1, None, 0),
      ('without_program_state_1_1', 1, 1, None, 0),
      ('without_program_state_1_2', 1, 2, None, 0),
      ('without_program_state_2_1', 2, 1, None, 0),
      ('without_program_state_2_2', 2, 2, None, 0),
      ('without_program_state_10_1', 10, 1, None, 0),
      ('without_program_state_10_5', 10, 5, None, 0),
      ('with_program_state_0_1', 0, 1, 'update', 1),
      ('with_program_state_1_1', 1, 1, 'update', 1),
      ('with_program_state_1_2', 1, 2, 'update', 1),
      ('with_program_state_2_1', 2, 1, 'update', 1),
      ('with_program_state_2_2', 2, 2, 'update', 1),
      ('with_program_state_10_1', 10, 1, 'update', 1),
      ('with_program_state_10_5', 10, 5, 'update', 1),
  )
  def test_program_state_manager_called(self, total_rounds,
                                        rounds_per_saving_program_state,
                                        program_state, version):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = ('update', {'metric': 0})
    training_selection_fn = mock.MagicMock()
    program_state_manager = mock.MagicMock()
    program_state_manager.load_latest.return_value = (program_state, version)

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds,
        program_state_manager=program_state_manager,
        rounds_per_saving_program_state=rounds_per_saving_program_state)

    self.assertEqual(program_state_manager.load_latest.call_count, 1)
    calls = []
    if version == 0:
      call = mock.call('initialize', 0)
      calls.append(call)
    for round_num in range(1, total_rounds + 1):
      if round_num % rounds_per_saving_program_state == 0:
        call = mock.call('update', round_num)
        calls.append(call)
    program_state_manager.save.assert_has_calls(calls)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
      ('10', 10),
  )
  def test_metrics_managers_called_without_evaluation(self, total_rounds):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = ('update', {'metric': 0})
    training_selection_fn = mock.MagicMock()
    metrics_manager_1 = mock.MagicMock()
    metrics_manager_2 = mock.MagicMock()
    metrics_manager_3 = mock.MagicMock()

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds,
        metrics_managers=[
            metrics_manager_1, metrics_manager_2, metrics_manager_3
        ])

    calls = []
    for round_num in range(1, total_rounds + 1):
      metrics = collections.OrderedDict([
          ('metric', 0),
          ('training_time_in_seconds', mock.ANY),
          ('training_rounds_per_hour', mock.ANY),
          ('round_number', round_num),
      ])
      call = mock.call(metrics, round_num)
      calls.append(call)
    metrics_manager_1.save_metrics.assert_has_calls(calls)
    metrics_manager_2.save_metrics.assert_has_calls(calls)
    metrics_manager_3.save_metrics.assert_has_calls(calls)

  @parameterized.named_parameters(
      ('0_1', 0, 1),
      ('1_1', 1, 1),
      ('1_2', 1, 2),
      ('2_1', 2, 1),
      ('2_2', 2, 2),
      ('10_1', 10, 1),
      ('10_5', 10, 5),
  )
  def test_metrics_managers_called_with_evaluation(self, total_rounds,
                                                   rounds_per_evaluation):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = ('update', {'metric': 0})
    training_selection_fn = mock.MagicMock()
    evaluation_fn = mock.create_autospec(
        computation_base.Computation, return_value={'metric': 0})
    evaluation_selection_fn = mock.MagicMock()
    metrics_manager_1 = mock.MagicMock()
    metrics_manager_2 = mock.MagicMock()
    metrics_manager_3 = mock.MagicMock()

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds,
        evaluation_fn=evaluation_fn,
        evaluation_selection_fn=evaluation_selection_fn,
        rounds_per_evaluation=rounds_per_evaluation,
        metrics_managers=[
            metrics_manager_1, metrics_manager_2, metrics_manager_3
        ])

    calls = []
    metrics = collections.OrderedDict([
        ('evaluation/metric', 0),
        ('evaluation/evaluation_time_in_seconds', mock.ANY),
    ])
    call = mock.call(metrics, 0)
    calls.append(call)
    for round_num in range(1, total_rounds + 1):
      metrics = collections.OrderedDict([
          ('metric', 0),
          ('training_time_in_seconds', mock.ANY),
          ('training_rounds_per_hour', mock.ANY),
          ('round_number', round_num),
      ])
      if round_num % rounds_per_evaluation == 0:
        metrics.update([
            ('evaluation/metric', 0),
            ('evaluation/evaluation_time_in_seconds', mock.ANY),
        ])
      call = mock.call(metrics, round_num)
      calls.append(call)
    metrics_manager_1.save_metrics.assert_has_calls(calls)
    metrics_manager_2.save_metrics.assert_has_calls(calls)
    metrics_manager_3.save_metrics.assert_has_calls(calls)

  def test_performance_metrics_with_training_time_0(self):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = ('update', {'metric': 0})
    training_selection_fn = mock.MagicMock()
    metrics_manager = mock.MagicMock()

    with mock.patch('time.time') as mock_time:
      mock_time.return_value = 0.0
      training_loop.run_training_process(
          training_process=training_process,
          training_selection_fn=training_selection_fn,
          total_rounds=1,
          metrics_managers=[metrics_manager])

    for call in metrics_manager.save_metrics.mock_calls:
      _, args, _ = call
      metrics, round_num = args
      self.assertEqual(metrics[training_loop.ROUND_NUMBER_KEY], 1)
      self.assertEqual(metrics[training_loop.TRAINING_TIME_KEY], 0)
      self.assertIsNone(metrics[training_loop.TRAINING_ROUNDS_PER_HOUR_KEY])
      self.assertEqual(round_num, 1)

  def test_performance_metrics_with_training_and_evaluation_time_10(self):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = ('update', {'metric': 0})
    training_selection_fn = mock.MagicMock()
    evaluation_fn = mock.create_autospec(
        computation_base.Computation, return_value={'metric': 0})
    evaluation_selection_fn = mock.MagicMock()
    metrics_manager = mock.MagicMock()

    with mock.patch('time.time') as mock_time:
      # Since absl.logging.info uses a call to time.time, we mock it out.
      with mock.patch('absl.logging.info'):
        mock_time.side_effect = [0.0, 10.0] * 3
        training_loop.run_training_process(
            training_process=training_process,
            training_selection_fn=training_selection_fn,
            total_rounds=1,
            evaluation_fn=evaluation_fn,
            evaluation_selection_fn=evaluation_selection_fn,
            metrics_managers=[metrics_manager])

    for index, call in enumerate(metrics_manager.save_metrics.mock_calls):
      _, args, _ = call
      metrics, round_num = args
      if round_num > 0:
        self.assertEqual(metrics[training_loop.ROUND_NUMBER_KEY], 1)
        self.assertEqual(metrics[training_loop.TRAINING_TIME_KEY], 10.0)
        self.assertEqual(metrics[training_loop.TRAINING_ROUNDS_PER_HOUR_KEY],
                         60.0 * 60.0 / 10.0)
      self.assertEqual(
          metrics[training_loop.EVALUATION_METRICS_PREFIX +
                  training_loop.EVALUATION_TIME_KEY], 10.0)
      self.assertEqual(round_num, index)


if __name__ == '__main__':
  absltest.main()
