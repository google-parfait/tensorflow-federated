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
from tensorflow_federated.python.program import release_manager as release_manager_lib
from tensorflow_federated.python.simulation import training_loop


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
      }
      self.assertDictEqual(output[i], expected_round_output)

  @mock.patch('time.time')
  def test_metrics_passed_to_metrics_managers(self, mock_time):
    mock_time.return_value = 0
    computation = mock.MagicMock(return_value={'a': 4, 'b': 5})
    client_selection_fn = mock.MagicMock()
    metric_manager1 = mock.create_autospec(release_manager_lib.ReleaseManager)
    metric_manager2 = mock.create_autospec(release_manager_lib.ReleaseManager)
    metrics_managers = [metric_manager1, metric_manager2]
    training_loop.run_stateless_simulation(
        computation,
        client_selection_fn,
        total_rounds=1,
        metrics_managers=metrics_managers)
    for manager in metrics_managers:
      manager.release.assert_called_once()
      save_call = manager.release.call_args
      self.assertEmpty(save_call[1])
      unnamed_args = save_call[0]
      self.assertLen(unnamed_args, 2)
      self.assertEqual(unnamed_args[1], 0)
      expected_metrics = {
          'a': 4,
          'b': 5,
          'round_num': 0,
          training_loop.ROUND_TIME_KEY: 0,
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
  def test_training_fns_called_with_tuple_next(self, total_rounds):
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
    self.assertEqual(training_selection_fn.call_args_list, calls)
    calls = []
    for round_num in range(1, total_rounds + 1):
      if round_num == 1:
        state = 'initialize'
      else:
        state = 'update'
      call = mock.call(state, [0])
      calls.append(call)
    self.assertEqual(training_process.next.call_args_list, calls)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
      ('10', 10),
  )
  def test_training_fns_called_with_odict_next(self, total_rounds):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = collections.OrderedDict(
        state='update', metrics={'metric': 0})
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
    self.assertEqual(training_selection_fn.call_args_list, calls)
    calls = []
    for round_num in range(1, total_rounds + 1):
      if round_num == 1:
        state = 'initialize'
      else:
        state = 'update'
      call = mock.call(state, [0])
      calls.append(call)
    self.assertEqual(training_process.next.call_args_list, calls)

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
    self.assertEqual(evaluation_selection_fn.call_args_list, calls)
    calls = [mock.call('initialize', [0])]
    for round_num in range(1, total_rounds + 1):
      if round_num % rounds_per_evaluation == 0:
        call = mock.call('update', [0])
        calls.append(call)
    self.assertEqual(evaluation_fn.call_args_list, calls)

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
    for round_num in range(version + 1, total_rounds + 1):
      if round_num % rounds_per_saving_program_state == 0:
        call = mock.call('update', round_num)
        calls.append(call)
    self.assertEqual(program_state_manager.save.call_args_list, calls)

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
          ('round_number', round_num),
      ])
      call = mock.call(metrics, round_num)
      calls.append(call)
    self.assertEqual(metrics_manager_1.release.call_args_list, calls)
    self.assertEqual(metrics_manager_2.release.call_args_list, calls)
    self.assertEqual(metrics_manager_3.release.call_args_list, calls)

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
    evaluation_fn = mock.MagicMock()
    evaluation_fn.return_value = {'metric': 0}
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
          ('round_number', round_num),
      ])
      if round_num % rounds_per_evaluation == 0:
        metrics.update([
            ('evaluation/metric', 0),
            ('evaluation/evaluation_time_in_seconds', mock.ANY),
        ])
      call = mock.call(metrics, round_num)
      calls.append(call)
    self.assertEqual(metrics_manager_1.release.call_args_list, calls)
    self.assertEqual(metrics_manager_2.release.call_args_list, calls)
    self.assertEqual(metrics_manager_3.release.call_args_list, calls)

  def test_performance_metrics_with_training_and_evaluation_time_10(self):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = ('update', {'metric': 0})
    training_selection_fn = mock.MagicMock()
    evaluation_fn = mock.MagicMock()
    evaluation_fn.return_value = {'metric': 0}
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

    for index, call in enumerate(metrics_manager.release.mock_calls):
      _, args, _ = call
      metrics, round_num = args
      if round_num > 0:
        self.assertEqual(metrics[training_loop.ROUND_NUMBER_KEY], 1)
        self.assertEqual(metrics[training_loop.TRAINING_TIME_KEY], 10.0)
      self.assertEqual(
          metrics[training_loop.EVALUATION_METRICS_PREFIX +
                  training_loop.EVALUATION_TIME_KEY], 10.0)
      self.assertEqual(round_num, index)

  @parameterized.named_parameters(
      ('0_1', 0, 1),
      ('0_3', 0, 3),
      ('1_0', 1, 0),
      ('1_1', 1, 1),
      ('2_0', 2, 0),
      ('2_2', 2, 2),
      ('2_3', 2, 3),
      ('2_5', 2, 5),
      ('5_2', 5, 2),
  )
  def test_program_state_manager_calls_on_existing_program_state(
      self, version, total_rounds):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = ('update', {'metric': 0})
    training_selection_fn = mock.MagicMock()
    program_state_manager = mock.MagicMock()
    program_state_manager.load_latest.return_value = ('program_state', version)

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds,
        program_state_manager=program_state_manager,
        rounds_per_saving_program_state=1)

    self.assertEqual(program_state_manager.load_latest.call_count, 1)
    calls = []
    for round_num in range(version + 1, total_rounds + 1):
      call = mock.call('update', round_num)
      calls.append(call)
    self.assertEqual(program_state_manager.save.call_args_list, calls)


if __name__ == '__main__':
  absltest.main()
