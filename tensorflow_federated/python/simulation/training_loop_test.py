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
import numpy as np

from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.simulation import training_loop


@federated_computation.federated_computation
def _test_init_fn():
  return intrinsics.federated_value(0, placements.SERVER)


@federated_computation.federated_computation([
    computation_types.FederatedType(np.int32, placements.SERVER),
    computation_types.FederatedType(np.int32, placements.CLIENTS),
])
def _test_next_fn(state, client_data):
  del state, client_data  # Unused
  updated_state = intrinsics.federated_value(1, placements.SERVER)
  metrics = collections.OrderedDict([('metric', 1.0)])
  output = intrinsics.federated_value(metrics, placements.SERVER)
  return updated_state, output


@federated_computation.federated_computation([
    computation_types.FederatedType(np.int32, placements.SERVER),
    computation_types.FederatedType(np.int32, placements.CLIENTS),
])
def _test_evaluation_fn(state, client_data):
  del state, client_data  # Unused
  metrics = collections.OrderedDict([('metric', 2.0)])
  output = intrinsics.federated_value(metrics, placements.SERVER)
  return output


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
    training_process.next.return_value = ('update', {'metric': 1.0})
    training_selection_fn = mock.MagicMock()
    training_selection_fn.return_value = [0]

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds,
    )

    training_process.initialize.assert_called_once()
    expected_calls = []
    for round_num in range(1, total_rounds + 1):
      call = mock.call(round_num)
      expected_calls.append(call)
    self.assertEqual(training_selection_fn.call_args_list, expected_calls)
    expected_calls = []
    for round_num in range(1, total_rounds + 1):
      if round_num == 1:
        state = 'initialize'
      else:
        state = 'update'
      call = mock.call(state, [0])
      expected_calls.append(call)
    self.assertEqual(training_process.next.call_args_list, expected_calls)

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
        state='update', metrics={'metric': 1.0}
    )
    training_selection_fn = mock.MagicMock()
    training_selection_fn.return_value = [0]

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds,
    )

    training_process.initialize.assert_called_once()
    expected_calls = []
    for round_num in range(1, total_rounds + 1):
      call = mock.call(round_num)
      expected_calls.append(call)
    self.assertEqual(training_selection_fn.call_args_list, expected_calls)
    expected_calls = []
    for round_num in range(1, total_rounds + 1):
      if round_num == 1:
        state = 'initialize'
      else:
        state = 'update'
      call = mock.call(state, [0])
      expected_calls.append(call)
    self.assertEqual(training_process.next.call_args_list, expected_calls)

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
    training_process.next.return_value = ('update', {'metric': 1.0})
    training_selection_fn = mock.MagicMock()
    evaluation_fn = mock.create_autospec(
        computation_base.Computation, return_value={'metric': 1.0}
    )
    evaluation_selection_fn = mock.MagicMock()
    evaluation_selection_fn.return_value = [0]

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds,
        evaluation_fn=evaluation_fn,
        evaluation_selection_fn=evaluation_selection_fn,
        rounds_per_evaluation=rounds_per_evaluation,
    )

    call = mock.call(0)
    expected_calls = [call]
    for round_num in range(1, total_rounds + 1):
      if round_num % rounds_per_evaluation == 0:
        call = mock.call(round_num)
        expected_calls.append(call)
    self.assertEqual(evaluation_selection_fn.call_args_list, expected_calls)
    call = mock.call('initialize', [0])
    expected_calls = [call]
    for round_num in range(1, total_rounds + 1):
      if round_num % rounds_per_evaluation == 0:
        call = mock.call('update', [0])
        expected_calls.append(call)
    self.assertEqual(evaluation_fn.call_args_list, expected_calls)

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
  def test_program_state_manager_called_without_existing_program_state(
      self,
      total_rounds,
      rounds_per_saving_program_state,
      program_state,
      version,
  ):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = ('update', {'metric': 1.0})
    training_selection_fn = mock.MagicMock()
    program_state_manager = mock.AsyncMock()
    program_state_manager.load_latest.return_value = (program_state, version)

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds,
        program_state_manager=program_state_manager,
        rounds_per_saving_program_state=rounds_per_saving_program_state,
    )

    program_state_manager.load_latest.assert_called_once()
    expected_calls = []
    if version == 0:
      call = mock.call('initialize', 0)
      expected_calls.append(call)
    for round_num in range(version + 1, total_rounds + 1):
      if round_num % rounds_per_saving_program_state == 0:
        call = mock.call('update', round_num)
        expected_calls.append(call)
    self.assertEqual(program_state_manager.save.call_args_list, expected_calls)

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
  def test_program_state_manager_called_with_existing_program_state(
      self, version, total_rounds
  ):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = ('update', {'metric': 1.0})
    training_selection_fn = mock.MagicMock()
    program_state_manager = mock.AsyncMock()
    program_state_manager.load_latest.return_value = ('program_state', version)

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds,
        program_state_manager=program_state_manager,
        rounds_per_saving_program_state=1,
    )

    program_state_manager.load_latest.assert_called_once()
    expected_calls = []
    for round_num in range(version + 1, total_rounds + 1):
      call = mock.call('update', round_num)
      expected_calls.append(call)
    self.assertEqual(program_state_manager.save.call_args_list, expected_calls)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
      ('10', 10),
  )
  def test_metrics_managers_called_without_evaluation(self, total_rounds):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.initialize.type_signature.return_value = (
        _test_init_fn.type_signature
    )
    training_process.next.return_value = ('update', {'metric': 1.0})
    training_process.next.type_signature = _test_next_fn.type_signature
    training_selection_fn = mock.MagicMock()
    metrics_manager_1 = mock.AsyncMock()
    metrics_manager_2 = mock.AsyncMock()
    metrics_manager_3 = mock.AsyncMock()
    metrics_managers = [metrics_manager_1, metrics_manager_2, metrics_manager_3]

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds,
        metrics_managers=metrics_managers,
    )

    expected_calls = []
    for round_num in range(1, total_rounds + 1):
      metrics = collections.OrderedDict([
          ('metric', 1.0),
          ('training_time_in_seconds', mock.ANY),
          ('round_number', round_num),
      ])
      call = mock.call(metrics, key=round_num)
      expected_calls.append(call)
    for metrics_manager in metrics_managers:
      self.assertEqual(metrics_manager.release.call_args_list, expected_calls)

  @parameterized.named_parameters(
      ('0_1', 0, 1),
      ('1_1', 1, 1),
      ('1_2', 1, 2),
      ('2_1', 2, 1),
      ('2_2', 2, 2),
      ('10_1', 10, 1),
      ('10_5', 10, 5),
  )
  def test_metrics_managers_called_with_evaluation(
      self, total_rounds, rounds_per_evaluation
  ):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.next.return_value = ('update', {'metric': 1.0})
    training_selection_fn = mock.MagicMock()
    evaluation_fn = mock.MagicMock()
    evaluation_fn.return_value = {'metric': 1.0}
    evaluation_selection_fn = mock.MagicMock()
    metrics_manager_1 = mock.AsyncMock()
    metrics_manager_2 = mock.AsyncMock()
    metrics_manager_3 = mock.AsyncMock()
    metrics_managers = [metrics_manager_1, metrics_manager_2, metrics_manager_3]

    training_loop.run_training_process(
        training_process=training_process,
        training_selection_fn=training_selection_fn,
        total_rounds=total_rounds,
        evaluation_fn=evaluation_fn,
        evaluation_selection_fn=evaluation_selection_fn,
        rounds_per_evaluation=rounds_per_evaluation,
        metrics_managers=metrics_managers,
    )

    expected_calls = []
    metrics = collections.OrderedDict([
        ('evaluation/metric', 1.0),
        ('evaluation/evaluation_time_in_seconds', mock.ANY),
    ])
    call = mock.call(metrics, key=0)
    expected_calls.append(call)
    for round_num in range(1, total_rounds + 1):
      if round_num % rounds_per_evaluation == 0:
        metrics = collections.OrderedDict([
            ('metric', 1.0),
            ('training_time_in_seconds', mock.ANY),
            ('round_number', round_num),
            ('evaluation/metric', 1.0),
            ('evaluation/evaluation_time_in_seconds', mock.ANY),
        ])
      else:
        metrics = collections.OrderedDict([
            ('metric', 1.0),
            ('training_time_in_seconds', mock.ANY),
            ('round_number', round_num),
        ])
      call = mock.call(metrics, key=round_num)
      expected_calls.append(call)
    for metrics_manager in metrics_managers:
      self.assertEqual(metrics_manager.release.call_args_list, expected_calls)

  def test_metrics_managers_called_with_training_and_evaluation_time_10(self):
    training_process = mock.create_autospec(iterative_process.IterativeProcess)
    training_process.initialize.return_value = 'initialize'
    training_process.initialize.type_signature.return_value = (
        _test_init_fn.type_signature
    )
    training_process.next.return_value = ('update', {'metric': 1.0})
    training_process.next.type_signature = _test_next_fn.type_signature
    training_selection_fn = mock.MagicMock()
    evaluation_fn = mock.MagicMock()
    evaluation_fn.return_value = {'metric': 1.0}
    evaluation_fn.type_signature = _test_evaluation_fn.type_signature
    evaluation_selection_fn = mock.MagicMock()
    metrics_manager = mock.AsyncMock()

    with mock.patch('time.time') as mock_time:
      # Since absl.logging.info uses a call to time.time, we mock it out.
      mock_time.side_effect = [0.0, 10.0] * 3
      with mock.patch('absl.logging.info'):
        training_loop.run_training_process(
            training_process=training_process,
            training_selection_fn=training_selection_fn,
            total_rounds=1,
            evaluation_fn=evaluation_fn,
            evaluation_selection_fn=evaluation_selection_fn,
            metrics_managers=[metrics_manager],
        )

    expected_calls = []
    metrics = collections.OrderedDict([
        ('evaluation/metric', 1.0),
        ('evaluation/evaluation_time_in_seconds', 10.0),
    ])
    call = mock.call(metrics, key=0)
    expected_calls.append(call)
    metrics = collections.OrderedDict([
        ('metric', 1.0),
        ('training_time_in_seconds', 10.0),
        ('round_number', 1),
        ('evaluation/metric', 1.0),
        ('evaluation/evaluation_time_in_seconds', 10.0),
    ])
    call = mock.call(metrics, key=1)
    expected_calls.append(call)
    self.assertEqual(metrics_manager.release.call_args_list, expected_calls)


if __name__ == '__main__':
  absltest.main()
