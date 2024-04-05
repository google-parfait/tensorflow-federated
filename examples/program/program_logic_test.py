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
"""An example of program logic test."""

import functools
from typing import Optional
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from examplesprogram import computations
from examplesprogram import program_logic


def _create_native_federated_context():
  context = tff.backends.native.create_async_local_cpp_execution_context()
  return tff.program.NativeFederatedContext(context)


def _create_mock_context() -> mock.Mock:
  return mock.create_autospec(
      tff.program.NativeFederatedContext, spec_set=True, instance=True
  )


def _create_mock_data_source_iterator(
    *,
    federated_type: tff.FederatedType,
    data: Optional[object] = None,
) -> mock.Mock:
  mock_data_source_iterator = mock.create_autospec(
      tff.program.FederatedDataSourceIterator, spec_set=True, instance=True
  )
  type(mock_data_source_iterator).federated_type = mock.PropertyMock(
      spec=tff.FederatedType, return_value=federated_type, spec_set=True
  )
  mock_data_source_iterator.select.side_effect = data
  return mock_data_source_iterator


def _create_mock_data_source(
    *,
    federated_type: tff.FederatedType,
    iterator: Optional[tff.program.FederatedDataSourceIterator] = None,
) -> mock.Mock:
  if iterator is None:
    iterator = _create_mock_data_source_iterator(federated_type=federated_type)

  mock_data_source = mock.create_autospec(
      tff.program.FederatedDataSource, spec_set=True, instance=True
  )
  type(mock_data_source).federated_type = mock.PropertyMock(
      spec=tff.Type, return_value=federated_type, spec_set=True
  )
  mock_data_source.iterator.return_value = iterator
  return mock_data_source


def _create_mock_initialize(
    *,
    state_type: tff.FederatedType,
    side_effect: Optional[object] = None,
) -> mock.Mock:
  mock_initialize = mock.create_autospec(
      tff.Computation, spec_set=True, side_effect=side_effect
  )
  type_signature = tff.FunctionType(None, state_type)
  type(mock_initialize).type_signature = mock.PropertyMock(
      spec=tff.Type, return_value=type_signature, spec_set=True
  )
  return mock_initialize


def _create_mock_train(
    *,
    state_type: tff.FederatedType,
    train_data_type: tff.FederatedType,
    train_metrics_type: tff.FederatedType,
    side_effect: Optional[object] = None,
) -> mock.Mock:
  mock_train = mock.create_autospec(
      tff.Computation, spec_set=True, side_effect=side_effect
  )
  type_signature = tff.FunctionType(
      [state_type, train_data_type],
      [state_type, train_metrics_type],
  )
  type(mock_train).type_signature = mock.PropertyMock(
      spec=tff.Type, return_value=type_signature, spec_set=True
  )
  return mock_train


def _create_mock_evaluation(
    *,
    state_type: tff.FederatedType,
    evaluation_data_type: tff.FederatedType,
    evaluation_metrics_type: tff.FederatedType,
    side_effect: Optional[object] = None,
) -> mock.Mock:
  mock_evaluation = mock.create_autospec(
      tff.Computation, spec_set=True, side_effect=side_effect
  )
  type_signature = tff.FunctionType(
      [state_type, evaluation_data_type], evaluation_metrics_type
  )
  type(mock_evaluation).type_signature = mock.PropertyMock(
      spec=tff.Type, return_value=type_signature, spec_set=True
  )
  return mock_evaluation


def _create_mock_program_state_manager(
    program_state: Optional[program_logic._ProgramState] = None,
    version: int = 0,
) -> mock.Mock:
  mock_program_state_manager = mock.create_autospec(
      tff.program.ProgramStateManager, spec_set=True, instance=True
  )
  mock_program_state_manager.load_latest.return_value = (program_state, version)
  return mock_program_state_manager


class CheckExpectedTypeSignaturesTest(parameterized.TestCase):

  def test_does_not_raise_unexpected_type_singature_error(self):
    state_type = tff.FederatedType(np.str_, tff.SERVER)
    train_data_type = tff.FederatedType(tff.SequenceType(np.str_), tff.CLIENTS)
    train_metrics_type = tff.FederatedType(np.str_, tff.SERVER)
    evaluation_data_type = tff.FederatedType(
        tff.SequenceType(np.int32), tff.CLIENTS
    )
    evaluation_metrics_type = tff.FederatedType(np.str_, tff.SERVER)

    mock_initialize = _create_mock_initialize(state_type=state_type)
    mock_train = _create_mock_train(
        state_type=state_type,
        train_data_type=train_data_type,
        train_metrics_type=train_metrics_type,
    )
    mock_train_data_source = _create_mock_data_source(
        federated_type=train_data_type
    )
    mock_evaluation = _create_mock_evaluation(
        state_type=state_type,
        evaluation_data_type=evaluation_data_type,
        evaluation_metrics_type=evaluation_metrics_type,
    )
    mock_evaluation_data_source = _create_mock_data_source(
        federated_type=evaluation_data_type
    )

    try:
      program_logic._check_expected_type_signatures(
          initialize=mock_initialize,
          train=mock_train,
          train_data_source=mock_train_data_source,
          evaluation=mock_evaluation,
          evaluation_data_source=mock_evaluation_data_source,
      )
    except program_logic.UnexpectedTypeSignatureError:
      self.fail('Raised `UnexpectedTypeSignatureError` unexpectedly.')

  @parameterized.named_parameters(
      (
          'mismatch_initialize_train_state_type',
          tff.FederatedType(np.int32, tff.SERVER),
          tff.FederatedType(np.str_, tff.SERVER),
          tff.FederatedType(np.str_, tff.SERVER),
      ),
      (
          'mismatch_train_evaluation_type_signatures',
          tff.FederatedType(np.str_, tff.SERVER),
          tff.FederatedType(np.str_, tff.SERVER),
          tff.FederatedType(np.int32, tff.SERVER),
      ),
  )
  def test_raise_unexpected_type_singature_error(
      self, initialize_state_type, train_state_type, evaluation_state_type
  ):
    train_data_type = tff.FederatedType(tff.SequenceType(np.str_), tff.CLIENTS)
    train_metrics_type = tff.FederatedType(np.str_, tff.SERVER)
    evaluation_data_type = tff.FederatedType(
        tff.SequenceType(np.int32), tff.CLIENTS
    )
    evaluation_metrics_type = tff.FederatedType(np.str_, tff.SERVER)

    mock_initialize = _create_mock_initialize(state_type=initialize_state_type)
    mock_train = _create_mock_train(
        state_type=train_state_type,
        train_data_type=train_data_type,
        train_metrics_type=train_metrics_type,
    )
    mock_train_data_source = _create_mock_data_source(
        federated_type=train_data_type
    )
    mock_evaluation = _create_mock_evaluation(
        state_type=evaluation_state_type,
        evaluation_data_type=evaluation_data_type,
        evaluation_metrics_type=evaluation_metrics_type,
    )
    mock_evaluation_data_source = _create_mock_data_source(
        federated_type=evaluation_data_type
    )

    with self.assertRaises(program_logic.UnexpectedTypeSignatureError):
      program_logic._check_expected_type_signatures(
          initialize=mock_initialize,
          train=mock_train,
          train_data_source=mock_train_data_source,
          evaluation=mock_evaluation,
          evaluation_data_source=mock_evaluation_data_source,
      )


class TrainFederatedModelTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      ('program_state_manager_none', 0, None),
      ('before_frist_round', 0, _create_mock_program_state_manager),
      ('after_first_round', 1, _create_mock_program_state_manager),
      ('less_than_total_rounds', 5, _create_mock_program_state_manager),
      ('equal_to_total_rounds', 10, _create_mock_program_state_manager),
      ('greater_than_total_rounds', 100, _create_mock_program_state_manager),
  )
  @tff.test.with_context(_create_mock_context)
  async def test_calls_program_components(
      self, round_num, mock_program_state_manager_factory
  ):
    total_rounds = 10
    num_clients = 3

    initial_state = 'initial_state'
    version = 1
    if round_num != 0:
      state = f'state_{round_num}'
      start_round = round_num + 1
    else:
      state = initial_state
      start_round = 1
    rounds = range(start_round, total_rounds + 1)

    states = [f'state_{x}' for x in rounds]
    state_type = tff.FederatedType(np.str_, tff.SERVER)

    train_data = [f'train_data_{x}' for x in rounds]
    train_data_type = tff.FederatedType(tff.SequenceType(np.str_), tff.CLIENTS)

    train_metrics = [f'train_metrics_{x}' for x in rounds]
    train_metrics_type = tff.FederatedType(np.str_, tff.SERVER)

    evaluation_data = 'evaluation_data_1'
    evaluation_data_type = tff.FederatedType(
        tff.SequenceType(np.int32), tff.CLIENTS
    )

    evaluation_metrics = 'evaluation_metrics_1'
    evaluation_metrics_type = tff.FederatedType(np.str_, tff.SERVER)

    mock_initialize = _create_mock_initialize(
        state_type=state_type, side_effect=[initial_state]
    )
    mock_train = _create_mock_train(
        state_type=state_type,
        train_data_type=train_data_type,
        train_metrics_type=train_metrics_type,
        side_effect=list(zip(states, train_metrics)),
    )
    mock_train_data_source_iterator = _create_mock_data_source_iterator(
        federated_type=train_data_type,
        data=train_data,
    )
    mock_train_data_source = _create_mock_data_source(
        federated_type=train_data_type, iterator=mock_train_data_source_iterator
    )
    mock_train_data_source_iterator = (
        mock_train_data_source.iterator.return_value
    )
    mock_evaluation = _create_mock_evaluation(
        state_type=state_type,
        evaluation_data_type=evaluation_data_type,
        evaluation_metrics_type=evaluation_metrics_type,
        side_effect=[evaluation_metrics],
    )
    mock_evaluation_data_source_iterator = _create_mock_data_source_iterator(
        federated_type=evaluation_data_type,
        data=[evaluation_data],
    )
    mock_evaluation_data_source = _create_mock_data_source(
        federated_type=evaluation_data_type,
        iterator=mock_evaluation_data_source_iterator,
    )
    mock_train_metrics_manager = mock.create_autospec(
        tff.program.ReleaseManager, spec_set=True, instance=True
    )
    mock_evaluation_metrics_manager = mock.create_autospec(
        tff.program.ReleaseManager, spec_set=True, instance=True
    )
    mock_model_output_manager = mock.create_autospec(
        tff.program.ReleaseManager, spec_set=True, instance=True
    )
    if mock_program_state_manager_factory is not None:
      if round_num != 0:
        program_state = program_logic._ProgramState(
            state=state,
            round_num=round_num,
            iterator=mock_train_data_source_iterator,
        )
      else:
        program_state = None
      mock_program_state_manager = mock_program_state_manager_factory(
          program_state=program_state, version=version
      )
    else:
      mock_program_state_manager = None

    with mock.patch.object(
        program_logic, '_check_expected_type_signatures'
    ) as mock_check:
      await program_logic.train_federated_model(
          initialize=mock_initialize,
          train=mock_train,
          train_data_source=mock_train_data_source,
          evaluation=mock_evaluation,
          evaluation_data_source=mock_evaluation_data_source,
          total_rounds=total_rounds,
          num_clients=num_clients,
          train_metrics_manager=mock_train_metrics_manager,
          evaluation_metrics_manager=mock_evaluation_metrics_manager,
          model_output_manager=mock_model_output_manager,
          program_state_manager=mock_program_state_manager,
      )
      mock_check.assert_called_once_with(
          initialize=mock_initialize,
          train=mock_train,
          train_data_source=mock_train_data_source,
          evaluation=mock_evaluation,
          evaluation_data_source=mock_evaluation_data_source,
      )

    # Assert that the `initialize` computation is invoked once.
    mock_initialize.assert_called_once_with()

    # Assert that the `train_data_source` iterator is created once.
    mock_train_data_source.iterator.assert_called_once_with()

    if mock_program_state_manager is not None:
      # Assert that the program state is loaded once.
      structure = (initial_state, 0, mock_train_data_source_iterator)
      mock_program_state_manager.load_latest.assert_called_once_with(structure)

    # Assert that train data is selected for each train round.
    mock_train_data_source_iterator = (
        mock_train_data_source.iterator.return_value
    )
    expected_calls = []
    for _ in rounds:
      call = mock.call(num_clients)
      expected_calls.append(call)
    self.assertLen(
        mock_train_data_source_iterator.select.mock_calls, len(expected_calls)
    )
    mock_train_data_source_iterator.select.assert_has_calls(expected_calls)

    # Assert that the `train` computation is invoked for each train round.
    expected_calls = []
    if rounds:
      expected_states = [state, *states[:-1]]
      for state, data in zip(expected_states, train_data):
        call = mock.call(state, data)
        expected_calls.append(call)
    self.assertLen(mock_train.mock_calls, len(expected_calls))
    mock_train.assert_has_calls(expected_calls)

    # Assert that train metrics are released for each train round.
    expected_calls = []
    if rounds:
      for round_num, metrics in zip(rounds, train_metrics):
        call = mock.call(metrics, key=round_num)
        expected_calls.append(call)
    self.assertLen(
        mock_train_metrics_manager.release.mock_calls, len(expected_calls)
    )
    mock_train_metrics_manager.release.assert_has_calls(expected_calls)

    if mock_program_state_manager is not None:
      # Assert that the program state is saved for each train round.
      if rounds:
        expected_calls = []
        versions = range(version + 1, version + 1 + len(rounds))
        for round_num, state, version in zip(rounds, states, versions):
          program_state = program_logic._ProgramState(
              state, round_num, mock_train_data_source_iterator
          )
          call = mock.call(program_state, version)
          expected_calls.append(call)
        self.assertLen(
            mock_program_state_manager.save.mock_calls, len(expected_calls)
        )
        mock_program_state_manager.save.assert_has_calls(expected_calls)

    # Assert that the `evaluation_data_source` iterator is created once.
    mock_evaluation_data_source.iterator.assert_called_once_with()

    # Assert that the evaluation data is selected once.
    mock_evaluation_data_source_iterator = (
        mock_evaluation_data_source.iterator.return_value
    )
    mock_evaluation_data_source_iterator.select.assert_called_once_with(
        num_clients
    )

    if rounds:
      expected_state = states[-1]
    else:
      expected_state = state

    # Assert that the `evaluation` computation is invoked once.
    mock_evaluation.assert_called_once_with(expected_state, evaluation_data)

    # Assert that evaluation metrics are released once.
    mock_evaluation_metrics_manager.release.assert_called_once_with(
        evaluation_metrics, key=total_rounds + 1
    )

    # Assert that the model output is released once.
    mock_model_output_manager.release.assert_called_once_with(
        expected_state, key=None
    )


class TrainFederatedModelIntegrationTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase
):

  @tff.test.with_context(_create_native_federated_context)
  async def test_fault_tolerance(self):
    datasets = [tf.data.Dataset.range(10, output_type=tf.int32)] * 3
    train_data_source = tff.program.DatasetDataSource(datasets)
    evaluation_data_source = tff.program.DatasetDataSource(datasets)
    num_clients = 3
    mock_train_metrics_manager = mock.create_autospec(
        tff.program.ReleaseManager, spec_set=True, instance=True
    )
    mock_evaluation_metrics_manager = mock.create_autospec(
        tff.program.ReleaseManager, spec_set=True, instance=True
    )
    mock_model_output_manager = mock.create_autospec(
        tff.program.ReleaseManager, spec_set=True, instance=True
    )
    program_state_dir = self.create_tempdir()
    program_state_manager = tff.program.FileProgramStateManager(
        program_state_dir
    )

    train_federated_model = functools.partial(
        program_logic.train_federated_model,
        initialize=computations.initialize,
        train=computations.train,
        train_data_source=train_data_source,
        evaluation=computations.evaluation,
        evaluation_data_source=evaluation_data_source,
        num_clients=num_clients,
        train_metrics_manager=mock_train_metrics_manager,
        evaluation_metrics_manager=mock_evaluation_metrics_manager,
        model_output_manager=mock_model_output_manager,
        program_state_manager=program_state_manager,
    )

    # Train first round.
    await train_federated_model(total_rounds=1)

    actual_versions = await program_state_manager.get_versions()
    self.assertEqual(actual_versions, [1])
    mock_train_metrics_manager.release.assert_called_once_with(
        {'total_sum': mock.ANY}, key=1
    )
    mock_evaluation_metrics_manager.release.assert_called_once_with(
        {'total_sum': mock.ANY}, key=2
    )
    mock_model_output_manager.release.assert_called_once_with(
        mock.ANY, key=None
    )

    # Reset the mocks instead of creating new mocks.
    mock_train_metrics_manager.reset_mock()
    mock_evaluation_metrics_manager.reset_mock()
    mock_model_output_manager.reset_mock()

    # Clear Train second round. This simulates a failure after the first round.
    await train_federated_model(total_rounds=2)

    actual_versions = await program_state_manager.get_versions()
    self.assertEqual(actual_versions, [1, 2])
    mock_train_metrics_manager.release.assert_called_once_with(
        {'total_sum': mock.ANY}, key=2
    )
    mock_evaluation_metrics_manager.release.assert_called_once_with(
        {'total_sum': mock.ANY}, key=3
    )
    mock_model_output_manager.release.assert_called_once_with(
        mock.ANY, key=None
    )


if __name__ == '__main__':
  absltest.main()
