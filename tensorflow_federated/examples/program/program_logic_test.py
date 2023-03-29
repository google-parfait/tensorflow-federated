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

from typing import Optional
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.examples.program import program_logic


def _create_mock_context() -> mock.Mock:
  return mock.create_autospec(
      tff.program.NativeFederatedContext, spec_set=True, instance=True
  )


def _create_mock_data_source(
    *,
    federated_type: tff.FederatedType,
    data: Optional[object] = None,
) -> mock.Mock:
  mock_data_source = mock.create_autospec(
      tff.program.FederatedDataSource, spec_set=True, instance=True
  )
  type(mock_data_source).federated_type = mock.PropertyMock(
      return_value=federated_type
  )
  mock_data_source.iterator.return_value.select.side_effect = data
  return mock_data_source


def _create_mock_initialize(
    *,
    state_type: tff.FederatedType,
    side_effect: Optional[object] = None,
) -> mock.Mock:
  mock_initialize = mock.create_autospec(
      tff.Computation, spec_set=True, side_effect=side_effect
  )
  type(mock_initialize).type_signature = mock.PropertyMock(
      return_value=tff.FunctionType(None, state_type)
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
  type(mock_train).type_signature = mock.PropertyMock(
      return_value=tff.FunctionType(
          [state_type, train_data_type],
          [state_type, train_metrics_type],
      )
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
  type(mock_evaluation).type_signature = mock.PropertyMock(
      return_value=tff.FunctionType(
          [state_type, evaluation_data_type], evaluation_metrics_type
      )
  )
  return mock_evaluation


def _create_mock_program_state_manager(
    latest_program_state: Optional[program_logic._ProgramState] = None,
) -> mock.Mock:
  mock_program_state_manager = mock.create_autospec(
      tff.program.ProgramStateManager, spec_set=True, instance=True
  )
  mock_program_state_manager.load_latest.return_value = (
      latest_program_state,
      mock.create_autospec(int, spec_set=True),
  )
  return mock_program_state_manager


class CheckExpectedTypeSignaturesTest(parameterized.TestCase):

  def test_does_not_raise_unexpected_type_singature_error(self):
    state_type = tff.FederatedType(tf.string, tff.SERVER)
    train_data_type = tff.FederatedType(
        tff.SequenceType(tf.string), tff.CLIENTS
    )
    train_metrics_type = tff.FederatedType(tf.string, tff.SERVER)
    evaluation_data_type = tff.FederatedType(
        tff.SequenceType(tf.int32), tff.CLIENTS
    )
    evaluation_metrics_type = tff.FederatedType(tf.string, tff.SERVER)

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

  # pyformat: disable
  @parameterized.named_parameters(
      ('mismatch_initialize_train_state_type',
       tff.FederatedType(tf.int32, tff.SERVER),
       tff.FederatedType(tf.string, tff.SERVER),
       tff.FederatedType(tf.string, tff.SERVER)),
      ('mismatch_train_evaluation_type_signatures',
       tff.FederatedType(tf.string, tff.SERVER),
       tff.FederatedType(tf.string, tff.SERVER),
       tff.FederatedType(tf.int32, tff.SERVER)),
  )
  # pyformat: enable
  def test_raise_unexpected_type_singature_error(
      self, initialize_state_type, train_state_type, evaluation_state_type
  ):
    train_data_type = tff.FederatedType(
        tff.SequenceType(tf.string), tff.CLIENTS
    )
    train_metrics_type = tff.FederatedType(tf.string, tff.SERVER)
    evaluation_data_type = tff.FederatedType(
        tff.SequenceType(tf.int32), tff.CLIENTS
    )
    evaluation_metrics_type = tff.FederatedType(tf.string, tff.SERVER)

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

  # pyformat: disable
  @parameterized.named_parameters(
      ('program_state_manager_none', None, None),
      ('program_state_none', None),
      ('program_state_less_than_rounds',
       program_logic._ProgramState('state_5', 5)),
      ('program_state_equal_to_rounds',
       program_logic._ProgramState('state_10', 10)),
      ('program_state_greater_than_rounds',
       program_logic._ProgramState('state_100', 100)),
  )
  # pyformat: enable
  @tff.test.with_context(_create_mock_context)
  async def test_calls_program_components(
      self,
      program_state,
      mock_program_state_manager_factory=_create_mock_program_state_manager,
  ):
    if program_state is not None:
      _, saved_round_num = program_state
      start_round = saved_round_num + 1
    else:
      start_round = 1
    total_rounds = 10
    num_clients = 3

    rounds = range(start_round, total_rounds + 1)
    initial_state = 'initial_state'
    states = [f'state_{x}' for x in rounds]
    state_type = tff.FederatedType(tf.string, tff.SERVER)

    train_data = [f'train_data_{x}' for x in rounds]
    train_data_type = tff.FederatedType(tff.SequenceType(tf.string), tff.CLIENTS)

    train_metrics = [f'train_metrics_{x}' for x in rounds]
    train_metrics_type = tff.FederatedType(tf.string, tff.SERVER)

    evaluation_data = 'evaluation_data_1'
    evaluation_data_type = tff.FederatedType(tff.SequenceType(tf.int32), tff.CLIENTS)

    evaluation_metrics = 'evaluation_metrics_1'
    evaluation_metrics_type = tff.FederatedType(tf.string, tff.SERVER)

    mock_initialize = _create_mock_initialize(
        state_type=state_type, side_effect=[initial_state]
    )
    mock_train = _create_mock_train(
        state_type=state_type,
        train_data_type=train_data_type,
        train_metrics_type=train_metrics_type,
        side_effect=list(zip(states, train_metrics)),
    )
    mock_train_data_source = _create_mock_data_source(
        federated_type=train_data_type, data=train_data
    )
    mock_evaluation = _create_mock_evaluation(
        state_type=state_type,
        evaluation_data_type=evaluation_data_type,
        evaluation_metrics_type=evaluation_metrics_type,
        side_effect=[evaluation_metrics],
    )
    mock_evaluation_data_source = _create_mock_data_source(
        federated_type=evaluation_data_type, data=[evaluation_data]
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
      mock_program_state_manager = mock_program_state_manager_factory(
          latest_program_state=program_state
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

    # Assert that train data is selected for each train round.
    mock_train_data_source_iterator = (
        mock_train_data_source.iterator.return_value
    )
    expected_calls = []
    for _ in rounds:
      expected_calls.append(mock.call(num_clients))
    self.assertLen(
        mock_train_data_source_iterator.select.mock_calls, len(expected_calls)
    )
    mock_train_data_source_iterator.select.assert_has_calls(expected_calls)

    # Assert that the `train` computation is invoked for each train round.
    expected_calls = []
    if rounds:
      if program_state is not None:
        saved_state, _ = program_state
        expected_states = [saved_state, *states[:-1]]
      else:
        expected_states = [initial_state, *states[:-1]]
      for state, data in zip(expected_states, train_data):
        expected_calls.append(mock.call(state, data))
    self.assertLen(mock_train.mock_calls, len(expected_calls))
    mock_train.assert_has_calls(expected_calls)

    # Assert that train metrics are released for each train round.
    expected_calls = []
    if rounds:
      for round_num, metrics in zip(rounds, train_metrics):
        call = mock.call(metrics, train_metrics_type.member, round_num)
        expected_calls.append(call)
    self.assertLen(
        mock_train_metrics_manager.release.mock_calls, len(expected_calls)
    )
    mock_train_metrics_manager.release.assert_has_calls(expected_calls)

    if mock_program_state_manager is not None:
      # Assert that the program state is loaded once.
      structure = (initial_state, 0)
      mock_program_state_manager.load_latest.assert_called_once_with(structure)

      # Assert that the program state is saved for each train round.
      if rounds:
        expected_calls = []
        for round_num, state in zip(rounds, states):
          program_state = (state, round_num)
          expected_calls.append(mock.call(program_state, mock.ANY))
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
      expected_final_state = states[-1]
    elif program_state is not None:
      saved_state, _ = program_state
      expected_final_state = saved_state
    else:
      expected_final_state = initial_state

    # Assert that the `evaluation` computation is invoked once.
    mock_evaluation.assert_called_once_with(
        expected_final_state, evaluation_data
    )

    # Assert that evaluation metrics are released once.
    mock_evaluation_metrics_manager.release.assert_called_once_with(
        evaluation_metrics, evaluation_metrics_type.member, total_rounds + 1
    )

    # Assert that the model output is released once.
    mock_model_output_manager.release.assert_called_once_with(
        expected_final_state,
        state_type.member,
        None,
    )


if __name__ == '__main__':
  absltest.main()
