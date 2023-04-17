# Copyright 2022, The TensorFlow Federated Authors.
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

import asyncio
import collections
import datetime
import unittest
from unittest import mock

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning.programs import evaluation_program_logic
from tensorflow_federated.python.learning.programs import program_logic
from tensorflow_federated.python.learning.programs import training_program_logic
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.program import data_source
from tensorflow_federated.python.program import federated_context
from tensorflow_federated.python.program import native_platform
from tensorflow_federated.python.program import program_state_manager
from tensorflow_federated.python.program import release_manager

# Convenience aliases.
TensorType = computation_types.TensorType


class WaitForTasksToFinishTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def test_empty_tasks_does_not_raise(self):
    try:
      await training_program_logic._wait_for_tasks_to_finish(set())
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'Test failed, expected no exceptions but got: {e}')

  async def test_one_task(self):
    with self.subTest('non_eval_task'):

      async def not_eval_task():
        return None

      tasks = set([asyncio.create_task(not_eval_task())])
      with mock.patch.object(
          training_program_logic, '_finalize_tasks'
      ) as mock_finalize:
        await training_program_logic._wait_for_tasks_to_finish(tasks)
        mock_finalize.assert_called_once_with(tasks)

    with self.subTest('eval_task'):
      train_round_num = 10

      async def eval_task():
        return train_round_num

      tasks = set([asyncio.create_task(eval_task())])
      with mock.patch.object(
          training_program_logic, '_finalize_tasks'
      ) as mock_finalize:
        await training_program_logic._wait_for_tasks_to_finish(tasks)
        mock_finalize.assert_called_once_with(tasks)

  async def test_multiple_tasks(self):
    results = [None, 5, None, None, 15, None]

    async def task(result):
      return result

    tasks = set([asyncio.create_task(task(r)) for r in results])
    with mock.patch.object(
        training_program_logic, '_finalize_tasks'
    ) as mock_finalize:
      await training_program_logic._wait_for_tasks_to_finish(tasks)
      mock_finalize.assert_called_once_with(tasks)


def _create_test_context() -> federated_context.FederatedContext:
  return native_platform.NativeFederatedContext(
      execution_contexts.create_async_local_cpp_execution_context()
  )


def _create_mock_datasource() -> mock.Mock:
  mock_datasource = mock.create_autospec(
      data_source.FederatedDataSource, instance=True, spec_set=True
  )

  def create_mock_iterator(*args, **kwargs) -> mock.Mock:
    del args  # Unused
    del kwargs  # Unused
    return mock.create_autospec(
        data_source.FederatedDataSourceIterator, instance=True, spec_set=True
    )

  mock_datasource.iterator.side_effect = create_mock_iterator
  return mock_datasource


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
  type(mock_process.next).type_signature = mock.PropertyMock(
      return_value=computation_types.FunctionType(
          parameter=(
              empty_state,
              computation_types.SequenceType(element=TensorType(tf.float32)),
          ),
          result=mock_process.next.return_value,
      )
  )
  return mock_process


def _create_metrics_release_call(*, key: int):
  return mock.call(mock.ANY, mock.ANY, key=key)


class TrainModelTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    # Print all failure explanations.
    self.maxDiff = None  # pylint: disable=invalid-name

  def test_is_train_model_program_logic(self):
    self.assertIsInstance(
        training_program_logic.train_model, program_logic.TrainModelProgramLogic
    )

  @context_stack_test_utils.with_context(_create_test_context)
  async def test_integration_runs_5_training_rounds_two_eval_rounds_from_scratch(
      self,
  ):
    train_num_clients = 5
    training_rounds = 5
    training_process = _create_mock_train_process()

    # Create a mock state manager that returns no previous state, starting
    # training from scratch.
    mock_program_state_manager = mock.create_autospec(
        program_state_manager.ProgramStateManager, instance=True, spec_set=True
    )
    mock_program_state_manager.load_latest.side_effect = [(None, 0)]

    mock_model_output_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True, spec_set=True
    )
    mock_train_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True, spec_set=True
    )
    for manager in (mock_model_output_manager, mock_train_metrics_manager):
      manager.release.return_value = None

    # Setup the meta evaluation manager to have no previous state, and launch
    # evaluations on the second, fourth, and last rounds (indexes 2, 4, and 5).
    mock_evaluation_manager = mock.create_autospec(
        evaluation_program_logic.EvaluationManager, instance=True, spec_set=True
    )

    # unittest.mock.AsyncMock doesn't appear to be a corourtine or asyncio.Task,
    # which is needed for the asyncio.wait calls inside the program logic.
    # Instead of mocking, we create fakes here.
    async def fake_evaluation(round_num) -> asyncio.Task:
      async def return_round_num() -> int:
        return round_num

      return asyncio.create_task(return_round_num())

    mock_evaluation_coros = tuple(
        fake_evaluation(train_round) for train_round in (2, 4, 5)
    )

    mock_evaluation_manager.start_evaluation.side_effect = list(
        mock_evaluation_coros
    )

    await training_program_logic.train_model(
        train_process=training_process,
        train_data_source=_create_mock_datasource(),
        train_per_round_clients=train_num_clients,
        train_total_rounds=training_rounds,
        program_state_manager=mock_program_state_manager,
        model_output_manager=mock_model_output_manager,
        evaluation_manager=mock_evaluation_manager,
        train_metrics_manager=mock_train_metrics_manager,
        evaluation_periodicity=2,
    )
    await asyncio.gather(*mock_evaluation_coros)

    # Assert that the program attempted to load a previous checkpoint and then
    # released the model state every round.
    any_algorithm_state = composers.LearningAlgorithmState(
        global_model_weights=mock.ANY,
        distributor=mock.ANY,
        client_work=mock.ANY,
        aggregator=mock.ANY,
        finalizer=mock.ANY,
    )
    self.assertEqual(
        mock_program_state_manager.load_latest.call_args_list,
        [mock.call((any_algorithm_state, 0))],
    )
    self.assertEqual(
        mock_program_state_manager.save.call_args_list,
        # Expect saving the initial state (version 0) and training rounds 1
        # through training_rounds.
        [
            mock.call((any_algorithm_state, round_num), version=round_num)
            for round_num in range(0, training_rounds + 1)
        ],
    )

    # Assert that training metrics were released every round.
    train_state_type, _ = training_process.next.type_signature.result
    self.assertSequenceEqual(
        [
            _create_metrics_release_call(key=round_num)
            for round_num in range(1, training_rounds + 1)
        ],
        mock_train_metrics_manager.release.call_args_list,
    )

    # Assert the model was output once at the end of training.
    self.assertSequenceEqual(
        [
            mock.call(
                any_algorithm_state,
                train_state_type,
                key=f'final_training_checkpoint_round_{training_rounds}',
            )
        ],
        mock_model_output_manager.release.call_args_list,
    )

    # Assert that evaluation were started in round 2, 4, and 5.
    mock_evaluation_manager.resume_from_previous_state.assert_called_once()
    self.assertSequenceEqual(
        [mock.call(round_num, mock.ANY, mock.ANY) for round_num in (2, 4, 5)],
        mock_evaluation_manager.start_evaluation.call_args_list,
    )
    mock_evaluation_manager.wait_for_evaluations_to_finish.assert_called_once()

  @context_stack_test_utils.with_context(_create_test_context)
  async def test_integration_runs_training_rounds_evaluates_on_time(self):
    train_num_clients = 5
    training_rounds = 5
    training_process = _create_mock_train_process()

    # Create a mock state manager that returns no previous state, starting
    # training from scratch.
    mock_program_state_manager = mock.create_autospec(
        program_state_manager.ProgramStateManager, instance=True, spec_set=True
    )
    mock_program_state_manager.load_latest.side_effect = [(None, 0)]

    mock_model_output_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True, spec_set=True
    )
    mock_train_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True, spec_set=True
    )
    for manager in (mock_model_output_manager, mock_train_metrics_manager):
      manager.release.return_value = None

    # Setup the meta evaluation manager to have no previous state, and launch
    # evaluations on the second and fourth rounds (indexes 1 and 3).
    mock_evaluation_manager = mock.create_autospec(
        evaluation_program_logic.EvaluationManager, instance=True, spec_set=True
    )

    # unittest.mock.AsyncMock doesn't appear to be a corourtine or asyncio.Task,
    # which is needed for the asyncio.wait calls inside the program logic.
    # Instead of mocking, we create fakes here.
    async def create_fake_evaluation(*args, **kwargs) -> asyncio.Task:
      del args  # Unused.
      del kwargs  # Unused.

      async def return_round_num() -> None:
        return

      return asyncio.create_task(return_round_num())

    mock_evaluation_manager.start_evaluation.side_effect = (
        create_fake_evaluation
    )

    # Patch `datetime.now` so that each round looks like it takes 20
    # milliseconds. With evaluation periodicity of 25 milliseconds and training
    # rounds finishing (relative) at [0, 20, 40, 60, 80], the test will expect
    # evaluations at round [1, 3, 5].
    with mock.patch(
        'datetime.datetime', wraps=datetime.datetime
    ) as mock_datetime:
      start_datetime = datetime.datetime(2022, 11, 17, 9, 0)
      mock_datetime.now.side_effect = [
          start_datetime + datetime.timedelta(milliseconds=20 * i)
          for i in range(training_rounds)
      ]
      await training_program_logic.train_model(
          train_process=training_process,
          train_data_source=_create_mock_datasource(),
          train_per_round_clients=train_num_clients,
          train_total_rounds=training_rounds,
          program_state_manager=mock_program_state_manager,
          model_output_manager=mock_model_output_manager,
          evaluation_manager=mock_evaluation_manager,
          train_metrics_manager=mock_train_metrics_manager,
          evaluation_periodicity=datetime.timedelta(milliseconds=25),
      )

    # Assert that the program attempted to load a previous checkpoint and then
    # released the model state every round.
    any_algorithm_state = composers.LearningAlgorithmState(
        global_model_weights=mock.ANY,
        distributor=mock.ANY,
        client_work=mock.ANY,
        aggregator=mock.ANY,
        finalizer=mock.ANY,
    )
    self.assertEqual(
        mock_program_state_manager.load_latest.call_args_list,
        [mock.call((any_algorithm_state, 0))],
    )
    self.assertEqual(
        mock_program_state_manager.save.call_args_list,
        # Expect saving the initial state (version 0) and training rounds 1
        # through training_rounds.
        [
            mock.call((any_algorithm_state, round_num), version=round_num)
            for round_num in range(0, training_rounds + 1)
        ],
    )

    # Assert that training metrics were released every round.
    train_state_type, _ = training_process.next.type_signature.result
    self.assertSequenceEqual(
        [
            _create_metrics_release_call(key=round_num)
            for round_num in range(1, training_rounds + 1)
        ],
        mock_train_metrics_manager.release.call_args_list,
    )

    # Assert the model was output once at the end of training.
    self.assertSequenceEqual(
        [
            mock.call(
                any_algorithm_state,
                train_state_type,
                key=f'final_training_checkpoint_round_{training_rounds}',
            )
        ],
        mock_model_output_manager.release.call_args_list,
    )

    # Assert that evaluations were started.
    mock_evaluation_manager.resume_from_previous_state.assert_called_once()
    mock_evaluation_manager.start_evaluation.assert_has_calls(
        [mock.call(round_num, mock.ANY, mock.ANY) for round_num in (1, 3, 5)]
    )
    mock_evaluation_manager.wait_for_evaluations_to_finish.assert_called_once()

  @context_stack_test_utils.with_context(_create_test_context)
  async def test_integration_runs_5_training_rounds_no_eval_manager(self):
    train_num_clients = 5
    training_rounds = 5
    training_process = _create_mock_train_process()

    # Create a mock state manager that returns no previous state, starting
    # training from scratch.
    mock_program_state_manager = mock.create_autospec(
        program_state_manager.ProgramStateManager, instance=True, spec_set=True
    )
    mock_program_state_manager.load_latest.side_effect = [(None, 0)]

    mock_model_output_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True, spec_set=True
    )
    mock_train_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True, spec_set=True
    )
    for manager in (mock_model_output_manager, mock_train_metrics_manager):
      manager.release.return_value = None

    await training_program_logic.train_model(
        train_process=training_process,
        train_data_source=_create_mock_datasource(),
        train_per_round_clients=train_num_clients,
        train_total_rounds=training_rounds,
        program_state_manager=mock_program_state_manager,
        model_output_manager=mock_model_output_manager,
        evaluation_manager=None,
        train_metrics_manager=mock_train_metrics_manager,
        evaluation_periodicity=100,
    )

    # Assert that the program attempted to load a previous checkpoint and then
    # released the model state every round.
    any_algorithm_state = composers.LearningAlgorithmState(
        global_model_weights=mock.ANY,
        distributor=mock.ANY,
        client_work=mock.ANY,
        aggregator=mock.ANY,
        finalizer=mock.ANY,
    )
    self.assertEqual(
        mock_program_state_manager.load_latest.call_args_list,
        [mock.call((any_algorithm_state, 0))],
    )
    self.assertEqual(
        mock_program_state_manager.save.call_args_list,
        # Expect saving the initial state (version 0) and training rounds 1
        # through training_rounds.
        [
            mock.call((any_algorithm_state, round_num), version=round_num)
            for round_num in range(0, training_rounds + 1)
        ],
    )

    # Assert that training metrics were released every round.
    train_state_type, _ = training_process.next.type_signature.result
    self.assertSequenceEqual(
        [
            _create_metrics_release_call(key=round_num)
            for round_num in range(1, training_rounds + 1)
        ],
        mock_train_metrics_manager.release.call_args_list,
    )

    # Assert the model was output once at the end of training.
    self.assertSequenceEqual(
        [
            mock.call(
                any_algorithm_state,
                train_state_type,
                key=f'final_training_checkpoint_round_{training_rounds}',
            )
        ],
        mock_model_output_manager.release.call_args_list,
    )

  @context_stack_test_utils.with_context(_create_test_context)
  async def test_resumes_from_previous_version_10_runs_one_round(self):
    train_num_clients = 5
    training_rounds = 11
    training_process = _create_mock_train_process()

    # Create a mock state manager that returns a previous state for round 10
    # (one before the last requested round).
    training_state = training_process.initialize()
    mock_program_state_manager = mock.create_autospec(
        program_state_manager.ProgramStateManager, instance=True
    )
    mock_program_state_manager.load_latest.side_effect = [(
        (training_state, training_rounds - 1),
        training_rounds - 1,
    )]

    mock_model_output_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    mock_train_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    for manager in (mock_model_output_manager, mock_train_metrics_manager):
      manager.release.return_value = None

    # Setup the meta evaluation manager to have no previous state and runs no
    # evaluations.
    mock_evaluation_manager = mock.create_autospec(
        evaluation_program_logic.EvaluationManager, instance=True
    )

    await training_program_logic.train_model(
        train_process=training_process,
        train_data_source=_create_mock_datasource(),
        train_per_round_clients=train_num_clients,
        train_total_rounds=training_rounds,
        program_state_manager=mock_program_state_manager,
        model_output_manager=mock_model_output_manager,
        evaluation_manager=mock_evaluation_manager,
        train_metrics_manager=mock_train_metrics_manager,
        evaluation_periodicity=100,
    )

    # Assert that the program attempted to load a previous checkpoint and then
    # released the model state on the next (11th) round.
    any_algorithm_state = composers.LearningAlgorithmState(
        global_model_weights=mock.ANY,
        distributor=mock.ANY,
        client_work=mock.ANY,
        aggregator=mock.ANY,
        finalizer=mock.ANY,
    )
    self.assertEqual(
        mock_program_state_manager.load_latest.call_args_list,
        [mock.call((any_algorithm_state, 0))],
    )
    self.assertEqual(
        mock_program_state_manager.save.call_args_list,
        # Expect saving the 11th version (running one round after loading the
        # 10th version)
        [
            mock.call(
                (any_algorithm_state, training_rounds), version=training_rounds
            )
        ],
        msg=mock_program_state_manager.save.call_args_list,
    )

    # Assert that training metrics were released for the resumed round and the
    # model as output after all training.
    train_state_type, _ = training_process.next.type_signature.result
    self.assertSequenceEqual(
        [_create_metrics_release_call(key=training_rounds)],
        mock_train_metrics_manager.release.call_args_list,
    )
    self.assertSequenceEqual(
        [
            mock.call(
                any_algorithm_state,
                train_state_type,
                key=f'final_training_checkpoint_round_{training_rounds}',
            )
        ],
        mock_model_output_manager.release.call_args_list,
    )

    # Assert no evaluations were created when attempting to resume evaluations.
    mock_evaluation_manager.resume_from_previous_state.assert_called_once()
    mock_evaluation_manager.wait_for_evaluations_to_finish.assert_called_once()

  @context_stack_test_utils.with_context(_create_test_context)
  async def test_resumes_from_previous_runs_no_train_rounds(self):
    train_num_clients = 5
    training_rounds = 10
    training_process = _create_mock_train_process()

    # Create a mock state manager that returns a previous state that already
    # completed the entire training process.
    training_state = training_process.initialize()
    mock_program_state_manager = mock.create_autospec(
        program_state_manager.ProgramStateManager, instance=True
    )
    mock_program_state_manager.load_latest.side_effect = [(
        (training_state, training_rounds),
        training_rounds,
    )]

    mock_model_output_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    mock_train_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )

    # Run an evaluation every round, but we assert none were run because no
    # training occurs after resuming from a previous version.
    mock_evaluation_manager = mock.create_autospec(
        evaluation_program_logic.EvaluationManager, instance=True, spec_set=True
    )

    await training_program_logic.train_model(
        train_process=training_process,
        train_data_source=_create_mock_datasource(),
        train_per_round_clients=train_num_clients,
        train_total_rounds=training_rounds,
        program_state_manager=mock_program_state_manager,
        model_output_manager=mock_model_output_manager,
        evaluation_manager=mock_evaluation_manager,
        train_metrics_manager=mock_train_metrics_manager,
        evaluation_periodicity=100,
    )

    # Assert that the program attempted to load a previous checkpoint and then
    # was not called again because no rounds were ran.
    any_algorithm_state = composers.LearningAlgorithmState(
        global_model_weights=mock.ANY,
        distributor=mock.ANY,
        client_work=mock.ANY,
        aggregator=mock.ANY,
        finalizer=mock.ANY,
    )
    self.assertSequenceEqual(
        mock_program_state_manager.load_latest.call_args_list,
        [mock.call((any_algorithm_state, 0))],
    )
    mock_program_state_manager.save.assert_not_called()

    # Assert no training metrics were released.
    mock_train_metrics_manager.release.assert_not_called()
    mock_model_output_manager.release.assert_not_called()

    # Assert no evaluations were created.
    mock_evaluation_manager.resume_from_previous_state.assert_called_once()
    mock_evaluation_manager.wait_for_evaluations_to_finish.assert_called_once()


if __name__ == '__main__':
  absltest.main()
