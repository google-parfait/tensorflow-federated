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
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.programs import evaluation_program_logic
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.program import data_source
from tensorflow_federated.python.program import federated_context
from tensorflow_federated.python.program import file_program_state_manager
from tensorflow_federated.python.program import native_platform
from tensorflow_federated.python.program import release_manager

# Convenience aliases.
TensorType = computation_types.TensorType


class _NumpyMatcher:
  """An object to assert equality against np.ndarray values in mock calls."""

  def __init__(self, value: np.ndarray):
    self._value = value

  def __eq__(self, other) -> bool:
    if not isinstance(other, np.ndarray):
      return False
    return np.all(self._value == other)

  def __repr__(self) -> str:
    return f'_NumpyMatcher({self._value})'


class ExtractAndRewrapMetricsTest(tf.test.TestCase):

  def test_extracts_substructure_adds_prefix(self):
    test_structure = collections.OrderedDict(
        test=collections.OrderedDict(
            path=collections.OrderedDict(test_a=[1, 2, 3], test_b=10.0),
            other=collections.OrderedDict(test_one=1.0, test_two=2.0),
        )
    )
    extracted_structure = evaluation_program_logic.extract_and_rewrap_metrics(
        test_structure, path=('test', 'path')
    )
    self.assertAllClose(
        extracted_structure,
        collections.OrderedDict([
            (
                evaluation_program_logic.MODEL_METRICS_PREFIX,
                collections.OrderedDict(test_a=[1, 2, 3], test_b=10.0),
            ),
            (
                'test',
                collections.OrderedDict(
                    other=collections.OrderedDict(test_one=1.0, test_two=2.0)
                ),
            ),
        ]),
    )

  def test_no_path_raises_error(self):
    with self.assertRaisesRegex(ValueError, '`path` is empty'):
      evaluation_program_logic.extract_and_rewrap_metrics(
          collections.OrderedDict(foo=1, bar=2), path=()
      )

  def test_path_does_not_exist_in_structure_fails(self):
    with self.subTest('early_path'):
      with self.assertRaisesRegex(KeyError, r'\[test\]'):
        evaluation_program_logic.extract_and_rewrap_metrics(
            collections.OrderedDict(foo=1, bar=2), path=('test', 'bad_path')
        )
    with self.subTest('last_path'):
      with self.assertRaisesRegex(KeyError, r'\[bad_path\]'):
        evaluation_program_logic.extract_and_rewrap_metrics(
            collections.OrderedDict(foo=collections.OrderedDict(bar=2)),
            path=('foo', 'bad_path'),
        )

  def test_federated_value_structure(self):

    def awaitable_value(value, value_type):
      async def _value():
        return value

      return native_platform.AwaitableValueReference(_value, value_type)

    test_value = collections.OrderedDict(
        a=awaitable_value('foo', computation_types.TensorType(np.str_)),
        b=collections.OrderedDict(
            x=awaitable_value('bar', computation_types.TensorType(np.str_)),
            z=awaitable_value(1.0, computation_types.TensorType(np.float32)),
        ),
    )

    try:
      evaluation_program_logic.extract_and_rewrap_metrics(
          metrics_structure=test_value, path=['b']
      )
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'Unexpected error raised: {e}')


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


def _create_per_round_eval_metrics_release_call(*, key: int):
  return mock.call(
      collections.OrderedDict([
          ('distributor', ()),
          ('client_work', mock.ANY),
          ('aggregator', mock.ANY),
          ('finalizer', ()),
          (evaluation_program_logic.MODEL_METRICS_PREFIX, mock.ANY),
      ]),
      key=key,
  )


def _create_mock_eval_process() -> mock.Mock:
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
      empty_state,
      collections.OrderedDict(
          distributor=(),
          client_work=collections.OrderedDict(
              eval=collections.OrderedDict(
                  current_round_metrics=collections.OrderedDict(),
                  total_rounds_metrics=collections.OrderedDict(),
              )
          ),
          aggregator=(),
          finalizer=(),
      ),
  )
  return mock_process


class EvaluationManagerTest(tf.test.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.maxDiff = None
    context_stack_impl.context_stack.set_default_context(_create_test_context())

  async def test_resume_nothing(self):
    mock_data_source = mock.create_autospec(
        data_source.FederatedDataSource, instance=True, spec_set=True
    )
    mock_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True, spec_set=True
    )
    mock_create_state_manager = mock.Mock()
    mock_meta_eval_manager = mock.create_autospec(
        file_program_state_manager.FileProgramStateManager,
        instance=True,
        spec_set=True,
    )
    mock_meta_eval_manager.load_latest.return_value = (None, 0)
    mock_create_state_manager.side_effect = [mock_meta_eval_manager]
    mock_create_process_fn = mock.Mock()
    manager = evaluation_program_logic.EvaluationManager(
        data_source=mock_data_source,
        aggregated_metrics_manager=mock_metrics_manager,
        create_state_manager_fn=mock_create_state_manager,
        create_process_fn=mock_create_process_fn,
        cohort_size=10,
    )
    await manager.resume_from_previous_state()
    self.assertEmpty(manager._pending_tasks)
    await manager.wait_for_evaluations_to_finish()
    self.assertEmpty(manager._pending_tasks)
    mock_create_process_fn.assert_not_called()
    self.assertSequenceEqual(
        mock_create_state_manager.call_args_list,
        [mock.call(evaluation_program_logic._EVAL_MANAGER_KEY)],
    )
    mock_meta_eval_manager.load_latest.assert_called_once()
    mock_create_process_fn.assert_not_called()

  async def test_start_evaluations(self):
    mock_data_source = mock.create_autospec(
        data_source.FederatedDataSource, instance=True, spec_set=True
    )
    mock_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True, spec_set=True
    )
    # Create a state manager with no previous evaluations.
    mock_meta_eval_manager = mock.create_autospec(
        file_program_state_manager.FileProgramStateManager,
        instance=True,
        spec_set=True,
    )
    mock_meta_eval_manager.load_latest.side_effect = [(
        (np.asarray([]).astype(np.int32), np.asarray([]).astype(np.int32)),
        0,
    )]
    test_train_rounds = [5, 15]
    test_train_times = [
        int(datetime.datetime(2022, 11, 2, 10, 7).timestamp()),
        int(datetime.datetime(2022, 11, 2, 10, 15).timestamp()),
    ]
    mock_eval_managers = [
        mock.create_autospec(
            file_program_state_manager.FileProgramStateManager,
            instance=True,
            spec_set=True,
        ),
        mock.create_autospec(
            file_program_state_manager.FileProgramStateManager,
            instance=True,
            spec_set=True,
        ),
    ]
    for mock_eval_manager in mock_eval_managers:
      mock_eval_manager.load_latest.side_effect = [(
          mock.create_autospec(composers.LearningAlgorithmState, instance=True),
          0,
      )]
    mock_create_state_manager = mock.Mock(
        side_effect=[mock_meta_eval_manager] + mock_eval_managers
    )
    processes = [_create_mock_eval_process(), _create_mock_eval_process()]
    metrics_managers = [
        mock.create_autospec(
            release_manager.ReleaseManager, instance=True, spec_set=True
        ),
        mock.create_autospec(
            release_manager.ReleaseManager, instance=True, spec_set=True
        ),
    ]
    mock_create_process_fn = mock.Mock(
        side_effect=[
            (process, metrics_manager)
            for process, metrics_manager in zip(processes, metrics_managers)
        ]
    )
    manager = evaluation_program_logic.EvaluationManager(
        data_source=mock_data_source,
        aggregated_metrics_manager=mock_metrics_manager,
        create_state_manager_fn=mock_create_state_manager,
        create_process_fn=mock_create_process_fn,
        cohort_size=10,
        duration=datetime.timedelta(milliseconds=10),
    )
    mock_create_state_manager.assert_called_once_with(
        evaluation_program_logic._EVAL_MANAGER_KEY
    )
    mock_create_state_manager.reset_mock()
    self.assertEmpty(manager._pending_tasks)
    await manager.resume_from_previous_state()
    self.assertEmpty(manager._pending_tasks)
    mock_create_process_fn.assert_not_called()
    test_model_weights = model_weights.ModelWeights([], [])
    for train_round, train_time in zip(test_train_rounds, test_train_times):
      await manager.start_evaluation(
          train_round=train_round,
          start_timestamp_seconds=train_time,
          model_weights=test_model_weights,
      )
      # Assert that an evlauation process and state manager was created for
      # the newly started evaluation.
      eval_name = evaluation_program_logic._EVAL_NAME_PATTERN.format(
          round_num=train_round
      )
      self.assertSequenceEqual(
          mock_create_state_manager.call_args_list, [mock.call(eval_name)]
      )
      mock_create_state_manager.reset_mock()
      self.assertSequenceEqual(
          mock_create_process_fn.call_args_list, [mock.call(eval_name)]
      )
      mock_create_process_fn.reset_mock()
    # Expect that there are one or two pending evaluations. Its possible that
    # the first evaluation has completed during `await`s while starting the
    # second evaluation, so we can't explicitly check for two pending tasks
    # here.
    self.assertNotEmpty(manager._pending_tasks)
    await manager.wait_for_evaluations_to_finish()
    self.assertEmpty(manager._pending_tasks)
    # Assert that two meta-state-manager save calls, each one adding the new
    # evaluations.
    self.assertSequenceEqual(
        mock_meta_eval_manager.save.call_args_list,
        [
            # First evaluation started.
            mock.call(
                (
                    _NumpyMatcher(test_train_rounds[0]),
                    _NumpyMatcher([test_train_times[0]]),
                ),
                version=1,
            ),
            # First evaluation ended.
            mock.call((_NumpyMatcher([]), _NumpyMatcher([[]])), version=2),
            # Both evaluations started and in-progress.
            mock.call(
                (
                    _NumpyMatcher(test_train_rounds[1]),
                    _NumpyMatcher([test_train_times[1]]),
                ),
                version=3,
            ),
            # Second evaluation ended.
            mock.call((_NumpyMatcher([]), _NumpyMatcher([[]])), version=4),
        ],
    )

  async def test_record_finished_evaluations_removes_from_state(self):
    mock_data_source = mock.create_autospec(
        data_source.FederatedDataSource, instance=True, spec_set=True
    )
    mock_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True, spec_set=True
    )
    # Create a state manager with two inflight evaluations.
    mock_meta_eval_manager = mock.create_autospec(
        file_program_state_manager.FileProgramStateManager,
        instance=True,
        spec_set=True,
    )
    mock_create_state_manager = mock.Mock(side_effect=[mock_meta_eval_manager])
    mock_create_process_fn = mock.Mock()
    manager = evaluation_program_logic.EvaluationManager(
        data_source=mock_data_source,
        aggregated_metrics_manager=mock_metrics_manager,
        create_state_manager_fn=mock_create_state_manager,
        create_process_fn=mock_create_process_fn,
        cohort_size=10,
        duration=datetime.timedelta(milliseconds=10),
    )
    # Directly set the state, avoid starting asyncio.Task for the resumed evals.
    manager._evaluating_training_checkpoints = np.asarray([5, 15]).astype(
        np.int32
    )
    manager._evaluation_start_timestamp_seconds = np.asarray([
        datetime.datetime(2022, 10, 28, 5, 15).timestamp(),
        datetime.datetime(2022, 10, 28, 5, 25).timestamp(),
    ]).astype(np.int32)
    manager._next_version = 1
    await manager.record_evaluations_finished(5)
    # Only train_round 15 should be saved after 5 finishes.
    self.assertSequenceEqual(
        mock_meta_eval_manager.save.call_args_list,
        [mock.call((_NumpyMatcher([15]), mock.ANY), version=1)],
    )
    mock_meta_eval_manager.reset_mock()
    await manager.record_evaluations_finished(15)
    # No train_rounds should be saved after 15 finishes.
    self.assertSequenceEqual(
        mock_meta_eval_manager.save.call_args_list,
        [mock.call((_NumpyMatcher([]), _NumpyMatcher([])), version=2)],
    )
    with self.assertRaisesRegex(
        RuntimeError, 'An internal error occurred where the EvaluationManager'
    ):
      await manager.record_evaluations_finished(7)
    await manager.wait_for_evaluations_to_finish()
    self.assertEmpty(manager._pending_tasks)

  async def test_resume_previous_evaluations(self):
    mock_data_source = mock.create_autospec(
        data_source.FederatedDataSource, instance=True, spec_set=True
    )
    mock_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True, spec_set=True
    )
    mock_create_state_manager = mock.Mock()
    mock_meta_eval_manager = mock.create_autospec(
        file_program_state_manager.FileProgramStateManager,
        instance=True,
        spec_set=True,
    )
    test_state_time = int(datetime.datetime(2022, 10, 28, 5, 15).timestamp())
    test_train_rounds = [5, 15]
    mock_meta_eval_manager.load_latest.side_effect = [(
        (
            np.asarray(test_train_rounds),
            np.asarray(
                [
                    test_state_time + test_train_round
                    for test_train_round in test_train_rounds
                ]
            ),
        ),
        0,
    )]
    mock_resumed_eval_managers = [
        mock.create_autospec(
            file_program_state_manager.FileProgramStateManager,
            instance=True,
            spec_set=True,
        ),
        mock.create_autospec(
            file_program_state_manager.FileProgramStateManager,
            instance=True,
            spec_set=True,
        ),
    ]

    processes = [_create_mock_eval_process(), _create_mock_eval_process()]
    for index, mock_resumed_eval_manager in enumerate(
        mock_resumed_eval_managers
    ):
      mock_resumed_eval_manager.load_latest.side_effect = [(
          processes[0].initialize(),
          index * 10,
      )]
    mock_create_state_manager.side_effect = [
        mock_meta_eval_manager
    ] + mock_resumed_eval_managers
    metrics_managers = [
        mock.create_autospec(
            release_manager.ReleaseManager, instance=True, spec_set=True
        ),
        mock.create_autospec(
            release_manager.ReleaseManager, instance=True, spec_set=True
        ),
    ]
    mock_create_process_fn = mock.Mock(
        side_effect=[
            (process, metrics_manager)
            for process, metrics_manager in zip(processes, metrics_managers)
        ]
    )
    manager = evaluation_program_logic.EvaluationManager(
        data_source=mock_data_source,
        aggregated_metrics_manager=mock_metrics_manager,
        create_state_manager_fn=mock_create_state_manager,
        create_process_fn=mock_create_process_fn,
        cohort_size=10,
        duration=datetime.timedelta(milliseconds=10),
    )
    self.assertEmpty(manager._pending_tasks)
    await manager.resume_from_previous_state()
    self.assertLen(manager._pending_tasks, len(test_train_rounds))
    await manager.wait_for_evaluations_to_finish()
    self.assertEmpty(manager._pending_tasks)
    eval_names = [
        evaluation_program_logic._EVAL_NAME_PATTERN.format(round_num=round_num)
        for round_num in test_train_rounds
    ]
    self.assertSequenceEqual(
        mock_create_state_manager.call_args_list,
        [mock.call(evaluation_program_logic._EVAL_MANAGER_KEY)]
        + [mock.call(eval_name) for eval_name in eval_names],
    )
    mock_meta_eval_manager.load_latest.assert_called_once()
    self.assertSequenceEqual(
        mock_create_process_fn.call_args_list,
        [mock.call(eval_name) for eval_name in eval_names],
    )
    for index, mock_resumed_eval_manager in enumerate(
        mock_resumed_eval_managers
    ):
      self.assertGreaterEqual(
          len(mock_resumed_eval_manager.save.call_args_list), 1
      )
      # Assert the first call to save is version 1, since version 0 is expected
      # to have already been done before starting the evaluation.
      self.assertEqual(
          mock_resumed_eval_manager.save.call_args_list[0],
          mock.call(mock.ANY, version=(index * 10) + 1),
      )
      mock_resumed_eval_manager.remove_all.assert_called_once()

  async def test_failed_evaluation_raises(self):
    mock_data_source = mock.create_autospec(
        data_source.FederatedDataSource, instance=True, spec_set=True
    )
    mock_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True, spec_set=True
    )
    mock_create_state_manager = mock.Mock()
    mock_create_process_fn = mock.Mock()
    manager = evaluation_program_logic.EvaluationManager(
        data_source=mock_data_source,
        aggregated_metrics_manager=mock_metrics_manager,
        create_state_manager_fn=mock_create_state_manager,
        create_process_fn=mock_create_process_fn,
        cohort_size=10,
        duration=datetime.timedelta(milliseconds=10),
    )

    async def foo():
      raise RuntimeError('Task failed!')

    manager._pending_tasks.clear()
    task = asyncio.create_task(foo())
    manager._pending_tasks.add(task)
    with self.assertRaisesRegex(RuntimeError, 'Task failed!'):
      await manager.wait_for_evaluations_to_finish()


class RunEvaluationTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.maxDiff = None
    self.time_now_called = [False]
    self.end_time = datetime.datetime(2023, 4, 2, 5, 15)
    self.time_before_end = self.end_time - datetime.timedelta(seconds=1)
    self.time_after_end = self.end_time + datetime.timedelta(seconds=1)

    def _return_time() -> datetime.datetime:
      for i in range(len(self.time_now_called)):
        if not self.time_now_called[i]:
          self.time_now_called[i] = True
          return self.time_before_end
      return self.time_after_end

    self._mock_return_time_fn = _return_time
    context_stack_impl.context_stack.set_default_context(_create_test_context())

  async def test_invalid_process_rasies(self):
    @federated_computation.federated_computation
    def empty_initialize():
      return intrinsics.federated_value((), placements.SERVER)

    @federated_computation.federated_computation(
        computation_types.FederatedType((), placements.SERVER),
        computation_types.FederatedType(
            computation_types.SequenceType(()), placements.CLIENTS
        ),
    )
    def next_fn(state, inputs):
      del inputs  # Unused.
      return state

    eval_process = iterative_process.IterativeProcess(empty_initialize, next_fn)
    state_manager = mock.create_autospec(
        file_program_state_manager.FileProgramStateManager, instance=True
    )
    state_manager.load_latest.return_value = (empty_initialize(), 0)
    num_clients = 3
    with self.assertRaisesRegex(
        TypeError, 'Expected a `tff.learning.templates.LearningProcessOutput`'
    ):
      await evaluation_program_logic._run_evaluation(
          train_round_num=1,
          state_manager=state_manager,
          evaluation_process=eval_process,
          evaluation_name='test_evaluation',
          evaluation_data_source=_create_mock_datasource(),
          evaluation_per_round_clients_number=num_clients,
          evaluation_end_time=self.end_time,
          per_round_metrics_manager=mock.create_autospec(
              release_manager.ReleaseManager, instance=True
          ),
          aggregated_metrics_manager=mock.create_autospec(
              release_manager.ReleaseManager, instance=True
          ),
      )

  async def test_no_zero_state_raises(self):
    eval_process = _create_mock_eval_process()
    state_manager = mock.create_autospec(
        file_program_state_manager.FileProgramStateManager, instance=True
    )
    # Return no initial state.
    state_manager.load_latest.return_value = (None, 0)
    num_clients = 3
    mock_per_round_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    mock_aggregated_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    train_round_num = 1
    with self.assertRaisesRegex(
        ValueError, 'No previous state found for evaluation'
    ):
      await evaluation_program_logic._run_evaluation(
          train_round_num=train_round_num,
          state_manager=state_manager,
          evaluation_process=eval_process,
          evaluation_name='test_evaluation',
          evaluation_data_source=_create_mock_datasource(),
          evaluation_per_round_clients_number=num_clients,
          evaluation_end_time=self.end_time,
          per_round_metrics_manager=mock_per_round_metrics_manager,
          aggregated_metrics_manager=mock_aggregated_metrics_manager,
      )

  async def test_passed_end_time_runs_one_round(self):
    eval_process = _create_mock_eval_process()
    state_manager = mock.create_autospec(
        file_program_state_manager.FileProgramStateManager, instance=True
    )
    state_manager.load_latest.return_value = (eval_process.initialize(), 0)
    num_clients = 3
    mock_per_round_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    mock_aggregated_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    train_round_num = 1
    await evaluation_program_logic._run_evaluation(
        train_round_num=train_round_num,
        state_manager=state_manager,
        evaluation_process=eval_process,
        evaluation_name='test_evaluation',
        evaluation_data_source=_create_mock_datasource(),
        evaluation_per_round_clients_number=num_clients,
        evaluation_end_time=self.end_time,
        per_round_metrics_manager=mock_per_round_metrics_manager,
        aggregated_metrics_manager=mock_aggregated_metrics_manager,
    )
    # Assert the evaluation state was loaded from the state manager, then the
    # first round of evaluation was saved.
    self.assertSequenceEqual(
        state_manager.load_latest.call_args_list, [mock.call(mock.ANY)]
    )
    self.assertSequenceEqual(
        state_manager.save.call_args_list, [mock.call(mock.ANY, version=1)]
    )
    # The evaluation end time should be passed when invoking the evaluation, we
    # should expect exactly one evaluation to have occurred for the training
    # round.
    self.assertSequenceEqual(
        [_create_per_round_eval_metrics_release_call(key=train_round_num)],
        mock_per_round_metrics_manager.release.call_args_list,
    )
    # Assert the aggregated metrics are output once at the end.
    self.assertSequenceEqual(
        mock_aggregated_metrics_manager.release.call_args_list,
        [mock.call(mock.ANY, key=train_round_num)],
    )

  async def test_future_end_time_runs_atleast_one_evaluation_round(self):
    eval_process = _create_mock_eval_process()
    state_manager = mock.create_autospec(
        file_program_state_manager.FileProgramStateManager, instance=True
    )
    state_manager.load_latest.return_value = (eval_process.initialize(), 0)
    num_clients = 3
    mock_per_round_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    mock_aggregated_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    train_round_num = 10
    with mock.patch.object(datetime, 'datetime') as m_datetime:
      m_datetime.now.side_effect = self._mock_return_time_fn
      self.time_now_called = [False]
      await evaluation_program_logic._run_evaluation(
          train_round_num=train_round_num,
          state_manager=state_manager,
          evaluation_process=eval_process,
          evaluation_name='test_evaluation',
          evaluation_data_source=_create_mock_datasource(),
          evaluation_per_round_clients_number=num_clients,
          evaluation_end_time=self.end_time,
          per_round_metrics_manager=mock_per_round_metrics_manager,
          aggregated_metrics_manager=mock_aggregated_metrics_manager,
      )
    # Assert the evaluation state was loaded from the state manager, then the
    # first round of evaluation was saved.
    self.assertEqual(
        state_manager.load_latest.call_args_list, [mock.call(mock.ANY)]
    )
    # Assert the evaluation runs two rounds: it always runs one round before
    # checking the current time, and the datatime.now() is called once.
    self.assertEqual(
        state_manager.save.call_args_list,
        [
            mock.call(mock.ANY, version=1),
            mock.call(mock.ANY, version=2),
        ],
    )
    self.assertEqual(
        mock_per_round_metrics_manager.release.call_args_list,
        [
            _create_per_round_eval_metrics_release_call(key=train_round_num),
            _create_per_round_eval_metrics_release_call(
                key=train_round_num + 1
            ),
        ],
    )
    # Assert the aggregated metrics are output once at the end.
    self.assertEqual(
        mock_aggregated_metrics_manager.release.call_args_list,
        [mock.call(mock.ANY, key=train_round_num)],
    )

  async def test_resume_evaluation_uses_correct_eval_round(self):
    eval_process = _create_mock_eval_process()
    state_manager = mock.create_autospec(
        file_program_state_manager.FileProgramStateManager, instance=True
    )
    # Setup a state with a version later than zero.
    latest_version = 5
    state_manager.load_latest.return_value = (
        eval_process.initialize(),
        latest_version,
    )
    num_clients = 3
    mock_per_round_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    mock_aggregated_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    train_round_num = 10
    with mock.patch.object(datetime, 'datetime') as m_datetime:
      m_datetime.now.side_effect = self._mock_return_time_fn
      self.time_now_called = [False]
      await evaluation_program_logic._run_evaluation(
          train_round_num=train_round_num,
          state_manager=state_manager,
          evaluation_process=eval_process,
          evaluation_name='test_evaluation',
          evaluation_data_source=_create_mock_datasource(),
          evaluation_per_round_clients_number=num_clients,
          evaluation_end_time=self.end_time,
          per_round_metrics_manager=mock_per_round_metrics_manager,
          aggregated_metrics_manager=mock_aggregated_metrics_manager,
      )
    # Assert the evaluation state was loaded from the state manager, then the
    # first round of evaluation was saved.
    self.assertEqual(
        state_manager.load_latest.call_args_list, [mock.call(mock.ANY)]
    )
    # Assert the evaluation runs two rounds: it always runs one round before
    # checking the current time, and the datatime.now() is called once.
    self.assertEqual(
        state_manager.save.call_args_list,
        [
            mock.call(mock.ANY, version=latest_version + 1),
            mock.call(mock.ANY, version=latest_version + 2),
        ],
    )
    self.assertEqual(
        mock_per_round_metrics_manager.release.call_args_list,
        [
            _create_per_round_eval_metrics_release_call(
                key=train_round_num + latest_version
            ),
            _create_per_round_eval_metrics_release_call(
                key=train_round_num + latest_version + 1
            ),
        ],
    )
    # Assert the aggregated metrics are output once at the end.
    self.assertEqual(
        mock_aggregated_metrics_manager.release.call_args_list,
        [mock.call(mock.ANY, key=train_round_num)],
    )

  async def test_resume_evaluation_uses_correct_end_time(self):
    eval_process = _create_mock_eval_process()
    state_manager = mock.create_autospec(
        file_program_state_manager.FileProgramStateManager, instance=True
    )
    # Setup a state with a version later than zero.
    latest_version = 5
    state_manager.load_latest.return_value = (
        eval_process.initialize(),
        latest_version,
    )
    num_clients = 3
    mock_per_round_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    mock_aggregated_metrics_manager = mock.create_autospec(
        release_manager.ReleaseManager, instance=True
    )
    train_round_num = 10
    await evaluation_program_logic._run_evaluation(
        train_round_num=train_round_num,
        state_manager=state_manager,
        evaluation_process=eval_process,
        evaluation_name='test_evaluation',
        evaluation_data_source=_create_mock_datasource(),
        evaluation_per_round_clients_number=num_clients,
        evaluation_end_time=self.end_time,
        per_round_metrics_manager=mock_per_round_metrics_manager,
        aggregated_metrics_manager=mock_aggregated_metrics_manager,
    )
    # Assert the evaluation state was loaded from the state manager, then the
    # first round of evaluation was saved.
    self.assertEqual(
        state_manager.load_latest.call_args_list, [mock.call(mock.ANY)]
    )
    # The evaluation end time should be passed when invoking the evaluation, we
    # should expect exactly one evaluation to have occurred for the training
    # round.
    state_manager.save.assert_has_calls([
        mock.call(mock.ANY, version=latest_version + 1),
    ])
    mock_per_round_metrics_manager.release.assert_has_calls([
        _create_per_round_eval_metrics_release_call(
            key=train_round_num + latest_version
        ),
    ])
    # Assert the aggregated metrics are output once at the end.
    self.assertEqual(
        mock_aggregated_metrics_manager.release.call_args_list,
        [mock.call(mock.ANY, key=train_round_num)],
    )


if __name__ == '__main__':
  absltest.main()
