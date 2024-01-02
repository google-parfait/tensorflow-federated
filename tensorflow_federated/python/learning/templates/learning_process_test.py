# Copyright 2021, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.learning.templates import learning_process

# Convenience aliases.
LearningProcessOutput = learning_process.LearningProcessOutput
SequenceType = computation_types.SequenceType
TensorType = computation_types.TensorType
federated_computation = federated_computation.federated_computation
tf_computation = tensorflow_computation.tf_computation

_LearningProcessConstructionError = (
    TypeError,
    errors.TemplateInitFnParamNotEmptyError,
    errors.TemplateNextFnNumArgsError,
    errors.TemplateStateNotAssignableError,
    learning_process.GetModelWeightsTypeSignatureError,
    learning_process.LearningProcessOutputError,
    learning_process.LearningProcessPlacementError,
    learning_process.SetModelWeightsTypeSignatureError,
)


def create_pass_through_get_model_weights(state_type):
  @federated_computation(state_type)
  def pass_through_get_model_weights_fn(state):
    return state

  return pass_through_get_model_weights_fn


def create_take_arg_set_model_weights(state_type, model_type):
  @federated_computation(state_type, model_type)
  def take_arg_set_model_weights(state, model_weights):
    del state  # Unused.
    return model_weights

  return take_arg_set_model_weights


@federated_computation
def test_init_fn():
  return intrinsics.federated_value(0, placements.SERVER)


@tf_computation(SequenceType(np.int32))
@tf.function
def sum_dataset(dataset):
  total = tf.zeros(
      shape=dataset.element_spec.shape, dtype=dataset.element_spec.dtype
  )
  for i in iter(dataset):
    total += i
  return total


@federated_computation(
    computation_types.FederatedType(np.int32, placements.SERVER),
    computation_types.FederatedType(SequenceType(np.int32), placements.CLIENTS),
)
def test_next_fn(state, data):
  client_sums = intrinsics.federated_map(sum_dataset, data)
  server_sum = intrinsics.federated_sum(client_sums)

  @tf_computation
  def add(x, y):
    """Function to hide `tf.add`'s `name` parameter from TFF."""
    return tf.add(x, y)

  result = intrinsics.federated_map(add, (state, server_sum))
  return LearningProcessOutput(
      state=result, metrics=intrinsics.federated_value((), placements.SERVER)
  )


test_get_model_weights_fn = create_pass_through_get_model_weights(np.int32)
test_set_model_weights_fn = create_take_arg_set_model_weights(
    np.int32, np.int32
)


class LearningProcessTest(absltest.TestCase):

  def test_construction_does_not_raise(self):
    try:
      learning_process.LearningProcess(
          test_init_fn,
          test_next_fn,
          test_get_model_weights_fn,
          test_set_model_weights_fn,
      )
    except _LearningProcessConstructionError:
      self.fail('Could not construct a valid LearningProcess.')

  def test_learning_process_can_be_reconstructed(self):
    process = learning_process.LearningProcess(
        test_init_fn,
        test_next_fn,
        test_get_model_weights_fn,
        test_set_model_weights_fn,
    )
    try:
      learning_process.LearningProcess(
          process.initialize,
          process.next,
          process.get_model_weights,
          process.set_model_weights,
      )
    except _LearningProcessConstructionError:
      self.fail('Could not reconstruct the LearningProcess.')

  def test_construction_with_empty_state_does_not_raise(self):
    empty_tuple = ()

    @federated_computation
    def empty_initialize_fn():
      return intrinsics.federated_value(empty_tuple, placements.SERVER)

    @federated_computation(
        computation_types.FederatedType(empty_tuple, placements.SERVER),
        computation_types.FederatedType(
            SequenceType(np.int32), placements.CLIENTS
        ),
    )
    def next_fn(state, value):
      del value  # Unused.
      return LearningProcessOutput(
          state=state,
          metrics=intrinsics.federated_value(empty_tuple, placements.SERVER),
      )

    try:
      learning_process.LearningProcess(
          empty_initialize_fn,
          next_fn,
          create_pass_through_get_model_weights(empty_tuple),
          create_take_arg_set_model_weights(empty_tuple, empty_tuple),
      )
    except _LearningProcessConstructionError:
      self.fail('Could not construct a LearningProcess with empty state.')

  def test_construction_with_unknown_dimension_does_not_raise(self):
    @federated_computation
    def initialize_fn():
      return intrinsics.federated_eval(
          tf_computation(lambda: tf.constant([], tf.string)), placements.SERVER
      )

    # This replicates a tensor that can grow in string length. The
    # `initialize_fn` will concretely start with shape `[0]`, but `next_fn` will
    # grow this, hence the need to define the shape as `[None]`.
    none_dimension_string_type = TensorType(np.str_, [None])

    @federated_computation(
        computation_types.FederatedType(
            none_dimension_string_type, placements.SERVER
        ),
        computation_types.FederatedType(
            SequenceType(np.str_), placements.CLIENTS
        ),
    )
    def next_fn(state, datasets):
      del datasets  # Unused.
      return LearningProcessOutput(
          state, intrinsics.federated_value((), placements.SERVER)
      )

    try:
      learning_process.LearningProcess(
          initialize_fn,
          next_fn,
          create_pass_through_get_model_weights(none_dimension_string_type),
          create_take_arg_set_model_weights(
              none_dimension_string_type, none_dimension_string_type
          ),
      )
    except _LearningProcessConstructionError:
      self.fail(
          'Could not construct a LearningProcess with state type having '
          'statically unknown shape.'
      )

  def test_construction_with_nested_datasets_does_not_raise(self):
    @federated_computation
    def initialize_fn():
      return intrinsics.federated_eval(
          tf_computation(lambda: tf.constant(0.0, tf.float32)),
          placements.SERVER,
      )

    # Test that clients can receive multiple datasets.
    datasets_type = (
        SequenceType(np.str_),
        (SequenceType(np.str_), SequenceType(np.str_)),
    )

    @federated_computation(
        computation_types.FederatedType(np.float32, placements.SERVER),
        computation_types.FederatedType(datasets_type, placements.CLIENTS),
    )
    def next_fn(state, datasets):
      del datasets  # Unused.
      return LearningProcessOutput(
          state, intrinsics.federated_value((), placements.SERVER)
      )

    try:
      learning_process.LearningProcess(
          initialize_fn,
          next_fn,
          create_pass_through_get_model_weights(np.float32),
          create_take_arg_set_model_weights(np.float32, np.float32),
      )
    except learning_process.LearningProcessSequenceTypeError:
      self.fail(
          'Could not construct a LearningProcess with second parameter '
          'type having nested sequences.'
      )

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      init_fn = lambda: 0
      learning_process.LearningProcess(
          init_fn,
          test_next_fn,
          test_get_model_weights_fn,
          test_set_model_weights_fn,
      )

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      learning_process.LearningProcess(
          initialize_fn=test_init_fn,
          next_fn=lambda state, client_data: LearningProcessOutput(state, ()),
          get_model_weights=test_get_model_weights_fn,
          set_model_weights=test_set_model_weights_fn,
      )

  def test_init_param_not_empty_raises(self):

    @federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER)
    )
    def one_arg_initialize_fn(x):
      return x

    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      learning_process.LearningProcess(
          one_arg_initialize_fn,
          test_next_fn,
          test_get_model_weights_fn,
          test_set_model_weights_fn,
      )

  def test_init_state_not_federated(self):
    @federated_computation
    def float_initialize_fn():
      return 0.0

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      learning_process.LearningProcess(
          float_initialize_fn,
          test_next_fn,
          test_get_model_weights_fn,
          test_set_model_weights_fn,
      )

  def test_next_state_not_federated(self):

    float_next_fn = create_pass_through_get_model_weights(np.float32)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      learning_process.LearningProcess(
          test_init_fn,
          float_next_fn,
          test_get_model_weights_fn,
          test_set_model_weights_fn,
      )

  def test_init_fn_with_client_placed_state_raises(self):
    @federated_computation
    def init_fn():
      return intrinsics.federated_value(0, placements.CLIENTS)

    @federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS),
        computation_types.FederatedType(
            SequenceType(np.int32), placements.CLIENTS
        ),
    )
    def next_fn(state, client_values):
      return LearningProcessOutput(state, client_values)

    with self.assertRaises(learning_process.LearningProcessPlacementError):
      learning_process.LearningProcess(
          init_fn, next_fn, test_get_model_weights_fn, test_set_model_weights_fn
      )

  def test_next_return_tuple_raises(self):

    @federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(
            SequenceType(np.int32), placements.CLIENTS
        ),
    )
    def tuple_next_fn(state, client_values):
      metrics = intrinsics.federated_map(sum_dataset, client_values)
      metrics = intrinsics.federated_sum(metrics)
      return (state, metrics)

    with self.assertRaises(learning_process.LearningProcessOutputError):
      learning_process.LearningProcess(
          test_init_fn,
          tuple_next_fn,
          test_get_model_weights_fn,
          test_set_model_weights_fn,
      )

  def test_next_return_namedtuple_raises(self):
    learning_process_output = collections.namedtuple(
        'LearningProcessOutput', ['state', 'metrics']
    )

    @federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(
            SequenceType(np.int32), placements.CLIENTS
        ),
    )
    def namedtuple_next_fn(state, client_values):
      metrics = intrinsics.federated_map(sum_dataset, client_values)
      metrics = intrinsics.federated_sum(metrics)
      return learning_process_output(state, metrics)

    with self.assertRaises(learning_process.LearningProcessOutputError):
      learning_process.LearningProcess(
          test_init_fn,
          namedtuple_next_fn,
          test_get_model_weights_fn,
          test_set_model_weights_fn,
      )

  def test_next_return_odict_raises(self):

    @federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(
            SequenceType(np.int32), placements.CLIENTS
        ),
    )
    def odict_next_fn(state, client_values):
      metrics = intrinsics.federated_map(sum_dataset, client_values)
      metrics = intrinsics.federated_sum(metrics)
      return collections.OrderedDict(state=state, metrics=metrics)

    with self.assertRaises(learning_process.LearningProcessOutputError):
      learning_process.LearningProcess(
          test_init_fn,
          odict_next_fn,
          test_get_model_weights_fn,
          test_set_model_weights_fn,
      )

  def test_next_fn_with_one_parameter_raises(self):

    @federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER)
    )
    def next_fn(state):
      return LearningProcessOutput(state, 0)

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      learning_process.LearningProcess(
          test_init_fn,
          next_fn,
          test_get_model_weights_fn,
          test_set_model_weights_fn,
      )

  def test_next_fn_with_three_parameters_raises(self):

    @federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(
            SequenceType(np.int32), placements.CLIENTS
        ),
        computation_types.FederatedType(np.int32, placements.SERVER),
    )
    def next_fn(state, client_values, second_state):
      del second_state  # Unused.
      metrics = intrinsics.federated_map(sum_dataset, client_values)
      metrics = intrinsics.federated_sum(metrics)
      return LearningProcessOutput(state, metrics)

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      learning_process.LearningProcess(
          test_init_fn,
          next_fn,
          test_get_model_weights_fn,
          test_set_model_weights_fn,
      )

  def test_next_fn_with_server_placed_second_arg_raises(self):

    @federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(
            SequenceType(np.int32), placements.SERVER
        ),
    )
    def next_fn(state, server_values):
      metrics = intrinsics.federated_map(sum_dataset, server_values)
      return LearningProcessOutput(state, metrics)

    with self.assertRaises(learning_process.LearningProcessPlacementError):
      learning_process.LearningProcess(
          test_init_fn,
          next_fn,
          test_get_model_weights_fn,
          test_set_model_weights_fn,
      )

  def test_next_fn_with_client_placed_metrics_result_raises(self):

    @federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(
            SequenceType(np.int32), placements.CLIENTS
        ),
    )
    def next_fn(state, metrics):
      return LearningProcessOutput(state, metrics)

    with self.assertRaises(learning_process.LearningProcessPlacementError):
      learning_process.LearningProcess(
          test_init_fn,
          next_fn,
          test_get_model_weights_fn,
          test_set_model_weights_fn,
      )

  def test_non_tff_computation_get_model_weights_raises(self):
    get_model_weights = lambda x: x
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      learning_process.LearningProcess(
          test_init_fn,
          test_next_fn,
          get_model_weights,
          test_set_model_weights_fn,
      )

  def test_non_functional_get_model_weights_raises(self):
    get_model_weights = computation_types.FederatedType(
        np.int32, placements.SERVER
    )
    with self.assertRaises(TypeError):
      learning_process.LearningProcess(
          test_init_fn,
          test_next_fn,
          get_model_weights,
          test_set_model_weights_fn,
      )

  def test_federated_get_model_weights_raises(self):
    bad_get_model_weights = create_pass_through_get_model_weights(
        computation_types.FederatedType(np.float32, placements.SERVER)
    )
    with self.assertRaises(learning_process.GetModelWeightsTypeSignatureError):
      learning_process.LearningProcess(
          test_init_fn,
          test_next_fn,
          bad_get_model_weights,
          test_set_model_weights_fn,
      )

  def test_get_model_weights_param_not_equivalent_to_next_fn(self):
    bad_get_model_weights = create_pass_through_get_model_weights(np.float32)
    with self.assertRaises(learning_process.GetModelWeightsTypeSignatureError):
      learning_process.LearningProcess(
          test_init_fn,
          test_next_fn,
          bad_get_model_weights,
          test_set_model_weights_fn,
      )

  def test_set_model_weights_param_not_equivalent_to_next_fn(self):
    bad_set_model_weights_fn = create_take_arg_set_model_weights(
        np.float32, np.int32
    )
    with self.assertRaises(learning_process.SetModelWeightsTypeSignatureError):
      learning_process.LearningProcess(
          test_init_fn,
          test_next_fn,
          test_get_model_weights_fn,
          bad_set_model_weights_fn,
      )

  def test_set_model_weights_result_not_assignable(self):
    bad_set_model_weights_fn = create_take_arg_set_model_weights(
        np.int32, np.float32
    )
    with self.assertRaises(learning_process.SetModelWeightsTypeSignatureError):
      learning_process.LearningProcess(
          test_init_fn,
          test_next_fn,
          test_get_model_weights_fn,
          bad_set_model_weights_fn,
      )


if __name__ == '__main__':
  absltest.main()
