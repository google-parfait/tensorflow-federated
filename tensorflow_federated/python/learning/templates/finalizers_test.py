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

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.learning.templates import hparams_base

SERVER_INT = computation_types.FederatedType(np.int32, placements.SERVER)
SERVER_FLOAT = computation_types.FederatedType(np.float32, placements.SERVER)
CLIENTS_INT = computation_types.FederatedType(np.int32, placements.CLIENTS)
CLIENTS_FLOAT = computation_types.FederatedType(np.float32, placements.CLIENTS)
MODEL_WEIGHTS_TYPE = computation_types.FederatedType(
    computation_types.to_type(
        model_weights.ModelWeights(np.float32, np.float32)
    ),
    placements.SERVER,
)
MeasuredProcessOutput = measured_process.MeasuredProcessOutput

_FinalizerProcessConstructionError = (
    errors.TemplateNextFnNumArgsError,
    errors.TemplateNotFederatedError,
    errors.TemplatePlacementError,
    finalizers.FinalizerResultTypeError,
    hparams_base.GetHparamsTypeError,
    hparams_base.SetHparamsTypeError,
)


def server_zero():
  """Returns zero integer placed at SERVER."""
  return intrinsics.federated_value(0, placements.SERVER)


def federated_add(a, b):
  return intrinsics.federated_map(
      tensorflow_computation.tf_computation(lambda x, y: x + y), (a, b)
  )


@federated_computation.federated_computation()
def test_initialize_fn():
  return server_zero()


def test_finalizer_result(weights, update):
  return intrinsics.federated_zip(
      model_weights.ModelWeights(
          federated_add(weights.trainable, update), weights.non_trainable
      )
  )


@federated_computation.federated_computation(
    SERVER_INT, MODEL_WEIGHTS_TYPE, SERVER_FLOAT
)
def test_next_fn(state, weights, update):
  return MeasuredProcessOutput(
      state,
      test_finalizer_result(weights, update),
      intrinsics.federated_value(1, placements.SERVER),
  )


class FinalizerTest(tf.test.TestCase):

  def test_construction_does_not_raise(self):
    try:
      finalizers.FinalizerProcess(test_initialize_fn, test_next_fn)
    except _FinalizerProcessConstructionError:
      self.fail('Could not construct a valid FinalizerProcess.')

  def test_construction_with_empty_state_does_not_raise(self):
    initialize_fn = federated_computation.federated_computation()(
        lambda: intrinsics.federated_value((), placements.SERVER)
    )
    model_weights_type = computation_types.StructWithPythonType(
        [('trainable', np.float32), ('non_trainable', ())],
        model_weights.ModelWeights,
    )
    server_model_weights_type = computation_types.FederatedType(
        model_weights_type, placements.SERVER
    )

    @federated_computation.federated_computation(
        initialize_fn.type_signature.result,
        server_model_weights_type,
        SERVER_FLOAT,
    )
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state,
          test_finalizer_result(weights, update),
          intrinsics.federated_value(1, placements.SERVER),
      )

    try:
      finalizers.FinalizerProcess(initialize_fn, next_fn)
    except _FinalizerProcessConstructionError:
      self.fail('Could not construct an FinalizerProcess with empty state.')

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      finalizers.FinalizerProcess(initialize_fn=lambda: 0, next_fn=test_next_fn)

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      finalizers.FinalizerProcess(
          initialize_fn=test_initialize_fn,
          next_fn=lambda state, w, u: MeasuredProcessOutput(state, w + u, ()),
      )

  def test_init_param_not_empty_raises(self):
    one_arg_initialize_fn = federated_computation.federated_computation(
        SERVER_INT
    )(lambda x: x)
    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      finalizers.FinalizerProcess(one_arg_initialize_fn, test_next_fn)

  def test_init_state_not_assignable(self):
    float_initialize_fn = federated_computation.federated_computation()(
        lambda: intrinsics.federated_value(0.0, placements.SERVER)
    )
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      finalizers.FinalizerProcess(float_initialize_fn, test_next_fn)

  def test_next_state_not_assignable(self):
    @federated_computation.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, SERVER_FLOAT
    )
    def float_next_fn(state, weights, update):
      del state
      return MeasuredProcessOutput(
          intrinsics.federated_value(0.0, placements.SERVER),
          test_finalizer_result(weights, update),
          intrinsics.federated_value(1, placements.SERVER),
      )

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      finalizers.FinalizerProcess(test_initialize_fn, float_next_fn)

  def test_next_return_tuple_raises(self):
    @federated_computation.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, SERVER_FLOAT
    )
    def tuple_next_fn(state, weights, update):
      return state, test_finalizer_result(weights, update), server_zero()

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      finalizers.FinalizerProcess(test_initialize_fn, tuple_next_fn)

  def test_next_return_namedtuple_raises(self):
    measured_process_output = collections.namedtuple(
        'MeasuredProcessOutput', ['state', 'result', 'measurements']
    )

    @federated_computation.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, SERVER_FLOAT
    )
    def namedtuple_next_fn(state, weights, update):
      return measured_process_output(
          state, test_finalizer_result(weights, update), server_zero()
      )

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      finalizers.FinalizerProcess(test_initialize_fn, namedtuple_next_fn)

  def test_next_return_odict_raises(self):
    @federated_computation.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, SERVER_FLOAT
    )
    def odict_next_fn(state, weights, update):
      return collections.OrderedDict(
          state=state,
          result=test_finalizer_result(weights, update),
          measurements=server_zero(),
      )

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      finalizers.FinalizerProcess(test_initialize_fn, odict_next_fn)

  # Tests specific only for the FinalizerProcess contract below.

  def test_non_federated_init_next_raises(self):
    initialize_fn = tensorflow_computation.tf_computation(lambda: 0)
    model_weights_type = computation_types.StructWithPythonType(
        [('trainable', np.float32), ('non_trainable', ())],
        model_weights.ModelWeights,
    )

    @tensorflow_computation.tf_computation(
        np.int32,
        model_weights_type,
        np.float32,
    )
    def next_fn(state, weights, update):
      new_weigths = model_weights.ModelWeights(weights.trainable + update, ())
      return MeasuredProcessOutput(state, new_weigths, 0)

    with self.assertRaises(errors.TemplateNotFederatedError):
      finalizers.FinalizerProcess(initialize_fn, next_fn)

  def test_init_tuple_of_federated_types_raises(self):
    initialize_fn = federated_computation.federated_computation()(
        lambda: (server_zero(), server_zero())
    )

    @federated_computation.federated_computation(
        initialize_fn.type_signature.result, MODEL_WEIGHTS_TYPE, SERVER_FLOAT
    )
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state, test_finalizer_result(weights, update), server_zero()
      )

    with self.assertRaises(errors.TemplateNotFederatedError):
      finalizers.FinalizerProcess(initialize_fn, next_fn)

  def test_non_server_placed_init_state_raises(self):
    initialize_fn = federated_computation.federated_computation(
        lambda: intrinsics.federated_value(0, placements.CLIENTS)
    )

    @federated_computation.federated_computation(
        CLIENTS_INT, MODEL_WEIGHTS_TYPE, SERVER_FLOAT
    )
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state, test_finalizer_result(weights, update), server_zero()
      )

    with self.assertRaises(errors.TemplatePlacementError):
      finalizers.FinalizerProcess(initialize_fn, next_fn)

  def test_two_param_next_raises(self):
    @federated_computation.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE)
    def next_fn(state, weights):
      return MeasuredProcessOutput(state, weights, server_zero())

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_weight_param_raises(self):

    @federated_computation.federated_computation(
        SERVER_INT,
        computation_types.FederatedType(
            MODEL_WEIGHTS_TYPE.member, placements.CLIENTS
        ),
        SERVER_FLOAT,
    )
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state,
          test_finalizer_result(intrinsics.federated_sum(weights), update),
          server_zero(),
      )

    with self.assertRaises(errors.TemplatePlacementError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)

  def test_constructs_with_non_model_weights_parameter(self):
    non_model_weights_type = computation_types.FederatedType(
        computation_types.to_type(
            collections.OrderedDict(trainable=np.float32, non_trainable=())
        ),
        placements.SERVER,
    )

    @federated_computation.federated_computation(
        SERVER_INT, non_model_weights_type, SERVER_FLOAT
    )
    def next_fn(state, weights, update):
      del update
      return MeasuredProcessOutput(state, weights, server_zero())

    try:
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)
    except _FinalizerProcessConstructionError:
      self.fail('Could not construct a valid FinalizerProcess.')

  def test_non_server_placed_next_update_param_raises(self):
    @federated_computation.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, CLIENTS_FLOAT
    )
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state,
          test_finalizer_result(weights, intrinsics.federated_sum(update)),
          server_zero(),
      )

    with self.assertRaises(errors.TemplatePlacementError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_result_raises(self):
    @federated_computation.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, SERVER_FLOAT
    )
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state,
          intrinsics.federated_broadcast(
              test_finalizer_result(weights, update)
          ),
          server_zero(),
      )

    with self.assertRaises(errors.TemplatePlacementError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)

  def test_result_not_assignable_to_weight_raises(self):
    bad_cast_fn = tensorflow_computation.tf_computation(
        lambda x: tf.nest.map_structure(lambda y: tf.cast(y, tf.float64), x)
    )

    @federated_computation.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, SERVER_FLOAT
    )
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state,
          intrinsics.federated_map(
              bad_cast_fn, test_finalizer_result(weights, update)
          ),
          server_zero(),
      )

    with self.assertRaises(finalizers.FinalizerResultTypeError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_measurements_raises(self):
    @federated_computation.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, SERVER_FLOAT
    )
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state,
          test_finalizer_result(weights, update),
          intrinsics.federated_value(1.0, placements.CLIENTS),
      )

    with self.assertRaises(errors.TemplatePlacementError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)


if __name__ == '__main__':
  tf.test.main()
