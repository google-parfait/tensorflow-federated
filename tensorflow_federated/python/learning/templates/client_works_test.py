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
import federated_language
import numpy as np

from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import hparams_base

SERVER_INT = federated_language.FederatedType(
    np.int32, federated_language.SERVER
)
SERVER_FLOAT = federated_language.FederatedType(
    np.float32, federated_language.SERVER
)
CLIENTS_FLOAT_SEQUENCE = federated_language.FederatedType(
    federated_language.SequenceType(np.float32), federated_language.CLIENTS
)
CLIENTS_FLOAT = federated_language.FederatedType(
    np.float32, federated_language.CLIENTS
)
CLIENTS_INT = federated_language.FederatedType(
    np.int32, federated_language.CLIENTS
)
MODEL_WEIGHTS_TYPE = federated_language.FederatedType(
    federated_language.to_type(
        model_weights.ModelWeights(np.float32, np.float32)
    ),
    federated_language.CLIENTS,
)
HPARAMS_TYPE = federated_language.to_type(collections.OrderedDict(a=np.int32))
MeasuredProcessOutput = measured_process.MeasuredProcessOutput

_IterativeProcessConstructionError = (
    TypeError,
    client_works.ClientDataTypeError,
    client_works.ClientResultTypeError,
    errors.TemplateNextFnNumArgsError,
    errors.TemplateNotFederatedError,
    errors.TemplatePlacementError,
    hparams_base.GetHparamsTypeError,
    hparams_base.SetHparamsTypeError,
)


def server_zero():
  return federated_language.federated_value(0, federated_language.SERVER)


def client_one():
  return federated_language.federated_value(1.0, federated_language.CLIENTS)


def federated_add(a, b):
  return federated_language.federated_map(
      tensorflow_computation.tf_computation(lambda x, y: x + y), (a, b)
  )


@tensorflow_computation.tf_computation()
def tf_data_sum(data):
  return data.reduce(0.0, lambda x, y: x + y)


@federated_language.federated_computation()
def test_initialize_fn():
  return server_zero()


def test_client_result(weights, data):
  reduced_data = federated_language.federated_map(tf_data_sum, data)
  return federated_language.federated_zip(
      client_works.ClientResult(
          update=federated_add(weights.trainable, reduced_data),
          update_weight=client_one(),
      )
  )


@federated_language.federated_computation(
    SERVER_INT, MODEL_WEIGHTS_TYPE, CLIENTS_FLOAT_SEQUENCE
)
def test_next_fn(state, weights, data):
  return MeasuredProcessOutput(
      state,
      test_client_result(weights, data),
      federated_language.federated_value(1, federated_language.SERVER),
  )


@tensorflow_computation.tf_computation(np.int32)
def test_get_hparams_fn(state):
  return collections.OrderedDict(a=state)


@tensorflow_computation.tf_computation(np.int32, HPARAMS_TYPE)
def test_set_hparams_fn(state, hparams):
  del state
  return hparams['a']


class ClientWorkTest(absltest.TestCase):

  def test_construction_does_not_raise(self):
    try:
      client_works.ClientWorkProcess(test_initialize_fn, test_next_fn)
    except _IterativeProcessConstructionError:
      self.fail('Could not construct a valid ClientWorkProcess.')

  def test_construction_with_empty_state_does_not_raise(self):
    initialize_fn = federated_language.federated_computation()(
        lambda: federated_language.federated_value(
            (), federated_language.SERVER
        )
    )

    @federated_language.federated_computation(
        initialize_fn.type_signature.result,
        MODEL_WEIGHTS_TYPE,
        CLIENTS_FLOAT_SEQUENCE,
    )
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state,
          test_client_result(weights, data),
          federated_language.federated_value(1, federated_language.SERVER),
      )

    try:
      client_works.ClientWorkProcess(initialize_fn, next_fn)
    except _IterativeProcessConstructionError:
      self.fail('Could not construct an ClientWorkProcess with empty state.')

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      client_works.ClientWorkProcess(
          initialize_fn=lambda: 0, next_fn=test_next_fn
      )

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      client_works.ClientWorkProcess(
          initialize_fn=test_initialize_fn,
          next_fn=lambda state, w, d: MeasuredProcessOutput(state, w + d, ()),
      )

  def test_get_hparams_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      client_works.ClientWorkProcess(
          initialize_fn=test_initialize_fn,
          next_fn=test_next_fn,
          get_hparams_fn=lambda x: 0,
      )

  def test_set_hparams_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      client_works.ClientWorkProcess(
          initialize_fn=test_initialize_fn,
          next_fn=test_next_fn,
          set_hparams_fn=lambda x: 0,
      )

  def test_init_param_not_empty_raises(self):
    one_arg_initialize_fn = federated_language.federated_computation(
        SERVER_INT
    )(lambda x: x)
    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      client_works.ClientWorkProcess(one_arg_initialize_fn, test_next_fn)

  def test_init_state_not_assignable(self):
    float_initialize_fn = federated_language.federated_computation()(
        lambda: federated_language.federated_value(
            0.0, federated_language.SERVER
        )
    )
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      client_works.ClientWorkProcess(float_initialize_fn, test_next_fn)

  def test_next_state_not_assignable(self):

    @federated_language.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, CLIENTS_FLOAT_SEQUENCE
    )
    def float_next_fn(state, weights, data):
      del state
      return MeasuredProcessOutput(
          federated_language.federated_value(0.0, federated_language.SERVER),
          test_client_result(weights, data),
          federated_language.federated_value(1, federated_language.SERVER),
      )

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      client_works.ClientWorkProcess(test_initialize_fn, float_next_fn)

  def test_next_return_tuple_raises(self):

    @federated_language.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, CLIENTS_FLOAT_SEQUENCE
    )
    def tuple_next_fn(state, weights, data):
      return (state, test_client_result(weights, data), server_zero())

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      client_works.ClientWorkProcess(test_initialize_fn, tuple_next_fn)

  def test_next_return_namedtuple_raises(self):
    measured_process_output = collections.namedtuple(
        'MeasuredProcessOutput', ['state', 'result', 'measurements']
    )

    @federated_language.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, CLIENTS_FLOAT_SEQUENCE
    )
    def namedtuple_next_fn(state, weights, data):
      return measured_process_output(
          state, test_client_result(weights, data), server_zero()
      )

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      client_works.ClientWorkProcess(test_initialize_fn, namedtuple_next_fn)

  def test_next_return_odict_raises(self):

    @federated_language.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, CLIENTS_FLOAT_SEQUENCE
    )
    def odict_next_fn(state, weights, data):
      return collections.OrderedDict(
          state=state,
          result=test_client_result(weights, data),
          measurements=server_zero(),
      )

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      client_works.ClientWorkProcess(test_initialize_fn, odict_next_fn)

  # Tests specific only for the ClientWorkProcess contract below.

  def test_non_federated_init_next_raises(self):
    initialize_fn = tensorflow_computation.tf_computation(lambda: 0)

    @tensorflow_computation.tf_computation(
        np.int32,
        MODEL_WEIGHTS_TYPE.member,
        federated_language.SequenceType(np.float32),
    )
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state,
          client_works.ClientResult(weights.trainable + tf_data_sum(data), ()),
          (),
      )

    with self.assertRaises(errors.TemplateNotFederatedError):
      client_works.ClientWorkProcess(initialize_fn, next_fn)

  def test_init_tuple_of_federated_types_raises(self):
    initialize_fn = federated_language.federated_computation()(
        lambda: (server_zero(), server_zero())
    )

    @federated_language.federated_computation(
        initialize_fn.type_signature.result,
        MODEL_WEIGHTS_TYPE,
        CLIENTS_FLOAT_SEQUENCE,
    )
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state, test_client_result(weights, data), server_zero()
      )

    with self.assertRaises(errors.TemplateNotFederatedError):
      client_works.ClientWorkProcess(initialize_fn, next_fn)

  def test_non_server_placed_init_state_raises(self):
    initialize_fn = federated_language.federated_computation(
        lambda: federated_language.federated_value(
            0, federated_language.CLIENTS
        )
    )

    @federated_language.federated_computation(
        initialize_fn.type_signature.result,
        MODEL_WEIGHTS_TYPE,
        CLIENTS_FLOAT_SEQUENCE,
    )
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state, test_client_result(weights, data), server_zero()
      )

    with self.assertRaises(errors.TemplatePlacementError):
      client_works.ClientWorkProcess(initialize_fn, next_fn)

  def test_two_param_next_raises(self):

    @federated_language.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE)
    def next_fn(state, weights):
      return MeasuredProcessOutput(state, weights.trainable, server_zero())

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_non_clients_placed_next_weights_param_raises(self):

    @federated_language.federated_computation(
        SERVER_INT,
        federated_language.FederatedType(
            MODEL_WEIGHTS_TYPE.member, federated_language.SERVER
        ),
        CLIENTS_FLOAT_SEQUENCE,
    )
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state,
          test_client_result(
              federated_language.federated_broadcast(weights), data
          ),
          server_zero(),
      )

    with self.assertRaises(errors.TemplatePlacementError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_constructs_with_non_model_weights_parameter(self):
    non_model_weights_type = federated_language.FederatedType(
        federated_language.to_type(
            collections.OrderedDict(trainable=np.float32, non_trainable=())
        ),
        federated_language.CLIENTS,
    )

    @federated_language.federated_computation(
        SERVER_INT, non_model_weights_type, CLIENTS_FLOAT_SEQUENCE
    )
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state, test_client_result(weights, data), server_zero()
      )

    try:
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)
    except client_works.ClientDataTypeError:
      self.fail('Could not construct a valid ClientWorkProcess.')

  def test_constructs_with_struct_of_client_data_parameter(self):

    @federated_language.federated_computation(
        SERVER_INT,
        MODEL_WEIGHTS_TYPE,
        federated_language.FederatedType(
            (
                federated_language.SequenceType(np.float32),
                (
                    federated_language.SequenceType(np.float32),
                    federated_language.SequenceType(np.float32),
                ),
            ),
            federated_language.CLIENTS,
        ),
    )
    def next_fn(state, unused_weights, unused_data):
      return MeasuredProcessOutput(
          state,
          federated_language.federated_value(
              client_works.ClientResult((), ()), federated_language.CLIENTS
          ),
          server_zero(),
      )

    try:
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)
    except client_works.ClientDataTypeError:
      self.fail('Could not construct a valid ClientWorkProcess.')

  def test_non_clients_placed_next_data_param_raises(self):
    server_sequence_float_type = federated_language.FederatedType(
        federated_language.SequenceType(np.float32), federated_language.SERVER
    )

    @federated_language.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, server_sequence_float_type
    )
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state,
          test_client_result(
              weights, federated_language.federated_broadcast(data)
          ),
          server_zero(),
      )

    with self.assertRaises(errors.TemplatePlacementError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_non_sequence_or_struct_next_data_param_raises(self):

    @federated_language.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, CLIENTS_FLOAT
    )
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state,
          federated_language.federated_zip(
              client_works.ClientResult(
                  federated_add(weights.trainable, data), client_one()
              )
          ),
          server_zero(),
      )

    with self.assertRaises(client_works.ClientDataTypeError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_non_clients_placed_next_result_raises(self):

    @federated_language.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, CLIENTS_FLOAT_SEQUENCE
    )
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state,
          federated_language.federated_sum(test_client_result(weights, data)),
          server_zero(),
      )

    with self.assertRaises(errors.TemplatePlacementError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_non_zipped_next_result_raises(self):

    @federated_language.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, CLIENTS_FLOAT_SEQUENCE
    )
    def next_fn(state, weights, data):
      reduced_data = federated_language.federated_map(tf_data_sum, data)
      return MeasuredProcessOutput(
          state,
          client_works.ClientResult(
              federated_add(weights.trainable, reduced_data), client_one()
          ),
          server_zero(),
      )

    with self.assertRaises(errors.TemplatePlacementError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_incorrect_client_result_container_raises(self):

    @federated_language.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, CLIENTS_FLOAT_SEQUENCE
    )
    def next_fn(state, weights, data):
      reduced_data = federated_language.federated_map(tf_data_sum, data)
      bad_client_result = federated_language.federated_zip(
          collections.OrderedDict(
              update=federated_add(weights.trainable, reduced_data),
              update_weight=client_one(),
          )
      )
      return MeasuredProcessOutput(state, bad_client_result, server_zero())

    with self.assertRaises(client_works.ClientResultTypeError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_measurements_raises(self):

    @federated_language.federated_computation(
        SERVER_INT, MODEL_WEIGHTS_TYPE, CLIENTS_FLOAT_SEQUENCE
    )
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state,
          test_client_result(weights, data),
          federated_language.federated_value(1.0, federated_language.CLIENTS),
      )

    with self.assertRaises(errors.TemplatePlacementError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)


if __name__ == '__main__':
  absltest.main()
