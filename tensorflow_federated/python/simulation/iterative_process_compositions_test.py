# Copyright 2020, The TensorFlow Federated Authors.
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

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.learning.templates import learning_process
from tensorflow_federated.python.simulation import iterative_process_compositions


def _create_whimsy_iterative_process():
  @tensorflow_computation.tf_computation()
  def init():
    return []

  @tensorflow_computation.tf_computation(init.type_signature.result)
  def next_fn(x):
    return x

  return iterative_process.IterativeProcess(initialize_fn=init, next_fn=next_fn)


def _create_federated_int_dataset_identity_iterative_process():
  @tensorflow_computation.tf_computation()
  def create_dataset():
    return tf.data.Dataset.range(5)

  @federated_computation.federated_computation()
  def init():
    return intrinsics.federated_eval(create_dataset, placements.CLIENTS)

  @federated_computation.federated_computation(init.type_signature.result)
  def next_fn(x):
    return x

  return iterative_process.IterativeProcess(initialize_fn=init, next_fn=next_fn)


def _create_stateless_int_dataset_reduction_iterative_process():
  @tensorflow_computation.tf_computation()
  def make_zero():
    return tf.cast(0, tf.int64)

  @federated_computation.federated_computation()
  def init():
    return intrinsics.federated_eval(make_zero, placements.SERVER)

  @tensorflow_computation.tf_computation(
      computation_types.SequenceType(np.int64)
  )
  def reduce_dataset(x):
    return x.reduce(tf.cast(0, tf.int64), lambda x, y: x + y)

  @federated_computation.federated_computation((
      init.type_signature.result,
      computation_types.FederatedType(
          computation_types.SequenceType(np.int64), placements.CLIENTS
      ),
  ))
  def next_fn(server_state, client_data):
    del server_state  # Unused
    return intrinsics.federated_sum(
        intrinsics.federated_map(reduce_dataset, client_data)
    )

  return iterative_process.IterativeProcess(initialize_fn=init, next_fn=next_fn)


def _create_stateless_int_vector_unknown_dim_dataset_reduction_iterative_process():
  # Tests handling client data of unknown shape and summing to fixed shape.

  @tensorflow_computation.tf_computation()
  def make_zero():
    return tf.reshape(tf.cast(0, tf.int64), shape=[1])

  @federated_computation.federated_computation()
  def init():
    return intrinsics.federated_eval(make_zero, placements.SERVER)

  @tensorflow_computation.tf_computation(
      computation_types.SequenceType(
          computation_types.TensorType(np.int64, shape=[None])
      )
  )
  def reduce_dataset(x):
    return x.reduce(
        tf.cast(tf.constant([0]), tf.int64), lambda x, y: x + tf.reduce_sum(y)
    )

  @federated_computation.federated_computation(
      computation_types.FederatedType(
          computation_types.TensorType(np.int64, shape=[None]),
          placements.SERVER,
      ),
      computation_types.FederatedType(
          computation_types.SequenceType(
              computation_types.TensorType(np.int64, shape=[None])
          ),
          placements.CLIENTS,
      ),
  )
  def next_fn(server_state, client_data):
    del server_state  # Unused
    return intrinsics.federated_sum(
        intrinsics.federated_map(reduce_dataset, client_data)
    )

  return iterative_process.IterativeProcess(initialize_fn=init, next_fn=next_fn)


@tensorflow_computation.tf_computation(np.str_)
def int_dataset_computation(x):
  del x  # Unused
  return tf.data.Dataset.range(5)


@tensorflow_computation.tf_computation(np.str_)
def vector_int_dataset_computation(x):
  del x  # Unused
  return tf.data.Dataset.range(5).map(lambda x: tf.reshape(x, shape=[1]))


@tensorflow_computation.tf_computation(np.str_)
def float_dataset_computation(x):
  del x  # Unused
  return tf.data.Dataset.range(5, output_type=tf.float32)


@tensorflow_computation.tf_computation(np.int32)
def int_identity(x):
  return x


@federated_computation.federated_computation(
    np.int32,
    computation_types.FederatedType(
        computation_types.SequenceType(np.int64), placements.CLIENTS
    ),
    np.float32,
)
def test_int64_sequence_struct_computation(a, dataset, b):
  return a, dataset, b


@federated_computation.federated_computation(
    np.int32,
    computation_types.StructType([
        np.int64,
        computation_types.FederatedType(
            computation_types.SequenceType(np.int64), placements.CLIENTS
        ),
        np.float64,
    ]),
    np.float32,
)
def test_int64_sequence_nested_struct_computation(a, dataset, b):
  return a, dataset, b


@federated_computation.federated_computation(
    computation_types.StructType([
        computation_types.FederatedType(
            computation_types.SequenceType(np.int64), placements.CLIENTS
        ),
    ]),
    computation_types.FederatedType(
        computation_types.SequenceType(np.int64), placements.CLIENTS
    ),
)
def test_int64_sequence_multiple_matching_federated_types_computation(a, b):
  return a, b


@federated_computation.federated_computation(
    computation_types.FederatedType(
        computation_types.SequenceType(np.int64), placements.CLIENTS
    )
)
def test_int64_sequence_computation(dataset):
  del dataset
  return intrinsics.federated_value(5, placements.SERVER)


class ConstructDatasetsOnClientsComputationTest(absltest.TestCase):

  def test_raises_non_computation_dataset_comp(self):
    fn = lambda _: []
    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation_with_computation(
          fn, test_int64_sequence_struct_computation
      )

  def test_raises_non_computation_outer_comp(self):
    non_comp = lambda x: x
    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation_with_computation(
          int_dataset_computation, non_comp
      )

  def test_raises_computation_not_returning_dataset(self):
    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation_with_computation(
          int_identity, test_int64_sequence_struct_computation
      )

  def test_raises_computation_no_dataset_parameter(self):
    no_dataset_comp = federated_computation.federated_computation(
        lambda x: x, [np.int32]
    )
    with self.assertRaises(
        iterative_process_compositions.SequenceTypeNotFoundError
    ):
      iterative_process_compositions.compose_dataset_computation_with_computation(
          int_dataset_computation, no_dataset_comp
      )

  def test_raises_mismatched_dataset_comp_return_type_and_sequence_type(self):
    with self.assertRaises(
        iterative_process_compositions.SequenceTypeNotAssignableError
    ):
      iterative_process_compositions.compose_dataset_computation_with_computation(
          float_dataset_computation, test_int64_sequence_struct_computation
      )

  def test_mutates_comp_accepting_only_dataset(self):
    expected_new_next_type_signature = computation_types.FunctionType(
        parameter=computation_types.FederatedType(np.str_, placements.CLIENTS),
        result=computation_types.FederatedType(np.int32, placements.SERVER),
    )
    new_comp = iterative_process_compositions.compose_dataset_computation_with_computation(
        int_dataset_computation, test_int64_sequence_computation
    )
    expected_new_next_type_signature.check_equivalent_to(
        new_comp.type_signature
    )

  def test_mutates_comp_accepting_dataset_in_second_index(self):
    expected_new_next_type_signature = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            a=np.int32,
            dataset=computation_types.FederatedType(
                np.str_, placements.CLIENTS
            ),
            b=np.float32,
        ),
        result=(
            np.int32,
            computation_types.FederatedType(
                computation_types.SequenceType(np.int64), placements.CLIENTS
            ),
            np.float32,
        ),
    )
    new_comp = iterative_process_compositions.compose_dataset_computation_with_computation(
        int_dataset_computation, test_int64_sequence_struct_computation
    )
    expected_new_next_type_signature.check_equivalent_to(
        new_comp.type_signature
    )

  def test_raises_computation_with_multiple_federated_types(self):
    with self.assertRaises(
        iterative_process_compositions.MultipleMatchingSequenceTypesError
    ):
      iterative_process_compositions.compose_dataset_computation_with_computation(
          int_dataset_computation,
          test_int64_sequence_multiple_matching_federated_types_computation,
      )

  def test_mutates_comp_accepting_deeply_nested_dataset(self):
    expected_new_next_type_signature = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            a=np.int32,
            dataset=computation_types.StructType([
                np.int64,
                computation_types.FederatedType(np.str_, placements.CLIENTS),
                np.float64,
            ]),
            b=np.float32,
        ),
        result=test_int64_sequence_nested_struct_computation.type_signature.result,
    )
    new_comp = iterative_process_compositions.compose_dataset_computation_with_computation(
        int_dataset_computation, test_int64_sequence_nested_struct_computation
    )
    expected_new_next_type_signature.check_equivalent_to(
        new_comp.type_signature
    )


class ConstructDatasetsOnClientsIterativeProcessTest(absltest.TestCase):

  def test_raises_non_computation(self):
    fn = lambda _: []
    iterproc = _create_federated_int_dataset_identity_iterative_process()

    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation_with_iterative_process(
          fn, iterproc
      )

  def test_raises_non_iterative_process(self):
    non_iterproc = lambda x: x

    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation_with_iterative_process(
          int_dataset_computation, non_iterproc
      )

  def test_raises_computation_not_returning_dataset(self):
    iterproc = _create_federated_int_dataset_identity_iterative_process()

    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation_with_iterative_process(
          int_identity, iterproc
      )

  def test_raises_iterative_process_no_dataset_parameter(self):
    iterproc = _create_whimsy_iterative_process()

    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation_with_iterative_process(
          int_dataset_computation, iterproc
      )

  def test_raises_mismatched_dataset_comp_return_type_and_iterproc_sequence_type(
      self,
  ):
    iterproc = _create_federated_int_dataset_identity_iterative_process()

    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation_with_iterative_process(
          float_dataset_computation, iterproc
      )

  def test_raises_iterproc_if_dataset_is_returned_by_init(self):
    iterproc = _create_federated_int_dataset_identity_iterative_process()

    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation_with_iterative_process(
          int_dataset_computation, iterproc
      )

  def test_mutates_iterproc_accepting_dataset_in_second_index_of_next(self):
    iterproc = _create_stateless_int_dataset_reduction_iterative_process()
    expected_new_next_type_signature = computation_types.FunctionType(
        collections.OrderedDict(
            server_state=computation_types.FederatedType(
                np.int64, placements.SERVER
            ),
            client_data=computation_types.FederatedType(
                np.str_, placements.CLIENTS
            ),
        ),
        computation_types.FederatedType(np.int64, placements.SERVER),
    )

    new_iterproc = iterative_process_compositions.compose_dataset_computation_with_iterative_process(
        int_dataset_computation, iterproc
    )

    expected_new_next_type_signature.check_equivalent_to(
        new_iterproc.next.type_signature
    )

  def test_mutates_iterproc_with_parameter_assignable_from_result(self):
    iterproc = (
        _create_stateless_int_vector_unknown_dim_dataset_reduction_iterative_process()
    )
    expected_new_next_type_signature = computation_types.FunctionType(
        collections.OrderedDict(
            server_state=computation_types.FederatedType(
                computation_types.TensorType(np.int64, shape=[None]),
                placements.SERVER,
            ),
            client_data=computation_types.FederatedType(
                np.str_, placements.CLIENTS
            ),
        ),
        computation_types.FederatedType(
            computation_types.TensorType(np.int64, shape=[1]), placements.SERVER
        ),
    )

    new_iterproc = iterative_process_compositions.compose_dataset_computation_with_iterative_process(
        vector_int_dataset_computation, iterproc
    )

    expected_new_next_type_signature.check_equivalent_to(
        new_iterproc.next.type_signature
    )

  def test_returns_iterproc_accepting_dataset_in_third_index_of_next(self):
    iterproc = _create_stateless_int_dataset_reduction_iterative_process()

    old_param_type = iterproc.next.type_signature.parameter

    new_param_elements = [old_param_type[0], np.int32, old_param_type[1]]

    @federated_computation.federated_computation(
        computation_types.StructType(new_param_elements)
    )
    def new_next(param):
      return iterproc.next([param[0], param[2]])

    iterproc_with_dataset_as_third_elem = iterative_process.IterativeProcess(
        iterproc.initialize, new_next
    )
    expected_new_next_type_signature = computation_types.FunctionType(
        [
            computation_types.FederatedType(np.int64, placements.SERVER),
            np.int32,
            computation_types.FederatedType(np.str_, placements.CLIENTS),
        ],
        computation_types.FederatedType(np.int64, placements.SERVER),
    )

    new_iterproc = iterative_process_compositions.compose_dataset_computation_with_iterative_process(
        int_dataset_computation, iterproc_with_dataset_as_third_elem
    )

    self.assertTrue(
        expected_new_next_type_signature.is_equivalent_to(
            new_iterproc.next.type_signature
        )
    )


class ConstructDatasetsOnClientsLearningProcessTest(absltest.TestCase):

  def test_returns_iterative_process_with_same_non_next_type_signatures(self):
    @tensorflow_computation.tf_computation()
    def make_zero():
      return tf.cast(0, tf.int64)

    @federated_computation.federated_computation()
    def initialize_fn():
      return intrinsics.federated_eval(make_zero, placements.SERVER)

    @federated_computation.federated_computation((
        initialize_fn.type_signature.result,
        computation_types.FederatedType(
            computation_types.SequenceType(np.int64), placements.CLIENTS
        ),
    ))
    def next_fn(server_state, client_data):
      del client_data
      return learning_process.LearningProcessOutput(
          state=server_state,
          metrics=intrinsics.federated_value((), placements.SERVER),
      )

    @tensorflow_computation.tf_computation(np.int64)
    def get_model_weights(server_state):
      return server_state + 1

    @tensorflow_computation.tf_computation(np.int64, np.int64)
    def set_model_weights(state, state_update):
      return state + state_update

    process = learning_process.LearningProcess(
        initialize_fn=initialize_fn,
        next_fn=next_fn,
        get_model_weights=get_model_weights,
        set_model_weights=set_model_weights,
    )
    new_process = iterative_process_compositions.compose_dataset_computation_with_learning_process(
        int_dataset_computation, process
    )

    self.assertIsInstance(new_process, iterative_process.IterativeProcess)
    self.assertTrue(hasattr(new_process, 'get_model_weights'))
    self.assertTrue(
        new_process.get_model_weights.type_signature.is_equivalent_to(
            process.get_model_weights.type_signature
        )
    )
    self.assertTrue(hasattr(new_process, 'set_model_weights'))
    self.assertTrue(
        new_process.set_model_weights.type_signature.is_equivalent_to(
            process.set_model_weights.type_signature
        )
    )


if __name__ == '__main__':
  absltest.main()
