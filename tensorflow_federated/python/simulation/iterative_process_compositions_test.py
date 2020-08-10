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
"""Tests for iterative_process_compositions."""

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.simulation import iterative_process_compositions


def _create_dummy_iterative_process():

  @computations.tf_computation()
  def init():
    return []

  @computations.tf_computation(init.type_signature.result)
  def next_fn(x):
    return x

  return iterative_process.IterativeProcess(initialize_fn=init, next_fn=next_fn)


def _create_federated_int_dataset_identity_iterative_process():

  @computations.tf_computation()
  def create_dataset():
    return tf.data.Dataset.range(5)

  @computations.federated_computation()
  def init():
    return intrinsics.federated_eval(create_dataset, placement_literals.CLIENTS)

  @computations.federated_computation(init.type_signature.result)
  def next_fn(x):
    return x

  return iterative_process.IterativeProcess(initialize_fn=init, next_fn=next_fn)


def _create_stateless_int_dataset_reduction_iterative_process():

  @computations.tf_computation()
  def make_zero():
    return tf.cast(0, tf.int64)

  @computations.federated_computation()
  def init():
    return intrinsics.federated_eval(make_zero, placement_literals.SERVER)

  @computations.tf_computation(computation_types.SequenceType(tf.int64))
  def reduce_dataset(x):
    return x.reduce(tf.cast(0, tf.int64), lambda x, y: x + y)

  @computations.federated_computation(
      (init.type_signature.result,
       computation_types.FederatedType(
           computation_types.SequenceType(tf.int64),
           placement_literals.CLIENTS)))
  def next_fn(empty_tup, x):
    del empty_tup  # Unused
    return intrinsics.federated_sum(intrinsics.federated_map(reduce_dataset, x))

  return iterative_process.IterativeProcess(initialize_fn=init, next_fn=next_fn)


@computations.tf_computation(tf.string)
def int_dataset_computation(x):
  del x  # Unused
  return tf.data.Dataset.range(5)


@computations.tf_computation(tf.string)
def float_dataset_computation(x):
  del x  # Unused
  return tf.data.Dataset.range(5, output_type=tf.float32)


@computations.tf_computation(tf.int32)
def int_identity(x):
  return x


class ConstructDatasetsOnClientsTest(absltest.TestCase):

  def test_raises_non_computation(self):

    fn = lambda _: []
    iterproc = _create_federated_int_dataset_identity_iterative_process()

    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation(fn, iterproc)

  def test_raises_non_iterative_process(self):
    non_iterproc = lambda x: x

    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation(
          int_dataset_computation, non_iterproc)

  def test_raises_computation_not_returning_dataset(self):
    iterproc = _create_federated_int_dataset_identity_iterative_process()

    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation(
          int_identity, iterproc)

  def test_raises_iterative_process_no_dataset_parameter(self):
    iterproc = _create_dummy_iterative_process()

    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation(
          int_dataset_computation, iterproc)

  def test_raises_mismatched_dataset_comp_return_type_and_iterproc_sequence_type(
      self):
    iterproc = _create_federated_int_dataset_identity_iterative_process()

    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation(
          float_dataset_computation, iterproc)

  def test_raises_iterproc_if_dataset_is_returned_by_init(self):
    iterproc = _create_federated_int_dataset_identity_iterative_process()

    with self.assertRaises(TypeError):
      iterative_process_compositions.compose_dataset_computation(
          int_dataset_computation, iterproc)

  def test_mutates_iterproc_accepting_dataset_in_second_index_of_next(self):
    iterproc = _create_stateless_int_dataset_reduction_iterative_process()
    expected_new_next_type_signature = computation_types.FunctionType([
        computation_types.FederatedType(tf.int64, placement_literals.SERVER),
        computation_types.FederatedType(tf.string, placement_literals.CLIENTS)
    ], computation_types.FederatedType(tf.int64, placement_literals.SERVER))

    new_iterproc = iterative_process_compositions.compose_dataset_computation(
        int_dataset_computation, iterproc)

    self.assertTrue(
        expected_new_next_type_signature.is_equivalent_to(
            new_iterproc.next.type_signature))

  def test_returns_iterproc_accepting_dataset_in_third_index_of_next(self):
    iterproc = _create_stateless_int_dataset_reduction_iterative_process()

    old_param_type = iterproc.next.type_signature.parameter

    new_param_elements = [old_param_type[0], tf.int32, old_param_type[1]]

    @computations.federated_computation(
        computation_types.StructType(new_param_elements))
    def new_next(param):
      return iterproc.next([param[0], param[2]])

    iterproc_with_dataset_as_third_elem = iterative_process.IterativeProcess(
        iterproc.initialize, new_next)
    expected_new_next_type_signature = computation_types.FunctionType([
        computation_types.FederatedType(tf.int64, placement_literals.SERVER),
        tf.int32,
        computation_types.FederatedType(tf.string, placement_literals.CLIENTS)
    ], computation_types.FederatedType(tf.int64, placement_literals.SERVER))

    new_iterproc = iterative_process_compositions.compose_dataset_computation(
        int_dataset_computation, iterproc_with_dataset_as_third_elem)

    self.assertTrue(
        expected_new_next_type_signature.is_equivalent_to(
            new_iterproc.next.type_signature))


if __name__ == '__main__':
  absltest.main()
