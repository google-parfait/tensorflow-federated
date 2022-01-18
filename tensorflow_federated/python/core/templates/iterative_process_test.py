# Copyright 2019, The TensorFlow Federated Authors.
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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import iterative_process


# Convenience aliases.
FederatedType = computation_types.FederatedType
StructType = computation_types.StructType
StructWithPythonType = computation_types.StructWithPythonType
TensorType = computation_types.TensorType


@computations.tf_computation()
def test_initialize_fn():
  return tf.constant(0, tf.int32)


@computations.tf_computation(tf.int32)
def test_next_fn(state):
  return state


class IterativeProcessTest(test_case.TestCase):

  def test_construction_does_not_raise(self):
    try:
      iterative_process.IterativeProcess(test_initialize_fn, test_next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct a valid IterativeProcess.')

  def test_construction_with_empty_state_does_not_raise(self):
    initialize_fn = computations.tf_computation()(lambda: ())
    next_fn = computations.tf_computation(())(lambda x: (x, 1.0))
    try:
      iterative_process.IterativeProcess(initialize_fn, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an IterativeProcess with empty state.')

  def test_construction_with_unknown_dimension_does_not_raise(self):
    initialize_fn = computations.tf_computation()(
        lambda: tf.constant([], dtype=tf.string))

    @computations.tf_computation(TensorType(shape=[None], dtype=tf.string))
    def next_fn(strings):
      return tf.concat([strings, tf.constant(['abc'])], axis=0)

    try:
      iterative_process.IterativeProcess(initialize_fn, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an IterativeProcess with parameter types '
                'with statically unknown shape.')

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      iterative_process.IterativeProcess(
          initialize_fn=lambda: 0, next_fn=test_next_fn)

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      iterative_process.IterativeProcess(
          initialize_fn=test_initialize_fn, next_fn=lambda state: state)

  def test_init_param_not_empty_raises(self):
    one_arg_initialize_fn = computations.tf_computation(tf.int32)(lambda x: x)
    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      iterative_process.IterativeProcess(one_arg_initialize_fn, test_next_fn)

  def test_init_state_not_assignable(self):
    float_initialize_fn = computations.tf_computation()(lambda: 0.0)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      iterative_process.IterativeProcess(float_initialize_fn, test_next_fn)

  def test_federated_init_state_not_assignable(self):
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(
        FederatedType(tf.int32, placements.CLIENTS))(lambda state: state)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      iterative_process.IterativeProcess(initialize_fn, next_fn)

  def test_next_state_not_assignable(self):
    float_next_fn = computations.tf_computation(
        tf.float32)(lambda state: tf.cast(state, tf.float32))
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      iterative_process.IterativeProcess(test_initialize_fn, float_next_fn)

  def test_federated_next_state_not_assignable(self):
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(
        initialize_fn.type_signature.result)(
            intrinsics.federated_broadcast)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      iterative_process.IterativeProcess(initialize_fn, next_fn)

  def test_next_state_not_assignable_tuple_result(self):
    float_next_fn = computations.tf_computation(
        tf.float32,
        tf.float32)(lambda state, x: (tf.cast(state, tf.float32), x))
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      iterative_process.IterativeProcess(test_initialize_fn, float_next_fn)


def create_test_process(
    type_spec: computation_types.Type) -> iterative_process.IterativeProcess:

  @computations.tf_computation
  def create_value():
    return type_conversions.structure_from_tensor_type_tree(
        lambda t: tf.zeros(dtype=t.dtype, shape=t.shape),
        type_spec.member if type_spec.is_federated() else type_spec)

  @computations.federated_computation
  def init_fn():
    if type_spec.is_federated():
      return intrinsics.federated_eval(create_value, type_spec.placement)
    else:
      return create_value()

  @computations.federated_computation(init_fn.type_signature.result, tf.int32)
  def next_fn(state, arg):
    return state, arg

  return iterative_process.IterativeProcess(init_fn, next_fn)


class HasEmptyStateTest(parameterized.TestCase, test_case.TestCase):

  @parameterized.named_parameters(
      ('simple', StructType([])),
      ('simple_with_python_container', StructWithPythonType([], list)),
      ('nested', StructType([StructType([]),
                             StructType([StructType([])])])),
      ('federated_simple', computation_types.at_server(StructType([]))),
      ('federated_nested',
       computation_types.at_server(
           StructType([StructType([]),
                       StructType([StructType([])])]))),
  )
  def test_stateless_process_is_false(self, state_type):
    process = create_test_process(state_type)
    self.assertFalse(iterative_process.is_stateful(process))

  @parameterized.named_parameters(
      ('tensor', TensorType(tf.int32)),
      ('struct_with_tensor', StructType([TensorType(tf.int32)])),
      ('struct_with_python_tensor',
       StructWithPythonType([TensorType(tf.int32)], list)),
      ('nested_state',
       StructType(
           [StructType([]),
            StructType([StructType([TensorType(tf.int32)])])])),
      ('federated_simple_state',
       computation_types.at_server(StructType([TensorType(tf.int32)]))),
      ('federated_nested_state',
       computation_types.at_server(
           StructType([
               StructType([TensorType(tf.int32)]),
               StructType([StructType([])])
           ]))),
  )
  def test_stateful_process_is_true(self, state_type):
    process = create_test_process(state_type)
    self.assertTrue(iterative_process.is_stateful(process))

if __name__ == '__main__':
  test_case.main()
