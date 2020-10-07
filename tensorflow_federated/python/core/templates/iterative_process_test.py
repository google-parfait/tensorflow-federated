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

import tensorflow as tf
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import iterative_process


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

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string))
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
        computation_types.FederatedType(
            tf.int32, placements.CLIENTS))(lambda state: state)
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


if __name__ == '__main__':
  test_case.main()
