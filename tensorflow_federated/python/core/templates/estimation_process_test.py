# Copyright 2020, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import iterative_process


@computations.tf_computation()
def test_initialize_fn():
  return tf.constant(0, tf.int32)


@computations.tf_computation(tf.int32)
def test_next_fn(state):
  return state + 1


@computations.tf_computation(tf.int32)
def test_get_estimate_fn(state):
  return tf.cast(state, tf.float32) / 2.0


test_estimation_process = estimation_process.EstimationProcess(
    initialize_fn=test_initialize_fn,
    next_fn=test_next_fn,
    get_estimate_fn=test_get_estimate_fn)


@computations.tf_computation(tf.float32)
def test_transform_fn(arg):
  return 3.0 * arg + 1.0


class EstimationProcessTest(test_case.TestCase):

  def test_get_estimate_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      estimation_process.EstimationProcess(
          initialize_fn=test_initialize_fn,
          next_fn=test_next_fn,
          get_estimate_fn=lambda state: 1)

  def test_init_state_not_assignable(self):
    float_initialize_fn = computations.tf_computation()(lambda: 0.0)
    float_next_fn = computations.tf_computation(tf.float32)(lambda x: x)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.EstimationProcess(float_initialize_fn, float_next_fn,
                                           test_next_fn)

  def test_apply_does_not_raise(self):
    try:
      estimation_process.apply(test_transform_fn, test_estimation_process)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct a valid TransformEstimationProcess.')

  def test_apply_arg_process_not_estimation_process_raises(self):
    non_estimation_process = iterative_process.IterativeProcess(
        test_initialize_fn, test_next_fn)
    with self.assertRaisesRegex(TypeError,
                                r'Expected .*\.EstimationProcess, .*'):
      estimation_process.apply(test_transform_fn, non_estimation_process)

  def test_apply_not_tff_computation_raises(self):
    non_tff_computation = lambda arg: 1.0
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      estimation_process.apply(non_tff_computation, test_estimation_process)

  def test_apply_arg_not_assignable(self):
    int_transform_fn = computations.tf_computation(tf.int32)(lambda x: 0.0)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.apply(int_transform_fn, test_estimation_process)

  def test_apply_execution(self):
    process = estimation_process.apply(test_transform_fn,
                                       test_estimation_process)

    state = process.initialize()
    self.assertAllClose(process.get_estimate(state), 1.0)
    state = process.next(state)
    self.assertAllClose(process.get_estimate(state), 2.5)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
