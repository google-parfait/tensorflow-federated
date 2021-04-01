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

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import estimation_process


@computations.tf_computation()
def test_initialize_fn():
  return tf.constant(0, tf.int32)


@computations.tf_computation(tf.int32)
def test_next_fn(state):
  return state


@computations.tf_computation(tf.int32)
def test_report_fn(state):
  return tf.cast(state, tf.float32)


@computations.tf_computation(tf.float32)
def test_map_fn(estimate):
  return tf.stack([estimate, estimate])


class EstimationProcessTest(test_case.TestCase):

  def test_construction_does_not_raise(self):
    try:
      estimation_process.EstimationProcess(test_initialize_fn, test_next_fn,
                                           test_report_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct a valid EstimationProcess.')

  def test_construction_with_empty_state_does_not_raise(self):
    initialize_fn = computations.tf_computation()(lambda: ())
    next_fn = computations.tf_computation(())(lambda x: (x, 1.0))
    report_fn = computations.tf_computation(())(lambda x: x)
    try:
      estimation_process.EstimationProcess(initialize_fn, next_fn, report_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an EstimationProcess with empty state.')

  def test_construction_with_unknown_dimension_does_not_raise(self):
    initialize_fn = computations.tf_computation()(
        lambda: tf.constant([], dtype=tf.string))

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string))
    def next_fn(strings):
      return tf.concat([strings, tf.constant(['abc'])], axis=0)

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string))
    def report_fn(strings):
      return strings

    try:
      estimation_process.EstimationProcess(initialize_fn, next_fn, report_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an EstimationProcess with parameter types '
                'with statically unknown shape.')

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      estimation_process.EstimationProcess(
          initialize_fn=lambda: 0,
          next_fn=test_next_fn,
          report_fn=test_report_fn)

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      estimation_process.EstimationProcess(
          initialize_fn=test_initialize_fn,
          next_fn=lambda state: state,
          report_fn=test_report_fn)

  def test_report_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      estimation_process.EstimationProcess(
          initialize_fn=test_initialize_fn,
          next_fn=test_next_fn,
          report_fn=lambda state: state)

  def test_init_param_not_empty_raises(self):
    one_arg_initialize_fn = computations.tf_computation(tf.int32)(lambda x: x)
    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      estimation_process.EstimationProcess(one_arg_initialize_fn, test_next_fn,
                                           test_report_fn)

  def test_init_state_not_assignable(self):
    float_initialize_fn = computations.tf_computation()(lambda: 0.0)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.EstimationProcess(float_initialize_fn, test_next_fn,
                                           test_report_fn)

  def test_federated_init_state_not_assignable(self):
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(
        computation_types.FederatedType(
            tf.int32, placements.CLIENTS))(lambda state: state)
    report_fn = computations.federated_computation(
        initialize_fn.type_signature.result)(lambda state: state)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.EstimationProcess(initialize_fn, next_fn, report_fn)

  def test_next_state_not_assignable(self):
    float_next_fn = computations.tf_computation(
        tf.float32)(lambda state: tf.cast(state, tf.float32))
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.EstimationProcess(test_initialize_fn, float_next_fn,
                                           test_report_fn)

  def test_federated_next_state_not_assignable(self):
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(
        initialize_fn.type_signature.result)(
            intrinsics.federated_broadcast)
    report_fn = computations.federated_computation(
        initialize_fn.type_signature.result)(lambda state: state)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.EstimationProcess(initialize_fn, next_fn, report_fn)

  def test_next_state_not_assignable_tuple_result(self):
    float_next_fn = computations.tf_computation(
        tf.float32,
        tf.float32)(lambda state, x: (tf.cast(state, tf.float32), x))
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.EstimationProcess(test_initialize_fn, float_next_fn,
                                           test_report_fn)

  # Tests specific only for the EstimationProcess contract below.

  def test_report_state_not_assignable(self):
    report_fn = computations.tf_computation(
        tf.float32)(lambda estimate: estimate)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.EstimationProcess(test_initialize_fn, test_next_fn,
                                           report_fn)

  def test_federated_report_state_not_assignable(self):
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(
        initialize_fn.type_signature.result)(lambda state: state)
    report_fn = computations.federated_computation(
        computation_types.FederatedType(
            tf.int32, placements.CLIENTS))(lambda state: state)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.EstimationProcess(initialize_fn, next_fn, report_fn)

  def test_mapped_process_as_expected(self):
    process = estimation_process.EstimationProcess(test_initialize_fn,
                                                   test_next_fn, test_report_fn)
    mapped_process = process.map(test_map_fn)

    self.assertIsInstance(mapped_process, estimation_process.EstimationProcess)
    self.assertEqual(process.initialize, mapped_process.initialize)
    self.assertEqual(process.next, mapped_process.next)
    self.assertEqual(process.report.type_signature.parameter,
                     mapped_process.report.type_signature.parameter)
    self.assertEqual(test_map_fn.type_signature.result,
                     mapped_process.report.type_signature.result)

  def test_federated_mapped_process_as_expected(self):
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(
        initialize_fn.type_signature.result)(lambda state: state)
    report_fn = computations.federated_computation(
        initialize_fn.type_signature.result)(
            lambda state: intrinsics.federated_map(test_report_fn, state))
    process = estimation_process.EstimationProcess(initialize_fn, next_fn,
                                                   report_fn)

    map_fn = computations.federated_computation(
        report_fn.type_signature.result)(
            lambda estimate: intrinsics.federated_map(test_map_fn, estimate))
    mapped_process = process.map(map_fn)

    self.assertIsInstance(mapped_process, estimation_process.EstimationProcess)
    self.assertEqual(process.initialize, mapped_process.initialize)
    self.assertEqual(process.next, mapped_process.next)
    self.assertEqual(process.report.type_signature.parameter,
                     mapped_process.report.type_signature.parameter)
    self.assertEqual(map_fn.type_signature.result,
                     mapped_process.report.type_signature.result)

  def test_map_estimate_not_assignable(self):
    map_fn = computations.tf_computation(tf.int32)(lambda estimate: estimate)
    process = estimation_process.EstimationProcess(test_initialize_fn,
                                                   test_next_fn, test_report_fn)
    with self.assertRaises(estimation_process.EstimateNotAssignableError):
      process.map(map_fn)

  def test_federated_map_estimate_not_assignable(self):
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(
        initialize_fn.type_signature.result)(lambda state: state)
    report_fn = computations.federated_computation(
        initialize_fn.type_signature.result)(
            lambda state: intrinsics.federated_map(test_report_fn, state))
    process = estimation_process.EstimationProcess(initialize_fn, next_fn,
                                                   report_fn)

    map_fn = computations.federated_computation(
        computation_types.FederatedType(
            tf.int32, placements.CLIENTS))(lambda estimate: estimate)
    with self.assertRaises(estimation_process.EstimateNotAssignableError):
      process.map(map_fn)


if __name__ == '__main__':
  test_case.main()
