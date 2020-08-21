# Copyright 2019, The TensorFlow Federated Authors.
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

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.backends.mapreduce import canonical_form
from tensorflow_federated.python.core.backends.mapreduce import test_utils


def _dummy_canonical_form_computations():

  @computations.tf_computation
  def initialize():
    return tf.constant(0)

  @computations.tf_computation(tf.int32)
  def prepare(server_state):
    del server_state  # Unused
    return tf.constant(1.0)

  @computations.tf_computation(
      computation_types.SequenceType(tf.float32), tf.float32)
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return True, []

  @computations.tf_computation
  def zero():
    return tf.constant(0), tf.constant(0)

  @computations.tf_computation((tf.int32, tf.int32), tf.bool)
  def accumulate(accumulator, client_update):
    del accumulator  # Unused
    del client_update  # Unused
    return tf.constant(1), tf.constant(1)

  @computations.tf_computation((tf.int32, tf.int32), (tf.int32, tf.int32))
  def merge(accumulator1, accumulator2):
    del accumulator1  # Unused
    del accumulator2  # Unused
    return tf.constant(1), tf.constant(1)

  @computations.tf_computation(tf.int32, tf.int32)
  def report(accumulator):
    del accumulator  # Unused
    return tf.constant(1.0)

  @computations.tf_computation
  def bitwidth():
    return []

  @computations.tf_computation(tf.int32,
                               (tf.float32, computation_types.StructType([])))
  def update(server_state, global_update):
    del server_state  # Unused
    del global_update  # Unused
    return tf.constant(1), []

  return (initialize, prepare, work, zero, accumulate, merge, report, bitwidth,
          update)


class CanonicalFormTest(absltest.TestCase):

  def test_init_does_not_raise_type_error(self):
    (initialize, prepare, work, zero, accumulate, merge, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    try:
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_init_does_not_raise_type_error_with_unknown_dimensions(self):

    @computations.tf_computation
    def initialize():
      return tf.constant(0)

    @computations.tf_computation(tf.int32)
    def prepare(server_state):
      del server_state  # Unused
      return tf.constant(1.0)

    @computations.tf_computation(
        computation_types.SequenceType(tf.float32), tf.float32)
    def work(client_data, client_input):
      del client_data  # Unused
      del client_input  # Unused
      return True, []

    @computations.tf_computation
    def zero():
      return tf.constant([], dtype=tf.string)

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string), tf.bool)
    def accumulate(accumulator, client_update):
      del accumulator  # Unused
      del client_update  # Unused
      return tf.constant(['abc'])

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string),
        computation_types.TensorType(shape=[None], dtype=tf.string))
    def merge(accumulator1, accumulator2):
      del accumulator1  # Unused
      del accumulator2  # Unused
      return tf.constant(['abc'])

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string))
    def report(accumulator):
      del accumulator  # Unused
      return tf.constant(1.0)

    @computations.tf_computation
    def bitwidth():
      return []

    @computations.tf_computation(tf.int32,
                                 (tf.float32, computation_types.StructType([])))
    def update(server_state, global_update):
      del server_state  # Unused
      del global_update  # Unused
      return tf.constant(1), []

    try:
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_init_raises_type_error_with_bad_initialize_result_type(self):
    (_, prepare, work, zero, accumulate, merge, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation
    def initialize():
      return tf.constant(0.0)

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_prepare_parameter_type(self):
    (initialize, _, work, zero, accumulate, merge, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation(tf.float32)
    def prepare(server_state):
      del server_state  # Unused
      return tf.constant(1.0)

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_prepare_result_type(self):
    (initialize, _, work, zero, accumulate, merge, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation(tf.int32)
    def prepare(server_state):
      del server_state  # Unused
      return tf.constant(1)

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_work_second_parameter_type(self):
    (initialize, prepare, _, zero, accumulate, merge, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation(
        computation_types.SequenceType(tf.float32), tf.int32)
    def work(client_data, client_input):
      del client_data  # Unused
      del client_input  # Unused
      return True, []

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_work_result_type(self):
    (initialize, prepare, _, zero, accumulate, merge, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation(
        computation_types.SequenceType(tf.float32), tf.float32)
    def work(client_data, client_input):
      del client_data  # Unused
      del client_input  # Unused
      return tf.constant('abc'), []

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_zero_result_type(self):
    (initialize, prepare, work, _, accumulate, merge, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation
    def zero():
      return tf.constant(0.0), tf.constant(0)

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_accumulate_first_parameter_type(
      self):
    (initialize, prepare, work, zero, _, merge, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation((tf.float32, tf.int32), tf.bool)
    def accumulate(accumulator, client_update):
      del accumulator  # Unused
      del client_update  # Unused
      return tf.constant(1), tf.constant(1)

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_accumulate_second_parameter_type(
      self):
    (initialize, prepare, work, zero, _, merge, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation((tf.float32, tf.float32), tf.string)
    def accumulate(accumulator, client_update):
      del accumulator  # Unused
      del client_update  # Unused
      return tf.constant(1), tf.constant(1)

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_accumulate_result_type(self):
    (initialize, prepare, work, zero, _, merge, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation((tf.float32, tf.float32), tf.bool)
    def accumulate(accumulator, client_update):
      del accumulator  # Unused
      del client_update  # Unused
      return tf.constant(1.0), tf.constant(1)

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_merge_first_parameter_type(self):
    (initialize, prepare, work, zero, accumulate, _, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation((tf.float32, tf.int32), (tf.int32, tf.int32))
    def merge(accumulator1, accumulator2):
      del accumulator1  # Unused
      del accumulator2  # Unused
      return tf.constant(1), tf.constant(1)

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_merge_second_parameter_type(self):
    (initialize, prepare, work, zero, accumulate, _, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation((tf.int32, tf.int32), (tf.float32, tf.int32))
    def merge(accumulator1, accumulator2):
      del accumulator1  # Unused
      del accumulator2  # Unused
      return tf.constant(1), tf.constant(1)

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_merge_result_type(self):
    (initialize, prepare, work, zero, accumulate, _, report, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation((tf.int32, tf.int32), (tf.int32, tf.int32))
    def merge(accumulator1, accumulator2):
      del accumulator1  # Unused
      del accumulator2  # Unused
      return tf.constant(1.0), tf.constant(1)

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_report_parameter_type(self):
    (initialize, prepare, work, zero, accumulate, merge, _, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation(tf.float32, tf.int32)
    def report(accumulator):
      del accumulator  # Unused
      return tf.constant(1.0)

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_report_result_type(self):
    (initialize, prepare, work, zero, accumulate, merge, _, bitwidth,
     update) = _dummy_canonical_form_computations()

    @computations.tf_computation(tf.int32, tf.int32)
    def report(accumulator):
      del accumulator  # Unused
      return tf.constant(1)

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_update_first_parameter_type(self):
    (initialize, prepare, work, zero, accumulate, merge, report, bitwidth,
     _) = _dummy_canonical_form_computations()

    @computations.tf_computation(tf.float32,
                                 (tf.float32, computation_types.StructType([])))
    def update(server_state, global_update):
      del server_state  # Unused
      del global_update  # Unused
      return tf.constant(1), []

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_update_second_parameter_type(self):
    (initialize, prepare, work, zero, accumulate, merge, report, bitwidth,
     _) = _dummy_canonical_form_computations()

    @computations.tf_computation(tf.int32,
                                 (tf.int32, computation_types.StructType([])))
    def update(server_state, global_update):
      del server_state  # Unused
      del global_update  # Unused
      return tf.constant(1), []

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_init_raises_type_error_with_bad_update_result_type(self):
    (initialize, prepare, work, zero, accumulate, merge, report, bitwidth,
     _) = _dummy_canonical_form_computations()

    @computations.tf_computation(tf.int32,
                                 (tf.float32, computation_types.StructType([])))
    def update(server_state, global_update):
      del server_state  # Unused
      del global_update  # Unused
      return tf.constant(1.0), []

    with self.assertRaises(TypeError):
      canonical_form.CanonicalForm(initialize, prepare, work, zero, accumulate,
                                   merge, report, bitwidth, update)

  def test_summary(self):
    cf = test_utils.get_temperature_sensor_example()

    class CapturePrint(object):

      def __init__(self):
        self.summary = ''

      def __call__(self, msg):
        self.summary += msg + '\n'

    capture = CapturePrint()
    cf.summary(print_fn=capture)
    # pyformat: disable
    self.assertEqual(
        capture.summary,
        'initialize: ( -> <num_rounds=int32>)\n'
        'prepare   : (<num_rounds=int32> -> <max_temperature=float32>)\n'
        'work      : (<data=float32*,state=<max_temperature=float32>> -> <<is_over=bool>,<>>)\n'
        'zero      : ( -> <num_total=int32,num_over=int32>)\n'
        'accumulate: (<accumulator=<num_total=int32,num_over=int32>,update=<is_over=bool>> -> <num_total=int32,num_over=int32>)\n'
        'merge     : (<accumulator1=<num_total=int32,num_over=int32>,accumulator2=<num_total=int32,num_over=int32>> -> <num_total=int32,num_over=int32>)\n'
        'report    : (<num_total=int32,num_over=int32> -> <ratio_over_threshold=float32>)\n'
        'bitwidth  : ( -> <>)\n'
        'update    : ( -> <num_rounds=int32>)\n'
    )
    # pyformat: enable


if __name__ == '__main__':
  absltest.main()
