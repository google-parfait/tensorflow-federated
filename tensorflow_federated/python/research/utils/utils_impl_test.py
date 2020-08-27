# Copyright 2018, The TensorFlow Federated Authors.
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
import contextlib
import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
import tensorflow as tf

from tensorflow_federated.python.research.utils import utils_impl

FLAGS = flags.FLAGS
TEST_CLIENT_FLAG_PREFIX = 'test_client'
TEST_SERVER_FLAG_PREFIX = 'test_server'


@contextlib.contextmanager
def flag_sandbox(flag_value_dict):

  def _set_flags(flag_dict):
    for name, value in flag_dict.items():
      FLAGS[name].value = value

  # Store the current values and override with the new.
  preserved_value_dict = {
      name: FLAGS[name].value for name in flag_value_dict.keys()
  }
  _set_flags(flag_value_dict)
  yield

  # Restore the saved values.
  for name in preserved_value_dict.keys():
    FLAGS[name].unparse()
  _set_flags(preserved_value_dict)


def setUpModule():
  # Create flags here to ensure duplicate flags are not created.
  utils_impl.define_optimizer_flags(TEST_SERVER_FLAG_PREFIX)
  utils_impl.define_optimizer_flags(TEST_CLIENT_FLAG_PREFIX)


# Create a list of `(test name, optimizer name flag value, optimizer class)`
# for parameterized tests.
_OPTIMIZERS_TO_TEST = [
    (name, name, cls) for name, cls in utils_impl._SUPPORTED_OPTIMIZERS.items()
]


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_optimizer_from_flags_invalid_optimizer(self):
    FLAGS['{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX)].value = 'foo'
    with self.assertRaisesRegex(ValueError, 'not a valid optimizer'):
      _ = utils_impl.create_optimizer_from_flags(TEST_CLIENT_FLAG_PREFIX)

  def test_create_optimizer_from_flags_invalid_overrides(self):
    with flag_sandbox({'{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX): 'sgd'}):
      with self.assertRaisesRegex(TypeError, 'type `collections.Mapping`'):
        _ = utils_impl.create_optimizer_from_flags(
            TEST_CLIENT_FLAG_PREFIX, overrides=[1, 2, 3])

  def test_create_optimizer_from_flags_flags_set_not_for_optimizer(self):
    with flag_sandbox({'{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX): 'sgd'}):
      # Set an Adam flag that isn't used in SGD.
      # We need to use `_parse_args` because that is the only way FLAGS is
      # notified that a non-default value is being used.
      bad_adam_flag = '{}_adam_beta_1'.format(TEST_CLIENT_FLAG_PREFIX)
      FLAGS._parse_args(
          args=['--{}=0.5'.format(bad_adam_flag)], known_only=True)
      with self.assertRaisesRegex(
          ValueError,
          r'Commandline flags for .*\[sgd\].*\'test_client_adam_beta_1\'.*'):
        _ = utils_impl.create_optimizer_from_flags(TEST_CLIENT_FLAG_PREFIX)
      FLAGS[bad_adam_flag].unparse()

  @parameterized.named_parameters(_OPTIMIZERS_TO_TEST)
  def test_create_client_optimizer_from_flags(self, optimizer_name,
                                              optimizer_cls):
    with flag_sandbox(
        {'{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX): optimizer_name}):
      # Construct a default optimizer.
      default_optimizer = utils_impl.create_optimizer_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      self.assertIsInstance(default_optimizer, optimizer_cls)
      # Override the default flag value.
      overridden_learning_rate = 5.0
      custom_optimizer = utils_impl.create_optimizer_from_flags(
          TEST_CLIENT_FLAG_PREFIX,
          overrides={'learning_rate': overridden_learning_rate})
      self.assertIsInstance(custom_optimizer, optimizer_cls)
      self.assertEqual(custom_optimizer.get_config()['learning_rate'],
                       overridden_learning_rate)
      # Override learning rate flag.
      commandline_set_learning_rate = 100.0
      with flag_sandbox({
          '{}_learning_rate'.format(TEST_CLIENT_FLAG_PREFIX):
              commandline_set_learning_rate
      }):
        custom_optimizer = utils_impl.create_optimizer_from_flags(
            TEST_CLIENT_FLAG_PREFIX)
        self.assertIsInstance(custom_optimizer, optimizer_cls)
        self.assertEqual(custom_optimizer.get_config()['learning_rate'],
                         commandline_set_learning_rate)

  @parameterized.named_parameters(_OPTIMIZERS_TO_TEST)
  def test_create_server_optimizer_from_flags(self, optimizer_name,
                                              optimizer_cls):
    with flag_sandbox(
        {'{}_optimizer'.format(TEST_SERVER_FLAG_PREFIX): optimizer_name}):
      FLAGS['{}_optimizer'.format(
          TEST_SERVER_FLAG_PREFIX)].value = optimizer_name
      # Construct a default optimizer.
      default_optimizer = utils_impl.create_optimizer_from_flags(
          TEST_SERVER_FLAG_PREFIX)
      self.assertIsInstance(default_optimizer, optimizer_cls)
      # Override the default flag value.
      overridden_learning_rate = 5.0
      custom_optimizer = utils_impl.create_optimizer_from_flags(
          TEST_SERVER_FLAG_PREFIX,
          overrides={'learning_rate': overridden_learning_rate})
      self.assertIsInstance(custom_optimizer, optimizer_cls)
      self.assertEqual(custom_optimizer.get_config()['learning_rate'],
                       overridden_learning_rate)
      # Set a flag to a non-default.
      commandline_set_learning_rate = 100.0
      with flag_sandbox({
          '{}_learning_rate'.format(TEST_SERVER_FLAG_PREFIX):
              commandline_set_learning_rate
      }):
        custom_optimizer = utils_impl.create_optimizer_from_flags(
            TEST_SERVER_FLAG_PREFIX)
        self.assertIsInstance(custom_optimizer, optimizer_cls)
        self.assertEqual(custom_optimizer.get_config()['learning_rate'],
                         commandline_set_learning_rate)

  def test_atomic_write(self):
    for name in ['foo.csv', 'baz.csv.bz2']:
      dataframe = pd.DataFrame(dict(a=[1, 2], b=[4.0, 5.0]))
      output_file = os.path.join(absltest.get_default_test_tmpdir(), name)
      utils_impl.atomic_write_to_csv(dataframe, output_file)
      dataframe2 = pd.read_csv(output_file, index_col=0)
      pd.testing.assert_frame_equal(dataframe, dataframe2)

      # Overwriting
      dataframe3 = pd.DataFrame(dict(a=[1, 2, 3], b=[4.0, 5.0, 6.0]))
      utils_impl.atomic_write_to_csv(dataframe3, output_file)
      dataframe4 = pd.read_csv(output_file, index_col=0)
      pd.testing.assert_frame_equal(dataframe3, dataframe4)

  def test_atomic_read(self):
    for name in ['foo.csv', 'baz.csv.bz2']:
      dataframe = pd.DataFrame(dict(a=[1, 2], b=[4.0, 5.0]))
      csv_file = os.path.join(absltest.get_default_test_tmpdir(), name)
      utils_impl.atomic_write_to_csv(dataframe, csv_file)

      dataframe2 = utils_impl.atomic_read_from_csv(csv_file)
      pd.testing.assert_frame_equal(dataframe, dataframe2)

  def test_iter_grid(self):
    grid = dict(a=[], b=[])
    self.assertCountEqual(list(utils_impl.iter_grid(grid)), [])

    grid = dict(a=[1])
    self.assertCountEqual(list(utils_impl.iter_grid(grid)), [dict(a=1)])

    grid = dict(a=[1, 2])
    self.assertCountEqual(
        list(utils_impl.iter_grid(grid)), [dict(a=1), dict(a=2)])

    grid = dict(a=[1, 2], b='b', c=[3.0, 4.0])
    self.assertCountEqual(
        list(utils_impl.iter_grid(grid)), [
            dict(a=1, b='b', c=3.0),
            dict(a=1, b='b', c=4.0),
            dict(a=2, b='b', c=3.0),
            dict(a=2, b='b', c=4.0)
        ])

  def test_record_new_flags(self):
    with utils_impl.record_new_flags() as hparam_flags:
      flags.DEFINE_string('exp_name', 'name', 'Unique name for the experiment.')
      flags.DEFINE_float('learning_rate', 0.1, 'Optimizer learning rate.')

    self.assertCountEqual(hparam_flags, ['exp_name', 'learning_rate'])

  def test_convert_flag_names_to_odict(self):
    with utils_impl.record_new_flags() as hparam_flags:
      flags.DEFINE_integer('flag1', 1, 'This is the first flag.')
      flags.DEFINE_float('flag2', 2.0, 'This is the second flag.')

    hparam_odict = utils_impl.lookup_flag_values(hparam_flags)
    expected_odict = collections.OrderedDict(flag1=1, flag2=2.0)

    self.assertEqual(hparam_odict, expected_odict)

  def test_convert_undefined_flag_names(self):
    with self.assertRaisesRegex(ValueError, '"bad_flag" is not a defined flag'):
      utils_impl.lookup_flag_values(['bad_flag'])

  def test_convert_nonstr_flag(self):
    with self.assertRaisesRegex(ValueError, 'All flag names must be strings'):
      utils_impl.lookup_flag_values([300])

  @mock.patch.object(utils_impl, 'multiprocessing')
  def test_launch_experiment(self, mock_multiprocessing):
    pool = mock_multiprocessing.Pool(processes=10)

    grid_dict = [
        collections.OrderedDict([('a_long', 1), ('b', 4.0)]),
        collections.OrderedDict([('a_long', 1), ('b', 5.0)])
    ]

    utils_impl.launch_experiment(
        'bazel run //research/emnist:run_experiment --',
        grid_dict,
        '/tmp_dir',
        short_names={'a_long': 'a'})
    expected = [
        'bazel run //research/emnist:run_experiment -- --a_long=1 --b=4.0 '
        '--root_output_dir=/tmp_dir --exp_name=0-a=1,b=4.0',
        'bazel run //research/emnist:run_experiment -- --a_long=1 --b=5.0 '
        '--root_output_dir=/tmp_dir --exp_name=1-a=1,b=5.0'
    ]
    result = pool.apply_async.call_args_list
    result = [args[0][1][0] for args in result]
    self.assertCountEqual(result, expected)

  def test_remove_unused_flags_without_optimizer_flag(self):
    hparam_dict = collections.OrderedDict([('client_opt_fn', 'sgd'),
                                           ('client_sgd_momentum', 0.3)])
    with self.assertRaisesRegex(ValueError,
                                'The flag client_optimizer was not defined.'):
      _ = utils_impl.remove_unused_flags('client', hparam_dict)

  def test_remove_unused_flags_with_empty_optimizer(self):
    hparam_dict = collections.OrderedDict([('optimizer', '')])

    with self.assertRaisesRegex(
        ValueError, 'The flag optimizer was not set. '
        'Unable to determine the relevant optimizer.'):
      _ = utils_impl.remove_unused_flags(prefix=None, hparam_dict=hparam_dict)

  def test_remove_unused_flags_with_prefix(self):
    hparam_dict = collections.OrderedDict([('client_optimizer', 'sgd'),
                                           ('non_client_value', 0.1),
                                           ('client_sgd_momentum', 0.3),
                                           ('client_adam_momentum', 0.5)])

    relevant_hparam_dict = utils_impl.remove_unused_flags('client', hparam_dict)
    expected_flag_names = [
        'client_optimizer', 'non_client_value', 'client_sgd_momentum'
    ]
    self.assertCountEqual(relevant_hparam_dict.keys(), expected_flag_names)
    self.assertEqual(relevant_hparam_dict['client_optimizer'], 'sgd')
    self.assertEqual(relevant_hparam_dict['non_client_value'], 0.1)
    self.assertEqual(relevant_hparam_dict['client_sgd_momentum'], 0.3)

  def test_remove_unused_flags_without_prefix(self):
    hparam_dict = collections.OrderedDict([('optimizer', 'sgd'), ('value', 0.1),
                                           ('sgd_momentum', 0.3),
                                           ('adam_momentum', 0.5)])
    relevant_hparam_dict = utils_impl.remove_unused_flags(
        prefix=None, hparam_dict=hparam_dict)
    expected_flag_names = ['optimizer', 'value', 'sgd_momentum']
    self.assertCountEqual(relevant_hparam_dict.keys(), expected_flag_names)
    self.assertEqual(relevant_hparam_dict['optimizer'], 'sgd')
    self.assertEqual(relevant_hparam_dict['value'], 0.1)
    self.assertEqual(relevant_hparam_dict['sgd_momentum'], 0.3)


if __name__ == '__main__':
  tf.test.main()
