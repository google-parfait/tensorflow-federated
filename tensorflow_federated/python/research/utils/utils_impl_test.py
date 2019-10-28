# Lint as: python3
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

  def test_get_optimizer_from_flags_invalid_optimizer(self):
    FLAGS['{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX)].value = 'foo'
    with self.assertRaisesRegex(ValueError, 'not a valid optimizer'):
      _ = utils_impl.get_optimizer_from_flags(TEST_CLIENT_FLAG_PREFIX)

  def test_get_optimizer_from_flags_flags_set_not_for_optimizer(self):
    FLAGS['{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX)].value = 'sgd'
    # Set an Adam flag that isn't used in SGD.
    # We need to use `_parse_args` because that is the only way FLAGS is
    # notified that a non-default value is being used.
    FLAGS._parse_args(
        args=['--{}_adam_beta_1=0.5'.format(TEST_CLIENT_FLAG_PREFIX)],
        known_only=True)
    with self.assertRaisesRegex(
        ValueError,
        r'Commandline flags for .*\[sgd\].*\'test_client_adam_beta_1\'.*'):
      _ = utils_impl.get_optimizer_from_flags(TEST_CLIENT_FLAG_PREFIX)

  @parameterized.named_parameters(_OPTIMIZERS_TO_TEST)
  def test_get_client_optimizer_from_flags(self, optimizer_name, optimizer_cls):
    FLAGS['{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX)].value = optimizer_name
    # Construct a default optimizer.
    default_optimizer = utils_impl.get_optimizer_from_flags(
        TEST_CLIENT_FLAG_PREFIX)
    self.assertIsInstance(default_optimizer, optimizer_cls)
    # Override learning rate flag.
    FLAGS['{}_{}'.format(TEST_CLIENT_FLAG_PREFIX,
                         'learning_rate')].value = 100.0
    custom_optimizer = utils_impl.get_optimizer_from_flags(
        TEST_CLIENT_FLAG_PREFIX)
    self.assertIsInstance(custom_optimizer, optimizer_cls)
    self.assertEqual(custom_optimizer.get_config()['learning_rate'], 100.0)
    # Override the flag value.
    custom_optimizer = utils_impl.get_optimizer_from_flags(
        TEST_CLIENT_FLAG_PREFIX, {'learning_rate': 5.0})
    self.assertIsInstance(custom_optimizer, optimizer_cls)
    self.assertEqual(custom_optimizer.get_config()['learning_rate'], 5.0)

  @parameterized.named_parameters(_OPTIMIZERS_TO_TEST)
  def test_get_server_optimizer_from_flags(self, optimizer_name, optimizer_cls):
    FLAGS['{}_optimizer'.format(TEST_SERVER_FLAG_PREFIX)].value = optimizer_name
    # Construct a default optimizer.
    default_optimizer = utils_impl.get_optimizer_from_flags(
        TEST_SERVER_FLAG_PREFIX)
    self.assertIsInstance(default_optimizer, optimizer_cls)
    # Set a flag to a non-default.
    FLAGS['{}_{}'.format(TEST_SERVER_FLAG_PREFIX,
                         'learning_rate')].value = 100.0
    custom_optimizer = utils_impl.get_optimizer_from_flags(
        TEST_SERVER_FLAG_PREFIX)
    self.assertIsInstance(custom_optimizer, optimizer_cls)
    self.assertEqual(custom_optimizer.get_config()['learning_rate'], 100.0)
    # Override the flag value.
    custom_optimizer = utils_impl.get_optimizer_from_flags(
        TEST_SERVER_FLAG_PREFIX, {'learning_rate': 5.0})
    self.assertIsInstance(custom_optimizer, optimizer_cls)
    self.assertEqual(custom_optimizer.get_config()['learning_rate'], 5.0)

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


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
