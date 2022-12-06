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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.simulation import sampling_utils


def create_tf_dataset_for_client(client_id):
  del client_id
  return tf.data.Dataset.range(1)


class SamplingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': '_int_no_replace',
          'a': range(100),
          'replace': False
      }, {
          'testcase_name': '_int_replace',
          'a': range(5),
          'replace': True
      }, {
          'testcase_name': '_sequence_no_replace',
          'a': [str(i) for i in range(100)],
          'replace': False
      }, {
          'testcase_name': '_sequence_replace',
          'a': [str(i) for i in range(5)],
          'replace': True
      })
  def test_build_uniform_sampling_fn_with_random_seed(self, a, replace):
    random_seed = 1
    round_num = 5

    sample_fn_1 = sampling_utils.build_uniform_sampling_fn(
        a, replace=replace, random_seed=random_seed)
    sample_1 = sample_fn_1(round_num, size=10)

    sample_fn_2 = sampling_utils.build_uniform_sampling_fn(
        a, replace=replace, random_seed=random_seed)
    sample_2 = sample_fn_2(round_num, size=10)

    self.assertEqual(sample_1, sample_2)

  @parameterized.named_parameters(
      {
          'testcase_name': '_int_no_replace',
          'a': range(100),
          'replace': False
      }, {
          'testcase_name': '_int_replace',
          'a': range(5),
          'replace': True
      }, {
          'testcase_name': '_sequence_no_replace',
          'a': [str(i) for i in range(100)],
          'replace': False
      }, {
          'testcase_name': '_sequence_replace',
          'a': [str(i) for i in range(5)],
          'replace': True
      })
  def test_build_sampling_fn_without_random_seed(self, a, replace):
    round_num = 5

    sample_fn_1 = sampling_utils.build_uniform_sampling_fn(a, replace=replace)
    sample_1 = sample_fn_1(round_num, size=10)

    sample_fn_2 = sampling_utils.build_uniform_sampling_fn(a, replace=replace)
    sample_2 = sample_fn_2(round_num, size=10)

    self.assertNotEqual(sample_1, sample_2)


if __name__ == '__main__':
  tf.test.main()
