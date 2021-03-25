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
from tensorflow_federated.python.simulation.datasets import client_data


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
    size = 10
    random_seed = 1
    round_num = 5

    sample_fn_1 = sampling_utils.build_uniform_sampling_fn(
        a, size, replace=replace, random_seed=random_seed)
    sample_1 = sample_fn_1(round_num)

    sample_fn_2 = sampling_utils.build_uniform_sampling_fn(
        a, size, replace=replace, random_seed=random_seed)
    sample_2 = sample_fn_2(round_num)

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
    size = 10
    round_num = 5

    sample_fn_1 = sampling_utils.build_uniform_sampling_fn(
        a, size, replace=replace)
    sample_1 = sample_fn_1(round_num)

    sample_fn_2 = sampling_utils.build_uniform_sampling_fn(
        a, size, replace=replace)
    sample_2 = sample_fn_2(round_num)

    self.assertNotEqual(sample_1, sample_2)

  def test_client_sampling_with_one_client(self):
    tff_dataset = client_data.ConcreteClientData([2],
                                                 create_tf_dataset_for_client)
    client_sampling_fn = sampling_utils.build_uniform_client_sampling_fn(
        tff_dataset, clients_per_round=1)
    client_ids = client_sampling_fn(round_num=7)
    self.assertEqual(client_ids, [2])

  def test_client_sampling_fn_with_random_seed(self):
    tff_dataset = client_data.ConcreteClientData([0, 1, 2, 3, 4],
                                                 create_tf_dataset_for_client)

    client_sampling_fn_1 = sampling_utils.build_uniform_client_sampling_fn(
        tff_dataset, clients_per_round=1, random_seed=363)
    client_ids_1 = client_sampling_fn_1(round_num=5)

    client_sampling_fn_2 = sampling_utils.build_uniform_client_sampling_fn(
        tff_dataset, clients_per_round=1, random_seed=363)
    client_ids_2 = client_sampling_fn_2(round_num=5)

    self.assertEqual(client_ids_1, client_ids_2)

  def test_different_random_seed_give_different_clients(self):
    tff_dataset = client_data.ConcreteClientData(
        list(range(100)), create_tf_dataset_for_client)

    client_sampling_fn_1 = sampling_utils.build_uniform_client_sampling_fn(
        tff_dataset, clients_per_round=50, random_seed=1)
    client_ids_1 = client_sampling_fn_1(round_num=1001)

    client_sampling_fn_2 = sampling_utils.build_uniform_client_sampling_fn(
        tff_dataset, clients_per_round=50, random_seed=2)
    client_ids_2 = client_sampling_fn_2(round_num=1001)

    self.assertNotEqual(client_ids_1, client_ids_2)

  def test_client_sampling_fn_without_random_seed(self):
    tff_dataset = client_data.ConcreteClientData(
        list(range(100)), create_tf_dataset_for_client)
    client_sampling_fn = sampling_utils.build_uniform_client_sampling_fn(
        tff_dataset, clients_per_round=50)
    client_ids_1 = client_sampling_fn(round_num=0)

    client_ids_2 = client_sampling_fn(round_num=0)
    self.assertNotEqual(client_ids_1, client_ids_2)


if __name__ == '__main__':
  tf.test.main()
