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

import collections

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import dataset_utils
from tensorflow_federated.python.simulation.datasets import from_tensor_slices_client_data


class DatasetUtilsTest(tf.test.TestCase):

  def test_deterministic_dataset_mixture(self):
    a = tf.data.Dataset.range(5)
    b = tf.data.Dataset.range(5).map(lambda x: x + 5)
    mixture1 = dataset_utils.build_dataset_mixture(
        a, b, a_probability=0.5, seed=0)
    mixture2 = dataset_utils.build_dataset_mixture(
        a, b, a_probability=0.5, seed=0)
    examples1 = [self.evaluate(x) for x in mixture1]
    examples2 = [self.evaluate(x) for x in mixture2]
    self.assertAllEqual(examples1, examples2)

  def test_deterministic_dataset_mixture_distribution(self):
    # Create a dataset of infinite eights.
    a = tf.data.Dataset.from_tensor_slices([8]).repeat(None)
    # Create a normal sampling of integers around mean=5
    b = tf.data.Dataset.from_tensor_slices(
        tf.cast(tf.random.normal(shape=[1000], mean=5, stddev=2.0), tf.int32))
    # Create a mixture of 1000 integers (bounded by the size of `b` since `a` is
    # infinite).
    mixture1 = dataset_utils.build_dataset_mixture(
        a, b, a_probability=0.8, seed=0)
    mixture2 = dataset_utils.build_dataset_mixture(
        a, b, a_probability=0.8, seed=0)
    mixture3 = dataset_utils.build_dataset_mixture(
        a, b, a_probability=0.8, seed=1)
    counts1 = collections.Counter(self.evaluate(x) for x in mixture1)
    counts2 = collections.Counter(self.evaluate(x) for x in mixture2)
    counts3 = collections.Counter(self.evaluate(x) for x in mixture3)
    self.assertEqual(counts1, counts2)
    self.assertNotEqual(counts1, counts3)

  def test_non_deterministic_dataset_mixture_different(self):
    num_examples = 100
    a = tf.data.Dataset.from_tensor_slices([0] * num_examples)
    b = tf.data.Dataset.from_tensor_slices([1] * num_examples)
    mixture_1 = dataset_utils.build_dataset_mixture(a, b, a_probability=0.5)
    mixture_2 = dataset_utils.build_dataset_mixture(a, b, a_probability=0.5)
    # The mixtures should produce different samples.
    self.assertNotEqual(
        self.evaluate(list(iter(mixture_1))),
        self.evaluate(list(iter(mixture_2))))

  def test_filter_single_label_dataset(self):
    # Create a uniform sampling of integers in [0, 10).
    d = tf.data.Dataset.from_tensor_slices({
        'label':
            tf.random.uniform(shape=[1000], minval=0, maxval=9, dtype=tf.int32),
    })

    filtered_d = dataset_utils.build_single_label_dataset(
        d, label_key='label', desired_label=6)
    filtered_examples = [self.evaluate(x) for x in filtered_d]
    # Expect close to 1000 / 10  = 100 examples.
    self.assertLen(filtered_examples, 103)
    self.assertTrue(all(x['label'] == 6 for x in filtered_d))

  def test_build_synthethic_iid_client_data(self):
    # Create a fake, very non-IID ClientData.
    client_datasets = collections.OrderedDict(a=[1] * 3, b=[2] * 5, c=[3] * 7)
    non_iid_client_data = from_tensor_slices_client_data.TestClientData(
        client_datasets)

    iid_client_data_iter = iter(
        dataset_utils.build_synthethic_iid_datasets(
            non_iid_client_data, client_dataset_size=5))

    num_synthethic_clients = 3
    run_results = []
    for _ in range(5):
      actual_iid_client_datasets = []
      for _ in range(num_synthethic_clients):
        dataset = next(iid_client_data_iter)
        actual_iid_client_datasets.append([self.evaluate(x) for x in dataset])
      # We expect 3 datasets: 15 examples in the global dataset, synthetic
      # non-iid configured for 5 examples per client.
      self.assertEqual([5, 5, 5], [len(d) for d in actual_iid_client_datasets])
      run_results.append(actual_iid_client_datasets)

    # Assert no run is the same. The chance that two runs are the same is far
    # less than 1 in a million, flakes should be imperceptible.
    for i, run_a in enumerate(run_results[:-1]):
      for run_b in run_results[i + 1:]:
        self.assertNotEqual(run_a, run_b, msg=str(run_results))


if __name__ == '__main__':
  tf.test.main()
