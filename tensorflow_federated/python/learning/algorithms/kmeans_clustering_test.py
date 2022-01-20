# Copyright 2022, The TensorFlow Federated Authors.
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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning.algorithms import kmeans_clustering

_WEIGHT_DTYPE = kmeans_clustering._WEIGHT_DTYPE


class ClientWorkTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('shape1', (1,)),
      ('shape2', (2,)),
      ('shape3', (2, 2)),
      ('shape4', (5, 7, 1, 6)),
  )
  def test_find_closest_centroid__with_different_shapes(self, shape):
    centroid1 = tf.fill(shape, -1)
    centroid2 = tf.fill(shape, 1)
    centroids = tf.convert_to_tensor([centroid1, centroid2])
    point = tf.fill(shape, 2)
    closest_centroid = kmeans_clustering._find_closest_centroid(
        centroids, point)
    self.assertEqual(closest_centroid, 1)

  @parameterized.named_parameters(
      ('int32', tf.int32),
      ('int64', tf.int64),
      ('float32', tf.float32),
      ('float64', tf.float64),
      ('bfloat16', tf.bfloat16),
  )
  def test_find_closest_centroid_with_different_dtypes(self, dtype):
    shape = (3, 2)
    value1 = tf.constant(-1, dtype=dtype)
    value2 = tf.constant(1, dtype=dtype)
    point = tf.constant(2, dtype=dtype)
    centroid1 = tf.fill(shape, value1)
    centroid2 = tf.fill(shape, value2)
    centroids = tf.convert_to_tensor([centroid1, centroid2])
    closest_centroid = kmeans_clustering._find_closest_centroid(
        centroids, point)
    self.assertEqual(closest_centroid, 1)

  @parameterized.named_parameters(
      ('shape1', (1,)),
      ('shape2', (2,)),
      ('shape3', (2, 2)),
      ('shape4', (5, 7, 1, 6)),
  )
  def test_kmeans_step_with_different_shapes(self, shape):
    centroid1 = tf.fill(shape, -1)
    centroid2 = tf.fill(shape, 1)
    centroids = tf.convert_to_tensor([centroid1, centroid2])
    cluster_zero_points = [tf.fill(shape, -2) for _ in range(2)]
    cluster_one_points = [tf.fill(shape, 2) for _ in range(3)]
    data = tf.data.Dataset.from_tensor_slices(cluster_zero_points +
                                              cluster_one_points)

    actual_result, actual_metrics = kmeans_clustering._compute_kmeans_step(
        centroids, data)
    expected_result_update = (tf.convert_to_tensor(
        [tf.fill(shape, -4), tf.fill(shape, 6)]), tf.constant([2, 3]))

    self.assertLen(actual_result.update, 2)
    self.assertAllEqual(actual_result.update[0], expected_result_update[0])
    self.assertAllEqual(actual_result.update[1], expected_result_update[1])
    self.assertEmpty(actual_result.update_weight)
    self.assertDictEqual(actual_metrics, {'num_examples': 5})

  @parameterized.named_parameters(
      ('int32', tf.int32),
      ('int64', tf.int64),
      ('float32', tf.float32),
      ('float64', tf.float64),
      ('bfloat16', tf.bfloat16),
  )
  def test_kmeans_step_with_different_dtypes(self, dtype):
    shape = (3, 2)
    centroid1 = tf.fill(shape, tf.constant(-1, dtype=dtype))
    centroid2 = tf.fill(shape, tf.constant(1, dtype=dtype))
    centroids = tf.convert_to_tensor([centroid1, centroid2])
    cluster_zero_points = [
        tf.fill(shape, tf.constant(-2, dtype=dtype)) for _ in range(2)
    ]
    cluster_one_points = [
        tf.fill(shape, tf.constant(2, dtype=dtype)) for _ in range(3)
    ]
    data = tf.data.Dataset.from_tensor_slices(cluster_zero_points +
                                              cluster_one_points)

    actual_result, actual_metrics = kmeans_clustering._compute_kmeans_step(
        centroids, data)
    expected_result_update = (tf.convert_to_tensor([
        tf.fill(shape, tf.constant(-4, dtype=dtype)),
        tf.fill(shape, tf.constant(6, dtype=dtype))
    ]), tf.constant([2, 3]))

    self.assertLen(actual_result.update, 2)
    self.assertEqual(actual_result.update[0].dtype, dtype)
    self.assertAllEqual(actual_result.update[0], expected_result_update[0])
    self.assertAllEqual(actual_result.update[1], expected_result_update[1])
    self.assertEmpty(actual_result.update_weight)
    self.assertDictEqual(actual_metrics, {'num_examples': 5})

  @parameterized.named_parameters(
      ('shape1', (1,)),
      ('shape2', (2,)),
      ('shape3', (2, 2)),
      ('shape4', (5, 7, 1, 6)),
  )
  def test_build_kmeans_client_work_with_different_shapes(self, shape):
    point_dtype = tf.float32
    num_clusters = 5
    centroids_shape = (num_clusters,) + shape
    centroids_type = computation_types.TensorType(point_dtype, centroids_shape)
    point_type = computation_types.TensorType(point_dtype, shape)
    data_type = computation_types.SequenceType(point_type)
    weight_type = computation_types.TensorType(_WEIGHT_DTYPE, (num_clusters,))
    empty_server_type = computation_types.at_server(())

    client_work = kmeans_clustering._build_kmeans_client_work(
        centroids_type, data_type)

    next_type = client_work.next.type_signature
    next_type.parameter[0].check_equivalent_to(empty_server_type)
    next_type.parameter[1].check_equivalent_to(
        computation_types.at_clients(centroids_type))
    next_type.parameter[2].check_equivalent_to(
        computation_types.at_clients(data_type))
    next_type.result[0].check_equivalent_to(empty_server_type)
    next_type.result[1].member.update.check_equivalent_to(
        computation_types.to_type((centroids_type, weight_type)))

    expected_measurements_type = computation_types.to_type(
        collections.OrderedDict(
            num_examples=computation_types.TensorType(_WEIGHT_DTYPE)))
    next_type.result[2].member.check_equivalent_to(expected_measurements_type)

  @parameterized.named_parameters(
      ('int32', tf.int32),
      ('int64', tf.int64),
      ('float32', tf.float32),
      ('float64', tf.float64),
      ('bfloat16', tf.bfloat16),
  )
  def test_build_kmeans_client_work_with_different_dtypes(self, point_dtype):
    shape = (3, 2)
    num_clusters = 5
    centroids_shape = (num_clusters,) + shape
    centroids_type = computation_types.TensorType(point_dtype, centroids_shape)
    point_type = computation_types.TensorType(point_dtype, shape)
    data_type = computation_types.SequenceType(point_type)
    weight_type = computation_types.TensorType(_WEIGHT_DTYPE, (num_clusters,))
    empty_server_type = computation_types.at_server(())

    client_work = kmeans_clustering._build_kmeans_client_work(
        centroids_type, data_type)

    next_type = client_work.next.type_signature
    next_type.parameter[0].check_equivalent_to(empty_server_type)
    next_type.parameter[1].check_equivalent_to(
        computation_types.at_clients(centroids_type))
    next_type.parameter[2].check_equivalent_to(
        computation_types.at_clients(data_type))
    next_type.result[0].check_equivalent_to(empty_server_type)
    next_type.result[1].member.update.check_equivalent_to(
        computation_types.to_type((centroids_type, weight_type)))

    expected_measurements_type = computation_types.to_type(
        collections.OrderedDict(
            num_examples=computation_types.TensorType(_WEIGHT_DTYPE)))
    next_type.result[2].member.check_equivalent_to(expected_measurements_type)


class FinalizerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('shape1', (1,)),
      ('shape2', (2,)),
      ('shape3', (2, 2)),
      ('shape4', (5, 7, 1, 6)),
  )
  def test_update_centroids_computes_average_with_weights_one(self, shape):
    num_clusters = 5
    centroids_shape = (num_clusters,) + shape
    current_centroids = tf.fill(centroids_shape, -3.0)
    new_cluster_sums = tf.fill(centroids_shape, 1.0)
    weights = tf.fill((num_clusters,), 1)
    updated_centroids, total_weights = kmeans_clustering._update_centroids(
        current_centroids, weights, new_cluster_sums, weights)

    expected_centroids = 0.5 * (current_centroids + new_cluster_sums)
    expected_weights = tf.fill((num_clusters,), 2)

    self.assertAllEqual(updated_centroids, expected_centroids)
    self.assertAllEqual(total_weights, expected_weights)

  @parameterized.named_parameters(
      ('shape1', (1,)),
      ('shape2', (2,)),
      ('shape3', (2, 2)),
      ('shape4', (5, 7, 1, 6)),
  )
  def test_update_centroids_is_no_op_on_new_weights_zero(self, shape):
    num_clusters = 5
    centroids_shape = (num_clusters,) + shape
    current_centroids = tf.fill(centroids_shape, -3.0)
    new_cluster_sums = tf.fill(centroids_shape, 1.0)
    current_weights = tf.fill((num_clusters,), 1)
    new_weights = tf.fill((num_clusters,), 0)
    updated_centroids, total_weights = kmeans_clustering._update_centroids(
        current_centroids, current_weights, new_cluster_sums, new_weights)

    self.assertAllEqual(total_weights, current_weights)
    self.assertAllEqual(updated_centroids, current_centroids)

  @parameterized.named_parameters(
      ('shape1', (1,)),
      ('shape2', (2,)),
      ('shape3', (2, 2)),
      ('shape4', (5, 7, 1, 6)),
  )
  def test_update_centroids_with_current_weight_zero(self, shape):
    num_clusters = 5
    centroids_shape = (num_clusters,) + shape
    current_centroids = tf.fill(centroids_shape, -3.0)
    new_cluster_sums = tf.fill(centroids_shape, 16.0)
    current_weights = tf.fill((num_clusters,), 0)
    new_weights = tf.fill((num_clusters,), 8)
    updated_centroids, total_weights = kmeans_clustering._update_centroids(
        current_centroids, current_weights, new_cluster_sums, new_weights)

    self.assertAllEqual(total_weights, new_weights)
    self.assertAllEqual(updated_centroids, tf.fill(centroids_shape, 2.0))

  def test_current_weights_applied_coordinate_wise(self):
    centroids_shape = (3, 2)
    current_centroids = tf.fill(centroids_shape, 1.0)
    new_cluster_sums = tf.fill(centroids_shape, 0.0)
    current_weights = tf.constant([1, 2, 3])
    new_weights = tf.constant([1, 1, 1])
    updated_centroids, total_weights = kmeans_clustering._update_centroids(
        current_centroids, current_weights, new_cluster_sums, new_weights)
    expected_updated_centroids = tf.constant([
        [1.0 / (1.0 + 1.0), 1.0 / (1.0 + 1.0)],
        [2.0 / (1.0 + 2.0), 2.0 / (1.0 + 2.0)],
        [3.0 / (1.0 + 3.0), 3.0 / (1.0 + 3.0)],
    ])

    self.assertAllEqual(total_weights, new_weights + current_weights)
    self.assertAllEqual(updated_centroids, expected_updated_centroids)

  def test_new_weights_applied_coordinate_wise(self):
    centroids_shape = (3, 2)
    current_centroids = tf.fill(centroids_shape, 0.0)
    new_cluster_sums = tf.fill(centroids_shape, 1.0)
    current_weights = tf.constant([0, 0, 0])
    new_weights = tf.constant([1, 2, 3])
    updated_centroids, total_weights = kmeans_clustering._update_centroids(
        current_centroids, current_weights, new_cluster_sums, new_weights)
    expected_updated_centroids = tf.constant([
        [1.0 / 1.0, 1.0 / 1.0],
        [1.0 / 2.0, 1.0 / 2.0],
        [1.0 / 3.0, 1.0 / 3.0],
    ])

    self.assertAllEqual(total_weights, new_weights + current_weights)
    self.assertAllEqual(updated_centroids, expected_updated_centroids)


class FederatedKmeansTest(test_case.TestCase):

  def test_constructs_with_pseudocounts_of_one(self):
    kmeans_process = kmeans_clustering.build_fed_kmeans(
        num_clusters=3, data_shape=(2, 2))
    state = kmeans_process.initialize()
    self.assertAllEqual(state.finalizer, tf.ones(3,))

  def test_initialize_uses_random_seed(self):
    data_shape = (3, 4, 5)
    kmeans_1 = kmeans_clustering.build_fed_kmeans(
        num_clusters=6, data_shape=data_shape, random_seed=(42, 2))
    kmeans_2 = kmeans_clustering.build_fed_kmeans(
        num_clusters=6, data_shape=data_shape, random_seed=(42, 2))
    kmeans_3 = kmeans_clustering.build_fed_kmeans(
        num_clusters=6, data_shape=data_shape, random_seed=(43, 2))
    kmeans_4 = kmeans_clustering.build_fed_kmeans(
        num_clusters=6, data_shape=data_shape, random_seed=(42, 3))
    init_value1 = kmeans_1.initialize().global_model_weights
    init_value2 = kmeans_2.initialize().global_model_weights
    init_value3 = kmeans_3.initialize().global_model_weights
    init_value4 = kmeans_4.initialize().global_model_weights

    self.assertAllClose(init_value1, init_value2)
    self.assertNotAllClose(init_value1, init_value3)
    self.assertNotAllClose(init_value1, init_value4)

  def test_single_step_with_one_client(self):
    data_shape = (3, 2)
    kmeans = kmeans_clustering.build_fed_kmeans(
        num_clusters=1, data_shape=data_shape, random_seed=(0, 0))
    point1 = tf.fill(data_shape, value=1.0)
    point2 = tf.fill(data_shape, value=2.0)
    dataset = tf.data.Dataset.from_tensor_slices([point1, point2])

    state = kmeans.initialize()
    initial_centroids = state.global_model_weights
    output = kmeans.next(state, [dataset])
    actual_centroids = output.state.global_model_weights
    weights = output.state.finalizer
    expected_centroids = (1 / 3) * (
        initial_centroids + tf.expand_dims(point1 + point2, axis=0))

    self.assertAllClose(actual_centroids, expected_centroids)
    self.assertAllEqual(weights, [3])

  def test_single_step_with_two_clients(self):
    data_shape = (3, 2)
    kmeans = kmeans_clustering.build_fed_kmeans(
        num_clusters=1, data_shape=data_shape, random_seed=(0, 0))
    point1 = tf.fill(data_shape, value=1.0)
    dataset1 = tf.data.Dataset.from_tensors(point1)
    point2 = tf.fill(data_shape, value=2.0)
    dataset2 = tf.data.Dataset.from_tensors(point2)

    state = kmeans.initialize()
    initial_centroids = state.global_model_weights
    output = kmeans.next(state, [dataset1, dataset2])
    actual_centroids = output.state.global_model_weights
    weights = output.state.finalizer
    expected_centroids = (1 / 3) * (
        initial_centroids + tf.expand_dims(point1 + point2, axis=0))

    self.assertAllClose(actual_centroids, expected_centroids)
    self.assertAllEqual(weights, [3])

  def test_two_steps_with_one_cluster(self):
    data_shape = (3, 2)
    kmeans = kmeans_clustering.build_fed_kmeans(
        num_clusters=1, data_shape=data_shape, random_seed=(0, 0))
    point1 = tf.fill(data_shape, value=1.0)
    dataset1 = tf.data.Dataset.from_tensors(point1)
    point2 = tf.fill(data_shape, value=2.0)
    dataset2 = tf.data.Dataset.from_tensors(point2)
    state = kmeans.initialize()
    initial_centroids = state.global_model_weights

    output = kmeans.next(state, [dataset1])
    centroids = output.state.global_model_weights
    weights = output.state.finalizer
    expected_step_1_centroids = 0.5 * (
        initial_centroids + tf.expand_dims(point1, axis=0))
    self.assertAllClose(centroids, expected_step_1_centroids)
    self.assertAllEqual(weights, [2])

    output = kmeans.next(output.state, [dataset2])
    centroids = output.state.global_model_weights
    weights = output.state.finalizer
    expected_step_2_centroids = (1 / 3) * (
        initial_centroids + tf.expand_dims(point1 + point2, axis=0))
    self.assertAllClose(centroids, expected_step_2_centroids)
    self.assertAllEqual(weights, [3])


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
