# Lint as: python3
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

import tensorflow as tf

from tensorflow_federated.python.research.optimization.shared import keras_metrics

tf.compat.v1.enable_v2_behavior()


class NumTokensCounterTest(tf.test.TestCase):

  def test_constructor_no_masked_token(self):
    metric_name = 'my_test_metric'
    metric = keras_metrics.NumTokensCounter(name=metric_name)
    self.assertIsInstance(metric, tf.keras.metrics.Metric)
    self.assertEqual(metric.name, metric_name)
    self.assertEqual(self.evaluate(metric.result()), 0)

  def test_counts_total_examples_without_zero_mask_no_sample_weight(self):
    metric = keras_metrics.NumTokensCounter()
    metric.update_state(
        y_true=[[1, 2, 3, 4], [0, 0, 0, 0]],
        y_pred=[
            0
            # y_pred is thrown away
        ])
    self.assertEqual(self.evaluate(metric.result()), 8)

  def test_counts_total_examples_with_zero_mask_no_sample_weight(self):
    metric = keras_metrics.NumTokensCounter(masked_tokens=[0])
    metric.update_state(y_true=[[1, 2, 3, 4], [0, 0, 0, 0]], y_pred=[0])
    self.assertEqual(self.evaluate(metric.result()), 4)

  def test_counts_total_examples_without_zero_mask_with_sample_weight(self):
    metric = keras_metrics.NumTokensCounter()
    metric.update_state(
        y_true=[[1, 2, 3, 4], [0, 0, 0, 0]],
        y_pred=[0],
        sample_weight=[[1, 2, 3, 4], [1, 1, 1, 1]])
    self.assertEqual(self.evaluate(metric.result()), 14)

  def test_counts_total_examples_with_zero_mask_with_sample_weight(self):
    metric = keras_metrics.NumTokensCounter(masked_tokens=[0])
    metric.update_state(
        y_true=[[1, 2, 3, 0], [1, 0, 0, 0]],
        y_pred=[0],
        sample_weight=[[1, 2, 3, 4], [1, 1, 1, 1]])
    self.assertEqual(self.evaluate(metric.result()), 7)


class MaskedCategoricalAccuracyTest(tf.test.TestCase):

  def test_constructor_no_masked_token(self):
    metric_name = 'my_test_metric'
    metric = keras_metrics.MaskedCategoricalAccuracy(name=metric_name)
    self.assertIsInstance(metric, tf.keras.metrics.Metric)
    self.assertEqual(metric.name, metric_name)
    self.assertAllEqual(metric.get_config()['masked_tokens'], [])
    self.assertEqual(self.evaluate(metric.result()), 0.0)

  def test_constructor_with_masked_token(self):
    metric_name = 'my_test_metric'
    metric = keras_metrics.MaskedCategoricalAccuracy(
        name=metric_name, masked_tokens=[100])
    self.assertIsInstance(metric, tf.keras.metrics.Metric)
    self.assertEqual(metric.name, metric_name)
    self.assertAllEqual(metric.get_config()['masked_tokens'], [100])
    self.assertEqual(self.evaluate(metric.result()), 0.0)

  def test_update_state_with_special_character(self):
    metric = keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[4])
    metric.update_state(
        y_true=[[1, 2, 3, 4], [0, 0, 0, 0]],
        y_pred=[
            # A batch with 100% accruacy.
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
            # A batch with 50% accruacy.
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.0],
            ],
        ])
    self.assertAllClose(self.evaluate(metric.result()), 5 / 7.0)
    metric.update_state(
        y_true=[[0, 4, 1, 2]],
        y_pred=[
            # A batch with 33% accruacy.
            [
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
        ])
    self.assertAllClose(self.evaluate(metric.result()), 6 / 10.0)

  def test_update_state_with_no_special_character(self):
    metric = keras_metrics.MaskedCategoricalAccuracy()
    metric.update_state(
        y_true=[[1, 2, 3, 4], [0, 0, 0, 0]],
        y_pred=[
            # A batch with 100% accruacy.
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
            # A batch with 50% accruacy.
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.0],
            ],
        ])
    self.assertEqual(self.evaluate(metric.result()), 6 / 8.0)
    metric.update_state(
        y_true=[[0, 4, 1, 2]],
        y_pred=[
            # A batch with 25% accruacy.
            [
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
        ])
    self.assertAllClose(self.evaluate(metric.result()), 8 / 12.0)

  def test_weighted_update_state_with_masked_token(self):
    metric = keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[4])
    metric.update_state(
        y_true=[[1, 2, 3, 4], [0, 0, 0, 0]],
        y_pred=[
            # A batch with 100% accuracy.
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
            # A batch with 50% accuracy.
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.0],
            ],
        ],
        # A weight for each `y_true` scalar.
        sample_weight=[[1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0]])
    self.assertAllClose(self.evaluate(metric.result()), (4 + 4) / 10.0)
    metric.update_state(
        y_true=[[0, 4, 1, 2]],
        y_pred=[
            # A batch with 25% accruacy.
            [
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
        ],
        sample_weight=[1.0, 1.0, 2.0, 2.0])
    self.assertAllClose(self.evaluate(metric.result()), (4 + 4 + 1) / 15.0)

  def test_weighted_update_state_no_special_character(self):
    metric = keras_metrics.MaskedCategoricalAccuracy()
    metric.update_state(
        y_true=[[1, 2, 3, 4], [0, 0, 0, 0]],
        y_pred=[
            # A batch with 100% accuracy.
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
            # A batch with 50% accuracy.
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.0],
            ],
        ],
        # A weight for each `y_true` scalar.
        sample_weight=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    self.assertAllClose(self.evaluate(metric.result()), (6 + 4) / 12.0)
    metric.update_state(
        y_true=[[0, 4, 1, 2]],
        y_pred=[
            # A batch with 25% accruacy.
            [
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
        ],
        sample_weight=[1.0, 1.0, 2.0, 2.0])
    self.assertAllClose(self.evaluate(metric.result()), (6 + 4 + 2) / 18.0)

  def test_weighted_update_state_no_special_character_rank_2_sample_weight(
      self):
    metric = keras_metrics.MaskedCategoricalAccuracy()
    metric.update_state(
        y_true=[[1, 2, 3, 4], [0, 0, 0, 0]],
        y_pred=[
            # A batch with 100% accuracy.
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
            # A batch with 50% accuracy.
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.0],
            ],
        ],
        # A weight for each `y_true` scalar.
        sample_weight=[[1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0]])
    self.assertAllClose(self.evaluate(metric.result()), (6 + 4) / 12.0)

  def test_weighted_update_state_with_scalar_weight(self):
    metric = keras_metrics.MaskedCategoricalAccuracy()
    metric.update_state(
        y_true=[[1, 2, 3, 4]],
        y_pred=[
            # A batch with 50% accuracy.
            [
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
        ],
        sample_weight=1.0)
    self.assertAllClose(self.evaluate(metric.result()), .5)

  def test_weighted_update_state_special_character_rank_2_sample_weight(self):
    metric = keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[4])
    metric.update_state(
        y_true=[[1, 2, 3, 4], [0, 0, 0, 0]],
        y_pred=[
            # A batch with 100% accuracy.
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
            # A batch with 50% accuracy.
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.0],
            ],
        ],
        # A weight for each `y_true` scalar.
        sample_weight=[[1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0]])
    self.assertAllClose(self.evaluate(metric.result()), (6 + 2) / 10.0)

  def test_update_state_with_multiple_tokens_masked(self):
    metric = keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[1, 2, 3, 4])
    metric.update_state(
        y_true=[[1, 2, 3, 4], [0, 0, 0, 0]],
        y_pred=[
            [
                # This batch should be masked.
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
            [
                # Batch with 50% accuracy
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.0],
            ],
        ])
    self.assertAllClose(self.evaluate(metric.result()), 0.5)

  def test_update_state_with_all_tokens_masked(self):
    metric = keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[1, 2, 3, 4])
    metric.update_state(
        # All batches should be masked.
        y_true=[[1, 2, 3, 4], [4, 3, 2, 1]],
        y_pred=[
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.9],
            ],
            [
                [0.1, 0.9, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.9, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.0],
            ],
        ])
    self.assertAllClose(self.evaluate(metric.result()), 0.0)


if __name__ == '__main__':
  tf.test.main()
