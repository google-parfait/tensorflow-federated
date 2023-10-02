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
import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.learning.metrics import counters
from tensorflow_federated.python.learning.metrics import keras_utils

# Names of Keras metrics to test.
BINARY_METRIC_NAMES = [
    'Accuracy',
    'BinaryAccuracy',
    'BinaryCrossentropy',
    'CategoricalAccuracy',
    'CategoricalCrossentropy',
    'CategoricalHinge',
    'CosineSimilarity',
    'FalseNegatives',
    'FalsePositives',
    'KLDivergence',
    'MeanAbsoluteError',
    'MeanSquaredError',
    'AUC',
]

# keras_utils assumes that the input to a metric's `update` method has a
# `prediction` attribute.
# TODO: b/259609586 - Remove this class when `FunctionalModel` has an explicit
# loss function.
_BatchOutput = collections.namedtuple(
    'BatchOutput',
    ['predictions'],
    defaults=[None],
)


class CreateFunctionalMetricTest(tf.test.TestCase, parameterized.TestCase):

  def assertAllEqual(self, a, b, msg=None):
    # assertAllEqual fails to walk nested structures nicely, but assertAllClose
    # does, so we can simply set the tolerance to 0.0 to achieve equality tests.
    self.assertAllClose(a, b, msg=msg, rtol=0.0, atol=0.0)

  def assert_no_variable_ops(self, graph_def):
    all_nodes = itertools.chain(
        graph_def.node, *[f.node_def for f in graph_def.library.function]
    )
    self.assertEmpty([node.op for node in all_nodes if 'Variable' in node.op])

  def assert_binary_metric_variableless_functions(
      self, initialize_fn, update_fn, finalize_fn
  ):
    self.assert_no_variable_ops(
        initialize_fn.get_concrete_function().graph.as_graph_def()
    )
    state_spec = tf.nest.map_structure(
        tf.TensorSpec.from_tensor, initialize_fn()
    )
    self.assert_no_variable_ops(
        update_fn.get_concrete_function(
            state=state_spec,
            labels=tf.TensorSpec(shape=[None], dtype=tf.float32),
            batch_output=_BatchOutput(
                predictions=tf.TensorSpec(shape=[None], dtype=tf.float32)
            ),
        ).graph.as_graph_def()
    )
    self.assert_no_variable_ops(
        finalize_fn.get_concrete_function(state=state_spec).graph.as_graph_def()
    )

  @parameterized.named_parameters(
      ('NumBatchesCounter', counters.NumBatchesCounter),
      ('NumExamplesCounter', counters.NumExamplesCounter),
  )
  def test_custom_tff_metrics(self, metric_constructor):
    metric = metric_constructor()
    initialize, update, finalize = keras_utils.create_functional_metric_fns(
        metric_constructor
    )
    self.assert_binary_metric_variableless_functions(
        initialize, update, finalize
    )
    state = initialize()
    self.assertAllEqual(metric.variables, state)
    predictions = np.asarray([0.0, 1.0])
    labels = np.asarray([1.0, 1.0])
    metric.update_state(y_pred=predictions, y_true=labels)
    batch_output = _BatchOutput(predictions=predictions)
    state = update(state, batch_output=batch_output, labels=labels)
    self.assertAllEqual(metric.variables, state)
    self.assertAllEqual(metric.result(), finalize(state))
    predictions = np.asarray([0.0, 1.0, 0.0])
    labels = np.asarray([1.0, 1.0, 0.0])
    metric.update_state(y_pred=predictions, y_true=labels)
    batch_output = _BatchOutput(predictions=predictions)
    state = update(state, batch_output=batch_output, labels=labels)
    self.assertAllEqual(metric.variables, state)
    self.assertAllEqual(metric.result(), finalize(state))

  @parameterized.named_parameters(
      (name, getattr(tf.keras.metrics, name)) for name in BINARY_METRIC_NAMES
  )
  def test_binary_metrics_graph(self, metric_constructor):
    with tf.Graph().as_default():
      with tf.compat.v1.Session() as sess:
        metric = metric_constructor()
        sess.run(tf.compat.v1.initializers.variables(metric.variables))
        initialize, update, finalize = keras_utils.create_functional_metric_fns(
            metric_constructor
        )
        self.assert_binary_metric_variableless_functions(
            initialize, update, finalize
        )
        state = initialize()
        self.assertAllEqual(self.evaluate(metric.variables), state)
        predictions = [0.0, 1.0]
        labels = [1.0, 1.0]
        self.evaluate(metric.update_state(y_pred=predictions, y_true=labels))
        batch_output = _BatchOutput(predictions=predictions)
        state = update(state, batch_output=batch_output, labels=labels)
        self.assertAllEqual(
            self.evaluate(metric.variables), self.evaluate(state)
        )
        self.assertAllEqual(
            self.evaluate(metric.result()), self.evaluate(finalize(state))
        )
        predictions = [0.0, 1.0, 0.0]
        labels = [1.0, 1.0, 0.0]
        self.evaluate(metric.update_state(y_pred=predictions, y_true=labels))
        batch_output = _BatchOutput(predictions=predictions)
        state = update(state, batch_output=batch_output, labels=labels)
        self.assertAllEqual(
            self.evaluate(metric.variables), self.evaluate(state)
        )
        self.assertAllEqual(
            self.evaluate(metric.result()), self.evaluate(finalize(state))
        )

  @parameterized.named_parameters(
      (name, getattr(tf.keras.metrics, name)) for name in BINARY_METRIC_NAMES
  )
  def test_binary_metrics_eager(self, metric_constructor):
    metric = metric_constructor()
    initialize, update, finalize = keras_utils.create_functional_metric_fns(
        metric_constructor
    )
    self.assert_binary_metric_variableless_functions(
        initialize, update, finalize
    )
    state = initialize()
    self.assertAllEqual(metric.variables, state)
    predictions = [0.0, 1.0]
    labels = [1.0, 1.0]
    metric.update_state(y_pred=predictions, y_true=labels)
    batch_output = _BatchOutput(predictions=predictions)
    state = update(state, batch_output=batch_output, labels=labels)
    self.assertAllEqual(metric.variables, state)
    self.assertAllEqual(metric.result(), finalize(state))
    predictions = [0.0, 1.0, 0.0]
    labels = [1.0, 1.0, 0.0]
    metric.update_state(y_pred=predictions, y_true=labels)
    batch_output = _BatchOutput(predictions=predictions)
    state = update(state, batch_output=batch_output, labels=labels)
    self.assertAllEqual(metric.variables, state)
    self.assertAllEqual(metric.result(), finalize(state))

  def test_non_default_metric(self):
    def metric_constructor():
      return tf.keras.metrics.Precision(thresholds=[0.25, 0.5, 0.75])

    metric = metric_constructor()
    initialize, update, finalize = keras_utils.create_functional_metric_fns(
        metric_constructor
    )
    state = initialize()
    self.assertAllEqual(metric.variables, state)
    predictions = [0.0, 1.0]
    labels = [1.0, 1.0]
    metric.update_state(y_pred=predictions, y_true=labels)
    batch_output = _BatchOutput(predictions=predictions)
    state = update(state, batch_output=batch_output, labels=labels)
    self.assertAllEqual(metric.variables, state)
    self.assertAllEqual(metric.result(), finalize(state))
    predictions = [0.0, 1.0, 0.0]
    labels = [1.0, 1.0, 0.0]
    metric.update_state(y_pred=predictions, y_true=labels)
    batch_output = _BatchOutput(predictions=predictions)
    state = update(state, batch_output=batch_output, labels=labels)
    self.assertAllEqual(metric.variables, state)
    self.assertAllEqual(metric.result(), finalize(state))

  def test_multiple_metrics_constructor(self):
    def metrics_constructor():
      return collections.OrderedDict(
          precisions=tf.keras.metrics.Precision(thresholds=[0.25, 0.5, 0.75]),
          accuracy=tf.keras.metrics.Accuracy(),
      )

    metrics = metrics_constructor()
    initialize, update, finalize = keras_utils.create_functional_metric_fns(
        metrics_constructor
    )
    state = initialize()
    self.assertAllEqual(
        tf.nest.map_structure(lambda m: m.variables, metrics), state
    )
    predictions = [0.0, 1.0]
    labels = [1.0, 1.0]
    tf.nest.map_structure(
        lambda m: m.update_state(y_pred=predictions, y_true=labels), metrics
    )
    batch_output = _BatchOutput(predictions=predictions)
    state = update(state, batch_output=batch_output, labels=labels)
    self.assertAllEqual(
        tf.nest.map_structure(lambda m: m.variables, metrics), state
    )
    self.assertAllEqual(
        tf.nest.map_structure(lambda m: m.result(), metrics), finalize(state)
    )
    predictions = [0.0, 1.0, 0.0]
    labels = [1.0, 1.0, 0.0]
    tf.nest.map_structure(
        lambda m: m.update_state(y_pred=predictions, y_true=labels), metrics
    )
    batch_output = _BatchOutput(predictions=predictions)
    state = update(state, batch_output=batch_output, labels=labels)
    self.assertAllEqual(
        tf.nest.map_structure(lambda m: m.variables, metrics), state
    )
    self.assertAllEqual(
        tf.nest.map_structure(lambda m: m.result(), metrics), finalize(state)
    )

  @parameterized.named_parameters(
      ('sum', tf.keras.metrics.Sum),
      ('mean', tf.keras.metrics.Mean),
  )
  def test_unary_metrics(self, metric_constructor):
    initialize, update, _ = keras_utils.create_functional_metric_fns(
        metric_constructor
    )
    state = initialize()
    batch_output = _BatchOutput(predictions=[0.0, 1.0])
    labels = [1.0, 1.0]
    with self.assertRaisesRegex(
        TypeError, 'got an unexpected keyword argument'
    ):
      update(state, batch_output=batch_output, labels=labels)

  def test_composite_metrics_fn(self):
    # We purposely use a constructor to an OrderedDict of metrics in
    # non-lexical-key-sorted order to ensure the test covered metric variables
    # be initialized _not_ in the order of the flattened structure return by
    # construction.
    def metrics_constructor():
      return collections.OrderedDict(
          precision_at_5=tf.keras.metrics.Precision(thresholds=[0.5]),
          num_batches=counters.NumBatchesCounter(dtype=tf.int64),
          accuracy=tf.keras.metrics.Accuracy(),
          num_examples=counters.NumExamplesCounter(dtype=tf.int64),
      )

    initialize, update, finalize = keras_utils.create_functional_metric_fns(
        metrics_constructor
    )
    metrics_by_name = metrics_constructor()
    state = initialize()
    self.assertAllClose(
        state, tf.nest.map_structure(lambda m: m.variables, metrics_by_name)
    )
    predictions = np.asarray([0.25, 0.5, 1.0])
    labels = np.asarray([0.0, 0.0, 1.0])
    batch_output = _BatchOutput(predictions=predictions)
    state = update(state, labels=labels, batch_output=batch_output)
    tf.nest.map_structure(
        lambda m: m.update_state(y_true=labels, y_pred=predictions),
        metrics_by_name,
    )
    self.assertAllClose(
        state, tf.nest.map_structure(lambda m: m.variables, metrics_by_name)
    )
    finalized_metrics = finalize(state)
    metric_result = tf.nest.map_structure(lambda m: m.result(), metrics_by_name)
    self.assertAllClose(finalized_metrics, metric_result)


if __name__ == '__main__':
  tf.test.main()
