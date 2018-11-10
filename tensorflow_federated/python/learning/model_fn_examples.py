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
"""Example ModelFns, also used in tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.learning import model_fn
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class NoFeaturesRegressionModelFn(model_fn.ModelFn):
  """A simple linear regression model."""

  def __init__(self, initial_model_param=1.0):
    self._initial_model_param = initial_model_param

  def build(self):
    parsed_batch = tf.parse_example(
        model_fn.model_input_tensor(), {
            'feature': tf.FixedLenFeature([], tf.float32),
            'label': tf.FixedLenFeature([], tf.float32)
        })
    # This simple model ignores the features for now.
    labels = parsed_batch['label']
    model_var = tf.get_variable(
        'model_param', shape=(),
        initializer=tf.constant_initializer(self._initial_model_param))
    predictions = model_var
    residuals = predictions - labels
    squared_error = 0.5 * tf.reduce_mean(tf.pow(residuals, 2))
    num_examples = tf.size(labels)

    metrics = []
    update_ops = []
    predictions_per_label = tf.broadcast_to(model_var, shape=tf.shape(labels))

    #
    # Set up sum metrics (a.k.a. counters).
    #

    # N.B. We only want to increment the number of minibatches if
    # we are actually processing data. Currently, the only
    # way to do this seems to be to add a control dependency here.
    # See also near fc.run_repeatedly in the client computation in
    # federated_averaging.py.
    with tf.control_dependencies([labels]):
      one = tf.constant(1, name='one_batch')
    counters = OrderedDict(
        num_examples=tensor_utils.metrics_sum(
            num_examples, name='num_examples'),
        num_minibatches=tensor_utils.metrics_sum(one, name='num_minibatches'),
        label_sum=tensor_utils.metrics_sum(labels, name='label_sum'))
    for name, (value, update_op) in counters.iteritems():
      metrics.append(model_fn.Metric.sum(name, value))
      update_ops.append(update_op)

    #
    # Set up weighted average metrics.
    #

    # When averaging metrics across clients, we weight by he total
    # number of examples summed across batches, which
    # we get via the value variable from the corresponding metrics_sum:
    total_client_weight = counters['num_examples'][0]
    avg_metrics = OrderedDict(
        avg_prediction=tf.metrics.mean(predictions_per_label,
                                       weights=num_examples,
                                       name='mean_prediction'),
        avg_label=tf.metrics.mean(labels, weights=num_examples,
                                  name='mean_label'),
        squared_error=tf.metrics.mean(squared_error, weights=num_examples,
                                      name='mean_squared_error'),
        abs_error=tf.metrics.mean_absolute_error(
            labels, predictions_per_label, weights=num_examples))
    for name, (value, update_op) in avg_metrics.iteritems():
      metrics.append(model_fn.Metric.average(name, value, total_client_weight))
      update_ops.append(update_op)

    return model_fn.ModelSpec(
        loss=squared_error,
        minibatch_update_ops=update_ops,
        metrics=metrics)

  @classmethod
  def make_tf_example(cls, x):
    """Returns a serialized example with feature x and label y = 2*x + 1."""
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'feature':
                    tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
                'label':
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=[2.0 * x + 1.0])),
            })).SerializeToString()

  @classmethod
  def client_dict_for_fc_data(cls, location, client_data=None):
    """Formats a dictionary of per-client data for passing to an environment().

    Typicall usage:
      client_dict_for_fc_data(
        LOCATION,
        client_data={
            'device_one': [[4.0, 4.0, 4.0], [0.0, 0.0]],
            'device_two': [[0.0]]})

    Args:
      location: The location string for the data.
      client_data: A dict keyed by a string client_id, with values being
        lists of features x which are mapped to tf.train.Examples.

    Returns:
      A dictionary to pass to the data argument of an environment.environment().
    """

    assert client_data is not None
    return {
        location: {
            client_name: [
                np.array(
                    [cls.make_tf_example(x) for x in batch], dtype=np.object)
                for batch in batch_stream
            ] for client_name, batch_stream in client_data.iteritems()
        }
    }
