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
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.research.utils.models import stackoverflow_models


class KerasSequenceModelsTest(absltest.TestCase):

  def test_constructs(self):
    model = stackoverflow_models.create_recurrent_model(10, name='rnn-lstm')
    self.assertIsInstance(model, tf.keras.Model)
    self.assertEqual('rnn-lstm', model.name)

  def test_shared_embedding_returns_dense_gradient_in_graph_mode(self):
    batch_size = 2
    sequence_length = 20
    batch_x = np.ones((batch_size, sequence_length), dtype=np.int32)
    batch_y = np.ones((batch_size, sequence_length), dtype=np.int32)
    graph = tf.Graph()
    with graph.as_default():
      model = stackoverflow_models.create_recurrent_model(shared_embedding=True)
      loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
      with tf.GradientTape() as tape:
        predictions = model(batch_x, training=True)
        loss = loss_fn(y_true=batch_y, y_pred=predictions)
      embedding_gradient = tape.gradient(loss, model.trainable_variables[0])
      init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      embedding_grad = sess.run(embedding_gradient)

    self.assertTrue(np.all(np.linalg.norm(embedding_grad, axis=1) > 0.0))


if __name__ == '__main__':
  absltest.main()
