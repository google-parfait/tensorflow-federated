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
"""Script to train EMNIST classifier model in emnist_classifier_model.py.

The emnist_classifier_model library provides a pretrained instance of the Keras
model. However, if wishing to train from scratch your own EMNIST classifier, you
can use this script.

To run, first build the binary:
  bazel build /path/to/this/script:train_emnist_classifier_model
Then execute it:
  ./path/to/binary/train_emnist_classifier_model --epochs=3
"""

from absl import app
from absl import flags
import tensorflow as tf

from tensorflow_federated.python.research.gans.experiments.emnist import emnist_data_utils
from tensorflow_federated.python.research.gans.experiments.emnist.classifier import emnist_classifier_model

Model = tf.keras.Model

flags.DEFINE_string(
    'path_to_save_model_checkpoint', '/tmp/emnist_classifier_checkpoint.hd5',
    'The path/name at which to save a trained checkpoint of the '
    'EMNIST classifier model, after --epochs of training.')
flags.DEFINE_integer('epochs', 1, 'The # of epochs to train for.')

FLAGS = flags.FLAGS

BATCH_SIZE = 32


def _load_and_preprocess_datasets():
  """Load raw EMNIST data and preprocess images and labels."""
  emnist_train, emnist_test = (
      emnist_data_utils.create_real_images_tff_client_data())

  # Raw image datasets.
  train_dataset = emnist_train.create_tf_dataset_from_all_clients()
  test_dataset = emnist_test.create_tf_dataset_from_all_clients()

  # Preprocessed image datasets.
  preprocessed_train_dataset = emnist_data_utils.preprocess_img_dataset(
      train_dataset, include_label=True, batch_size=BATCH_SIZE, shuffle=True)
  preprocessed_test_dataset = emnist_data_utils.preprocess_img_dataset(
      test_dataset, include_label=True, batch_size=BATCH_SIZE, shuffle=False)

  return preprocessed_train_dataset, preprocessed_test_dataset


def _train_and_evaluate(preprocessed_train_dataset, preprocessed_test_dataset,
                        epochs):
  """Train (and evaluate) the classifier model defined in this module."""

  # Model.
  # This is the model we actually want to save and use in the future. It outputs
  # 'logits', i.e., the values that then would get passed to a softmax layer to
  # calculate the probabilities of each category in the output space.
  logits_model = emnist_classifier_model.get_emnist_classifier_model()
  # For training purposes, we need to make use of this model, which outputs the
  # probabilities directly (i.e., post softmax). The outputs of this model will
  # be what's used to calculate loss during the course of training/evaluation.
  inputs = tf.keras.Input(shape=(28, 28, 1))  # Returns a placeholder tensor
  probs_model = tf.keras.Model(
      inputs=inputs, outputs=tf.nn.softmax(logits_model(inputs)))

  # Loss and Optimizer.
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam()

  probs_model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

  # Training.
  probs_model.fit(preprocessed_train_dataset, epochs=epochs)

  # Evaluation.
  eval_loss, eval_accuracy = probs_model.evaluate(preprocessed_test_dataset)
  print('Evaluation loss and accuracy are...')
  print(eval_loss)
  print(eval_accuracy)

  return logits_model


def _save(model, path_to_save_model_checkpoint):
  model.save_weights(path_to_save_model_checkpoint, save_format='h5')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Datasets.
  preprocessed_train_dataset, preprocessed_test_dataset = (
      _load_and_preprocess_datasets())

  # Train and Evaluate Model.
  model = _train_and_evaluate(
      preprocessed_train_dataset,
      preprocessed_test_dataset,
      epochs=FLAGS.epochs)

  # Save Model.
  _save(model, FLAGS.path_to_save_model_checkpoint)


if __name__ == '__main__':
  app.run(main)
