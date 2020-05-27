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
"""Library providing untrained or trained models for EMNIST classification."""

import os.path

import tensorflow as tf

layers = tf.keras.layers
Model = tf.keras.Model

BASE_URL = 'https://storage.googleapis.com/tff-experiments-public/'
CLASSIFIER_CKPTS_PATH = 'gans/classifier/checkpoints/'

# This checkpoint was trained on EMNIST data for 3 epochs and has ~89.88% test
# accuracy.
CHECKPOINT_FILENAME = 'emnist_classifier_checkpoint_epochs_3_acc_89p88.hd5'


def _get_emnist_classifier_model_v1():
  """Model architecture for classifying EMNIST images into one of 36 classes.

  Returns:
    A Keras model that takes 28x28x1 images and returns logit vectors.

  This model is based on the one in the TF 2.0 Alpha page (Beginner Tutorials >
  Images > Convolutional NNs).
  https://www.tensorflow.org/alpha/tutorials/images/intro_to_cnn

  The only difference is changing the output dimension from 10 (# of MNIST
  classes) to 36 (the # of output classes in our problem: 10 numeric digits + 26
  letters).

  To simplify classification, we don't distinguish between letter case.
  Lowercase and uppercase of a letter are grouped together under a single output
  label. This gives greater classification accuracy (b/c we've removed case
  situations that are understandably hard for a CNN to disambiguate, e.g., 'x'
  from 'X'), and this higher baseline allows for more dramatic illustration of
  the drop in classification performance when a bug is introduced upstream of
  this network.
  """
  inputs = tf.keras.Input(shape=(28, 28, 1))  # Returns a placeholder tensor

  x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
  x = layers.MaxPooling2D((2, 2))(x)
  x = layers.Conv2D(64, (3, 3), activation='relu')(x)
  x = layers.MaxPooling2D((2, 2))(x)
  x = layers.Conv2D(64, (3, 3), activation='relu')(x)
  x = layers.Flatten()(x)
  x = layers.Dense(64, activation='relu')(x)

  logits = layers.Dense(36, activation='linear')(x)

  return Model(inputs=inputs, outputs=logits)


def get_emnist_classifier_model():
  """Untrained model for EMNIST image classification (into 36 classes)."""
  # If changing the architecture exposed, new checkpoint needs to be checked in.
  return _get_emnist_classifier_model_v1()


def _get_path_to_checkpoint(cache_dir=None):
  path = tf.keras.utils.get_file(
      fname=CHECKPOINT_FILENAME,
      cache_subdir=CLASSIFIER_CKPTS_PATH,
      cache_dir=cache_dir,
      origin=os.path.join(BASE_URL, CLASSIFIER_CKPTS_PATH, CHECKPOINT_FILENAME))
  return path


def get_trained_emnist_classifier_model(cache_dir=None):
  """Already trained model for EMNIST image classification (into 36 classes)."""
  model = get_emnist_classifier_model()
  model.load_weights(_get_path_to_checkpoint(cache_dir))
  return model
