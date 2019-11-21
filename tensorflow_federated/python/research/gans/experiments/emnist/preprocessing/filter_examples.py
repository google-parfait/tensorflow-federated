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
r"""Filters EMNIST users' examples to correctly and incorrectly classified sets.

This script applies the image inversion bug to a percentage of the users in the
Federated EMNIST dataset and runs the EMNIST classifier (the `primary model`) on
all users' examples. It then considers two distinct sub-datasets: all user
examples that correctly classified (primary model predicts the correct label)
and all user examples that incorrectly classified (primary model predicts the
wrong label).

It then saves these two sub-datasets so that training can occur without
needing to rerun the EMNIST classifier on all the data each time. Specifically,
for each subpopulation we save two csvs. One is a csv of key/value pairs where
the keys are the Federated EMNIST client IDs and the values are a boolean for
whether the data is inverted or not. The other csv contains key/value pairs
where the key is the client ID and the value is a list (in string form) of
indices of the user's examples that classify correctly or incorrectly.

To run, first build the binary:
  bazel build /path/to/this/script:filter_examples
Then execute it:
  ./path/to/binary/filter_examples --invert_imagery_likelihood=0.1 \
    --min_num_examples=5
"""

import collections
import csv

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.research.gans.experiments.emnist import emnist_data_utils
from tensorflow_federated.python.research.gans.experiments.emnist.classifier import emnist_classifier_model as ecm
from tensorflow_federated.python.research.utils import utils_impl

with utils_impl.record_new_flags() as hparam_flags:
  flags.DEFINE_float(
      'invert_imagery_likelihood', 0.0,
      'The likelihood that a given client has a flipped sign on the pixels of '
      'its EMNIST raw images.')
  flags.DEFINE_integer(
      'min_num_examples', 5,
      'The minimum number of examples a client needs to be included in a '
      'training sub-population. E.g., if this is set to 5, and client X\'s '
      'local dataset contains 5 correctly classified examples and 4 '
      'incorrectly classified examples, then client X will be included as a '
      'key in the datastructures storing the clients with correct examples, '
      'but will not be included as a key in the datastructures storing the '
      'clients with incorrect examples.')
  flags.DEFINE_string(
      'path_to_save_clients_with_correct_examples_csv',
      '/tmp/emnist_client_ids_with_correct_examples.csv',
      'The path/name at which to save a csv file of a dictionary, where the '
      'keys are correct example-containing client IDs from the Federated '
      'EMNIST dataset, and the values are a boolean for whether the image '
      'intensity was inverted for that client id. If a client ID is not a key '
      'in this map, it means there were less than min_num_examples '
      'correctly-classified examples in the client\'s local dataset.')
  flags.DEFINE_string(
      'path_to_save_clients_with_incorrect_examples_csv',
      '/tmp/emnist_client_ids_with_incorrect_examples.csv',
      'The path/name at which to save a csv file of a dictionary, where the '
      'keys are incorrect example-containing client IDs from the Federated '
      'EMNIST dataset, and the values are a boolean for whether the image '
      'intensity was inverted for that client id. If a client ID is not a key '
      'in this map, it means there were less than min_num_examples '
      'incorrectly-classified examples in the client\'s local dataset.')
  flags.DEFINE_string(
      'path_to_save_correct_example_indices_csv',
      '/tmp/emnist_client_ids_correct_examples_map.csv',
      'The path/name at which to save a csv file of a dictionary, where the '
      'keys are client IDs from the Federated EMNIST dataset, and the values '
      'are a list of indices of examples that classified correctly on the '
      'client ID.')
  flags.DEFINE_string(
      'path_to_save_incorrect_example_indices_csv',
      '/tmp/emnist_client_ids_incorrect_examples_map.csv',
      'The path/name at which to save a csv file of a dictionary, where the '
      'keys are client IDs from the Federated EMNIST dataset, and the values '
      'are a list of indices of examples that classified incorrectly on the '
      'client ID.')
FLAGS = flags.FLAGS


def _analyze_classifier(images_ds, classifier_model):
  """Measure accuracy of the classifier model against the labeled dataset."""
  correct_indices = []
  incorrect_indices = []

  index = 0
  for image, label in images_ds:
    logit = tf.squeeze(classifier_model(tf.expand_dims(image, axis=0)), axis=0)
    predicted_label = tf.math.argmax(logit, output_type=tf.int32)
    if tf.math.equal(label, predicted_label).numpy():
      correct_indices.append(index)
    else:
      incorrect_indices.append(index)

    index += 1

  return correct_indices, incorrect_indices


def _get_client_ids_and_examples_based_on_classification(
    train_tff_data, min_num_examples, invert_imagery_likelihood,
    classifier_model):
  """Get maps storing whether imagery inverted and how examples classified."""
  client_ids_with_correct_examples_map = {}
  client_ids_with_incorrect_examples_map = {}
  client_ids_correct_example_indices_map = {}
  client_ids_incorrect_example_indices_map = {}

  for client_id in train_tff_data.client_ids:
    invert_imagery = (1 == np.random.binomial(n=1, p=invert_imagery_likelihood))

    # TF Dataset for particular client.
    raw_images_ds = train_tff_data.create_tf_dataset_for_client(client_id)
    # Preprocess into format expected by classifier.
    images_ds = emnist_data_utils.preprocess_img_dataset(
        raw_images_ds,
        invert_imagery=invert_imagery,
        include_label=True,
        batch_size=None,
        shuffle=False,
        repeat=False)
    # Run classifier on all data on client, return lists of indices of examples
    # classified correctly and incorrectly.
    correct_indices, incorrect_indices = _analyze_classifier(
        images_ds, classifier_model)

    if len(correct_indices) >= min_num_examples:
      client_ids_with_correct_examples_map[client_id] = invert_imagery
      client_ids_correct_example_indices_map[client_id] = correct_indices

    if len(incorrect_indices) >= min_num_examples:
      client_ids_with_incorrect_examples_map[client_id] = invert_imagery
      client_ids_incorrect_example_indices_map[client_id] = incorrect_indices

  return (client_ids_with_correct_examples_map,
          client_ids_with_incorrect_examples_map,
          client_ids_correct_example_indices_map,
          client_ids_incorrect_example_indices_map)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.compat.v1.enable_v2_behavior()

  # Flags.
  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])
  for k in hparam_dict.keys():
    if hparam_dict[k] is None:
      hparam_dict[k] = 'None'
  for k, v in hparam_dict.items():
    print('{} : {} '.format(k, v))

  if FLAGS.invert_imagery_likelihood > 1.0:
    raise ValueError('invert_imagery_likelihood cannot be greater than 1.0')

  # Training datasets.
  client_real_images_train_tff_data = (
      emnist_data_utils.create_real_images_tff_client_data('train'))

  print('There are %d unique clients.' %
        len(client_real_images_train_tff_data.client_ids))

  # Trained classifier model.
  classifier_model = ecm.get_trained_emnist_classifier_model()

  # Filter down to those client IDs that fall within some accuracy cutoff.
  (client_ids_with_correct_examples_map, client_ids_with_incorrect_examples_map,
   client_ids_correct_example_indices_map,
   client_ids_incorrect_example_indices_map) = (
       _get_client_ids_and_examples_based_on_classification(
           client_real_images_train_tff_data, FLAGS.min_num_examples,
           FLAGS.invert_imagery_likelihood, classifier_model))

  print('There are %d unique clients with at least %d correct examples.' %
        (len(client_ids_with_correct_examples_map), FLAGS.min_num_examples))
  print('There are %d unique clients with at least %d incorrect examples.' %
        (len(client_ids_with_incorrect_examples_map), FLAGS.min_num_examples))

  # Save client id dictionarys to csv.
  with tf.io.gfile.GFile(FLAGS.path_to_save_clients_with_correct_examples_csv,
                         'w') as csvfile:
    w = csv.writer(csvfile)
    for key, val in client_ids_with_correct_examples_map.items():
      w.writerow([key, val])

  with tf.io.gfile.GFile(FLAGS.path_to_save_clients_with_incorrect_examples_csv,
                         'w') as csvfile:
    w = csv.writer(csvfile)
    for key, val in client_ids_with_incorrect_examples_map.items():
      w.writerow([key, val])

  with tf.io.gfile.GFile(FLAGS.path_to_save_correct_example_indices_csv,
                         'w') as csvfile:
    w = csv.writer(csvfile)
    for key, val in client_ids_correct_example_indices_map.items():
      w.writerow([key, val])

  with tf.io.gfile.GFile(FLAGS.path_to_save_incorrect_example_indices_csv,
                         'w') as csvfile:
    w = csv.writer(csvfile)
    for key, val in client_ids_incorrect_example_indices_map.items():
      w.writerow([key, val])

  print('CSV files with selected Federated EMNIST clients and lists of '
        'classified/misclassified examples have been saved.')


if __name__ == '__main__':
  app.run(main)
