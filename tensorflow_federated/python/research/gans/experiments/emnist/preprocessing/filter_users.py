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
r"""Filter users in Federated EMNIST, based on user example classification.

This script applies the image inversion bug to a percentage of the users in the
Federated EMNIST dataset, runs the EMNIST classifier on all examples from all
users, and computes for each user an overall classification accuracy. It then
identifies whether a user belongs to a subpopulation of 'good' users (i.e., with
accuracy exceeding a cutoff) and whether a user a belongs to a subpopulation of
'bad' users (i.e., with accuracy not meeting a cutoff).

It then saves these two subpopulations so that training can occur without
needing to rerun the EMNIST classifier on all the data each time. Specifically,
for each subpopulation, we save a csv of key/value pairs, where the keys are the
Federated EMNIST client IDs and the values are a boolean for whether the data is
inverted or not.

To run, first build the binary:
  bazel build /path/to/this/script:filter_users
Then execute it:
  ./path/to/binary/filter_users --invert_imagery_likelihood=0.1 \
    --bad_accuracy_cutoff=0.5 --good_accuracy_cutoff=0.9
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
  flags.DEFINE_float(
      'bad_accuracy_cutoff', 0.862,
      'The classification accuracy (when classifying all the data in a client\'s '
      'dataset) to use as a threshold for deciding whether a client is included '
      'in the population of poorly-performing devices for which we\'ll train a '
      'GAN.')
  flags.DEFINE_float(
      'good_accuracy_cutoff', 0.947,
      'The classification accuracy (when classifying all the data in a client\'s '
      'dataset) to use as a threshold for deciding whether a client is included '
      'in the population of well-performing devices for which we\'ll train a '
      'GAN.')
  flags.DEFINE_string(
      'path_to_save_bad_clients_csv',
      '/tmp/selected_emnist_bad_client_ids_map.csv',
      'The path/name at which to save a csv file of a dictionary, where the keys '
      'are poorly-performing device client ids from the Federated EMNIST '
      'dataset, and the values are a boolean for whether the image intensity was '
      'inverted for that client id.')
  flags.DEFINE_string(
      'path_to_save_good_clients_csv',
      '/tmp/selected_emnist_good_client_ids_map.csv',
      'The path/name at which to save a csv file of a dictionary, where the keys '
      'are well-performing device client ids from the Federated EMNIST dataset, '
      'and the values are a boolean for whether the image intensity was inverted '
      'for that client id.')
FLAGS = flags.FLAGS


def _analyze_classifier(images_ds, classifier_model):
  """Measure accuracy of the classifier model against the labeled dataset."""

  def reduce_fn(total_count_and_correct_count, image_and_label):
    """Reduces total and correctly classified example counts across a client."""
    total_count, correct_count = total_count_and_correct_count
    image, label = image_and_label

    logit = tf.squeeze(classifier_model(tf.expand_dims(image, axis=0)), axis=0)
    predicted_label = tf.math.argmax(logit, output_type=tf.int32)

    return (total_count + 1, correct_count +
            tf.cast(tf.math.equal(label, predicted_label), tf.int32))

  total_count, correct_count = images_ds.reduce((0, 0), reduce_fn)

  return total_count, correct_count


def _get_client_ids_meeting_condition(train_tff_data, bad_accuracy_cutoff,
                                      good_accuracy_cutoff,
                                      invert_imagery_likelihood,
                                      classifier_model):
  """Get clients that classify <bad_accuracy_cutoff or >good_accuracy_cutoff."""
  bad_client_ids_inversion_map = {}
  good_client_ids_inversion_map = {}
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
    # Run classifier on all data on client, compute % classified correctly.
    total_count, correct_count = _analyze_classifier(images_ds,
                                                     classifier_model)
    accuracy = float(correct_count) / float(total_count)

    if accuracy < bad_accuracy_cutoff:
      bad_client_ids_inversion_map[client_id] = invert_imagery
    if accuracy > good_accuracy_cutoff:
      good_client_ids_inversion_map[client_id] = invert_imagery

  return bad_client_ids_inversion_map, good_client_ids_inversion_map


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
  if FLAGS.bad_accuracy_cutoff > 1.0:
    raise ValueError('bad_accuracy_cutoff cannot be greater than 1.0')
  if FLAGS.good_accuracy_cutoff > 1.0:
    raise ValueError('good_accuracy_cutoff cannot be greater than 1.0')

  # Training datasets.
  client_real_images_train_tff_data = (
      emnist_data_utils.create_real_images_tff_client_data('train'))

  print('There are %d unique clients.' %
        len(client_real_images_train_tff_data.client_ids))

  # Trained classifier model.
  classifier_model = ecm.get_trained_emnist_classifier_model()

  # Filter down to those client IDs that fall within some accuracy cutoff.
  bad_client_ids_inversion_map, good_client_ids_inversion_map = (
      _get_client_ids_meeting_condition(client_real_images_train_tff_data,
                                        FLAGS.bad_accuracy_cutoff,
                                        FLAGS.good_accuracy_cutoff,
                                        FLAGS.invert_imagery_likelihood,
                                        classifier_model))

  print('There are %d unique clients meeting bad accuracy cutoff condition.' %
        len(bad_client_ids_inversion_map))
  print('There are %d unique clients meeting good accuracy cutoff condition.' %
        len(good_client_ids_inversion_map))

  # Save selected client id dictionary to csv.
  with tf.io.gfile.GFile(FLAGS.path_to_save_bad_clients_csv, 'w') as csvfile:
    w = csv.writer(csvfile)
    for key, val in bad_client_ids_inversion_map.items():
      w.writerow([key, val])

  with tf.io.gfile.GFile(FLAGS.path_to_save_good_clients_csv, 'w') as csvfile:
    w = csv.writer(csvfile)
    for key, val in good_client_ids_inversion_map.items():
      w.writerow([key, val])

  print('CSV files with selected Federated EMNIST clients have been saved.')


if __name__ == '__main__':
  app.run(main)
