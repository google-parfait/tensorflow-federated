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
r"""Script that measures the classification accuracy amongst different clients.

We consider a bug in the data pipeline: due to a faulty application update, the
pixel intensities of the handwriting images are inverted (i.e., black becomes
white, and white becomes black). For users that have updated their app, all data
examples are affected.

This script provides statistics on how well the pretrained
emnist_classifier_model performs in the presence of this bug. A user sets via
input flag the `invert_imagery_likelihood`; this represents the percentage of
users in the overall population who have received the bad application update.
The classification accuracy of each user is computed, and summary statistics
(histogram and 25th and 75th percentiles of user accuracy) are output. It can be
observed that as `invert_imagery_likelihood` is increased, the rate of
misclassification of user data increases.

The given percentiles can then be used to identify thresholds to classify users
as "low accuracy" (bottom 25%) and "high accuracy" (top 25%). In the case where
we filter by user, these thresholds are then used to identify two subpopulations
of users on which we will train Federated GANs.

To run, first build the binary:
  bazel build /path/to/this/script:measure_misclassification_of_users
Then execute it:
  ./path/to/binary/measure_misclassification_of_users \
    --invert_imagery_likelihood=0.0
"""

import math

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.research.gans.experiments.emnist import emnist_data_utils
from tensorflow_federated.python.research.gans.experiments.emnist.classifier import emnist_classifier_model as ecm

flags.DEFINE_float(
    'invert_imagery_likelihood', 0.0,
    'The likelihood that a given client has black and white flipped in its '
    'EMNIST images. I.e., if the pixels are represented as floats in the range '
    '[-1.0, 1.0], this is the likelihood a given raw image has been multiplied '
    'by -1.0.')
FLAGS = flags.FLAGS


def _analyze_classifier(images_ds, classifier_model):
  """Measure accuracy of the classifier model against the labeled dataset."""

  def reduce_fn(total_count_and_correct_count, image_and_label):
    total_count, correct_count = total_count_and_correct_count
    image, label = image_and_label

    logit = tf.squeeze(classifier_model(tf.expand_dims(image, axis=0)), axis=0)
    predicted_label = tf.math.argmax(logit, output_type=tf.int32)

    return (total_count + 1, correct_count +
            tf.cast(tf.math.equal(label, predicted_label), tf.int32))

  total_count, correct_count = images_ds.reduce((0, 0), reduce_fn)

  return total_count, correct_count


def _compute_histogram(accuracy_list, bin_width=1):
  """Compute a histogram of per user cumulative classification accuracy."""
  n_bins = tf.cast(100 / bin_width, dtype=tf.int32)

  histogram = tf.zeros([n_bins], dtype=tf.int32)
  for accuracy in accuracy_list:
    histogram_value = math.floor(100 * accuracy / float(bin_width))
    if histogram_value == 100.0:
      histogram_value -= 1.0
    histogram_index = tf.cast(histogram_value, dtype=tf.int32)
    one_hot = tf.one_hot(indices=histogram_index, depth=n_bins, dtype=tf.int32)
    histogram += one_hot

  return histogram


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  invert_imagery_likelihood = FLAGS.invert_imagery_likelihood
  print('invert_imagery_likelihood is %s' % invert_imagery_likelihood)
  if invert_imagery_likelihood > 1.0:
    raise ValueError('invert_imagery_likelihood cannot be greater than 1.0')

  # TFF Dataset.
  client_real_images_tff_data = (
      emnist_data_utils.create_real_images_tff_client_data(split='train'))
  print('There are %d unique clients.' %
        len(client_real_images_tff_data.client_ids))

  # EMNIST Classifier.
  classifier_model = ecm.get_trained_emnist_classifier_model()

  accuracy_list = []
  overall_total_count = 0
  overall_correct_count = 0
  for client_id in client_real_images_tff_data.client_ids:
    invert_imagery = (1 == np.random.binomial(n=1, p=invert_imagery_likelihood))

    # TF Dataset for particular client.
    raw_images_ds = client_real_images_tff_data.create_tf_dataset_for_client(
        client_id)
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
    accuracy_list.append(accuracy)

    overall_total_count += total_count
    overall_correct_count += correct_count

  # Calculate histogram.
  bin_width = 1
  histogram = _compute_histogram(accuracy_list, bin_width)
  print('\nHistogram:')
  print(histogram.numpy())
  # Sanity check (should be 3400)
  print('(Histogram sum):')
  print(sum(histogram.numpy()))

  # Calculate percentile values.
  percentile_25, percentile_75 = np.percentile(accuracy_list, q=(25, 75))
  print('\nPercentiles...')
  print('25th Percentile : %f' % percentile_25)
  print('75th Percentile : %f' % percentile_75)

  overall_accuracy = (float(overall_correct_count) / float(overall_total_count))
  print('\nOverall classification success percentage: %d / %d (%f)' %
        (overall_correct_count, overall_total_count, overall_accuracy))


if __name__ == '__main__':
  app.run(main)
