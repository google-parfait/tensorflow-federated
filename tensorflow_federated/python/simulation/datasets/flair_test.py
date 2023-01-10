# Copyright 2023, The TensorFlow Federated Authors.
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
import json
import os.path

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import flair


METADATA_1 = {
    'user_id': 'user_a',
    'image_id': 0,
    'labels': ['label1', 'label2', 'label3'],
    'partition': 'train',
    'fine_grained_labels': ['fine1', 'fine2'],
}
METADATA_2 = {
    'user_id': 'user_b',
    'image_id': 1,
    'labels': ['label1', 'label3'],
    'partition': 'val',
    'fine_grained_labels': ['fine2'],
}
METADATA_3 = {
    'user_id': 'user_a',
    'image_id': 2,
    'labels': ['label4'],
    'partition': 'train',
    'fine_grained_labels': ['fine1'],
}
METADATA_4 = {
    'user_id': 'user_c',
    'image_id': 3,
    'labels': ['label3', 'label3'],
    'partition': 'test',
    'fine_grained_labels': ['fine1', 'fine4'],
}
METADATA_5 = {
    'user_id': 'user_d',
    'image_id': 4,
    'labels': ['label3', 'label4'],
    'partition': 'train',
    'fine_grained_labels': ['fine1', 'fine3'],
}
TEST_METADATA = [METADATA_1, METADATA_2, METADATA_3, METADATA_4, METADATA_5]
TEST_CLIENT_METADATA = collections.OrderedDict(
    user_a=[METADATA_1, METADATA_3],
    user_b=[METADATA_2],
    user_c=[METADATA_4],
    user_d=[METADATA_5],
)
TEST_USER_IDS = ['user_a', 'user_b', 'user_c', 'user_d']
TEST_IMAGE_IDS = [0, 1, 2, 3, 4]
TEST_LABELS_TO_INDEX = collections.OrderedDict(
    label1=0, label2=1, label3=2, label4=3
)
TEST_FINE_GRAINED_LABELS_TO_INDEX = collections.OrderedDict(
    fine1=0, fine2=1, fine3=2, fine4=3
)


def _generate_image_bytes(seed: int) -> bytes:
  random_seed = (0, seed)
  image = tf.random.stateless_uniform(
      shape=flair._IMAGE_SHAPE, seed=random_seed, minval=None, dtype=tf.uint32
  )
  image = tf.cast(image, tf.uint8)
  return tf.io.encode_jpeg(image).numpy()


def _write_test_images(
    image_ids: list[int], image_dir: str
) -> collections.OrderedDict[int, bytes]:
  test_images = collections.OrderedDict()
  for image_id in image_ids:
    image_bytes = _generate_image_bytes(image_id)
    test_images[image_id] = image_bytes
    image_file = os.path.join(image_dir, f'{image_id}.jpg')
    tf.io.write_file(image_file, image_bytes)
  return test_images


class FlairTest(tf.test.TestCase):

  def test_load_client_metadata(self):
    tmp_dir = self.create_tempdir()
    metadata_file = os.path.join(tmp_dir, 'client_metadata.json')
    with tf.io.gfile.GFile(metadata_file, 'w') as f:
      json.dump(TEST_METADATA, f)
    client_metadata, label_set, fine_grained_label_set = (
        flair._load_client_metadata(metadata_file)
    )
    self.assertDictEqual(TEST_CLIENT_METADATA, client_metadata)

    expected_label_set = set(['label1', 'label2', 'label3', 'label4'])
    self.assertEqual(expected_label_set, label_set)

    expected_fine_grained_label_set = set(['fine1', 'fine2', 'fine3', 'fine4'])
    self.assertEqual(expected_fine_grained_label_set, fine_grained_label_set)

  def test_create_example(self):
    image_bytes = b'these are image bytes'
    labels = [0, 3, 5]
    fine_grained_labels = [1, 2, 3, 4]

    example = flair._create_example(image_bytes, labels, fine_grained_labels)
    features = example.features.feature

    expected_keys = [
        flair._KEY_IMAGE_BYTES,
        flair._KEY_LABELS,
        flair._KEY_FINE_GRAINED_LABELS,
    ]
    self.assertCountEqual(list(features.keys()), expected_keys)

    image_feature = features[flair._KEY_IMAGE_BYTES]
    self.assertTrue(hasattr(image_feature, 'bytes_list'))
    self.assertEqual(image_feature.bytes_list.value[0], image_bytes)

    labels_feature = features[flair._KEY_LABELS]
    self.assertTrue(hasattr(labels_feature, 'int64_list'))
    self.assertEqual(labels_feature.int64_list.value, labels)

    fine_grained_labels_feature = features[flair._KEY_FINE_GRAINED_LABELS]
    self.assertTrue(hasattr(fine_grained_labels_feature, 'int64_list'))
    self.assertEqual(
        fine_grained_labels_feature.int64_list.value, fine_grained_labels
    )

  def test_create_label_index(self):
    labels = set(['c', 'd', 'e', 'f', 'a'])
    label_index = flair._create_label_index(labels)
    expected_label_index = collections.OrderedDict(a=0, c=1, d=2, e=3, f=4)
    self.assertEqual(expected_label_index, label_index)

  def test_write_tfrecords_writes_label_index_file(self):
    image_dir = self.create_tempdir()
    tfrecords_dir = self.create_tempdir()
    _write_test_images(TEST_IMAGE_IDS, image_dir)
    flair._write_tfrecords(
        client_metadata=TEST_CLIENT_METADATA,
        labels_to_index=TEST_LABELS_TO_INDEX,
        fine_grained_labels_to_index=TEST_FINE_GRAINED_LABELS_TO_INDEX,
        image_dir=image_dir,
        tfrecords_dir=tfrecords_dir,
    )
    expected_filename = os.path.join(tfrecords_dir, flair._LABEL_INDEX_FILE)
    self.assertTrue(tf.io.gfile.exists(expected_filename))

  def test_write_tfrecords_writes_expected_tfrecords_files(self):
    image_dir = self.create_tempdir()
    tfrecords_dir = self.create_tempdir()
    _write_test_images(TEST_IMAGE_IDS, image_dir)
    flair._write_tfrecords(
        client_metadata=TEST_CLIENT_METADATA,
        labels_to_index=TEST_LABELS_TO_INDEX,
        fine_grained_labels_to_index=TEST_FINE_GRAINED_LABELS_TO_INDEX,
        image_dir=image_dir,
        tfrecords_dir=tfrecords_dir,
    )
    actual_tfrecords_files = tf.io.gfile.glob(
        tfrecords_dir.full_path + '/*/' + '*.tfrecords'
    )
    expected_tfrecords_files = [
        os.path.join(tfrecords_dir, 'train', 'user_a.tfrecords'),
        os.path.join(tfrecords_dir, 'train', 'user_d.tfrecords'),
        os.path.join(tfrecords_dir, 'val', 'user_b.tfrecords'),
        os.path.join(tfrecords_dir, 'test', 'user_c.tfrecords'),
    ]
    self.assertCountEqual(expected_tfrecords_files, actual_tfrecords_files)

  def test_parse_example_undoes_create_example(self):
    image_bytes = _generate_image_bytes(seed=0)
    labels = [0, 3, 5]
    fine_grained_labels = [1, 2, 3, 4]
    example = flair._create_example(image_bytes, labels, fine_grained_labels)
    serialized_example = example.SerializeToString()

    parsed_example = flair._parse_example(serialized_example)

    expected_keys = [
        flair._KEY_IMAGE,
        flair._KEY_LABELS,
        flair._KEY_FINE_GRAINED_LABELS,
    ]
    self.assertCountEqual(list(parsed_example.keys()), expected_keys)
    expected_image = tf.io.decode_jpeg(image_bytes)
    self.assertAllEqual(parsed_example[flair._KEY_IMAGE], expected_image)
    expected_label = [1 if i in labels else 0 for i in range(flair._NUM_LABELS)]
    self.assertAllEqual(parsed_example[flair._KEY_LABELS], expected_label)
    expected_fine_grained_label = [
        1 if i in fine_grained_labels else 0
        for i in range(flair._NUM_FINEGRAINED_LABELS)
    ]
    self.assertAllEqual(
        parsed_example[flair._KEY_FINE_GRAINED_LABELS],
        expected_fine_grained_label,
    )

  def test_get_client_ids_to_files(self):
    tfrecords_dir = self.create_tempdir()
    tf.io.gfile.makedirs(
        os.path.join(tfrecords_dir, 'train', 'user_a.tfrecords')
    )
    tf.io.gfile.makedirs(os.path.join(tfrecords_dir, 'val', 'user_b.tfrecords'))
    tf.io.gfile.mkdir(os.path.join(tfrecords_dir, 'train', 'user_c.tfrecords'))
    train_client_ids_to_files = flair._get_client_ids_to_files(
        tfrecords_dir, 'train'
    )
    expected_train_client_ids_to_files = {
        'user_a': tfrecords_dir.full_path + '/train/user_a.tfrecords',
        'user_c': tfrecords_dir.full_path + '/train/user_c.tfrecords',
    }
    self.assertDictEqual(
        train_client_ids_to_files, expected_train_client_ids_to_files
    )

    val_client_ids_to_files = flair._get_client_ids_to_files(
        tfrecords_dir, 'val'
    )
    expected_val_client_ids_to_files = {
        'user_b': tfrecords_dir.full_path + '/val/user_b.tfrecords'
    }
    self.assertDictEqual(
        val_client_ids_to_files, expected_val_client_ids_to_files
    )


if __name__ == '__main__':
  tf.test.main()
