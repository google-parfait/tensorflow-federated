# Copyright 2021, The TensorFlow Federated Authors.
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

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import inaturalist
from tensorflow_federated.python.simulation.datasets import vision_datasets_utils

EXPECTED_ELEMENT_TYPE = collections.OrderedDict([
    ('image/decoded', tf.TensorSpec(shape=(128, 128, 3), dtype=tf.uint8)),
    ('class', tf.TensorSpec(shape=(), dtype=tf.int64)),
])


class INatualistTest(tf.test.TestCase):

  def test_get_synthetic(self):
    client_data = inaturalist.get_synthetic()
    self.assertLen(client_data.client_ids, 1)
    self.assertEqual(client_data.element_type_structure, EXPECTED_ELEMENT_TYPE)

    data = client_data.create_tf_dataset_for_client(client_data.client_ids[0])
    images = [x['image/decoded'].numpy() for x in data]
    classes = [x['class'].numpy() for x in data]
    self.assertLen(images, 3)
    self.assertCountEqual(classes, [0, 1, 2])

  def test_create_dataset_from_mapping(self):
    tmp_dir = self.create_tempdir(name='images')
    images = {'image_1': b'somebytes', 'image_2': b'someotherbytes'}
    image_map = {}
    for image_name, content in images.items():
      tmp_file = tmp_dir.create_file(
          file_path=image_name + '.jpg', content=content
      )
      image_map[image_name] = tmp_file.full_path

    mapping = [
        {'image_id': 'image_1', 'class': '0'},
        {'image_id': 'image_2', 'class': '12'},
    ]

    expected_features = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    vision_datasets_utils.KEY_IMAGE_BYTES: tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b'somebytes'])
                    ),
                    vision_datasets_utils.KEY_CLASS: tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[0])
                    ),
                }
            )
        ),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    vision_datasets_utils.KEY_IMAGE_BYTES: tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b'someotherbytes'])
                    ),
                    vision_datasets_utils.KEY_CLASS: tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[12])
                    ),
                }
            )
        ),
    ]

    features = inaturalist._create_dataset_with_mapping(image_map, mapping)
    self.assertSequenceEqual(features, expected_features)


if __name__ == '__main__':
  tf.test.main()
