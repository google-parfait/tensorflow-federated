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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs.model_compression import encoders
from tensorflow_federated.python.tensorflow_libs.model_compression.core import test_utils


class EncoderComposerTest(tf.test.TestCase):

  def test_add_parent(self):
    encoder = (
        encoders.EncoderComposer(
            test_utils.ReduceMeanEncodingStage()).add_parent(
                test_utils.PlusOneEncodingStage(), 'values').add_parent(
                    test_utils.TimesTwoEncodingStage(), 'values').make())

    self.assertIsInstance(encoder, encoders.Encoder)
    self.assertIsInstance(encoder.stage._wrapped_stage,
                          test_utils.TimesTwoEncodingStage)
    self.assertIsInstance(encoder.children['values'], encoders.Encoder)
    self.assertIsInstance(encoder.children['values'].stage._wrapped_stage,
                          test_utils.PlusOneEncodingStage)
    self.assertIsInstance(encoder.children['values'].children['values'],
                          encoders.Encoder)
    self.assertIsInstance(
        encoder.children['values'].children['values'].stage._wrapped_stage,
        test_utils.ReduceMeanEncodingStage)

  def test_add_child(self):
    encoder = encoders.EncoderComposer(test_utils.TimesTwoEncodingStage())
    encoder.add_child(test_utils.PlusOneEncodingStage(), 'values').add_child(
        test_utils.ReduceMeanEncodingStage(), 'values')
    encoder = encoder.make()

    self.assertIsInstance(encoder, encoders.Encoder)
    self.assertIsInstance(encoder.stage._wrapped_stage,
                          test_utils.TimesTwoEncodingStage)
    self.assertIsInstance(encoder.children['values'], encoders.Encoder)
    self.assertIsInstance(encoder.children['values'].stage._wrapped_stage,
                          test_utils.PlusOneEncodingStage)
    self.assertIsInstance(encoder.children['values'].children['values'],
                          encoders.Encoder)
    self.assertIsInstance(
        encoder.children['values'].children['values'].stage._wrapped_stage,
        test_utils.ReduceMeanEncodingStage)

  def test_add_child_semantics(self):
    composer = encoders.EncoderComposer(test_utils.TimesTwoEncodingStage())
    composer.add_child(test_utils.PlusOneEncodingStage(), 'values')
    encoder_1 = composer.make()
    encoder_2 = encoders.EncoderComposer(
        test_utils.TimesTwoEncodingStage()).add_child(
            test_utils.PlusOneEncodingStage(), 'values').make()

    # Assert that these produce different trees. The add_child method returns
    # the newly created node, and thus the make creates only the child node.
    self.assertNotEqual(encoder_1.children.keys(), encoder_2.children.keys())

  def test_constructor_raises(self):
    with self.assertRaises(TypeError):
      encoders.EncoderComposer('not an encoding stage')

  def test_add_child_parent_bad_key_raises(self):
    encoder = encoders.EncoderComposer(test_utils.TimesTwoEncodingStage())
    with self.assertRaises(KeyError):
      encoder.add_child(test_utils.PlusOneEncodingStage(), '___bad_key')
    with self.assertRaises(KeyError):
      encoder.add_parent(test_utils.PlusOneEncodingStage(), '___bad_key')

  def test_add_child_repeat_key_raises(self):
    encoder = encoders.EncoderComposer(test_utils.TimesTwoEncodingStage())
    encoder.add_child(test_utils.PlusOneEncodingStage(), 'values')
    with self.assertRaises(KeyError):
      encoder.add_child(test_utils.PlusOneEncodingStage(), 'values')


if __name__ == '__main__':
  tf.test.main()
