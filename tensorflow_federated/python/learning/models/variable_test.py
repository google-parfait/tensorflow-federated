# Copyright 2026, The TensorFlow Federated Authors.
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

from collections.abc import Sequence
import warnings

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.learning.models import variable


class _TestModel(variable.VariableModel):

  @property
  def trainable_variables(self) -> Sequence[tf.Variable]:
    return []

  @property
  def non_trainable_variables(self) -> Sequence[tf.Variable]:
    return []

  @property
  def local_variables(self) -> Sequence[tf.Variable]:
    return []

  @property
  def input_spec(self):
    return (
        tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
    )

  def forward_pass(self, batch_input, training=True) -> variable.BatchOutput:
    del batch_input, training  # Unused.
    return variable.BatchOutput(loss=tf.constant(0.0))

  def predict_on_batch(self, batch_input, training=True):
    del batch_input, training  # Unused.
    return tf.constant(0.0)

  def report_local_unfinalized_metrics(self):
    return {}

  def metric_finalizers(self):
    return {}

  def reset_metrics(self) -> None:
    pass


class VariableModelTest(absltest.TestCase):

  def test_deprecation_warning_on_init(self):
    with warnings.catch_warnings(record=True) as warning_list:
      warnings.simplefilter(action='always', category=DeprecationWarning)
      _TestModel()
      deprecation_warnings = [
          w
          for w in warning_list
          if issubclass(w.category, DeprecationWarning)
          and 'tff.learning.models.VariableModel is deprecated'
          in str(w.message)
      ]
      self.assertNotEmpty(deprecation_warnings)
      self.assertIn('FunctionalModel', str(deprecation_warnings[0].message))

  def test_cannot_instantiate_abstract_class(self):
    with self.assertRaises(TypeError):
      variable.VariableModel()  # pyrefly: ignore[abstract-class-instantiation]

  def test_deprecated_attribute(self):
    self.assertTrue(hasattr(variable.VariableModel, '__deprecated__'))
    self.assertIn(
        'tff.learning.models.VariableModel is deprecated',
        variable.VariableModel.__deprecated__,  # pyrefly: ignore[missing-attribute]
    )


if __name__ == '__main__':
  absltest.main()
