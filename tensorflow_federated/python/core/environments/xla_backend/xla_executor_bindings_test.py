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

from absl.testing import absltest

from tensorflow_federated.python.core.environments.xla_backend import xla_executor_bindings


class XlaExecutorBindingsTest(absltest.TestCase):

  def test_create(self):
    try:
      xla_executor_bindings.create_xla_executor()
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'Exception: {e}')

  def test_materialize_on_unkown_fails(self):
    executor = xla_executor_bindings.create_xla_executor()
    with self.assertRaisesRegex(Exception, 'NOT_FOUND'):
      executor.materialize(0)


if __name__ == '__main__':
  absltest.main()
