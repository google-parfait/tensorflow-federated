# Copyright 2018, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.execution_contexts import compiler_pipeline


class _FakeComputation(computation_base.Computation):

  def __init__(self, value: int):
    self.value = value

  def __call__(self):
    raise NotImplementedError()

  def __hash__(self):
    return hash(self.value)

  def type_signature(self):
    raise NotImplementedError()


class CompilerPipelineTest(absltest.TestCase):

  def test_compile_computation_with_identity(self):
    comp = _FakeComputation(5)
    pipeline = compiler_pipeline.CompilerPipeline(lambda x: x)

    compiled_comp = pipeline.compile(comp)
    self.assertEqual(compiled_comp.value, 5)

    # TODO: b/113123410 - Expand the test with more structural invariants.


if __name__ == '__main__':
  absltest.main()
