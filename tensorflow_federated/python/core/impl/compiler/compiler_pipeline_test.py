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

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.compiler import compiler_pipeline


class CompilerPipelineTest(test_case.TestCase):

  def test_compile_computation_with_identity(self):

    class BogusComputation(computation_base.Computation):

      def __init__(self, v: int):
        self.v = v

      def __call__(self):
        raise NotImplementedError()

      def __hash__(self):
        return hash(self.v)

      def type_signature(self):
        raise NotImplementedError()

    id_pipeline = compiler_pipeline.CompilerPipeline(lambda x: x)
    compiled_bogus = id_pipeline.compile(BogusComputation(5))
    self.assertEqual(compiled_bogus.v, 5)

    # TODO(b/113123410): Expand the test with more structural invariants.


if __name__ == '__main__':
  test_case.main()
