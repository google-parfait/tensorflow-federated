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

from absl.testing import parameterized
import logging
import numpy as np
import tempfile
import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.utils import execution_tracing

mock_repr = "mock!"


class Mock:

  def __repr__(self):
    return mock_repr

  @tracing.trace
  def create_value(self, x):
    return x

  @tracing.trace
  def _compute_intrinsic_xxyyzz(self, x):
    arg = self.create_value(x)
    res = self._weird_other_method(arg)
    return res

  @tracing.trace
  def _weird_other_method(self, x):

    @tracing.trace
    def _weird_local_trace(y):
      return y

    return _weird_local_trace(x)


expected_mock_format = "{} {}".format(mock_repr, Mock)
modified_format_strategy = lambda x: "hello from {x}".format(x=x)


class ExecutionTracingProviderTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('default', execution_tracing.DEFAULT_FORMAT_STRATEGY,
       expected_mock_format),
      ('modified', modified_format_strategy, "hello from mock!"),
  )
  def test_default_formatting_strategy(self, strategy, expected):
    obj = Mock()
    tracer = execution_tracing.ExecutionTracingProvider(
        default_format_strategy=strategy)
    self.assertEqual(tracer._format_object(obj), expected)

  @parameterized.named_parameters(
      ('values', execution_tracing.TraceFilterMode.VALUES, [True, False, False
                                                           ]),
      ('intrinsics', execution_tracing.TraceFilterMode.INTRINSICS,
       [True, True, False]),
      ('all', execution_tracing.TraceFilterMode.ALL, [True] * 3))
  def test_trace_filter_mode(self, mode, expected):
    obj = Mock()
    testable_methods = [
        'create_value',
        '_compute_intrinsic_xxyyzz',
        '_weird_other_method',
    ]
    tracer = execution_tracing.ExecutionTracingProvider(trace_filter_mode=mode)
    regex = tracer._is_traceable_regex
    override = regex is None
    result = [
        tracer._is_traceable(m, regex, override) for m in testable_methods
    ]
    self.assertAllEqual(result, expected)


if __name__ == '__main__':
  test.main()
