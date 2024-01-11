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
"""Typing information for metrics finalizers."""

import collections
from collections.abc import Callable
from typing import Any, Optional, Protocol

from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.types import computation_types


MetricFinalizersType = collections.OrderedDict[str, Callable[[Any], Any]]
MetricsState = collections.OrderedDict[str, Any]
FunctionalMetricFinalizersType = Callable[[MetricsState], MetricsState]


# TODO: b/319261270 - delete the local_unfinalized_metrics_type entirely.
class MetricsAggregatorType(Protocol):

  def __call__(
      self,
      metric_finalizers: MetricFinalizersType,
      local_unfinalized_metrics_type: Optional[
          computation_types.StructWithPythonType
      ] = None,
  ) -> computation_base.Computation:
    ...
