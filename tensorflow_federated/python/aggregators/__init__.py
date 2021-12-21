# Copyright 2020, The TensorFlow Federated Authors.
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
"""Libraries for constructing federated aggregation.

For example uses of the symbols in this module, see
[Tuning recommended aggregations for learning](
https://www.tensorflow.org/federated/tutorials/tuning_recommended_aggregators)
tutorial, and for details of the design and how to implement new aggregations,
see [Implementing Custom Aggregations](
https://www.tensorflow.org/federated/tutorials/custom_aggregators) tutorial.
"""

from tensorflow_federated.python.aggregators.concat import concat_factory
from tensorflow_federated.python.aggregators.differential_privacy import DifferentiallyPrivateFactory
from tensorflow_federated.python.aggregators.encoded import EncodedSumFactory
from tensorflow_federated.python.aggregators.factory import AggregationFactory
from tensorflow_federated.python.aggregators.factory import UnweightedAggregationFactory
from tensorflow_federated.python.aggregators.factory import WeightedAggregationFactory
from tensorflow_federated.python.aggregators.mean import MeanFactory
from tensorflow_federated.python.aggregators.mean import UnweightedMeanFactory
from tensorflow_federated.python.aggregators.measurements import add_measurements
from tensorflow_federated.python.aggregators.primitives import federated_max
from tensorflow_federated.python.aggregators.primitives import federated_min
from tensorflow_federated.python.aggregators.primitives import federated_sample
from tensorflow_federated.python.aggregators.primitives import secure_quantized_sum
from tensorflow_federated.python.aggregators.quantile_estimation import PrivateQuantileEstimationProcess
from tensorflow_federated.python.aggregators.robust import clipping_factory
from tensorflow_federated.python.aggregators.robust import zeroing_factory
from tensorflow_federated.python.aggregators.rotation import DiscreteFourierTransformFactory
from tensorflow_federated.python.aggregators.rotation import HadamardTransformFactory
from tensorflow_federated.python.aggregators.sampling import UnweightedReservoirSamplingFactory
from tensorflow_federated.python.aggregators.secure import SecureSumFactory
from tensorflow_federated.python.aggregators.sum_factory import SumFactory
