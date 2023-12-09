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

from absl.testing import absltest
import numpy as np

from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.learning.templates import hparams_base


class HparamsBaseTest(absltest.TestCase):

  def test_get_hparams_with_compatible_state_type_does_not_raise(self):
    state_type = computation_types.TensorType(np.int32)

    @tensorflow_computation.tf_computation(np.int32)
    def get_hparams_fn(state):
      return collections.OrderedDict(a=state)

    hparams_base.type_check_get_hparams_fn(get_hparams_fn, state_type)

  def test_get_hparams_with_incompatible_state_type(self):
    state_type = computation_types.TensorType(np.int32)

    @tensorflow_computation.tf_computation(np.float32)
    def get_hparams_fn(state):
      return collections.OrderedDict(a=state)

    with self.assertRaises(hparams_base.GetHparamsTypeError):
      hparams_base.type_check_get_hparams_fn(get_hparams_fn, state_type)

  def test_set_hparams_fn_with_one_input_arg_raises(self):
    state_type = computation_types.TensorType(np.int32)

    @tensorflow_computation.tf_computation(np.int32)
    def set_hparams_fn(state):
      return state

    with self.assertRaises(hparams_base.SetHparamsTypeError):
      hparams_base.type_check_set_hparams_fn(set_hparams_fn, state_type)

  def test_set_hparams_fn_with_three_input_args_raises(self):
    state_type = computation_types.TensorType(np.int32)

    @tensorflow_computation.tf_computation(np.int32, np.int32, np.int32)
    def set_hparams_fn(state, x, y):
      del x
      del y
      return state

    with self.assertRaises(hparams_base.SetHparamsTypeError):
      hparams_base.type_check_set_hparams_fn(set_hparams_fn, state_type)

  def test_set_hparams_fn_with_compatible_state_type_does_not_raise(self):
    state_type = computation_types.TensorType(np.int32)
    hparams_type = computation_types.to_type(
        collections.OrderedDict(a=np.int32)
    )

    @tensorflow_computation.tf_computation(np.int32, hparams_type)
    def set_hparams_fn(state, hparams):
      del state
      return hparams['a']

    hparams_base.type_check_set_hparams_fn(set_hparams_fn, state_type)

  def test_set_hparams_fn_with_incompatible_input_state_type_raises(self):
    state_type = computation_types.TensorType(np.int32)
    hparams_type = computation_types.to_type(
        collections.OrderedDict(a=np.int32)
    )

    @tensorflow_computation.tf_computation(np.float32, hparams_type)
    def set_hparams_fn(state, hparams):
      del state
      return hparams['a']

    with self.assertRaises(hparams_base.SetHparamsTypeError):
      hparams_base.type_check_set_hparams_fn(set_hparams_fn, state_type)

  def test_set_hparams_fn_with_incompatible_outputput_state_type_raises(self):
    state_type = computation_types.TensorType(np.int32)
    hparams_type = computation_types.to_type(
        collections.OrderedDict(a=np.float32)
    )

    @tensorflow_computation.tf_computation(np.int32, hparams_type)
    def set_hparams_fn(state, hparams):
      del state
      return hparams['a']

    with self.assertRaises(hparams_base.SetHparamsTypeError):
      hparams_base.type_check_set_hparams_fn(set_hparams_fn, state_type)

  def test_default_get_hparams_returns_empty_dict(self):
    state_type = computation_types.TensorType(np.int32)
    get_hparams_fn = hparams_base.build_basic_hparams_getter(state_type)
    expected_hparams_type = computation_types.to_type(collections.OrderedDict())
    expected_function_type = computation_types.FunctionType(
        parameter=state_type, result=expected_hparams_type
    )
    type_test_utils.assert_types_equivalent(
        get_hparams_fn.type_signature, expected_function_type
    )

  def test_default_set_hparams_returns_state_of_matching_type(self):
    state_type = computation_types.TensorType(np.int32)
    hparams_type = computation_types.to_type(
        collections.OrderedDict(a=np.float32)
    )
    set_hparams_fn = hparams_base.build_basic_hparams_setter(
        state_type, hparams_type
    )
    expected_function_type = computation_types.FunctionType(
        parameter=computation_types.StructType(
            [('state', state_type), ('hparams', hparams_type)]
        ),
        result=state_type,
    )
    type_test_utils.assert_types_equivalent(
        set_hparams_fn.type_signature, expected_function_type
    )


if __name__ == '__main__':
  absltest.main()
