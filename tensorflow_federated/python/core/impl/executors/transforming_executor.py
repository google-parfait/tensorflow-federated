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
"""An executor that transforms computations prior to executing them."""

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.executors import executor_base


class TransformingExecutor(executor_base.Executor):
  """This executor transforms computations prior to executing them.

  This executor only performs transformations. All other aspects of execution
  are delegated to the underlying target executor.
  """

  def __init__(self, transformation_fn, target_executor):
    """Creates a transforming executor backed by a given target executor.

    Args:
      transformation_fn: A callable that accepts as single parameter that is an
        instance of `building_blocks.ComputationBuildingBlock`, and returns a
        result of the same type. This callable is used to transform any kind of
        computations before they are relayed to the target executor.
      target_executor: The target executor to delegate all the execution to.
    """
    py_typecheck.check_callable(transformation_fn)
    py_typecheck.check_type(target_executor, executor_base.Executor)
    self._transformation_fn = transformation_fn
    self._target_executor = target_executor

  # TODO(b/134543154): Add support for the case where embedded values might be
  # nested structures with computations in them (also to be transformed).

  async def create_value(self, value, type_spec=None):
    if isinstance(value, computation_impl.ComputationImpl):
      return await self.create_value(
          computation_impl.ComputationImpl.get_proto(value),
          type_utils.reconcile_value_with_type_spec(value, type_spec))
    elif isinstance(value, pb.Computation):
      return await self.create_value(
          building_blocks.ComputationBuildingBlock.from_proto(value), type_spec)
    elif isinstance(value, building_blocks.ComputationBuildingBlock):
      value = self._transformation_fn(value)
      py_typecheck.check_type(value, building_blocks.ComputationBuildingBlock)
      return await self._target_executor.create_value(value.proto, type_spec)
    else:
      return await self._target_executor.create_value(value, type_spec)

  async def create_call(self, comp, arg=None):
    return await self._target_executor.create_call(comp, arg)

  async def create_struct(self, elements):
    return await self._target_executor.create_struct(elements)

  async def create_selection(self, source, index=None, name=None):
    return await self._target_executor.create_selection(
        source, index=index, name=name)

  def close(self):
    self._target_executor.close()
