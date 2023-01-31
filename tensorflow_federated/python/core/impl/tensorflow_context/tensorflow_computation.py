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
"""Definition of a tensorflow computation."""

from collections.abc import Generator, Mapping, Sequence
from typing import Any, Callable, Optional, Union

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import computation_wrapper
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization


def _tf_wrapper_fn(parameter_type, name, **kwargs):
  """Wrapper function to plug Tensorflow logic into the TFF framework."""
  del name  # Unused.
  if 'layout_map' in kwargs:
    layout_map = kwargs['layout_map']
  else:
    layout_map = None
  if not type_analysis.is_tensorflow_compatible_type(parameter_type):
    raise TypeError(
        '`tf_computation`s can accept only parameter types with '
        'constituents `SequenceType`, `StructType` '
        'and `TensorType`; you have attempted to create one '
        'with the type {}.'.format(parameter_type)
    )
  ctx_stack = context_stack_impl.context_stack
  tf_serializer = tensorflow_serialization.tf_computation_serializer(
      parameter_type, ctx_stack, layout_map
  )
  arg = next(tf_serializer)
  try:
    result = yield arg
  except Exception as e:  # pylint: disable=broad-except
    tf_serializer.throw(e)
  comp_pb, extra_type_spec = tf_serializer.send(result)
  tf_serializer.close()
  yield computation_impl.ConcreteComputation(
      comp_pb, ctx_stack, extra_type_spec
  )


tf_computation = computation_wrapper.ComputationWrapper(
    computation_wrapper.PythonTracingStrategy(_tf_wrapper_fn)
)
tf_computation.__doc__ = """Decorates/wraps Python functions and defuns as TFF TensorFlow computations.

  This symbol can be used as either a decorator or a wrapper applied to a
  function given to it as an argument. The supported patterns and examples of
  usage are as follows:

  1. Convert an existing function inline into a TFF computation. This is the
     simplest mode of usage, and how one can embed existing non-TFF code for
     use with the TFF framework. In this mode, one invokes
     `tff.tf_computation` with a pair of arguments, the first being a
     function/defun that contains the logic, and the second being the TFF type
     of the parameter:

     ```python
     foo = tff.tf_computation(lambda x: x > 10, tf.int32)
     ```

     After executing the above code snippet, `foo` becomes an instance of the
     abstract base class `Computation`. Like all computations, it has the
     `type_signature` property:

     ```python
     str(foo.type_signature) == '(int32 -> bool)'
     ```

     The function passed as a parameter doesn't have to be a lambda, it can
     also be an existing Python function or a defun. Here's how to construct
     a computation from the standard TensorFlow operator `tf.add`:

     ```python
     foo = tff.tf_computation(tf.add, (tf.int32, tf.int32))
     ```

     The resulting type signature is as expected:

     ```python
     str(foo.type_signature) == '(<int32,int32> -> int32)'
     ```

     If one intends to create a computation that doesn't accept any arguments,
     the type argument is simply omitted. The function must be a no-argument
     function as well:

     ```python
     foo = tf_computation(lambda: tf.constant(10))
     ```

  2. Decorate a Python function or a TensorFlow defun with a TFF type to wrap
     it as a TFF computation. The only difference between this mode of usage
     and the one mentioned above is that instead of passing the function/defun
     as an argument, `tff.tf_computation` along with the optional type specifier
     is written above the function/defun's body.

     Here's an example of a computation that accepts a parameter:

     ```python
     @tff.tf_computation(tf.int32)
     def foo(x):
       return x > 10
     ```

     One can think of this mode of usage as merely a syntactic sugar for the
     example already given earlier:

     ```python
     foo = tff.tf_computation(lambda x: x > 10, tf.int32)
     ```

     Here's an example of a no-parameter computation:

     ```python
     @tff.tf_computation
     def foo():
       return tf.constant(10)
     ```

     Again, this is merely syntactic sugar for the example given earlier:

     ```python
     foo = tff.tf_computation(lambda: tf.constant(10))
     ```

     If the Python function has multiple decorators, `tff.tf_computation` should
     be the outermost one (the one that appears first in the sequence).

  3. Create a polymorphic callable to be instantiated based on arguments,
     similarly to TensorFlow defuns that have been defined without an input
     signature.

     This mode of usage is symmetric to those above. One simply omits the type
     specifier, and applies `tff.tf_computation` as a decorator or wrapper to a
     function/defun that does expect parameters.

     Here's an example of wrapping a lambda as a polymorphic callable:

     ```python
     foo = tff.tf_computation(lambda x, y: x > y)
     ```

     The resulting `foo` can be used in the same ways as if it were had the
     type been declared; the corresponding computation is simply created on
     demand, in the same way as how polymorphic TensorFlow defuns create and
     cache concrete function definitions for each combination of argument
     types.

     ```python
     ...foo(1, 2)...
     ...foo(0.5, 0.3)...
     ```

     Here's an example of creating a polymorphic callable via decorator:

     ```python
     @tff.tf_computation
     def foo(x, y):
       return x > y
     ```

     The syntax is symmetric to all examples already shown.

  Args:
    *args: Either a function/defun, or TFF type spec, or both (function first),
      or neither, as documented in the 3 patterns and examples of usage above.

  Returns:
    If invoked with a function as an argument, returns an instance of a TFF
    computation constructed based on this function. If called without one, as
    in the typical decorator style of usage, returns a callable that expects
    to be called with the function definition supplied as a parameter; see the
    patterns and examples of usage above.
  """


class CapturedVariableError(Exception):
  """Error raised when TFF tracing encountered variables captured from an outer scope."""


def _no_lifting_creator(next_creator_fn, **kwargs):
  """Variable creator that disables variable lifitng."""
  # We require disabling the lifting of variables outside the `tf.function` in
  # TFF. TFF follows a functional programming model, where functions are "pure"
  # in that they do not (can not) rely on external state. Because of this only,
  # the FunctionDef will be serialized and sent for execution. So, we disable
  # `tf.variable` lifting at the outermost computation wrapper scope.
  kwargs['experimental_enable_variable_lifting'] = False
  return next_creator_fn(**kwargs)


# The ArgDef protocol buffer message type isn't exposed in the `tensorflow`
# package, this is a less than ideal workaround.
ArgDef = type(
    tf.compat.v1.GraphDef().library.function.add().signature.input_arg.add()
)
TensorStructureType = Union[
    computation_types.TensorType,
    computation_types.StructType,
    computation_types.SequenceType,
]


def _extract_bindings(
    type_spec: Optional[TensorStructureType],
    arg_defs: Sequence[ArgDef],
) -> pb.TensorFlowFunction.Binding:
  """Constructs a Binding given a type and function arguments."""

  def extract(
      type_spec: Optional[TensorStructureType],
      function_args: Sequence[ArgDef],
      arg_index=0,
  ) -> tuple[Optional[pb.TensorFlowFunction.Binding], int]:
    if type_spec is None:
      return None, 0
    elif type_spec.is_tensor():
      arg_def = function_args[arg_index]
      if arg_def.type != type_spec.dtype.as_datatype_enum:
        raise TypeError(
            f'Argument at position {arg_index} had binding type '
            f'{tf.dtypes.as_dtype(arg_def.type)}, but '
            f'type signature expected type {type_spec.dtype}.'
        )
      return (
          pb.TensorFlowFunction.Binding(
              tensor=pb.TensorFlowFunction.TensorBinding(arg_name=arg_def.name)
          ),
          1,
      )
    elif type_spec.is_sequence():
      arg_def = function_args[arg_index]
      if arg_def.type != tf.variant:
        raise TypeError(
            f'Argument at position {arg_index} had binding type '
            f'{tf.dtypes.as_dtype(arg_def.type)}, but '
            'sequence type signature expected tf.variant.'
        )
      return (
          pb.TensorFlowFunction.Binding(
              sequence=pb.TensorFlowFunction.SequenceBinding(
                  arg_name=arg_def.name
              )
          ),
          1,
      )

    elif type_spec.is_struct():
      # tf.function tracing uses tf.nest.flatten to destructure input arguments.
      # The `GetValueIterator` method in tensorflow/python/util/util.cc sorts
      # the keys in Mapping types (note: namedtuple is not a mapping), though
      # TFF does _NOT_ sort when constructing StructType. Here we need to change
      # the method of iteration depending on the container type.
      # TODO(b/257258116): Refactor to a generic component that can be used
      # across TFF when needing to reconcile TFF Struct traversal with
      # tf.nest.flatten.
      def field_iterator(
          struct_type: computation_types.StructType,
      ) -> Generator[computation_types.Type, None, None]:
        if struct_type.is_struct_with_python() and issubclass(
            computation_types.StructWithPythonType.get_container_type(
                struct_type
            ),
            Mapping,
        ):
          for field_name in sorted(structure.name_list(struct_type)):
            yield struct_type[field_name]
        else:
          for field in struct_type:
            yield field

      # Note: this code ignores the names of the elements, as all bindings are
      # only structural; the execution runtimes do not utilize names.
      args_consumed = 0
      elements = []
      for field in field_iterator(type_spec):
        element, args_used = extract(
            field, function_args, arg_index + args_consumed
        )
        elements.append(element)
        args_consumed += args_used
      return (
          pb.TensorFlowFunction.Binding(
              structure=pb.TensorFlowFunction.StructBinding(element=elements)
          ),
          args_consumed,
      )
    else:
      raise TypeError(
          'Cannot build bindings for type signature, must be '
          'TensorType, SequenceType, or StructType. '
          f'Got: {type_spec!r}'
      )

  try:
    binding, args_consumed = extract(type_spec, arg_defs)
  except TypeError as e:
    raise TypeError(
        f'Failed to creating bindings for type {type_spec!r} with '
        f'arguments {arg_defs}'
    ) from e
  if args_consumed != len(arg_defs):
    raise ValueError(
        'Number of args is not compatible with type '
        f'{type_spec!r}. Expected {args_consumed} args to bind, '
        f'but got {len(arg_defs)} args.'
    )
  return binding


class _TensorFlowFunctionTracingStrategy:
  """A tracing strategy that relies on `tf.function` tracing."""

  def __call__(
      self,
      fn_to_wrap: Callable[..., Any],
      fn_name: Optional[str],
      parameter_type: Optional[computation_types.Type],
      unpack: Optional[bool],
      **kwargs,
  ) -> computation_impl.ConcreteComputation:
    if not type_analysis.is_tensorflow_compatible_type(parameter_type):
      raise TypeError(
          '`tf_computation`s can accept only parameter types with '
          'constituents `SequenceType`, `StructType` '
          'and `TensorType`; you have attempted to create one '
          'with the type {}.'.format(parameter_type)
      )
    ctx_stack = context_stack_impl.context_stack
    unpack_arguments_fn = function_utils.create_argument_unpacking_fn(
        fn_to_wrap, parameter_type, unpack=unpack
    )

    # Disabling variable lifting does not work on XLA devices so it is required
    # that the `tf.function` tracing _NOT_ jit compile.
    # TODO(b/210930091): remove explicit jit_compile=False after local
    # (non-lifted) variables are supported in TF2XLA Bridge.
    @tf.function(jit_compile=False)
    def fn_without_variable_lifting(packed_args=None):
      # TFF's `Struct` type is not compatible with `tf.nest`, which is needed
      # by the `tf.function` APIs. However, `unpack_arguments_fn` expects the
      # `Struct` type. So `packed_args` will come in as a Python container,
      # and we wrap it in a `Struct` type here to be compatible.
      if (
          packed_args is not None
          and parameter_type is not None
          and parameter_type.is_struct()
      ):
        packed_args = structure.from_container(packed_args, recursive=False)
      args, kwargs = unpack_arguments_fn(packed_args)
      with tf.variable_creator_scope(_no_lifting_creator):
        return fn_to_wrap(*args, **kwargs)

    TensorFlowSpec = Union[
        tf.data.DatasetSpec,
        tf.TensorSpec,
        tf.SparseTensorSpec,
        tf.RaggedTensorSpec,
    ]

    def _tf_spec_from_tff_type(
        type_spec: computation_types.Type,
    ) -> Union[
        TensorFlowSpec, Sequence[TensorFlowSpec], Mapping[str, TensorFlowSpec]
    ]:
      if type_spec.is_tensor():
        return tf.TensorSpec(shape=type_spec.shape, dtype=type_spec.dtype)
      elif type_spec.is_sequence():
        return tf.data.DatasetSpec(_tf_spec_from_tff_type(type_spec.element))
      elif type_spec.is_struct():
        container_type = type_spec.python_container
        if container_type is tf.SparseTensor:
          [rank] = type_spec['dense_shape'].shape
          return tf.SparseTensorSpec(
              shape=[None] * rank, dtype=type_spec['values'].dtype
          )
        elif container_type is tf.RaggedTensor:
          flat_values_type_spec = type_spec['flat_values']
          flat_values_spec = tf.TensorSpec(
              shape=flat_values_type_spec.shape,
              dtype=flat_values_type_spec.dtype,
          )
          nested_row_splits_type_spec = type_spec['nested_row_splits']
          row_splits_dtype = nested_row_splits_type_spec[0].dtype
          return tf.RaggedTensorSpec(
              dtype=flat_values_spec.dtype,
              ragged_rank=len(nested_row_splits_type_spec),
              row_splits_dtype=row_splits_dtype,
              flat_values_spec=flat_values_spec,
          )
        else:
          structure_of_type_specs = structure.Struct(
              [
                  (name, _tf_spec_from_tff_type(child_type))
                  for name, child_type in structure.iter_elements(type_spec)
              ]
          )
          return type_conversions.type_to_py_container(
              structure_of_type_specs, type_spec
          )
      else:
        raise TypeError(
            f'Cannot trace functions with arguments of type: {type_spec!r}'
        )

    if parameter_type is None:
      concrete_fn = fn_without_variable_lifting.get_concrete_function()
    else:
      tensorflow_input_spec = _tf_spec_from_tff_type(parameter_type)
      concrete_fn = fn_without_variable_lifting.get_concrete_function(
          tensorflow_input_spec
      )

    if concrete_fn.variables:
      raise CapturedVariableError(
          'Traced function references variables from an outer scope, '
          f'this is not allowed. Found variables: {concrete_fn.variables}'
      )

    if concrete_fn.structured_outputs is None:
      raise computation_wrapper.ComputationReturnedNoneError(fn_to_wrap)

    def _symbolic_tensors_to_tf_specs(tensors):
      if isinstance(tensors, tf.Tensor):
        return tf.TensorSpec.from_tensor(tensors)
      elif isinstance(tensors, tf.SparseTensor):
        return tf.SparseTensorSpec.from_value(tensors)
      elif isinstance(tensors, tf.RaggedTensor):
        return tf.RaggedTensorSpec.from_value(tensors)
      elif isinstance(tensors, tf.data.Dataset):
        return tf.data.DatasetSpec(element_spec=tensors.element_spec)
      else:
        raise TypeError(
            'Function output must be a `tf.Tensor`, `tf.SparseTensor`, '
            '`tf.RaggedTensor`, or a `tf.data.Dataset`. '
            f'Unknown tensor value type: {type(tensors)!r}.'
        )

    type_signature = computation_types.FunctionType(
        parameter=parameter_type,
        result=computation_types.to_type(
            tf.nest.map_structure(
                _symbolic_tensors_to_tf_specs, concrete_fn.structured_outputs
            )
        ),
    )

    parameter_binding = _extract_bindings(
        type_signature.parameter, concrete_fn.function_def.signature.input_arg
    )
    result_binding = _extract_bindings(
        type_signature.result, concrete_fn.function_def.signature.output_arg
    )
    if 'layout_map' in kwargs:
      layout_map = kwargs['layout_map']
    else:
      layout_map = None
    comp_pb = pb.Computation(
        type=type_serialization.serialize_type(type_signature),
        tensorflow_function=pb.TensorFlowFunction(
            function_def=serialization_utils.pack_function_def(
                concrete_fn.function_def
            ),
            parameter=parameter_binding,
            result=result_binding,
            layout_map=pb.TensorFlowFunction.LayoutMap(
                name_to_sharding_spec=layout_map
            ),
        ),
    )
    return computation_impl.ConcreteComputation(
        comp_pb, ctx_stack, annotated_type=type_signature
    )


experimental_tf_fn_computation = computation_wrapper.ComputationWrapper(
    _TensorFlowFunctionTracingStrategy()
)
experimental_tf_fn_computation.__doc__ = """Decorates/wraps functions as TFF TensorFlow computations.

  This symbol can be used as either a decorator or a wrapper applied to a
  function given to it as an argument.

  IMPORTANT: This symbol will decorate the function argument with `tf.function`
  consequently apply TensorFlow's Auto-Control-Dependencies tracing to the logic
  (eager-mode-like semantics, _not_ graph-mode-like semantics).

  The supported patterns and examples of usage are as follows:

  1. Convert an existing function inline into a TFF computation. This is the
     simplest mode of usage, and how one can embed existing non-TFF code for
     use with the TFF framework. In this mode, one invokes
     `tff.experimental_tf_fn_computation` with a pair of arguments, the first
     being a function that contains the logic, and the second being the TFF type
     of the parameter:

     ```python
     foo = tff.experimental_tf_fn_computation(lambda x: x > 10, tf.int32)
     ```

     After executing the above code snippet, `foo` becomes an instance of the
     abstract base class `Computation`. Like all computations, it has the
     `type_signature` property:

     ```python
     str(foo.type_signature)
     >>> '(int32 -> bool)'
     ```

     The function passed as a parameter doesn't have to be a lambda, it can
     be any Python callable. One notable exception is that TFF does not handle
     arguments with default values.

     If one intends to create a computation that doesn't accept any arguments,
     the type argument is simply omitted. The function must be a no-argument
     function as well:

     ```python
     foo = tff.experimental_tf_fn_computation(lambda: tf.constant(10))
     str(foo.type_signature)
     >>> '( -> tf.int32)'
     ```

  2. Decorate a callable with a TFF type to wrap it as a TFF computation. The
     only difference between this mode of usage and the one mentioned above is
     that instead of passing the callable as an argument,
     `tff.experimetnal_tf_func_computation` along with the optional type
     specifier is written above the callable's body.

     Here's an example of a computation that accepts a parameter:

     ```python
     @tff.experimental_tf_fn_computation(tf.int32)
     def foo(x):
       return x > 10
     ```

     One can think of this mode of usage as merely a syntactic sugar for the
     example already given earlier:

     ```python
     foo = tff.tf_computation(lambda x: x > 10, tf.int32)
     ```

     Here's an example of a no-parameter computation:

     ```python
     @tff.tf_computation
     def foo():
       return tf.constant(10)
     ```

     Again, this is merely syntactic sugar for the example given earlier:

     ```python
     foo = tff.tf_computation(lambda: tf.constant(10))
     ```

     If the Python callable has multiple decorators, `tff.tf_computation` should
     be the outermost decorator (the one that appears first, or at the top).

  3. Create a polymorphic callable to be instantiated based on arguments,
     similarly to `tf.function`s that have been defined without an input
     signature.

     This mode of usage is symmetric to those above. One simply omits the type
     specifier, and applies `tff.experimental_tf_fn_computation` as a
     decorator or wrapper to a function/defun that does expect parameters.

     Here's an example of wrapping a lambda as a polymorphic callable:

     ```python
     foo = tff.tf_computation(lambda x, y: x > y)
     ```

     The resulting `foo` can be used in the same ways as if it were had the
     type been declared; the corresponding computation is simply created on
     demand, in the same way as how polymorphic `tf.function`s create and
     cache concrete function definitions for each combination of argument
     types.

     ```python
     ...foo(1, 2)...
     ...foo(0.5, 0.3)...
     ```

     Here's an example of creating a polymorphic callable via decorator:

     ```python
     @tff.tf_computation
     def foo(x, y):
       return x > y
     ```

     The syntax is symmetric to all examples already shown.

  Args:
    *args: Either a python callable, or TFF type spec, or both (with callable
    first), or neither, as documented in the 3 patterns and examples of usage
    above.

  Returns:
    If invoked with a function as an argument, returns an instance of a TFF
    computation constructed based on this function. If called without one, as
    in the typical decorator style of usage, returns a callable that expects
    to be called with the function definition supplied as a parameter; see the
    patterns and examples of usage above.
  """
