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
"""A library of all transformations to be performed by the compiler pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import federated_computation_utils


def transform_postorder(comp, fn):
  """Traverses `comp` recursively postorder and replaces its constituents.

  For each element of `comp` viewed as an expression tree, the transformation
  `fn` is applied first to building blocks it is parameterized by, then the
  element itself. The transformation `fn` should act as an identity function
  on the kinds of elements (computation building blocks) it does not care to
  transform. This corresponds to a post-order traversal of the expression tree,
  i.e., parameters are alwaysd transformed left-to-right (in the order in which
  they are listed in building block constructors), then the parent is visited
  and transformed with the already-visited, and possibly transformed arguments
  in place.

  NOTE: In particular, in `Call(f,x)`, both `f` and `x` are arguments to `Call`.
  Therefore, `f` is transformed into `f'`, next `x` into `x'` and finally,
  `Call(f',x')` is transformed at the end.

  Args:
    comp: The computation to traverse and transform bottom-up.
    fn: The transformation to apply locally to each building block in `comp`.
      It is a Python function that accepts a building block at input, and should
      return either the same, or transformed building block at output. Both the
      intput and output of `fn` are instances of `ComputationBuildingBlock`.

  Returns:
    The result of applying `fn` to parts of `comp` in a bottom-up fashion.

  Raises:
    TypeError: If the arguments are of the wrong computation_types.
    NotImplementedError: If the argument is a kind of computation building block
      that is currently not recognized.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  if isinstance(
      comp,
      (computation_building_blocks.CompiledComputation,
       computation_building_blocks.Data, computation_building_blocks.Intrinsic,
       computation_building_blocks.Placement,
       computation_building_blocks.Reference)):
    return fn(comp)
  elif isinstance(comp, computation_building_blocks.Selection):
    return fn(
        computation_building_blocks.Selection(
            transform_postorder(comp.source, fn), comp.name, comp.index))
  elif isinstance(comp, computation_building_blocks.Tuple):
    return fn(
        computation_building_blocks.Tuple([(k, transform_postorder(
            v, fn)) for k, v in anonymous_tuple.to_elements(comp)]))
  elif isinstance(comp, computation_building_blocks.Call):
    transformed_fn = transform_postorder(comp.function, fn)
    if comp.argument is not None:
      transformed_arg = transform_postorder(comp.argument, fn)
    else:
      transformed_arg = None
    return fn(computation_building_blocks.Call(transformed_fn, transformed_arg))
  elif isinstance(comp, computation_building_blocks.Lambda):
    transformed_result = transform_postorder(comp.result, fn)
    return fn(
        computation_building_blocks.Lambda(
            comp.parameter_name, comp.parameter_type, transformed_result))
  elif isinstance(comp, computation_building_blocks.Block):
    return fn(
        computation_building_blocks.Block(
            [(k, transform_postorder(v, fn)) for k, v in comp.locals],
            transform_postorder(comp.result, fn)))
  else:
    raise NotImplementedError(
        'Unrecognized computation building block: {}'.format(str(comp)))


def name_compiled_computations(comp):
  """Labels all compiled computations with names that are unique within `comp`.

  Args:
    comp: The computation building block in which to perform the replacements.

  Returns:
    A modified variant of `comp` with all compiled computations given unique
    names.
  """

  def _name_generator():
    n = 0
    while True:
      n = n + 1
      yield str(n)

  def _transformation_fn(x, name_sequence):
    if not isinstance(x, computation_building_blocks.CompiledComputation):
      return x
    else:
      return computation_building_blocks.CompiledComputation(
          x.proto, six.next(name_sequence))

  name_sequence = _name_generator()
  return transform_postorder(
      comp, lambda x: _transformation_fn(x, name_sequence))


def replace_intrinsic(comp, uri, body, context_stack):
  """Replaces all occurrences of an intrinsic.

  Args:
    comp: The computation building block in which to perform the replacements.
    uri: The URI of the intrinsic to replace.
    body: A polymorphic callable that represents the body of the implementation
      of the intrinsic, i.e., one that given the parameter of the intrinsic
      constructs the intended result. This will typically be a Python function
      decorated with `@federated_computation` to make it into a polymorphic
      callable.
    context_stack: The context stack to use.

  Returns:
    A modified variant of `comp` with all occurrences of the intrinsic with
    the URI equal to `uri` replaced with the logic constructed by `replacement`.

  Raises:
    TypeError: if types do not match somewhere in the course of replacement.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(uri, six.string_types)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  if not callable(body):
    raise TypeError('The body of the intrinsic must be a callable.')

  def _transformation_fn(comp, uri, body):
    """Internal function to replace occurrences of an intrinsic."""
    if not isinstance(comp, computation_building_blocks.Intrinsic):
      return comp
    elif comp.uri != uri:
      return comp
    else:
      py_typecheck.check_type(comp.type_signature,
                              computation_types.FunctionType)
      # We need 'wrapped_body' to accept exactly one argument.
      wrapped_body = lambda x: body(x)  # pylint: disable=unnecessary-lambda
      return federated_computation_utils.zero_or_one_arg_fn_to_building_block(
          wrapped_body,
          'arg',
          comp.type_signature.parameter,
          context_stack,
          suggested_name=uri)

  return transform_postorder(comp, lambda x: _transformation_fn(x, uri, body))


def replace_called_lambdas_with_block(comp):
  """Replaces occurrences of Call(Lambda(...), ...)) with Block(...).

  This transformation is used to facilitate the merging of TFF orchestration
  logic, in particular to remove unnecessary lambda expressions and as a
  stepping stone for merging Blocks together to maximal effect.

  Args:
    comp: The computation building block in which to perform the replacements.

  Returns:
    A modified version of `comp` with all occurrences of the pattern Call(
    Lambda(...), args) replaced with an equivalent Block(...), or the original
    `comp` if it does not follow the `Call(Lambda(...), args)` pattern.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)

  def _transform(comp):
    """Internal function to break down Call-Lambda and build Block."""
    if not isinstance(comp, computation_building_blocks.Call):
      return comp
    elif not isinstance(comp.function, computation_building_blocks.Lambda):
      return comp
    py_typecheck.check_type(
        comp.argument, computation_building_blocks.ComputationBuildingBlock)
    arg = comp.argument
    lam = comp.function
    param_name = lam.parameter_name
    result = lam.result
    return computation_building_blocks.Block([(param_name, arg)], result)

  return transform_postorder(comp, _transform)


@six.add_metaclass(abc.ABCMeta)
class ScopedSnapshot(object):
  """Callable to allow for taking snapshot of scopes (contexts) in an AST.

  `ScopedSnapshot` is an abstract class, designed to provide the basic
  functionality for reading off information from an AST by variable scopes
  or contexts. That is, constructing a dict keyed by identifiers of the
  `computation_building_blocks.Block` and `computation_building_blocks.Lambda`
  computations in the AST, and mapping to some information about the variables
  these functions declare. The particular information tracked by
  `ScopedSnapshot` is determined by the implementations of the classes
  that inherit from it, in particular the functions
  `initialize_local_symbols_dict`,
  `update_local_symbols_dict`, and `merge_symbols_under_comp`.

  Unfortunately, this implementation essentially requires the tree traversals to
  be of quadratic complexity in the number of nodes in the AST, since it is
  designed to stop on the way up the tree whenever it encounters a
  `computation_building_blocks.ComputationBuildingBlock` defining a variable
  scope, and use the information this scope defines to re-traverse under the
  scope.

  The end result of calling this callable is the `global_snapshot` attribute,
  which is a nested dictionary representing the information read off from the
  AST. The outer dict is a mapping from string representations of computations
  defining scopes to inner `dict`s containing the information read off about
  this scope. The inner dicts in general are keyed by the names of variables
  defined in these scopes, and the values are determined initially by
  `initialize_local_symbols_dict`, and updated by `update_local_symbols_dict`.
  """

  # TODO(b/123428953): Implement reduction of complexity in AST transforms
  # from quadratic to linear.

  # TODO(b/123428953): There is nothing special about the string reps here--
  # they could be replaced by addressing by position in a postorder traversal.
  # However, since the nested walk happens anyway, this is lower priority to
  # remove.

  def __init__(self):
    self.global_snapshot = {}

  @abc.abstractmethod
  def merge_symbols_under_comp(self, scope):
    """Updates `current_names` with the information present in arg `scope`.

    The traversal formalized in `ScopedSnapshot` walks the AST postorder,
    and when it encounters a `computation_building_blocks.Block` or
    `computation_building_blocks.Lambda`, kicks off a new postorder traversal
    under this node. For this reason, some nodes can be traversed multiple
    times. Therefore it is necessary to use the information previously read
    from scopes under the calling computation to correct for this multiple
    traversal. `merge_under_scope` is the function designed to implement
    this correction.

    Args:
      scope: A `dict` representing the information read off from some function
        scope underneath the calling `computation_building_blocks.Block` or
        `computation_building_blocks.Lambda` in the AST to which `current_names`
        is bound. `scope` is keyed with names the references which its
        associated context defines.
    """
    pass

  @abc.abstractmethod
  def initialize_local_symbols_dict(self, comp, assigned_var):
    """Returns initial values for `current_names` associated to `comp`.

    `ScopedSnapshot` generates a nested dictionary; the outer dict is keyed
    by some identifier of scopes in the AST (e.g. string representation of
    the computation defining this scope), while the inner dict is keyed by
    names of variables within this scope. The inner dict can track whatever
    information it likes about this variable, as defined by the classes
    that inherit from `ScopedSnapshot`. `initialize_local_symbols_dict` must
    take care
    of initializing the values in this inner dictionary.

    Args:
      comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
        which we would like to track in `global_snapshot`. Generally, str(comp)
        will be used as the key to the outer dict `global_snapshot`.
        `initialize_local_symbols_dict` is intended to simply return the `value`
        with which we would like to initialize the inner `dict` representing
        information read from the scope bound to `comp`. `comp` is passed in
        here since it may be necessary to treat different computations that
        define scope differently (e.g. `computation_building_blocks.Block`s may
        be treated differently from `computation_building_blocks.Lambda`s), and
        `assigned_var` is passed in as we may in general want any information
        the computation `assigned_var` contains, up to and including the entire
        computation itself.
      assigned_var: Instance of
        `computation_building_blocks.ComputationBuildingBlock` assigned to the
        variable we are looking to initialize in the inner dict.
    """
    pass

  @abc.abstractmethod
  def update_local_symbols_dict(self, comp):
    """Updates the `dict` `current_names` with the information from `comp`.

    For example, if we are tracking number of references to a variable in an
    enclosing context, `update_local_symbols_dict` called on a reference to that
    variable
    would add 1 to the `current_names` dict keyed by the name of the variable.

    Args:
      comp: The `computation_building_blocks.ComputationBuildingBlock` instance
        to use for updating `current_names`.
    """
    pass

  def _merge_under_scope(self, inner_comp):
    """Calls `merge_symbols_under_comp` if `inner_comp` has its own scope.

    That is, if `inner_comp` is an instance of
    `computation_building_blocks.Block` or `computation_building_blocks.Lambda`.
    Since we are traversing the tree postorder, and calling
    `_merge_under_scope` from within a nested traversal, all the
    information associated to the scope `inner_comp` defines has already been
    read into `global_snapshot`. Therefore `_merge_under_scope` addresses
    `global_snapshot` by the string representation of `inner_comp`, and
    passes this inner dict to `merge_symbols_under_comp`.

    Args:
      inner_comp: `computation_building_blocks.ComputationBuildingBlock`
        instance for which to pass corresponding scope information up to calling
        context.

    Returns:
      `inner_comp` unchanged.
    """
    if isinstance(inner_comp, (computation_building_blocks.Block,
                               computation_building_blocks.Lambda)):
      self.merge_symbols_under_comp(self.global_snapshot[str(inner_comp)])
    return inner_comp

  def __call__(self, comp):
    """Populates `global_snapshot` with values scoped to `comp`.

    If a `computation_building_blocks.Block` declares a Local, in the AST under
    this `computation_building_blocks.Block`, the name of this
    Local is bound to its value in the declaration unless overridden by a
    narrower scope.
    Similarly if a `computation_building_blocks.Lambda` declares a Parameter
    name.

    `__call__` will be called in postorder fashion on a given AST; for ease of
    understanding we may consider the loop invariant respected by `__call__`.
    Before every call to `__call__`, we assume that all Blocks and
    Lambdas strictly under `comp` in the AST (i.e. scopes narrower than that
    which may be bound to `comp`) have had their information
    correctly read off into the dict `global_snapshot`, where it is keyed
    by the string representation of the computation. To verify `__call__` we
    must verify that this invariant is respected.

    This assertion depends on the interaction of `update_local_symbols_dict` and
    `merge_symbols_under_comp`; every class which inherits from `ScopedSnapshot`
    must
    ensure that after:
      * `update_local_symbols_dict(underlying_comp)` has been called for every
      `underlying_comp` under `comp` in the AST
      * `merge_symbols_under_comp(scope)` has been called for every `scope`
      associated to a `computation_building_blocks.Block` or
      `computation_building_blocks.Lambda` strictly underneath `comp` in the
      AST
    `current_names` represents the correct information that `comp`
    is to contribute to reading out of the AST. `__call__` then attaches
    this information to `global_snapshot`, respecting the invariant
    as we make our way back up the tree.

    Args:
      comp: `computation_building_blocks.ComputationBuildingBlock` instance for
        which to generate information on its associated abstract syntax tree
        broken down by contexts, e.g. counts of usages of variables of name 'x'
        in scope 'y'.

    Returns:
      comp: Unchanged.
    """

    if isinstance(comp, computation_building_blocks.Block):
      self.current_names = {}
      for bound_names in comp.locals:
        self.current_names[bound_names[0]] = self.initialize_local_symbols_dict(
            comp, bound_names[1])
      transform_postorder(comp.result, self.update_local_symbols_dict)
      transform_postorder(comp.result, self._merge_under_scope)
      self.global_snapshot[str(comp)] = self.current_names

    elif isinstance(comp, computation_building_blocks.Lambda):
      self.current_names = {}
      self.current_names[
          comp.parameter_name] = self.initialize_local_symbols_dict(comp, None)
      transform_postorder(comp.result, self.update_local_symbols_dict)
      transform_postorder(comp.result, self._merge_under_scope)
      self.global_snapshot[str(comp)] = self.current_names

    return comp


class CountContextOccurrences(ScopedSnapshot):
  """Callable to count variable instances by context.

  Scope here is defined to be the nodes of the AST under `comp` on which
  this callable is invoked which are enclosed by variable declarations,
  either as parameters to `computation_building_blocks.Lambda`s or
  `computation_building_blocks.Block`s. Note that the locals declaration
  of a given block defines the scope for its result, and so is in a different
  variable scope.
  """

  def initialize_local_symbols_dict(self, comp, var_name):
    del comp, var_name
    return 0

  def update_local_symbols_dict(self, comp):
    """Adds 1 to counter of occurrences of variable's name in current scope."""
    if isinstance(comp, computation_building_blocks.Reference):
      if self.current_names.get(comp.name, None) is not None:
        self.current_names[comp.name] += 1
    return comp

  def merge_symbols_under_comp(self, prev_vars):
    """Subtracts references in `prev_vars` from those in `current_names`.

    All bound variables strictly under the calling `comp` are counted by
    `update_local_symbols_dict`; if their names conflict with a bound variable
    in a
    narrower scope this is incorrect and must be remedied. As we are using
    postorder traversal to walk our AST this must be done after the fact;
    we cannot tell by inspecting our node or the value of an instance
    variable which scope we are currently in as we walk the tree.

    Since the argument `scope` is always narrower than `current_names` and
    `ConutScopeOccurrences` is looking to track how many times the variables
    its scope declares are referenced or selected from, the appropriate action
    is simply to subtract the number of times the variable with the same name
    is referenced in the subscope `scope`.

    Args:
      prev_vars: `dict` containing (variable name, number of references in
        `scope`) pairs corresponding to `scope`. `scope` is always a strict
        subscope of `current_names`, by construction in `__call__`.
    """
    for key in prev_vars:
      if key in list(self.current_names.keys()):
        self.current_names[key] -= prev_vars[key]


def scope_count_snapshot(comp):
  photog = CountContextOccurrences()
  _ = transform_postorder(comp, photog)
  return photog.global_snapshot


def list_comp_names(comp):
  """Canonical list of string representations of nodes in `comp`.

  Used as a helper function to generate static name-to-index mappings.

  Args:
    comp: The root of the AST for which we wish to generate a list of string
      representations of all nodes.

  Returns:
    names: Python `list` of string representations of nodes under `comp`.
      This list is generated by walking the AST of `comp` in postorder fashion
      and thus is deterministic.
  """
  names = []

  def _string_rep(inner_comp):
    names.append(str(inner_comp))
    return inner_comp

  transform_postorder(comp, _string_rep)
  return names


class InlineReferences(object):
  """Stateful callable that inlines local variables when appropriate."""

  def __init__(self, inlining_threshold, counter_snapshot, initial_comp):
    """Initializes object with inlining cutoff and counter for mutations.

    Args:
      inlining_threshold: Int to use as cutoff for inlining. That is, any value
        which is referenced `inlining_threshold` or fewer times under a
        `computation_building_blocks.Block` is to be used as a replacement for
        the `computation_building_blocks.Reference`s which reference it.
      counter_snapshot: `dict` of `dict`s, each dict containing bindings from
        variable name to number of times referenced in its parent scope. Outer
        dicts are keyed by string representations of parent computations owning
        these scopes.
      initial_comp: The root of the AST which we are inlining. Used to generate
        a map from initial string representations of the AST's nodes to indices
        when the tree is walked in postorder fashion. As `counter_snapshot` is
        keyed by string representations, we use this map with the counter `idx`
        to address `counter_snapshot`.
    """
    self.inlining_threshold = inlining_threshold
    self.counts = counter_snapshot
    self.initial_comp_names = list_comp_names(initial_comp)
    self.idx = -1

  def __call__(self, comp):
    """Counts references to locals under Block and performs inlining.

    If the `comp` argument is a `computation_building_blocks.Block`, `__call__`
    selects the locals on which to perform inlining based on the threshold
    defined in `inlining_threshold` and the snapshot of the calling AST
    before any transformations are executed, stored as `counts`, before
    executing the inlining itself.

    Args:
      comp: The `computation_building_blocks.ComputationBuildingBlock` to be
        checked for the possibility of inlining.

    Returns:
      comp: A transformed version of `comp`, with locals of any of its
        `computation_building_blocks.Block`s which are referenced
        `inlining_threshold` or fewer times replaced with their
        associated values. All local declarations no longer referenced
        in the body are removed.
    """
    self.idx += 1
    if isinstance(comp, (computation_building_blocks.Block)):
      bound_dict = self.counts[self.initial_comp_names[self.idx]]
      values_to_replace = [
          k for k, v in bound_dict.items() if v <= self.inlining_threshold
      ]
      names_and_values = {
          x[0]: x[1] for x in comp.locals if x[0] in values_to_replace
      }

      def _execute_inlining_from_bound_dict(inner_comp):
        """Uses `dict` bound to calling comp to inline as appropriate.

        Args:
          inner_comp: The `computation_building_blocks.ComputationBuildingBlock`
            to potentially inline.

        Returns:
          `computation_building_blocks.ComputationBuildingBlock`, `inner_comp`
          unchanged if `inner_comp` is not a
          `computation_building_blocks.Reference` whose name  appears in
          `bound_dict`; otherwise the appropriate local definition.
        """
        if (isinstance(inner_comp, computation_building_blocks.Reference) and
            names_and_values.get(inner_comp.name)):
          py_typecheck.check_type(
              names_and_values[inner_comp.name],
              computation_building_blocks.ComputationBuildingBlock)
          return names_and_values[inner_comp.name]
        return inner_comp

      remaining_locals = [(name, val)
                          for name, val in comp.locals
                          if name not in values_to_replace]
      return computation_building_blocks.Block(
          remaining_locals,
          transform_postorder(comp.result, _execute_inlining_from_bound_dict))
    else:
      return comp


def inline_blocks_with_n_referenced_locals(comp, inlining_threshold=1):
  """Replaces locals referenced few times in `comp` with bound values.

  Args:
    comp: The computation building block in which to inline the locals which
      occur only `inlining_threshold` times in the result computation.
    inlining_threshold: The threshhold below which to inline computations. E.g.
      if `inlining_threshold` is 1, locals which are referenced exactly once
      will be inlined, but locals which are referenced twice or more will not.

  Returns:
    A modified version of `comp` for which all occurrences of
    `computation_building_blocks.Block`s with locals which
      are referenced `inlining_threshold` or fewer times inlined with the value
      of the local.
  """

  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  count_snapshot = scope_count_snapshot(comp)
  op = InlineReferences(inlining_threshold, count_snapshot, comp)
  return transform_postorder(comp, op)
