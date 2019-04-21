# Lint as: python3
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
"""A library of transformation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import enum
import itertools

import six
from six.moves import range
from six.moves import zip

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import type_utils


def transform_postorder(comp, transform):
  """Traverses `comp` recursively postorder and replaces its constituents.

  For each element of `comp` viewed as an expression tree, the transformation
  `transform` is applied first to building blocks it is parameterized by, then
  the element itself. The transformation `transform` should act as an identity
  function on the kinds of elements (computation building blocks) it does not
  care to transform. This corresponds to a post-order traversal of the
  expression tree, i.e., parameters are alwaysd transformed left-to-right (in
  the order in which they are listed in building block constructors), then the
  parent is visited and transformed with the already-visited, and possibly
  transformed arguments in place.

  NOTE: In particular, in `Call(f,x)`, both `f` and `x` are arguments to `Call`.
  Therefore, `f` is transformed into `f'`, next `x` into `x'` and finally,
  `Call(f',x')` is transformed at the end.

  Args:
    comp: The computation to traverse and transform bottom-up.
    transform: The transformation to apply locally to each building block in
      `comp`. It is a Python function that accepts a building block at input,
      and should return either the same, or transformed building block as
      output. Both the input and output of `transform` are instances of
      `ComputationBuildingBlock`.

  Returns:
    The result of applying `transform` to parts of `comp` in a bottom-up
    fashion.

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
    return transform(comp)
  elif isinstance(comp, computation_building_blocks.Selection):
    return transform(
        computation_building_blocks.Selection(
            transform_postorder(comp.source, transform), comp.name, comp.index))
  elif isinstance(comp, computation_building_blocks.Tuple):
    return transform(
        computation_building_blocks.Tuple([
            (k, transform_postorder(v, transform))
            for k, v in anonymous_tuple.to_elements(comp)
        ]))
  elif isinstance(comp, computation_building_blocks.Call):
    transformed_transform = transform_postorder(comp.function, transform)
    if comp.argument is not None:
      transformed_arg = transform_postorder(comp.argument, transform)
    else:
      transformed_arg = None
    return transform(
        computation_building_blocks.Call(transformed_transform,
                                         transformed_arg))
  elif isinstance(comp, computation_building_blocks.Lambda):
    transformed_result = transform_postorder(comp.result, transform)
    return transform(
        computation_building_blocks.Lambda(comp.parameter_name,
                                           comp.parameter_type,
                                           transformed_result))
  elif isinstance(comp, computation_building_blocks.Block):
    return transform(
        computation_building_blocks.Block(
            [(k, transform_postorder(v, transform)) for k, v in comp.locals],
            transform_postorder(comp.result, transform)))
  else:
    raise NotImplementedError(
        'Unrecognized computation building block: {}'.format(str(comp)))


def transform_postorder_with_symbol_bindings(comp, transform, symbol_tree):
  """Uses symbol binding hooks to execute transformations.

  `transform_postorder_with_symbol_bindings` hooks into the preorder traversal
  that is defined by walking down the tree to its leaves, using
  the variable bindings along this path to push information onto
  the given `SymbolTree`. Once we hit the leaves, we walk back up the
  tree in a postorder fashion, calling `transform` as we go.

  The transformations `transform_postorder_with_symbol_bindings` executes are
  therefore stateful in some sense. Here 'stateful' means that a transformation
  executed on a given AST node in general depends on not only the node itself
  or its immediate vicinity; possibly there is some global information on which
  this transformation depends. `transform_postorder_with_symbol_bindings` is
  functional 'from AST to AST' (where `comp` represents the root of an AST) but
  not 'from node to node'.

  One important fact to note: there are recursion invariants that the
  `transform_postorder_with_symbol_bindings` uses the `SymbolTree` data
  structure
  to enforce. In particular, within a `transform` call the following
  invariants hold, for `symbol_tree` the SymbolTree argument to `transform`:

    * `symbol_tree.update_payload_tracking_reference` with an argument `ref` of
      type `Reference` will call `update` on the `BoundVariableTracker` in
      `symbol_tree` which tracks the value of `ref` active in the current
      lexical scope. Will raise a `NameError` if none exists.

    * `symbol_tree.get_payload_with_name` with a string argument `name`
       will return the `BoundVariableTracker` instance from `symbol_tree`
      which corresponds to the computation bound to the variable `name` in
      the current lexical scope. Will raise a `NameError` if none exists.

  These recursion invariants are enforced by the framework, and should be
  relied on when designing new transformations that depend on variable
  bindings.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock` to
      read information from or transform.
    transform: Python function accepting `comp` and `symbol_tree` arguments and
      returning `transformed_comp`.
    symbol_tree: Instance of `SymbolTree`, the data structure into which we may
      read information about variable bindings, and from which we may read.

  Returns:
    Returns a possibly modified version of `comp`, an instance
    of `computation_building_blocks.ComputationBuildingBlock`.
  """
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(symbol_tree, SymbolTree)
  identifier_seq = itertools.count(start=1)

  def _transform_postorder_with_symbol_bindings_switch(comp, transform_fn,
                                                       ctxt_tree,
                                                       identifier_sequence):
    """Recursive helper function delegated to after binding comp_id sequence."""
    if isinstance(comp, (computation_building_blocks.CompiledComputation,
                         computation_building_blocks.Data,
                         computation_building_blocks.Intrinsic,
                         computation_building_blocks.Placement,
                         computation_building_blocks.Reference)):
      return _traverse_leaf(comp, transform_fn, ctxt_tree, identifier_sequence)
    elif isinstance(comp, computation_building_blocks.Selection):
      return _traverse_selection(comp, transform, ctxt_tree,
                                 identifier_sequence)
    elif isinstance(comp, computation_building_blocks.Tuple):
      return _traverse_tuple(comp, transform, ctxt_tree, identifier_sequence)
    elif isinstance(comp, computation_building_blocks.Call):
      return _traverse_call(comp, transform, ctxt_tree, identifier_sequence)
    elif isinstance(comp, computation_building_blocks.Lambda):
      return _traverse_lambda(comp, transform, ctxt_tree, identifier_sequence)
    elif isinstance(comp, computation_building_blocks.Block):
      return _traverse_block(comp, transform, ctxt_tree, identifier_sequence)
    else:
      raise NotImplementedError(
          'Unrecognized computation building block: {}'.format(str(comp)))

  def _traverse_leaf(comp, transform, context_tree, identifier_seq):
    """Helper function holding traversal logic for leaf nodes."""
    _ = six.next(identifier_seq)
    return transform(comp, context_tree)

  def _traverse_selection(comp, transform, context_tree, identifier_seq):
    """Helper function holding traversal logic for selection nodes."""
    _ = six.next(identifier_seq)
    transformed_source = _transform_postorder_with_symbol_bindings_switch(
        comp.source, transform, context_tree, identifier_seq)
    transformed_comp = transform(
        computation_building_blocks.Selection(transformed_source, comp.name,
                                              comp.index), context_tree)
    return transformed_comp

  def _traverse_tuple(comp, transform, context_tree, identifier_seq):
    """Helper function holding traversal logic for tuple nodes."""
    _ = six.next(identifier_seq)
    new_elems = []
    for k, v in anonymous_tuple.to_elements(comp):
      transformed_elem = _transform_postorder_with_symbol_bindings_switch(
          v, transform, context_tree, identifier_seq)
      new_elems.append((k, transformed_elem))
    transformed_comp = transform(
        computation_building_blocks.Tuple(new_elems), context_tree)
    return transformed_comp

  def _traverse_call(comp, transform, context_tree, identifier_seq):
    """Helper function holding traversal logic for call nodes."""
    _ = six.next(identifier_seq)
    transformed_func = _transform_postorder_with_symbol_bindings_switch(
        comp.function, transform, context_tree, identifier_seq)
    if comp.argument is not None:
      transformed_arg = _transform_postorder_with_symbol_bindings_switch(
          comp.argument, transform, context_tree, identifier_seq)
    else:
      transformed_arg = None
    transformed_comp = transform(
        computation_building_blocks.Call(transformed_func, transformed_arg),
        context_tree)
    return transformed_comp

  def _traverse_lambda(comp, transform, context_tree, identifier_seq):
    """Helper function holding traversal logic for lambda nodes."""
    comp_id = six.next(identifier_seq)
    context_tree.ingest_variable_binding(
        name=comp.parameter_name,
        value=None,
        mode=MutationMode.CHILD,
        comp_id=comp_id)
    transformed_result = _transform_postorder_with_symbol_bindings_switch(
        comp.result, transform, context_tree, identifier_seq)
    transformed_comp = transform(
        computation_building_blocks.Lambda(comp.parameter_name,
                                           comp.parameter_type,
                                           transformed_result), context_tree)
    context_tree.move_to_parent_context()
    return transformed_comp

  def _traverse_block(comp, transform, context_tree, identifier_seq):
    """Helper function holding traversal logic for block nodes."""
    comp_id = six.next(identifier_seq)
    transformed_locals = []
    if comp.locals:
      first_local_name = comp.locals[0][0]
      first_local_comp = comp.locals[0][1]
      new_value = _transform_postorder_with_symbol_bindings_switch(
          first_local_comp, transform, context_tree, identifier_seq)
      transformed_locals.append((first_local_name, new_value))
      context_tree.ingest_variable_binding(
          name=transformed_locals[0][0],
          value=transformed_locals[0][1],
          mode=MutationMode.CHILD,
          comp_id=comp_id)
    for k in range(1, len(comp.locals)):
      new_value = _transform_postorder_with_symbol_bindings_switch(
          comp.locals[k][1], transform, context_tree, identifier_seq)
      transformed_locals.append((comp.locals[k][0], new_value))
      context_tree.ingest_variable_binding(
          name=transformed_locals[k][0],
          value=transformed_locals[k][1],
          mode=MutationMode.SIBLING)
    transformed_result = _transform_postorder_with_symbol_bindings_switch(
        comp.result, transform, context_tree, identifier_seq)
    transformed_comp = transform(
        computation_building_blocks.Block(transformed_locals,
                                          transformed_result), context_tree)
    if comp.locals:
      context_tree.move_to_parent_context()
    return transformed_comp

  return _transform_postorder_with_symbol_bindings_switch(
      comp, transform, symbol_tree, identifier_seq)


class MutationMode(enum.Enum):
  CHILD = 1
  SIBLING = 2


class SymbolTree(object):
  """Data structure to hold variable bindings as we walk an AST.

  `SymbolTree` is designed to be constructed and mutatated as we traverse an
  AST, maintaining a pointer to an active node representing the variable
  bindings we currently have available as we walk the AST.

  Each instance of the node class can be used at most once in the symbol tree,
  as checked by memory location. This disallows circular tree structures that
  could cause an infinite loop in recursive equality testing or printing.
  """

  def __init__(self, payload_type):
    """Initializes `SymbolTree` with its payload type.

    Args:
      payload_type: Class which subclasses BoundVariableTracker; the type of
        payloads to be constructed and held in this SymbolTree.
    """
    initial_node = SequentialBindingNode(OuterContextPointer())
    py_typecheck.check_subclass(payload_type, BoundVariableTracker)
    self.active_node = initial_node
    self.payload_type = payload_type
    self._node_ids = {id(initial_node): 1}

  def get_payload_with_name(self, name):
    """Returns payload corresponding to `name` in active variable bindings.

    Args:
      name: String name to find in currently active context.

    Returns:
      Returns instance of `BoundVariableTracker` corresponding to `name`
      in context represented by `active_comp`, or `None` if the requested
      name is unbound in the current context.

    Raises:
      NameError: If requested `name` is not found among the bound names
      currently available in `self`.

    """
    py_typecheck.check_type(name, six.string_types)
    comp = self.active_node
    while not isinstance(comp.payload, OuterContextPointer):
      if name == comp.payload.name:
        return comp.payload
      if comp.older_sibling is not None:
        comp = comp.older_sibling
      elif comp.parent is not None:
        comp = comp.parent
    raise NameError('Name {} is not available in {}'.format(name, self))

  def update_payload_tracking_reference(self, ref):
    """Calls `update` if it finds its Reference arg among the available symbols.

    If there is no such available symbol, simply does nothing.

    Args:
      ref: Instance of `computation_building_blocks.Reference`; generally, this
        is the variable a walker has encountered in a TFF AST, and which it is
        relying on `SymbolTable` to address correctly.

    Raises:
      NameError: If `ref` is not found among the bound names currently
        available in `self`.
    """
    py_typecheck.check_type(ref, computation_building_blocks.Reference)
    comp = self.active_node
    while not isinstance(comp.payload, OuterContextPointer):
      if ref.name == comp.payload.name:
        comp.payload.update(ref)
        break
      if comp.older_sibling is not None:
        comp = comp.older_sibling
      elif comp.parent is not None:
        comp = comp.parent
    if isinstance(comp.payload, OuterContextPointer):
      raise NameError('The reference {} is not available in {}'.format(
          ref, self))

  def ingest_variable_binding(self, name, value, mode, comp_id=None):
    """Constructs or updates node in symbol tree as AST is walked.

    Passes `name` and `value` onto the symbol tree's node constructor, with
    `mode` determining how the node being constructed or updated
    relates to the symbol tree's `active_node`.

    If there is no preexisting node in the symbol tree bearing the
    requested relationship to the active node, a new one will be constructed and
    initialized. If there is an existing node, `ingest_variable_binding` checks
    that this node has the correct `payload.name`, and overwrites its
    `payload.value` with the `value` argument.

    Args:
      name: The string name of the `CompTracker` instance we are constructing or
        updating.
      value: Instance of `computation_building_blocks.ComputationBuildingBlock`
        or `None`, as in the `value` to pass to symbol tree's node payload
        constructor.
      mode: Enum indicating the relationship the desired node should bear to the
        symbol tree's active node. Can be either CHILD or SIBLING.
      comp_id: Integer `comp_id` generated by walking the tree, used to address
        children of nodes in the symbol tree. Only necessary if `mode` is
        'child'.

    Raises:
      ValueError: If we are passed a name-mode pair such that a
        preexisting node in the symbol tree bears this relationship with
        the active node, but has a different name. This is an indication
        that either a transformation has failed to happen in the symbol tree
        or that we have a symbol tree instance that does not match the
        computation we are currently processing.
    """
    py_typecheck.check_type(mode, MutationMode)
    if mode == MutationMode.CHILD:
      py_typecheck.check_type(comp_id, int)
    py_typecheck.check_type(name, six.string_types)
    if value is not None:
      py_typecheck.check_type(
          value, computation_building_blocks.ComputationBuildingBlock)
    node = SequentialBindingNode(self.payload_type(name=name, value=value))
    if mode == MutationMode.SIBLING:
      if self.active_node.younger_sibling is None:
        self._add_younger_sibling(node)
        self._move_to_younger_sibling()
      else:
        if self.active_node.younger_sibling.payload.name != name:
          raise ValueError(
              'You have a mismatch between your symbol tree and the '
              'computation you are trying to process; your symbol tree is {} '
              'and you are looking for a BoundVariableTracker with name {} '
              'and value {}'.format(self, name, value))
        self._move_to_younger_sibling()
        self.active_node.payload.value = value
    else:
      if self.active_node.children.get(comp_id) is None:
        self._add_child(comp_id, node)
        self._move_to_child(comp_id)
      else:
        if self.active_node.children[comp_id].payload.name != name:
          raise ValueError(
              'You have a mismatch between your symbol tree and the '
              'computation you are trying to process; your symbol tree is {} '
              'and you are looking for a BoundVariableTracker with name {} '
              'and value {}'.format(self, name, value))
        self._move_to_child(comp_id)
        self.active_node.payload.value = value

  def move_to_parent_context(self):
    """Moves `active_node` to the parent of current active node.

    Of the `active_node` manipulation methods, this is the only one exposed.
    This is because the parent-child relationship corresponds directly to
    passing through a scope-introducing TFF AST node in a postorder traversal;
    therefore it is convenient to expose this as a mechanism to a TFF AST
    traversal function. The rest of these manipulation methods are more easily
    exposed via `ingest_variable_binding`.

    Raises:
      Raises ValueError if the active node has no parent.
    """
    if self.active_node.parent:
      self.active_node = self.active_node.parent
    else:
      raise ValueError('You have tried to move to a nonexistent parent.')

  def _add_younger_sibling(self, comp_tracker):
    """Appends comp as younger sibling of current `active_node`."""
    py_typecheck.check_type(comp_tracker, SequentialBindingNode)
    if self._node_ids.get(id(comp_tracker)):
      raise ValueError(
          'Each instance of {} can only appear once in a given symbol tree.'
          .format(self.payload_type))
    if self.active_node.younger_sibling is not None:
      raise ValueError('Ambiguity in adding a younger sibling')
    if self.active_node.parent is not None:
      comp_tracker.set_parent(self.active_node.parent)
    comp_tracker.set_older_sibling(self.active_node)
    self.active_node.set_younger_sibling(comp_tracker)
    self._node_ids[id(comp_tracker)] = 1

  def _move_to_younger_sibling(self):
    """Moves `active_node` to the younger sibling of the current active node.

    Raises:
      Raises ValueError if the active node has no younger sibling.
    """
    if self.active_node.younger_sibling:
      self.active_node = self.active_node.younger_sibling
    else:
      raise ValueError('You have tried to move to a '
                       'nonexistent younger sibling in ' + str(self))

  def _move_to_older_sibling(self):
    """Moves `active_node` to the older sibling of the current active node.

    Raises:
      Raises ValueError if the active node has no older sibling.
    """
    if self.active_node.older_sibling:
      self.active_node = self.active_node.older_sibling
    else:
      raise ValueError('You have tried to move to a '
                       'nonexistent older sibling in ' + str(self))

  def _add_child(self, constructing_comp_id, comp_tracker):
    """Writes `comp_tracker` to children of active node.

    Each `SequentialBindingNode` keeps a `dict` of its children; `_add_child`
    updates the value of this `dict` with key `constructing_comp_id` to be
    `comp_tracker`.

    Notice that `constructing_comp_id` is simply a way of addressing the
    children in this dict; it is not necessarily globally unique, as long
    as it is sufficient to address child scopes.

    Args:
      constructing_comp_id: Key to identify child being constructed from the
        parent scope.
      comp_tracker: Instance of `SequentialBindingNode`, the node to add as a
        child of `active_node`.
    """
    py_typecheck.check_type(comp_tracker, SequentialBindingNode)
    if self._node_ids.get(id(comp_tracker)):
      raise ValueError('Each node can only appear once in a given'
                       'symbol tree. You have tried to add {} '
                       'twice.'.format(comp_tracker.payload))
    comp_tracker.set_parent(self.active_node)
    self.active_node.add_child(constructing_comp_id, comp_tracker)
    self._node_ids[id(comp_tracker)] = 1

  def _move_to_child(self, comp_id):
    """Moves `active_node` to child of current active node with key `comp_id`.

    Args:
      comp_id: Integer representing the position of the child we wish to update
        `active_node` to point to in a preorder traversal of the AST.

    Raises:
      ValueError: If the active node has no child with the correct id.
    """
    if self.active_node.children.get(comp_id) is not None:
      self.active_node = self.active_node.get_child(comp_id)
    else:
      raise ValueError('You have tried to move to a nonexistent child.')

  def _equal_under_node(self, self_node, other_node):
    """Recursive helper function to check equality of `SymbolTree`s."""
    if self_node is None and other_node is None:
      return True
    if self_node is None or other_node is None:
      return False
    if self_node.payload != other_node.payload:
      return False
    if len(self_node.children) != len(other_node.children):
      return False
    for (_, val_1), (_, val_2) in zip(
        six.iteritems(self_node.children), six.iteritems(other_node.children)):
      # keys not compared to avoid coupling walking logic to `SymbolTree`.
      if not self._equal_under_node(val_1, val_2):
        return False
    return self._equal_under_node(self_node.younger_sibling,
                                  other_node.younger_sibling)

  def __eq__(self, other):
    """Walks to root of `self` and `other` before testing equality of subtrees.

    Args:
      other: Instance of `SymbolTree` to test for equality with `self`.

    Returns:
      Returns `True` if and only if `self` and `other` are the same
      structurally (each node has the same number of children and siblings) and
      each node of `self` compares as equal with the node in the corresponding
      position of `other`.
    """
    if self is other:
      return True
    if not isinstance(other, SymbolTree):
      return NotImplemented
    self_node = _walk_to_root(self.active_node)
    other_node = _walk_to_root(other.active_node)
    return self._equal_under_node(self_node, other_node)

  def __ne__(self, other):
    return not self == other

  def _string_under_node(self, node):
    """Rescursive helper function to generate string reps of `SymbolTree`s."""
    py_typecheck.check_type(node, SequentialBindingNode)
    if node is self.active_node:
      active_node_indicator = '*'
    else:
      active_node_indicator = ''
    symbol_tree_string = '[' + str(node.payload) + active_node_indicator + ']'
    if node.children:
      symbol_tree_string += '->{'
      for _, child_node in six.iteritems(node.children):
        if not child_node.older_sibling:
          symbol_tree_string += '('
          symbol_tree_string += self._string_under_node(child_node)
          symbol_tree_string += '),('
      symbol_tree_string = symbol_tree_string[:-2]
      symbol_tree_string += '}'
    if node.younger_sibling:
      symbol_tree_string += '-' + self._string_under_node(node.younger_sibling)
    return symbol_tree_string

  def __str__(self):
    """Generates a string representation of this `SymbolTree`.

    First we walk up to the root node, then we walk down
    the tree generating string rep of this symbol tree.

    Returns:
      Returns a string representation of the current `SymbolTree`, with
      the node labeled the active node identified with a *.
    """
    node = self.active_node
    root_node = _walk_to_root(node)
    py_typecheck.check_type(root_node.payload, OuterContextPointer)
    return self._string_under_node(root_node)


def _walk_to_root(node):
  while node.parent is not None:
    node = node.parent
  while node.older_sibling is not None:
    node = node.older_sibling
  return node


class SequentialBindingNode(object):
  """Represents a node in a context tree with sequential-binding semantics.

  `SequentialBindingNode`s are designed to be constructed and pushed into
  a context tree as an AST representing a given computation is walked.

  Each `SequentialBindingNode` holds as payload a variable binding in the AST.
  The node-node relationships encoded by the `SequentialBindingNode` data
  structure determine how the context tree must be walked in order to resolve
  variables and track their values in the AST.

  Parent-child relationships represent relationships between levels of the AST,
  meaning, moving through an AST node which defines a variable scope in preorder
  corresponds to moving from a `SequentialBindingNode` to one of its children,
  and moving through such a node postorder corresponds to moving from a
  `SequentialBindingNode` to its parent.

  Sibling-sibling relationships are particular to sequential binding of
  variables in `computation_building_blocks.Block` constructs; binding
  a new variable in such a construct corresponds to moving from a
  `SequentialBindingNode` to its (unique) younger sibling.
  """

  def __init__(self, payload):
    """Initializes `SequentialBindingNode`.

    Args:
      payload: Instance of BoundVariableTracker representing the payload of this
        node.
    """
    py_typecheck.check_type(payload, BoundVariableTracker)
    self.payload = payload
    self._children = collections.OrderedDict()
    self._parent = None
    self._older_sibling = None
    self._younger_sibling = None

  @property
  def parent(self):
    return self._parent

  @property
  def children(self):
    return self._children

  @property
  def older_sibling(self):
    return self._older_sibling

  @property
  def younger_sibling(self):
    return self._younger_sibling

  def set_parent(self, node):
    """Sets the _parent scope of `self` to the binding embodied by `node`.

    This method should not be assumed to be efficient.

    Args:
      node: Instance of `SequentialBindingNode` to set as parent of `self`.
    """
    py_typecheck.check_type(node, SequentialBindingNode)
    self._parent = node

  def set_older_sibling(self, node):
    """Sets the older sibling scope of `self` to `node`.

    This method should not be assumed to be efficient.

    Args:
      node: Instance of `SequentialBindingNode` to set as older sibling of
        `self`.
    """
    py_typecheck.check_type(node, SequentialBindingNode)
    self._older_sibling = node

  def set_younger_sibling(self, node):
    """Sets the younger sibling scope of `self` to `node`.

    This corresponds to binding a new variable in a
    `computation_building_blocks.Block` construct.

    This method should not be assumed to be efficient.

    Args:
      node: Instance of `SequentialBindingNode` representing this new binding.
    """
    py_typecheck.check_type(node, SequentialBindingNode)
    self._younger_sibling = node

  def add_child(self, comp_id, node):
    """Sets the child scope of `self` indexed by `comp_id` to `node`.

    This corresponds to encountering a node in a TFF AST which defines a
    variable scope.

    If a child with this `comp_id` already exists, it is replaced, as in a
    `dict`.

    Args:
      comp_id: The identifier of the computation generating this scope.
      node: Instance of `SequentialBindingNode` representing this new binding.
    """
    py_typecheck.check_type(node, SequentialBindingNode)
    self._children[comp_id] = node

  def get_child(self, comp_id):
    """Returns the child of `self` identified by `comp_id` if one exists.

    Args:
      comp_id: Integer used to address child of `self` by position of
        corresponding AST node in a preorder traversal of the AST.

    Returns:
      Instance of `SequentialBindingNode` if an appropriate child of `self`
      exists, or `None`.
    """
    return self._children.get(comp_id)


@six.add_metaclass(abc.ABCMeta)
class BoundVariableTracker(object):
  """Abstract class representing a mutable variable binding."""

  def __init__(self, name, value):
    """Initializes `BoundVariableTracker`.

    The initializer is likely to be overwritten by subclasses in order to
    attach more state to the `BoundVariableTracker`. Each of them must
    satisfy the same interface, however. This is simply because the
    `BoundVariableTracker` represents a variable binding in a TFF AST;
    no more information is avaiable to it than the `name`-`value` pair
    being bound together.

    Args:
      name: String name of variable to be bound.
      value: Value to bind to this name. Can be instance of
        `computation_building_blocks.ComputationBuildingBlock` if this
        `BoundVariableTracker` represents a concrete binding to a variable (e.g.
        in a block locals declaration), or `None`, if this
        `BoundVariableTracker` represents merely a variable declaration (e.g. in
        a lambda).
    """
    py_typecheck.check_type(name, six.string_types)
    if value is not None:
      py_typecheck.check_type(
          value, computation_building_blocks.ComputationBuildingBlock)
    self.name = name
    self.value = value

  @abc.abstractmethod
  def update(self, value=None):
    """Abstract method defining the way information is read into this node.

    Args:
      value: Similar to `value` argument in initializer.
    """
    pass

  @abc.abstractmethod
  def __str__(self):
    """Abstract string method required as context tree will delegate."""
    pass

  def __eq__(self, other):
    """Base class equality checks names and values equal."""
    # TODO(b/130890785): Delegate value-checking to
    # `computation_building_blocks.ComputationBuildingBlock`.
    if self is other:
      return True
    if not isinstance(other, BoundVariableTracker):
      return NotImplemented
    if self.name != other.name:
      return False
    if isinstance(self.value,
                  computation_building_blocks.ComputationBuildingBlock):
      return isinstance(
          other.value, computation_building_blocks.ComputationBuildingBlock
      ) and self.value.tff_repr == other.value.tff_repr and type_utils.are_equivalent_types(
          self.value.type_signature, other.value.type_signature)
    return self.value is other.value

  def __ne__(self, other):
    """Implementing __ne__ to enforce in Python2 the Python3 standard."""
    return not self == other


class OuterContextPointer(BoundVariableTracker):
  """Sentinel node class representing the context 'outside' a given AST."""

  def __init__(self, name=None, value=None):
    if name is not None or value is not None:
      raise ValueError('Please don\'t pass a name or value to '
                       'OuterContextPointer; it will simply be ignored.')
    super(OuterContextPointer, self).__init__('OuterContext', None)

  def update(self, comp=None):
    del comp
    raise RuntimeError('We shouldn\'t be trying to update the outer context.')

  def __str__(self):
    return self.name

  def __eq__(self, other):
    """Returns `True` iff `other` is also an `OuterContextPointer`.

    OuterContextPointer simply refers to a global notion of 'external',
    so all instances are identical--"outside" of every building refers
    to the same location.

    Args:
      other: Value for equality comparison.

    Returns:
      Returns true iff `other` is also an instance of `OuterContextPointer`.
    """
    # Using explicit type comparisons here to prevent a subclass from passing.
    # pylint: disable=unidiomatic-typecheck
    return type(other) is OuterContextPointer
    # pylint: enable=unidiomatic-typecheck


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


class ReferenceCounter(BoundVariableTracker):
  """Data container to track number References to a variable in an AST.


  Attributes:
    name: The string name representing the variable whose binding is represented
      by an instance of `ReferenceCounter`.
    value: The value bound to `name`. Can be an instance of
      `computation_building_blocks.ComputationBuildingBlock` or None if this
      binding is simply a placeholder, e.g. in a Lambda.
    count: An integer tracking how many times the variable an instance of
      `ReferenceCounter` represents is referenced in a TFF AST.
  """

  def __init__(self, name, value):
    super(ReferenceCounter, self).__init__(name, value)
    self.count = 0

  def update(self, reference=None):
    del reference  # Unused
    self.count += 1

  def __str__(self):
    return 'Instance count: {}; value: {}; name: {}.'.format(
        self.count, self.value, self.name)

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    if self is other:
      return True
    if not isinstance(other, ReferenceCounter):
      return NotImplemented
    if not super(ReferenceCounter, self).__eq__(other):
      return False
    return self.count == other.count


def get_count_of_references_to_variables(comp):
  """Returns `SymbolTree` counting references to each bound variable in `comp`.

  Args:
    comp: Instance of `computation_building_blocks.ComputationBuildingBlock`
      representing the root of the AST for which we want to read total reference
      counts by context.

  Returns:
    An instance of `SymbolTree` representing root of context tree
    populated with `transformation_utils.ReferenceCounter`s which
    contain the number of times each variable bound by a
    `computation_building_blocks.Lambda`
    or `computation_building_blocks.Block` are referenced in their computation's
    body.
  """

  reference_counter = SymbolTree(ReferenceCounter)

  def _should_transform(comp, context_tree):
    del context_tree  # Unused
    return isinstance(comp, computation_building_blocks.Reference)

  def transform_fn(comp, context_tree):
    if _should_transform(comp, context_tree):
      context_tree.update_payload_tracking_reference(comp)
    return comp

  transform_postorder_with_symbol_bindings(comp, transform_fn,
                                           reference_counter)
  return reference_counter
