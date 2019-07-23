<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.IterativeProcess" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="next"/>
<meta itemprop="property" content="__init__"/>
</div>

# tff.utils.IterativeProcess

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/computation_utils.py">View
source</a>

## Class `IterativeProcess`

A process that includes an initialization and iterated computation.

<!-- Placeholder for "Used in" -->

An iterated process will usually be driven by a control loop like:

```python
def initialize():
  ...

def next(state):
  ...

iterative_process = IterativeProcess(initialize, next)
state = iterative_process.initialize()
for round in range(num_rounds):
  state = iterative_process.next(state)
```

The iteration step can accept arguments in addition to `state` (which must be
the first argument), and return additional arguments:

```python
def next(state, item):
  ...

iterative_process = ...
state = iterative_process.initialize()
for round in range(num_rounds):
  state, output = iterative_process.next(state, round)
```

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/computation_utils.py">View
source</a>

```python
__init__(
    initialize_fn,
    next_fn
)
```

Creates a `tff.IterativeProcess`.

#### Args:

*   <b>`initialize_fn`</b>: a no-arg
    <a href="../../tff/Computation.md"><code>tff.Computation</code></a> that
    creates the initial state of the chained computation.
*   <b>`next_fn`</b>: a
    <a href="../../tff/Computation.md"><code>tff.Computation</code></a> that
    defines an iterated function. If `initialize_fn` returns a type _T_, then
    `next_fn` must also return type _T_ or multiple values where the first is of
    type _T_, and accept either a single argument of type _T_ or multiple
    arguments where the first argument must be of type _T_.

#### Raises:

*   <b>`TypeError`</b>: `initialize_fn` and `next_fn` are not compatible
    function types.

## Properties

<h3 id="initialize"><code>initialize</code></h3>

A no-arg <a href="../../tff/Computation.md"><code>tff.Computation</code></a>
that returns the initial state.

<h3 id="next"><code>next</code></h3>

A <a href="../../tff/Computation.md"><code>tff.Computation</code></a> that
produces the next state.

The first argument of should always be the current state (originally produced by
`tff.IterativeProcess.initialize`), and the first (or only) returned value is
the updated state.

#### Returns:

A <a href="../../tff/Computation.md"><code>tff.Computation</code></a>.
