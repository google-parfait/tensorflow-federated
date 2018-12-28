<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework.SequentialTffComputation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="run_one_round"/>
<meta itemprop="property" content="__init__"/>
</div>

# tff.learning.framework.SequentialTffComputation

## Class `SequentialTffComputation`



Container for a pair of TFF computations defining sequential processing.

A sequential computation will usually be driven by a control loop like:

      seq_comp = ...
      state = seq_comp.initialize()
      for round in range(num_rounds):
        state = seq_comp.run_one_round(state, ...)

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    initialize,
    run_one_round
)
```

Creates a SequentialTffComputation.



## Properties

<h3 id="initialize"><code>initialize</code></h3>

A no-arg <a href="../../../tff/Computation.md"><code>tff.Computation</code></a> that returns the initial server state.

<h3 id="run_one_round"><code>run_one_round</code></h3>

A <a href="../../../tff/Computation.md"><code>tff.Computation</code></a> that updates the server state.

The first argument of this computation should always be the current
server state as produced by `initialize`, and the return value is the
updated server state.

#### Returns:

A <a href="../../../tff/Computation.md"><code>tff.Computation</code></a>.



