<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.backends.mapreduce.get_canonical_form_for_iterative_process" />
<meta itemprop="path" content="Stable" />
</div>

# tff.backends.mapreduce.get_canonical_form_for_iterative_process

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/backends/mapreduce/canonical_form_utils.py">View
source</a>

Constructs
<a href="../../../tff/backends/mapreduce/CanonicalForm.md"><code>tff.backends.mapreduce.CanonicalForm</code></a>
given iterative process.

```python
tff.backends.mapreduce.get_canonical_form_for_iterative_process(iterative_process)
```

<!-- Placeholder for "Used in" -->

This function transforms computations from the input `iterative_process` into an
instance of
<a href="../../../tff/backends/mapreduce/CanonicalForm.md"><code>tff.backends.mapreduce.CanonicalForm</code></a>.

#### Args:

*   <b>`iterative_process`</b>: An instance of
    <a href="../../../tff/utils/IterativeProcess.md"><code>tff.utils.IterativeProcess</code></a>.

#### Returns:

An instance of
<a href="../../../tff/backends/mapreduce/CanonicalForm.md"><code>tff.backends.mapreduce.CanonicalForm</code></a>
equivalent to this process.

#### Raises:

*   <b>`TypeError`</b>: If the arguments are of the wrong types.
*   <b>`transformations.CanonicalFormCompilationError`</b>: If the compilation
    process fails.
