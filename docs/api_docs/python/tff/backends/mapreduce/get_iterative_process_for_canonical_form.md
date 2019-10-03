<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.backends.mapreduce.get_iterative_process_for_canonical_form" />
<meta itemprop="path" content="Stable" />
</div>

# tff.backends.mapreduce.get_iterative_process_for_canonical_form

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/backends/mapreduce/canonical_form_utils.py">View
source</a>

Creates
<a href="../../../tff/utils/IterativeProcess.md"><code>tff.utils.IterativeProcess</code></a>
from a canonical form.

```python
tff.backends.mapreduce.get_iterative_process_for_canonical_form(cf)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`cf`</b>: An instance of
    <a href="../../../tff/backends/mapreduce/CanonicalForm.md"><code>tff.backends.mapreduce.CanonicalForm</code></a>.

#### Returns:

An instance of
<a href="../../../tff/utils/IterativeProcess.md"><code>tff.utils.IterativeProcess</code></a>
that corresponds to `cf`.

#### Raises:

*   <b>`TypeError`</b>: If the arguments are of the wrong types.
