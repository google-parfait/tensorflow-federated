<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.uniquify_reference_names" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.uniquify_reference_names

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py">View
source</a>

Replaces all the bound reference names in `comp` with unique names.

```python
tff.framework.uniquify_reference_names(comp)
```

<!-- Placeholder for "Used in" -->

Notice that `uniquify_reference_names` simply leaves alone any reference which
is unbound under `comp`.

#### Args:

*   <b>`comp`</b>: The computation building block in which to perform the
    replacements.

#### Returns:

Returns a transformed version of comp inside of which all variable names are
guaranteed to be unique.
