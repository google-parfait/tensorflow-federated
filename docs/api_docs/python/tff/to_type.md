<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.to_type" />
<meta itemprop="path" content="Stable" />
</div>

# tff.to_type

Converts the argument into an instance of
<a href="../tff/Type.md"><code>tff.Type</code></a>.

```python
tff.to_type(spec)
```

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_types.py">View
source</a>

<!-- Placeholder for "Used in" -->

Examples of arguments convertible to tensor types:

```python
tf.int32
(tf.int32, [10])
(tf.int32, [None])
```

Examples of arguments convertible to flat named tuple types:

```python
[tf.int32, tf.bool]
(tf.int32, tf.bool)
[('a', tf.int32), ('b', tf.bool)]
('a', tf.int32)
collections.OrderedDict([('a', tf.int32), ('b', tf.bool)])
```

Examples of arguments convertible to nested named tuple types:

```python
(tf.int32, (tf.float32, tf.bool))
(tf.int32, (('x', tf.float32), tf.bool))
((tf.int32, [1]), (('x', (tf.float32, [2])), (tf.bool, [3])))
```

#### Args:

*   <b>`spec`</b>: Either an instance of
    <a href="../tff/Type.md"><code>tff.Type</code></a>, or an argument
    convertible to <a href="../tff/Type.md"><code>tff.Type</code></a>.

#### Returns:

An instance of <a href="../tff/Type.md"><code>tff.Type</code></a> corresponding
to the given `spec`.
