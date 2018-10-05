<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.to_type" />
<meta itemprop="path" content="Stable" />
</div>

# tff.to_type

``` python
tff.to_type(spec)
```

Converts the argument into an instance of Type.

#### Args:

* <b>`spec`</b>: Either an instance of Type, or an argument convertible to Type.
    Assorted examples of type specifications are included below.

    Examples of arguments convertible to tensor types:

      tf.int32
      (tf.int32, [10])

    Examples of arguments convertible to named tuple types:

      [tf.int32, tf.bool]
      (tf.int32, tf.bool)
      [('a', tf.int32), ('b', tf.bool)]
      collections.OrderedDict([('a', tf.int32), ('b', tf.bool)])
      (tf.int32, (tf.float32, tf.bool))
      (tf.int32, (('x', tf.float32), tf.bool))
      ((tf.int32, [1]), (('x', (tf.float32, [2])), (tf.bool, [3])))


#### Returns:

An instance of tb.Type corresponding to the given spec.