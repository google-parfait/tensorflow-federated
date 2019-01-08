<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework.EnhancedModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="local_variables"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="aggregated_outputs"/>
<meta itemprop="property" content="forward_pass"/>
</div>

# tff.learning.framework.EnhancedModel

## Class `EnhancedModel`

Inherits From: [`Model`](../../../tff/learning/Model.md)

A wrapper around a Model that adds sanity checking and metadata helpers.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(model)
```





## Properties

<h3 id="local_variables"><code>local_variables</code></h3>



<h3 id="non_trainable_variables"><code>non_trainable_variables</code></h3>



<h3 id="trainable_variables"><code>trainable_variables</code></h3>



<h3 id="weights"><code>weights</code></h3>

Returns a `tff.learning.ModelWeights`.



## Methods

<h3 id="aggregated_outputs"><code>aggregated_outputs</code></h3>

``` python
aggregated_outputs()
```



<h3 id="forward_pass"><code>forward_pass</code></h3>

``` python
forward_pass(
    batch,
    training=True
)
```





