<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework.ModelWeights" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="trainable"/>
<meta itemprop="property" content="non_trainable"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="from_model"/>
</div>

# tff.learning.framework.ModelWeights

## Class `ModelWeights`



A container for the trainable and non-trainable variables of a `Model`.

Note this does not include the model's local variables.

It may also be used to hold other values that are parallel to these variables,
e.g., tensors corresponding to variable values, or updates to model variables.

<h2 id="__new__"><code>__new__</code></h2>

``` python
@staticmethod
__new__(
    cls,
    trainable,
    non_trainable
)
```

Create new instance of ModelWeightsBase(trainable, non_trainable)



## Properties

<h3 id="trainable"><code>trainable</code></h3>



<h3 id="non_trainable"><code>non_trainable</code></h3>





## Methods

<h3 id="from_model"><code>from_model</code></h3>

``` python
@classmethod
from_model(
    cls,
    model
)
```





