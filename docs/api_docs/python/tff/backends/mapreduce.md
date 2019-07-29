<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.backends.mapreduce" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff.backends.mapreduce

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/backends/mapreduce/__init__.py">View
source</a>

Utility classes and functions for integration with MapReduce-like backends.

<!-- Placeholder for "Used in" -->

This module contains utility components for interfacing between TFF and backend
systems that offer MapReduce-like capabilities, i.e., systems that can perform
parallel processing on a set of clients, and then aggregate the results of such
processing on the server. Systems of this type do not support the full
expressiveness of TFF, but they are common enough in practice to warrant a
dedicated set of utility functions, and many examples of TFF computations,
including those constructed by
<a href="../../tff/learning.md"><code>tff.learning</code></a>, can be compiled
by TFF into a form that can be deployed on such systems.

## Classes

[`class CanonicalForm`](../../tff/backends/mapreduce/CanonicalForm.md):
Standardized representation of logic deployable to MapReduce-like systems.
