<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff.utils

Utility classes/functions built on top of TensorFlow Federated Core API.

All components that depend on utils should import symbols from this file rather
than directly importing individual modules. For this reason, the visibility for
the latter is set to private and should remain such. The code in utils must not
depend on implementation classes. It should be written against the Core API.

## Functions

[`get_variables(...)`](../tff/utils/get_variables.md): Creates a set of variables that matches the given `type_spec`.

