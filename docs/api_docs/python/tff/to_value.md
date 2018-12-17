<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.to_value" />
<meta itemprop="path" content="Stable" />
</div>

# tff.to_value

``` python
tff.to_value(
    val,
    type_spec=None
)
```

Converts the argument into an instance of the abstract base class `Value`.

Instances of `Value` represent TFF values that appear internally in federated
computations. This helper function can be used to wrap a variety of Python
objects as `Value` instances to allow them to be passed as arguments, used as
functions, or otherwise manipulated within bodies of federated computations.

This function is also invoked when attempting to execute a TFF computation.
All arguments supplied in the invocation are converted into TFF values prior
to execution. The types of Python objects that can be passed as arguments to
computations thus matches the types listed here.

#### Args:

* <b>`val`</b>: An instance of one of the Python types that are convertible to TFF
    values (instances of `Value`). At the moment, the supported types include
    the following:

    * Simple constants of `str`, `int`, `float`, and `bool` types, mapped to
      values of a TFF tensor type.

    * Numpy arrays (`ndarray` objects), also mapped to TFF tensors.

    * Dictionaries (`OrderedDict` and unordered `dict`), `list`s, `tuple`s,
      `namedtuple`s, and `AnonymousTuple`s, all of which are mapped to TFF
      named tuples.

    * Computations (constructed with either the `tf_computation` or with the
      `federated_computation` decorator), typically mapped to TFF functions.

    * Placement literals (`CLIENTS`, `SERVER`), mapped to valeus of the TFF
      placement type.

* <b>`type_spec`</b>: An optional type specifier that allows for disambiguating the
    target type (e.g., when two TFF types can be mapped to the same Python
    representations). If not specified, TFF tried to determine the type of
    the TFF value automatically.


#### Returns:

An instance of `Value` of a TFF type as described above.