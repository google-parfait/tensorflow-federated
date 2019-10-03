# Frequently Asked Questions

## Can TensorFlow Federated be used in production setting, e.g., on mobile phones?

Currently not. Although we designed TFF with deployment to real devices in mind,
at this stage we do not currently provide any tools for this purpose. The
current release is intended for experimentation uses, such as expressing novel
federated algorithms, or trying out federated learning with your own datasets,
using the included simulation runtime.

We anticipate that over time the open source ecosystem around TFF will evolve to
include runtimes targeting physical deployment platforms.

## How do I use TFF to experiments with large datasets?

The default runtime included in the initial release of TFF is intended only for
small experiments such as those described in our tutorials in which all your
data (across all the simulated clients) simultaneously fits in memory on a
single machine, and the entire experiment runs locally within the colab
notebook.

Our near-term future roadmap includes a high-performance runtime for experiments
with very large data sets and large numbers of clients.

## How can I ensure randomness in TFF matches my expectations?

Since TFF has federated computing baked into its core, the writer of TFF should
not assume control over where and how TensorFlow `Session`s are entered, or
`run` is called within those sessions. The semantics of randomness can depend on
entry and exit of TensorFlow `Session`s if seeds are set. We recommend using
TensorFlow 2-style radomness, using for example
`tf.random.experimental.Generator` as of TF 1.14. This uses a `tf.Variable` to
manage its internal state.

To help manage expectations, TFF allows for the TensorFlow it serializes to have
op-level seeds set, but not graph-level seeds. This is because the semantics of
op-level seeds should be clearer in the TFF setting: a deterministic sequence
will be generated upon each invocation of a function wrapped as a
`tf_computation`, and only within this invocation will any guarantees made by
the pseudorandom number generator hold. Notice that this is not quite the same
as the semantics of calling a `tf.function` in eager mode; TFF effectively
enters and exits a unique `tf.Session` each time the `tf_computation` is
invoked, while repeatedly calling a function in eager mode is analogous to
calling `sess.run` on the output tensor repeatedly within the same session.

## How can I contribute?

See the [README](../README.md) and
[guidelines for contributors](../CONTRIBUTING.md).
