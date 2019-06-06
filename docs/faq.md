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

## How can I contribute?

See the [README](../README.md) and
[guidelines for contributors](../CONTRIBUTING.md).
