# Datasets

This directory contains various datasets for federated learning research in
TensorFlow Federated. The libraries load from datasets using TFF (see the list
of datasets
[here](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets))
and apply various kinds of pre-processing.

These datasets are intended primarily for research in TFF (see
[Using TFF for Federated Learning Research](https://github.com/tensorflow/federated/blob/master/docs/tff_for_research.md))
and are not necessarily intended to be production-ready. Moreover, these
datasets are tied to prior and ongoing research. While changes designed to clean
up or speed up the dataset preprocessing are welcome, any fundamental changes to
the datasets should be instead implemented as new datasets.
