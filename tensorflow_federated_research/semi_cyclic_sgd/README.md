This directory contains source code for reproducting the results in the ICML
publication [Eichner, Koren, McMahan, Srebro, Talwar. Semi-Cyclic Stochastic
Gradient Descent](https://arxiv.org/abs/1904.10120). While these experiments are
highly relevant to the federated learning setting, we experiment with standard
SGD (without partitioning the data among users), and so these experiments use
vanilla TF rather than TFF.

## Running the experiments

The experiments can be run with `run.sh`, which will

*   download the
    [Sentiment140 dataset](http://help.sentiment140.com/for-students)
*   preprocess it
*   train + evaluate models for the parameter settings used in the paper.
    *   This step will take several hours on a single machine.
    *   Each run persists its configuration and results to a separate log file

## Analyzing the results

[`logs_analysis.ipynb`](logs_analysis.ipynb) contains a
[Jupyter](https://jupyter.org/index.html) note book that parses the log files,
analyzes the results, and plots them. This notebook was used to produce the
figures shown in the publication.
