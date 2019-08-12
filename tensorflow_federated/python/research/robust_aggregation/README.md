# Robust Federated Aggregation (RFA)

This is a TFF implmentation of the RFA algorithm,
a federated learning algorithm robust to corruptions in client updates.
As detailed [here](https://krishnap25.github.io/papers/2019_rfa.pdf), 
the robust aggregation is achieved by computing an approximate Geometric median
with the smoothed Weiszfeld algorithm. 
See the [paper](https://krishnap25.github.io/papers/2019_rfa.pdf) for details.

### Use
This directory gives a function `build_robust_federated_aggregation_process`
which builds a federated aggregation process implementing the RFA algorithm.
It works as a drop-in replacement to `tff.learning.build_federated_averaging_process`
for vanilla FedAvg. It can be used in the 
[TFF EMIST tutorial](https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification)
as follows:

```
from tensorflow_federated.python.research.robust_aggregation import build_robust_federated_aggregation_process
# dataset setup, model setup, etc. from tutorial
....
# change definition of iterative process from
# iterative_process = tff.learning.build_federated_averaging_process(model_fn)
# to
iterative_process = build_robust_federated_aggregation_process(model_fn)
# Rest of the code remains unchanged
```

In addition, the directory gives a standalone function `build_stateless_robust_aggregation`
build a robust aggregation oracle as an instance of `tff.utils.StatefulAggregateFn`.


## Reproducing Experimental Results
To reproduce experimental results from the [paper](https://krishnap25.github.io/papers/2019_rfa.pdf), 
see the [official repository](https://github.com/krishnap25/RFA). 


## Citation
If you find this implementation useful, please cite the following paper:

```
@article{pillutla2019robust,
title={{R}obust {A}ggregation for {F}ederated {L}earning},
author={Pillutla, Krishna and  Kakade, Sham M. and Harchaoui, Zaid},
journal={Preprint},
year={2019}
}
```
