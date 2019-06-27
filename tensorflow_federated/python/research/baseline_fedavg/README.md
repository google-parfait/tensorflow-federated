# Minimal Stand-Alone Implementation of Federated Averaging

This is intended to be a flexible and full-featured implementation of Federated
Averaging, and the code is designed to be modular and re-usable. See
[federated_averaging.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning/federated_averaging.py)
for a more full-featured implementation.

## Instructions

Modify the update in the server through the function `server_update`. Modify the
update in clients through the function `client_update`.

## Citation

```
@inproceedings{mcmahan2017communication,
  title={Communication-Efficient Learning of Deep Networks from
  Decentralized Data},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson,
  Seth and y Arcas, Blaise Aguera},
  booktitle={Artificial Intelligence and Statistics},
  pages={1273--1282},
  year={2017}
  }
```
