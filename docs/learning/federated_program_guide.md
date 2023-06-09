# Learning Federated Program Developer Guide

This documentation is for anyone who is interested in authoring
[federated program logic](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic)
in
[`tff.learning`](https://www.tensorflow.org/federated/api_docs/python/tff/learning).
It assumes knowledge of `tff.learning` and the
[Federated Program Developer Guide](https://github.com/tensorflow/federated/blob/main/docs/program/guide.md).

[TOC]

## Program Logic

This section defines guidelines for how
[program logic](http://g3doc/third_party/tensorflow_federated/g3doc/program/federated_program.md#program-logic)
should be authored **in `tff.learning`**.

### Learning Components

**Do** use learning components in program logic (e.g.
[`tff.learning.templates.LearningProcess`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/templates/LearningProcess)
and
[`tff.learning.programs.EvaluationManager`](https://www.tensorflow.org/federated/api_docs/python/tff/learning/programs/EvaluationManager)).

## Program

Typically
[programs](http://g3doc/third_party/tensorflow_federated/g3doc/program/federated_program.md#programs)
are not authored in `tff.learning`.
