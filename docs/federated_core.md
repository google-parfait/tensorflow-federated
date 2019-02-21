# Federated Core

This document introduces the core layer of TFF that serves as a foundation for
[Federated Learning](federated_learning.md), and possible future non-learning
federated algorithms.

For a gentle introduction to Federated Core, please read the following
tutorials, as they introduce some of the fundamental concepts by example and
demonstrate step-by-step the construction of a simple federated averaging
algorithm.

* [Custom Federated Algorithms, Part 1: Introduction to the Federated Core]
  (tutorials/custom_federated_algorithms_1.ipynb).

* [Custom Federated Algorithms, Part 2: Implementing Federated Averaging]
  (tutorials/custom_federated_algorithms_2.ipynb).

We would also encourage you to familiarize yourself with
[Federated Learning](federated_learning.md) and the associated tutorials on
[image classification]
(tutorials/federated_learning_for_image_classification.ipynb) and
[text generation]
(tutorials/federated_learning_for_text_generation.ipynb), as the uses of the
Federated Core API (FC API) for federated learning provide important context
for some of the choices we've made in designing this layer.

## Overview

### Goals, Intended Uses, and Scope

Federated Core (FC) is best understood as a programming environment for
implementing distributed computations i.e., computations that involve multiple
computers (mobile phones, tablets, embedded devices, desktop computers, sensors,
database servers, etc.) that may each perform non-trivial processing locally,
and communicate across the network to coordinate their work.

The term *distributed* is very generic, and TFF does not target all possible
types of distributed algorithms out there, so we prefer to use the less generic
term *federated computation* to describe the types of algorithms that can be
expressed in this framework.

While defining the term *federated computation* in a fully formal manner is
outside the scope of this document, think of the types of algorithms you might
see expressed in pseudocode in a [research publication]
(https://arxiv.org/pdf/1602.05629.pdf) that describes a new distributed
learning algorithm.

The goal of FC, in a nusthell, is to enable similarly compact representation,
at a similar pseudocode-like level of abstraction, of program logic that is
*not* pseudocode, but rather, that's executable in a variety of target
environments.

The key defining characteristic of the kinds of algorithms that FC is designed
to express is that actions of system participants are described in a collective
manner. Thus, we tend to talk about *each device* locally transforming data,
and the devices coordinating work by a centralized coordinator *broadcasting*,
*collecting*, or *aggregating* their results.

While TFF has been designed to be able to go beyond simple *client-server*
architectures, the notion of collective processing is fundamental. This is
due to the origins of TFF in federated learning, a technology originally
designed to support computations on potentially sensitive data that remains
under control of client devices, and that may not be simply downloaded to a
centralized location for privacy reasons. While each client in such systems
contributes data and processing power towards computing a result by the
system (a result that we would generally expect to be of value to all the
participants), we also strive at preserving each client's privacy and anonymity.

Thus, while most frameworks for distributed computing are designed to express
processing from the perspective of individual participants - that is, at the
level of individual point-to-point message exchanges, and the interdependence
of the participant's local state transitions with incoming and outgoing
messages, TFF's Federated Core is designed to describe the behavior of the
system from the *global* system-wide perspective (similarly to, e.g.,
[MapReduce](https://ai.google/research/pubs/pub62.pdf)).

Consequently, while distributed frameworks for general purposes may offer
operations such as *send* and *receive* as building blocks, FC provides building
blocks such as `tff.federated_sum`, `tff.federated_reduce`, or
`tff.federated_broadcast` that encapsulate simple distributed protocols.
