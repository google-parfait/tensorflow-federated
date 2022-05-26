# Multi-Framework Support in TensorFlow Federated

TensorFlow Federated (TFF) has been designed to support a broad range of
federated computations, expressed through a combination of TFF's federated
operators that model distributed communication, and local processing logic.

Currently local processing logic can be expressed using TensorFlow APIs (via
`@tff.tf_computation`) at the frontend, and is executed via the TensorFlow
runtime at the backend. However, we aim to support multiple other
(non-TensorFlow) frontend and backend frameworks for local computations,
including non-ML frameworks (e.g., for logic expressed in SQL or general-purpose
programming languages).

In this section, we'll include information on:

*   Mechanisms that TFF provides to support alternative frameworks, and how you
    can add support for your preferred type of frontend or backend to TFF.

*   Experimental implementations of support for non-TensorFlow frameworks, with
    examples.

*   Tentative future roadmap for graduating these capabilities beyond the
    experimental status.
