# Federated Program

This documentation is for anyone who is interested in a high-level overview of
federated program concepts. It assumes knowledge of TensorFlow Federated,
especially its type system.

For more information about federated program, see:

*   [API Documentation](https://www.tensorflow.org/federated/api_docs/python/tff/program)
*   [Examples](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program)
*   [Developer Guide](https://github.com/tensorflow/federated/blob/main/docs/program/guide.md)

[TOC]

## What is a federated program?

A **federated program** is a program that executes computations and other
processing logic in a federated environment.

More, specifically a **federated program**:

*   executes [computations](#computations)
*   using [program logic](#program-logic)
*   with [platform-specific components](#platform-specific-components)
*   and [platform-agnostic components](#platform-agnostic-components)
*   given [parameters](#parameters) set by the [program](#program)
*   and [parameters](#parameters) set by the [customer](#customer)
*   when the [customer](#customer) runs the [program](#program)
*   and may [materialize](#materialize) data in
    [platform storage](#platform storage) to:
    *   use in Python logic
    *   implement [fault tolerance](#fault tolerance)
*   and may [release](#release) data to [customer storage](#customer storage)

Defining these [concepts](#concepts) and abstractions make it possible to
describe the relationships between the [components](#components) of a federated
program and allows these components to be owned and authored by different
[roles](#roles). This decoupling allows developers to compose federated program
using components that are shared with other federated programs, typically this
means executing the same program logic on many different platforms.

TFF's federated program library
([tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program))
defines the abstractions required to create a federated program and provides
[platform-agnostic components](#platform-agnostic-components).

## Components

The **components** of TFF's federated program library are designed so they can
be owned and authored by different [roles](#roles).

Note: This is a high-level overview of the components, see
[tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program)
for documentation of specific APIs.

### Program

The **program** is a Python binary that:

1.  defines [parameters](#parameters) (e.g. flags)
1.  constructs [platform-specific components](#platform-specific-components) and
    [platform-agnostic components](#platform-agnostic-components)
1.  executes [computations](#computations) using [program logic](#program_logic)
    in a federated context

For example:

```python
# Parameters set by the customer.
flags.DEFINE_string('output_dir', None, 'The output path.')

def main() -> None:

  # Parameters set by the program.
  total_rounds = 10
  num_clients = 3

  # Construct the platform-specific components.
  context = tff.program.NativeFederatedContext(...)
  data_source = tff.program.DatasetDataSource(...)

  # Construct the platform-agnostic components.
  summary_dir = os.path.join(FLAGS.output_dir, 'summary')
  metrics_manager = tff.program.GroupingReleaseManager([
      tff.program.LoggingReleaseManager(),
      tff.program.TensorBoardReleaseManager(summary_dir),
  ])
  program_state_dir = os.path.join(..., 'program_state')
  program_state_manager = tff.program.FileProgramStateManager(program_state_dir)

  # Define the computations.
  initialize = ...
  train = ...

  # Execute the computations using program logic.
  tff.framework.set_default_context(context)
  asyncio.run(
      train_federated_model(
          initialize=initialize,
          train=train,
          data_source=data_source,
          total_rounds=total_rounds,
          num_clients=num_clients,
          metrics_manager=metrics_manager,
          program_state_manager=program_state_manager,
      )
  )
```

### Parameters

The **parameters** are the inputs to the [program](#program), these inputs may
be set by the [customer](#customer), if they are exposed as flags, or they may
be set by the program. In the example above, `output_dir` is a parameter that is
set by the [customer](#customer), and `total_rounds` and `num_clients` are
parameters set by the program.

### Platform-Specific Components

The **platform-specific components** are the components provided by a
[platform](#platform) implementing the abstract interfaces defined by TFF's
federated program library.

### Platform-Agnostic Components

The **platform-agnostic components** are the components provided by a
[library](#library) (e.g. TFF) implementing the abstract interfaces defined by
TFF's federated program library.

### Computations

The **computations** are implementations of the abstract interface
[`tff.Computation`](https://www.tensorflow.org/federated/api_docs/python/tff/Computation).

For example, in the TFF platform you can use the
[`tff.tf_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/tf_computation)
or
[`tff.federated_computation`](https://www.tensorflow.org/federated/api_docs/python/tff/federated_computation)
decorators to create a
[`tff.framework.ConcreteComputation`](https://www.tensorflow.org/federated/api_docs/python/tff/framework/ConcreteComputation):

See
[life of a computation](https://github.com/tensorflow/federated/blob/main/docs/design/life_of_a_computation.md)
for more information.

### Program Logic

The **program logic** is a Python function that takes as an input:

*   [parameters](#parameters) set by the [customer](#customer) and the
    [program](#program)
*   [platform-specific components](#platform-specific-components)
*   [platform-agnostic components](#platform-agnostic-components)
*   [computations](#computations)

and performs some operations, which typically includes:

*   executing [computations](#computations)
*   executing Python logic
*   [materializing](#materialize) data in [platform storage](#platform storage)
    to:
    *   use in Python logic
    *   implement [fault tolerance](#fault tolerance)

and may yields some output, which typically includes:

*   [releasing](#release) data to [customer storage](#customer storage) as
    [metrics](#metrics)

For example:

```python
async def program_logic(
    initialize: tff.Computation,
    train: tff.Computation,
    data_source: tff.program.FederatedDataSource,
    total_rounds: int,
    num_clients: int,
    metrics_manager: tff.program.ReleaseManager[
        tff.program.ReleasableStructure, int
    ],
) -> None:
  state = initialize()
  start_round = 1

  data_iterator = data_source.iterator()
  for round_number in range(1, total_rounds + 1):
    train_data = data_iterator.select(num_clients)
    state, metrics = train(state, train_data)

    _, metrics_type = train.type_signature.result
    metrics_manager.release(metrics, metrics_type, round_number)
```

## Roles

There are three **roles** that are useful to define when discussing federated
programs: the [customer](#customer), the [platform](#platform), and the
[library](#library). Each of these roles owns and authors some of the
[components](#components) used to create a federated program. However, it is
possible for a single entity or group to fulfill multiple roles.

### Customer

The **customer** typically:

*   owns [customer storage](#customer-storage)
*   launches the [program](#program)

but may:

*   author the [program](#program)
*   fulfill any of the capabilities of the [platform](#platform)

### Platform

The **platform** typically:

*   owns [platform storage](#platform-storage)
*   authors [platform-specific components](#platform-specific-components)

but may:

*   author the [program](#program)
*   fulfill any of the capabilities of the [library](#library)

### Library

A **library** typically:

*   authors [platform-agnostic components](#platform-agnostic-components)
*   authors [computations](#computations)
*   authors [program logic](#program-logic)

## Concepts

There are a few **concepts** that are useful to define when discussing federated
programs.

### Customer Storage

**Customer storage** is storage that the [customer](#customer) has read and
write access to and that the [platform](#platform) has write access to.

### Platform Storage

**Platform storage** is storage that only the [platform](#platform) has read and
write access to.

### Release

**Releasing** a value makes the value available to
[customer storage](#customer-storage) (e.g. publishing the value to a dashboard,
logging the value, or writing the value to disk).

### Materialize

**Materializing** a value reference makes the referenced value available to the
[program](#program). Often materializing a value reference is required to
[release](#release) the value or to make [program logic](#program-logic)
[fault tolerant](#fault-tolerance).

### Fault Tolerance

**Fault tolerance** is the capability of the [program logic](#program-logic) to
recover from a failure when executing a computations. For example, if you
successfully train the first 90 rounds out of 100 and then experience a failure,
is the program logic capable of resuming training from round 91 or does training
need to be restarted at round 1?
