# TensorFlow Federated's Program API

[TOC]

This documentation is for users of
[TensorFlow Federated's (TFF) federated program API](#what-is-tffs-federated-program-api)
or anyone who is interested in a high-level overview of
[federated programs](#what-is-a-federated-program) or this API.

*   See
    [federated program examples](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/...)
    for runnable examples of the components and concepts described here.
*   See
    [tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program)
    for more detailed documentation about specific components.

## What is a federated program?

A **federated program** is a binary that executes federated computations and
other processing logic.

## What is TFF's federated program API?

TFF's federated program API is a library
([tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program))
that provides [components](#components) and defines [concepts](#concepts)
developers can use to create and understand federated programs.

Specifically a **federated program**:

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
*   and may [release](#release) data to [customer storage](#customer storage) as
    metrics

These abstractions describe the relationships between the components of a
federated program and allows these components to be owned and authored by
different [roles](#roles). This decoupling enables developers to compose
federated program using components that are shared with other federated
programs, typically this means executing the same program logic on many
different platforms.

## Components

The **components** of the TFF's federated program API are designed so they can
be owned and authored by different [roles](#roles).

Note: This is a high-level overview of the components, see
[tff.program](https://www.tensorflow.org/federated/api_docs/python/tff/program)
for more detailed documentation about specific components.

### Program

The **program** is a Python binary that composes all the other
[components](#components) of TFF's federated program API to create a federated
program.

For example:

```
# Parameters set by the customer.
flags.DEFINE_string('output_dir', None, 'The output path.')

def main():

  # Parameters set by the program.
  total_rounds = 10
  number_of_clients = 3

  # Configure the platform-specific components.
  context = tff.program.NativeFederatedContext(...)
  data_source = tff.program.DatasetDataSource(...)

  # Configure the platform-agnostic components.
  summary_dir = os.path.join(FLAGS.output_dir, 'summary')
  output_managers = [
      tff.program.LoggingReleaseManager(),
      tensorboard_manager = tff.program.TensorBoardReleaseManager(summary_dir),
  ]
  program_state_dir = os.path.join(..., 'program_state')
  program_state_manager = tff.program.FileProgramStateManager(program_state_dir)

  # Define the computations.
  initialize = ...
  train = ...

  # Execute the computations using program logic.
  tff.framework.set_default_context(context)
  train_federated_model(
      initialize=initialize,
      train=train,
      data_source=data_source,
      total_rounds=total_rounds,
      number_of_clients=number_of_clients,
      output_managers=output_managers,
      program_state_manager=program_state_manager)
```

### Parameters

The **Parameters** are the inputs to the [program](#program), these inputs may
be set by the [customer](#customer) if they are exposed as flags to the program
or they may be set by the program. For example, `total_rounds` and
`number_of_clients` are parameters in the example of a program above.

### Platform-Specific Components

The **Platform-specific components** are the components provided by a
[platform](#platform) implementing the abstract interfaces defined by TFF's
federated program API that *require* a platform-specific implementation.

The "platform-specific" abstract interfaces are:

*   [`tff.program.FederatedContext`](https://www.tensorflow.org/federated/api_docs/python/tff/program/FederatedContext)
*   [`tff.program.FederatedDataSourceIterator`](https://www.tensorflow.org/federated/api_docs/python/tff/program/FederatedDataSourceIterator)
*   [`tff.program.FederatedDataSource`](https://www.tensorflow.org/federated/api_docs/python/tff/program/FederatedDataSource)
*   [`tff.program.MaterializableValueReference`](https://www.tensorflow.org/federated/api_docs/python/tff/program/MaterializableValueReference)

Because it is often useful when running simulations or for testing, TFF's
federated program API provides a **native platform** comprised of
platform-specific components:

*   [`tff.program.CoroValueReference`](https://www.tensorflow.org/federated/api_docs/python/tff/program/CoroValueReference)
*   [`tff.program.NativeFederatedContext`](https://www.tensorflow.org/federated/api_docs/python/tff/program/NativeFederatedContext)
*   [`tff.program.DatasetDataSourceIterator`](https://www.tensorflow.org/federated/api_docs/python/tff/program/DatasetDataSourceIterator)
*   [`tff.program.DatasetDataSource`](https://www.tensorflow.org/federated/api_docs/python/tff/program/DatasetDataSource)

### Platform-Agnostic Components

The **Platform-agnostic components** are the components provided by a
[library](#library) (such as TFF itself) implementing the abstract interfaces
defined by TFF's federated program API that *may* have a platform-agnostic
implementation.

The "platform-agnostic" abstract interfaces are:

*   [`tff.program.ProgramStateManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/ProgramStateManager)
*   [`tff.program.ReleaseManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/ReleaseManager)

Because these abstract interfaces may have a platform-agnostic implementation,
TFF's federated program API provides some platform-agnostic components:

*   [`tff.program.CSVFileReleaseManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/CSVFileReleaseManager)
*   [`tff.program.FileProgramStateManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/FileProgramStateManager)
*   [`tff.program.LoggingReleaseManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/LoggingReleaseManager)
*   [`tff.program.MemoryReleaseManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/MemoryReleaseManager)
*   [`tff.program.SavedModelFileReleaseManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/SavedModelFileReleaseManager)
*   [`tff.program.TensorboardReleaseManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/TensorboardReleaseManager)

Note: Because a component may have a platform-agnostic implementation does not
limit a [platform](#platform) from providing a platform-specific implementation
to these abstract interfaces.

### Computations

The **computations** are implementations of
[`tff.Computation`](https://www.tensorflow.org/federated/api_docs/python/tff/Computation)s.

For example:

```
@tff.tf_computation()
def add_one(x):
  return x + 1
```

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

```
def train_federated_model(
    initialize: tff.Computation,
    train: tff.Computation,
    data_source: tff.program.FederatedDataSource,
    total_rounds: int,
    number_of_clients: int,
    output_managers: Optional[List[tff.program.ReleaseManager]] = None,
    program_state_manager: Optional[tff.program.ProgramStateManager] = None):
  state = initialize()
  start_round = 1

  data_iterator = data_source.iterator()
  for round_number in range(1, total_rounds + 1):
    train_data = data_iterator.select(number_of_clients)
    state, metrics = train(state, train_data)

    if output_managers is not None:
      for output_manager in output_managers:
        output_manager.release(metrics, round_number)
```

## Roles

There are primarily three **roles** that are useful to define when discussing
TFF's federated program API: the [customer](#customer), the
[platform](#platform), and the [library](#library). Each of these roles
typically owns some of the [components](#components) used to to create a
federated program. However it is possible for a single entity or group to
fulfill multiple roles.

Note: The author of the [program](#program) is not typically determined by the
role, but rather by the use case.

### Customer

The **customer** typically:

*   owns [customer storage](#customer-storage)
*   launches the [program](#program)

but may:

*   authors the [program](#program)
*   fulfill any of the capabilities of the [platform](#platform)

### Platform

The **platform** typically:

*   owns [platform storage](#platform-storage)
*   authors [platform-specific components](#platform-specific-components)

but may:

*   authors the [program](#program)
*   fulfill any of the capabilities of the [library](#library)

### Library

A **library** typically:

*   authors [platform-agnostic components](#platform-agnostic-components)
*   authors [computations](#computations)
*   authors [program logic](#program-logic)

## Concepts

There are a few **concepts** that are useful to document when discussing TFF's
federated program API.

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

**Fault Tolerance** is the ability of the [program logic](#program-logic) to
recover from a failure when executing a computations. For example, if you
successfully train rounds 90 out of 100 rounds and then restart training after a
failure during traning.
