# Federated Program Developer Guide

This documentation is for anyone who is interested in authoring
[federated program logic](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic)
or a
[federated program](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program).
It assumes knowledge of TensorFlow Federated, especially its type system, and
[federated programs](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md).

[TOC]

## Program Logic

This section defines guidelines for how
[program logic](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program-logic)
should be authored.

See the example
[program_logic.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program_logic.py)
for more information.

### Document Type Signatures

**Do** document the TFF type signature for each parameter supplied to the
program logic that has a type signature.

```python {.good}
async def program_logic(
    train: tff.Computation,
    data_source: tff.program.FederatedDataSource,
    ...
) -> None:
  """Trains a federated model for some number of rounds.

  The following types signatures are required:

  1.  `train`:       `(<S@SERVER, D@CLIENTS> -> <S@SERVER, M@SERVER>)`
  2.  `data_source`: `D@CLIENTS`

  Where:

  *   `S`: The server state.
  *   `M`: The train metrics.
  *   `D`: The train client data.
  """
```

### Check Type Signatures

**Do** check the TFF type signature (at runtime) for each parameter supplied to
the program logic that has a type signature.

```python {.good}
def _check_program_logic_type_signatures(
    train: tff.Computation,
    data_source: tff.program.FederatedDataSource,
    ...
) -> None:
  ...

async def program_logic(
    train: tff.Computation,
    data_source: tff.program.FederatedDataSource,
    ...
) -> None:
  _check_program_logic_type_signatures(
      train=train,
      data_source=data_source,
  )
  ...
```

### Type Annotations

**Do** provide a well defined Python type for each
[`tff.program.ReleaseManager`](https://www.tensorflow.org/federated/api_docs/python/tff/program/ReleaseManager)
parameter supplied to the program logic.

```python {.good}
async def program_logic(
    metrics_manager: Optional[
        tff.program.ReleaseManager[tff.program.ReleasableStructure, int]
    ] = None,
    ...
) -> None:
  ...
```

Not

```python {.bad}
async def program_logic(
    metrics_manager,
    ...
) -> None:
  ...
```

```python {.bad}
async def program_logic(
    metrics_manager: Optional[tff.program.ReleaseManager] = None,
    ...
) -> None:
  ...
```

### Program State

**Do** provide a well defined structure describing the program state of the
program logic.

```python {.good}
class _ProgramState(NamedTuple):
  state: object
  round_num: int

async def program_loic(...) -> None:
  initial_state = ...

  # Load the program state
  if program_state_manager is not None:
    structure = _ProgramState(initial_state, round_num=0)
    program_state, version = await program_state_manager.load_latest(structure)
  else:
    program_state = None
    version = 0

  # Assign state and round_num
  if program_state is not None:
    state = program_state.state
    start_round = program_state.round_num + 1
  else:
    state = initial_state
    start_round = 1

  for round_num in range(start_round, ...):
    state, _ = train(state, ...)

    # Save the program state
    program_state = _ProgramState(state, round_num)
    version = version + 1
    program_state_manager.save(program_state, version)
```

### Document Released Values

**Do** document the values released from the program logic.

```python {.good}
async def program_logic(
    metrics_manager: Optional[tff.program.ReleaseManager] = None,
    ...
) -> None:
  """Trains a federated model for some number of rounds.

  Each round, `loss` is released to the `metrics_manager`.
  """
```

### Release Specific Values

**Do** not release more values from the program logic than is required.

```python {.good}
async def program_logic(...) -> None:

  for round_number in range(...):
    state, metrics = train(state, ...)

    _, metrics_type = train.type_signature.result
    loss = metrics['loss']
    loss_type = metrics_type['loss']
    metrics_manager.release(loss, loss_type, round_number)
```

Not

```python {.bad}
async def program_loic(...) -> None:

  for round_number in range(...):
    state, metrics = train(state, ...)

    _, metrics_type = train.type_signature.result
    metrics_manager.release(metrics, metrics_type, round_number)
```

Note: It is ok to release all the values if that is what is required.

### Asynchronous Functions

**Do** define the program logic as an
[asynchronous function](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition).
The
[components](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#components)
of TFF's federated program library use
[asyncio](https://docs.python.org/3/library/asyncio.html) to execute Python
concurrently and defining the program logic as an asynchronous function makes it
easier to interact with those components.

```python {.good}
async def program_logic(...) -> None:
  ...
```

Not

```python {.bad}
def program_logic(...) -> None:
  ...
```

### Tests

**Do** provide unit tests for the program logic (e.g.
[program_logic_test.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program_logic_test.py)).

## Program

This section defines guidelines for how a
[program](https://github.com/tensorflow/federated/blob/main/docs/program/federated_program.md#program)
should be authored.

See the example
[program.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program.py)
for more information.

### Document the Program

**Do** document the details of the program to the customer in the docstring of
the module (e.g.
[program.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/program/program.py)):

*   How to manually run the program.
*   What platform, computations, and data sources are used in the program.
*   How a customer should access information released from the program to
    customer storage.

### Too Many Parameters

**Don't** parameterize the program such that there are mutually exclusive
collections of parameters. For example, if `foo` is set to `X` then you are also
required to set parameters `bar`, `baz`, otherwise these parameters must be
`None`. This indicates that you could have made two different programs for
different values of `foo`.

### Group Parameters

**Do** use proto to define related but complex or verbose parameters instead of
defining many FLAGS (go/absl.flags).

> Note: Proto can be read from disk and used to construct Python objects, for
> example:
>
> ```python
> with tf.io.gfile.GFile(config_path) as f:
>   proto = text_format.Parse(f.read(), vizier_pb2.StudyConfig())
> return pyvizier.StudyConfig.from_proto(proto)
> ```

### Python Logic

**Don't** write logic (e.g. control flow, invoking computations, anything that
needs to be tested) in the program. Instead, move the logic into a private
library that can be tested or into the program logic the program invokes.

### Asynchronous Functions

**Don't** write
[asynchronous functions](https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition)
in the program. Instead, move the function into a private library that can be
tested or into the program logic the program invokes.

### Tests

**Don't** write unit tests for the program, if testing the program is useful
write those tests in terms of integration tests.

Note: It should be unlikely that testing the program is useful if Python logic
and asynchronous functions are moved into libraries and tested.
