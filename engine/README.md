# TFF Engine Libraries Overview

The TFF Engine libraries provide C++ and Java APIs, as well as command-line tools, for working with federated learning artifacts such as TensorFlow checkpoints and aggregation plans. These libraries are designed to facilitate the serialization, parsing, and aggregation of model parameters and plans in distributed and privacy-preserving machine learning workflows.

## Key Components

### 1. Core Aggregation Library (C++)
- Implements tensor and checkpoint serialization/deserialization.
- Provides Tensor, TensorProto, TensorShape, and aggregation utilities.
- Used for building and parsing binary checkpoint files compatible with federated learning systems.

### 2. Java Aggregation Tool
- Provides a Java interface for aggregation sessions.
- Supports JNI integration with native C++ aggregation logic.
- Useful for orchestrating federated aggregation from Java-based systems.

### 3. Command-Line Tools
- **checkpoint_tool**: Build or parse TensorFlow checkpoints from JSON or binary files.
- **plan_tool**: Build, parse, or sample aggregation plans.
- **ass_tool**: Java-based aggregation session tool for running aggregation jobs.

## Example Usage

### Building a Checkpoint from JSON
```sh
# Input: checkpoint.json (list of tensor specs and values)
checkpoint_tool build model.ckpt < checkpoint.json
```

### Parsing a Checkpoint to JSON
```sh
# Output: checkpoint.json (human-readable tensor specs and values)
checkpoint_tool parse model.ckpt > checkpoint.json
```

### Example JSON Format for Checkpoint
```json
[
  {
    "name": "weights",
    "dtype": "DT_FLOAT",
    "shape": { "dim_sizes": [2, 2] },
    "float_val": [1.0, 2.0, 3.0, 4.0]
  },
  {
    "name": "labels",
    "dtype": "DT_STRING",
    "shape": { "dim_sizes": [2] },
    "string_val": ["cat", "dog"]
  }
]
```

### Aggregating Checkpoints (Java)
1. Build agg_tool:
```sh
bazel build //engine/java/src/tool/org/jetbrains/tff/engine:agg_tool_deploy.jar
```

2. Use it:
```sh
# Run aggregation session tool (Java)
java -jar agg_tool_deploy.jar plan.bin ckpt1.bin ckpt2.bin agg_ckpt.bin
```

### Building and Parsing Plans
```sh
# Build a plan from JSON
plan_tool build plan.bin < plan.json

# Parse a plan to JSON
plan_tool parse plan.bin > plan.json
```


## Notes
- All tools support both build (JSON → binary) and parse (binary → JSON) operations.
- Tensor dtypes supported: DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT64, DT_STRING.
- **DT_STRING deserialization:** is not supported yet.
- For custom aggregation or orchestration, use the Java APIs or extend the C++ core.

For more details, see the documentation in the `docs/` directory.
