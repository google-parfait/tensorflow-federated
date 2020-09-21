# Golden Testing

TFF includes a small library called `golden` that makes it easy to write and
maintain golden tests.

## What are golden tests? When should I use them?

Golden tests are used when you want a developer to know that their code altered
the output of a function. They violate many characteristics of good unit tests
in that they make promises about the exact outputs of functions, rather than
testing a specific set of clear, documented properties. It's sometimes not clear
when a change to a golden output is "expected" or whether it is violating some
property that the golden test saught to enforce. As such, a well-factored unit
test is usually preferable to a golden test.

However, golden tests can be extremely useful for validating the exact contents
of error messages, diagnostics, or generated code. In these cases, golden tests
can be a helpful confidence check that any changes to the generated output "look
right."

## How should I write tests using `golden`?

`golden.check_string(filename, value)` is the primary entrypoint into the
`golden` library. It will check the `value` string against the contents of a
file whose last path element is `filename`. The full path to `filename` must be
provided via a commandline `--golden <path_to_file>` argument. Similarly, these
files must be made available to tests using the `data` argument to the `py_test`
BUILD rule. Use the `location` function to generate a correct appropriate
relative path:

```
py_string_test(
  ...
  args = [
    "--golden",
    "$(location path/to/first_test_output.expected)",
    ...
    "--golden",
    "$(location path/to/last_test_output.expected)",
  ],
  data = [
    "path/to/first_test_output.expected",
    ...
    "path/to/last_test_output.expected",
  ],
  ...
)
```

By convention, golden files should be placed in a sibling directory with the
same name as their test target, suffixed with `_goldens`:

```
path/
  to/
    some_test.py
    some_test_goldens/
      test_case_one.expected
      ...
      test_case_last.expected
```

## How do I update `.expected` files?

`.expected` files can be updated by running the affected test target with the
arguments `--test_arg=--update_goldens --test_strategy=local`. The resulting
diff should be checked for unanticipated changes.
