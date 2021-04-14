"""TensorFlow Federated build macros and rules."""

load("@rules_python//python:defs.bzl", "py_test")

def py_cpu_gpu_test(name, main = None, tags = [], **kwargs):
    """A version of `py_test` that tests both cpu and gpu.

    It accepts all `py_test` arguments.

    Args:
      name: A unique name for this target.
      main: The name of the source file that is the main entry point of
          the application
      tags: List of arbitrary text tags.
      **kwargs: `py_test` keyword arguments.
    """
    if main == None:
        main = name + ".py"
    py_test(
        name = name + "_cpu",
        main = main,
        tags = tags,
        **kwargs
    )

    # TODO(b/185290076): Remove this line when tests are passing again.
    if "disable_gpu" not in tags:
        py_test(
            name = name + "_gpu",
            main = main,
            tags = tags + ["requires-gpu-nvidia"],
            **kwargs
        )
    native.test_suite(
        name = name,
        tests = [
            name + "_cpu",
            name + "_gpu",
        ],
    )

def cc_cpu_gpu_test(name, tags = [], **kwargs):
    """A version of `cc_test` that tests both cpu and gpu.

    It accepts all `cc_test` arguments.

    Args:
      name: A unique name for this target.
      tags: List of arbitrary text tags.
      **kwargs: `cc_test` keyword arguments.
    """
    native.cc_test(
        name = name + "_cpu",
        tags = tags,
        **kwargs
    )
    native.cc_test(
        name = name + "_gpu",
        tags = tags + ["requires-gpu-nvidia"],
        **kwargs
    )
    native.test_suite(
        name = name,
        tests = [
            name + "_cpu",
            name + "_gpu",
        ],
    )
