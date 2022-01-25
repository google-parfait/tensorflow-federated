load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = [":test_packages"])

package_group(
    name = "test_packages",
    packages = ["//third_party/tensorflow_federated/experimental/python/tests/..."],
)

licenses(["notice"])

py_test(
    name = "jax_training_test",
    srcs = ["jax_training_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//third_party/tensorflow_federated"],
)
