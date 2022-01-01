load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = [":test_packages"])

package_group(
    name = "test_packages",
    packages = ["//tensorflow_federated/python/tests/..."],
)

licenses(["notice"])

py_test(
    name = "ast_generation_test",
    size = "small",
    srcs = ["ast_generation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "backend_test",
    timeout = "long",
    srcs = ["backend_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":remote_runtime_test_utils",
        ":temperature_sensor_example",
        ":test_contexts",
        "//tensorflow_federated",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras",
    ],
)

py_cpu_gpu_test(
    name = "backend_accelerators_test",
    srcs = ["backend_accelerators_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "composite_tensor_e2e_test",
    size = "small",
    srcs = ["composite_tensor_e2e_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "federated_averaging_e2e_convergence_test",
    timeout = "eternal",
    srcs = ["federated_averaging_e2e_convergence_test.py"],
    python_version = "PY3",
    shard_count = 7,
    srcs_version = "PY3",
    tags = [
        "guitar",
        "learning",
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow_federated",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras",
    ],
)

py_test(
    name = "federated_averaging_integration_test",
    size = "large",
    srcs = ["federated_averaging_integration_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "guitar",
        "learning",
        "manual",
    ],
    deps = [
        ":learning_test_models",
        "//tensorflow_federated",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras",
    ],
)

py_test(
    name = "federated_sgd_e2e_convergence_test",
    timeout = "eternal",
    srcs = ["federated_sgd_e2e_convergence_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    tags = [
        "guitar",
        "learning",
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow_federated",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras",
    ],
)

py_test(
    name = "federated_sgd_integration_test",
    size = "large",
    srcs = ["federated_sgd_integration_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "guitar",
        "learning",
        "manual",
    ],
    deps = [
        ":learning_test_models",
        "//tensorflow_federated",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras",
    ],
)

py_library(
    name = "learning_test_models",
    testonly = True,
    srcs = ["learning_test_models.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras",
    ],
)

py_test(
    name = "map_reduce_form_test",
    size = "medium",
    srcs = ["map_reduce_form_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras",
    ],
)

py_test(
    name = "remote_runtime_integration_test",
    timeout = "moderate",
    srcs = ["remote_runtime_integration_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":remote_runtime_test_utils",
        "//tensorflow_federated",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "remote_runtime_test_utils",
    testonly = True,
    srcs = ["remote_runtime_test_utils.py"],
    data = [":test_aggregator"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "@absl_py//absl/logging",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "simulation_test",
    size = "medium",
    timeout = "long",
    srcs = ["simulation_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras",
    ],
)

py_library(
    name = "temperature_sensor_example",
    srcs = ["temperature_sensor_example.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "test_aggregator",
    srcs = ["test_aggregator.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_library(
    name = "test_contexts",
    testonly = True,
    srcs = ["test_contexts.py"],
    srcs_version = "PY3",
    deps = [
        ":remote_runtime_test_utils",
        "//tensorflow_federated",
        "@absl_py//absl/testing:parameterized",
    ],
)
