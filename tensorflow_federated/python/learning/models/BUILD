load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":models_packages",
        "//tensorflow_federated/python/learning:learning_users",
        "//tensorflow_federated/python/learning/algorithms:algorithms_packages",
        "//tensorflow_federated/python/learning/programs:programs_packages",
        "//tensorflow_federated/python/learning/templates:templates_packages",
    ],
)

package_group(
    name = "models_packages",
    packages = ["//tensorflow_federated/python/learning/models/..."],
)

licenses(["notice"])

py_library(
    name = "model_weights",
    srcs = ["model_weights.py"],
    deps = [
        ":variable",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "models",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":functional",
        ":keras_utils",
        ":model_weights",
        ":reconstruction_model",
        ":serialization",
        ":variable",
    ],
)

py_library(
    name = "functional",
    srcs = ["functional.py"],
    deps = [
        ":variable",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:variable_utils",
        "//tensorflow_federated/python/learning/metrics:keras_finalizer",
        "//tensorflow_federated/python/learning/metrics:keras_utils",
        "//tensorflow_federated/python/learning/metrics:types",
    ],
)

py_test(
    name = "functional_test",
    srcs = ["functional_test.py"],
    deps = [
        ":functional",
        ":variable",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:variable_utils",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/learning/metrics:types",
    ],
)

py_library(
    name = "keras_utils",
    srcs = ["keras_utils.py"],
    deps = [
        ":variable",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/learning/metrics:counters",
        "//tensorflow_federated/python/learning/metrics:keras_finalizer",
    ],
)

py_test(
    name = "keras_utils_test",
    size = "medium",
    srcs = ["keras_utils_test.py"],
    deps = [
        ":keras_utils",
        ":model_examples",
        ":model_weights",
        ":variable",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/learning/metrics:counters",
    ],
)

py_library(
    name = "model_examples",
    srcs = ["model_examples.py"],
    deps = [":variable"],
)

py_test(
    name = "model_examples_test",
    size = "small",
    srcs = ["model_examples_test.py"],
    deps = [
        ":model_examples",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
    ],
)

py_library(
    name = "serialization",
    srcs = ["serialization.py"],
    deps = [
        ":functional",
        ":variable",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_test(
    name = "model_weights_test",
    srcs = ["model_weights_test.py"],
    deps = [
        ":model_weights",
        ":variable",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "reconstruction_model",
    srcs = ["reconstruction_model.py"],
    deps = [
        ":model_weights",
        ":variable",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_test(
    name = "reconstruction_model_test",
    srcs = ["reconstruction_model_test.py"],
    deps = [
        ":model_weights",
        ":reconstruction_model",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_test(
    name = "serialization_test",
    srcs = ["serialization_test.py"],
    deps = [
        ":functional",
        ":keras_utils",
        ":model_examples",
        ":serialization",
        ":test_models",
        ":variable",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "test_models",
    testonly = True,
    srcs = ["test_models.py"],
    deps = [
        ":functional",
        ":variable",
        "//tensorflow_federated/python/learning/metrics:types",
    ],
)

py_library(
    name = "variable",
    srcs = ["variable.py"],
    deps = ["//tensorflow_federated/python/learning/metrics:types"],
)
