load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":templates_packages",
        "//tensorflow_federated/python/learning:learning_users",
        "//tensorflow_federated/python/learning/algorithms:algorithms_packages",
        "//tensorflow_federated/python/learning/programs:programs_packages",
    ],
)

package_group(
    name = "templates_packages",
    packages = ["//tensorflow_federated/python/learning/templates/..."],
)

licenses(["notice"])

py_library(
    name = "templates",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":apply_optimizer_finalizer",
        ":client_works",
        ":composers",
        ":distributors",
        ":finalizers",
        ":learning_process",
        ":model_delta_client_work",
    ],
)

py_library(
    name = "apply_optimizer_finalizer",
    srcs = ["apply_optimizer_finalizer.py"],
    deps = [
        ":finalizers",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:tensor_utils",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "apply_optimizer_finalizer_test",
    srcs = ["apply_optimizer_finalizer_test.py"],
    deps = [
        ":apply_optimizer_finalizer",
        ":finalizers",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "client_works",
    srcs = ["client_works.py"],
    deps = [
        ":hparams_base",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "client_works_test",
    size = "small",
    srcs = ["client_works_test.py"],
    deps = [
        ":client_works",
        ":hparams_base",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/models:model_weights",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "composers",
    srcs = ["composers.py"],
    deps = [
        ":apply_optimizer_finalizer",
        ":client_works",
        ":distributors",
        ":finalizers",
        ":learning_process",
        ":model_delta_client_work",
        ":type_checks",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "composers_test",
    size = "small",
    timeout = "moderate",
    srcs = ["composers_test.py"],
    deps = [
        ":apply_optimizer_finalizer",
        ":client_works",
        ":composers",
        ":distributors",
        ":finalizers",
        ":learning_process",
        ":model_delta_client_work",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning/models:keras_utils",
        "//tensorflow_federated/python/learning/models:model_examples",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "distributors",
    srcs = ["distributors.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "distributors_test",
    size = "small",
    srcs = ["distributors_test.py"],
    deps = [
        ":distributors",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "finalizers",
    srcs = ["finalizers.py"],
    deps = [
        ":hparams_base",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "finalizers_test",
    size = "small",
    srcs = ["finalizers_test.py"],
    deps = [
        ":finalizers",
        ":hparams_base",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/models:model_weights",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "hparams_base",
    srcs = ["hparams_base.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "hparams_base_test",
    srcs = ["hparams_base_test.py"],
    deps = [
        ":hparams_base",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "learning_process",
    srcs = ["learning_process.py"],
    deps = [
        ":hparams_base",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/templates:errors",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "learning_process_test",
    size = "small",
    srcs = ["learning_process_test.py"],
    deps = [
        ":learning_process",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:errors",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "model_delta_client_work",
    srcs = ["model_delta_client_work.py"],
    deps = [
        ":client_works",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning:loop_builder",
        "//tensorflow_federated/python/learning:tensor_utils",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/learning/metrics:types",
        "//tensorflow_federated/python/learning/models:functional",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "model_delta_client_work_test",
    size = "small",
    timeout = "moderate",
    srcs = ["model_delta_client_work_test.py"],
    deps = [
        ":client_works",
        ":model_delta_client_work",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning:loop_builder",
        "//tensorflow_federated/python/learning/metrics:counters",
        "//tensorflow_federated/python/learning/models:functional",
        "//tensorflow_federated/python/learning/models:keras_utils",
        "//tensorflow_federated/python/learning/models:model_examples",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "proximal_client_work",
    srcs = ["proximal_client_work.py"],
    deps = [
        ":client_works",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning:loop_builder",
        "//tensorflow_federated/python/learning:tensor_utils",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/learning/metrics:types",
        "//tensorflow_federated/python/learning/models:functional",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/models:variable",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "proximal_client_work_test",
    size = "small",
    timeout = "moderate",
    srcs = ["proximal_client_work_test.py"],
    deps = [
        ":client_works",
        ":proximal_client_work",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning:loop_builder",
        "//tensorflow_federated/python/learning/models:functional",
        "//tensorflow_federated/python/learning/models:model_examples",
        "//tensorflow_federated/python/learning/models:model_weights",
        "//tensorflow_federated/python/learning/models:test_models",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "type_checks",
    srcs = ["type_checks.py"],
    deps = [
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "type_checks_test",
    srcs = ["type_checks_test.py"],
    deps = [
        ":type_checks",
        "@federated_language//federated_language",
    ],
)
