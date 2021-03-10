load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/simulation:simulation_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/simulation/baselines/stackoverflow/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "stackoverflow",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/simulation/baselines:__pkg__"],
)

py_library(
    name = "tag_prediction_preprocessing",
    srcs = ["tag_prediction_preprocessing.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tag_prediction_preprocessing_test",
    size = "medium",
    srcs = ["tag_prediction_preprocessing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tag_prediction_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "word_prediction_models",
    srcs = ["word_prediction_models.py"],
    srcs_version = "PY3",
    deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
)

py_test(
    name = "word_prediction_models_test",
    srcs = ["word_prediction_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":word_prediction_models",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "word_prediction_preprocessing",
    srcs = ["word_prediction_preprocessing.py"],
    srcs_version = "PY3",
    deps = [
        requirement("attrs"),
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "word_prediction_preprocessing_test",
    size = "medium",
    srcs = ["word_prediction_preprocessing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":word_prediction_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
