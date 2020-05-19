package(default_visibility = [
    "//tensorflow_federated/python/core/backends:__subpackages__",
    "//tensorflow_federated/python/core/framework:__subpackages__",
    "//tensorflow_federated/python/core/utils:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "templates",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow_federated:__pkg__",
        "//tensorflow_federated/python/core:__pkg__",
    ],
    deps = [
        ":iterative_process",
        ":measured_process",
    ],
)

py_library(
    name = "iterative_process",
    srcs = ["iterative_process.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
    ],
)

py_test(
    name = "iterative_process_test",
    size = "small",
    srcs = ["iterative_process_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":iterative_process",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:values",
        "//tensorflow_federated/python/core/impl/executors:default_executor",
        "//tensorflow_federated/python/core/impl/executors:executor_stacks",
    ],
)

py_library(
    name = "measured_process",
    srcs = ["measured_process.py"],
    srcs_version = "PY3",
    deps = [
        ":iterative_process",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "measured_process_test",
    size = "small",
    srcs = ["measured_process_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":measured_process",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:values",
        "//tensorflow_federated/python/core/impl/executors:default_executor",
        "//tensorflow_federated/python/core/impl/executors:executor_stacks",
    ],
)
