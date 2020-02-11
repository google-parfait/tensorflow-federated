default_visibility = [
    "//tensorflow_federated/python/core/api:__subpackages__",
    "//tensorflow_federated/python/core/backends:__subpackages__",
    "//tensorflow_federated/python/core/framework:__subpackages__",
    "//tensorflow_federated/python/core/impl/wrappers:__subpackages__",
    "//tensorflow_federated/python/core/utils:__subpackages__",
]

package(default_visibility = default_visibility)

licenses(["notice"])

py_library(
    name = "wrappers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "computation_wrapper",
    srcs = ["computation_wrapper.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:type_utils",
        "//tensorflow_federated/python/core/impl/utils:function_utils",
    ],
)

py_test(
    name = "computation_wrapper_test",
    size = "small",
    srcs = ["computation_wrapper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_wrapper",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:context_base",
        "//tensorflow_federated/python/core/impl:context_stack_impl",
        "//tensorflow_federated/python/core/impl/utils:function_utils",
    ],
)

py_library(
    name = "computation_wrapper_instances",
    srcs = ["computation_wrapper_instances.py"],
    srcs_version = "PY3",
    # TODO(b/143772942): Explicitly setting visibility is temporary and can be
    # removed once  all the appropriate modules have been moved out of the
    # `impl` package.
    visibility = default_visibility + [
        "//tensorflow_federated/python/core/impl:__pkg__",
    ],
    deps = [
        ":computation_wrapper",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl:context_stack_impl",
        "//tensorflow_federated/python/core/impl:federated_computation_utils",
        "//tensorflow_federated/python/core/impl:tensorflow_serialization",
        "//tensorflow_federated/python/core/impl:type_utils",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/utils:function_utils",
    ],
)

py_test(
    name = "computation_wrapper_instances_test",
    size = "small",
    srcs = ["computation_wrapper_instances_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_wrapper_instances",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
    ],
)

py_library(
    name = "set_default_executor",
    srcs = ["set_default_executor.py"],
    srcs_version = "PY3",
    # TODO(b/143772942): Explicitly setting visibility is temporary and can be
    # removed once  all the appropriate modules have been moved out of the
    # `impl` package.
    visibility = default_visibility + [
        "//tensorflow_federated/python/core/impl:__pkg__",
    ],
    deps = [
        "//tensorflow_federated/python/core/impl:context_stack_impl",
        "//tensorflow_federated/python/core/impl:execution_context",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
    ],
)

py_test(
    name = "set_default_executor_test",
    size = "small",
    srcs = ["set_default_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":set_default_executor",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl:context_stack_impl",
        "//tensorflow_federated/python/core/impl/executors:eager_executor",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
    ],
)

py_library(
    name = "get_context_stack",
    srcs = ["get_context_stack.py"],
    srcs_version = "PY3",
    # TODO(b/143772942): Explicitly setting visibility is temporary and can be
    # removed once  all the appropriate modules have been moved out of the
    # `impl` package.
    visibility = default_visibility + [
        "//tensorflow_federated/python/core/impl:__pkg__",
    ],
    deps = ["//tensorflow_federated/python/core/impl:context_stack_impl"],
)

py_test(
    name = "get_context_stack_test",
    size = "small",
    srcs = ["get_context_stack_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":get_context_stack",
        "//tensorflow_federated/python/core/impl:context_stack_impl",
        "//tensorflow_federated/python/core/impl:context_stack_test_utils",
        "//tensorflow_federated/python/core/impl:execution_context",
    ],
)

py_library(
    name = "set_default_context",
    srcs = ["set_default_context.py"],
    srcs_version = "PY3",
    # TODO(b/143772942): Explicitly setting visibility is temporary and can be
    # removed once  all the appropriate modules have been moved out of the
    # `impl` package.
    visibility = default_visibility + [
        "//tensorflow_federated/python/core/impl:__pkg__",
    ],
    deps = ["//tensorflow_federated/python/core/impl:context_stack_impl"],
)

py_test(
    name = "set_default_context_test",
    size = "small",
    srcs = ["set_default_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":get_context_stack",
        ":set_default_context",
        "//tensorflow_federated/python/core/impl:context_stack_test_utils",
        "//tensorflow_federated/python/core/impl:execution_context",
    ],
)
