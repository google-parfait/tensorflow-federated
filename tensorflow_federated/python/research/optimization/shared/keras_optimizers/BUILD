package(default_visibility = ["//tensorflow_federated/python/research"])

licenses(["notice"])

py_library(
    name = "lars",
    srcs = ["lars.py"],
    srcs_version = "PY3",
)

py_test(
    name = "lars_test",
    size = "medium",
    srcs = ["lars_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":lars"],
)

py_library(
    name = "yogi",
    srcs = ["yogi.py"],
    srcs_version = "PY3",
)

py_test(
    name = "yogi_test",
    size = "medium",
    srcs = ["yogi_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":yogi"],
)

py_library(
    name = "shampoo",
    srcs = ["shampoo.py"],
    srcs_version = "PY3",
)

py_test(
    name = "shampoo_test",
    size = "medium",
    srcs = ["shampoo_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":shampoo"],
)
