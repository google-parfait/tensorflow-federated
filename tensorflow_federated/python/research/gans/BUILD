package(default_visibility = ["//tensorflow_federated/python/research:__subpackages__"])

licenses(["notice"])

py_library(
    name = "gan_losses",
    srcs = ["gan_losses.py"],
    srcs_version = "PY3",
)

py_library(
    name = "gan_training_tf_fns",
    srcs = ["gan_training_tf_fns.py"],
    srcs_version = "PY3",
    deps = [
        ":gan_losses",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_test(
    name = "gan_training_tf_fns_test",
    srcs = ["gan_training_tf_fns_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gan_losses",
        ":gan_training_tf_fns",
        ":one_dim_gan",
    ],
)

py_library(
    name = "one_dim_gan",
    srcs = ["one_dim_gan.py"],
    srcs_version = "PY3",
)

py_library(
    name = "tff_gans",
    srcs = ["tff_gans.py"],
    srcs_version = "PY3",
    deps = [
        ":gan_training_tf_fns",
        "//tensorflow_federated",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "tff_gans_test",
    size = "large",
    srcs = ["tff_gans_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    # TODO(b/158558930): re-enable tests after removing change dectection.
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":gan_losses",
        ":gan_training_tf_fns",
        ":one_dim_gan",
        ":tff_gans",
        "//tensorflow_federated",
    ],
)

py_library(
    name = "training_loops",
    srcs = ["training_loops.py"],
    srcs_version = "PY3",
    deps = [
        ":gan_training_tf_fns",
        ":tff_gans",
        "//tensorflow_federated/python/research/utils:checkpoint_utils",
    ],
)

py_test(
    name = "training_loops_test",
    size = "large",
    srcs = ["training_loops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gan_losses",
        ":gan_training_tf_fns",
        ":one_dim_gan",
        ":tff_gans",
        ":training_loops",
    ],
)
